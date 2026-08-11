"""
embeddings/embedder.py
----------------------
Phase 4: BGE Embedding Model Wrapper

Model: BAAI/bge-base-en-v1.5
  - Output dimensions: 768
  - Max sequence length: 512 tokens
  - Normalize embeddings: True (required for cosine similarity in the vector store)

BGE Asymmetric Retrieval Design:
  - Documents are embedded WITHOUT any instruction prefix.
  - Queries are embedded WITH the instruction prefix defined in QUERY_INSTRUCTION.
  This asymmetry is required for best retrieval quality with BGE models.

Singleton Pattern:
  Module-level _embedder_instance ensures the 440MB model is loaded exactly once
  per process lifetime. FastAPI Phase 9 will call get_embedder() at startup.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import Optional, Union

from dotenv import load_dotenv
import httpx

# Load environment variables at module initialization time
load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME    = "BAAI/bge-base-en-v1.5"
EMBEDDING_DIM = 768
BATCH_SIZE    = 100      # Maximized to 100 for cloud API batchEmbedContents limit.
NORMALIZE     = True     # Mandatory for cosine similarity correctness in the vector store.
MAX_SEQ_LEN   = 512      # BGE-base hard token limit.

# BGE instruction prefix — applied to queries ONLY at retrieval time.
# Documents at indexing time: NO prefix.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_embedder_instance: Optional[Union[BGEEmbedder, GeminiEmbedder, HuggingFaceEmbedder]] = None


def get_embedder() -> Union[BGEEmbedder, GeminiEmbedder, HuggingFaceEmbedder]:
    """
    Returns the process-level BGEEmbedder, GeminiEmbedder, or HuggingFaceEmbedder singleton.
    Loads the model/API on first call; subsequent calls return the cached instance.
    Use this function everywhere — never instantiate embedder classes directly.
    """
    global _embedder_instance
    if _embedder_instance is None:
        backend = os.environ.get("EMBEDDING_BACKEND", "hf").strip().lower()
        if backend == "gemini":
            _embedder_instance = GeminiEmbedder()
        elif backend in ("hf", "huggingface"):
            _embedder_instance = HuggingFaceEmbedder()
        else:
            _embedder_instance = BGEEmbedder()
    if not _embedder_instance.is_loaded:
        _embedder_instance.load()
    return _embedder_instance


# ---------------------------------------------------------------------------
# BGEEmbedder
# ---------------------------------------------------------------------------

class BGEEmbedder:
    """
    Wraps SentenceTransformer for BAAI/bge-base-en-v1.5.

    Public API:
        embed_documents(texts)  → list[list[float]]   (no prefix)
        embed_query(query)      → list[float]          (with BGE query prefix)
        embedding_dim           → int (768)
    """

    def __init__(self) -> None:
        self._model = None
        self._device: str = "cpu"

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        """
        Loads BAAI/bge-base-en-v1.5 from the local Hugging Face cache.
        Detects CUDA automatically; falls back to CPU.
        Logs model path and device so the operator knows where the model lives.
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            logger.error(
                "[EMBEDDER] sentence-transformers is not installed. "
                "Run: pip install sentence-transformers"
            )
            raise

        try:
            import torch
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            self._device = "cpu"

        logger.info(f"[EMBEDDER] Loading {MODEL_NAME} on device={self._device} ...")

        self._model = SentenceTransformer(
            MODEL_NAME,
            device=self._device,
        )

        # Override the model's default max_seq_length to enforce our limit.
        self._model.max_seq_length = MAX_SEQ_LEN

        logger.info(
            f"[EMBEDDER] Model loaded. "
            f"dim={self.embedding_dim}, "
            f"max_seq_len={MAX_SEQ_LEN}, "
            f"device={self._device}"
        )

    # ------------------------------------------------------------------
    # Public embedding API
    # ------------------------------------------------------------------

    def embed_documents(
        self,
        texts: list[str],
        batch_size: int = BATCH_SIZE,
        show_progress: bool = True,
    ) -> list[list[float]]:
        """
        Embeds a list of document texts in batches.

        Documents are embedded WITHOUT any instruction prefix.
        Normalization is always applied (required for cosine similarity).

        Args:
            texts:         List of raw chunk content strings.
            batch_size:    Number of texts per inference call. Default: 32.
            show_progress: If True, logs batch progress at INFO level.

        Returns:
            List of 768-dimensional float vectors, one per input text.
        """
        self._assert_loaded()
        if not texts:
            return []

        total = len(texts)
        embeddings: list[list[float]] = []

        for start in range(0, total, batch_size):
            batch = texts[start : start + batch_size]
            batch_embeddings = self._model.encode(
                batch,
                normalize_embeddings=NORMALIZE,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            embeddings.extend(batch_embeddings.tolist())

            if show_progress:
                done = min(start + batch_size, total)
                logger.info(f"[EMBEDDER] Embedded {done}/{total} documents.")

        return embeddings

    def embed_query(self, query: str) -> list[float]:
        """
        Embeds a single query string WITH the BGE instruction prefix.

        This asymmetric prefix is critical for retrieval quality — BGE models
        are trained to expect "Represent this sentence for searching relevant
        passages: <query>" at query time.

        Args:
            query: The user's natural-language question.

        Returns:
            A single 768-dimensional float vector.
        """
        self._assert_loaded()
        prefixed = QUERY_INSTRUCTION + query.strip()
        embedding = self._model.encode(
            [prefixed],
            normalize_embeddings=NORMALIZE,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embedding[0].tolist()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def embedding_dim(self) -> int:
        return EMBEDDING_DIM

    @property
    def model_name(self) -> str:
        return MODEL_NAME

    @property
    def device(self) -> str:
        return self._device

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _assert_loaded(self) -> None:
        if self._model is None:
            raise RuntimeError(
                "[EMBEDDER] Model not loaded. Call load() or use get_embedder()."
            )


# ---------------------------------------------------------------------------
# GeminiEmbedder
# ---------------------------------------------------------------------------

class GeminiEmbedder:
    """
    Wraps the Google Gemini embedding-2 API.
    """

    def __init__(self) -> None:
        self.api_key = os.environ.get("GEMINI_API_KEY", "")
        self._model_name = "models/gemini-embedding-001"
        self._device = "cloud"
        self._is_loaded = False

    def load(self) -> None:
        if not self.api_key:
            logger.warning("[EMBEDDER] GEMINI_API_KEY is not set. Cloud embedding requests will fail.")
        self._is_loaded = True
        logger.info(f"[EMBEDDER] GeminiEmbedder initialized using {self._model_name} (cloud).")

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def embedding_dim(self) -> int:
        return 768

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def device(self) -> str:
        return self._device

    def embed_documents(
        self,
        texts: list[str],
        batch_size: int = BATCH_SIZE,
        show_progress: bool = True,
    ) -> list[list[float]]:
        if not texts:
            return []

        url = f"https://generativelanguage.googleapis.com/v1beta/{self._model_name}:batchEmbedContents?key={self.api_key}"
        embeddings: list[list[float]] = []
        total = len(texts)

        for start in range(0, total, batch_size):
            batch = texts[start : start + batch_size]
            requests = [
                {
                    "model": self._model_name,
                    "taskType": "RETRIEVAL_DOCUMENT",
                    "content": {"parts": [{"text": text}]},
                    "outputDimensionality": 768
                }
                for text in batch
            ]

            payload = {"requests": requests}
            max_attempts = 6
            attempt_delay = 5.0
            for attempt in range(max_attempts):
                try:
                    with httpx.Client(timeout=60.0) as client:
                        response = client.post(url, json=payload)
                        if response.status_code == 429:
                            logger.warning(
                                f"[EMBEDDER] Rate limited (429) on batch {start // batch_size + 1}. "
                                f"Retrying in {attempt_delay}s... (attempt {attempt + 1}/{max_attempts})"
                            )
                            time.sleep(attempt_delay)
                            attempt_delay *= 2
                            continue
                        if response.status_code != 200:
                            raise RuntimeError(f"Gemini Embedding API error: {response.text}")
                        data = response.json()
                        for emb in data.get("embeddings", []):
                            embeddings.append(emb["values"])
                        break
                except Exception as e:
                    if attempt == max_attempts - 1:
                        logger.error(f"[EMBEDDER] Failed to embed batch: {e}")
                        raise
                    time.sleep(attempt_delay)
                    attempt_delay *= 2

            # Small safety delay between successful batches
            time.sleep(1.0)

            if show_progress:
                done = min(start + batch_size, total)
                logger.info(f"[EMBEDDER] Embedded {done}/{total} documents via Gemini API.")

        return embeddings

    def embed_query(self, query: str) -> list[float]:
        url = f"https://generativelanguage.googleapis.com/v1beta/{self._model_name}:embedContent?key={self.api_key}"
        payload = {
            "model": self._model_name,
            "taskType": "RETRIEVAL_QUERY",
            "content": {"parts": [{"text": query.strip()}]},
            "outputDimensionality": 768
        }
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(url, json=payload)
                if response.status_code != 200:
                    raise RuntimeError(f"Gemini Embedding API error: {response.text}")
                data = response.json()
                return data["embedding"]["values"]
        except Exception as e:
            logger.error(f"[EMBEDDER] Failed to embed query: {e}")
            raise


class HuggingFaceEmbedder:
    """
    Wraps Hugging Face Serverless Inference API for BAAI/bge-base-en-v1.5.
    """
    def __init__(self) -> None:
        self.api_token = os.environ.get("HF_API_TOKEN", "")
        self._model_name = "BAAI/bge-base-en-v1.5"
        self._device = "cloud-hf"
        self._is_loaded = False

    def load(self) -> None:
        self._is_loaded = True
        logger.info(f"[EMBEDDER] HuggingFaceEmbedder initialized using {self._model_name} (cloud API).")

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def embedding_dim(self) -> int:
        return 768

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def device(self) -> str:
        return self._device

    def _call_api(self, texts: list[str]) -> list[list[float]]:
        headers = {}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"
            
        url = f"https://router.huggingface.co/hf-inference/models/{self._model_name}/pipeline/feature-extraction"
        payload = {"inputs": texts, "options": {"wait_for_model": True}}

        with httpx.Client(timeout=60.0) as client:
            response = client.post(url, json=payload, headers=headers)
            if response.status_code != 200:
                raise RuntimeError(f"HuggingFace Inference API error: {response.text}")
            res = response.json()
            # The API returns a list of list of floats, or a single list if one input
            # If it returns a list of floats (1D), wrap it in a list
            if isinstance(res, list) and len(res) > 0 and not isinstance(res[0], list):
                res = [res]
            return res

    def embed_documents(
        self,
        texts: list[str],
        batch_size: int = 20,
        show_progress: bool = True,
    ) -> list[list[float]]:
        if not texts:
            return []
        
        embeddings = []
        total = len(texts)
        for start in range(0, total, batch_size):
            batch = texts[start : start + batch_size]
            max_attempts = 3
            for attempt in range(max_attempts):
                try:
                    res = self._call_api(batch)
                    embeddings.extend(res)
                    break
                except Exception as e:
                    if attempt == max_attempts - 1:
                        logger.error(f"[EMBEDDER] HuggingFace embedding failed: {e}")
                        raise
                    time.sleep(3.0)
            
            if show_progress:
                done = min(start + batch_size, total)
                logger.info(f"[EMBEDDER] Embedded {done}/{total} documents via HF API.")
                
        return embeddings

    def embed_query(self, query: str) -> list[float]:
        prefixed_query = QUERY_INSTRUCTION + query
        headers = {}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"
            
        url = f"https://router.huggingface.co/hf-inference/models/{self._model_name}/pipeline/feature-extraction"
        payload = {"inputs": [prefixed_query], "options": {"wait_for_model": True}}
        
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(url, json=payload, headers=headers)
                if response.status_code != 200:
                    raise RuntimeError(f"HuggingFace Inference API error: {response.text}")
                res = response.json()
                # HF API returns a list of list of floats (or a list of floats if single input depending on response)
                if isinstance(res, list) and len(res) > 0:
                    if isinstance(res[0], list):
                        return res[0]
                    return res
                raise RuntimeError(f"Unexpected response format: {res}")
        except Exception as e:
            logger.error(f"[EMBEDDER] Failed to embed query via HF API: {e}")
            raise

