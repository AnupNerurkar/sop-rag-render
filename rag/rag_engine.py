"""
rag/rag_engine.py
------------------
RAG Engine — converts BuiltPrompt into a grounded answer via the configured LLM.

Supported backends (set LLM_BACKEND in .env):
    ollama  (default) — local Ollama server  (OllamaClient)
    hf                — HuggingFace Inference API (HFClient)

Pipeline:
    BuiltPrompt.messages
        ↓  <client>.chat() / .chat_stream()
    raw LLM response
        ↓  RAGEngine._build_response()
    RAGResponse

This module knows nothing about retrieval, chunking, or embeddings.
It only cares about: prompt in, answer out.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Iterator, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

from rag.ollama_client import OllamaClient, OllamaConfig, get_ollama_client
from rag.prompt_schema import BuiltPrompt

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RAGResponse
# ---------------------------------------------------------------------------

class RAGResponse(BaseModel):
    """
    The complete output of a RAG generation call.
    All fields required for the API response layer (Phase 9).
    """

    answer:             str   = Field(..., description="The generated answer text.")
    model_name:         str   = Field(..., description="Ollama model that produced this answer.")
    finish_reason:      str   = Field(default="stop",
                                      description="'stop' | 'length' — why generation ended.")

    # Token accounting
    prompt_tokens:      int   = Field(default=0)
    completion_tokens:  int   = Field(default=0)
    total_tokens:       int   = Field(default=0)

    # Timing
    latency_ms:         float = Field(default=0.0,
                                      description="Wall-clock time from request to response (ms).")
    generation_time_ms: float = Field(default=0.0,
                                      description="Model eval time reported by Ollama (ms).")

    # Provenance (from BuiltPrompt)
    chunks_used:        int   = Field(default=0)
    has_conflicts:      bool  = Field(default=False)
    template_used:      str   = Field(default="default")

    @property
    def tokens_per_second(self) -> float:
        if self.generation_time_ms > 0:
            return round(self.completion_tokens / (self.generation_time_ms / 1000), 1)
        return 0.0

    def summary(self) -> str:
        return (
            f"model={self.model_name} | "
            f"tokens={self.prompt_tokens}+{self.completion_tokens} | "
            f"latency={self.latency_ms:.0f}ms | "
            f"gen={self.generation_time_ms:.0f}ms | "
            f"finish={self.finish_reason}"
        )


# ---------------------------------------------------------------------------
# RAGEngine
# ---------------------------------------------------------------------------

class RAGEngine:
    """
    Orchestrates BuiltPrompt → RAGResponse using an OllamaClient.

    Swap note: pass a different OllamaClient implementation to target a
    different model or provider without changing any calling code.
    """

    def __init__(self, client: Optional[OllamaClient] = None) -> None:
        self._client = client or get_ollama_client()

    @property
    def client(self):
        """The active backend client (whichever LLM_BACKEND selected -- see
        get_rag_engine() below). Public so diagnostics (verify_rag.py) can
        probe the real active backend instead of reaching into a private
        attribute or hardcoding an assumption about which one is live."""
        return self._client

    # ------------------------------------------------------------------
    # Non-streaming
    # ------------------------------------------------------------------

    def generate(
        self,
        built_prompt:   BuiltPrompt,
        *,
        temperature:    Optional[float] = None,
        max_tokens:     Optional[int]   = None,
        top_p:          Optional[float] = None,
        repeat_penalty: Optional[float] = None,
    ) -> RAGResponse:
        """
        Sends built_prompt.messages to Ollama and returns a complete RAGResponse.

        Args:
            built_prompt:   Output of PromptBuilder.build().
            temperature:    Per-call override (uses OllamaConfig default if None).
            max_tokens:     Per-call override.
            top_p:          Per-call override.
            repeat_penalty: Per-call override.

        Raises:
            OllamaConnectionError, OllamaTimeoutError,
            OllamaModelNotFoundError, OllamaResponseError
        """
        logger.info(
            "[RAG] generate | chunks=%d conflicts=%d template=%s",
            built_prompt.chunks_included,
            len(built_prompt.conflicts),
            built_prompt.template_used.value,
        )

        raw = self._client.chat(
            built_prompt.messages,
            temperature    = temperature,
            max_tokens     = max_tokens,
            top_p          = top_p,
            repeat_penalty = repeat_penalty,
        )

        return self._build_response(raw, built_prompt)

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def generate_stream(
        self,
        built_prompt:   BuiltPrompt,
        *,
        temperature:    Optional[float] = None,
        max_tokens:     Optional[int]   = None,
        top_p:          Optional[float] = None,
        repeat_penalty: Optional[float] = None,
    ) -> Iterator[str]:
        """
        Streaming generation. Yields text chunk strings as they arrive.

        Usage:
            for chunk in engine.generate_stream(prompt):
                print(chunk, end="", flush=True)

        Note: token counts and latency are not available from streaming.
        Use generate() if you need those metrics.
        """
        logger.info(
            "[RAG] generate_stream | chunks=%d template=%s",
            built_prompt.chunks_included,
            built_prompt.template_used.value,
        )
        yield from self._client.chat_stream(
            built_prompt.messages,
            temperature    = temperature,
            max_tokens     = max_tokens,
            top_p          = top_p,
            repeat_penalty = repeat_penalty,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_response(self, raw: dict, built_prompt: BuiltPrompt) -> RAGResponse:
        p_tokens = raw.get("prompt_tokens", 0)
        c_tokens = raw.get("completion_tokens", 0)
        return RAGResponse(
            answer             = raw["answer"],
            model_name         = raw.get("model_name", self._client.model),
            finish_reason      = raw.get("finish_reason", "stop"),
            prompt_tokens      = p_tokens,
            completion_tokens  = c_tokens,
            total_tokens       = p_tokens + c_tokens,
            latency_ms         = raw.get("latency_ms", 0.0),
            generation_time_ms = raw.get("generation_time_ms", 0.0),
            chunks_used        = built_prompt.chunks_included,
            has_conflicts      = built_prompt.has_conflicts,
            template_used      = built_prompt.template_used.value,
        )


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_engine_instance: Optional[RAGEngine] = None


def get_rag_engine(
    config: Optional[OllamaConfig] = None,
    client=None,
) -> RAGEngine:
    """
    Returns the process-level RAGEngine singleton.

    Backend selection via LLM_BACKEND environment variable:
        LLM_BACKEND=ollama  (default) — uses OllamaClient
        LLM_BACKEND=hf              — uses HFClient (HuggingFace Inference API)

    Pass config or client to configure the first call; subsequent calls
    return the same instance regardless of arguments.
    """
    global _engine_instance
    if _engine_instance is None:
        if client is not None:
            _engine_instance = RAGEngine(client)
        else:
            backend = os.environ.get("LLM_BACKEND", "ollama").strip().lower()
            if backend == "hf":
                from rag.hf_client import get_hf_client
                logger.info("[RAG_ENGINE] Using HuggingFace backend (LLM_BACKEND=hf)")
                _engine_instance = RAGEngine(get_hf_client())
            elif backend == "gemini":
                from rag.gemini_client import get_gemini_client
                logger.info("[RAG_ENGINE] Using Gemini backend (LLM_BACKEND=gemini)")
                _engine_instance = RAGEngine(get_gemini_client())
            elif backend == "groq":
                from rag.groq_client import get_groq_client
                logger.info("[RAG_ENGINE] Using Groq backend (LLM_BACKEND=groq)")
                _engine_instance = RAGEngine(get_groq_client())
            else:
                logger.info("[RAG_ENGINE] Using Ollama backend (LLM_BACKEND=ollama)")
                _engine_instance = RAGEngine(
                    get_ollama_client(config) if config else get_ollama_client()
                )
    return _engine_instance


def reset_rag_engine() -> None:
    """Clears the singleton — primarily for testing."""
    global _engine_instance
    _engine_instance = None
