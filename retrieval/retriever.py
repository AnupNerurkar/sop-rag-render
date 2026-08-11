"""
retrieval/retriever.py
-----------------------
Retrieval Engine — Public API (Phase 6, upgraded with hybrid + reranking)

Full pipeline:
    RetrievalQuery
        ↓ QueryPreprocessor.preprocess()
    clean_query_text
        ↓ FilterBuilder.build()
    where_clause
        ↓ BGEEmbedder.embed_query()
    query_vector (768-dim)
        ↓ DenseSearchBackend  →  top_k_dense results
          BM25SearchBackend   →  top_k_bm25  results  (if use_bm25)
        ↓ RecipRankFusion.fuse()
    fused candidates (top_k_fusion)
        ↓ GroqListwiseReranker.rerank()            (if use_reranker)
    top_k_final results
        ↓ _build_results()
    RetrievalResponse

Mode summary (controlled per-query via RetrievalQuery flags):
    use_bm25=False, use_reranker=False  →  Dense only
    use_bm25=True,  use_reranker=False  →  Hybrid (Dense + BM25 + RRF)
    use_bm25=True,  use_reranker=True   →  Hybrid + Reranker  ← default
    use_bm25=False, use_reranker=True   →  Dense + Reranker
"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Optional

from retrieval.retrieval_schema import (
    DEFAULT_TOP_K,
    DEFAULT_TOP_K_FINAL,
    RetrievalFilter,
    RetrievalQuery,
    RetrievalResponse,
    RetrievalResult,
    SourceCitation,
)
from retrieval.filters import FilterBuilder
from retrieval.hybrid_search import (
    DenseSearchBackend,
    BM25SearchBackend,
    RawSearchResult,
)
from retrieval.fusion import RecipRankFusion

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_retriever_instance: Optional["Retriever"] = None


def get_retriever() -> "Retriever":
    """Returns the process-level Retriever singleton."""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = Retriever()
    return _retriever_instance


# ---------------------------------------------------------------------------
# Query Preprocessor
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "to", "of", "in", "on", "at", "by", "for", "with", "about", "as",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "and", "but", "or", "nor",
    "so", "yet", "both", "either", "not", "no", "nor", "only", "own",
    "same", "than", "too", "very", "just", "it", "its", "this", "that",
    "these", "those", "i", "me", "my", "we", "our", "you", "your",
    "he", "she", "they", "them", "their", "what", "which", "who",
})
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WHITESPACE_RE   = re.compile(r"\s+")


class QueryPreprocessor:
    def preprocess(self, text: str, remove_stopwords: bool = False) -> str:
        text = _CONTROL_CHAR_RE.sub(" ", text)
        text = _WHITESPACE_RE.sub(" ", text).strip().lower()

        # Split common run-on typos with 'the' followed by keyword
        for term in ["attendance", "admission", "scholarship", "examination", "library", "budget", "finance", "academics", "placement", "alumni", "fees", "billing", "mms"]:
            text = text.replace(f"the{term}", f"the {term}")

        if remove_stopwords:
            tokens = text.split()
            if len(tokens) > 5:
                tokens = [t for t in tokens if t not in _STOPWORDS]
            text = " ".join(tokens) if tokens else text
        return text


# ---------------------------------------------------------------------------
# Citation builder
# ---------------------------------------------------------------------------

def _build_citation(metadata: dict) -> SourceCitation:
    source_file = metadata.get("source_file", "")
    title       = metadata.get("title", "").strip()

    if not title:
        stem = os.path.splitext(os.path.basename(source_file))[0]
        stem = re.sub(r"^\d+[\.\s]+", "", stem).strip()
        display_name = stem or metadata.get("department", "Unknown Document")
    else:
        display_name = title

    return SourceCitation(
        doc_id          = metadata.get("doc_id", ""),
        source_file     = source_file,
        display_name    = display_name,
        department      = metadata.get("department", "General"),
        category        = metadata.get("category", "SOP"),
        version         = metadata.get("version", "1.0"),
        section_heading = metadata.get("section_heading", ""),
        chunk_index     = int(metadata.get("chunk_index", 0)),
        total_chunks    = int(metadata.get("total_chunks", 0)),
    )


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class Retriever:
    """
    Orchestrates the full retrieval pipeline (dense + BM25 + RRF + reranker).

    All three phases (dense, BM25, reranker) are lazy-loaded on first use.
    Each can be disabled per-query via RetrievalQuery flags.
    """

    def __init__(self) -> None:
        self._preprocessor  = QueryPreprocessor()
        self._dense_backend = DenseSearchBackend()
        self._bm25_backend  = BM25SearchBackend()
        self._fuser         = RecipRankFusion(k=60)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, query: RetrievalQuery) -> RetrievalResponse:
        """
        Full retrieval pipeline. The single entry point for all callers.

        Args:
            query: Validated RetrievalQuery Pydantic model.

        Returns:
            RetrievalResponse with ranked results, citations, timing, and mode.
        """
        t_start = time.perf_counter()

        # --- 1. Preprocess ---
        clean_text = self._preprocessor.preprocess(
            query.text, remove_stopwords=query.remove_stopwords
        )

        # --- 2. Build metadata filter ---
        filter_builder = FilterBuilder(role=query.role, filters=query.filters)
        where_clause   = filter_builder.build()
        applied        = filter_builder.describe()

        # --- 3. Embed query (needed for dense; skip if dense disabled) ---
        from embeddings.embedder import get_embedder
        query_vector = get_embedder().embed_query(clean_text)

        # --- 4. Dense retrieval ---
        dense_results = self._dense_backend.search(
            query_text   = clean_text,
            query_vector = query_vector,
            where_clause = where_clause,
            n_results    = query.top_k_dense,
        )

        # --- 5. BM25 retrieval (optional) ---
        bm25_results: list[RawSearchResult] = []
        if query.use_bm25:
            try:
                bm25_results = self._bm25_backend.search(
                    query_text   = clean_text,
                    query_vector = query_vector,
                    where_clause = where_clause,
                    n_results    = query.top_k_bm25,
                )
            except Exception as e:
                logger.warning(f"[RETRIEVER] BM25 search failed, using dense only: {e}")

        # --- 6. Fuse with RRF ---
        if bm25_results:
            candidate_pool = self._fuser.fuse(
                [dense_results, bm25_results],
                n_results = query.top_k_fusion,
            )
        else:
            # No BM25 or BM25 failed → dense results are the candidate pool
            candidate_pool = dense_results[: query.top_k_fusion]

        # --- 7. Rerank (optional) ---
        # rerank() itself never raises for expected failure modes (timeout,
        # bad JSON, an invalid permutation, the circuit breaker being open)
        # -- it reports them through the `reranked` boolean instead, which
        # is what makes the mode label below trustworthy. A try/except here
        # would only mask a genuine bug in that contract.
        reranked = False
        rerank_method: Optional[str] = None
        if query.use_reranker and candidate_pool:
            from retrieval.reranker import get_reranker
            candidate_pool, reranked = get_reranker().rerank(
                query      = clean_text,
                candidates = candidate_pool,
                top_k      = query.top_k_final,
            )
            if reranked:
                rerank_method = "groq_listwise"
        else:
            # Use top_k (legacy) or top_k_final, whichever the caller set
            final_k = query.top_k_final if query.use_reranker else query.top_k
            candidate_pool = candidate_pool[:final_k]

        # --- 8. Determine retrieval mode label ---
        if reranked and bm25_results:
            mode = "hybrid+rerank"
        elif reranked:
            mode = "dense+rerank"
        elif bm25_results:
            mode = "hybrid"
        else:
            mode = "dense"

        # --- 9. Build typed results ---
        results = self._build_results(candidate_pool[:query.top_k], mode)

        latency_ms = (time.perf_counter() - t_start) * 1000
        logger.info(
            f"[RETRIEVER] '{clean_text[:60]}' | {mode} | "
            f"results={len(results)} | {latency_ms:.0f}ms"
        )

        return RetrievalResponse(
            query_text       = query.text,
            clean_query_text = clean_text,
            role             = query.role,
            results          = results,
            total_results    = len(results),
            top_k_requested  = query.top_k,
            latency_ms       = round(latency_ms, 2),
            reranked         = reranked,
            rerank_method    = rerank_method,
            retrieval_mode   = mode,
            applied_filters  = applied,
        )

    def retrieve_by_text(
        self,
        text:             str,
        role:             str            = "Public",
        top_k:            int            = DEFAULT_TOP_K,
        department:       Optional[str]  = None,
        category:         Optional[str]  = None,
        doc_id:           Optional[str]  = None,
        version:          Optional[str]  = None,
        remove_stopwords: bool           = False,
        use_bm25:         bool           = True,
        use_reranker:     bool           = False,
        top_k_dense:      int            = 25,
        top_k_bm25:       int            = 25,
        top_k_fusion:     int            = 25,
        top_k_final:      int            = DEFAULT_TOP_K_FINAL,
    ) -> RetrievalResponse:
        """Convenience wrapper — builds a RetrievalQuery and calls retrieve()."""
        query = RetrievalQuery(
            text             = text,
            role             = role,
            top_k            = top_k,
            filters          = RetrievalFilter(
                department=department, category=category,
                doc_id=doc_id, version=version,
            ),
            remove_stopwords = remove_stopwords,
            use_bm25         = use_bm25,
            use_reranker     = use_reranker,
            top_k_dense      = top_k_dense,
            top_k_bm25       = top_k_bm25,
            top_k_fusion     = top_k_fusion,
            top_k_final      = top_k_final,
        )
        return self.retrieve(query)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_results(
        self,
        raw: list[RawSearchResult],
        mode: str,
    ) -> list[RetrievalResult]:
        results: list[RetrievalResult] = []
        for rank, r in enumerate(raw, start=1):
            citation = _build_citation(r.metadata)
            logger.info(
                f"[RETRIEVAL] [CHUNK] rank={rank} doc_id={r.metadata.get('doc_id')} "
                f"chunk_id={r.chunk_id} filepath={r.metadata.get('source_file')} score={r.score:.4f}"
            )
            results.append(
                RetrievalResult(
                    rank           = rank,
                    chunk_id       = r.chunk_id,
                    content        = r.content,
                    score          = r.score,
                    distance       = r.distance,
                    rerank_score   = r.rerank_score,
                    retrieval_mode = mode,
                    citation       = citation,
                    metadata       = r.metadata,
                )
            )
        return results
