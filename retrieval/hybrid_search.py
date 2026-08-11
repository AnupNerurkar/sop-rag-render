"""
retrieval/hybrid_search.py
---------------------------
Phase 6: Search Backend Abstraction (BM25-Ready)

Defines the interface that all search backends implement, plus the
HybridSearchEngine that combines them.

Current state  : DenseSearchBackend only (the vector store + BGE vectors).
Future state   : BM25SearchBackend (SQLite FTS or rank_bm25) + RRF fusion.

Architecture:

    HybridSearchEngine
        │
        ├── DenseSearchBackend   ← active now
        │       └── SQLiteVectorStore.query_with_filter()
        │
        └── BM25SearchBackend    ← STUB (raises NotImplementedError)
                └── SQLite FTS / rank_bm25  (Phase 6.5 or later)

Hybrid score fusion (when BM25 is added):
    Reciprocal Rank Fusion (RRF):
        score(d) = Σ_b  1 / (k + rank_b(d))     k=60 (standard default)
    Or weighted linear combination:
        score(d) = alpha * dense_score + (1 - alpha) * bm25_score

    alpha is configurable (default 1.0 = dense only until BM25 is implemented).

Interface contract:
    BaseSearchBackend.search(query_text, query_vector, where_clause, n_results)
        → list[RawSearchResult]

    Both dense and BM25 backends implement this interface.
    Dense backend uses query_vector; ignores query_text.
    BM25  backend uses query_text;   ignores query_vector (and where_clause format).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Raw result (internal, backend-level)
# ---------------------------------------------------------------------------

@dataclass
class RawSearchResult:
    """
    Internal result type returned by search backends.
    Converted to RetrievalResult by the Retriever after citation building.
    """
    chunk_id:     str
    content:      str
    score:        float          # Relevance score [0, 1]. Higher = better.
    distance:     Optional[float]  # Cosine distance for dense hits; None for FTS-only hits
                                    # (a BM25/FTS score is not on the same scale as cosine
                                    # distance -- synthesizing one, as the old code did with
                                    # `1.0 - norm_score`, corrupted confidence and thresholding
                                    # with a value that only looked like a real distance).
    metadata:     dict           = field(default_factory=dict)
    backend:      str            = "dense"  # "dense" | "bm25" | "hybrid" | "hybrid+rerank"
    rerank_score: Optional[float]= None     # Set by reranker.py after cross-encoder scoring


# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------

class BaseSearchBackend(ABC):
    """
    Abstract base class for all search backends.

    Implementors:
        DenseSearchBackend  — cosine similarity
        BM25SearchBackend   — Full-text keyword search (future)

    Contract:
        - Returns at most `n_results` results.
        - Results are sorted by score descending (highest relevance first).
        - Never raises on empty results — returns an empty list.
        - Never modifies `where_clause` — it is owned by the caller.
    """

    @abstractmethod
    def search(
        self,
        query_text:   str,
        query_vector: list[float],
        where_clause: Optional[dict],
        n_results:    int,
    ) -> list[RawSearchResult]:
        ...

    @property
    @abstractmethod
    def backend_name(self) -> str:
        ...


# ---------------------------------------------------------------------------
# Dense backend (the vector store + BGE)
# ---------------------------------------------------------------------------

class DenseSearchBackend(BaseSearchBackend):
    """
    Semantic similarity search using the vector store and BAAI/bge-base-en-v1.5.

    Uses query_vector; query_text is ignored (the vector already encodes it).
    where_clause is passed directly to the vector store.query_with_filter().
    """

    @property
    def backend_name(self) -> str:
        return "dense"

    def search(
        self,
        query_text:   str,
        query_vector: list[float],
        where_clause: Optional[dict],
        n_results:    int,
    ) -> list[RawSearchResult]:
        """
        Runs a nearest-neighbour query.

        Args:
            query_text:   Ignored by this backend (vector already encodes it).
            query_vector: 768-dim BGE embedding of the (preprocessed) query.
            where_clause: Optional the vector store metadata where-clause from filters.py.
            n_results:    Number of results to return.

        Returns:
            List of RawSearchResult sorted by score descending.
        """
        from vector_store.sqlite_store import get_vector_store

        store = get_vector_store()
        raw = store.query_with_filter(
            query_embedding = query_vector,
            where_clause    = where_clause,
            n_results       = n_results,
        )

        return [
            RawSearchResult(
                chunk_id = r["chunk_id"],
                content  = r["content"],
                score    = r["score"],
                distance = r["distance"],
                metadata = r["metadata"],
                backend  = "dense",
            )
            for r in raw
        ]


# ---------------------------------------------------------------------------
# BM25 backend (SQLite FTS5)
# ---------------------------------------------------------------------------

class BM25SearchBackend(BaseSearchBackend):
    """
    Keyword retrieval via SQLite FTS5 (retrieval/fts5.py).

    Class name kept as BM25SearchBackend even though the implementation is
    FTS5 -- callers (Retriever, evaluate_retrieval.py) refer to this as "the
    keyword backend" / "bm25 mode", and FTS5's own ranking function is also
    called bm25(). Renaming would touch a lot of call sites for a label with
    no behavioral consequence.
    """

    @property
    def backend_name(self) -> str:
        return "bm25"

    def search(
        self,
        query_text:   str,
        query_vector: list[float],
        where_clause: Optional[dict],
        n_results:    int,
    ) -> list[RawSearchResult]:
        """
        Runs FTS5 keyword search.
        Uses query_text; query_vector is ignored.
        where_clause is translated to SQL and applied as a JOIN predicate
        (see retrieval/fts5.py), not a Python post-filter.
        """
        import ledger
        from retrieval import fts5
        from vector_store.sqlite_store import _where_clause_to_sql

        where_sql, where_params = _where_clause_to_sql(where_clause)

        conn = ledger.get_connection()
        try:
            raw = fts5.search(conn, query_text, where_sql, where_params, n_results)
        finally:
            conn.close()

        return [
            RawSearchResult(
                chunk_id = chunk_id,
                content  = content,
                score    = norm_score,
                distance = None,   # FTS score isn't on the cosine scale -- see RawSearchResult.distance
                metadata = metadata,
                backend  = "bm25",
            )
            for chunk_id, content, norm_score, metadata in raw
        ]


# ---------------------------------------------------------------------------
# Hybrid search engine
# ---------------------------------------------------------------------------

class HybridSearchEngine:
    """
    Combines multiple search backends with configurable fusion.

    Current state (alpha=1.0): delegates entirely to DenseSearchBackend.
    When BM25 is implemented, set alpha < 1.0 to enable fusion.

    Fusion strategy: Reciprocal Rank Fusion (RRF)
        final_score(d) = Σ_b  1 / (RRF_K + rank_b(d))
        where RRF_K=60 is the standard constant that limits the influence
        of very high-ranked documents.

    Args:
        alpha:       Weight of dense scores in [0, 1]. 1.0 = dense only.
        rrf_k:       RRF constant. Default 60 (standard literature value).
        dense_extra: Multiplier on n_results passed to dense backend to ensure
                     enough candidates for fusion. Default 2.
    """

    RRF_K = 60

    def __init__(
        self,
        alpha:       float = 1.0,
        dense_extra: int   = 2,
    ) -> None:
        self._alpha       = alpha
        self._dense_extra = dense_extra
        self._dense       = DenseSearchBackend()
        self._bm25        = BM25SearchBackend()

    def search(
        self,
        query_text:   str,
        query_vector: list[float],
        where_clause: Optional[dict],
        n_results:    int,
    ) -> list[RawSearchResult]:
        """
        Runs the configured backends and returns fused, ranked results.

        With alpha=1.0 (default): pure dense retrieval, no fusion overhead.
        With alpha<1.0: RRF fusion across dense + BM25 results.
        """
        if self._alpha >= 1.0 or self._alpha < 0.0:
            # Pure dense — no fusion needed
            return self._dense.search(query_text, query_vector, where_clause, n_results)

        # --- Hybrid path (BM25 not yet implemented — guard) ---
        fetch_k = n_results * self._dense_extra

        dense_results = self._dense.search(query_text, query_vector, where_clause, fetch_k)
        try:
            bm25_results = self._bm25.search(query_text, query_vector, where_clause, fetch_k)
        except NotImplementedError:
            logger.warning(
                "[HYBRID] BM25 backend not implemented; falling back to dense-only."
            )
            return dense_results[:n_results]

        return self._rrf_fuse(dense_results, bm25_results, n_results)

    def _rrf_fuse(
        self,
        dense_results: list[RawSearchResult],
        bm25_results:  list[RawSearchResult],
        n_results:     int,
    ) -> list[RawSearchResult]:
        """
        Reciprocal Rank Fusion over two ranked result lists.

        For each unique chunk, sums   1/(RRF_K + rank)   across backends.
        Re-ranks by fused score. Preserves metadata from dense results
        (they carry full the vector store metadata).
        """
        scores: dict[str, float] = {}
        meta:   dict[str, RawSearchResult] = {}

        for rank, r in enumerate(dense_results, start=1):
            scores[r.chunk_id] = scores.get(r.chunk_id, 0.0) + (1.0 / (self.RRF_K + rank))
            meta[r.chunk_id]   = r

        for rank, r in enumerate(bm25_results, start=1):
            scores[r.chunk_id] = scores.get(r.chunk_id, 0.0) + (1.0 / (self.RRF_K + rank))
            if r.chunk_id not in meta:
                meta[r.chunk_id] = r

        fused = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:n_results]

        return [
            RawSearchResult(
                chunk_id = cid,
                content  = meta[cid].content,
                score    = round(score, 6),
                distance = meta[cid].distance,
                metadata = meta[cid].metadata,
                backend  = "hybrid",
            )
            for cid, score in fused
        ]
