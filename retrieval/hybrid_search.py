"""
retrieval/hybrid_search.py
---------------------------
Search backend implementations: DenseSearchBackend (the vector store +
BGE) and BM25SearchBackend (SQLite FTS5, despite the class name -- see its
own docstring). Both implement the same BaseSearchBackend interface so
retrieval/retriever.py can call either uniformly; retriever.py runs both
and fuses the results itself via retrieval/fusion.py's RecipRankFusion
(Reciprocal Rank Fusion, k=60) -- that fusion is not done in this module.

Interface contract:
    BaseSearchBackend.search(query_text, query_vector, where_clause, n_results)
        → list[RawSearchResult]

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

# HybridSearchEngine (a second, duplicate RRF implementation with its own
# alpha-weighted fusion) was removed in Phase 8 -- never instantiated
# anywhere; retrieval/retriever.py calls DenseSearchBackend and
# BM25SearchBackend directly and does its own fusion via retrieval/fusion.py's
# RecipRankFusion, which is what's actually live.
