"""
retrieval/
----------
Phase 6: Retrieval Engine

Exports:
    Retriever          — main retrieval class (use get_retriever() for singleton)
    get_retriever      — returns process-level singleton
    RetrievalQuery     — query specification model
    RetrievalResponse  — response model with results, citations, and stats
    RetrievalResult    — single ranked result
    RetrievalFilter    — optional metadata pre-filters
    SourceCitation     — source provenance model

Keyword search is SQLite FTS5 (retrieval/fts5.py), not a process-level
index -- there is no singleton to import here. See fts5.ensure_schema /
fts5.search.
"""

from retrieval.retrieval_schema import (
    RetrievalFilter,
    RetrievalQuery,
    RetrievalResponse,
    RetrievalResult,
    SourceCitation,
)
from retrieval.retriever import Retriever, get_retriever
from retrieval.fusion import RecipRankFusion
from retrieval.reranker import get_reranker

__all__ = [
    "Retriever", "get_retriever",
    "RetrievalQuery", "RetrievalResponse", "RetrievalResult",
    "RetrievalFilter", "SourceCitation",
    "RecipRankFusion",
    "get_reranker",
]
