"""
retrieval/bm25.py
------------------
BM25 keyword retrieval backend.

Corpus: all 627 chunks loaded from SQLite on first search (lazy, ~0.3s).
Index:  BM25Okapi from rank_bm25, rebuilt whenever new chunks are ingested.
Filter: post-filters BM25 results using a small metadata where-clause
        interpreter supporting $eq, $in, $and.

Why BM25 alongside dense vectors?
    Dense retrieval excels at semantic similarity but misses exact keyword
    matches. BM25 excels at exact term matching (acronyms, procedure codes,
    specific SOP names like "8A", "Sub Process 1.3"). RRF fusion captures
    both signals.

Tokenization strategy for SOP corpus:
    - Lowercase
    - Expand hyphens / slashes to spaces (common in SOP step numbering)
    - Remove non-alphanumeric chars (except space)
    - Filter tokens shorter than 2 characters
    No stemming — VIT SOP terms are domain-specific and stemming
    ("admissions" → "admiss") destroys more than it normalizes.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# Tokenizer regex: keep alphanumeric, expand separators to space
_EXPAND_RE    = re.compile(r"[-/|]")
_STRIP_RE     = re.compile(r"[^a-z0-9\s]")
_WHITESPACE   = re.compile(r"\s+")


def _tokenize(text: str) -> list[str]:
    """Lightweight SOP-aware tokenizer for BM25 corpus and queries."""
    text = text.lower()
    text = _EXPAND_RE.sub(" ", text)
    text = _STRIP_RE.sub(" ", text)
    text = _WHITESPACE.sub(" ", text).strip()
    return [t for t in text.split() if len(t) >= 2]


# ---------------------------------------------------------------------------
# Where-clause interpreter (the vector store → Python predicate)
# ---------------------------------------------------------------------------

def _matches_where(metadata: dict, where: Optional[dict]) -> bool:
    """
    Evaluates a metadata where-clause against a metadata dict.

    Supports the operators produced by filters.FilterBuilder:
        {"field": {"$eq": value}}
        {"field": {"$in": [v1, v2, ...]}}
        {"$and": [condition, ...]}

    Returns True (passes) when where is None.
    """
    if where is None:
        return True

    if "$and" in where:
        return all(_matches_where(metadata, c) for c in where["$and"])

    # Single field condition
    if len(where) == 1:
        field_name, condition = next(iter(where.items()))
        if not isinstance(condition, dict):
            return False
        if "$eq" in condition:
            return metadata.get(field_name) == condition["$eq"]
        if "$in" in condition:
            return metadata.get(field_name) in condition["$in"]

    return True  # Unknown operator → pass through (safe default)


# ---------------------------------------------------------------------------
# Corpus entry
# ---------------------------------------------------------------------------

@dataclass
class CorpusEntry:
    chunk_id: str
    tokens:   list[str]
    content:  str
    metadata: dict


# ---------------------------------------------------------------------------
# BM25Index — lazy-loaded singleton
# ---------------------------------------------------------------------------

class BM25Index:
    """
    In-memory BM25 index over all chunks loaded from SQLite.

    Singleton accessed via get_bm25_index().
    Call invalidate() after new document ingestion to force a rebuild.
    """

    def __init__(self) -> None:
        self._corpus:  list[CorpusEntry] = []
        self._model    = None
        self._built    = False

    def build(self, db_path: Optional[str] = None) -> None:
        """Load all chunks from SQLite, tokenize, build BM25Okapi index."""
        from rank_bm25 import BM25Okapi
        import os

        if db_path is None:
            db_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "..", "ingestion_ledger.db"
            )

        logger.info("[BM25] Building index from active database...")
        import ledger
        conn = ledger.get_connection()
        cur  = conn.cursor()
        cur.execute(
            """
            SELECT chunk_id, content, doc_id, source_file, category,
                   department, access_level, version, section_heading,
                   chunk_index, total_chunks
            FROM chunks
            ORDER BY doc_id, chunk_index
            """
        )
        rows = cur.fetchall()
        conn.close()

        self._corpus = [
            CorpusEntry(
                chunk_id = row["chunk_id"],
                tokens   = _tokenize(row["content"]),
                content  = row["content"],
                metadata = {
                    "doc_id":          row["doc_id"],
                    "source_file":     row["source_file"],
                    "title":           "",
                    "category":        row["category"],
                    "department":      row["department"],
                    "access_level":    row["access_level"],
                    "version":         row["version"],
                    "upload_date":     "",
                    "section_heading": row["section_heading"] or "",
                    "chunk_index":     row["chunk_index"],
                    "total_chunks":    row["total_chunks"],
                },
            )
            for row in rows
        ]

        tokenized = [e.tokens for e in self._corpus]
        self._model = BM25Okapi(tokenized)
        self._built = True
        logger.info(f"[BM25] Index ready. {len(self._corpus)} documents indexed.")

    def search(
        self,
        query: str,
        where_clause: Optional[dict],
        n_results: int,
        fetch_multiplier: int = 4,
    ) -> list[tuple[str, str, float, dict]]:
        """
        Run BM25 search, post-filter by where_clause, return top-n.

        Returns list of (chunk_id, content, normalized_score, metadata).

        The fetch_multiplier ensures we always retrieve enough candidates
        before post-filtering by RBAC / metadata constraints.
        """
        if not self._built:
            raise RuntimeError("[BM25] Index not built. Call build() first.")

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        raw_scores = self._model.get_scores(query_tokens)   # numpy array, one per doc

        # Normalize scores to [0, 1] by max score
        max_score = float(raw_scores.max()) if raw_scores.max() > 0 else 1.0

        # Collect (index, score) pairs, sort descending, fetch more than needed
        fetch_n = min(n_results * fetch_multiplier, len(self._corpus))
        top_indices = raw_scores.argsort()[::-1][:fetch_n]

        results = []
        for idx in top_indices:
            entry = self._corpus[idx]
            if not _matches_where(entry.metadata, where_clause):
                continue
            norm_score = float(raw_scores[idx]) / max_score
            results.append((entry.chunk_id, entry.content, norm_score, entry.metadata))
            if len(results) >= n_results:
                break

        return results

    def invalidate(self) -> None:
        """Force rebuild on next search (call after new document ingestion)."""
        self._built = False
        self._corpus = []
        self._model  = None
        logger.info("[BM25] Index invalidated. Will rebuild on next search.")

    @property
    def is_built(self) -> bool:
        return self._built

    @property
    def corpus_size(self) -> int:
        return len(self._corpus)


# ---------------------------------------------------------------------------
# Process-level singleton
# ---------------------------------------------------------------------------

_bm25_index: Optional[BM25Index] = None


def get_bm25_index(db_path: Optional[str] = None) -> BM25Index:
    """Returns the process-level BM25Index singleton, building it on first call."""
    global _bm25_index
    if _bm25_index is None:
        _bm25_index = BM25Index()
    if not _bm25_index.is_built:
        _bm25_index.build(db_path=db_path)
    return _bm25_index
