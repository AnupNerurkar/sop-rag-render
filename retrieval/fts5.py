"""
retrieval/fts5.py
------------------
SQLite FTS5 keyword search, replacing the old in-memory rank_bm25 index
(retrieval/bm25.py, deleted).

Why: the old index was a process-global Python object, built once from the
`chunks` table on first search and never invalidated on delete. Deleting a
document removed it from SQLite but not from that in-memory corpus, so a
"deleted" document kept surfacing in keyword search and got cited -- the
reported bug. A second uvicorn worker would have the same problem in the
other direction (new documents invisible until its own first build), and
holding every chunk's text and tokens in RAM permanently isn't free on a
5.7GB box either.

FTS5 fixes this structurally rather than procedurally: chunks_fts is a
table in the same database file as `chunks`, kept in sync by SQL triggers
(see ensure_schema). There is no invalidate() to forget to call and no
per-process copy to fall out of sync -- a delete that commits to `chunks`
is, by construction, already reflected in `chunks_fts`.

Ranking: FTS5's built-in bm25() auxiliary function (same algorithm family
as the old rank_bm25 package, different implementation -- unrelated code,
coincidentally similar name). Smaller/more negative bm25() values mean a
better match; results here are flipped and normalized to [0, 1] within
each result set so the sign convention matches every other score in the
pipeline ("higher is better").
"""

from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _sanitize_query(text: str) -> str:
    """
    Converts free text into a safe FTS5 MATCH expression.

    Raw user text passed straight to MATCH throws on FTS5 query-syntax
    characters (? - " * etc) -- exactly the kind of exception that used to
    degrade a query silently to dense-only. Tokenizing and rejoining as
    quoted terms sidesteps the syntax entirely (a quoted string can never be
    misparsed as an operator) and preserves the old bm25.py's "any keyword
    can contribute a match" semantics, which RRF then reconciles against
    the dense signal.
    """
    tokens = [t for t in _TOKEN_RE.findall(text.lower()) if len(t) >= 2]
    if not tokens:
        return ""
    return " OR ".join(f'"{t}"' for t in tokens)


def ensure_schema(conn) -> None:
    """
    Creates the FTS5 table and sync triggers if they don't exist, and
    backfills from `chunks` if the table is new. Idempotent -- safe to call
    on every startup.

    Standalone (not external-content) FTS5 table: `chunks` has only an
    implicit rowid, and it is not stable across a VACUUM, so tying the FTS5
    table's identity to it would be a latent corruption bug. chunk_id (the
    real primary key) is stored as an UNINDEXED column instead and used for
    all sync/delete operations -- a linear scan of an FTS5 shadow table at
    this corpus size (low thousands of rows) is sub-millisecond.
    """
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            chunk_id UNINDEXED,
            content,
            section_heading,
            tokenize = 'unicode61 remove_diacritics 2',
            prefix = '2 3'
        )
    """)

    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS chunks_fts_ai AFTER INSERT ON chunks BEGIN
            INSERT INTO chunks_fts(chunk_id, content, section_heading)
            VALUES (new.chunk_id, new.content, new.section_heading);
        END
    """)
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS chunks_fts_ad AFTER DELETE ON chunks BEGIN
            DELETE FROM chunks_fts WHERE chunk_id = old.chunk_id;
        END
    """)
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS chunks_fts_au
        AFTER UPDATE OF content, section_heading ON chunks BEGIN
            DELETE FROM chunks_fts WHERE chunk_id = old.chunk_id;
            INSERT INTO chunks_fts(chunk_id, content, section_heading)
            VALUES (new.chunk_id, new.content, new.section_heading);
        END
    """)

    fts_count = conn.execute("SELECT COUNT(*) AS n FROM chunks_fts").fetchone()["n"]
    if fts_count == 0:
        chunk_count = conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
        if chunk_count > 0:
            conn.execute("""
                INSERT INTO chunks_fts(chunk_id, content, section_heading)
                SELECT chunk_id, content, section_heading FROM chunks
            """)
            logger.info(f"[FTS5] Backfilled {chunk_count} existing chunks into chunks_fts.")
    conn.commit()


def search(
    conn,
    query_text: str,
    where_sql: str,
    where_params: list,
    n_results: int,
) -> list[tuple[str, str, float, dict]]:
    """
    Runs a keyword search and returns up to n_results
    (chunk_id, content, normalized_score, metadata) tuples, sorted by
    relevance descending.

    where_sql/where_params come from vector_store.sqlite_store's existing
    where-clause translator, applied here as a JOIN predicate against
    `chunks` rather than a Python post-filter -- so a restrictive RBAC
    filter can no longer silently return fewer than n_results just because
    the over-fetch multiplier guessed too low (the old bm25.py's
    fetch_multiplier=4 was exactly that guess).
    """
    match_expr = _sanitize_query(query_text)
    if not match_expr:
        return []

    # title comes from a correlated scalar subquery, not a JOIN to
    # `documents` -- `documents` has its own access_level/department/
    # category/version/doc_id columns mirroring `chunks`, so joining it
    # made every bare column name in where_sql ambiguous (filters.py emits
    # unqualified names like "access_level = ?", which is exactly what the
    # comment in sqlite_store.py's query_with_filter warns about). The
    # subquery keeps `chunks` (aliased c) as the only table in scope for
    # the WHERE clause, so those bare names resolve without qualification.
    sql = """
        SELECT c.chunk_id, c.content, bm25(chunks_fts) AS rank,
               c.doc_id,
               (SELECT title FROM documents WHERE doc_id = c.doc_id) AS title,
               c.category, c.department, c.access_level,
               c.version, c.section_heading, c.chunk_index, c.total_chunks,
               c.source_file
        FROM chunks_fts
        JOIN chunks c ON c.chunk_id = chunks_fts.chunk_id
        WHERE chunks_fts MATCH ?
    """
    params: list = [match_expr]
    if where_sql:
        sql += f" AND ({where_sql})"
        params.extend(where_params)
    sql += " ORDER BY rank LIMIT ?"
    params.append(n_results)

    cursor = conn.cursor()
    try:
        cursor.execute(sql, params)
    except Exception as e:
        # A malformed MATCH expression or FTS5 syntax edge case should
        # degrade to "no keyword results" for this query, not crash it --
        # RRF fusion falls back to dense-only, same as before.
        logger.warning(f"[FTS5] Query failed for expr {match_expr!r}: {e}")
        return []

    rows = cursor.fetchall()
    if not rows:
        return []

    # FTS5's bm25(): smaller (more negative) = better match. Flip sign so
    # "higher is better" holds everywhere else in the pipeline, then
    # normalize to [0, 1] within this result set.
    raw_scores = [-r["rank"] for r in rows]
    max_score = max(raw_scores) if raw_scores else 1.0
    if max_score <= 0:
        max_score = 1.0

    results = []
    for row, raw in zip(rows, raw_scores):
        norm_score = raw / max_score
        metadata = {
            "doc_id":          row["doc_id"],
            "source_file":     row["source_file"] or "",
            "title":           row["title"] or "",
            "category":        row["category"],
            "department":      row["department"],
            "access_level":    row["access_level"],
            "version":         row["version"],
            "section_heading": row["section_heading"] or "",
            "chunk_index":     row["chunk_index"],
            "total_chunks":    row["total_chunks"],
        }
        results.append((row["chunk_id"], row["content"], norm_score, metadata))
    return results
