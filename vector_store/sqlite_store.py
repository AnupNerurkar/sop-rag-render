"""
vector_store/sqlite_store.py
-----------------------------
The vector store. Everything is local: vectors live in the same database file
as the ledger's `chunks` table (ingestion_ledger.db, via
ledger.get_connection()), and that colocation is what makes the citation JOIN
possible. Nothing here talks to the network.

Similarity search is a brute-force numpy dot product. Vectors are L2-normalized
at write time, so the dot product IS cosine similarity, and:

    distance = 1.0 - dot      (the cosine distance pgvector used to return)
    score    = dot

Score parity with the old pgvector backend is load-bearing even now that it is
gone, because the thresholds calibrated against it remain:
response_schema.compute_confidence
hard-thresholds at similarity < 0.70, so a scale mismatch would silently turn
every answer into the insufficient-evidence fallback without raising anything.

At ~750 chunks the full matrix is ~2.3 MB, so a scan is single-digit ms even on
a Pi 3B — negligible beside the HuggingFace embedding round-trip. No matrix
cache: it would need invalidating across upsert, delete_by_doc_id and the
ingestion path, which is where the bugs would be.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ledger import get_connection

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 768

# Logical name of the vector collection. Retained because the ledger rows and
# the operational scripts refer to it; it is not a vector collection.
COLLECTION_NAME = "vit_institutional_kb"

# Which access levels each role may read. A role always sees its own level and
# everything below it.
ACCESS_HIERARCHY: dict[str, list[str]] = {
    "Admin":   ["Public", "Student", "Faculty", "Admin"],
    "Faculty": ["Public", "Student", "Faculty"],
    "Student": ["Public", "Student"],
    "Public":  ["Public"],
}


def _condition_to_sql(field: str, op_dict: dict) -> tuple[str, list]:
    """Converts a single {field: {"$eq"|"$in": value}} condition to SQL."""
    if "$eq" in op_dict:
        return f"{field} = ?", [op_dict["$eq"]]
    if "$in" in op_dict:
        values = list(op_dict["$in"])
        placeholders = ",".join(["?"] * len(values))
        return f"{field} IN ({placeholders})", values
    raise ValueError(f"Unsupported where-clause operator for field '{field}': {op_dict}")


def _where_clause_to_sql(where_clause: Optional[dict]) -> tuple[str, list]:
    """
    Converts a nested metadata where-clause (see retrieval/filters.py) into a
    SQL WHERE fragment (without the "WHERE" keyword) and its params.
    Supports $and, $or, $eq, $in — the operators filters.py actually emits.
    """
    if not where_clause:
        return "", []
    if "$and" in where_clause:
        parts, params = [], []
        for cond in where_clause["$and"]:
            sql, p = _where_clause_to_sql(cond)
            parts.append(f"({sql})")
            params.extend(p)
        return " AND ".join(parts), params
    if "$or" in where_clause:
        parts, params = [], []
        for cond in where_clause["$or"]:
            sql, p = _where_clause_to_sql(cond)
            parts.append(f"({sql})")
            params.extend(p)
        return " OR ".join(parts), params
    field, op_dict = next(iter(where_clause.items()))
    return _condition_to_sql(field, op_dict)


_vector_store_instance = None


def get_vector_store() -> "SQLiteVectorStore":
    """Process-level vector store singleton. There is one backend: local SQLite."""
    global _vector_store_instance
    if _vector_store_instance is None:
        logger.info("[VECTOR_STORE] Instantiating SQLiteVectorStore (local)")
        _vector_store_instance = SQLiteVectorStore()
        _vector_store_instance.initialize()
    return _vector_store_instance


def _pack(vector) -> bytes:
    """L2-normalizes a vector and packs it as a float32 blob."""
    v = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(v))
    if norm > 0.0:
        v = v / norm
    return v.astype(np.float32).tobytes()


def _normalize_query(vector) -> np.ndarray:
    """L2-normalizes the query vector. The embedder is not assumed to return unit vectors."""
    q = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(q))
    if norm > 0.0:
        q = q / norm
    return q


class SQLiteVectorStore:
    """
    Local SQLite + numpy vector store.

    Implements the vector-store API so that
    retrieval/hybrid_search.py depends only on this surface.
    """

    def __init__(self) -> None:
        self._initialized = False

    def initialize(self) -> None:
        if self._initialized:
            return

        conn = get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding BLOB NOT NULL,
                dim INTEGER NOT NULL DEFAULT 768,
                access_level TEXT,
                department TEXT,
                category TEXT,
                title TEXT,
                version TEXT
            )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_doc_id ON embeddings(doc_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_access_level ON embeddings(access_level)")
            conn.commit()
            self._initialized = True
            logger.info("[SQLITE_VEC] Local SQLite vector store initialized.")
        except Exception as e:
            conn.rollback()
            logger.error(f"[SQLITE_VEC] Failed to initialize: {e}")
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def upsert(self, payloads: list) -> int:
        if not payloads:
            return 0

        self.initialize()
        conn = get_connection()
        cursor = conn.cursor()
        total = 0
        try:
            for p in payloads:
                meta = p.metadata
                blob = _pack(p.embedding)
                cursor.execute("""
                INSERT INTO embeddings (
                    id, doc_id, content, embedding, dim,
                    access_level, department, category, title, version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    doc_id = excluded.doc_id,
                    content = excluded.content,
                    embedding = excluded.embedding,
                    dim = excluded.dim,
                    access_level = excluded.access_level,
                    department = excluded.department,
                    category = excluded.category,
                    title = excluded.title,
                    version = excluded.version
                """, (
                    p.chunk_id,
                    meta.get("doc_id", ""),
                    p.content,
                    blob,
                    len(blob) // 4,
                    meta.get("access_level", "Public"),
                    meta.get("department", ""),
                    meta.get("category", ""),
                    meta.get("title", ""),
                    meta.get("version", "1.0"),
                ))
                total += 1
            conn.commit()
            logger.info(f"[SQLITE_VEC] Upserted {total} vectors.")
            return total
        except Exception as e:
            conn.rollback()
            logger.error(f"[SQLITE_VEC] Upsert failed: {e}")
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def _rank(self, cursor, where_sql: str, params: list, query_vector, n_results: int):
        """
        Phase 1 of a search: scan candidate vectors and return the top-n
        [(id, distance), ...] sorted by distance ascending.

        Only id + embedding are selected — pulling `content` for every row
        would multiply per-query I/O by the size of the corpus text.
        """
        sql = "SELECT id, embedding FROM embeddings"
        if where_sql:
            sql += f" WHERE {where_sql}"
        cursor.execute(sql, params)
        rows = cursor.fetchall()
        if not rows:
            return []

        ids = [r["id"] for r in rows]
        matrix = np.frombuffer(b"".join(r["embedding"] for r in rows), dtype=np.float32)
        if matrix.size != len(rows) * EMBEDDING_DIM:
            # A truncated or wrong-dimension blob would silently reshape into
            # garbage and return nonsense rankings, so fail loudly instead.
            raise ValueError(
                f"[SQLITE_VEC] Embedding blob size mismatch: got {matrix.size} floats "
                f"for {len(rows)} rows (expected {len(rows) * EMBEDDING_DIM})."
            )
        matrix = matrix.reshape(len(rows), EMBEDDING_DIM)

        sims = matrix @ _normalize_query(query_vector)

        k = min(n_results, sims.shape[0])
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]

        return [(ids[i], 1.0 - float(sims[i])) for i in top]

    def query_with_filter(
        self,
        query_embedding: list,
        where_clause: Optional[dict],
        n_results: int = 10,
    ) -> list:
        """
        Nearest-neighbour search with a pre-built metadata filter
        (see retrieval/filters.py). Mirrors the query_with_filter
        contract so retrieval/hybrid_search.py works against either backend.
        """
        self.initialize()
        conn = get_connection()
        cursor = conn.cursor()
        try:
            where_sql, where_params = _where_clause_to_sql(where_clause)
            ranked = self._rank(cursor, where_sql, where_params, query_embedding, n_results)
            if not ranked:
                return []

            # Phase 2: hydrate only the winners. The chunk_index/total_chunks/
            # section_heading/source_file columns live on the ledger's `chunks`
            # table, not on `embeddings` — without this join every citation
            # defaults to chunk_index 0 and every "open source" click lands in
            # the same spot (see commit 409ee15).
            #
            # The filter never reaches this statement: the only predicate here
            # is on e.id, which we control. filters.py emits bare column names
            # that also exist on `chunks`, so letting it near a join would raise
            # "ambiguous column name".
            ids = [cid for cid, _ in ranked]
            placeholders = ",".join(["?"] * len(ids))
            cursor.execute(
                f"""
                SELECT e.id, e.content, e.doc_id, e.access_level, e.department,
                       e.category, e.title, e.version,
                       c.chunk_index, c.total_chunks, c.section_heading, c.source_file
                FROM embeddings e
                LEFT JOIN chunks c ON c.chunk_id = e.id
                WHERE e.id IN ({placeholders})
                """,
                ids,
            )
            by_id = {r["id"]: r for r in cursor.fetchall()}

            results = []
            for chunk_id, distance in ranked:
                r = by_id.get(chunk_id)
                if r is None:
                    continue
                results.append({
                    "chunk_id": chunk_id,
                    "content":  r["content"],
                    "distance": distance,
                    "score":    1.0 - distance,
                    "metadata": {
                        "doc_id":          r["doc_id"],
                        "access_level":    r["access_level"],
                        "department":      r["department"],
                        "category":        r["category"],
                        "title":           r["title"],
                        "version":         r["version"],
                        "chunk_index":     r["chunk_index"] if r["chunk_index"] is not None else 0,
                        "total_chunks":    r["total_chunks"] if r["total_chunks"] is not None else 0,
                        "section_heading": r["section_heading"] or "",
                        "source_file":     r["source_file"] or "",
                    },
                })
            return results
        except Exception as e:
            logger.error(f"[SQLITE_VEC] query_with_filter failed: {e}")
            raise
        finally:
            conn.close()

    def query(
        self,
        embedding: list,
        role: str,
        n_results: int = 10,
        department_filter: Optional[str] = None,
    ) -> list:
        """
        RBAC-filtered nearest-neighbour search.

        Return shape uses "id" as the key
        (not "chunk_id") and there is deliberately no "distance" key.
        """
        self.initialize()

        allowed_levels = ACCESS_HIERARCHY.get(role, ["Public"])

        conn = get_connection()
        cursor = conn.cursor()
        try:
            placeholders = ",".join(["?"] * len(allowed_levels))
            where_sql = f"access_level IN ({placeholders})"
            params = list(allowed_levels)
            if department_filter:
                where_sql += " AND department = ?"
                params.append(department_filter)

            ranked = self._rank(cursor, where_sql, params, embedding, n_results)
            if not ranked:
                return []

            ids = [cid for cid, _ in ranked]
            id_placeholders = ",".join(["?"] * len(ids))
            cursor.execute(
                f"""
                SELECT id, content, doc_id, access_level, department, category, title, version
                FROM embeddings WHERE id IN ({id_placeholders})
                """,
                ids,
            )
            by_id = {r["id"]: r for r in cursor.fetchall()}

            results = []
            for chunk_id, distance in ranked:
                r = by_id.get(chunk_id)
                if r is None:
                    continue
                results.append({
                    "id":      chunk_id,
                    "content": r["content"],
                    "score":   1.0 - distance,
                    "metadata": {
                        "doc_id":       r["doc_id"],
                        "access_level": r["access_level"],
                        "department":   r["department"],
                        "category":     r["category"],
                        "title":        r["title"],
                        "version":      r["version"],
                    },
                })
            return results
        except Exception as e:
            logger.error(f"[SQLITE_VEC] Query failed: {e}")
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def get_collection_stats(self) -> dict:
        self.initialize()
        conn = get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT COUNT(*) as cnt FROM embeddings")
            count = cursor.fetchone()["cnt"]

            cursor.execute("SELECT COUNT(DISTINCT doc_id) as doc_cnt FROM embeddings")
            doc_count = cursor.fetchone()["doc_cnt"]

            from ledger import DB_PATH
            return {
                "count": count,
                "vector_count": count,
                "document_count": doc_count,
                "db_path": DB_PATH,
                "backend": "sqlite",
            }
        except Exception as e:
            logger.error(f"[SQLITE_VEC] Failed to get stats: {e}")
            return {
                "count": 0, "vector_count": 0, "document_count": 0,
                "db_path": "", "backend": "sqlite",
            }
        finally:
            conn.close()

    def delete_by_doc_id(self, doc_id: str) -> None:
        self.initialize()
        conn = get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("DELETE FROM embeddings WHERE doc_id = ?", (doc_id,))
            conn.commit()
            logger.info(f"[SQLITE_VEC] Deleted all embeddings for doc_id: {doc_id}")
        except Exception as e:
            conn.rollback()
            logger.error(f"[SQLITE_VEC] Failed to delete embeddings for doc_id {doc_id}: {e}")
            raise
        finally:
            conn.close()

    def collection_exists(self) -> bool:
        self.initialize()
        return self.get_collection_stats()["count"] > 0
