"""
vector_store/chroma_store.py
-----------------------------
Phase 5: ChromaDB Persistent Vector Store

Wraps chromadb.PersistentClient with:
  - Cosine distance collection (mandatory for normalized BGE vectors)
  - RBAC-filtered queries (access_level metadata where-clause)
  - Idempotent upsert (safe for re-runs and incremental ingestion)
  - Source citation support (metadata returned with every result)
  - Singleton process-level client (one PersistentClient per process)

RBAC Access Hierarchy (enforced at query time via ChromaDB metadata filter):
    Admin   → no filter     (sees all levels)
    Faculty → Public, Student, Faculty
    Student → Public, Student
    Public  → Public only

Collection Design:
    Name   : vit_institutional_kb
    Metric : cosine  (BGE vectors are L2-normalized — cosine == dot product)
    Storage: vector_store/chroma_db/  (persisted to disk)
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

COLLECTION_NAME = "vit_institutional_kb"
CHROMA_DB_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")

# RBAC access hierarchy: maps a role to the set of access_level values it may see.
ACCESS_HIERARCHY: dict[str, list[str]] = {
    "Admin":   ["Public", "Student", "Faculty", "Admin"],
    "Faculty": ["Public", "Student", "Faculty"],
    "Student": ["Public", "Student"],
    "Public":  ["Public"],
}

# Default number of results to return per query.
DEFAULT_N_RESULTS = 10

# ChromaDB upsert batch size — keeps memory usage flat for large collections.
UPSERT_BATCH_SIZE = 100


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_vector_store_instance = None


def get_chroma_store():
    """
    Returns the process-level vector store singleton.
    Dynamically chooses between ChromaStore and PGVectorStore based on environment variables.
    """
    global _vector_store_instance
    if _vector_store_instance is None:
        backend = os.getenv("VECTOR_STORE_BACKEND", "").strip().lower()
        db_url = os.getenv("DATABASE_URL", "")
        is_postgres_db = db_url.startswith("postgresql://") or db_url.startswith("postgres://")
        chroma_host = os.getenv("CHROMA_SERVER_HOST")

        # Ordered so an explicit VECTOR_STORE_BACKEND always wins, and the
        # fallback is SQLite rather than Chroma — chromadb isn't in
        # requirements.txt, so falling through to ChromaStore raises
        # ImportError at initialize() rather than degrading.
        if backend in ("supabase", "pgvector"):
            from vector_store.pgvector_store import PGVectorStore
            logger.info("[VECTOR_STORE] Instantiating PGVectorStore (pgvector backend, explicit)")
            _vector_store_instance = PGVectorStore()
        elif backend == "sqlite":
            from vector_store.sqlite_store import SQLiteVectorStore
            logger.info("[VECTOR_STORE] Instantiating SQLiteVectorStore (local SQLite backend, explicit)")
            _vector_store_instance = SQLiteVectorStore()
        elif backend == "chroma" or chroma_host:
            logger.info("[VECTOR_STORE] Instantiating ChromaStore (ChromaDB backend)")
            _vector_store_instance = ChromaStore()
        elif is_postgres_db:
            from vector_store.pgvector_store import PGVectorStore
            logger.info("[VECTOR_STORE] Instantiating PGVectorStore (inferred from Postgres DATABASE_URL)")
            _vector_store_instance = PGVectorStore()
        else:
            from vector_store.sqlite_store import SQLiteVectorStore
            logger.info("[VECTOR_STORE] Instantiating SQLiteVectorStore (default)")
            _vector_store_instance = SQLiteVectorStore()


        _vector_store_instance.initialize()
    return _vector_store_instance


# ---------------------------------------------------------------------------
# ChromaStore
# ---------------------------------------------------------------------------

class ChromaStore:
    """
    Persistent ChromaDB vector store for the VIT institutional knowledge base.

    Public API:
        initialize()                           — connect, create/get collection
        upsert(payloads)                       — batch upsert EmbeddingPayload list
        query(embedding, role, n_results, ...) — RBAC-filtered nearest-neighbour search
        get_collection_stats()                 — count, metadata summary
        delete_by_doc_id(doc_id)               — remove all chunks for a document
        collection_exists()                    — True if collection is non-empty
    """

    def __init__(self, db_path: str = CHROMA_DB_PATH) -> None:
        self._db_path   = db_path
        self._client    = None
        self._collection = None

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """
        Connects to ChromaDB Client (PersistentClient locally or HttpClient for cloud)
        and gets/creates the collection.
        """
        try:
            import chromadb
        except ImportError:
            raise ImportError(
                "chromadb is not installed. Run: pip install chromadb"
            )

        host = os.getenv("CHROMA_SERVER_HOST")
        port = os.getenv("CHROMA_SERVER_PORT")
        ssl = os.getenv("CHROMA_SERVER_SSL", "false").lower() in ("true", "1", "yes")
        auth_provider = os.getenv("CHROMA_SERVER_AUTH_PROVIDER")
        auth_credentials = os.getenv("CHROMA_SERVER_AUTH_CREDENTIALS")

        if host:
            logger.info(f"[CHROMA] Connecting to HttpClient at: {host}:{port or 8000} (ssl={ssl})")
            client_kwargs = {
                "host": host,
                "ssl": ssl,
            }
            if port:
                client_kwargs["port"] = int(port)
            if auth_provider and auth_credentials:
                client_kwargs["settings"] = chromadb.Settings(
                    chroma_client_auth_provider=auth_provider,
                    chroma_client_auth_credentials=auth_credentials,
                )
            self._client = chromadb.HttpClient(**client_kwargs)
        else:
            os.makedirs(self._db_path, exist_ok=True)
            logger.info(f"[CHROMA] Connecting to PersistentClient at: {self._db_path}")
            self._client = chromadb.PersistentClient(path=self._db_path)

        self._collection = self._client.get_or_create_collection(
            name     = COLLECTION_NAME,
            metadata = {"hnsw:space": "cosine"},
        )

        count = self._collection.count()
        logger.info(
            f"[CHROMA] Collection '{COLLECTION_NAME}' ready. "
            f"Current vector count: {count}"
        )

    # ------------------------------------------------------------------
    # Upsert
    # ------------------------------------------------------------------

    def upsert(self, payloads: list) -> int:
        """
        Batch-upserts EmbeddingPayload objects into ChromaDB.

        Upsert is idempotent: re-upserting the same chunk_id overwrites the
        existing entry without creating duplicates. Safe for re-runs.

        Args:
            payloads: List of EmbeddingPayload (from embed_pipeline.py).

        Returns:
            Number of vectors successfully upserted.
        """
        self._assert_initialized()
        if not payloads:
            return 0

        total_upserted = 0
        total = len(payloads)

        for start in range(0, total, UPSERT_BATCH_SIZE):
            batch = payloads[start : start + UPSERT_BATCH_SIZE]

            ids        = [p.chunk_id  for p in batch]
            documents  = [p.content   for p in batch]
            embeddings = [p.embedding for p in batch]
            metadatas  = [p.metadata  for p in batch]

            self._collection.upsert(
                ids        = ids,
                documents  = documents,
                embeddings = embeddings,
                metadatas  = metadatas,
            )

            total_upserted += len(batch)
            done = min(start + UPSERT_BATCH_SIZE, total)
            logger.info(f"[CHROMA] Upserted {done}/{total} vectors.")

        logger.info(
            f"[CHROMA] Upsert complete. "
            f"Total in collection: {self._collection.count()}"
        )
        return total_upserted

    # ------------------------------------------------------------------
    # Query (RBAC-filtered semantic search)
    # ------------------------------------------------------------------

    def query(
        self,
        query_embedding: list[float],
        role: str = "Public",
        n_results: int = DEFAULT_N_RESULTS,
        department: Optional[str] = None,
        category: Optional[str] = None,
    ) -> list[dict]:
        """
        Performs a nearest-neighbour semantic search with RBAC enforcement.

        Access is controlled by the `role` parameter:
          - "Public"  → only Public chunks
          - "Student" → Public + Student chunks
          - "Faculty" → Public + Student + Faculty chunks
          - "Admin"   → all chunks (no filter)

        Args:
            query_embedding: 768-dim query vector from BGEEmbedder.embed_query().
            role:            Caller's role for access filtering.
            n_results:       Number of results to return.
            department:      Optional metadata pre-filter (e.g. 'Admissions').
            category:        Optional metadata pre-filter (e.g. 'SOP').

        Returns:
            List of result dicts, each containing:
                chunk_id, content, distance, score, metadata (full)
            Sorted by relevance (lowest cosine distance first).
        """
        self._assert_initialized()

        # Build where-clause for RBAC + optional filters
        where = self._build_where_clause(role, department, category)

        query_kwargs: dict = {
            "query_embeddings": [query_embedding],
            "n_results":        min(n_results, self._collection.count() or 1),
            "include":          ["documents", "metadatas", "distances"],
        }
        if where:
            query_kwargs["where"] = where

        raw = self._collection.query(**query_kwargs)

        return self._format_results(raw)

    def query_with_filter(
        self,
        query_embedding: list[float],
        where_clause:    Optional[dict],
        n_results:       int = DEFAULT_N_RESULTS,
    ) -> list[dict]:
        """
        Nearest-neighbour search with a pre-built ChromaDB where-clause.

        Used by Phase 6 (retrieval/hybrid_search.py).
        The caller is responsible for building the where_clause (filters.py
        handles RBAC + metadata filter construction).

        Args:
            query_embedding: 768-dim normalized BGE vector.
            where_clause:    ChromaDB metadata filter dict, or None for unfiltered.
            n_results:       Number of results to return.

        Returns:
            List of result dicts — same format as query().
        """
        self._assert_initialized()

        safe_n = min(n_results, self._collection.count() or 1)

        query_kwargs: dict = {
            "query_embeddings": [query_embedding],
            "n_results":        safe_n,
            "include":          ["documents", "metadatas", "distances"],
        }
        if where_clause:
            query_kwargs["where"] = where_clause

        raw = self._collection.query(**query_kwargs)
        return self._format_results(raw)

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def delete_by_doc_id(self, doc_id: str) -> int:
        """
        Removes all chunk vectors belonging to a document from ChromaDB.
        Used during version supersession — old document's vectors are purged
        before the new version's vectors are upserted.

        Returns:
            Number of vectors deleted.
        """
        self._assert_initialized()

        existing = self._collection.get(
            where   = {"doc_id": {"$eq": doc_id}},
            include = [],
        )
        ids_to_delete = existing["ids"]

        if ids_to_delete:
            self._collection.delete(ids=ids_to_delete)
            logger.info(
                f"[CHROMA] Deleted {len(ids_to_delete)} vectors for doc_id={doc_id[:12]}..."
            )
        return len(ids_to_delete)

    def get_collection_stats(self) -> dict:
        """Returns a summary dict of collection statistics."""
        self._assert_initialized()
        count = self._collection.count()
        return {
            "collection_name": COLLECTION_NAME,
            "vector_count":    count,
            "db_path":         self._db_path,
        }

    def collection_exists(self) -> bool:
        """Returns True if the collection has at least one vector."""
        self._assert_initialized()
        return self._collection.count() > 0

    @property
    def is_initialized(self) -> bool:
        return self._collection is not None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_where_clause(
        self,
        role: str,
        department: Optional[str],
        category: Optional[str],
    ) -> Optional[dict]:
        """
        Constructs a ChromaDB metadata where-clause that enforces RBAC and
        optional department/category pre-filters.

        ChromaDB logical operators:
            $and → all conditions must match
            $or  → any condition must match
            $in  → value is in a list (used for multi-level RBAC)
            $eq  → exact match
        """
        allowed_levels = ACCESS_HIERARCHY.get(role, ACCESS_HIERARCHY["Public"])

        conditions: list[dict] = []

        # RBAC filter: access_level must be in the allowed list
        if len(allowed_levels) == 1:
            conditions.append({"access_level": {"$eq": allowed_levels[0]}})
        else:
            conditions.append({"access_level": {"$in": allowed_levels}})

        if department:
            conditions.append({"department": {"$eq": department}})

        if category:
            conditions.append({"category": {"$eq": category}})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    def _format_results(self, raw: dict) -> list[dict]:
        """
        Converts raw ChromaDB query output into clean result dicts.

        Each result includes:
            chunk_id : str    — deterministic chunk identifier
            content  : str    — raw chunk text (for RAG context window)
            distance : float  — cosine distance (lower = more similar)
            score    : float  — relevance score = 1 - distance (higher = better)
            metadata : dict   — full ChunkMetadata (source, department, access_level…)
        """
        results: list[dict] = []

        ids        = raw.get("ids",        [[]])[0]
        documents  = raw.get("documents",  [[]])[0]
        metadatas  = raw.get("metadatas",  [[]])[0]
        distances  = raw.get("distances",  [[]])[0]

        for chunk_id, content, metadata, distance in zip(
            ids, documents, metadatas, distances
        ):
            results.append({
                "chunk_id": chunk_id,
                "content":  content,
                "distance": round(distance, 6),
                "score":    round(1.0 - distance, 6),
                "metadata": metadata,
            })

        return results

    def _assert_initialized(self) -> None:
        if self._collection is None:
            raise RuntimeError(
                "[CHROMA] ChromaStore not initialized. "
                "Call initialize() or use get_chroma_store()."
            )
