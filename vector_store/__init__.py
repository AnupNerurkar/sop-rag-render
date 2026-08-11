"""
vector_store/
-------------
Local SQLite vector store and the indexing pipeline that fills it.

Exports:
    SQLiteVectorStore — local vector store with RBAC filtering
    get_vector_store  — process-level singleton accessor
    run_indexing      — end-to-end orchestrator (embed → upsert → stamp)
"""

from vector_store.sqlite_store import SQLiteVectorStore, get_vector_store
from vector_store.index_pipeline import run_indexing

__all__ = ["SQLiteVectorStore", "get_vector_store", "run_indexing"]
