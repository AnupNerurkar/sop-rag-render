"""
backend/document_manager.py
-----------------------------
Admin Document Management: upload → ingest → embed → index.

Handles:
  - SHA-256 deduplication (reject exact duplicate content)
  - Version supersession detection (same department, new version)
  - Auto-trigger ingestion → chunking → embedding → vector indexing
  - Support for .docx files (the VIT SOP corpus format)
  - File staging to data/staging/ before ingestion

Public API:
  ingest_uploaded_file(filename, content_bytes)  → IngestionResult
  get_document_stats()                           → dict
  get_ingestion_logs(limit)                      → list[dict]
"""

from __future__ import annotations

import hashlib
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Ensure project root on path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Global indexing status
# ---------------------------------------------------------------------------
# Ingestion (extract → chunk → embed → index) runs synchronously inside the
# request that triggered it, but FastAPI serves other requests concurrently via
# its threadpool. This shared flag lets any client (including guests) poll
# GET /api/indexing-status and show a "knowledge base updating" banner while a
# document is being processed.
import threading

_indexing_lock = threading.Lock()
_indexing_state: dict = {"active": False, "filename": None, "started_at": None}

# Serializes actual ingestion runs. Each upload/approval schedules its own
# independent FastAPI background task, so without this, multiple uploads
# submitted close together would run their full extract/chunk/embed/index
# pipelines concurrently — racing on _indexing_state/_last_result and, worse,
# competing for the same constrained CPU/DB/embedding-API resources, which is
# what caused larger/later uploads in a batch to silently stall mid-pipeline.
_ingestion_run_lock = threading.Lock()


def get_indexing_state() -> dict:
    with _indexing_lock:
        state = dict(_indexing_state)
    state["last_result"] = get_last_result()
    return state


def _set_indexing(active: bool, filename: Optional[str] = None) -> None:
    with _indexing_lock:
        _indexing_state["active"] = active
        _indexing_state["filename"] = filename if active else None
        _indexing_state["started_at"] = datetime.utcnow().isoformat() if active else None


# ---------------------------------------------------------------------------
# Last-result tracking (for background/async ingestion callers to poll)
# ---------------------------------------------------------------------------

_last_result_lock = threading.Lock()
_last_result: Optional[dict] = None


def get_last_result() -> Optional[dict]:
    with _last_result_lock:
        return dict(_last_result) if _last_result else None


def _store_last_result(result: "IngestionResult") -> None:
    global _last_result
    with _last_result_lock:
        _last_result = {
            "filename":       result.filename,
            "status":         result.status,
            "doc_id":         result.doc_id,
            "chunks_created": result.chunks_created,
            "vectors_added":  result.vectors_added,
            "error":          result.error,
            "completed_at":   datetime.utcnow().isoformat(),
        }


def run_ingestion_background(
    filename: str,
    content_bytes: bytes,
    forced_access_level: Optional[str] = None,
    uploaded_by: Optional[str] = None,
) -> None:
    """
    Runs ingest_uploaded_file() and records the outcome for polling clients.

    Intended for use as a FastAPI BackgroundTasks target — the HTTP request
    returns immediately (avoiding Render free-tier request/health-check
    timeouts on slow embedding calls) while /api/indexing-status reports
    progress and, once done, the recorded last_result (ingest_uploaded_file
    records it automatically, so this just needs to invoke it).
    """
    ingest_uploaded_file(filename, content_bytes, forced_access_level, uploaded_by)

STAGING_DIR = Path(_PROJECT_ROOT) / "data" / "staging"
# Admin-uploadable formats. Text extraction for each is handled by the shared
# ingestion pipeline (ingestion_pipeline.ingest_document); chunking, embedding,
# vector indexing and BM25 indexing are identical regardless of format.
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc"}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class IngestionResult:
    filename:       str
    status:         str            # "ingested" | "duplicate" | "superseded" | "failed"
    doc_id:         Optional[str]  = None
    department:     Optional[str]  = None
    version:        Optional[str]  = None
    chunks_created: int            = 0
    vectors_added:  int            = 0
    superseded_doc: Optional[str]  = None
    error:          Optional[str]  = None
    processing_ms:  float          = 0.0

    @property
    def success(self) -> bool:
        return self.status in ("ingested", "superseded")


# ---------------------------------------------------------------------------
# Main ingestion entry point
# ---------------------------------------------------------------------------

def ingest_uploaded_file(
    filename: str,
    content_bytes: bytes,
    forced_access_level: Optional[str] = None,
    uploaded_by: Optional[str] = None,
) -> IngestionResult:
    """
    Full pipeline for an admin-uploaded document:
      1. Validate file type
      2. SHA-256 dedup check
      3. Save to staging directory
      4. Phase 2: ingest (text extraction + metadata + ledger)
      5. Phase 3: chunk (already called by ingestion_pipeline.ingest_document)
      6. Phase 4+5: embed + index to the vector store

    Args:
        forced_access_level: if provided, overrides the content-heuristic
            access_level detected during ingestion (e.g. "Student" for
            committee-head SOP submissions approved by an admin).
        uploaded_by: username to record as the document's contributor (the
            admin uploader, or the committee head whose submission was approved).

    Returns IngestionResult with full status.
    """
    import time
    with _ingestion_run_lock:  # serialize — see _ingestion_run_lock's docstring
        t_start = time.perf_counter()
        _set_indexing(True, filename)
        try:
            result = _ingest_uploaded_file_inner(
                filename, content_bytes, forced_access_level, uploaded_by, t_start
            )
        except Exception as exc:
            logger.error("[DOC_MANAGER] Unhandled ingestion error for '%s': %s", filename, exc, exc_info=True)
            result = IngestionResult(filename=filename, status="failed", error=str(exc))
        finally:
            _set_indexing(False)
        _store_last_result(result)
        return result


def _ingest_uploaded_file_inner(
    filename: str,
    content_bytes: bytes,
    forced_access_level: Optional[str],
    uploaded_by: Optional[str],
    t_start: float,
) -> IngestionResult:

    import time

    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return IngestionResult(
            filename=filename,
            status="failed",
            error=f"Unsupported file type '{ext}'. Accepted: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    if not content_bytes:
        return IngestionResult(filename=filename, status="failed", error="File is empty.")

    # --- SHA-256 dedup: check if identical content already indexed ---
    file_hash = hashlib.sha256(content_bytes).hexdigest()

    try:
        import ledger
        ledger.initialize_db()
        existing = ledger.get_document_by_doc_id(file_hash)
        if existing and existing.get("status") not in ("superseded",):
            return IngestionResult(
                filename=filename,
                status="duplicate",
                doc_id=file_hash,
                department=existing.get("department"),
                version=existing.get("version"),
                error="Identical document already indexed (SHA-256 match).",
            )
    except Exception as exc:
        logger.warning("[DOC_MANAGER] Ledger dedup check failed: %s", exc)

    # --- Save to staging directory ---
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    dest_path = STAGING_DIR / filename

    try:
        with open(dest_path, "wb") as f:
            f.write(content_bytes)
        logger.info("[DOC_MANAGER] Saved '%s' to staging (%d bytes)", filename, len(content_bytes))
    except Exception as exc:
        return IngestionResult(
            filename=filename, status="failed",
            error=f"Could not save file: {exc}",
        )

    # --- Phase 2+3: ingest_document (text extraction + chunking + ledger) ---
    try:
        from ingestion_pipeline import ingest_document
        from chunk_schema import IngestionSummary
        from chunker import run_chunking

        summary = IngestionSummary(
            run_id=datetime.utcnow().strftime("%Y%m%d_%H%M%S"),
            started_at=datetime.utcnow().isoformat(),
        )
        result = ingest_document(str(dest_path), summary)

        if result is None:
            # Skipped (unchanged hash) or failed
            if summary.total_docs_skipped > 0:
                return IngestionResult(
                    filename=filename,
                    status="duplicate",
                    doc_id=file_hash,
                    error="Document content unchanged — already indexed.",
                )
            error_msg = "; ".join(summary.errors) if summary.errors else "Ingestion returned no output."
            return IngestionResult(filename=filename, status="failed", error=error_msg)

        doc_record, cleaned_text = result

        if forced_access_level:
            from chunk_schema import AccessLevel
            doc_record.access_level = AccessLevel(forced_access_level)
            import ledger as _ledger
            _ledger.upsert_document(doc_record.to_ledger_dict())

        # Run chunking
        all_chunks = run_chunking([doc_record], {doc_record.doc_id: cleaned_text})
        chunks_for_doc = all_chunks.get(doc_record.doc_id, [])
        n_chunks = len(chunks_for_doc)

    except Exception as exc:
        logger.error("[DOC_MANAGER] Phase 2/3 failed for '%s': %s", filename, exc, exc_info=True)
        return IngestionResult(filename=filename, status="failed", error=f"Ingestion failed: {exc}")

    # --- Phase 4+5: embed + index ---
    n_vectors = 0
    index_error: Optional[str] = None
    try:
        from vector_store.index_pipeline import run_indexing
        index_summary = run_indexing(doc_id=doc_record.doc_id)
        n_vectors = index_summary.chunks_upserted
        if index_summary.errors:
            index_error = "; ".join(index_summary.errors)
            logger.warning("[DOC_MANAGER] Indexing errors for '%s': %s", filename, index_summary.errors)
    except Exception as exc:
        logger.error("[DOC_MANAGER] Phase 4/5 failed for '%s': %s", filename, exc, exc_info=True)
        index_error = f"Embedding/indexing failed: {exc}"
        # Non-fatal to ingestion — doc is chunked and saved, just not yet vectorized

    # --- Record contributor for the admin document registry ---
    if uploaded_by:
        try:
            import ledger as _ledger
            _ledger.set_document_uploader(doc_record.doc_id, uploaded_by)
        except Exception as exc:
            logger.warning("[DOC_MANAGER] Could not set uploader for '%s': %s", filename, exc)

    elapsed_ms = (time.perf_counter() - t_start) * 1000
    superseded = summary.total_docs_superseded > 0

    logger.info(
        "[DOC_MANAGER] '%s' ingested: chunks=%d vectors=%d superseded=%s elapsed=%.0fms",
        filename, n_chunks, n_vectors, superseded, elapsed_ms,
    )

    return IngestionResult(
        filename=filename,
        status="indexing_failed" if n_vectors == 0 and index_error else ("superseded" if superseded else "ingested"),
        doc_id=doc_record.doc_id,
        department=doc_record.department,
        version=doc_record.version,
        chunks_created=n_chunks,
        vectors_added=n_vectors,
        error=index_error,
        processing_ms=round(elapsed_ms, 1),
    )


# ---------------------------------------------------------------------------
# Dashboard helpers
# ---------------------------------------------------------------------------

def get_document_stats() -> dict:
    """Returns counts from the SQLite ledger for the admin dashboard."""
    try:
        import ledger
        ledger.initialize_db()
        docs = ledger.get_all_documents()
        total      = len(docs)
        embedded   = sum(1 for d in docs if d.get("status") == "embedded")
        superseded = sum(1 for d in docs if d.get("status") == "superseded")
        chunked    = sum(1 for d in docs if d.get("status") in ("chunked", "embedded"))
        chunk_count = ledger.get_chunk_count()
        vector_count = ledger.get_embedded_chunk_count()

        return {
            "total_documents":    total,
            "embedded_documents": embedded,
            "superseded_documents": superseded,
            "chunked_documents":  chunked,
            "total_chunks":       chunk_count,
            "embedded_vectors":   vector_count,
        }
    except Exception as exc:
        logger.error("[DOC_MANAGER] get_document_stats failed: %s", exc)
        return {
            "total_documents": 0, "embedded_documents": 0,
            "superseded_documents": 0, "chunked_documents": 0,
            "total_chunks": 0, "embedded_vectors": 0,
        }


def get_document_list() -> list[dict]:
    """Returns the full document list from the ledger."""
    try:
        import ledger
        return ledger.get_all_documents()
    except Exception as exc:
        logger.error("[DOC_MANAGER] get_document_list failed: %s", exc)
        return []


def get_ingestion_logs(limit: int = 50) -> list[dict]:
    """Returns recent audit log events."""
    try:
        import sqlite3
        import ledger
        conn = ledger.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM process_logs ORDER BY timestamp DESC LIMIT ?", (limit,)
            )
            rows = cursor.fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()
    except Exception as exc:
        logger.error("[DOC_MANAGER] get_ingestion_logs failed: %s", exc)
        return []


def delete_document(doc_id: str) -> dict:
    """Removes a document from the vector store, the ledger (document + chunks),
    and its staged file, in a single transaction. Returns
    {removed: bool, vectors_removed: int}.

    Previously the vector delete and the ledger delete were two independent
    calls with the first one's failure only logged, not propagated -- a
    vector-delete exception left the ledger delete to run anyway, producing
    an orphaned embeddings row that stayed fully searchable (a "deleted"
    document could still be retrieved and cited). Folding both into one
    connection means either everything disappears or nothing does, and a
    failure here is a real exception the caller sees, not a silent partial
    delete.
    """
    import ledger
    from vector_store.sqlite_store import get_vector_store

    doc = ledger.get_document_by_doc_id(doc_id)
    # source_file is what ingestion actually populates (relative to project
    # root, e.g. "data/staging/x.docx"); the older `filepath` column is left
    # NULL by upsert_document's INSERT path -- reading it here always missed.
    filepath = doc.get("source_file") if doc else None

    conn = ledger.get_connection()
    try:
        store = get_vector_store()
        vectors_removed = store.delete_by_doc_id(doc_id, conn=conn)
        removed = ledger.delete_document(doc_id, conn=conn)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    # Staged file removal is best-effort and deliberately outside the DB
    # transaction (a filesystem delete can't be rolled back with it anyway).
    # Left in place, it would resurrect on the next full run_ingestion() scan
    # of data/staging/ -- the actual "deleted doc comes back" complaint.
    if filepath:
        try:
            abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", filepath))
            if os.path.isfile(abs_path):
                os.remove(abs_path)
                logger.info("[DOC_MANAGER] Removed staged file for %s: %s", doc_id[:12], filepath)
        except Exception as exc:
            logger.warning("[DOC_MANAGER] Could not remove staged file '%s' for %s: %s", filepath, doc_id[:12], exc)

    logger.info("[DOC_MANAGER] Deleted doc %s (ledger=%s, vectors=%s)", doc_id[:12], removed, vectors_removed)
    return {"removed": removed, "vectors_removed": vectors_removed}


def get_vector_count() -> int:
    """Returns the current vector count in the local store."""
    try:
        from vector_store.sqlite_store import get_vector_store
        store = get_vector_store()
        stats = store.get_collection_stats()
        return stats.get("vector_count", stats.get("total_vectors", 0))
    except Exception:
        return 0
