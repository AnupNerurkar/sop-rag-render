from __future__ import annotations

import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "ingestion_ledger.db"))


def get_connection():
    """Opens a connection to the local ledger database, creating the file if needed."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    # Every ledger call opens and closes its own connection, and FastAPI
    # runs `def` endpoints in a threadpool, so concurrent writers are real.
    # SQLite's default busy timeout is zero — it raises "database is locked"
    # immediately instead of waiting. WAL additionally keeps readers from
    # blocking the writer. busy_timeout is per-connection, so it has to be
    # set here on every connect, not once at setup.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def _ensure_column(cursor, table: str, column: str, definition: str) -> None:
    """Adds a column to an existing table if it is not already present."""
    cursor.execute(f"PRAGMA table_info({table})")
    existing = {row[1] for row in cursor.fetchall()}

    if column.lower() not in existing:
        cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


_PHASE2_DOCUMENT_COLUMNS = {
    "doc_id": "TEXT",
    "source_file": "TEXT",
    "original_file": "TEXT",
    "upload_date": "TEXT",
    "total_chunks": "INTEGER DEFAULT 0",
    "ingested_at": "TEXT",
    "uploaded_by": "TEXT",
}


def _migrate_documents_table(cursor) -> None:
    """Extends the Phase 1 documents table with Phase 2/3 columns."""
    for column, definition in _PHASE2_DOCUMENT_COLUMNS.items():
        if column == "doc_id":
            # doc_id is referenced by chunks, so it needs a UNIQUE constraint
            _ensure_column(cursor, "documents", column, "TEXT UNIQUE")
        else:
            _ensure_column(cursor, "documents", column, definition)

    cursor.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_doc_id ON documents(doc_id)"
    )
    cursor.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_source_file ON documents(source_file)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_documents_department ON documents(department)"
    )


def initialize_db():
    """Initializes the database tables if they do not exist."""
    conn = get_connection()
    try:
        cursor = conn.cursor()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id TEXT UNIQUE,
            filepath TEXT UNIQUE,
            sha256_hash TEXT,
            status TEXT,
            title TEXT,
            category TEXT,
            department TEXT,
            version TEXT,
            date TEXT,
            access_level TEXT,
            created_at TEXT,
            last_processed TEXT
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS process_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filepath TEXT,
            action TEXT,
            status TEXT,
            message TEXT,
            timestamp TEXT
        )
        """)

        # Migrate documents table BEFORE creating chunks, so doc_id column exists
        _migrate_documents_table(cursor)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id TEXT PRIMARY KEY,
            doc_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            section_heading TEXT,
            category TEXT,
            department TEXT,
            access_level TEXT,
            version TEXT,
            source_file TEXT,
            total_chunks INTEGER,
            created_at TEXT,
            FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
        )
        """)

        _migrate_chunks_embedding_columns(cursor)

        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id)"
        )

        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _row_to_dict(row) -> dict:
    return dict(row) if row else None


# ---------------------------------------------------------------------------
# Phase 1 API (repository assessment)
# ---------------------------------------------------------------------------

def register_document(filepath, sha256_hash, status="assessed", metadata=None):
    """Registers or updates a document entry in the ledger (Phase 1 assessment)."""
    if metadata is None:
        metadata = {}

    title = metadata.get("title", "")
    category = metadata.get("category", "Unknown")
    department = metadata.get("department", "Unknown")
    version = metadata.get("version", "1.0")
    date = metadata.get("date", "")
    access_level = metadata.get("access_level", "Public")

    now = datetime.utcnow().isoformat()

    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
        INSERT INTO documents (
            filepath, sha256_hash, status, title, category, department,
            version, date, access_level, created_at, last_processed
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(filepath) DO UPDATE SET
            sha256_hash = excluded.sha256_hash,
            status = excluded.status,
            title = excluded.title,
            category = excluded.category,
            department = excluded.department,
            version = excluded.version,
            date = excluded.date,
            access_level = excluded.access_level,
            last_processed = excluded.last_processed
        """, (
            filepath, sha256_hash, status, title, category, department,
            version, date, access_level, now, now,
        ))
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()


def get_document(filepath):
    """Retrieves document record by original filepath (Phase 1 key)."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM documents WHERE filepath = ?", (filepath,))
        row = cursor.fetchone()
        return _row_to_dict(row)
    finally:
        conn.close()


def get_all_documents():
    """Retrieves all registered documents from ledger."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM documents ORDER BY id")
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Phase 2/3 API (ingestion + chunking)
# ---------------------------------------------------------------------------

def upsert_document(record: dict) -> None:
    """
    Inserts or updates a document from Phase 2 ingestion.

    record keys match DocumentRecord.to_ledger_dict() plus sha256_hash
    derived from doc_id when not explicitly provided.
    """
    doc_id = record["doc_id"]
    sha256_hash = record.get("sha256_hash", doc_id)
    now = datetime.utcnow().isoformat()

    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT id FROM documents WHERE source_file = ?", (record["source_file"],))
        existing = cursor.fetchone()

        if existing:
            cursor.execute("""
            UPDATE documents SET
                doc_id = ?,
                sha256_hash = ?,
                filepath = COALESCE(?, filepath),
                original_file = ?,
                status = ?,
                title = ?,
                category = ?,
                department = ?,
                version = ?,
                upload_date = ?,
                date = COALESCE(?, date),
                access_level = ?,
                total_chunks = ?,
                ingested_at = COALESCE(ingested_at, ?),
                last_processed = ?
            WHERE source_file = ?
            """, (
                doc_id,
                sha256_hash,
                record.get("original_file") or record["source_file"],
                record.get("original_file", ""),
                record["status"],
                record["title"],
                record["category"],
                record["department"],
                record["version"],
                record.get("upload_date", ""),
                record.get("upload_date", ""),
                record["access_level"],
                record.get("total_chunks", 0),
                record.get("ingested_at", now),
                now,
                record["source_file"],
            ))
        else:
            cursor.execute("""
            INSERT INTO documents (
                doc_id, source_file, original_file, filepath, sha256_hash,
                status, title, category, department, version,
                upload_date, date, access_level, total_chunks,
                ingested_at, created_at, last_processed
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                doc_id,
                record["source_file"],
                record.get("original_file", ""),
                record.get("original_file") or record["source_file"],
                sha256_hash,
                record["status"],
                record["title"],
                record["category"],
                record["department"],
                record["version"],
                record.get("upload_date", ""),
                record.get("upload_date", ""),
                record["access_level"],
                record.get("total_chunks", 0),
                record.get("ingested_at", now),
                now,
                now,
            ))
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()


def set_document_uploader(doc_id: str, username: str) -> None:
    """Records which user contributed a document (admin uploader or approved
    committee head). Used by the admin document registry."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "UPDATE documents SET uploaded_by = ? WHERE doc_id = ?", (username, doc_id)
        )
        conn.commit()
    finally:
        conn.close()


def delete_document(doc_id: str) -> bool:
    """Removes a document and all its chunks from the ledger. Returns True if a
    document row was deleted. vectors must be removed separately by the
    caller (vector_store.delete_by_doc_id)."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        cursor.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
        deleted = cursor.rowcount > 0
        conn.commit()
        return deleted
    finally:
        conn.close()


def get_document_by_source(source_file: str) -> dict | None:
    """Retrieves a document by its staged source_file path."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM documents WHERE source_file = ?", (source_file,))
        row = cursor.fetchone()
        return _row_to_dict(row)
    finally:
        conn.close()


def get_document_by_doc_id(doc_id: str) -> dict | None:
    """Retrieves a document by its doc_id (SHA-256 of staged file)."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,))
        row = cursor.fetchone()
        return _row_to_dict(row)
    finally:
        conn.close()


def get_document_by_department(department: str) -> dict | None:
    """
    Returns the most recently processed non-superseded document for a department.
    Used for version supersession detection during incremental ingestion.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM documents
        WHERE department = ?
          AND status != 'superseded'
          AND doc_id IS NOT NULL
        ORDER BY last_processed DESC, id DESC
        LIMIT 1
    """, (department,))
    row = cursor.fetchone()
    conn.close()
    return _row_to_dict(row)


def mark_document_superseded(doc_id: str) -> None:
    """Marks an older document version as superseded."""
    now = datetime.utcnow().isoformat()
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            UPDATE documents
            SET status = 'superseded', last_processed = ?
            WHERE doc_id = ?
        """, (now, doc_id))
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()


def update_document_post_chunking(doc_id: str, total_chunks: int) -> None:
    """Updates document status and chunk count after Phase 3 chunking."""
    now = datetime.utcnow().isoformat()
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            UPDATE documents
            SET status = 'chunked',
                total_chunks = ?,
                last_processed = ?
            WHERE doc_id = ?
        """, (total_chunks, now, doc_id))
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()


def save_chunks(records: list[dict]) -> None:
    """
    Persists chunk records to SQLite.
    Replaces all existing chunks for the affected doc_id(s) to support re-ingestion.
    """
    if not records:
        return

    doc_ids = {r["doc_id"] for r in records}
    conn = get_connection()
    cursor = conn.cursor()
    try:
        for doc_id in doc_ids:
            cursor.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))

        cursor.executemany("""
            INSERT INTO chunks (
                chunk_id, doc_id, chunk_index, content, section_heading,
                category, department, access_level, version, source_file,
                total_chunks, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            (
                r["chunk_id"],
                r["doc_id"],
                r["chunk_index"],
                r["content"],
                r.get("section_heading", ""),
                r.get("category", ""),
                r.get("department", ""),
                r.get("access_level", "Public"),
                r.get("version", ""),
                r.get("source_file", ""),
                r.get("total_chunks", 0),
                r.get("created_at", datetime.utcnow().isoformat()),
            )
            for r in records
        ])
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()


def get_chunks_by_doc_id(doc_id: str) -> list[dict]:
    """Retrieves all chunks for a document, ordered by chunk_index."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM chunks WHERE doc_id = ? ORDER BY chunk_index",
            (doc_id,),
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def get_chunk_count() -> int:
    """Returns total number of chunks stored in the ledger."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM chunks")
        count = cursor.fetchone()[0]
        return count
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Phase 4 API (embeddings)
# ---------------------------------------------------------------------------

def _migrate_chunks_embedding_columns(cursor) -> None:
    """Adds Phase 4 columns to the chunks table if not already present."""
    _ensure_column(cursor, "chunks", "embedded_at", "TEXT")
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_chunks_embedded_at ON chunks(embedded_at)"
    )


def get_chunks_pending_embedding(doc_id: str | None = None) -> list[dict]:
    """
    Returns chunks that have not yet been embedded (embedded_at IS NULL).

    Args:
        doc_id: Optional filter to fetch only chunks for a specific document.

    Returns:
        List of chunk row dicts ordered by doc_id, chunk_index.
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()
        if doc_id:
            cursor.execute(
                """
                SELECT * FROM chunks
                WHERE embedded_at IS NULL AND doc_id = ?
                ORDER BY doc_id, chunk_index
                """,
                (doc_id,),
            )
        else:
            cursor.execute(
                """
                SELECT * FROM chunks
                WHERE embedded_at IS NULL
                ORDER BY doc_id, chunk_index
                """
            )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def mark_chunks_embedded(chunk_ids: list[str]) -> None:
    """
    Stamps embedded_at on a batch of chunks after successful vector-store upsert.
    Uses a single transaction for atomicity — if the vector store fails, do not call this.
    """
    if not chunk_ids:
        return
    now = datetime.utcnow().isoformat()
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.executemany(
            "UPDATE chunks SET embedded_at = ? WHERE chunk_id = ?",
            [(now, cid) for cid in chunk_ids],
        )
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()


def update_document_post_embedding(doc_id: str) -> None:
    """Updates document status to 'embedded' after all its chunks are vectorized."""
    now = datetime.utcnow().isoformat()
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            UPDATE documents
            SET status = 'embedded', last_processed = ?
            WHERE doc_id = ?
            """,
            (now, doc_id),
        )
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()


def get_embedded_chunk_count() -> int:
    """Returns the number of chunks that have been successfully embedded."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM chunks WHERE embedded_at IS NOT NULL")
    count = cursor.fetchone()[0]
    conn.close()
    return count


# ---------------------------------------------------------------------------
# Audit log (shared across phases)
# ---------------------------------------------------------------------------

def log_event(filepath, action, status, message):
    """Logs an action to the audit history."""
    now = datetime.utcnow().isoformat()
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
        INSERT INTO process_logs (filepath, action, status, message, timestamp)
        VALUES (?, ?, ?, ?, ?)
        """, (filepath, action, status, message, now))
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()
