"""
scripts/migrate_neon_to_sqlite.py
----------------------------------
One-shot migration: Neon (Postgres + pgvector) -> local SQLite files for the Pi.

Runs on the PC, where psycopg2 can reach Neon. Writes two files to dist/pi/:

    institutional.db      users, signup_domains, chat_messages, committee_uploads
    ingestion_ledger.db   documents, process_logs, chunks, embeddings

Both are built fresh by calling the application's own DDL -- never hand-written
schema, which is how the two drift apart.

Usage (PowerShell):
    $env:NEON_DATABASE_URL = "postgresql://..."
    python scripts/migrate_neon_to_sqlite.py

The URL is read from the environment and never written to disk. The output
contains real user data (password hashes, chat history) -- delete dist/ once
the files are on the Pi and verified.

Exits non-zero if any post-migration check fails.
"""

from __future__ import annotations

import os
import sys

# ---------------------------------------------------------------------------
# Environment must be set BEFORE any project import: backend/database.py builds
# its engine at import time, and ledger/chroma_store read env at import time too.
# ---------------------------------------------------------------------------

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

DIST = os.path.join(_ROOT, "dist", "pi")
os.makedirs(DIST, exist_ok=True)

APP_DB    = os.path.join(DIST, "institutional.db")
LEDGER_DB = os.path.join(DIST, "ingestion_ledger.db")

NEON_URL = os.environ.get("NEON_DATABASE_URL", "").strip()
if not NEON_URL:
    sys.exit("NEON_DATABASE_URL is not set. Export it and re-run (it is never written to disk).")

os.environ["DATABASE_URL"] = "sqlite:///" + APP_DB.replace("\\", "/")
os.environ["VECTOR_STORE_BACKEND"] = "sqlite"
os.environ.pop("LEDGER_DATABASE_URL", None)

import json
import sqlite3

import numpy as np
import psycopg2
import psycopg2.extras

import ledger

# get_connection() reads this module global at call time, so plain assignment
# is enough to redirect the ledger at the dist copy.
ledger.DB_PATH = LEDGER_DB

from backend.database import init_db
from vector_store.sqlite_store import SQLiteVectorStore, EMBEDDING_DIM, _pack

# Absolute Windows paths in committee_uploads.stored_path won't resolve on the
# Pi -- rebase them onto the Pi's data directory.
PI_PENDING_UPLOADS = "/home/mypi/edumind/data/pending_uploads"

LEDGER_TABLES = ["documents", "process_logs", "chunks"]
APP_TABLES    = ["users", "signup_domains", "chat_messages", "committee_uploads"]

errors: list = []
report: list = []


def note(line: str) -> None:
    print(line)
    report.append(line)


def sqlite_columns(conn, table: str) -> list:
    return [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]


def copy_table(pg_cur, dest: sqlite3.Connection, table: str, transform=None) -> int:
    """
    Copies one table by intersecting source and destination column names, so a
    column that exists on only one side is skipped rather than crashing. Any
    dropped column is reported.
    """
    target_cols = sqlite_columns(dest, table)
    if not target_cols:
        errors.append(f"{table}: destination table does not exist")
        return 0

    pg_cur.execute(f"SELECT * FROM {table}")
    rows = pg_cur.fetchall()
    if not rows:
        note(f"  {table:<20} 0 rows (source empty)")
        return 0

    source_cols = list(rows[0].keys())
    cols = [c for c in source_cols if c in target_cols]
    skipped = [c for c in source_cols if c not in target_cols]
    if skipped:
        note(f"  {table:<20} NOTE: source columns not in target, skipped: {skipped}")

    placeholders = ",".join(["?"] * len(cols))
    dest.execute(f"DELETE FROM {table}")
    payload = []
    for r in rows:
        d = {c: r[c] for c in cols}
        if transform:
            transform(d)
        payload.append(tuple(d[c] for c in cols))

    dest.executemany(
        f"INSERT INTO {table} ({','.join(cols)}) VALUES ({placeholders})", payload
    )
    dest.commit()
    note(f"  {table:<20} {len(payload)} rows")
    return len(payload)


def rewrite_stored_path(row: dict) -> None:
    old = row.get("stored_path")
    if not old:
        return
    new = PI_PENDING_UPLOADS + "/" + os.path.basename(str(old).replace("\\", "/"))
    if new != old:
        note(f"    stored_path: {old}  ->  {new}")
    row["stored_path"] = new


def migrate_embeddings(pg_cur, dest: sqlite3.Connection) -> int:
    """
    Copies pgvector rows into float32 blobs.

    The ::text cast renders the vector as '[0.1,0.2,...]', which avoids needing
    a pgvector Python adapter on this machine. _pack L2-normalizes, which is
    what makes query-time cosine a plain dot product in SQLiteVectorStore.
    """
    pg_cur.execute("""
        SELECT id, doc_id, content, access_level, department, category, title, version,
               embedding::text AS emb
        FROM embeddings
    """)
    rows = pg_cur.fetchall()

    dest.execute("DELETE FROM embeddings")
    payload = []
    bad_dims = 0
    for r in rows:
        vec = json.loads(r["emb"])
        if len(vec) != EMBEDDING_DIM:
            bad_dims += 1
            errors.append(f"embeddings: id={r['id']} has dim {len(vec)}, expected {EMBEDDING_DIM}")
            continue
        blob = _pack(vec)
        payload.append((
            r["id"], r["doc_id"], r["content"], blob, EMBEDDING_DIM,
            r["access_level"], r["department"], r["category"], r["title"], r["version"],
        ))

    dest.executemany("""
        INSERT INTO embeddings (id, doc_id, content, embedding, dim,
                                access_level, department, category, title, version)
        VALUES (?,?,?,?,?,?,?,?,?,?)
    """, payload)
    dest.commit()
    note(f"  {'embeddings':<20} {len(payload)} rows" + (f"  ({bad_dims} rejected)" if bad_dims else ""))
    return len(payload)


def verify(app: sqlite3.Connection, led: sqlite3.Connection) -> None:
    """
    Post-migration gates. A silent partial migration is worse than a loud
    failure, so each of these is a hard error rather than a warning.
    """
    note("")
    note("Verification")

    users = app.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    note(f"  users                {users}")
    if users == 0:
        # backend/app.py seeds admin_test/Admin@123 into an empty users table,
        # and this box is LAN-exposed.
        errors.append("users table is EMPTY -- the app would auto-seed default accounts on startup")

    embs = led.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
    note(f"  embeddings           {embs}")
    if embs == 0:
        errors.append("embeddings table is EMPTY -- retrieval would return nothing")

    orphans = led.execute("""
        SELECT COUNT(*) FROM embeddings e
        LEFT JOIN chunks c ON c.chunk_id = e.id
        WHERE c.chunk_id IS NULL
    """).fetchone()[0]
    note(f"  orphan embeddings    {orphans}  (no matching chunks row)")
    if orphans:
        errors.append(f"{orphans} embeddings have no matching chunk -- their citations would lose chunk_index")

    blobs = led.execute(
        "SELECT COUNT(*) FROM embeddings WHERE LENGTH(embedding) != ?", (EMBEDDING_DIM * 4,)
    ).fetchone()[0]
    note(f"  wrong-size blobs     {blobs}")
    if blobs:
        errors.append(f"{blobs} embedding blobs are not {EMBEDDING_DIM * 4} bytes")

    # A normalized vector dotted with itself is 1.0; anything else means the
    # write-time normalization didn't take, which would break score parity with
    # pgvector and silently trip the 0.70 confidence threshold.
    row = led.execute("SELECT embedding FROM embeddings LIMIT 1").fetchone()
    if row:
        v = np.frombuffer(row[0], dtype=np.float32)
        norm = float(np.linalg.norm(v))
        note(f"  sample vector norm   {norm:.6f}  (must be 1.0)")
        if abs(norm - 1.0) > 1e-3:
            errors.append(f"embeddings are not L2-normalized (sample norm {norm})")


def main() -> int:
    note(f"Target: {DIST}")
    note("")

    for path in (APP_DB, LEDGER_DB):
        if os.path.exists(path):
            os.remove(path)

    note("Creating schema from the application's own DDL...")
    ledger.initialize_db()
    init_db()
    SQLiteVectorStore().initialize()

    pg = psycopg2.connect(NEON_URL)
    pg_cur = pg.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    app = sqlite3.connect(APP_DB)
    led = sqlite3.connect(LEDGER_DB)
    try:
        note("")
        note("Ledger tables -> ingestion_ledger.db")
        for t in LEDGER_TABLES:
            copy_table(pg_cur, led, t)
        migrate_embeddings(pg_cur, led)

        note("")
        note("Application tables -> institutional.db")
        for t in APP_TABLES:
            copy_table(pg_cur, app, t,
                       transform=rewrite_stored_path if t == "committee_uploads" else None)

        verify(app, led)
    finally:
        pg_cur.close()
        pg.close()
        app.close()
        led.close()

    note("")
    if errors:
        note(f"FAILED -- {len(errors)} problem(s):")
        for e in errors:
            note(f"  - {e}")
        return 1

    note("OK -- both files written. Next: V2 (verify_rag.py against dist/pi), then scp to the Pi.")
    note("Remember to delete dist/ once the transfer is verified -- it contains real user data.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
