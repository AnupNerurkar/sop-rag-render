"""
vector_store/index_pipeline.py
-------------------------------
Phase 5: End-to-End Indexing Orchestrator

Connects Phase 4 (Embeddings) and Phase 5 (the vector store) into one run:

    SQLite chunks (pending)
         ↓  embed_pipeline.run()
    EmbeddingPayload list (768-dim vectors)
         ↓  sqlite_store.upsert()
    vector collection
         ↓  EmbedPipeline.mark_embedded()
    SQLite (embedded_at stamped, document status → 'embedded')

Two-phase commit guarantee:
    embedded_at is ONLY stamped in SQLite AFTER the vector store confirms the upsert.
    If the upsert fails, chunks remain pending and will be retried on the next run.

Incremental design:
    Only chunks with embedded_at IS NULL are processed.
    Re-running this script on an already-indexed corpus is a no-op.

Entry Points:
    run_indexing()              — index all pending chunks
    run_indexing(doc_id=...)    — index one document (admin upload workflow)
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Run summary
# ---------------------------------------------------------------------------

@dataclass
class IndexingRunSummary:
    """Combined Phase 4 + Phase 5 run statistics."""
    run_id:            str = ""
    started_at:        str = ""
    completed_at:      str = ""
    chunks_pending:    int = 0
    chunks_embedded:   int = 0
    chunks_upserted:   int = 0
    docs_updated:      int = 0
    errors:            list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return len(self.errors) == 0 and self.chunks_upserted == self.chunks_embedded

    def report(self) -> str:
        status = "SUCCESS" if self.success else "PARTIAL / FAILED"
        lines = [
            "=" * 60,
            f"INDEXING RUN [{status}]: {self.run_id}",
            f"  Started  : {self.started_at}",
            f"  Completed: {self.completed_at}",
            "-" * 60,
            f"  Chunks pending   : {self.chunks_pending}",
            f"  Chunks embedded  : {self.chunks_embedded}",
            f"  Vectors upserted : {self.chunks_upserted}",
            f"  Documents updated: {self.docs_updated}",
        ]
        if self.errors:
            lines.append("-" * 60)
            lines.append("  Errors:")
            for e in self.errors:
                lines.append(f"    * {e}")
        lines.append("=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_indexing(
    doc_id: Optional[str] = None,
    batch_size: int = 100,
) -> IndexingRunSummary:
    """
    Phase 4 + Phase 5 combined orchestrator.

    Args:
        doc_id:     Optional — restrict to one document (admin upload path).
        batch_size: Embedding batch size override.

    Returns:
        IndexingRunSummary with full statistics.
    """
    from embeddings.embed_pipeline import EmbedPipeline, run_embedding
    from vector_store.sqlite_store import get_vector_store

    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    summary = IndexingRunSummary(
        run_id     = run_id,
        started_at = datetime.utcnow().isoformat(),
    )

    logger.info(f"[INDEX PIPELINE] Run {run_id} started.")

    # ----------------------------------------------------------------
    # Phase 4: Generate embeddings for all pending chunks
    # ----------------------------------------------------------------
    logger.info("[INDEX PIPELINE] Phase 4: Generating embeddings...")
    try:
        pipeline = EmbedPipeline(batch_size=batch_size)
        payloads, embed_summary = pipeline.run(doc_id=doc_id)
    except Exception as e:
        msg = f"Phase 4 embedding failed: {e}"
        logger.error(f"[INDEX PIPELINE] {msg}", exc_info=True)
        summary.errors.append(msg)
        summary.completed_at = datetime.utcnow().isoformat()
        return summary

    summary.chunks_pending  = embed_summary.total_pending
    summary.chunks_embedded = embed_summary.total_embedded
    summary.errors.extend(embed_summary.errors)

    if not payloads:
        if embed_summary.total_pending == 0:
            logger.info("[INDEX PIPELINE] No pending chunks. Collection is up to date.")
        else:
            logger.warning("[INDEX PIPELINE] Embedding produced no payloads. Check errors.")
        summary.completed_at = datetime.utcnow().isoformat()
        return summary

    # ----------------------------------------------------------------
    # Phase 5: Upsert into the vector store
    # ----------------------------------------------------------------
    logger.info(f"[INDEX PIPELINE] Phase 5: Upserting {len(payloads)} vectors into the vector store...")
    try:
        store = get_vector_store()
        upserted = store.upsert(payloads)
        summary.chunks_upserted = upserted
    except Exception as e:
        msg = f"vector-store upsert failed: {e}"
        logger.error(f"[INDEX PIPELINE] {msg}", exc_info=True)
        summary.errors.append(msg)
        summary.completed_at = datetime.utcnow().isoformat()
        # Do NOT stamp embedded_at — two-phase commit guarantee
        return summary

    # ----------------------------------------------------------------
    # Ledger stamp: mark chunks embedded ONLY after confirmed upsert
    # ----------------------------------------------------------------
    try:
        docs_updated = EmbedPipeline.mark_embedded(payloads)
        summary.docs_updated = docs_updated
        logger.info(
            f"[INDEX PIPELINE] Ledger updated. "
            f"{len(payloads)} chunks stamped, {docs_updated} docs → 'embedded'."
        )
    except Exception as e:
        msg = f"Ledger stamp failed (vector-store upsert succeeded): {e}"
        logger.error(f"[INDEX PIPELINE] {msg}", exc_info=True)
        summary.errors.append(msg)

    summary.completed_at = datetime.utcnow().isoformat()
    logger.info(f"[INDEX PIPELINE] Run {run_id} complete.")
    return summary


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os as _os
    import sys as _sys
    # Ensure project root is on sys.path when running as a script from any cwd.
    _project_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
    if _project_root not in _sys.path:
        _sys.path.insert(0, _project_root)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info("Phase 4 + 5: Starting embedding and indexing pipeline...")
    summary = run_indexing()
    print(summary.report())
    sys.exit(0 if summary.success else 1)
