"""
embeddings/embed_pipeline.py
-----------------------------
Phase 4: Embedding Orchestration Pipeline

Reads chunks from SQLite → embeds with BGEEmbedder → returns vector-store-ready
payloads → stamps embedded_at in SQLite → updates document status to 'embedded'.

Embedding Payload Format (output of this module, consumed by Phase 5):
    {
        "chunk_id":  str,           # Deterministic 16-char hex ID
        "content":   str,           # Raw chunk text (the vector store 'document')
        "embedding": list[float],   # 768-dim BGE vector
        "metadata":  dict,          # Flat flat metadata dict
    }

Incremental Design:
    - Only chunks with embedded_at IS NULL are processed.
    - embedded_at is stamped AFTER successful vector-store upsert (Phase 5 calls
      ledger.mark_chunks_embedded()). The pipeline itself returns payloads
      without writing to the vector store — that is Phase 5's responsibility.
    - This keeps Phase 4 and Phase 5 independently testable.

Entry Points:
    run_embedding()               → embeds all pending chunks (full initial run)
    EmbedPipeline.run(doc_id=...) → embeds pending chunks for one document only
                                    (used by incremental admin upload workflow)
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import ledger
from embeddings.embedder import get_embedder, BATCH_SIZE

logger = logging.getLogger(__name__)

# How much of the previous chunk's tail to prepend as context when building
# the text actually sent to the embedder. Most chunks have zero overlap
# with their neighbours: chunker.py keeps SOP sub-process blocks atomic and
# unsplit up to 1500 chars, so CHUNK_OVERLAP=150 only ever applies inside
# the (rarer) oversized blocks that get a secondary split. This recovers
# cross-boundary context for the *vector* -- a step chunk that reads "Then
# submit the form to the HOD" gets embedded with the end of the previous
# chunk (which named the form) prepended, without duplicating any text in
# what's actually stored, cited, or shown to the LLM (payload.content stays
# the original chunk text; only the embedding input is augmented).
OVERLAP_TAIL_CHARS = 150


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class EmbeddingPayload:
    """
    Single chunk payload ready for vector-store upsert.
    Produced by EmbedPipeline; consumed by Phase 5 (vector_store/sqlite_store.py).
    """
    chunk_id:  str
    content:   str
    embedding: list[float]
    metadata:  dict


def _snap_to_word_boundary(tail: str) -> str:
    """
    Drops a leading partial word from a string sliced mid-word (a fixed
    character-count tail from the end of the previous chunk will usually
    start partway through a word). Only snaps if the boundary is close by
    -- a tail with no early space is probably one long token (a URL, a code
    like "8A-3"), which is left intact rather than truncated further.
    """
    sp = tail.find(" ")
    if 0 <= sp < 30:
        return tail[sp + 1:]
    return tail


def _build_embedding_text(content: str, section_heading: str, prev_content: Optional[str]) -> str:
    """
    Assembles the text actually sent to the embedder: heading + tail of the
    previous chunk (word-boundary snapped) + this chunk's own content. The
    first chunk of a document (prev_content is None) gets no overlap --
    there is nothing before it to pull context from.
    """
    parts: list[str] = []
    heading = (section_heading or "").strip()
    if heading:
        parts.append(heading)
    if prev_content:
        tail = prev_content.strip()
        if len(tail) > OVERLAP_TAIL_CHARS:
            tail = _snap_to_word_boundary(tail[-OVERLAP_TAIL_CHARS:])
        if tail:
            parts.append(tail)
    parts.append(content)
    return "\n".join(parts)


@dataclass
class EmbeddingRunSummary:
    """Statistics for a single embedding pipeline run."""
    run_id:            str = ""
    started_at:        str = ""
    completed_at:      str = ""
    total_pending:     int = 0
    total_embedded:    int = 0
    total_skipped:     int = 0   # Chunks already embedded in a previous run
    total_failed:      int = 0
    docs_updated:      int = 0
    errors:            list[str] = field(default_factory=list)

    def report(self) -> str:
        lines = [
            "=" * 60,
            f"EMBEDDING RUN: {self.run_id}",
            f"  Started  : {self.started_at}",
            f"  Completed: {self.completed_at}",
            "-" * 60,
            f"  Pending chunks     : {self.total_pending}",
            f"  Newly embedded     : {self.total_embedded}",
            f"  Already embedded   : {self.total_skipped}",
            f"  Failed             : {self.total_failed}",
            f"  Documents updated  : {self.docs_updated}",
        ]
        if self.errors:
            lines.append("-" * 60)
            lines.append("  Errors:")
            for e in self.errors:
                lines.append(f"    * {e}")
        lines.append("=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class EmbedPipeline:
    """
    Orchestrates Phase 4 embedding across all (or one) document's pending chunks.

    Usage:
        pipeline = EmbedPipeline()
        payloads, summary = pipeline.run()
        # payloads → pass to Phase 5 (vector-store upsert)
        # summary  → log / return from API
    """

    def __init__(self, batch_size: int = BATCH_SIZE) -> None:
        self._batch_size = batch_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        doc_id: Optional[str] = None,
    ) -> tuple[list[EmbeddingPayload], EmbeddingRunSummary]:
        """
        Embeds all chunks that have not yet been embedded.

        Args:
            doc_id: If provided, only embeds chunks for this specific document.
                    Used by the incremental admin upload workflow.

        Returns:
            (payloads, summary)
            payloads: List of EmbeddingPayload objects ready for vector-store upsert.
            summary:  EmbeddingRunSummary with statistics.

        NOTE: This method does NOT write to the vector store or stamp embedded_at.
              Phase 5 (sqlite_store.py) calls ledger.mark_chunks_embedded()
              after a successful upsert, keeping the two phases decoupled.
        """
        run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        summary = EmbeddingRunSummary(
            run_id     = run_id,
            started_at = datetime.utcnow().isoformat(),
        )

        # --- Ensure schema is up to date ---
        ledger.initialize_db()

        # --- Load pending chunks from SQLite ---
        pending_rows = ledger.get_chunks_pending_embedding(doc_id=doc_id)
        summary.total_pending = len(pending_rows)

        if not pending_rows:
            logger.info("[EMBED PIPELINE] No pending chunks found. Nothing to embed.")
            summary.completed_at = datetime.utcnow().isoformat()
            return [], summary

        logger.info(
            f"[EMBED PIPELINE] Run {run_id}: "
            f"{summary.total_pending} chunks to embed."
        )

        # --- Load the embedder (lazy singleton) ---
        try:
            embedder = get_embedder()
        except Exception as e:
            msg = f"Model load failed: {e}"
            logger.error(f"[EMBED PIPELINE] {msg}")
            summary.errors.append(msg)
            summary.completed_at = datetime.utcnow().isoformat()
            return [], summary

        # --- Build embedding texts (heading + prior-chunk tail + content) ---
        # `pending_rows` is ordered by (doc_id, chunk_index), so the
        # previous chunk is usually the prior row in this same batch; when
        # it isn't (e.g. only chunk_index=3 of a document is pending
        # because 0-2 were already embedded in an earlier run), it's
        # fetched directly. chunk_index=0 gets no overlap -- there is no
        # previous chunk in the document.
        batch_content_by_key = {
            (row["doc_id"], row["chunk_index"]): row["content"] for row in pending_rows
        }
        texts: list[str] = []
        for row in pending_rows:
            prev_content = None
            if row["chunk_index"] > 0:
                prev_key = (row["doc_id"], row["chunk_index"] - 1)
                prev_content = batch_content_by_key.get(prev_key)
                if prev_content is None:
                    prev_content = ledger.get_chunk_content_by_index(row["doc_id"], row["chunk_index"] - 1)
            texts.append(_build_embedding_text(row["content"], row.get("section_heading", ""), prev_content))

        # --- Embed in batches ---
        try:
            vectors = embedder.embed_documents(
                texts,
                batch_size=self._batch_size,
                show_progress=True,
            )
        except Exception as e:
            msg = f"Embedding failed: {e}"
            logger.error(f"[EMBED PIPELINE] {msg}", exc_info=True)
            summary.errors.append(msg)
            summary.completed_at = datetime.utcnow().isoformat()
            return [], summary

        # --- Build EmbeddingPayload list ---
        payloads: list[EmbeddingPayload] = []
        failed_chunk_ids: list[str] = []

        for row, vector in zip(pending_rows, vectors):
            try:
                payload = EmbeddingPayload(
                    chunk_id  = row["chunk_id"],
                    content   = row["content"],
                    embedding = vector,
                    metadata  = {
                        "doc_id":          row["doc_id"],
                        "source_file":     row["source_file"],
                        "title":           row.get("title", ""),
                        "category":        row["category"],
                        "department":      row["department"],
                        "version":         row["version"],
                        "access_level":    row["access_level"],
                        "upload_date":     row.get("upload_date", ""),
                        "chunk_index":     row["chunk_index"],
                        "total_chunks":    row["total_chunks"],
                        "section_heading": row.get("section_heading", ""),
                    },
                )
                payloads.append(payload)
            except Exception as e:
                failed_chunk_ids.append(row["chunk_id"])
                summary.errors.append(f"Payload build failed for {row['chunk_id']}: {e}")
                logger.error(f"[EMBED PIPELINE] Payload build error: {e}")

        summary.total_embedded = len(payloads)
        summary.total_failed   = len(failed_chunk_ids)

        summary.completed_at = datetime.utcnow().isoformat()

        logger.info(
            f"[EMBED PIPELINE] Complete. "
            f"Embedded={summary.total_embedded}, "
            f"Failed={summary.total_failed}"
        )

        return payloads, summary

    # ------------------------------------------------------------------
    # Post-upsert ledger update (called by Phase 5 after vector-store upsert)
    # ------------------------------------------------------------------

    @staticmethod
    def mark_embedded(payloads: list[EmbeddingPayload]) -> int:
        """
        Stamps embedded_at on all successfully upserted chunks and updates
        document statuses to 'embedded'.

        Called by Phase 5 (sqlite_store.py) AFTER a successful vector-store upsert.
        Returns the number of documents whose status was updated.

        This two-phase commit pattern (embed → upsert → stamp) ensures that
        embedded_at in SQLite is only set when the vector is confirmed in the vector store.
        """
        if not payloads:
            return 0

        chunk_ids = [p.chunk_id for p in payloads]
        ledger.mark_chunks_embedded(chunk_ids)

        # Determine which doc_ids are now fully embedded
        doc_ids = {p.metadata["doc_id"] for p in payloads}
        docs_updated = 0
        for doc_id in doc_ids:
            # Only mark the document as embedded if ALL its chunks are now embedded.
            pending = ledger.get_chunks_pending_embedding(doc_id=doc_id)
            if not pending:
                ledger.update_document_post_embedding(doc_id)
                docs_updated += 1
                logger.info(f"[EMBED PIPELINE] Document {doc_id[:12]}... → status=embedded")

        return docs_updated


# ---------------------------------------------------------------------------
# Module-level entry point
# ---------------------------------------------------------------------------

def run_embedding(
    doc_id: Optional[str] = None,
    batch_size: int = BATCH_SIZE,
) -> tuple[list[EmbeddingPayload], EmbeddingRunSummary]:
    """
    Convenience wrapper for running Phase 4 embedding.

    Returns (payloads, summary). Phase 5 receives payloads for vector-store upsert.

    Args:
        doc_id:     Optional — restrict to one document (incremental upload path).
        batch_size: Override default batch size.
    """
    pipeline = EmbedPipeline(batch_size=batch_size)
    return pipeline.run(doc_id=doc_id)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os as _os
    import sys as _sys
    _project_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
    if _project_root not in _sys.path:
        _sys.path.insert(0, _project_root)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info("Phase 4: Starting embedding pipeline...")
    payloads, summary = run_embedding()
    print(summary.report())
    print(f"\nPayloads ready for Phase 5 vector-store upsert: {len(payloads)}")

    sys.exit(0 if summary.total_failed == 0 else 1)
