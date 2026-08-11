"""
chunker.py
----------
Phase 3: Chunking

Splits serialized document text into metadata-rich, semantically bounded chunks
using LangChain's RecursiveCharacterTextSplitter.

Key Design Decisions (approved):
  - SOP sub-process blocks are treated as atomic units up to 1,500 chars.
  - Chunks may exceed 800 chars to avoid splitting a procedure step mid-way.
  - Hard split only when a block exceeds 1,500 chars.
  - Section headings are tracked and propagated to every chunk.
  - chunk_id is deterministic: SHA-256(doc_id::chunk_index)[:16].
  - total_chunks is back-filled on all chunks after the full document is split.
  - All chunks are written to the SQLite `chunks` table via ledger.py.
  - to_embedding_payload() produces embedding-payload output for Phase 4.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

import ledger
from chunk_schema import (
    AccessLevel,
    ChunkMetadata,
    ChunkRecord,
    DocumentRecord,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

# Target chunk size: allows one complete SOP sub-process block to fit.
CHUNK_SIZE = 1200

# Overlap preserves cross-boundary context (e.g. a step referencing the prior one).
CHUNK_OVERLAP = 150

# Absolute maximum for an atomic block before forcing a split.
ATOMIC_BLOCK_MAX = 1500

# Separator priority: try largest semantic breaks first.
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


# ---------------------------------------------------------------------------
# Chunk ID generation
# ---------------------------------------------------------------------------

def _make_chunk_id(doc_id: str, chunk_index: int) -> str:
    """
    Generates a deterministic 16-character hex chunk ID.
    SHA-256(doc_id::chunk_index)[:16]
    Same document version + same index always produces the same ID.
    """
    raw = f"{doc_id}::{chunk_index}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Section heading tracker
# ---------------------------------------------------------------------------

def _is_sub_process_marker(line: str) -> bool:
    """Returns True for [SUB PROCESS] blocks and legacy [SUB-PROCESS: ...] markers."""
    stripped = line.strip()
    if stripped == "[SUB PROCESS]":
        return True
    return stripped.startswith("[SUB-PROCESS:")


def _extract_section_headings(text: str) -> list[tuple[int, str]]:
    """
    Returns a list of (char_offset, heading_text) pairs found in the text.
    Detects:
      1. [SUB PROCESS] blocks (uses following 'Name:' line when present).
      2. Legacy [SUB-PROCESS: ...] markers.
      3. [PROCESS: ...] top-level process headings.
      4. Lines starting with 'Process:' or 'Process -'.
    """
    headings: list[tuple[int, str]] = []
    offset = 0
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped == "[SUB PROCESS]":
            heading = "[SUB PROCESS]"
            if i + 1 < len(lines) and lines[i + 1].strip().lower().startswith("name:"):
                heading = lines[i + 1].strip()
            headings.append((offset, heading))
        elif stripped.startswith("[SUB-PROCESS:") or stripped.startswith("[PROCESS:"):
            headings.append((offset, stripped.strip("[]")))
        elif stripped.lower().startswith("process:") or stripped.lower().startswith("process -"):
            headings.append((offset, stripped))

        offset += len(line) + 1
        i += 1
    return headings


def _find_heading_for_offset(char_offset: int, headings: list[tuple[int, str]]) -> str:
    """
    Returns the most recent heading whose offset is <= the given char_offset.
    Falls back to empty string if no heading precedes the offset.
    """
    result = ""
    for h_offset, h_text in headings:
        if h_offset <= char_offset:
            result = h_text
        else:
            break
    return result


# ---------------------------------------------------------------------------
# Atomic block splitter
# ---------------------------------------------------------------------------

def _split_into_atomic_blocks(text: str) -> list[tuple[str, int]]:
    """
    Splits text into atomic blocks at SUB-PROCESS / PROCESS boundaries.
    Each block tries to keep one complete SOP sub-process section together.

    Returns:
        List of (block_text, start_char_offset) tuples.
        Blocks exceeding ATOMIC_BLOCK_MAX are themselves passed to the
        RecursiveCharacterTextSplitter for secondary splitting.
    """
    blocks: list[tuple[str, int]] = []
    current_lines: list[str] = []
    current_start: int = 0
    offset: int = 0

    lines = text.split("\n")
    for line in lines:
        stripped = line.strip()
        is_boundary = (
            _is_sub_process_marker(stripped)
            or stripped.startswith("[PROCESS:")
            or (stripped.lower().startswith("process:") and len(stripped) < 80)
        )

        if is_boundary and current_lines:
            block_text = "\n".join(current_lines).strip()
            if block_text:
                blocks.append((block_text, current_start))
            current_lines = [line]
            current_start = offset
        else:
            current_lines.append(line)

        offset += len(line) + 1  # +1 for the '\n'

    # Flush last block
    if current_lines:
        block_text = "\n".join(current_lines).strip()
        if block_text:
            blocks.append((block_text, current_start))

    return blocks


# ---------------------------------------------------------------------------
# Core chunker
# ---------------------------------------------------------------------------

def build_splitter() -> RecursiveCharacterTextSplitter:
    """Instantiates the configured RecursiveCharacterTextSplitter."""
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
        length_function=len,
        is_separator_regex=False,
    )


def chunk_document(
    doc_record: DocumentRecord,
    text: str,
    primary_heading: Optional[str] = None,
) -> list[ChunkRecord]:
    """
    Splits a document's serialized text into ChunkRecord objects.

    Strategy:
      1. Split text into atomic SOP blocks (one per sub-process).
      2. Blocks ≤ ATOMIC_BLOCK_MAX → kept as single chunks.
      3. Blocks > ATOMIC_BLOCK_MAX → secondary split via RecursiveCharacterTextSplitter.
      4. Each chunk gets section_heading based on nearest heading before its position.
      5. chunk_id is deterministic. total_chunks is back-filled after all chunks are known.

    Args:
        doc_record: The DocumentRecord for the source document.
        text: Cleaned, serialized full-document text from ingestion_pipeline.py.
        primary_heading: Optional top-level heading (e.g. 'Process: Admissions').

    Returns:
        List of ChunkRecord objects (total_chunks NOT yet set — caller must back-fill).
    """
    if not text or not text.strip():
        logger.warning(f"[CHUNKER] Empty text for doc: {doc_record.source_file}. Skipping.")
        return []

    splitter = build_splitter()
    headings = _extract_section_headings(text)
    atomic_blocks = _split_into_atomic_blocks(text)

    chunk_index = 0
    chunks: list[ChunkRecord] = []

    for block_text, block_start in atomic_blocks:
        if not block_text.strip():
            continue

        # Determine section heading for this block
        section_heading = _find_heading_for_offset(block_start, headings) or primary_heading or ""

        if len(block_text) <= ATOMIC_BLOCK_MAX:
            # Keep the block atomic — no secondary split
            sub_chunks = [block_text]
        else:
            # Secondary split: block is too large for a single chunk
            logger.debug(
                f"[CHUNKER] Block at offset {block_start} length={len(block_text)} "
                f"> {ATOMIC_BLOCK_MAX}. Applying secondary split."
            )
            sub_chunks = splitter.split_text(block_text)

        for sub_text in sub_chunks:
            sub_text = sub_text.strip()
            if not sub_text:
                continue

            chunk_id = _make_chunk_id(doc_record.doc_id, chunk_index)

            metadata = ChunkMetadata(
                doc_id          = doc_record.doc_id,
                source_file     = doc_record.source_file,
                title           = doc_record.title,
                category        = doc_record.category.value,
                department      = doc_record.department,
                version         = doc_record.version,
                access_level    = doc_record.access_level.value,
                upload_date     = doc_record.upload_date,
                chunk_index     = chunk_index,
                total_chunks    = 0,  # Back-filled below
                section_heading = section_heading,
            )

            chunk = ChunkRecord(
                chunk_id    = chunk_id,
                doc_id      = doc_record.doc_id,
                chunk_index = chunk_index,
                content     = sub_text,
                metadata    = metadata,
            )
            chunks.append(chunk)
            chunk_index += 1

    # Back-fill total_chunks across all chunks now that we know the final count
    total = len(chunks)
    for chunk in chunks:
        chunk.metadata.total_chunks = total

    logger.info(
        f"[CHUNKER] '{doc_record.department}' → {total} chunks "
        f"(avg {int(sum(len(c.content) for c in chunks) / max(total, 1))} chars)"
    )
    return chunks


# ---------------------------------------------------------------------------
# SQLite persistence
# ---------------------------------------------------------------------------

def save_chunks_to_ledger(chunks: list[ChunkRecord]) -> None:
    """Writes all ChunkRecord objects to the SQLite chunks table via ledger.py."""
    if not chunks:
        return
    records = [c.to_ledger_dict() for c in chunks]
    ledger.save_chunks(records)
    logger.info(f"[CHUNKER] Saved {len(chunks)} chunks to ledger.")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_chunking(doc_records: list[DocumentRecord], doc_texts: dict[str, str]) -> dict[str, list[ChunkRecord]]:
    """
    Orchestrates chunking across all ingested documents.

    Args:
        doc_records: List of DocumentRecord objects from ingestion.
        doc_texts:   Mapping of doc_id → serialized text string.

    Returns:
        Mapping of doc_id → list of ChunkRecord objects.
        Also writes all chunks to SQLite and updates document status to 'chunked'.
    """
    all_chunks: dict[str, list[ChunkRecord]] = {}

    for doc in doc_records:
        text = doc_texts.get(doc.doc_id, "")
        if not text:
            logger.warning(f"[CHUNKER] No text found for doc_id={doc.doc_id}. Skipping.")
            continue

        try:
            chunks = chunk_document(doc, text)
            if not chunks:
                logger.warning(f"[CHUNKER] Zero chunks produced for: {doc.source_file}")
                continue

            # Persist to SQLite
            save_chunks_to_ledger(chunks)

            # Update document record: total_chunks + status → 'chunked'
            ledger.update_document_post_chunking(
                doc_id=doc.doc_id,
                total_chunks=len(chunks),
            )

            all_chunks[doc.doc_id] = chunks

        except Exception as e:
            logger.error(f"[CHUNKER] Failed to chunk '{doc.source_file}': {e}", exc_info=True)
            ledger.log_event(doc.source_file, "chunking", "failed", str(e))

    total_chunks = sum(len(v) for v in all_chunks.values())
    logger.info(f"[CHUNKER] Chunking complete. Total chunks generated: {total_chunks}")
    return all_chunks


# ---------------------------------------------------------------------------
# Embedding integration point (Phase 4 hook)
# ---------------------------------------------------------------------------

def get_embedding_payloads(chunks: list[ChunkRecord]) -> list[dict]:
    """
    Converts a list of ChunkRecords into the payload format expected by
    Phase 4 (Embeddings) and Phase 5 (Vector Indexing).

    Returns:
        List of dicts: { "chunk_id", "content", "metadata" (flat dict) }
    """
    return [chunk.to_embedding_payload() for chunk in chunks]
