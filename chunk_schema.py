"""
chunk_schema.py
---------------
Pydantic data models for Document and Chunk records.
Used across ingestion_pipeline.py, chunker.py, and future embeddings + the vector store phases.

Access Level Hierarchy (RBAC Foundation):
    Admin   → can access all levels
    Faculty → can access Faculty, Student, Public
    Student → can access Student, Public
    Public  → can access Public only (default fallback)
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class AccessLevel(str, Enum):
    """Role-based access tier for documents and chunks."""
    PUBLIC  = "Public"
    STUDENT = "Student"
    FACULTY = "Faculty"
    ADMIN   = "Admin"


class DocumentCategory(str, Enum):
    """Institutional document taxonomy."""
    SOP           = "SOP"
    POLICY        = "Policy"
    CIRCULAR      = "Circular"
    HANDBOOK      = "Handbook"
    ACCREDITATION = "Accreditation"
    GUIDELINE     = "Guideline"
    FORM          = "Form"
    UNKNOWN       = "Unknown"


class DocumentStatus(str, Enum):
    """Processing pipeline stage for a document."""
    ASSESSED  = "assessed"
    INGESTED  = "ingested"
    CHUNKED   = "chunked"
    EMBEDDED  = "embedded"     # populated in Phase 4
    INDEXED   = "indexed"      # populated in Phase 5
    SUPERSEDED = "superseded"  # set when a newer version replaces this doc


# ---------------------------------------------------------------------------
# Document Record
# ---------------------------------------------------------------------------

class DocumentRecord(BaseModel):
    """
    Represents one source document in the knowledge base.
    Populated during Phase 2 (Ingestion).
    Stored in the SQLite `documents` table.
    """

    doc_id: str = Field(
        ...,
        description="SHA-256 hash of the staged .docx file. Used as the stable, "
                    "version-agnostic foreign key across all child chunks."
    )
    source_file: str = Field(
        ...,
        description="Relative path to the staged .docx file (e.g. 'data/staging/1. VIT Admissions.docx')."
    )
    original_file: str = Field(
        default="",
        description="Relative path to the original raw file before conversion (e.g. 'data/1. VIT Admissions.doc')."
    )
    title: str = Field(
        default="",
        description="Document title extracted from the first-page header text."
    )
    category: DocumentCategory = Field(
        default=DocumentCategory.SOP,
        description="Institutional document type (SOP, Policy, Circular, etc.)."
    )
    department: str = Field(
        default="General",
        description="VIT administrative module / department owning this document."
    )
    version: str = Field(
        default="1.0",
        description="Document revision identifier extracted from filename or body text."
    )
    access_level: AccessLevel = Field(
        default=AccessLevel.PUBLIC,
        description="Access tier that controls retrieval visibility. Defaults to Public."
    )
    upload_date: str = Field(
        default="",
        description="ISO 8601 date string. Falls back to file system modification date."
    )
    total_chunks: int = Field(
        default=0,
        description="Number of chunks generated from this document. Back-filled after Phase 3."
    )
    status: DocumentStatus = Field(
        default=DocumentStatus.INGESTED,
        description="Current pipeline stage of the document."
    )
    ingested_at: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="UTC timestamp when this document was first ingested."
    )

    @field_validator("access_level", mode="before")
    @classmethod
    def coerce_access_level(cls, v):
        """Normalize and validate access level. Unknown values default to Public."""
        if isinstance(v, str):
            normalized = v.strip().title()
            try:
                return AccessLevel(normalized)
            except ValueError:
                return AccessLevel.PUBLIC
        return v

    @field_validator("category", mode="before")
    @classmethod
    def coerce_category(cls, v):
        """Normalize category string. Unknown values default to SOP for VIT corpus."""
        if isinstance(v, str):
            normalized = v.strip().title()
            try:
                return DocumentCategory(normalized)
            except ValueError:
                return DocumentCategory.SOP
        return v

    def to_ledger_dict(self) -> dict:
        """Serializes the record for SQLite storage."""
        return {
            "doc_id":        self.doc_id,
            "source_file":   self.source_file,
            "original_file": self.original_file,
            "title":         self.title,
            "category":      self.category.value,
            "department":    self.department,
            "version":       self.version,
            "access_level":  self.access_level.value,
            "upload_date":   self.upload_date,
            "total_chunks":  self.total_chunks,
            "status":        self.status.value,
            "ingested_at":   self.ingested_at,
        }


# ---------------------------------------------------------------------------
# Chunk Metadata (embeds in every chunk, propagated to the vector store)
# ---------------------------------------------------------------------------

class ChunkMetadata(BaseModel):
    """
    Metadata payload attached to every chunk.
    Propagated in full to vector-store metadata dict in Phase 5.
    Enables retrieval-time filtering by access_level, category, department, etc.
    """

    doc_id: str = Field(..., description="Parent DocumentRecord.doc_id.")
    source_file: str = Field(..., description="Relative path to the staged source file.")
    title: str = Field(default="", description="Parent document title.")
    category: str = Field(default="SOP", description="Parent document category.")
    department: str = Field(default="General", description="Parent document department/module.")
    version: str = Field(default="1.0", description="Parent document version.")

    # --- RBAC field: must propagate from parent DocumentRecord ---
    access_level: str = Field(
        default="Public",
        description="Access tier propagated from parent document. "
                    "Used for query-time metadata filtering."
    )

    upload_date: str = Field(default="", description="Source document upload/issue date.")
    chunk_index: int = Field(..., description="0-based position of this chunk within its document.")
    total_chunks: int = Field(
        default=0,
        description="Total number of chunks in the parent document. Back-filled after chunking."
    )
    section_heading: str = Field(
        default="",
        description="Nearest section heading above this chunk (Word Heading style or SUB-PROCESS marker)."
    )

    def to_metadata_dict(self) -> dict:
        """
        Returns a flat dict compatible with vector-store metadata storage.
        The metadata store only supports str / int / float / bool values — no nested objects.
        """
        return {
            "doc_id":          self.doc_id,
            "source_file":     self.source_file,
            "title":           self.title,
            "category":        self.category,
            "department":      self.department,
            "version":         self.version,
            "access_level":    self.access_level,   # KEY field for RBAC filtering
            "upload_date":     self.upload_date,
            "chunk_index":     self.chunk_index,
            "total_chunks":    self.total_chunks,
            "section_heading": self.section_heading,
        }


# ---------------------------------------------------------------------------
# Chunk Record
# ---------------------------------------------------------------------------

class ChunkRecord(BaseModel):
    """
    Represents one text chunk derived from a DocumentRecord.
    Populated during Phase 3 (Chunking).
    Stored in the SQLite `chunks` table.
    Passed to Phase 4 (Embeddings) as the unit of vectorization.
    """

    chunk_id: str = Field(
        ...,
        description="Deterministic 16-character hex ID: SHA-256(doc_id::chunk_index)[:16]. "
                    "Stable across re-ingestion of the same document version."
    )
    doc_id: str = Field(..., description="Foreign key → DocumentRecord.doc_id.")
    chunk_index: int = Field(..., description="0-based position within the parent document.")
    content: str = Field(..., description="The raw text content of this chunk.")
    metadata: ChunkMetadata = Field(..., description="Full metadata payload for this chunk.")
    created_at: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="UTC timestamp when this chunk was generated."
    )

    def to_embedding_payload(self) -> dict:
        """
        Returns the dict expected by the Phase 4 embeddings module.
        Structure: { chunk_id, content, metadata (flat metadata dict) }
        """
        return {
            "chunk_id": self.chunk_id,
            "content":  self.content,
            "metadata": self.metadata.to_metadata_dict(),
        }

    def to_ledger_dict(self) -> dict:
        """Serializes the chunk record for SQLite storage."""
        return {
            "chunk_id":        self.chunk_id,
            "doc_id":          self.doc_id,
            "chunk_index":     self.chunk_index,
            "content":         self.content,
            "section_heading": self.metadata.section_heading,
            "category":        self.metadata.category,
            "department":      self.metadata.department,
            "access_level":    self.metadata.access_level,
            "version":         self.metadata.version,
            "source_file":     self.metadata.source_file,
            "total_chunks":    self.metadata.total_chunks,
            "created_at":      self.created_at,
        }


# ---------------------------------------------------------------------------
# Ingestion Run Summary (for admin dashboard / logging)
# ---------------------------------------------------------------------------

class IngestionSummary(BaseModel):
    """Summary emitted at the end of a full ingestion + chunking run."""

    run_id: str = Field(..., description="Unique ID for this ingestion run (timestamp-based).")
    started_at: str
    completed_at: str = ""
    total_docs_discovered: int = 0
    total_docs_processed: int = 0
    total_docs_skipped: int = 0      # unchanged hash — already indexed
    total_docs_superseded: int = 0   # old version replaced by new
    total_docs_failed: int = 0
    total_chunks_generated: int = 0
    errors: list[str] = Field(default_factory=list)

    def report(self) -> str:
        lines = [
            "=" * 60,
            f"INGESTION RUN: {self.run_id}",
            f"  Started  : {self.started_at}",
            f"  Completed: {self.completed_at}",
            "-" * 60,
            f"  Docs Discovered : {self.total_docs_discovered}",
            f"  Docs Processed  : {self.total_docs_processed}",
            f"  Docs Skipped    : {self.total_docs_skipped} (unchanged)",
            f"  Docs Superseded : {self.total_docs_superseded}",
            f"  Docs Failed     : {self.total_docs_failed}",
            f"  Chunks Generated: {self.total_chunks_generated}",
        ]
        if self.errors:
            lines.append("-" * 60)
            lines.append("  Errors:")
            for e in self.errors:
                lines.append(f"    • {e}")
        lines.append("=" * 60)
        return "\n".join(lines)
