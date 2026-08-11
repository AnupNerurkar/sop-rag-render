"""
retrieval/retrieval_schema.py
------------------------------
Phase 6: Retrieval Data Models

All Pydantic models used by the retrieval layer.
No imports from other project modules — deliberately dependency-free
so every other module can import from here without circular risk.

Model hierarchy:
    RetrievalFilter      — optional metadata pre-filters (department, category, doc_id, version)
    RetrievalQuery       — the full query spec (text + role + filters + options)
    SourceCitation       — human-readable provenance for a single result
    RetrievalResult      — one ranked chunk (content + score + citation + metadata)
    RetrievalResponse    — the full response (results + stats)

RBAC roles (mirrors chunk_schema.AccessLevel):
    Admin   → can query all access levels
    Faculty → Public, Student, Faculty
    Student → Public, Student
    Public  → Public only  (default)
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_ROLES = {"Admin", "Faculty", "Student", "Public"}

DEFAULT_TOP_K       = 10    # Final results returned to caller
MAX_TOP_K           = 50
MIN_QUERY_LENGTH    = 2

# Hybrid retrieval defaults
DEFAULT_TOP_K_DENSE  = 25   # Dense candidates fed into RRF
DEFAULT_TOP_K_BM25   = 25   # BM25 candidates fed into RRF
DEFAULT_TOP_K_FUSION = 25   # Candidates entering the reranker
DEFAULT_TOP_K_FINAL  = 5    # Final results after reranking


# ---------------------------------------------------------------------------
# RetrievalFilter
# ---------------------------------------------------------------------------

class RetrievalFilter(BaseModel):
    """
    Optional metadata pre-filters applied before vector similarity search.
    All fields are optional — omitting a field means no filter on that dimension.

    Multiple fields are combined with AND logic inside the store.

    Note on doc_id: pass the full 64-character SHA-256 hex string to retrieve
    chunks from one specific document version only.
    """

    department: Optional[str] = Field(
        default=None,
        description="Exact department/module name (e.g. 'Admissions', 'Examination').",
    )
    category: Optional[str] = Field(
        default=None,
        description="Document category (e.g. 'SOP', 'Policy', 'Circular').",
    )
    doc_id: Optional[str] = Field(
        default=None,
        description="Full SHA-256 doc_id to restrict retrieval to one document version.",
    )
    version: Optional[str] = Field(
        default=None,
        description="Document version string (e.g. '1.0', '3.0', 'Final').",
    )

    @field_validator("department", "category", "version", mode="before")
    @classmethod
    def strip_strings(cls, v):
        return v.strip() if isinstance(v, str) else v

    @property
    def is_empty(self) -> bool:
        """True when no filters are set — avoids adding a no-op where clause."""
        return all(
            f is None
            for f in (self.department, self.category, self.doc_id, self.version)
        )


# ---------------------------------------------------------------------------
# RetrievalQuery
# ---------------------------------------------------------------------------

class RetrievalQuery(BaseModel):
    """
    The complete specification for a single retrieval request.

    Validated at construction — if `top_k` is out of range or the query text
    is empty, Pydantic raises before any embedding or the vector store work happens.
    """

    text: str = Field(
        ...,
        description="The user's natural-language question or search phrase.",
    )
    role: str = Field(
        default="Public",
        description="Caller's RBAC role. Controls which access_level chunks are visible.",
    )
    filters: RetrievalFilter = Field(
        default_factory=RetrievalFilter,
        description="Optional metadata pre-filters.",
    )
    top_k: int = Field(
        default=DEFAULT_TOP_K,
        ge=1,
        le=MAX_TOP_K,
        description=f"Number of chunks to return. Range [1, {MAX_TOP_K}].",
    )
    remove_stopwords: bool = Field(
        default=False,
        description=(
            "If True, common English stopwords are removed from the query before "
            "embedding. Disabled by default — stopword removal can destroy intent "
            "in short queries like 'what is the admission process'."
        ),
    )
    normalize_whitespace: bool = Field(
        default=True,
        description="Collapse runs of whitespace and strip the query.",
    )

    # --- Hybrid retrieval controls ---
    use_bm25: bool = Field(
        default=True,
        description="Enable BM25 keyword retrieval alongside dense vector retrieval.",
    )
    use_reranker: bool = Field(
        default=True,
        description="Enable cross-encoder reranking of fused candidates.",
    )
    top_k_dense: int = Field(
        default=DEFAULT_TOP_K_DENSE,
        ge=1, le=100,
        description="Number of candidates fetched from dense retrieval.",
    )
    top_k_bm25: int = Field(
        default=DEFAULT_TOP_K_BM25,
        ge=1, le=100,
        description="Number of candidates fetched from BM25 keyword retrieval.",
    )
    top_k_fusion: int = Field(
        default=DEFAULT_TOP_K_FUSION,
        ge=1, le=100,
        description="Number of candidates entering the reranker after RRF fusion.",
    )
    top_k_final: int = Field(
        default=DEFAULT_TOP_K_FINAL,
        ge=1, le=MAX_TOP_K,
        description="Final number of results returned after reranking.",
    )

    @field_validator("role", mode="before")
    @classmethod
    def normalize_role(cls, v):
        if isinstance(v, str):
            normalized = v.strip().title()
            if normalized not in VALID_ROLES:
                return "Public"
            return normalized
        return "Public"

    @field_validator("text", mode="before")
    @classmethod
    def validate_text(cls, v):
        if not isinstance(v, str) or len(v.strip()) < MIN_QUERY_LENGTH:
            raise ValueError(
                f"Query text must be at least {MIN_QUERY_LENGTH} characters."
            )
        return v


# ---------------------------------------------------------------------------
# SourceCitation
# ---------------------------------------------------------------------------

class SourceCitation(BaseModel):
    """
    Human-readable provenance attached to every RetrievalResult.
    Used by Phase 7 RAG to build the citation block in the LLM prompt,
    and by Phase 8 to render citations in the API response.

    Design principle: every field has a non-empty fallback so citations
    are always renderable even when metadata is sparse.
    """

    doc_id:          str = Field(..., description="Parent document SHA-256 hash.")
    source_file:     str = Field(..., description="Relative path to the staged .docx file.")
    display_name:    str = Field(..., description="Human-readable document name (title or filename stem).")
    department:      str = Field(default="General", description="Owning department/module.")
    category:        str = Field(default="SOP",     description="Document category.")
    version:         str = Field(default="1.0",     description="Document version.")
    section_heading: str = Field(default="",        description="Nearest section heading above this chunk.")
    chunk_index:     int = Field(default=0,         description="0-based position of this chunk in the document.")
    total_chunks:    int = Field(default=0,         description="Total chunks in the parent document.")

    def to_inline_citation(self) -> str:
        """
        Returns a compact inline citation string for use inside LLM prompts.
        Example: '[Admissions SOP v1.0 § Conducting Term Test (chunk 3/27)]'
        """
        heading_part = f" § {self.section_heading}" if self.section_heading else ""
        chunk_part   = f" (chunk {self.chunk_index + 1}/{self.total_chunks})" if self.total_chunks > 0 else ""
        return f"[{self.display_name} v{self.version}{heading_part}{chunk_part}]"

    def to_display_citation(self) -> str:
        """
        Returns a full display citation for the API response.
        Example: 'VIT Admissions SOP (v1.0) — Conducting Term Test — chunk 3 of 27'
        """
        parts = [f"{self.display_name} (v{self.version})"]
        if self.section_heading:
            parts.append(self.section_heading)
        if self.total_chunks > 0:
            parts.append(f"chunk {self.chunk_index + 1} of {self.total_chunks}")
        return " — ".join(parts)


# ---------------------------------------------------------------------------
# RetrievalResult
# ---------------------------------------------------------------------------

class RetrievalResult(BaseModel):
    """
    A single ranked retrieval result.

    Returned as part of RetrievalResponse. Phase 7 (RAG) reads `content` and
    `citation` to assemble the context window. Phase 8 (Source Citations) renders
    `citation.to_display_citation()` in the API response.
    """

    rank:          int            = Field(..., description="1-based rank in this result set.")
    chunk_id:      str            = Field(..., description="Deterministic 16-char chunk identifier.")
    content:       str            = Field(..., description="Raw chunk text passed to the LLM context window.")
    score:         float          = Field(..., description="Relevance score in [0, 1]. Higher is better.")
    distance:      Optional[float]= Field(default=None, description="Cosine distance. Lower is more similar. None for FTS-only (keyword) hits, which aren't on the cosine scale.")
    rerank_score:  Optional[float]= Field(default=None, description="Cross-encoder rerank score. None if reranker not used.")
    retrieval_mode: str           = Field(default="dense", description="'dense' | 'bm25' | 'hybrid' | 'hybrid+rerank'")
    citation:      SourceCitation = Field(..., description="Full source provenance.")

    # Raw metadata dict preserved for downstream filtering and debug
    metadata: dict = Field(default_factory=dict)

    def to_context_block(self) -> str:
        """
        Formats this result as a context block for the RAG prompt.

        Format:
            [SOURCE: citation]
            <content>
        """
        return f"[SOURCE: {self.citation.to_inline_citation()}]\n{self.content}"


# ---------------------------------------------------------------------------
# RetrievalResponse
# ---------------------------------------------------------------------------

class RetrievalResponse(BaseModel):
    """
    The complete response from a single retrieval call.
    Wraps results with query echo, statistics, and timing for the API layer.
    """

    query_text:       str   = Field(..., description="The original (pre-preprocessing) query text.")
    clean_query_text: str   = Field(..., description="The preprocessed query text that was embedded.")
    role:             str   = Field(..., description="The RBAC role used for this query.")
    results:          list[RetrievalResult] = Field(default_factory=list)

    # Statistics
    total_results:    int   = Field(default=0, description="Number of results returned.")
    top_k_requested:  int   = Field(default=DEFAULT_TOP_K)
    retrieved_at:     str   = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="UTC timestamp of the retrieval call.",
    )
    latency_ms:       float = Field(default=0.0, description="Total retrieval latency in milliseconds.")
    reranked:         bool  = Field(default=False, description="True only if the reranker actually reordered results -- not just requested.")
    rerank_method:    Optional[str] = Field(default=None, description="'groq_listwise' if reranking happened; None otherwise (circuit open, call failed, invalid response, etc).")
    retrieval_mode:   str   = Field(default="dense", description="'dense' | 'hybrid' | 'hybrid+rerank'")

    # Applied filters (echoed back for transparency)
    applied_filters:  dict  = Field(
        default_factory=dict,
        description="The filters that were active during this retrieval.",
    )

    @property
    def has_results(self) -> bool:
        return len(self.results) > 0

    def to_context_window(self) -> str:
        """
        Assembles all result blocks into a single string for the RAG prompt.
        Each block is separated by a blank line.
        """
        return "\n\n".join(r.to_context_block() for r in self.results)

    def summary(self) -> str:
        lines = [
            f"Query    : {self.query_text}",
            f"Role     : {self.role}",
            f"Results  : {self.total_results}",
            f"Latency  : {self.latency_ms:.1f} ms",
        ]
        if self.applied_filters:
            lines.append(f"Filters  : {self.applied_filters}")
        if self.results:
            top = self.results[0]
            lines.append(
                f"Top match: score={top.score:.4f} | "
                f"{top.citation.department} | "
                f"{top.citation.section_heading[:60]}"
            )
        return "\n".join(lines)
