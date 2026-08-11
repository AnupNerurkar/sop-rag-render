"""
rag/citation_schema.py
-----------------------
Data models for Phase 8: Source Citations.

Dependency-free (no imports from other project modules).
Pydantic v2.

Model hierarchy:
    Citation     — one cited document (may cover multiple retrieved chunks)
    CitationList — ordered, deduplicated citation set with rendering methods
"""

from __future__ import annotations

import json
from typing import Optional
from pydantic import BaseModel, Field


class Citation(BaseModel):
    """
    One cited source document as surfaced to the end user.

    A single Citation may represent several retrieved chunks from the same
    document — chunk_ids holds all of them; score reflects the best one.
    """

    citation_id:       str   = Field(...,
        description="Stable ID derived from doc_id. Safe for frontend anchors.")
    rank:              int   = Field(...,
        description="1-based position in the sorted citation list.")
    inline_ref:        str   = Field(...,
        description="Footnote-style reference, e.g. '[1]'.")

    # Document identity
    doc_id:            str   = Field(...,  description="SHA-256 document hash.")
    display_name:      str   = Field(...,  description="Human-readable document title.")
    title:             str   = Field(default="", description="Parent document title.")
    department:        str   = Field(default="General")
    category:          str   = Field(default="SOP")
    version:           str   = Field(default="1.0")
    source_file:       str   = Field(default="", description="Relative path to source .docx")
    filepath:          str   = Field(default="", description="Absolute or resolved file path.")
    section_heading:   str   = Field(default="")
    access_level:      str   = Field(default="Public")

    # Chunk provenance
    chunk_ids:         list[str] = Field(default_factory=list,
        description="All chunk IDs from this document included in the retrieval set.")
    chunk_id:          str   = Field(default="", description="Main chunk ID referenced.")
    chunk_index:       int   = Field(default=0,
        description="0-based index of the earliest included chunk (document position).")
    total_chunks:      int   = Field(default=0,
        description="Total chunks in parent document. 0 if unknown.")
    page_number:       int   = Field(default=0,
        description="Approximate 1-based position: chunk_index + 1. 0 if unknown.")

    # Scores
    score:             float         = Field(..., description="Best effective score [0, 1].")
    rerank_score:      Optional[float] = Field(default=None)
    retrieval_score:   float         = Field(default=0.0, description="Main chunk retrieval score.")


    # Version conflict flag
    is_latest_version: bool  = Field(default=True,
        description="False when a newer version of this document is also in the result set.")

    # Future frontend link
    url:               Optional[str] = Field(default=None,
        description="Clickable link for the frontend. Populated by the API layer.")

    # ----------------------------------------------------------------
    # Rendering
    # ----------------------------------------------------------------

    def to_markdown_entry(self) -> str:
        """
        Single-line Markdown entry for the References section.

        Example:
            [1] **Fee Payment SOP** (v2.0) — Finance — SOP
                Section: Fee Deadlines | Chunk: 3/27 | Score: 0.934 | Access: Public
        """
        heading = f" | Section: {self.section_heading}" if self.section_heading else ""
        chunk_part = (
            f" | Chunk: {self.chunk_index + 1}/{self.total_chunks}"
            if self.total_chunks > 0 else ""
        )
        eff_score = self.rerank_score if self.rerank_score is not None else self.score
        superseded = " ⚠ older version" if not self.is_latest_version else ""
        line1 = f"{self.inline_ref} **{self.display_name}** (v{self.version}) — {self.department} — {self.category}{superseded}"
        line2 = f"    Score: {eff_score:.3f}{heading}{chunk_part} | Access: {self.access_level}"
        return f"{line1}\n{line2}"

    def to_dict_entry(self) -> dict:
        """Clean dict for JSON serialisation. Scores rounded to 4 dp."""
        eff_score = self.rerank_score if self.rerank_score is not None else self.score
        return {
            "rank":              self.rank,
            "inline_ref":        self.inline_ref,
            "citation_id":       self.citation_id,
            "doc_id":            self.doc_id,
            "display_name":      self.display_name,
            "title":             self.title or self.display_name,
            "department":        self.department,
            "category":          self.category,
            "version":           self.version,
            "source_file":       self.source_file,
            "filepath":          self.filepath,
            "section_heading":   self.section_heading,
            "chunk_ids":         self.chunk_ids,
            "chunk_id":          self.chunk_id,
            "chunk_index":       self.chunk_index,
            "total_chunks":      self.total_chunks,
            "page_number":       self.page_number,
            "score":             round(eff_score, 4),
            "rerank_score":      round(self.rerank_score, 4) if self.rerank_score is not None else None,
            "retrieval_score":   round(self.retrieval_score, 4),
            "access_level":      self.access_level,
            "is_latest_version": self.is_latest_version,
            "url":               self.url,
        }


class CitationList(BaseModel):
    """
    Ordered, deduplicated citation set produced by CitationEngine.

    Primary outputs:
        answer_with_refs  — answer text with [SOURCE N] replaced by [N]
        to_markdown()     — Markdown references block
        to_json()         — JSON string of citations + annotated answer
    """

    citations:             list[Citation]
    answer_with_refs:      str  = Field(...,
        description="Generated answer with [SOURCE N] placeholders replaced by inline [N] refs.")
    original_answer:       str  = Field(...,
        description="Answer as returned by RAGEngine, before ref injection.")
    total_citations:       int  = Field(default=0)
    has_version_conflicts: bool = Field(default=False,
        description="True when any citation is superseded by a newer version in the same set.")
    citations_inferred: bool = Field(default=False,
        description=(
            "True when the answer emitted no valid [SOURCE N] markers and these "
            "citations were inferred (top-ranked chunks actually shown to the "
            "model) rather than read directly off the answer text. The UI should "
            "label these differently -- e.g. 'related sources' rather than "
            "claiming the answer cited them."
        ))
    hallucinated_marker_count: int = Field(default=0,
        description="[SOURCE N] markers in the raw answer that referenced a source "
                     "number never shown to the model. Stripped from answer_with_refs; "
                     "a nonzero count here is a hallucination signal worth logging.")

    # Internal: source_number (1-based from [SOURCE N]) → citation rank
    # Stored as dict[str, int] because Pydantic serialises int keys as strings in JSON.
    source_number_map: dict[str, int] = Field(default_factory=dict)

    # ----------------------------------------------------------------
    # Rendering
    # ----------------------------------------------------------------

    def to_markdown(self) -> str:
        """
        Full Markdown output: annotated answer + references section.

        Format:
            <answer with inline [N] refs>

            ---
            **References**

            [1] **Doc Title** (v2.0) — Department — Category
                Score: 0.934 | ...
        """
        ref_entries = "\n\n".join(c.to_markdown_entry() for c in self.citations)
        conflict_note = (
            "\n\n> ⚠ Some documents have multiple versions in the result set. "
            "Older versions are marked accordingly."
            if self.has_version_conflicts else ""
        )
        return (
            f"{self.answer_with_refs}"
            f"{conflict_note}"
            f"\n\n---\n**References**\n\n{ref_entries}"
        )

    def to_json(self) -> str:
        """JSON string with answer_with_refs and structured citation list."""
        payload = {
            "answer":                self.answer_with_refs,
            "total_citations":       self.total_citations,
            "has_version_conflicts": self.has_version_conflicts,
            "citations":             [c.to_dict_entry() for c in self.citations],
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def get_by_rank(self, rank: int) -> Optional[Citation]:
        """Returns the citation at the given 1-based rank, or None."""
        for c in self.citations:
            if c.rank == rank:
                return c
        return None

    def get_by_source_number(self, source_number: int) -> Optional[Citation]:
        """Maps an original [SOURCE N] number back to its merged Citation."""
        rank = self.source_number_map.get(str(source_number))
        if rank is None:
            return None
        return self.get_by_rank(rank)
