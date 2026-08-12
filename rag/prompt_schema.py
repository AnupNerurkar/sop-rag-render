"""
rag/prompt_schema.py
---------------------
Data models for the Prompt Builder layer.

Intentionally dependency-free (no imports from other project modules)
so every downstream module — including future LangGraph agents — can
import from here without circular risk.

Model hierarchy
    PromptTemplate      — enum: DEFAULT | CONCISE | STRICT_CITATION
    ConfidenceLabel     — enum: High | Medium | Low | Unknown
    PromptConfig        — configurable knobs passed to PromptBuilder.build()
    ContextChunk        — one prepared chunk ready for the context block
    ConflictGroup       — detected version conflict across chunks
    BuiltPrompt         — the fully assembled prompt, ready for the active LLM backend

LangGraph note:
    BuiltPrompt.messages is list[dict] in Ollama/OpenAI chat format.
    Phase 10 agents can pass it directly to ChatOllama or OllamaLLM
    without any adapter.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class PromptTemplate(str, Enum):
    DEFAULT         = "default"
    CONCISE         = "concise"
    STRICT_CITATION = "strict_citation"


def confidence_from_score(score: Optional[float]) -> str:
    """Converts a numeric score in [0, 1] to a percentage string."""
    if score is None:
        return "0%"
    return f"{round(score * 100)}%"


# ---------------------------------------------------------------------------
# ContextChunk
# ---------------------------------------------------------------------------

class ContextChunk(BaseModel):
    """
    A single retrieved chunk as prepared for the prompt context block.
    All fields needed to render a numbered [SOURCE N] block.
    """

    source_number:    int            = Field(..., description="1-based source number shown to the LLM.")
    chunk_id:         str            = Field(..., description="Original chunk_id from the retrieval layer.")
    content:          str            = Field(..., description="Chunk text (may be truncated to fit budget).")
    inline_citation:  str            = Field(..., description="'[DisplayName v1.0 § Heading (chunk N/M)]'")
    display_citation: str            = Field(..., description="Full display citation for the API response.")
    department:       str            = Field(default="General")
    category:         str            = Field(default="SOP")
    version:          str            = Field(default="1.0")
    doc_id:           str            = Field(default="")
    section_heading:  str            = Field(default="")
    rank:             int            = Field(..., description="Retrieval rank (1 = best).")
    score:            float          = Field(..., description="Relevance score [0, 1].")
    rerank_score:     Optional[float] = Field(default=None)
    confidence:       str            = Field(default="0%")

    def to_prompt_block(self, include_metadata: bool = True) -> str:
        """
        Renders this chunk as a numbered context block.

        Example output:
            [SOURCE 1] VIT Admissions SOP (v1.0) — Admission Process — chunk 3 of 27
            Department: Admissions | Category: SOP | Version: 1.0 | Score: 0.934 | Confidence: 93%
            ---
            <chunk text>
        """
        lines = [f"[SOURCE {self.source_number}] {self.display_citation}"]
        if include_metadata:
            eff_score = self.rerank_score if self.rerank_score is not None else self.score
            lines.append(
                f"Department: {self.department} | Category: {self.category} | "
                f"Version: {self.version} | Score: {eff_score:.3f} | "
                f"Confidence: {self.confidence}"
            )
        lines.append("---")
        lines.append(self.content)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# ConflictGroup
# ---------------------------------------------------------------------------

class ConflictGroup(BaseModel):
    """
    A version conflict: the same document appears under different version strings
    within a single result set.

    The system prompt injects a WARNING for each ConflictGroup so the LLM
    explicitly acknowledges the conflict and prefers the latest version.
    """

    display_name:   str       = Field(..., description="Shared document display name.")
    department:     str       = Field(...)
    category:       str       = Field(...)
    versions:       list[str] = Field(..., description="All distinct versions, newest-first.")
    source_numbers: list[int] = Field(..., description="[SOURCE N] indices involved.")
    latest_version: str       = Field(..., description="Version string judged to be newest.")

    def to_warning(self) -> str:
        versions_str = ", ".join(f"v{v}" for v in self.versions)
        sources_str  = ", ".join(f"[SOURCE {n}]" for n in self.source_numbers)
        return (
            f"CONFLICT DETECTED: '{self.display_name}' ({self.department} — {self.category}) "
            f"appears in {versions_str}. Prefer v{self.latest_version} ({sources_str}). "
            f"Explicitly state when an earlier-version source contradicts the latest."
        )


# ---------------------------------------------------------------------------
# PromptConfig
# ---------------------------------------------------------------------------

class PromptConfig(BaseModel):
    """
    Configurable parameters for PromptBuilder.build().
    Every field has a safe default; callers only override what they need.
    """

    template: PromptTemplate = Field(
        default=PromptTemplate.DEFAULT,
        description="System-prompt template variant.",
    )
    max_context_chars: int = Field(
        default=8000,
        ge=500,
        le=32000,
        description=(
            "Hard cap on total context block characters. "
            "Chunks are trimmed to fit; the last chunk is truncated if needed."
        ),
    )
    include_metadata: bool = Field(
        default=True,
        description="Include department/category/confidence metadata row per chunk.",
    )
    max_chunks: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of chunks included regardless of budget.",
    )
    confidence_threshold: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Drop chunks whose effective score (rerank_score if present, else score) "
            "is below this threshold."
        ),
    )


# ---------------------------------------------------------------------------
# BuiltPrompt
# ---------------------------------------------------------------------------

class BuiltPrompt(BaseModel):
    """
    The fully assembled prompt, ready for the active LLM backend (LLM_BACKEND
    selects it -- see rag/rag_engine.py; Groq in production).

    Primary consumer interface
    --------------------------
    Phase 9 (LLM call):
        pass built_prompt.messages to POST /api/chat

    Phase 10 (LangGraph agents):
        pass built_prompt.messages to ChatOllama(messages=...)
        or convert with LangChain's convert_to_messages helper

    Debugging / logging:
        built_prompt.to_flat_string()   — single-string view
        built_prompt.summary()          — one-line stats
    """

    # --- Core prompt components ---
    user_question:  str  = Field(..., description="Original user question (not preprocessed).")
    system_prompt:  str  = Field(..., description="Full system instruction block.")
    context_block:  str  = Field(..., description="Assembled numbered [SOURCE N] context blocks.")
    user_message:   str  = Field(..., description="User turn sent to Qwen2.5 (context + question).")

    # --- Ollama/OpenAI chat messages (primary output) ---
    messages: list[dict] = Field(
        ...,
        description=(
            "Chat messages: [{'role': 'system', 'content': ...}, {'role': 'user', 'content': ...}]. "
            "Passed directly to Ollama /api/chat in Phase 9 and to LangGraph ChatOllama in Phase 10."
        ),
    )

    # --- Stats ---
    chunks_included: int = Field(..., description="Chunks in the context block.")
    chunks_dropped:  int = Field(default=0, description="Chunks excluded (budget or threshold).")
    context_chars:   int = Field(..., description="Character count of context_block.")
    template_used:   PromptTemplate = Field(...)

    # --- Conflict detection ---
    conflicts:    list[ConflictGroup] = Field(default_factory=list)
    has_conflicts: bool               = Field(default=False)

    # --- Citation list for Phase 8 (Source Citations API) ---
    source_citations: list[str] = Field(
        default_factory=list,
        description="Ordered display citation strings matching [SOURCE 1], [SOURCE 2], ... numbering.",
    )

    # --- Prepared chunks (for testing and downstream inspection) ---
    context_chunks: list[ContextChunk] = Field(
        default_factory=list,
        description="Structured chunk objects used to assemble context_block.",
    )

    def to_flat_string(self) -> str:
        """
        Single-string rendering for non-chat Ollama endpoints and for debugging.

        Format:
            <system>...</system>
            <context>...</context>
            <question>...</question>
        """
        return (
            f"<system>\n{self.system_prompt}\n</system>\n\n"
            f"<context>\n{self.context_block}\n</context>\n\n"
            f"<question>\n{self.user_question}\n</question>"
        )

    def summary(self) -> str:
        lines = [
            f"Template : {self.template_used.value}",
            f"Chunks   : {self.chunks_included} included, {self.chunks_dropped} dropped",
            f"Chars    : {self.context_chars}",
            f"Conflicts: {len(self.conflicts)}",
        ]
        return "\n".join(lines)
