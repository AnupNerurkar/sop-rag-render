"""
response_schema.py
------------------
Phase 9: Structured response schema for the integrated RAG pipeline.

Designed to be the stable contract between RAGPipeline (Phase 9) and
the future FastAPI layer (Phase 10) and LangGraph agents (Phase 11).

Dependency notes:
    - Imports ConfidenceLabel from rag.prompt_schema (pure enum, no I/O).
    - Imports Citation from rag.citation_schema (pure Pydantic model).
    - Does NOT import from rag_pipeline -- no circular dependency risk.
    - TYPE_CHECKING guard on RetrievalResult prevents runtime circular imports.

LangGraph note:
    RAGPipelineResponse can be serialized to/from JSON via .model_dump_json().
    Future LangGraph state graphs can pass it as a typed state field.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Optional, TYPE_CHECKING

from pydantic import BaseModel, Field

from rag.citation_schema import Citation

if TYPE_CHECKING:
    from retrieval.retrieval_schema import RetrievalResult


# ---------------------------------------------------------------------------
# Fallback answer
# ---------------------------------------------------------------------------

# The single canonical "could not answer" text. Previously three different
# strings existed for what were really the same situation from the user's
# perspective (no results at all, low-relevance results, or the model
# reading real context and still finding nothing): this one
# (response_schema.FALLBACK_ANSWER), a shorter one hardcoded into every
# prompt template's rule 2, and a third one in agents/agent_state.py. Only
# the first was ever checked by is_fallback, so an answer produced by
# either of the other two paths silently failed that check -- e.g. the
# recorded baseline case where the model correctly said "I could not find
# this information..." but is_fallback still returned False and the
# response still carried four unrelated citations. prompt_builder.py and
# agents/agent_state.py both now import this constant instead of hardcoding
# their own wording, so all three paths produce byte-identical text and
# is_fallback (below) is a single accurate check regardless of which path
# produced the answer.
FALLBACK_ANSWER: str = (
    "I could not find sufficient institutional evidence to answer this question. "
    "The knowledge base may not contain information on this topic, or access may "
    "be restricted for your role."
)


# ---------------------------------------------------------------------------
# Relevance gate
# ---------------------------------------------------------------------------

# Below this dense cosine similarity, retrieved context is treated as noise
# and the LLM is never called -- this is what actually stops the model from
# confidently answering off irrelevant context (the recorded baseline case:
# "What is the capital of France?" retrieved four SOP chunks from unrelated
# departments, and the model produced a correctly-worded refusal that still
# shipped with four fabricated citations, because nothing gated generation
# on retrieval quality in the first place). Env-overridable so it can be
# tuned on the live box without a rebuild -- it should be calibrated against
# the unanswerable-question bucket in eval-results, not guessed.
RELEVANCE_FLOOR = float(os.environ.get("RELEVANCE_FLOOR", "0.60"))

# Below this many chunks clearing the floor, the pipeline still refuses to
# answer even if the top hit alone is strong -- a single borderline match is
# a weaker basis for a grounded institutional answer than the presence of
# corroborating context, though it doesn't have to be a hard gate on its own
# (top1 clearing RELEVANCE_FLOOR is sufficient by itself; see
# passes_relevance_gate).
MIN_SUPPORTING_CHUNKS = int(os.environ.get("MIN_SUPPORTING_CHUNKS", "1"))


def passes_relevance_gate(results: "list[RetrievalResult]") -> bool:
    """
    True if retrieval quality is high enough to bother calling the LLM at
    all. False means: skip generation, return FALLBACK_ANSWER, zero
    citations -- exactly the same treatment as retrieving zero results, and
    for the same reason (nothing here supports a grounded answer).
    """
    if not results:
        return False
    _, score = compute_confidence(results)
    return score >= RELEVANCE_FLOOR


# ---------------------------------------------------------------------------
# Confidence computation
# ---------------------------------------------------------------------------

def compute_confidence(
    results: "list[RetrievalResult]",
) -> tuple[str, float]:
    """
    Infers answer confidence purely from retrieval signals -- no LLM call,
    and no longer even a channel for one: this used to be a fallback for
    when the model didn't self-report a [Confidence: N%] line, but trusting
    the model's own claim meant "confidence" was usually just whatever
    number it decided to write, disconnected from whether the retrieved
    context actually supported the answer. There is now exactly one
    confidence signal in the whole pipeline, and it comes from here.

    Uses dense cosine similarity only -- FTS-only hits (Phase 2) carry
    distance=None and are skipped, since a BM25/FTS relevance score isn't
    on the cosine scale and treating it as one (as the pre-Phase-2 code did
    with `distance = 1.0 - norm_score`) silently corrupted this exact
    computation.

    Blends the top result with the next two rather than reading rank 0
    alone, so one lucky top hit surrounded by weak support scores lower
    than three consistently strong hits -- and removes the old hard
    similarity<0.70 cliff, which made confidence effectively binary (0% or
    "70%+") and hid everything in between.

    Args:
        results: list[RetrievalResult] in rank order (index 0 = best).

    Returns:
        (confidence_percentage_str, raw_confidence_score)
    """
    dense = [r for r in results if r.distance is not None]
    if not dense:
        return "0%", 0.0

    similarities = [max(0.0, min(1.0, 1.0 - r.distance)) for r in dense[:3]]
    top1 = similarities[0]
    mean_top3 = sum(similarities) / len(similarities)

    raw_confidence_score = 0.7 * top1 + 0.3 * mean_top3
    raw_confidence_score = max(0.0, min(1.0, raw_confidence_score))
    confidence_percentage = round(raw_confidence_score * 100)
    return f"{confidence_percentage}%", raw_confidence_score


# ---------------------------------------------------------------------------
# Sources block formatter
# ---------------------------------------------------------------------------

def format_sources_block(citations: list[Citation]) -> str:
    """
    Renders a numbered 'Sources' section from a CitationList.

    Format per line:
        N. <display_name> | <department> | v<version> | Page <page> | Chunk <n>/<total>

    Returns an empty string when citations is empty.
    """
    if not citations:
        return ""

    lines: list[str] = ["", "Sources:"]
    for c in citations:
        if c.total_chunks > 0:
            chunk_str = f"Chunk {c.chunk_index + 1}/{c.total_chunks}"
        else:
            chunk_str = f"Page {c.page_number}" if c.page_number > 0 else ""

        parts = [
            c.display_name,
            c.department,
            f"v{c.version}",
            f"Page {c.page_number}" if c.page_number > 0 else "",
            chunk_str,
        ]
        line = f"{c.rank}. " + " | ".join(p for p in parts if p)
        lines.append(line)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# RAGPipelineResponse
# ---------------------------------------------------------------------------

class RAGPipelineResponse(BaseModel):
    """
    The complete, structured output of a single RAGPipeline.run() call.

    Primary consumer contract — this is the object that:
        - Phase 9 tests validate
        - Phase 10 FastAPI serializes to JSON (via .model_dump_json())
        - Phase 11 LangGraph agents pass through their state graph

    Field groups:
        answer          — core LLM output in three forms
        citations       — deduplicated, ranked, formatted source references
        query meta      — echoed input fields
        timing          — per-phase latency measurements
        confidence      — retrieval-signal-based confidence estimate
        provenance      — model / template / retrieval mode labels
    """

    # --- Answer (three forms) ---
    answer: str = Field(
        ...,
        description="Raw LLM-generated answer, exactly as returned by Qwen2.5.",
    )
    answer_with_refs: str = Field(
        ...,
        description=(
            "Answer with [SOURCE N] placeholders replaced by [N] inline refs "
            "after citation deduplication and ranking."
        ),
    )
    formatted_answer: str = Field(
        ...,
        description=(
            "answer_with_refs followed by a formatted 'Sources:' block. "
            "This is the field shown to end users."
        ),
    )

    # --- Citations ---
    citations: list[Citation] = Field(
        default_factory=list,
        description="Deduplicated, ranked source citations from CitationEngine.",
    )
    retrieved_documents: int = Field(
        default=0,
        description="Number of unique documents in the retrieval result set.",
    )
    retrieved_chunks: int = Field(
        default=0,
        description="Total retrieved chunk count (before deduplication).",
    )

    # --- Query echo ---
    query: str = Field(..., description="Original user question.")
    role: str = Field(default="Public", description="RBAC role used for this query.")

    # --- Timing (all in milliseconds) ---
    processing_time_ms: float = Field(
        default=0.0,
        description="Wall-clock time for the complete pipeline (retrieval + LLM + citation).",
    )
    retrieval_time_ms: float = Field(
        default=0.0,
        description=(
            "Time for the retrieval phase (dense + BM25 + RRF + reranker). "
            "Reported by Retriever.latency_ms."
        ),
    )
    generation_time_ms: float = Field(
        default=0.0,
        description="Model eval time reported by Ollama (eval_duration / 1_000_000).",
    )
    total_tokens: int = Field(
        default=0,
        description="prompt_tokens + completion_tokens from Ollama.",
    )

    # --- Confidence ---
    confidence: str = Field(
        default="0%",
        description="Retrieval-signal-based confidence percentage: e.g., '87%'.",
    )
    confidence_score: float = Field(
        default=0.0,
        description="compute_confidence()'s blended dense-cosine score: 0.7*top1 + 0.3*mean(top3).",
    )

    # --- Provenance ---
    retrieval_mode: str = Field(
        default="unknown",
        description="'dense' | 'hybrid' | 'dense+rerank' | 'hybrid+rerank'",
    )
    rerank_method: Optional[str] = Field(
        default=None,
        description="'groq_listwise' if reranking actually happened; None otherwise.",
    )
    citations_inferred: bool = Field(
        default=False,
        description=(
            "True when the model emitted no valid [SOURCE N] markers and "
            "the citations shown are inferred (top-ranked chunks actually "
            "in the prompt) rather than read off the answer text. The UI "
            "should label these as 'related sources', not as cited."
        ),
    )
    model_name: str = Field(
        default="",
        description="Name of the model that generated the answer (see rag/rag_engine.py for the active backend).",
    )
    template_used: str = Field(
        default="default",
        description="PromptTemplate variant used.",
    )
    has_conflicts: bool = Field(
        default=False,
        description="True when version conflicts were detected in the retrieved context.",
    )
    chunks_in_context: int = Field(
        default=0,
        description="Number of chunks included in the LLM context window.",
    )

    # --- Metadata ---
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="UTC ISO-8601 timestamp of when this response was generated.",
    )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def is_fallback(self) -> bool:
        """
        True when the answer is the standard insufficient-evidence fallback,
        from any of the three paths that can produce it (no results, below
        the relevance gate, or the model reading real context and still
        finding nothing) -- they all now emit the same FALLBACK_ANSWER text,
        so one prefix check against the actual constant covers all three.
        """
        return self.answer.startswith(FALLBACK_ANSWER[:40])

    def short_summary(self) -> str:
        """One-line summary for logging / CLI output."""
        return (
            f"query={self.query[:50]!r} | "
            f"role={self.role} | "
            f"docs={self.retrieved_documents} | "
            f"confidence={self.confidence} | "
            f"tokens={self.total_tokens} | "
            f"latency={self.processing_time_ms:.0f}ms"
        )

    def to_display(self) -> str:
        """
        Human-readable multi-line summary for CLI / demo scripts.
        Suitable for verify_rag.py and demo.py output.
        """
        sep = "-" * 60
        lines = [
            sep,
            f"Query      : {self.query}",
            f"Role       : {self.role}",
            f"Mode       : {self.retrieval_mode}",
            f"Chunks     : {self.chunks_in_context} in context | "
            f"{self.retrieved_chunks} retrieved",
            f"Confidence : {self.confidence} (score={self.confidence_score:.4f})",
            f"Tokens     : {self.total_tokens}",
            f"Latency    : retrieval={self.retrieval_time_ms:.0f}ms  "
            f"generation={self.generation_time_ms:.0f}ms  "
            f"total={self.processing_time_ms:.0f}ms",
            sep,
            "Answer:",
            self.formatted_answer,
            sep,
        ]
        return "\n".join(lines)
