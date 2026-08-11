"""
agents/agent_state.py
----------------------
Graph state model for the multi-agent LangGraph workflow.

This module is intentionally dependency-light: it only defines the typed state
that flows between nodes plus a few small, pure helper dataclasses. Keeping it
free of heavy imports (retriever, ollama, etc.) means the graph definition and
the unit tests can import the state model without spinning up models.

State lifecycle (which node writes which keys):

    query_analyzer      → intent, is_followup, filters, ambiguous,
                          analysis_note, effective_query
    planner             → plan, is_complex, plan_note
    retrieval_agent     → raw_results, retrieval_mode, retrieval_latency_ms,
                          retrieval_iterations
    context_validator   → validated_results, dropped_count, validation_note
    response_generator  → answer, generation_time_ms, model_name
    citation_formatter  → citations, source_documents, answer_with_refs
    confidence_evaluator→ confidence, confidence_score, hallucination_risk,
                          insufficient_context, control
    reflection_agent    → effective_query, filters, plan, retry_count,
                          reflection_note
    final_response      → processing_time_ms, memory, finished
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, TypedDict


# ---------------------------------------------------------------------------
# Tunable thresholds (single source of truth — imported by the graph + tests)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AgentConfig:
    """Thresholds and limits governing the agentic control flow."""

    # Reflection / retry loop
    max_reflection_retries: int = 1          # extra full retrieval+generate passes
    reflection_confidence_floor: float = 0.40  # below → reflect (if retries remain)

    # Retrieval
    top_k_per_step: int = 5                  # results fetched per plan step
    max_plan_steps: int = 3                  # cap complex-question decomposition

    # Context validation
    min_relevance_score: float = 0.15        # drop chunks below this (keep >=1)
    max_context_chunks: int = 6              # cap chunks handed to the generator

    # Confidence / hallucination heuristics
    insufficient_top_score: float = 0.30     # top score below → insufficient context
    memory_max_turns: int = 20               # trim conversation memory


DEFAULT_AGENT_CONFIG = AgentConfig()

# Kept byte-identical to response_schema.FALLBACK_ANSWER (not imported --
# this module is deliberately dependency-light, see the module docstring)
# so RAGPipelineResponse.is_fallback recognizes an agent-path fallback the
# same way it recognizes one from the main pipeline. Previously this was a
# different, shorter string, so an agent response that correctly declined
# to answer would still fail that check.
FALLBACK_ANSWER = (
    "I could not find sufficient institutional evidence to answer this question. "
    "The knowledge base may not contain information on this topic, or access may "
    "be restricted for your role."
)


# ---------------------------------------------------------------------------
# Small value objects (pure data, easy to assert on in tests)
# ---------------------------------------------------------------------------

@dataclass
class QueryAnalysis:
    """Structured output of the Query Analyzer node."""
    intent: str = "rag"                      # rag | greeting | out_of_scope
    is_followup: bool = False
    ambiguous: bool = False
    filters: dict = field(default_factory=dict)   # {department?, category?, version?}
    effective_query: str = ""
    note: str = ""


# ---------------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------------

class AgentState(TypedDict, total=False):
    """
    The complete state object passed between graph nodes.

    `total=False` means every key is optional — nodes contribute the keys they
    own and LangGraph merges (last-write-wins) them into the running state.
    """

    # ---- Inputs ----------------------------------------------------------
    question:            str
    role:                str
    memory:              list[dict]          # [{"role": "...", "content": "..."}]

    # ---- Query Analyzer --------------------------------------------------
    intent:              str                 # rag | greeting | out_of_scope
    is_followup:         bool
    ambiguous:           bool
    filters:             dict
    analysis_note:       str
    effective_query:     str

    # ---- Planner ---------------------------------------------------------
    plan:                list[str]
    is_complex:          bool
    plan_note:           str

    # ---- Retrieval Agent -------------------------------------------------
    raw_results:         list[Any]           # list[RetrievalResult]
    retrieval_mode:      str
    retrieval_latency_ms: float
    retrieval_iterations: int

    # ---- Context Validator ----------------------------------------------
    validated_results:   list[Any]           # list[RetrievalResult]
    dropped_count:       int
    validation_note:     str

    # ---- Response Generator ---------------------------------------------
    answer:              str
    generation_time_ms:  float
    model_name:          str

    # ---- Citation Formatter ---------------------------------------------
    citations:           list[dict]
    source_documents:    list[str]
    answer_with_refs:    str

    # ---- Confidence Evaluator -------------------------------------------
    confidence:          str                 # HIGH | MEDIUM | LOW | UNKNOWN
    confidence_score:    float
    hallucination_risk:  str                 # low | medium | high
    insufficient_context: bool

    # ---- Reflection ------------------------------------------------------
    retry_count:         int
    reflection_note:     str

    # ---- Control / final -------------------------------------------------
    control:             str                 # "reflect" | "finalize" (router key)
    processing_time_ms:  float
    error:               str
    finished:            bool
