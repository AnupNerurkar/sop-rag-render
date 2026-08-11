"""
agents/multi_agent_graph.py
----------------------------
Production multi-agent LangGraph workflow built on top of the existing RAG
stack (hybrid retrieval + RRF + BGE reranker + citation engine + Ollama).

This does NOT replace or rewrite the underlying components — every node calls
the same building blocks the linear `rag_pipeline` uses. It orchestrates them
into an agentic graph with planning, validation, confidence evaluation and a
self-correcting reflection loop.

Graph:

    START
      │
      ▼
    query_analyzer ──(greeting/out_of_scope)──► final_response ──► END
      │ (rag)
      ▼
    planner
      │
      ▼
    retrieval_agent ◄───────────────┐
      │                             │
      ▼                             │
    context_validator               │ (reflect: broaden & retry)
      │                             │
      ▼                             │
    response_generator              │
      │                             │
      ▼                             │
    citation_formatter              │
      │                             │
      ▼                             │
    confidence_evaluator ──(reflect)──► reflection_agent ─┘
      │ (finalize)
      ▼
    final_response ──► END

Public API:
    run_multi_agent(question, role, memory) -> dict     (backward-compatible)
    get_multi_agent()                                   compiled graph singleton
"""

from __future__ import annotations

import logging
import os
import re
import sys
import time
from typing import Any, Optional

from agents.agent_state import (
    DEFAULT_AGENT_CONFIG,
    FALLBACK_ANSWER,
    AgentConfig,
    AgentState,
)

logger = logging.getLogger(__name__)

# Ensure project root on path so reused modules import cleanly under uvicorn.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

CFG = DEFAULT_AGENT_CONFIG


# ===========================================================================
# Lightweight NLU heuristics (no LLM call — keeps the control plane fast)
# ===========================================================================

_GREETINGS = frozenset({
    "hi", "hello", "hey", "yo", "hiya", "howdy", "greetings",
    "good morning", "good afternoon", "good evening", "thanks", "thank you",
})

_OUT_OF_SCOPE = frozenset({
    "weather", "cricket", "football", "movie", "recipe", "joke", "song",
    "stock price", "forex", "bitcoin", "horoscope", "lottery",
})

# Follow-up signals — short queries or anaphora that only make sense with context.
_FOLLOWUP_MARKERS = frozenset({
    "it", "that", "this", "those", "they", "them", "more", "also",
    "what about", "and the", "how about", "tell me more", "elaborate",
})

# Category keyword → canonical category value used in document metadata.
_CATEGORY_KEYWORDS = {
    "circular":  "Circular",
    "policy":    "Policy",
    "policies":  "Policy",
    "sop":       "SOP",
    "procedure": "SOP",
    "guideline": "SOP",
    "guidelines": "SOP",
}

# Department keyword → canonical department hint.
_DEPARTMENT_KEYWORDS = {
    "exam":        "Examination",
    "examination": "Examination",
    "admission":   "Admissions",
    "admissions":  "Admissions",
    "research":    "Research",
    "grant":       "Research",
    "scholarship": "Finance",
    "fee":         "Finance",
    "fees":        "Finance",
    "hostel":      "Hostel",
}

_VERSION_RE = re.compile(r"\bv(?:ersion)?\s*\.?\s*([0-9]+(?:\.[0-9]+)?)\b", re.I)

# Phrases stripped when broadening/rewriting a query.
_STRIP_PHRASES = (
    "can you explain", "could you explain", "please explain", "tell me about",
    "i want to know", "what is the", "what are the", "what is", "what are",
    "how do i", "how to", "how can i", "explain", "describe", "summarise",
    "summarize",
)


def _last_user_topic(memory: list[dict]) -> str:
    """Returns the most recent user turn from memory, for follow-up resolution."""
    for turn in reversed(memory or []):
        if turn.get("role") == "user" and turn.get("content"):
            return turn["content"]
    return ""


def _strip_question_phrases(text: str) -> str:
    q = text.lower()
    for phrase in _STRIP_PHRASES:
        q = q.replace(phrase, " ")
    return re.sub(r"\s+", " ", q).strip(" ?.,!")


# ===========================================================================
# NODE 1 — Query Analyzer
# ===========================================================================

def node_query_analyzer(state: AgentState) -> dict:
    """
    Understands the query: intent, follow-up resolution, metadata filters and
    ambiguity. Greetings / out-of-scope queries are short-circuited here with a
    canned answer so the retrieval stack is never touched needlessly.
    """
    question = (state.get("question") or "").strip()
    memory = state.get("memory") or []
    q = question.lower().strip(" ?.!")
    words = q.split()

    # --- Intent ---
    if q in _GREETINGS or (len(words) <= 3 and any(g in q for g in _GREETINGS)):
        logger.info("[MAGENT] analyzer: intent=greeting q=%r", question[:60])
        return {
            "intent": "greeting",
            "effective_query": question,
            "analysis_note": "Greeting detected — answered directly.",
            "answer": (
                "Hello! I'm your institutional knowledge assistant. Ask me "
                "anything about institutional policies, procedures, academics, "
                "examinations or administration."
            ),
            "source_documents": [], "citations": [], "answer_with_refs": "",
            "confidence": "HIGH", "confidence_score": 1.0,
            "hallucination_risk": "low", "insufficient_context": False,
        }

    if any(kw in q for kw in _OUT_OF_SCOPE):
        logger.info("[MAGENT] analyzer: intent=out_of_scope q=%r", question[:60])
        return {
            "intent": "out_of_scope",
            "effective_query": question,
            "analysis_note": "Out-of-scope topic detected.",
            "answer": (
                "I'm specialised in institutional knowledge. Please ask about "
                "academics, policies, examinations, admissions, research or "
                "administrative procedures."
            ),
            "source_documents": [], "citations": [], "answer_with_refs": "",
            "confidence": "HIGH", "confidence_score": 1.0,
            "hallucination_risk": "low", "insufficient_context": False,
        }

    # --- Follow-up resolution ---
    is_followup = bool(memory) and (
        len(words) <= 4 or any(m in q for m in _FOLLOWUP_MARKERS)
    )
    effective_query = question
    if is_followup:
        topic = _last_user_topic(memory)
        if topic:
            effective_query = f"{topic.rstrip(' ?.!')} — {question}"

    # --- Filter detection ---
    filters: dict = {}
    for kw, cat in _CATEGORY_KEYWORDS.items():
        if kw in q:
            filters["category"] = cat
            break
    for kw, dept in _DEPARTMENT_KEYWORDS.items():
        if kw in q:
            filters["department"] = dept
            break
    m = _VERSION_RE.search(question)
    if m:
        filters["version"] = m.group(1)

    # --- Ambiguity ---
    ambiguous = (len(words) <= 2 and not is_followup) or (
        any(marker in words for marker in ("it", "that", "this")) and not memory
    )

    note_parts = []
    if is_followup:
        note_parts.append("follow-up resolved from history")
    if filters:
        note_parts.append(f"filters={filters}")
    if ambiguous:
        note_parts.append("query is ambiguous — proceeding with best effort")
    note = "; ".join(note_parts) or "standalone factual query"

    logger.info(
        "[MAGENT] analyzer: intent=rag followup=%s ambiguous=%s filters=%s",
        is_followup, ambiguous, filters,
    )
    return {
        "intent": "rag",
        "is_followup": is_followup,
        "ambiguous": ambiguous,
        "filters": filters,
        "effective_query": effective_query,
        "analysis_note": note,
        "retry_count": state.get("retry_count", 0),
    }


# ===========================================================================
# NODE 2 — Planner
# ===========================================================================

_SPLIT_RE = re.compile(r"\s+and\s+|\s*;\s*|\s*\?\s*", re.I)


def node_planner(state: AgentState) -> dict:
    """
    Decomposes complex / multi-part questions into an ordered list of retrieval
    sub-queries. Simple questions yield a single-step plan.
    """
    query = state.get("effective_query") or state.get("question", "")
    lowered = query.lower()

    is_complex = bool(
        re.search(r"\band\b|\bcompare\b|\bdifference between\b|\bboth\b|;", lowered)
    )

    plan: list[str] = []
    if is_complex:
        parts = [p.strip(" ?.,") for p in _SPLIT_RE.split(query)]
        plan = [p for p in parts if len(p.split()) >= 3]

    if not plan:
        plan = [query]
        is_complex = False

    plan = plan[: CFG.max_plan_steps]
    note = (
        f"decomposed into {len(plan)} retrieval steps"
        if is_complex else "single-step plan"
    )
    logger.info("[MAGENT] planner: is_complex=%s steps=%d", is_complex, len(plan))
    return {"plan": plan, "is_complex": is_complex, "plan_note": note}


# ===========================================================================
# NODE 3 — Retrieval Agent
# ===========================================================================

def node_retrieval_agent(state: AgentState) -> dict:
    """
    Executes hybrid (dense + BM25 + RRF + reranker) retrieval for every step in
    the plan, applying any detected metadata filters. Accumulates results across
    steps. Re-entrant: the reflection loop re-runs this node with a broadened
    plan / dropped filters and a bumped iteration counter.
    """
    plan = state.get("plan") or [state.get("effective_query") or state.get("question", "")]
    role = state.get("role", "Public")
    filters = state.get("filters") or {}
    iteration = state.get("retrieval_iterations", 0) + 1

    t0 = time.perf_counter()
    collected: list[Any] = []
    mode = "hybrid+rerank"

    try:
        from retrieval.retriever import get_retriever
        retriever = get_retriever()
        for step in plan:
            if not step or len(step.strip()) < 2:
                continue
            resp = retriever.retrieve_by_text(
                text=step,
                role=role,
                top_k=CFG.top_k_per_step,
                department=filters.get("department"),
                category=filters.get("category"),
                version=filters.get("version"),
                use_bm25=True,
                use_reranker=True,
            )
            mode = resp.retrieval_mode or mode
            collected.extend(resp.results)
            logger.info(
                "[MAGENT] retrieval iter=%d step=%r results=%d filters=%s",
                iteration, step[:50], len(resp.results), filters,
            )
    except Exception as exc:
        logger.error("[MAGENT] retrieval_agent failed: %s", exc, exc_info=True)
        return {
            "raw_results": [],
            "retrieval_mode": "error",
            "retrieval_latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            "retrieval_iterations": iteration,
            "error": f"retrieval: {exc}",
        }

    latency = round((time.perf_counter() - t0) * 1000, 1)
    logger.info(
        "[MAGENT] retrieval_agent: iter=%d total_raw=%d mode=%s %.0fms",
        iteration, len(collected), mode, latency,
    )
    return {
        "raw_results": collected,
        "retrieval_mode": mode,
        "retrieval_latency_ms": latency,
        "retrieval_iterations": iteration,
    }


# ===========================================================================
# NODE 4 — Context Validator
# ===========================================================================

def node_context_validator(state: AgentState) -> dict:
    """
    Cleans the retrieved context before generation:
      - de-duplicates chunks (by chunk_id, then by normalised content)
      - drops irrelevant chunks below the relevance floor (keeps >=1)
      - validates citation quality (must have a usable display_name)
      - re-ranks by score and caps the context size
    """
    raw = state.get("raw_results") or []
    if not raw:
        return {
            "validated_results": [],
            "dropped_count": 0,
            "validation_note": "no context retrieved",
        }

    seen_ids: set[str] = set()
    seen_text: set[str] = set()
    deduped: list[Any] = []
    for r in raw:
        cid = getattr(r, "chunk_id", None)
        text_key = (getattr(r, "content", "") or "")[:160].strip().lower()
        if cid and cid in seen_ids:
            continue
        if text_key and text_key in seen_text:
            continue
        # Citation quality: a usable, named source is required.
        citation = getattr(r, "citation", None)
        if citation is None or not getattr(citation, "display_name", "").strip():
            continue
        if cid:
            seen_ids.add(cid)
        if text_key:
            seen_text.add(text_key)
        deduped.append(r)

    # Relevance filter — but never drop everything.
    above = [r for r in deduped if getattr(r, "score", 0.0) >= CFG.min_relevance_score]
    kept = above if above else deduped

    kept.sort(key=lambda r: getattr(r, "score", 0.0), reverse=True)
    kept = kept[: CFG.max_context_chunks]

    dropped = len(raw) - len(kept)
    note = (
        f"kept {len(kept)}/{len(raw)} chunks "
        f"(deduped + relevance>={CFG.min_relevance_score})"
    )
    logger.info("[MAGENT] context_validator: %s", note)
    return {
        "validated_results": kept,
        "dropped_count": max(dropped, 0),
        "validation_note": note,
    }


# ===========================================================================
# NODE 5 — Response Generator
# ===========================================================================

def node_response_generator(state: AgentState) -> dict:
    """
    Generates a grounded answer from the validated context only. Reuses the
    exact prompt builder + Ollama engine that the linear pipeline uses, so the
    grounding/citation contract is identical. No context → deterministic
    fallback (no LLM call, no hallucination).
    """
    results = state.get("validated_results") or []
    query = state.get("effective_query") or state.get("question", "")

    if not results:
        logger.info("[MAGENT] response_generator: no context → fallback")
        return {
            "answer": FALLBACK_ANSWER,
            "generation_time_ms": 0.0,
            "model_name": "",
        }

    try:
        from rag_pipeline import PipelineConfig
        from rag.prompt_builder import build_prompt
        from rag.rag_engine import get_rag_engine

        cfg = PipelineConfig()
        built = build_prompt(query, results, cfg.to_prompt_config())
        resp = get_rag_engine().generate(
            built,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            top_p=cfg.top_p,
            repeat_penalty=cfg.repeat_penalty,
        )
        logger.info(
            "[MAGENT] response_generator: answer_len=%d %.0fms model=%s",
            len(resp.answer), getattr(resp, "generation_time_ms", 0.0),
            getattr(resp, "model_name", "?"),
        )
        return {
            "answer": resp.answer or FALLBACK_ANSWER,
            "generation_time_ms": round(getattr(resp, "generation_time_ms", 0.0), 1),
            "model_name": getattr(resp, "model_name", ""),
        }
    except Exception as exc:
        logger.error("[MAGENT] response_generator failed: %s", exc, exc_info=True)
        return {
            "answer": "I encountered an error while generating the answer. Please try again.",
            "generation_time_ms": 0.0,
            "model_name": "",
            "error": f"generation: {exc}",
        }


# ===========================================================================
# NODE 6 — Citation Formatter
# ===========================================================================

def node_citation_formatter(state: AgentState) -> dict:
    """
    Builds citations from the validated context using the existing citation
    engine — single source of truth for citation rendering across the project.
    """
    results = state.get("validated_results") or []
    answer = state.get("answer", "")

    if not results:
        return {"citations": [], "source_documents": [], "answer_with_refs": answer}

    try:
        from rag.citation_engine import get_citation_engine
        citation_list = get_citation_engine().build(results, answer)
        citations = [
            {
                "doc_id":       c.doc_id,
                "display_name": c.display_name,
                "department":   c.department,
                "version":      c.version,
                "access_level": c.access_level,
                "score":        round(c.score, 4),
                "page_number":  c.page_number,
                "chunk_index":  c.chunk_index,
                "total_chunks": c.total_chunks,
                "source_file":  c.source_file,
            }
            for c in citation_list.citations
        ]
        source_documents = [c.display_name for c in citation_list.citations]
        logger.info("[MAGENT] citation_formatter: %d citations", len(citations))
        return {
            "citations": citations,
            "source_documents": source_documents,
            "answer_with_refs": citation_list.answer_with_refs,
        }
    except Exception as exc:
        logger.error("[MAGENT] citation_formatter failed: %s", exc, exc_info=True)
        # Degrade gracefully — answer still returns, just without rich citations.
        return {
            "citations": [],
            "source_documents": [
                getattr(getattr(r, "citation", None), "display_name", "")
                for r in results
            ],
            "answer_with_refs": answer,
        }


# ===========================================================================
# NODE 7 — Confidence Evaluator
# ===========================================================================

def node_confidence_evaluator(state: AgentState) -> dict:
    """
    Scores answer trustworthiness from retrieval signals and flags hallucination
    risk / insufficient context. Sets the `control` key that routes the graph to
    either reflection (retry) or finalisation.
    """
    results = state.get("validated_results") or []
    answer = state.get("answer", "") or ""
    retry_count = state.get("retry_count", 0)

    try:
        from response_schema import compute_confidence
        pct_str, raw_score = compute_confidence(results)
        confidence = pct_str
        confidence_score = round(float(raw_score), 4)
    except Exception as exc:
        logger.warning("[MAGENT] compute_confidence failed: %s", exc)
        confidence, confidence_score = "0%", 0.0

    # The underlying logic for "LOW" or "UNKNOWN" confidence was:
    # effective_score < 0.40 and n_unique_docs < 2, or no results.
    n_unique_docs = len({r.citation.doc_id for r in results}) if results else 0
    is_low_or_unknown = (confidence_score < 0.40 and n_unique_docs < 2) or not results

    top_score = max((getattr(r, "score", 0.0) for r in results), default=0.0)
    is_fallback = answer.strip().startswith(FALLBACK_ANSWER[:40])
    insufficient_context = (not results) or (top_score < CFG.insufficient_top_score)

    if insufficient_context and not is_fallback:
        hallucination_risk = "high"
    elif is_low_or_unknown:
        hallucination_risk = "medium"
    else:
        hallucination_risk = "low"

    # Decide whether to reflect (retry) or finalise.
    needs_retry = (
        (is_low_or_unknown or insufficient_context or hallucination_risk == "high")
        and not is_fallback
        and confidence_score < CFG.reflection_confidence_floor
        and retry_count < CFG.max_reflection_retries
    )
    control = "reflect" if needs_retry else "finalize"

    logger.info(
        "[MAGENT] confidence_evaluator: conf=%s score=%.3f top=%.3f "
        "halluc=%s insufficient=%s → %s",
        confidence, confidence_score, top_score, hallucination_risk,
        insufficient_context, control,
    )
    return {
        "confidence": confidence,
        "confidence_score": confidence_score,
        "hallucination_risk": hallucination_risk,
        "insufficient_context": insufficient_context,
        "control": control,
    }


# ===========================================================================
# NODE 8 — Reflection Agent
# ===========================================================================

def node_reflection_agent(state: AgentState) -> dict:
    """
    Self-correction step. Triggered only when confidence is below threshold and
    retries remain. Broadens the search by dropping (possibly over-narrow)
    metadata filters and simplifying the query into a keyword-focused form, then
    the graph loops back to the retrieval agent for another pass.
    """
    original = state.get("question", "")
    broadened = _strip_question_phrases(original) or original
    retry_count = state.get("retry_count", 0) + 1

    note = (
        f"reflection #{retry_count}: confidence "
        f"{state.get('confidence')} ({state.get('confidence_score')}) too low — "
        f"broadening query to {broadened!r} and dropping filters"
    )
    logger.info("[MAGENT] reflection_agent: %s", note)
    return {
        # Broaden: keyword query, no filters, fresh single-step plan.
        "effective_query": broadened,
        "filters": {},
        "plan": [broadened],
        "retry_count": retry_count,
        "reflection_note": note,
    }


# ===========================================================================
# NODE 9 — Final Response
# ===========================================================================

def node_final_response(state: AgentState) -> dict:
    """
    Assembles the final turn: updates conversation memory and stamps total
    processing time. The answer/citations/confidence are already in state.
    """
    memory = list(state.get("memory") or [])
    memory.append({"role": "user", "content": state.get("question", "")})
    memory.append({"role": "assistant", "content": state.get("answer", "")})
    if len(memory) > CFG.memory_max_turns:
        memory = memory[-CFG.memory_max_turns:]

    note = state.get("reflection_note", "")
    if not note:
        if state.get("insufficient_context"):
            note = "Limited supporting context — answer may be incomplete."
        elif str(state.get("confidence", "")).upper() in ("LOW", "UNKNOWN"):
            note = "Low confidence — consider rephrasing with more specific terms."

    logger.info(
        "[MAGENT] final_response: intent=%s conf=%s sources=%d iters=%s",
        state.get("intent"), state.get("confidence"),
        len(state.get("source_documents") or []),
        state.get("retrieval_iterations"),
    )
    return {"memory": memory, "reflection_note": note, "finished": True}


# ===========================================================================
# Graph construction
# ===========================================================================

def _route_after_analyzer(state: AgentState) -> str:
    return "rag" if state.get("intent", "rag") == "rag" else "direct"


def _route_after_confidence(state: AgentState) -> str:
    return state.get("control", "finalize")


def _build_graph():
    """Builds and compiles the multi-agent StateGraph."""
    try:
        from langgraph.graph import StateGraph, END
    except ImportError as exc:  # pragma: no cover - environment guard
        raise ImportError("langgraph is required. Run: pip install langgraph>=0.2.0") from exc

    g = StateGraph(AgentState)

    g.add_node("query_analyzer",       node_query_analyzer)
    g.add_node("planner",              node_planner)
    g.add_node("retrieval_agent",      node_retrieval_agent)
    g.add_node("context_validator",    node_context_validator)
    g.add_node("response_generator",   node_response_generator)
    g.add_node("citation_formatter",   node_citation_formatter)
    g.add_node("confidence_evaluator", node_confidence_evaluator)
    g.add_node("reflection_agent",     node_reflection_agent)
    g.add_node("final_response",       node_final_response)

    g.set_entry_point("query_analyzer")

    # Greeting / out-of-scope short-circuit straight to the final node.
    g.add_conditional_edges(
        "query_analyzer",
        _route_after_analyzer,
        {"rag": "planner", "direct": "final_response"},
    )

    g.add_edge("planner",            "retrieval_agent")
    g.add_edge("retrieval_agent",    "context_validator")
    g.add_edge("context_validator",  "response_generator")
    g.add_edge("response_generator", "citation_formatter")
    g.add_edge("citation_formatter", "confidence_evaluator")

    # Confidence gate — reflect (retry) or finalise.
    g.add_conditional_edges(
        "confidence_evaluator",
        _route_after_confidence,
        {"reflect": "reflection_agent", "finalize": "final_response"},
    )

    # Reflection loops back into retrieval for another, broadened pass.
    g.add_edge("reflection_agent", "retrieval_agent")
    g.add_edge("final_response",   END)

    return g.compile()


# Lazy compiled singleton
_compiled_graph = None


def get_multi_agent():
    """Returns the compiled multi-agent graph (singleton)."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = _build_graph()
    return _compiled_graph


def reset_multi_agent() -> None:
    """Clears the compiled-graph singleton — for testing."""
    global _compiled_graph
    _compiled_graph = None


# ===========================================================================
# Public run helper (backward-compatible response shape)
# ===========================================================================

def run_multi_agent(question: str, role: str, memory: Optional[list] = None) -> dict:
    """
    Runs the multi-agent workflow for a single turn.

    Returns a dict that is a superset of the legacy `run_agent` response, so it
    is a drop-in replacement for the FastAPI `/api/agent/chat` endpoint.
    """
    t_start = time.perf_counter()
    initial_state: AgentState = {
        "question": question or "",
        "role": role or "Public",
        "memory": memory or [],
        "retry_count": 0,
    }

    try:
        final_state = get_multi_agent().invoke(initial_state)
    except Exception as exc:
        logger.error("[MAGENT] run_multi_agent failed: %s", exc, exc_info=True)
        return {
            "answer": "The agent encountered an error. Please try again.",
            "answer_with_refs": "",
            "source_documents": [], "citations": [],
            "confidence": "0%", "confidence_score": 0.0,
            "retrieval_mode": "agent_error",
            "processing_time_ms": round((time.perf_counter() - t_start) * 1000, 1),
            "reflection_note": str(exc),
            "memory": memory or [],
            "intent": "error", "hallucination_risk": "high",
            "insufficient_context": True, "retrieval_iterations": 0,
        }

    wall_ms = round((time.perf_counter() - t_start) * 1000, 1)
    return {
        # --- legacy/back-compat keys ---
        "answer":             final_state.get("answer", ""),
        "answer_with_refs":   final_state.get("answer_with_refs") or final_state.get("answer", ""),
        "source_documents":   final_state.get("source_documents", []),
        "citations":          final_state.get("citations", []),
        "confidence":         final_state.get("confidence", "UNKNOWN"),
        "confidence_score":   final_state.get("confidence_score", 0.0),
        "retrieval_mode":     final_state.get("retrieval_mode", "agent"),
        "processing_time_ms": final_state.get("processing_time_ms") or wall_ms,
        "reflection_note":    final_state.get("reflection_note", ""),
        "memory":             final_state.get("memory", []),
        # --- new agentic transparency keys (additive, backward-compatible) ---
        "intent":               final_state.get("intent", "rag"),
        "is_followup":          final_state.get("is_followup", False),
        "ambiguous":            final_state.get("ambiguous", False),
        "filters":              final_state.get("filters", {}),
        "plan":                 final_state.get("plan", []),
        "is_complex":           final_state.get("is_complex", False),
        "hallucination_risk":   final_state.get("hallucination_risk", "low"),
        "insufficient_context": final_state.get("insufficient_context", False),
        "retrieval_iterations": final_state.get("retrieval_iterations", 0),
        "generation_time_ms":   final_state.get("generation_time_ms", 0.0),
        "model_name":           final_state.get("model_name", ""),
    }
