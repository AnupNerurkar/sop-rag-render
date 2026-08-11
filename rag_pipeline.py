"""
rag_pipeline.py
----------------
Phase 9: Integrated RAG Pipeline.

Wires together every completed phase into a single coherent call:

    User query + role
        |
        v  Retriever (dense + SQLite FTS5 + RRF + Groq listwise rerank)
    list[RetrievalResult]
        |
        v  PromptBuilder (system prompt + context block + question)
    BuiltPrompt
        |
        v  RAGEngine  ->  Groq (LLM_BACKEND selects the active backend;
                           Groq is what actually runs in production)
    RAGResponse  (answer + token counts + timing)
        |
        v  CitationEngine (dedup, version-prefer, inline refs)
    CitationList
        |
        v  response_schema.py assembler
    RAGPipelineResponse

Design choices
--------------
* Lazy sub-system resolution: get_retriever() / get_rag_engine() / etc. are
  called inside run(), not __init__().  Construction is O(1); the heavy
  models load on the first actual query call.  This also lets tests patch
  the singleton factories cleanly.

* Stateless per-call: RAGPipeline holds no call-level state between run()
  invocations.  Multiple concurrent callers are safe.

* Relevance gate: retrieval results below RELEVANCE_FLOOR (see
  response_schema.passes_relevance_gate) are treated the same as zero
  results -- skip the LLM entirely, return the standard fallback with no
  citations. This is what actually prevents a confident-sounding answer
  being generated off irrelevant context; a zero-results check alone
  only catches the case where nothing was retrieved at all.

* Role normalization: any case variant ("student", "STUDENT") is silently
  normalised to title-case before hitting the RBAC filter.

* PipelineConfig immutability: config is validated once at construction.
  Per-call overrides are passed as a separate PipelineConfig instance;
  the pipeline's own config is never mutated.

* Streaming: run_stream() yields LLM text tokens as they arrive.
  It does NOT run the citation engine (which needs the complete answer).
  For callers that need citations, use run() instead.

LangGraph note:
  PipelineConfig, RAGPipeline, and RAGPipelineResponse are all designed
  to be usable as LangGraph state fields and node callables with zero
  modification.
"""

from __future__ import annotations

import logging
import time
from typing import Iterator, Optional

from pydantic import BaseModel, Field

# Pure data-model imports -- no I/O, no model loading, no circular risk.
from rag.prompt_schema import PromptConfig, PromptTemplate
from response_schema import (
    FALLBACK_ANSWER,
    RAGPipelineResponse,
    compute_confidence,
    passes_relevance_gate,
    format_sources_block,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Role normalisation
# ---------------------------------------------------------------------------

_VALID_ROLES = frozenset({"Admin", "Faculty", "Student", "Public"})


def _normalize_role(role: str) -> str:
    """Title-cases the role string and falls back to 'Public' if unrecognised."""
    normalized = role.strip().title()
    return normalized if normalized in _VALID_ROLES else "Public"


# ---------------------------------------------------------------------------
# PipelineConfig
# ---------------------------------------------------------------------------

class PipelineConfig(BaseModel):
    """
    All tunable knobs for a single RAGPipeline instance.

    Group 1 -- Retrieval
        Controls how many candidates are fetched and which retrieval modes run.
    Group 2 -- Prompt
        Controls context window size, template variant, and metadata display.
    Group 3 -- Generation
        Controls sampling parameters for whatever LLM_BACKEND selects
        (Groq in production; see rag/rag_engine.py).
    """

    # ---- Retrieval ---------------------------------------------------------
    top_k_dense: int = Field(
        default=25, ge=1, le=100,
        description="Dense candidates fetched from the vector store.",
    )
    top_k_bm25: int = Field(
        default=25, ge=1, le=100,
        description="Keyword (SQLite FTS5) candidates fetched.",
    )
    top_k_fusion: int = Field(
        default=20, ge=1, le=100,
        description="Candidates entering the reranker after RRF.",
    )
    top_k_final: int = Field(
        default=5, ge=1, le=50,
        description="Final results returned after reranking.",
    )
    use_bm25: bool = Field(
        default=True,
        description="Enable BM25 keyword retrieval alongside dense retrieval.",
    )
    use_reranker: bool = Field(
        default=True,
        description=(
            "Rerank fused candidates via one extra Groq listwise call. "
            "Was False when this meant a local cross-encoder that was never "
            "actually installed (see retrieval/reranker.py) -- every query "
            "silently returned unreranked results while claiming otherwise. "
            "Now real: measured nDCG@5 0.681->0.728 and Hit@3 0.95->1.00 on "
            "the eval corpus for ~80ms added latency (eval-results/"
            "phase3-retrieval.json vs baseline-retrieval.json)."
        ),
    )

    # ---- Prompt ------------------------------------------------------------
    prompt_template: PromptTemplate = Field(
        default=PromptTemplate.DEFAULT,
        description="System-prompt template variant.",
    )
    max_chunks: int = Field(
        default=6, ge=1, le=20,
        description="Maximum context chunks passed to the LLM.",
    )
    max_context_chars: int = Field(
        default=12000, ge=500, le=32000,
        description="Hard cap on total context block characters.",
    )
    include_metadata: bool = Field(
        default=True,
        description="Include dept/category/confidence metadata row per source chunk.",
    )
    confidence_threshold: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Drop chunks below this effective score before building the prompt.",
    )

    # ---- Generation --------------------------------------------------------
    temperature: float = Field(
        default=0.2, ge=0.0, le=2.0,
        description=(
            "Sampling temperature. Lowered from 0.7 -- grounded citation "
            "work (answer only from supplied context, cite every claim) "
            "wants determinism, not creative variance; 0.7 is a "
            "conversational-assistant default that doesn't fit this task."
        ),
    )
    max_tokens: int = Field(
        default=1500, ge=1, le=32768,
        description=(
            "Maximum completion tokens. Raised from 512, which could cut "
            "off a multi-step SOP procedure mid-answer with no indication "
            "to the user that the answer was truncated rather than complete."
        ),
    )
    top_p: float = Field(
        default=0.9, ge=0.0, le=1.0,
    )
    repeat_penalty: float = Field(
        default=1.1, ge=0.0, le=2.0,
    )

    def to_prompt_config(self) -> PromptConfig:
        """Converts retrieval+prompt fields to a PromptConfig for the prompt builder."""
        return PromptConfig(
            template             = self.prompt_template,
            max_chunks           = self.max_chunks,
            max_context_chars    = self.max_context_chars,
            include_metadata     = self.include_metadata,
            confidence_threshold = self.confidence_threshold,
        )


# ---------------------------------------------------------------------------
# RAGPipeline
# ---------------------------------------------------------------------------

class RAGPipeline:
    """
    Orchestrates the full Retrieval-Augmented Generation pipeline.

    Usage
    -----
        pipeline = get_pipeline()
        response = pipeline.run("What is the attendance policy?", role="Student")
        print(response.formatted_answer)

    Streaming (text tokens only, no citations):
        for token in pipeline.run_stream("Summarise fee policy", role="Faculty"):
            print(token, end="", flush=True)
    """

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self._config = config or PipelineConfig()

    # ------------------------------------------------------------------
    # Primary API -- non-streaming
    # ------------------------------------------------------------------

    def run(
        self,
        question: str,
        role: str = "Public",
        *,
        config_overrides: Optional[PipelineConfig] = None,
    ) -> RAGPipelineResponse:
        """
        Execute the full RAG pipeline and return a structured response.

        Args:
            question:         The user's natural-language question.
            role:             RBAC role: 'Admin' | 'Faculty' | 'Student' | 'Public'.
                              Case-insensitive; defaults to 'Public' if unrecognised.
            config_overrides: Optional per-call PipelineConfig that overrides the
                              instance config for this call only.

        Returns:
            RAGPipelineResponse with answer, citations, timing, and confidence.

        Notes:
            - If the retriever finds no results, the LLM is skipped entirely.
            - Role-filtered documents are never exposed, even in error messages.
        """
        cfg          = config_overrides or self._config
        role         = _normalize_role(role)
        t_wall_start = time.perf_counter()

        # ---- Phase 1: Retrieval ----------------------------------------
        retrieval_response = self._retrieve(question, role, cfg)
        retrieval_time_ms  = retrieval_response.latency_ms
        results            = retrieval_response.results

        # ---- Short-circuit: no results, or results too weak to trust ----
        # Both cases get identical treatment: skip the LLM, zero citations,
        # FALLBACK_ANSWER. The relevance gate is what actually stops a
        # confident-sounding answer from being generated off irrelevant
        # context -- an empty-results check alone doesn't catch "we found
        # four chunks, none of them are about this question."
        if not results or not passes_relevance_gate(results):
            if not results:
                logger.info(
                    "[PIPELINE] No results for query=%r role=%s -- returning fallback.",
                    question[:80], role,
                )
            else:
                logger.info(
                    "[PIPELINE] Relevance gate failed for query=%r role=%s "
                    "(top score below RELEVANCE_FLOOR) -- returning fallback.",
                    question[:80], role,
                )
            wall_ms = (time.perf_counter() - t_wall_start) * 1000
            return RAGPipelineResponse(
                answer               = FALLBACK_ANSWER,
                answer_with_refs     = FALLBACK_ANSWER,
                formatted_answer     = FALLBACK_ANSWER,
                citations            = [],
                retrieved_documents  = 0,
                retrieved_chunks     = len(results),
                query                = question,
                role                 = role,
                processing_time_ms   = round(wall_ms, 2),
                retrieval_time_ms    = round(retrieval_time_ms, 2),
                generation_time_ms   = 0.0,
                total_tokens         = 0,
                confidence           = "0%",
                confidence_score     = 0.0,
                retrieval_mode       = retrieval_response.retrieval_mode,
                model_name           = "",
                template_used        = cfg.prompt_template.value,
                has_conflicts        = False,
                chunks_in_context    = 0,
            )

        # ---- Phase 2: Prompt Builder ------------------------------------
        from rag.prompt_builder import build_prompt
        built_prompt = build_prompt(question, results, cfg.to_prompt_config())

        # ---- Phase 3: LLM Generation ------------------------------------
        from rag.rag_engine import get_rag_engine
        rag_response = get_rag_engine().generate(
            built_prompt,
            temperature    = cfg.temperature,
            max_tokens     = cfg.max_tokens,
            top_p          = cfg.top_p,
            repeat_penalty = cfg.repeat_penalty,
        )

        # ---- Phase 4: Citation Engine -----------------------------------
        from rag.citation_engine import get_citation_engine
        confidence, conf_score = compute_confidence(results)
        citation_list = get_citation_engine().build(results, rag_response.answer)

        # ---- Assemble structured response --------------------------------
        sources_block          = format_sources_block(citation_list.citations)
        formatted_answer       = citation_list.answer_with_refs + sources_block

        n_unique_docs = len({r.citation.doc_id for r in results})
        wall_ms       = (time.perf_counter() - t_wall_start) * 1000

        response = RAGPipelineResponse(
            answer               = rag_response.answer,
            answer_with_refs     = citation_list.answer_with_refs,
            formatted_answer     = formatted_answer,
            citations            = citation_list.citations,
            retrieved_documents  = n_unique_docs,
            retrieved_chunks     = len(results),
            query                = question,
            role                 = role,
            processing_time_ms   = round(wall_ms, 2),
            retrieval_time_ms    = round(retrieval_time_ms, 2),
            generation_time_ms   = round(rag_response.generation_time_ms, 2),
            total_tokens         = rag_response.total_tokens,
            confidence           = confidence,
            confidence_score     = round(conf_score, 6),
            retrieval_mode       = retrieval_response.retrieval_mode,
            model_name           = rag_response.model_name,
            template_used        = rag_response.template_used,
            has_conflicts        = built_prompt.has_conflicts,
            chunks_in_context    = built_prompt.chunks_included,
        )

        logger.info("[PIPELINE] %s", response.short_summary())
        return response

    # ------------------------------------------------------------------
    # Streaming API -- yields text tokens, no citation processing
    # ------------------------------------------------------------------

    def run_stream(
        self,
        question: str,
        role: str = "Public",
        *,
        config_overrides: Optional[PipelineConfig] = None,
    ) -> Iterator[str]:
        """
        Streaming variant -- yields LLM text tokens as they arrive.

        Citation processing is skipped on this path because the citation
        engine requires the complete answer text.  Use run() for citations.

        Yields:
            str -- incremental text tokens from Qwen2.5:7B.
        """
        cfg  = config_overrides or self._config
        role = _normalize_role(role)

        retrieval_response = self._retrieve(question, role, cfg)
        results            = retrieval_response.results

        if not results:
            yield FALLBACK_ANSWER
            return

        from rag.prompt_builder import build_prompt
        from rag.rag_engine     import get_rag_engine

        built_prompt = build_prompt(question, results, cfg.to_prompt_config())
        yield from get_rag_engine().generate_stream(
            built_prompt,
            temperature    = cfg.temperature,
            max_tokens     = cfg.max_tokens,
            top_p          = cfg.top_p,
            repeat_penalty = cfg.repeat_penalty,
        )

    # ------------------------------------------------------------------
    # Streaming API (structured) -- yields live tokens AND final metadata
    # ------------------------------------------------------------------

    def run_stream_structured(
        self,
        question: str,
        role: str = "Public",
        *,
        config_overrides: Optional[PipelineConfig] = None,
    ) -> Iterator[tuple]:
        """
        Live-streaming variant that ALSO surfaces citations + confidence.

        Unlike run_stream() (tokens only) this drives the same retrieve →
        prompt → generate path but yields a structured 2-tuple stream so an
        SSE endpoint can paint tokens immediately and still emit a final
        metadata event:

            ("token", "<text chunk>")   -- repeated, as Qwen emits them
            ("meta",  {... citations, confidence, answer, timing ...})  -- once

        The full answer is accumulated as tokens arrive, so citations are
        built from the complete text after generation finishes — identical
        output to run(), just delivered live.
        """
        cfg          = config_overrides or self._config
        role         = _normalize_role(role)
        t_wall_start = time.perf_counter()

        retrieval_response = self._retrieve(question, role, cfg)
        retrieval_time_ms  = retrieval_response.latency_ms
        results            = retrieval_response.results

        # Same relevance gate as run() -- previously this path only checked
        # for zero results, so a streamed answer could confidently narrate
        # off irrelevant context in exactly the case run() was already
        # guarding against. See run()'s short-circuit for why the gate
        # exists.
        if not results or not passes_relevance_gate(results):
            wall_ms = (time.perf_counter() - t_wall_start) * 1000
            yield ("token", FALLBACK_ANSWER)
            yield ("meta", {
                "answer":             FALLBACK_ANSWER,
                "source_documents":   [],
                "citations":          [],
                "confidence":         "0%",
                "confidence_score":   0.0,
                "retrieval_mode":     retrieval_response.retrieval_mode,
                "processing_time_ms": round(wall_ms, 2),
            })
            return

        from rag.prompt_builder import build_prompt
        from rag.rag_engine     import get_rag_engine

        built_prompt = build_prompt(question, results, cfg.to_prompt_config())

        # ---- Stream tokens live, accumulating the full answer ----------
        chunks: list[str] = []
        for token in get_rag_engine().generate_stream(
            built_prompt,
            temperature    = cfg.temperature,
            max_tokens     = cfg.max_tokens,
            top_p          = cfg.top_p,
            repeat_penalty = cfg.repeat_penalty,
        ):
            chunks.append(token)
            yield ("token", token)

        full_answer = "".join(chunks).strip() or FALLBACK_ANSWER

        # ---- Citations + confidence from the complete answer -----------
        # confidence is a retrieval signal (see response_schema.
        # compute_confidence) -- it doesn't depend on the generated text,
        # so it's identical whether computed here or in run(). Passing
        # full_answer into it would have been meaningless anyway; the old
        # parse_llm_confidence only used the answer text to scrape a
        # self-reported number the model no longer writes.
        from rag.citation_engine import get_citation_engine
        citation_list          = get_citation_engine().build(results, full_answer)
        confidence, conf_score = compute_confidence(results)
        wall_ms                = (time.perf_counter() - t_wall_start) * 1000

        citations_payload = [
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

        yield ("meta", {
            "answer":             full_answer,
            "source_documents":   [c.display_name for c in citation_list.citations],
            "citations":          citations_payload,
            "confidence":         confidence,
            "confidence_score":   round(conf_score, 4),
            "retrieval_mode":     retrieval_response.retrieval_mode,
            "processing_time_ms": round(wall_ms, 2),
        })

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _retrieve(self, question: str, role: str, cfg: PipelineConfig):
        from retrieval.retriever import get_retriever
        return get_retriever().retrieve_by_text(
            text         = question,
            role         = role,
            top_k        = cfg.top_k_final,
            use_bm25     = cfg.use_bm25,
            use_reranker = cfg.use_reranker,
            top_k_dense  = cfg.top_k_dense,
            top_k_bm25   = cfg.top_k_bm25,
            top_k_fusion = cfg.top_k_fusion,
            top_k_final  = cfg.top_k_final,
        )

    @property
    def config(self) -> PipelineConfig:
        return self._config


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_pipeline_instance: Optional[RAGPipeline] = None


def get_pipeline(config: Optional[PipelineConfig] = None) -> RAGPipeline:
    """
    Returns the process-level RAGPipeline singleton.

    The first call may pass a PipelineConfig to configure the instance.
    Subsequent calls return the same instance regardless of the argument.
    """
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = RAGPipeline(config)
    return _pipeline_instance


def reset_pipeline() -> None:
    """Clears the singleton -- primarily for testing."""
    global _pipeline_instance
    _pipeline_instance = None
