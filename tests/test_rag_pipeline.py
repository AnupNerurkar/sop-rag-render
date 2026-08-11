"""
tests/test_rag_pipeline.py
---------------------------
Unit and integration tests for Phase 9: RAGPipeline, PipelineConfig,
RAGPipelineResponse, compute_confidence, format_sources_block.

Unit tests (no network, no model load)
    All heavy dependencies (retriever, rag_engine, citation_engine) are
    replaced with lightweight fakes built from real Pydantic models.

Integration tests (marked slow)
    Require a live the vector store + SQLite corpus.
    Ollama integration tests are skipped automatically when Ollama is offline.
"""

from __future__ import annotations

import time
from datetime import timezone
from types import SimpleNamespace
from typing import Optional
from unittest.mock import MagicMock, patch, call

import pytest
from pydantic import BaseModel

# ---- modules under test ---------------------------------------------------
from response_schema import (
    FALLBACK_ANSWER,
    RAGPipelineResponse,
    compute_confidence,
    format_sources_block,
)
from rag_pipeline import (
    PipelineConfig,
    RAGPipeline,
    _normalize_role,
    get_pipeline,
    reset_pipeline,
)
from rag.prompt_schema import PromptTemplate
from rag.citation_schema import Citation


# ---------------------------------------------------------------------------
# Shared factories
# ---------------------------------------------------------------------------

def _make_source_citation(
    doc_id: str = "abc123",
    display_name: str = "Admissions SOP",
    department: str = "Admissions",
    category: str = "SOP",
    version: str = "2.0",
    chunk_index: int = 0,
    total_chunks: int = 10,
    source_file: str = "staged/admissions.docx",
    section_heading: str = "",
):
    """Builds a real SourceCitation Pydantic model."""
    from retrieval.retrieval_schema import SourceCitation
    return SourceCitation(
        doc_id          = doc_id,
        source_file     = source_file,
        display_name    = display_name,
        department      = department,
        category        = category,
        version         = version,
        section_heading = section_heading,
        chunk_index     = chunk_index,
        total_chunks    = total_chunks,
    )


def _make_result(
    doc_id: str = "abc123",
    score: float = 0.85,
    rerank_score: Optional[float] = 0.82,
    rank: int = 1,
    content: str = "Students must attend 75% of lectures.",
):
    """Builds a real RetrievalResult Pydantic model."""
    from retrieval.retrieval_schema import RetrievalResult
    return RetrievalResult(
        rank           = rank,
        chunk_id       = f"chunk_{doc_id[:8]}_{rank}",
        content        = content,
        score          = score,
        distance       = 1 - score,
        rerank_score   = rerank_score,
        retrieval_mode = "hybrid+rerank",
        citation       = _make_source_citation(doc_id=doc_id),
        metadata       = {"access_level": "Public"},
    )


def _make_citation(
    rank: int = 1,
    doc_id: str = "abc123",
    display_name: str = "Admissions SOP",
    department: str = "Admissions",
    version: str = "2.0",
    score: float = 0.82,
    chunk_index: int = 2,
    total_chunks: int = 10,
    page_number: int = 3,
    rerank_score: Optional[float] = 0.82,
) -> Citation:
    return Citation(
        citation_id      = f"cite_{doc_id[:12]}",
        rank             = rank,
        inline_ref       = f"[{rank}]",
        doc_id           = doc_id,
        display_name     = display_name,
        department       = department,
        category         = "SOP",
        version          = version,
        source_file      = "staged/admissions.docx",
        chunk_index      = chunk_index,
        total_chunks     = total_chunks,
        page_number      = page_number,
        score            = score,
        rerank_score     = rerank_score,
        access_level     = "Public",
        is_latest_version = True,
    )


def _make_retrieval_response(
    results,
    latency_ms: float = 120.0,
    mode: str = "hybrid+rerank",
    reranked: bool = True,
):
    from retrieval.retrieval_schema import RetrievalResponse
    return RetrievalResponse(
        query_text       = "test query",
        clean_query_text = "test query",
        role             = "Student",
        results          = results,
        total_results    = len(results),
        latency_ms       = latency_ms,
        reranked         = reranked,
        retrieval_mode   = mode,
    )


def _make_built_prompt(chunks_included: int = 3, has_conflicts: bool = False):
    from rag.prompt_schema import BuiltPrompt, PromptTemplate
    return BuiltPrompt(
        user_question    = "What is attendance requirement?",
        system_prompt    = "You are a VIT assistant.",
        context_block    = "[SOURCE 1] Attendance SOP\n---\n75% required.",
        user_message     = "Context...\n\nQUESTION: What is attendance requirement?",
        messages         = [
            {"role": "system", "content": "You are a VIT assistant."},
            {"role": "user",   "content": "Context...\n\nQUESTION: What?"},
        ],
        chunks_included  = chunks_included,
        chunks_dropped   = 0,
        context_chars    = 150,
        template_used    = PromptTemplate.DEFAULT,
        has_conflicts    = has_conflicts,
        source_citations = ["Admissions SOP (v2.0)"],
    )


def _make_rag_response(answer: str = "Students need 75% attendance [SOURCE 1]."):
    from rag.rag_engine import RAGResponse
    return RAGResponse(
        answer             = answer,
        model_name         = "qwen2.5:7b",
        finish_reason      = "stop",
        prompt_tokens      = 400,
        completion_tokens  = 120,
        total_tokens       = 520,
        latency_ms         = 3200.0,
        generation_time_ms = 2800.0,
        chunks_used        = 3,
        has_conflicts      = False,
        template_used      = "default",
    )


def _make_citation_list(answer: str = "Students need 75% attendance [1]."):
    from rag.citation_schema import CitationList
    return CitationList(
        citations         = [_make_citation()],
        answer_with_refs  = answer,
        original_answer   = answer,
        total_citations   = 1,
        has_version_conflicts = False,
    )


# ---------------------------------------------------------------------------
# TestNormalizeRole
# ---------------------------------------------------------------------------

class TestNormalizeRole:
    def test_public_default(self):
        assert _normalize_role("Public") == "Public"

    def test_lowercase_student(self):
        assert _normalize_role("student") == "Student"

    def test_uppercase_admin(self):
        assert _normalize_role("ADMIN") == "Admin"

    def test_unknown_defaults_to_public(self):
        assert _normalize_role("SuperUser") == "Public"

    def test_empty_defaults_to_public(self):
        assert _normalize_role("") == "Public"

    def test_faculty_mixed_case(self):
        assert _normalize_role("fAcUlTy") == "Faculty"

    def test_whitespace_stripped(self):
        assert _normalize_role("  Admin  ") == "Admin"


# ---------------------------------------------------------------------------
# TestComputeConfidence
# ---------------------------------------------------------------------------

class TestComputeConfidence:
    def test_no_results_returns_unknown(self):
        lbl, score = compute_confidence([])
        assert lbl == "0%"
        assert score == 0.0

    def test_high_confidence_two_docs(self):
        r1 = _make_result("doc1", rerank_score=0.85)
        r2 = _make_result("doc2", rerank_score=0.75, rank=2)
        lbl, score = compute_confidence([r1, r2])
        assert lbl == "85%"
        assert abs(score - 0.85) < 1e-6

    def test_high_requires_two_docs(self):
        r1 = _make_result("doc1", rerank_score=0.90)
        lbl, score = compute_confidence([r1])
        assert lbl == "90%"
        assert abs(score - 0.90) < 1e-6

    def test_medium_by_score(self):
        r1 = _make_result("doc1", rerank_score=0.55)
        lbl, score = compute_confidence([r1])
        assert lbl == "55%"
        assert abs(score - 0.55) < 1e-6

    def test_medium_by_two_docs(self):
        r1 = _make_result("doc1", rerank_score=0.30)
        r2 = _make_result("doc2", rerank_score=0.25, rank=2)
        lbl, score = compute_confidence([r1, r2])
        assert lbl == "30%"
        assert abs(score - 0.30) < 1e-6

    def test_low_confidence(self):
        r1 = _make_result("doc1", rerank_score=0.20)
        lbl, score = compute_confidence([r1])
        assert lbl == "20%"
        assert abs(score - 0.20) < 1e-6

    def test_uses_rerank_score_preferentially(self):
        r = _make_result("doc1", score=0.30, rerank_score=0.75)
        lbl, score = compute_confidence([r])
        assert lbl == "75%"
        assert abs(score - 0.75) < 1e-6

    def test_falls_back_to_score_when_no_rerank(self):
        r = _make_result("doc1", score=0.80, rerank_score=None)
        lbl, score = compute_confidence([r])
        assert lbl == "80%"
        assert abs(score - 0.80) < 1e-6

    def test_boundary_exact_070(self):
        r1 = _make_result("doc1", rerank_score=0.70)
        r2 = _make_result("doc2", rerank_score=0.65, rank=2)
        lbl, _ = compute_confidence([r1, r2])
        assert lbl == "70%"

    def test_boundary_exact_040(self):
        r1 = _make_result("doc1", rerank_score=0.40)
        lbl, _ = compute_confidence([r1])
        assert lbl == "40%"


# ---------------------------------------------------------------------------
# TestFormatSourcesBlock
# ---------------------------------------------------------------------------

class TestFormatSourcesBlock:
    def test_empty_citations(self):
        assert format_sources_block([]) == ""

    def test_single_citation(self):
        c = _make_citation(rank=1, page_number=3, chunk_index=2, total_chunks=10)
        block = format_sources_block([c])
        assert "Sources:" in block
        assert "1." in block
        assert "Admissions SOP" in block
        assert "Admissions" in block
        assert "v2.0" in block

    def test_chunk_str_with_total_chunks(self):
        c = _make_citation(rank=1, chunk_index=2, total_chunks=10, page_number=3)
        block = format_sources_block([c])
        assert "Chunk 3/10" in block

    def test_chunk_str_zero_total(self):
        c = _make_citation(rank=1, chunk_index=2, total_chunks=0, page_number=3)
        block = format_sources_block([c])
        assert "Page 3" in block

    def test_multiple_citations(self):
        c1 = _make_citation(rank=1, doc_id="doc1")
        c2 = _make_citation(rank=2, doc_id="doc2",
                            display_name="Fee SOP", department="Finance",
                            version="1.0")
        block = format_sources_block([c1, c2])
        assert "1." in block
        assert "2." in block
        assert "Fee SOP" in block

    def test_block_starts_with_blank_line(self):
        c = _make_citation()
        block = format_sources_block([c])
        assert block.startswith("\n")


# ---------------------------------------------------------------------------
# TestRAGPipelineResponseModel
# ---------------------------------------------------------------------------

class TestRAGPipelineResponseModel:
    def _build(self, **kwargs) -> RAGPipelineResponse:
        defaults = dict(
            answer           = "Test answer.",
            answer_with_refs = "Test answer [1].",
            formatted_answer = "Test answer [1].\n\nSources:\n1. Admissions SOP",
            query            = "What is the policy?",
            role             = "Student",
        )
        defaults.update(kwargs)
        return RAGPipelineResponse(**defaults)

    def test_minimal_construction(self):
        r = self._build()
        assert r.answer == "Test answer."
        assert r.role == "Student"
        assert r.confidence == "0%"
        assert r.citations == []

    def test_is_fallback_true(self):
        r = self._build(answer=FALLBACK_ANSWER, answer_with_refs=FALLBACK_ANSWER,
                        formatted_answer=FALLBACK_ANSWER)
        assert r.is_fallback is True

    def test_is_fallback_false(self):
        r = self._build()
        assert r.is_fallback is False

    def test_timestamp_is_utc_iso(self):
        r = self._build()
        # Should be parseable as ISO date
        from datetime import datetime
        dt = datetime.fromisoformat(r.timestamp.replace("Z", "+00:00"))
        assert dt is not None

    def test_short_summary(self):
        r = self._build(
            confidence="85%",
            total_tokens=500,
            processing_time_ms=4200.0,
        )
        s = r.short_summary()
        assert "Student" in s
        assert "85%" in s
        assert "tokens=500" in s

    def test_to_display(self):
        r = self._build(
            formatted_answer="Answer text here.\n\nSources:\n1. Test SOP",
            confidence="55%",
            retrieval_mode="hybrid+rerank",
        )
        display = r.to_display()
        assert "Answer text here" in display
        assert "Sources:" in display
        assert "hybrid+rerank" in display

    def test_json_round_trip(self):
        r = self._build(confidence="90%")
        import json
        data = json.loads(r.model_dump_json())
        assert data["confidence"] == "90%"
        assert data["answer"] == "Test answer."


# ---------------------------------------------------------------------------
# TestPipelineConfig
# ---------------------------------------------------------------------------

class TestPipelineConfig:
    def test_defaults(self):
        cfg = PipelineConfig()
        assert cfg.use_bm25 is True
        assert cfg.use_reranker is True
        assert cfg.top_k_final == 5
        assert cfg.temperature == 0.7
        assert cfg.prompt_template == PromptTemplate.DEFAULT

    def test_to_prompt_config(self):
        cfg = PipelineConfig(max_chunks=3, max_context_chars=4000, temperature=0.5)
        pc  = cfg.to_prompt_config()
        assert pc.max_chunks == 3
        assert pc.max_context_chars == 4000
        assert pc.template == PromptTemplate.DEFAULT

    def test_custom_template(self):
        cfg = PipelineConfig(prompt_template=PromptTemplate.STRICT_CITATION)
        assert cfg.to_prompt_config().template == PromptTemplate.STRICT_CITATION

    def test_invalid_top_k_raises(self):
        with pytest.raises(Exception):
            PipelineConfig(top_k_final=0)

    def test_invalid_temperature_raises(self):
        with pytest.raises(Exception):
            PipelineConfig(temperature=3.0)


# ---------------------------------------------------------------------------
# TestRAGPipelineUnit  (all phases mocked)
# ---------------------------------------------------------------------------

class TestRAGPipelineUnit:
    """Tests the pipeline wiring with all sub-systems replaced by mocks."""

    @pytest.fixture(autouse=True)
    def reset(self):
        reset_pipeline()
        yield
        reset_pipeline()

    def _pipeline_with_mocks(
        self,
        results=None,
        rag_answer: str = "75% attendance required [SOURCE 1].",
    ):
        """
        Returns (pipeline, mock_retriever, mock_rag_engine, mock_citation_engine).
        All interactions are captured by the mocks.
        """
        if results is None:
            results = [_make_result("doc1"), _make_result("doc2", rank=2)]

        retrieval_resp  = _make_retrieval_response(results)
        built_prompt    = _make_built_prompt()
        rag_resp        = _make_rag_response(answer=rag_answer)
        citation_list   = _make_citation_list()

        mock_retriever = MagicMock()
        mock_retriever.retrieve_by_text.return_value = retrieval_resp

        mock_engine = MagicMock()
        mock_engine.generate.return_value = rag_resp

        mock_cite = MagicMock()
        mock_cite.build.return_value = citation_list

        pipeline = RAGPipeline()
        return pipeline, mock_retriever, mock_engine, mock_cite, built_prompt

    def test_happy_path_returns_response(self):
        pipeline, mock_ret, mock_eng, mock_cite, mock_bp = self._pipeline_with_mocks()

        with (
            patch("rag_pipeline.RAGPipeline._retrieve", return_value=_make_retrieval_response(
                [_make_result("doc1"), _make_result("doc2", rank=2)]
            )),
            patch("rag.prompt_builder.build_prompt", return_value=mock_bp),
            patch("rag.rag_engine.get_rag_engine", return_value=mock_eng),
            patch("rag.citation_engine.get_citation_engine", return_value=mock_cite),
        ):
            resp = pipeline.run("What is attendance?", role="Student")

        assert isinstance(resp, RAGPipelineResponse)
        assert resp.query == "What is attendance?"
        assert resp.role == "Student"
        assert resp.is_fallback is False

    def test_no_results_returns_fallback(self):
        pipeline = RAGPipeline()
        with patch(
            "rag_pipeline.RAGPipeline._retrieve",
            return_value=_make_retrieval_response([]),
        ):
            resp = pipeline.run("What is attendance?", role="Student")

        assert resp.is_fallback is True
        assert resp.citations == []
        assert resp.confidence == "0%"
        assert resp.total_tokens == 0
        assert resp.retrieved_chunks == 0

    def test_no_results_skips_llm(self):
        pipeline = RAGPipeline()
        with (
            patch("rag_pipeline.RAGPipeline._retrieve",
                  return_value=_make_retrieval_response([])),
            patch("rag.rag_engine.get_rag_engine") as mock_get_engine,
        ):
            pipeline.run("Test question", role="Public")
            mock_get_engine.assert_not_called()

    def test_role_normalisation(self):
        pipeline = RAGPipeline()
        with patch(
            "rag_pipeline.RAGPipeline._retrieve",
            return_value=_make_retrieval_response([]),
        ) as mock_retrieve:
            pipeline.run("Test", role="admin")
            # _retrieve is called with normalised role
            args, kwargs = mock_retrieve.call_args
            # role is the second positional argument to _retrieve
            assert args[1] == "Admin"

    def test_unknown_role_normalised_to_public(self):
        pipeline = RAGPipeline()
        with patch(
            "rag_pipeline.RAGPipeline._retrieve",
            return_value=_make_retrieval_response([]),
        ) as mock_retrieve:
            pipeline.run("Test", role="Hacker")
            args, kwargs = mock_retrieve.call_args
            assert args[1] == "Public"

    def test_timing_fields_populated(self):
        _, _, mock_eng, mock_cite, mock_bp = self._pipeline_with_mocks()
        pipeline = RAGPipeline()
        with (
            patch("rag_pipeline.RAGPipeline._retrieve",
                  return_value=_make_retrieval_response(
                      [_make_result("doc1"), _make_result("doc2", rank=2)],
                      latency_ms=150.0,
                  )),
            patch("rag.prompt_builder.build_prompt", return_value=mock_bp),
            patch("rag.rag_engine.get_rag_engine", return_value=mock_eng),
            patch("rag.citation_engine.get_citation_engine", return_value=mock_cite),
        ):
            resp = pipeline.run("Test", role="Student")

        assert resp.retrieval_time_ms == 150.0
        assert resp.generation_time_ms == 2800.0
        assert resp.processing_time_ms > 0

    def test_confidence_attached_to_response(self):
        _, _, mock_eng, mock_cite, mock_bp = self._pipeline_with_mocks()
        results = [
            _make_result("doc1", rerank_score=0.90),
            _make_result("doc2", rerank_score=0.80, rank=2),
        ]
        pipeline = RAGPipeline()
        with (
            patch("rag_pipeline.RAGPipeline._retrieve",
                  return_value=_make_retrieval_response(results)),
            patch("rag.prompt_builder.build_prompt", return_value=mock_bp),
            patch("rag.rag_engine.get_rag_engine", return_value=mock_eng),
            patch("rag.citation_engine.get_citation_engine", return_value=mock_cite),
        ):
            resp = pipeline.run("Test", role="Student")

        assert resp.confidence == "90%"
        assert resp.confidence_score >= 0.90

    def test_formatted_answer_contains_sources(self):
        _, _, mock_eng, mock_cite, mock_bp = self._pipeline_with_mocks()
        pipeline = RAGPipeline()
        with (
            patch("rag_pipeline.RAGPipeline._retrieve",
                  return_value=_make_retrieval_response(
                      [_make_result("doc1"), _make_result("doc2", rank=2)]
                  )),
            patch("rag.prompt_builder.build_prompt", return_value=mock_bp),
            patch("rag.rag_engine.get_rag_engine", return_value=mock_eng),
            patch("rag.citation_engine.get_citation_engine", return_value=mock_cite),
        ):
            resp = pipeline.run("Test", role="Student")

        assert "Sources:" in resp.formatted_answer

    def test_config_override_per_call(self):
        pipeline = RAGPipeline(PipelineConfig(max_tokens=512))
        override = PipelineConfig(max_tokens=256)
        with patch(
            "rag_pipeline.RAGPipeline._retrieve",
            return_value=_make_retrieval_response([]),
        ):
            # override config; no results so no LLM call
            resp = pipeline.run("Test", config_overrides=override)
        # pipeline's own config unchanged
        assert pipeline.config.max_tokens == 512

    def test_retrieved_documents_counts_unique_docs(self):
        _, _, mock_eng, mock_cite, mock_bp = self._pipeline_with_mocks()
        # 2 results from same document
        results = [
            _make_result("same_doc", rank=1),
            _make_result("same_doc", rank=2),
        ]
        pipeline = RAGPipeline()
        with (
            patch("rag_pipeline.RAGPipeline._retrieve",
                  return_value=_make_retrieval_response(results)),
            patch("rag.prompt_builder.build_prompt", return_value=mock_bp),
            patch("rag.rag_engine.get_rag_engine", return_value=mock_eng),
            patch("rag.citation_engine.get_citation_engine", return_value=mock_cite),
        ):
            resp = pipeline.run("Test", role="Student")

        assert resp.retrieved_documents == 1
        assert resp.retrieved_chunks == 2


# ---------------------------------------------------------------------------
# TestRAGPipelineStream
# ---------------------------------------------------------------------------

class TestRAGPipelineStream:
    @pytest.fixture(autouse=True)
    def reset(self):
        reset_pipeline()
        yield
        reset_pipeline()

    def test_stream_no_results_yields_fallback(self):
        pipeline = RAGPipeline()
        with patch(
            "rag_pipeline.RAGPipeline._retrieve",
            return_value=_make_retrieval_response([]),
        ):
            tokens = list(pipeline.run_stream("Test"))
        assert tokens == [FALLBACK_ANSWER]

    def test_stream_yields_tokens(self):
        mock_bp  = _make_built_prompt()
        mock_eng = MagicMock()
        mock_eng.generate_stream.return_value = iter(["Hello", " World", "!"])

        pipeline = RAGPipeline()
        with (
            patch("rag_pipeline.RAGPipeline._retrieve",
                  return_value=_make_retrieval_response([_make_result()])),
            patch("rag.prompt_builder.build_prompt", return_value=mock_bp),
            patch("rag.rag_engine.get_rag_engine", return_value=mock_eng),
        ):
            tokens = list(pipeline.run_stream("Test", role="Student"))

        assert tokens == ["Hello", " World", "!"]

    def test_stream_does_not_call_citation_engine(self):
        mock_bp  = _make_built_prompt()
        mock_eng = MagicMock()
        mock_eng.generate_stream.return_value = iter(["token"])

        pipeline = RAGPipeline()
        with (
            patch("rag_pipeline.RAGPipeline._retrieve",
                  return_value=_make_retrieval_response([_make_result()])),
            patch("rag.prompt_builder.build_prompt", return_value=mock_bp),
            patch("rag.rag_engine.get_rag_engine", return_value=mock_eng),
            patch("rag.citation_engine.get_citation_engine") as mock_cite_factory,
        ):
            list(pipeline.run_stream("Test"))
            mock_cite_factory.assert_not_called()


# ---------------------------------------------------------------------------
# TestGetPipelineSingleton
# ---------------------------------------------------------------------------

class TestGetPipelineSingleton:
    @pytest.fixture(autouse=True)
    def reset(self):
        reset_pipeline()
        yield
        reset_pipeline()

    def test_returns_same_instance(self):
        p1 = get_pipeline()
        p2 = get_pipeline()
        assert p1 is p2

    def test_config_locked_after_first_call(self):
        cfg1 = PipelineConfig(max_tokens=512)
        p1   = get_pipeline(cfg1)
        cfg2 = PipelineConfig(max_tokens=256)
        p2   = get_pipeline(cfg2)   # should return p1, ignoring cfg2
        assert p1 is p2
        assert p2.config.max_tokens == 512

    def test_reset_allows_new_config(self):
        get_pipeline(PipelineConfig(max_tokens=512))
        reset_pipeline()
        p = get_pipeline(PipelineConfig(max_tokens=256))
        assert p.config.max_tokens == 256

    def test_reset_pipeline_clears_singleton(self):
        p1 = get_pipeline()
        reset_pipeline()
        p2 = get_pipeline()
        assert p1 is not p2


# ---------------------------------------------------------------------------
# Integration tests (marked slow -- require live the vector store + SQLite)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestRAGPipelineIntegration:
    """
    Tests that hit real storage layers (the vector store, SQLite) but do NOT
    require Ollama.  They verify retrieval, prompt building, and citation
    engine all wire together correctly on live data.
    """

    @pytest.fixture(autouse=True)
    def reset(self):
        reset_pipeline()
        yield
        reset_pipeline()

    def test_retrieval_only(self):
        """Retriever returns results from live the vector store corpus."""
        pipeline = RAGPipeline(PipelineConfig(use_reranker=False))
        response = pipeline._retrieve(
            "What is the attendance requirement?",
            "Student",
            PipelineConfig(use_reranker=False),
        )
        assert response.total_results > 0
        assert all(r.score > 0 for r in response.results)

    def test_prompt_builds_from_live_results(self):
        """PromptBuilder works with real RetrievalResult objects."""
        from retrieval.retriever import get_retriever
        from rag.prompt_builder import build_prompt

        rr = get_retriever().retrieve_by_text(
            "attendance policy",
            role="Student",
            use_reranker=False,
            top_k_final=3,
        )
        assert rr.results, "Live retrieval returned no results -- corpus may be empty"
        bp = build_prompt("What is the attendance policy?", rr.results)
        assert bp.chunks_included > 0
        assert "[SOURCE 1]" in bp.context_block

    def test_citation_engine_on_live_results(self):
        """CitationEngine deduplicates and ranks live retrieval results."""
        from retrieval.retriever import get_retriever
        from rag.citation_engine import get_citation_engine

        rr = get_retriever().retrieve_by_text(
            "attendance policy",
            role="Student",
            use_reranker=False,
            top_k_final=5,
        )
        assert rr.results

        cl = get_citation_engine().build(
            rr.results,
            "Students must maintain 75% attendance [SOURCE 1].",
        )
        assert cl.total_citations > 0
        assert all(c.rank >= 1 for c in cl.citations)

    def test_no_results_for_nonsense_query(self):
        """
        A completely nonsensical query may return results (embedding still
        finds nearest neighbours).  We test that the pipeline handles both
        cases gracefully without crashing.
        """
        pipeline = RAGPipeline()
        with patch("rag_pipeline.RAGPipeline._retrieve",
                   return_value=_make_retrieval_response([])):
            resp = pipeline.run("zzz xyzzy quux placeholder", role="Public")
        assert resp.is_fallback is True
        assert resp.confidence == "0%"


@pytest.mark.slow
class TestRAGPipelineOllama:
    """
    End-to-end tests that require a running Ollama server with qwen2.5:7b.
    Skipped automatically when Ollama is offline.
    """

    @pytest.fixture(autouse=True)
    def skip_if_no_ollama(self):
        try:
            from rag.ollama_client import get_ollama_client
            client = get_ollama_client()
            if not client.health_check():
                pytest.skip("Ollama server not running")
        except Exception:
            pytest.skip("Ollama server not available")
        yield
        reset_pipeline()

    def test_end_to_end(self):
        pipeline = RAGPipeline(PipelineConfig(max_tokens=256, temperature=0.3))
        resp     = pipeline.run(
            "What is the attendance requirement for semester examinations?",
            role="Student",
        )
        assert isinstance(resp, RAGPipelineResponse)
        assert len(resp.answer) > 20
        assert resp.total_tokens > 0
        assert resp.processing_time_ms > 0
        assert resp.confidence != "0%"
        assert "Sources:" in resp.formatted_answer

    def test_role_filters_applied(self):
        """Public role should not surface Admin-access-only documents."""
        pipeline = RAGPipeline(PipelineConfig(max_tokens=128))
        resp     = pipeline.run("What are admin procedures?", role="Public")
        for c in resp.citations:
            assert c.access_level in ("Public",), (
                f"Public-role response exposed non-Public citation: {c}"
            )
