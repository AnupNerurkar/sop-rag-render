"""
tests/test_prompt_builder.py
-----------------------------
Unit tests for the Prompt Builder layer (rag/prompt_builder.py, rag/prompt_schema.py).

All tests are pure-Python — no the vector store, no models, no SQLite.
Run: pytest tests/test_prompt_builder.py -v
"""

from __future__ import annotations

import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.prompt_schema import (
    BuiltPrompt,
    PromptTemplate,
    PromptConfig,
    ContextChunk,
    ConflictGroup,
    confidence_from_score,
)
from rag.prompt_builder import PromptBuilder, _version_key, build_prompt


# ===========================================================================
# Fixtures & helpers
# ===========================================================================

def _make_retrieval_result(
    rank: int = 1,
    content: str = "The fee payment deadline is 15th of each month.",
    score: float = 0.85,
    rerank_score: float = 0.90,
    department: str = "Finance",
    category: str = "SOP",
    version: str = "2.0",
    doc_id: str = "abc123",
    display_name: str = "Fee Payment SOP",
    section_heading: str = "Fee Deadlines",
):
    """Constructs a minimal mock RetrievalResult without importing the full retrieval stack."""
    from types import SimpleNamespace
    citation = SimpleNamespace(
        doc_id           = doc_id,
        source_file      = f"docs/{display_name}.docx",
        display_name     = display_name,
        department       = department,
        category         = category,
        version          = version,
        section_heading  = section_heading,
        chunk_index      = rank - 1,
        total_chunks     = 10,
        to_inline_citation  = lambda: f"[{display_name} v{version} § {section_heading} (chunk {rank}/10)]",
        to_display_citation = lambda: f"{display_name} (v{version}) — {section_heading} — chunk {rank} of 10",
    )
    return SimpleNamespace(
        rank         = rank,
        chunk_id     = f"chunk_{rank:02d}",
        content      = content,
        score        = score,
        distance     = 1 - score,
        rerank_score = rerank_score,
        retrieval_mode = "hybrid+rerank",
        citation     = citation,
        metadata     = {"department": department, "access_level": "Public"},
    )


def _make_results(n: int = 3) -> list:
    """Returns n distinct mock RetrievalResults."""
    contents = [
        "The fee payment deadline is 15th of each month.",
        "Late fee of 500 INR applies after the deadline.",
        "Online payment is accepted via the student portal.",
        "Cash payment at the accounts office is also accepted.",
        "Fee receipts must be preserved for re-admission.",
    ]
    return [
        _make_retrieval_result(
            rank         = i + 1,
            content      = contents[i % len(contents)],
            score        = 0.95 - i * 0.05,
            rerank_score = 0.92 - i * 0.05,
        )
        for i in range(n)
    ]


@pytest.fixture
def builder() -> PromptBuilder:
    return PromptBuilder()


@pytest.fixture
def default_config() -> PromptConfig:
    return PromptConfig()


# ===========================================================================
# confidence_from_score
# ===========================================================================

class TestConfidenceFromScore:
    def test_none_returns_unknown(self):
        assert confidence_from_score(None) == "0%"

    def test_high_threshold(self):
        assert confidence_from_score(0.70) == "70%"
        assert confidence_from_score(1.00) == "100%"
        assert confidence_from_score(0.71) == "71%"

    def test_medium_threshold(self):
        assert confidence_from_score(0.40) == "40%"
        assert confidence_from_score(0.55) == "55%"
        assert confidence_from_score(0.69) == "69%"

    def test_low_threshold(self):
        assert confidence_from_score(0.00) == "0%"
        assert confidence_from_score(0.20) == "20%"
        assert confidence_from_score(0.39) == "39%"


# ===========================================================================
# _version_key
# ===========================================================================

class TestVersionKey:
    def test_numeric_ordering(self):
        assert _version_key("2.0") > _version_key("1.5")
        assert _version_key("1.5") > _version_key("1.0")
        assert _version_key("3.0") > _version_key("2.0")

    def test_non_numeric_lower_than_numeric(self):
        # Non-numeric maps to (0,) which is less than any real version
        assert _version_key("Final") <= _version_key("1.0")

    def test_same_version_equal(self):
        assert _version_key("1.0") == _version_key("1.0")

    def test_multi_part(self):
        assert _version_key("2.1") > _version_key("2.0")


# ===========================================================================
# ContextChunk.to_prompt_block
# ===========================================================================

class TestContextChunkToPromptBlock:
    @pytest.fixture
    def chunk(self) -> ContextChunk:
        return ContextChunk(
            source_number    = 1,
            chunk_id         = "c01",
            content          = "Fee must be paid by 15th.",
            inline_citation  = "[Fee SOP v2.0 § Deadlines (chunk 1/10)]",
            display_citation = "Fee SOP (v2.0) — Deadlines — chunk 1 of 10",
            department       = "Finance",
            category         = "SOP",
            version          = "2.0",
            doc_id           = "abc123",
            rank             = 1,
            score            = 0.90,
            rerank_score     = 0.92,
            confidence       = "92%",
        )

    def test_header_contains_source_number(self, chunk):
        block = chunk.to_prompt_block()
        assert "[SOURCE 1]" in block

    def test_header_contains_display_citation(self, chunk):
        block = chunk.to_prompt_block()
        assert "Fee SOP (v2.0) — Deadlines — chunk 1 of 10" in block

    def test_metadata_row_included_by_default(self, chunk):
        block = chunk.to_prompt_block(include_metadata=True)
        assert "Finance" in block
        assert "Confidence: 92%" in block

    def test_metadata_row_excluded(self, chunk):
        block = chunk.to_prompt_block(include_metadata=False)
        assert "Department:" not in block
        assert "Confidence:" not in block

    def test_content_included(self, chunk):
        block = chunk.to_prompt_block()
        assert "Fee must be paid by 15th." in block

    def test_separator_present(self, chunk):
        block = chunk.to_prompt_block()
        assert "---" in block

    def test_rerank_score_used_in_metadata(self, chunk):
        block = chunk.to_prompt_block(include_metadata=True)
        assert "0.920" in block


# ===========================================================================
# ConflictGroup.to_warning
# ===========================================================================

class TestConflictGroupToWarning:
    def test_warning_contains_display_name(self):
        cg = ConflictGroup(
            display_name   = "Admissions SOP",
            department     = "Admissions",
            category       = "SOP",
            versions       = ["2.0", "1.0"],
            source_numbers = [1, 3],
            latest_version = "2.0",
        )
        w = cg.to_warning()
        assert "Admissions SOP" in w

    def test_warning_contains_latest_version(self):
        cg = ConflictGroup(
            display_name   = "Admissions SOP",
            department     = "Admissions",
            category       = "SOP",
            versions       = ["2.0", "1.0"],
            source_numbers = [1, 3],
            latest_version = "2.0",
        )
        w = cg.to_warning()
        assert "v2.0" in w

    def test_warning_contains_source_numbers(self):
        cg = ConflictGroup(
            display_name   = "Admissions SOP",
            department     = "Admissions",
            category       = "SOP",
            versions       = ["2.0", "1.0"],
            source_numbers = [1, 3],
            latest_version = "2.0",
        )
        w = cg.to_warning()
        assert "[SOURCE 1]" in w
        assert "[SOURCE 3]" in w


# ===========================================================================
# PromptConfig validation
# ===========================================================================

class TestPromptConfig:
    def test_defaults(self):
        cfg = PromptConfig()
        assert cfg.template == PromptTemplate.DEFAULT
        assert cfg.max_context_chars == 8000
        assert cfg.include_metadata is True
        assert cfg.max_chunks == 5
        assert cfg.confidence_threshold == 0.0

    def test_custom_values(self):
        cfg = PromptConfig(
            template           = PromptTemplate.CONCISE,
            max_context_chars  = 4000,
            include_metadata   = False,
            max_chunks         = 3,
            confidence_threshold = 0.5,
        )
        assert cfg.template == PromptTemplate.CONCISE
        assert cfg.max_context_chars == 4000
        assert cfg.max_chunks == 3

    def test_max_context_chars_bounds(self):
        with pytest.raises(Exception):
            PromptConfig(max_context_chars=100)   # below ge=500
        with pytest.raises(Exception):
            PromptConfig(max_context_chars=99999)  # above le=32000


# ===========================================================================
# PromptBuilder._prepare_chunks
# ===========================================================================

class TestPrepareChunks:
    def test_returns_context_chunks(self, builder, default_config):
        results = _make_results(3)
        chunks, dropped = builder._prepare_chunks(results, default_config)
        assert len(chunks) == 3
        assert dropped == 0
        assert all(isinstance(c, ContextChunk) for c in chunks)

    def test_source_numbers_start_at_1(self, builder, default_config):
        results = _make_results(3)
        chunks, _ = builder._prepare_chunks(results, default_config)
        assert [c.source_number for c in chunks] == [1, 2, 3]

    def test_max_chunks_respected(self, builder):
        cfg = PromptConfig(max_chunks=2)
        results = _make_results(5)
        chunks, dropped = builder._prepare_chunks(results, cfg)
        assert len(chunks) == 2
        assert dropped == 3

    def test_confidence_threshold_filters(self, builder):
        results = _make_results(3)
        # All rerank_scores are 0.92, 0.87, 0.82 — threshold 0.85 should filter last
        cfg = PromptConfig(confidence_threshold=0.85)
        chunks, dropped = builder._prepare_chunks(results, cfg)
        for c in chunks:
            eff = c.rerank_score if c.rerank_score is not None else c.score
            assert eff >= 0.85

    def test_budget_cap_respected(self, builder):
        # Use a very small budget — only first chunk should fit
        cfg = PromptConfig(max_context_chars=500)
        results = _make_results(5)
        chunks, _ = builder._prepare_chunks(results, cfg)
        total_chars = sum(len(c.to_prompt_block()) for c in chunks)
        assert total_chars <= 600  # some slack for truncated last chunk

    def test_rerank_score_used_for_confidence(self, builder, default_config):
        results = [_make_retrieval_result(rerank_score=0.95)]
        chunks, _ = builder._prepare_chunks(results, default_config)
        assert chunks[0].confidence == "95%"

    def test_score_fallback_when_no_rerank(self, builder, default_config):
        r = _make_retrieval_result(score=0.45, rerank_score=None)
        r.rerank_score = None
        chunks, _ = builder._prepare_chunks([r], default_config)
        assert chunks[0].confidence == "45%"

    def test_empty_results_returns_empty(self, builder, default_config):
        chunks, dropped = builder._prepare_chunks([], default_config)
        assert chunks == []
        assert dropped == 0


# ===========================================================================
# PromptBuilder._detect_conflicts
# ===========================================================================

class TestDetectConflicts:
    def _make_chunk(
        self, source_number: int, display_name: str,
        dept: str, cat: str, version: str,
    ) -> ContextChunk:
        return ContextChunk(
            source_number    = source_number,
            chunk_id         = f"c{source_number:02d}",
            content          = "Sample content.",
            inline_citation  = f"[{display_name} v{version}]",
            display_citation = f"{display_name} (v{version}) — Section",
            department       = dept,
            category         = cat,
            version          = version,
            doc_id           = f"doc_{source_number}",
            rank             = source_number,
            score            = 0.9,
        )

    def test_no_conflict_single_version(self, builder):
        chunks = [
            self._make_chunk(1, "Fee SOP", "Finance", "SOP", "2.0"),
            self._make_chunk(2, "Fee SOP", "Finance", "SOP", "2.0"),
        ]
        assert builder._detect_conflicts(chunks) == []

    def test_conflict_different_versions(self, builder):
        chunks = [
            self._make_chunk(1, "Fee SOP", "Finance", "SOP", "1.0"),
            self._make_chunk(2, "Fee SOP", "Finance", "SOP", "2.0"),
        ]
        conflicts = builder._detect_conflicts(chunks)
        assert len(conflicts) == 1
        assert conflicts[0].latest_version == "2.0"
        assert set(conflicts[0].versions) == {"1.0", "2.0"}

    def test_latest_version_correct(self, builder):
        chunks = [
            self._make_chunk(1, "Admissions SOP", "Admissions", "SOP", "1.0"),
            self._make_chunk(2, "Admissions SOP", "Admissions", "SOP", "3.0"),
            self._make_chunk(3, "Admissions SOP", "Admissions", "SOP", "2.0"),
        ]
        conflicts = builder._detect_conflicts(chunks)
        assert conflicts[0].latest_version == "3.0"

    def test_no_conflict_different_docs(self, builder):
        chunks = [
            self._make_chunk(1, "Fee SOP",        "Finance",    "SOP", "1.0"),
            self._make_chunk(2, "Admissions SOP", "Admissions", "SOP", "1.0"),
        ]
        assert builder._detect_conflicts(chunks) == []

    def test_multiple_conflict_groups(self, builder):
        chunks = [
            self._make_chunk(1, "Fee SOP",        "Finance",    "SOP", "1.0"),
            self._make_chunk(2, "Fee SOP",        "Finance",    "SOP", "2.0"),
            self._make_chunk(3, "Admissions SOP", "Admissions", "SOP", "1.0"),
            self._make_chunk(4, "Admissions SOP", "Admissions", "SOP", "3.0"),
        ]
        conflicts = builder._detect_conflicts(chunks)
        assert len(conflicts) == 2


# ===========================================================================
# PromptBuilder._format_context_block
# ===========================================================================

class TestFormatContextBlock:
    def _make_chunk(self, n: int) -> ContextChunk:
        return ContextChunk(
            source_number    = n,
            chunk_id         = f"c{n:02d}",
            content          = f"Content of chunk {n}.",
            inline_citation  = f"[Doc v1.0 (chunk {n}/5)]",
            display_citation = f"Doc (v1.0) — Section — chunk {n} of 5",
            department       = "Test",
            category         = "SOP",
            version          = "1.0",
            doc_id           = "d1",
            rank             = n,
            score            = 0.9,
        )

    def test_empty_returns_notice(self, builder):
        block = builder._format_context_block([], include_metadata=True)
        assert "No relevant context" in block

    def test_header_shows_chunk_count(self, builder):
        chunks = [self._make_chunk(i) for i in range(1, 4)]
        block = builder._format_context_block(chunks, include_metadata=True)
        assert "3 sources" in block

    def test_all_source_numbers_present(self, builder):
        chunks = [self._make_chunk(i) for i in range(1, 4)]
        block = builder._format_context_block(chunks, include_metadata=True)
        for i in range(1, 4):
            assert f"[SOURCE {i}]" in block

    def test_singular_source_label(self, builder):
        chunks = [self._make_chunk(1)]
        block = builder._format_context_block(chunks, include_metadata=True)
        assert "1 source" in block
        assert "1 sources" not in block


# ===========================================================================
# PromptBuilder._render_system_prompt
# ===========================================================================

class TestRenderSystemPrompt:
    def test_default_template_contains_rules(self, builder):
        prompt = builder._render_system_prompt(PromptTemplate.DEFAULT, [])
        for rule_phrase in [
            "CONTEXT ONLY",
            "UNAVAILABLE INFORMATION",
            "PREFER LATEST VERSION",
            "CONFLICTING SOPs",
            "CONFIDENCE",
            "CITATIONS",
            "NO FABRICATION",
        ]:
            assert rule_phrase in prompt, f"Rule '{rule_phrase}' missing from DEFAULT template"

    def test_concise_template_shorter(self, builder):
        default_len = len(builder._render_system_prompt(PromptTemplate.DEFAULT, []))
        concise_len = len(builder._render_system_prompt(PromptTemplate.CONCISE, []))
        assert concise_len < default_len

    def test_strict_citation_template_exists(self, builder):
        prompt = builder._render_system_prompt(PromptTemplate.STRICT_CITATION, [])
        assert "citation" in prompt.lower()

    def test_conflict_warning_injected(self, builder):
        cg = ConflictGroup(
            display_name="Fee SOP", department="Finance", category="SOP",
            versions=["2.0", "1.0"], source_numbers=[1, 2], latest_version="2.0",
        )
        prompt = builder._render_system_prompt(PromptTemplate.DEFAULT, [cg])
        assert "CONFLICT" in prompt
        assert "Fee SOP" in prompt

    def test_no_conflict_no_conflict_section(self, builder):
        prompt = builder._render_system_prompt(PromptTemplate.DEFAULT, [])
        assert "CONFLICT NOTICES" not in prompt


# ===========================================================================
# PromptBuilder.build — integration of all steps
# ===========================================================================

class TestBuild:
    def test_returns_built_prompt(self, builder):
        results = _make_results(3)
        prompt = builder.build("What is the fee payment deadline?", results)
        assert isinstance(prompt, BuiltPrompt)

    def test_messages_format(self, builder):
        results = _make_results(3)
        prompt = builder.build("What is the fee?", results)
        assert isinstance(prompt.messages, list)
        assert len(prompt.messages) == 2
        assert prompt.messages[0]["role"] == "system"
        assert prompt.messages[1]["role"] == "user"
        assert isinstance(prompt.messages[0]["content"], str)
        assert isinstance(prompt.messages[1]["content"], str)

    def test_question_in_user_message(self, builder):
        question = "What is the fee payment deadline?"
        results = _make_results(3)
        prompt = builder.build(question, results)
        assert question in prompt.user_message

    def test_context_block_in_user_message(self, builder):
        results = _make_results(2)
        prompt = builder.build("Fee?", results)
        assert "[SOURCE 1]" in prompt.user_message
        assert "[SOURCE 2]" in prompt.user_message

    def test_chunks_included_count(self, builder):
        results = _make_results(4)
        prompt = builder.build("Fee?", results)
        assert prompt.chunks_included == 4

    def test_empty_results_no_error(self, builder):
        prompt = builder.build("What is the fee?", [])
        assert prompt.chunks_included == 0
        assert "No relevant context" in prompt.context_block

    def test_empty_question_raises(self, builder):
        with pytest.raises(ValueError):
            builder.build("   ", _make_results(1))

    def test_source_citations_list_matches_chunks(self, builder):
        results = _make_results(3)
        prompt = builder.build("Fee?", results)
        assert len(prompt.source_citations) == prompt.chunks_included

    def test_has_conflicts_false_by_default(self, builder):
        results = _make_results(3)
        prompt = builder.build("Fee?", results)
        assert prompt.has_conflicts is False

    def test_has_conflicts_true_when_versions_differ(self, builder):
        # Two chunks from the same doc but different versions
        r1 = _make_retrieval_result(rank=1, version="1.0", rerank_score=0.9)
        r2 = _make_retrieval_result(rank=2, version="2.0", rerank_score=0.85)
        prompt = builder.build("Fee payment?", [r1, r2])
        assert prompt.has_conflicts is True
        assert len(prompt.conflicts) == 1

    def test_conflict_warning_in_system_prompt(self, builder):
        r1 = _make_retrieval_result(rank=1, version="1.0", rerank_score=0.9)
        r2 = _make_retrieval_result(rank=2, version="2.0", rerank_score=0.85)
        prompt = builder.build("Fee payment?", [r1, r2])
        assert "CONFLICT" in prompt.system_prompt

    def test_template_concise(self, builder):
        cfg = PromptConfig(template=PromptTemplate.CONCISE)
        results = _make_results(2)
        prompt = builder.build("Fee?", results, config=cfg)
        assert prompt.template_used == PromptTemplate.CONCISE

    def test_max_chunks_config_respected(self, builder):
        cfg = PromptConfig(max_chunks=2)
        results = _make_results(5)
        prompt = builder.build("Fee?", results, config=cfg)
        assert prompt.chunks_included == 2
        assert prompt.chunks_dropped == 3

    def test_to_flat_string_format(self, builder):
        results = _make_results(2)
        prompt = builder.build("Fee?", results)
        flat = prompt.to_flat_string()
        assert "<system>" in flat
        assert "<context>" in flat
        assert "<question>" in flat

    def test_summary_returns_string(self, builder):
        results = _make_results(2)
        prompt = builder.build("Fee?", results)
        s = prompt.summary()
        assert "Chunks" in s
        assert "Template" in s

    def test_context_chunks_preserved(self, builder):
        results = _make_results(3)
        prompt = builder.build("Fee?", results)
        assert len(prompt.context_chunks) == 3
        assert all(isinstance(c, ContextChunk) for c in prompt.context_chunks)


# ===========================================================================
# Module-level build_prompt convenience function
# ===========================================================================

class TestBuildPromptConvenience:
    def test_returns_built_prompt(self):
        results = _make_results(2)
        prompt = build_prompt("Fee payment?", results)
        assert isinstance(prompt, BuiltPrompt)

    def test_with_config(self):
        cfg = PromptConfig(template=PromptTemplate.STRICT_CITATION, max_chunks=1)
        results = _make_results(3)
        prompt = build_prompt("Admission?", results, config=cfg)
        assert prompt.chunks_included == 1
        assert prompt.template_used == PromptTemplate.STRICT_CITATION


# ===========================================================================
# BuiltPrompt.to_flat_string structure
# ===========================================================================

class TestBuiltPromptFlatString:
    def test_has_system_section(self):
        results = _make_results(1)
        prompt = build_prompt("Test?", results)
        flat = prompt.to_flat_string()
        assert flat.startswith("<system>")
        assert "</system>" in flat

    def test_has_context_section(self):
        results = _make_results(1)
        prompt = build_prompt("Test?", results)
        flat = prompt.to_flat_string()
        assert "<context>" in flat
        assert "</context>" in flat

    def test_has_question_section(self):
        results = _make_results(1)
        prompt = build_prompt("Test question?", results)
        flat = prompt.to_flat_string()
        assert "<question>" in flat
        assert "Test question?" in flat
