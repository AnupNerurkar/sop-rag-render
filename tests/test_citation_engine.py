"""
tests/test_citation_engine.py
-------------------------------
Unit tests for rag/citation_engine.py and rag/citation_schema.py.

All tests are offline — no Ollama, the vector store, or model load.
Run: pytest tests/test_citation_engine.py -v
"""

from __future__ import annotations

import json
import sys
import os
import pytest
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.citation_schema import Citation, CitationList
from rag.citation_engine import (
    CitationEngine,
    _inject_inline_refs,
    _extract_display_name,
    _version_key,
    get_citation_engine,
    reset_citation_engine,
)
from rag.prompt_schema import ContextChunk


# ===========================================================================
# Fixtures / helpers
# ===========================================================================

@pytest.fixture(autouse=True)
def _reset():
    reset_citation_engine()
    yield
    reset_citation_engine()


def _make_source_citation(
    doc_id:          str   = "abc123def456" * 5,
    display_name:    str   = "Fee Payment SOP",
    department:      str   = "Finance",
    category:        str   = "SOP",
    version:         str   = "2.0",
    source_file:     str   = "docs/Fee_Payment_SOP.docx",
    section_heading: str   = "Fee Deadlines",
    chunk_index:     int   = 2,
    total_chunks:    int   = 27,
):
    return SimpleNamespace(
        doc_id          = doc_id,
        display_name    = display_name,
        department      = department,
        category        = category,
        version         = version,
        source_file     = source_file,
        section_heading = section_heading,
        chunk_index     = chunk_index,
        total_chunks    = total_chunks,
        to_inline_citation  = lambda: f"[{display_name} v{version}]",
        to_display_citation = lambda: f"{display_name} (v{version}) — {section_heading} — chunk {chunk_index+1} of {total_chunks}",
    )


def _make_result(
    rank:            int   = 1,
    chunk_id:        str   = "chunk_01",
    score:           float = 0.90,
    rerank_score:    Optional[float] = 0.92,
    doc_id:          str   = "abc123def456" * 5,
    display_name:    str   = "Fee Payment SOP",
    department:      str   = "Finance",
    category:        str   = "SOP",
    version:         str   = "2.0",
    source_file:     str   = "docs/Fee_Payment_SOP.docx",
    section_heading: str   = "Fee Deadlines",
    chunk_index:     int   = 2,
    total_chunks:    int   = 27,
    access_level:    str   = "Public",
):
    from typing import Optional as _Opt
    sc = _make_source_citation(
        doc_id=doc_id, display_name=display_name, department=department,
        category=category, version=version, source_file=source_file,
        section_heading=section_heading, chunk_index=chunk_index,
        total_chunks=total_chunks,
    )
    return SimpleNamespace(
        rank         = rank,
        chunk_id     = chunk_id,
        score        = score,
        rerank_score = rerank_score,
        citation     = sc,
        metadata     = {"access_level": access_level},
    )


from typing import Optional


def _make_context_chunk(
    source_number:   int   = 1,
    chunk_id:        str   = "chunk_01",
    doc_id:          str   = "docabc123",
    display_name:    str   = "Admissions SOP",
    department:      str   = "Admissions",
    category:        str   = "SOP",
    version:         str   = "1.0",
    section_heading: str   = "Admission Process",
    rank:            int   = 1,
    score:           float = 0.85,
    rerank_score:    Optional[float] = 0.88,
) -> ContextChunk:
    return ContextChunk(
        source_number    = source_number,
        chunk_id         = chunk_id,
        content          = "Sample content for testing.",
        inline_citation  = f"[{display_name} v{version} § {section_heading}]",
        display_citation = f"{display_name} (v{version}) — {section_heading}",
        department       = department,
        category         = category,
        version          = version,
        doc_id           = doc_id,
        section_heading  = section_heading,
        rank             = rank,
        score            = score,
        rerank_score     = rerank_score,
    )


# ===========================================================================
# _version_key
# ===========================================================================

class TestVersionKey:
    def test_numeric_ascending(self):
        assert _version_key("2.0") > _version_key("1.0")
        assert _version_key("3.0") > _version_key("2.5")

    def test_non_numeric_sorts_last(self):
        assert _version_key("Final") <= _version_key("1.0")

    def test_same_version_equal(self):
        assert _version_key("1.5") == _version_key("1.5")

    def test_multipart(self):
        assert _version_key("2.1") > _version_key("2.0")


# ===========================================================================
# _inject_inline_refs
# ===========================================================================

class TestInjectInlineRefs:
    def test_replaces_single_ref(self):
        answer = "See [SOURCE 1] for details."
        result = _inject_inline_refs(answer, {1: 1})
        assert result == "See [1] for details."

    def test_replaces_multiple_refs(self):
        answer = "[SOURCE 1] and [SOURCE 2] agree."
        result = _inject_inline_refs(answer, {1: 1, 2: 2})
        assert result == "[1] and [2] agree."

    def test_remaps_merged_refs(self):
        # [SOURCE 1] and [SOURCE 3] both came from the same doc → rank 1
        answer = "[SOURCE 1] also confirmed by [SOURCE 3]."
        result = _inject_inline_refs(answer, {1: 1, 3: 1})
        assert result == "[1] also confirmed by [1]."

    def test_unknown_source_stripped(self):
        # Phase 5: an unresolved marker (referencing a source that was
        # never shown to the model) is stripped entirely rather than left
        # as raw "[SOURCE 99]" text leaking into what the user reads.
        answer = "Check [SOURCE 99] for more."
        result = _inject_inline_refs(answer, {1: 1})
        assert "[SOURCE 99]" not in result
        assert "Check  for more." == result

    def test_case_insensitive(self):
        answer = "See [source 1] and [Source 2]."
        result = _inject_inline_refs(answer, {1: 1, 2: 2})
        assert result == "See [1] and [2]."

    def test_no_refs_unchanged(self):
        answer = "The fee is due on the 15th."
        result = _inject_inline_refs(answer, {1: 1})
        assert result == answer

    def test_empty_answer(self):
        assert _inject_inline_refs("", {}) == ""

    def test_bundled_marker_resolved(self):
        # Caught live: the model doesn't always emit one marker per
        # citation -- it sometimes bundles several into a single bracket.
        # The original single-number pattern didn't recognize this form at
        # all, so it passed through completely unresolved.
        answer = "...late returns [SOURCE 1, SOURCE 2, SOURCE 3]."
        result = _inject_inline_refs(answer, {1: 1, 2: 2, 3: 1})
        # ranks 1 and 2 (deduped, 3 also maps to rank 1)
        assert result == "...late returns [1, 2]."

    def test_bundled_marker_partially_hallucinated(self):
        # One number in the bundle resolves, one doesn't -- keep the real
        # one, drop the fake one, not the whole bracket.
        answer = "[SOURCE 1, SOURCE 99] confirm this."
        result = _inject_inline_refs(answer, {1: 1})
        assert result == "[1] confirm this."

    def test_bundled_marker_all_hallucinated_stripped(self):
        answer = "[SOURCE 98, SOURCE 99] confirm this."
        result = _inject_inline_refs(answer, {1: 1})
        assert result == " confirm this."


# ===========================================================================
# _extract_display_name
# ===========================================================================

class TestExtractDisplayName:
    def test_extracts_from_standard_format(self):
        assert _extract_display_name("Fee SOP (v2.0) — Section") == "Fee SOP"

    def test_handles_no_version(self):
        name = _extract_display_name("Fee SOP")
        assert "Fee SOP" in name

    def test_does_not_include_version(self):
        name = _extract_display_name("Admissions SOP (v1.5) — Process — chunk 1 of 10")
        assert "v1.5" not in name
        assert "Admissions SOP" in name


# ===========================================================================
# Citation.to_markdown_entry
# ===========================================================================

class TestCitationMarkdownEntry:
    @pytest.fixture
    def citation(self):
        return Citation(
            citation_id      = "cite_abc123",
            rank             = 1,
            inline_ref       = "[1]",
            doc_id           = "abc123",
            display_name     = "Fee Payment SOP",
            department       = "Finance",
            category         = "SOP",
            version          = "2.0",
            source_file      = "docs/Fee.docx",
            section_heading  = "Fee Deadlines",
            chunk_ids        = ["c01"],
            chunk_index      = 2,
            total_chunks     = 27,
            page_number      = 3,
            score            = 0.92,
            rerank_score     = 0.93,
            access_level     = "Public",
            is_latest_version = True,
        )

    def test_contains_inline_ref(self, citation):
        assert "[1]" in citation.to_markdown_entry()

    def test_contains_display_name(self, citation):
        assert "Fee Payment SOP" in citation.to_markdown_entry()

    def test_contains_version(self, citation):
        assert "v2.0" in citation.to_markdown_entry()

    def test_contains_score(self, citation):
        entry = citation.to_markdown_entry()
        assert "0.930" in entry

    def test_contains_section_heading(self, citation):
        assert "Fee Deadlines" in citation.to_markdown_entry()

    def test_superseded_marked(self, citation):
        old = citation.model_copy(update={"is_latest_version": False})
        assert "older version" in old.to_markdown_entry()

    def test_latest_not_marked(self, citation):
        assert "older version" not in citation.to_markdown_entry()


# ===========================================================================
# Citation.to_dict_entry
# ===========================================================================

class TestCitationDictEntry:
    def test_has_required_keys(self):
        c = Citation(
            citation_id="cite_x", rank=1, inline_ref="[1]",
            doc_id="d1", display_name="Test SOP", score=0.9,
        )
        d = c.to_dict_entry()
        for key in ("rank", "inline_ref", "doc_id", "display_name", "version",
                    "department", "category", "score", "access_level", "chunk_ids",
                    "page_number", "is_latest_version"):
            assert key in d, f"Missing key: {key}"

    def test_score_rounded(self):
        c = Citation(
            citation_id="cite_x", rank=1, inline_ref="[1]",
            doc_id="d1", display_name="Test SOP", score=0.912345,
        )
        assert len(str(c.to_dict_entry()["score"]).split(".")[-1]) <= 4

    def test_json_serialisable(self):
        c = Citation(
            citation_id="cite_x", rank=1, inline_ref="[1]",
            doc_id="d1", display_name="Test SOP", score=0.9,
        )
        json.dumps(c.to_dict_entry())  # must not raise


# ===========================================================================
# CitationEngine.build — core logic
# ===========================================================================

class TestCitationEngineBuild:
    @pytest.fixture
    def engine(self):
        return CitationEngine()

    def test_empty_results_returns_empty_list(self, engine):
        cl = engine.build([], "No answer available.")
        assert cl.citations == []
        assert cl.total_citations == 0

    def test_single_result(self, engine):
        r  = _make_result()
        cl = engine.build([r], "The fee deadline is [SOURCE 1].")
        assert cl.total_citations == 1
        assert cl.citations[0].rank == 1
        assert cl.citations[0].inline_ref == "[1]"

    def test_returns_citation_list(self, engine):
        from rag.citation_schema import CitationList
        cl = engine.build([_make_result()], "answer")
        assert isinstance(cl, CitationList)

    def test_answer_with_refs_populated(self, engine):
        r  = _make_result()
        cl = engine.build([r], "The deadline is [SOURCE 1].")
        assert "[1]" in cl.answer_with_refs
        assert "[SOURCE 1]" not in cl.answer_with_refs

    def test_original_answer_preserved(self, engine):
        r       = _make_result()
        answer  = "Original [SOURCE 1] answer."
        cl      = engine.build([r], answer)
        assert cl.original_answer == answer

    def test_citation_fields_populated(self, engine):
        r  = _make_result(
            display_name="Fee SOP", department="Finance",
            category="SOP", version="2.0", access_level="Student",
            chunk_index=3, total_chunks=20, score=0.88, rerank_score=0.91,
        )
        cl = engine.build([r], "answer")
        c  = cl.citations[0]
        assert c.display_name  == "Fee SOP"
        assert c.department    == "Finance"
        assert c.version       == "2.0"
        assert c.access_level  == "Student"
        assert c.total_chunks  == 20
        assert c.page_number   == 4    # chunk_index=3 → page 4
        assert c.score         == pytest.approx(0.91, abs=1e-4)  # rerank_score preferred


# ===========================================================================
# Merge duplicates
# ===========================================================================

class TestMergeDuplicates:
    @pytest.fixture
    def engine(self):
        return CitationEngine()

    def test_two_chunks_same_doc_merge_to_one(self, engine):
        r1 = _make_result(rank=1, chunk_id="c01", score=0.90, rerank_score=0.92, chunk_index=0)
        r2 = _make_result(rank=2, chunk_id="c02", score=0.80, rerank_score=0.82, chunk_index=5)
        cl = engine.build([r1, r2], "answer")
        assert cl.total_citations == 1
        c = cl.citations[0]
        assert "c01" in c.chunk_ids
        assert "c02" in c.chunk_ids

    def test_merged_score_is_best(self, engine):
        r1 = _make_result(chunk_id="c01", score=0.70, rerank_score=0.75, chunk_index=0)
        r2 = _make_result(chunk_id="c02", score=0.90, rerank_score=0.95, chunk_index=3)
        cl = engine.build([r1, r2], "answer")
        assert cl.citations[0].score == pytest.approx(0.95, abs=1e-4)

    def test_merged_chunk_index_is_earliest(self, engine):
        r1 = _make_result(chunk_id="c01", chunk_index=5)
        r2 = _make_result(chunk_id="c02", chunk_index=2)
        cl = engine.build([r1, r2], "answer")
        assert cl.citations[0].chunk_index == 2

    def test_different_docs_not_merged(self, engine):
        r1 = _make_result(chunk_id="c01", doc_id="doc_aaa" * 10)
        r2 = _make_result(chunk_id="c02", doc_id="doc_bbb" * 10, display_name="Exam SOP",
                          score=0.85, rerank_score=0.88, department="Examination")
        cl = engine.build([r1, r2], "answer")
        assert cl.total_citations == 2

    def test_source_map_maps_both_chunks_to_same_rank(self, engine):
        r1 = _make_result(rank=1, chunk_id="c01", chunk_index=0)
        r2 = _make_result(rank=2, chunk_id="c02", chunk_index=3)
        cl = engine.build([r1, r2], "[SOURCE 1] [SOURCE 2]")
        # Both [SOURCE 1] and [SOURCE 2] → same doc → same citation [1]
        assert cl.answer_with_refs == "[1] [1]"


# ===========================================================================
# Sort and determinism
# ===========================================================================

class TestSortAndDeterminism:
    @pytest.fixture
    def engine(self):
        return CitationEngine()

    def test_sorted_by_score_desc(self, engine):
        r1 = _make_result(chunk_id="c01", doc_id="doc_aaa" * 10,
                          score=0.70, rerank_score=0.72, display_name="A Doc", department="Dept A")
        r2 = _make_result(chunk_id="c02", doc_id="doc_bbb" * 10,
                          score=0.90, rerank_score=0.92, display_name="B Doc", department="Dept B")
        cl = engine.build([r1, r2], "answer")
        assert cl.citations[0].score > cl.citations[1].score

    def test_same_score_sorted_alphabetically(self, engine):
        r1 = _make_result(chunk_id="c01", doc_id="doc_aaa" * 10,
                          score=0.90, rerank_score=None, display_name="Zebra SOP", department="Z")
        r2 = _make_result(chunk_id="c02", doc_id="doc_bbb" * 10,
                          score=0.90, rerank_score=None, display_name="Alpha SOP", department="A")
        cl = engine.build([r1, r2], "answer")
        assert cl.citations[0].display_name == "Alpha SOP"
        assert cl.citations[1].display_name == "Zebra SOP"

    def test_deterministic_same_input_same_output(self, engine):
        results = [
            _make_result(chunk_id=f"c{i:02d}", doc_id=f"doc_{chr(65+i)}" * 10,
                         score=0.9 - i * 0.1, rerank_score=None,
                         display_name=f"Doc {chr(65+i)}", department=f"Dept {i}")
            for i in range(4)
        ]
        cl1 = engine.build(results, "answer")
        cl2 = engine.build(results, "answer")
        assert [c.rank for c in cl1.citations] == [c.rank for c in cl2.citations]
        assert [c.display_name for c in cl1.citations] == [c.display_name for c in cl2.citations]

    def test_ranks_are_sequential(self, engine):
        results = [
            _make_result(chunk_id=f"c{i:02d}", doc_id=f"doc_{chr(65+i)}" * 10,
                         score=0.9 - i * 0.1, rerank_score=None,
                         display_name=f"Doc {chr(65+i)}", department=f"D{i}")
            for i in range(3)
        ]
        # "answer" has no [SOURCE N] markers -> fallback mode, capped at
        # CITATION_FALLBACK_TOP_N (2) rather than citing all 3 retrieved
        # documents regardless of whether the answer used them.
        cl = engine.build(results, "answer")
        assert [c.rank for c in cl.citations] == [1, 2]
        assert cl.citations_inferred is True


# ===========================================================================
# Version conflict resolution
# ===========================================================================

class TestVersionConflicts:
    @pytest.fixture
    def engine(self):
        return CitationEngine()

    def test_same_sop_different_versions_both_present(self, engine):
        r1 = _make_result(chunk_id="c01", doc_id="doc_v1" * 10, version="1.0",
                          score=0.85, rerank_score=None, display_name="Fee SOP")
        r2 = _make_result(chunk_id="c02", doc_id="doc_v2" * 10, version="2.0",
                          score=0.80, rerank_score=None, display_name="Fee SOP")
        cl = engine.build([r1, r2], "answer")
        assert cl.total_citations == 2
        assert cl.has_version_conflicts is True

    def test_latest_version_is_latest(self, engine):
        r1 = _make_result(chunk_id="c01", doc_id="doc_v1" * 10, version="1.0",
                          score=0.85, rerank_score=None, display_name="Fee SOP")
        r2 = _make_result(chunk_id="c02", doc_id="doc_v2" * 10, version="2.0",
                          score=0.80, rerank_score=None, display_name="Fee SOP")
        cl = engine.build([r1, r2], "answer")
        latest = [c for c in cl.citations if c.is_latest_version]
        assert len(latest) == 1
        assert latest[0].version == "2.0"

    def test_older_version_marked_not_latest(self, engine):
        r1 = _make_result(chunk_id="c01", doc_id="doc_v1" * 10, version="1.0",
                          score=0.85, rerank_score=None, display_name="Fee SOP")
        r2 = _make_result(chunk_id="c02", doc_id="doc_v2" * 10, version="2.0",
                          score=0.80, rerank_score=None, display_name="Fee SOP")
        cl = engine.build([r1, r2], "answer")
        old = [c for c in cl.citations if not c.is_latest_version]
        assert len(old) == 1
        assert old[0].version == "1.0"

    def test_no_conflict_when_single_version(self, engine):
        r1 = _make_result(chunk_id="c01", doc_id="doc_a" * 10, display_name="Fee SOP")
        cl = engine.build([r1], "answer")
        assert cl.has_version_conflicts is False
        assert cl.citations[0].is_latest_version is True

    def test_different_docs_no_conflict(self, engine):
        r1 = _make_result(chunk_id="c01", doc_id="doc_a" * 10, display_name="Fee SOP")
        r2 = _make_result(chunk_id="c02", doc_id="doc_b" * 10, display_name="Exam SOP",
                          score=0.80, rerank_score=None, department="Examination")
        cl = engine.build([r1, r2], "answer")
        assert cl.has_version_conflicts is False


# ===========================================================================
# CitationEngine.build_from_chunks
# ===========================================================================

class TestBuildFromChunks:
    @pytest.fixture
    def engine(self):
        return CitationEngine()

    def test_single_chunk(self, engine):
        chunk = _make_context_chunk()
        cl    = engine.build_from_chunks([chunk], "The answer is [SOURCE 1].")
        assert cl.total_citations == 1
        assert "[1]" in cl.answer_with_refs

    def test_merges_same_doc(self, engine):
        c1 = _make_context_chunk(source_number=1, chunk_id="c01", doc_id="same_doc", rank=1)
        c2 = _make_context_chunk(source_number=2, chunk_id="c02", doc_id="same_doc", rank=2,
                                 score=0.80, rerank_score=0.82)
        cl = engine.build_from_chunks([c1, c2], "answer")
        assert cl.total_citations == 1
        assert set(cl.citations[0].chunk_ids) == {"c01", "c02"}

    def test_two_different_docs(self, engine):
        c1 = _make_context_chunk(source_number=1, doc_id="doc_a", display_name="Fee SOP")
        c2 = _make_context_chunk(source_number=2, doc_id="doc_b", display_name="Exam SOP",
                                 rank=2, score=0.80, rerank_score=0.82)
        cl = engine.build_from_chunks([c1, c2], "answer")
        assert cl.total_citations == 2

    def test_empty_chunks_returns_empty(self, engine):
        cl = engine.build_from_chunks([], "answer")
        assert cl.citations == []

    def test_version_conflict_detected(self, engine):
        c1 = _make_context_chunk(source_number=1, doc_id="doc_v1", version="1.0",
                                 display_name="Fee SOP")
        c2 = _make_context_chunk(source_number=2, doc_id="doc_v2", version="2.0",
                                 display_name="Fee SOP", rank=2, score=0.80, rerank_score=0.82)
        cl = engine.build_from_chunks([c1, c2], "answer")
        assert cl.has_version_conflicts is True


# ===========================================================================
# CitationList.to_markdown
# ===========================================================================

class TestCitationListMarkdown:
    @pytest.fixture
    def citation_list(self):
        engine = CitationEngine()
        r1     = _make_result(chunk_id="c01", display_name="Fee SOP", section_heading="Deadlines")
        r2     = _make_result(chunk_id="c02", doc_id="doc_bbb" * 10, display_name="Exam SOP",
                              score=0.80, rerank_score=0.82, department="Examination",
                              section_heading="Question Setting")
        return engine.build([r1, r2], "See [SOURCE 1] and [SOURCE 2].")

    def test_contains_references_header(self, citation_list):
        md = citation_list.to_markdown()
        assert "**References**" in md

    def test_contains_both_source_inline_refs(self, citation_list):
        md = citation_list.to_markdown()
        assert "[1]" in md
        assert "[2]" in md

    def test_answer_at_top(self, citation_list):
        md = citation_list.to_markdown()
        assert md.startswith("See [1] and [2].")

    def test_separator_present(self, citation_list):
        md = citation_list.to_markdown()
        assert "---" in md

    def test_conflict_note_absent_without_conflict(self, citation_list):
        assert not citation_list.has_version_conflicts
        assert "⚠" not in citation_list.to_markdown()

    def test_conflict_note_present_when_versions_differ(self):
        engine = CitationEngine()
        r1 = _make_result(chunk_id="c01", doc_id="doc_v1" * 10, version="1.0",
                          score=0.90, rerank_score=None, display_name="Fee SOP")
        r2 = _make_result(chunk_id="c02", doc_id="doc_v2" * 10, version="2.0",
                          score=0.85, rerank_score=None, display_name="Fee SOP")
        cl = engine.build([r1, r2], "answer")
        assert "⚠" in cl.to_markdown()


# ===========================================================================
# CitationList.to_json
# ===========================================================================

class TestCitationListJSON:
    @pytest.fixture
    def citation_list(self):
        engine = CitationEngine()
        r      = _make_result()
        return engine.build([r], "Answer [SOURCE 1].")

    def test_valid_json(self, citation_list):
        j = citation_list.to_json()
        parsed = json.loads(j)
        assert isinstance(parsed, dict)

    def test_has_answer_key(self, citation_list):
        parsed = json.loads(citation_list.to_json())
        assert "answer" in parsed
        assert "[1]" in parsed["answer"]

    def test_has_citations_key(self, citation_list):
        parsed = json.loads(citation_list.to_json())
        assert "citations" in parsed
        assert len(parsed["citations"]) == 1

    def test_citation_has_required_fields(self, citation_list):
        parsed = json.loads(citation_list.to_json())
        c = parsed["citations"][0]
        for field in ("rank", "inline_ref", "doc_id", "display_name", "version",
                      "score", "access_level", "page_number"):
            assert field in c, f"Missing field: {field}"

    def test_total_citations_in_json(self, citation_list):
        parsed = json.loads(citation_list.to_json())
        assert "total_citations" in parsed


# ===========================================================================
# CitationList.get_by_rank / get_by_source_number
# ===========================================================================

class TestCitationListLookup:
    @pytest.fixture
    def citation_list(self):
        engine = CitationEngine()
        r1     = _make_result(chunk_id="c01")
        r2     = _make_result(chunk_id="c02", doc_id="doc_bbb" * 10, display_name="Exam SOP",
                              score=0.80, rerank_score=0.82, department="Examination")
        return engine.build([r1, r2], "[SOURCE 1] and [SOURCE 2].")

    def test_get_by_rank_1(self, citation_list):
        c = citation_list.get_by_rank(1)
        assert c is not None
        assert c.rank == 1

    def test_get_by_rank_2(self, citation_list):
        c = citation_list.get_by_rank(2)
        assert c is not None
        assert c.rank == 2

    def test_get_by_rank_out_of_range(self, citation_list):
        assert citation_list.get_by_rank(99) is None

    def test_get_by_source_number(self, citation_list):
        c = citation_list.get_by_source_number(1)
        assert c is not None

    def test_get_by_unknown_source_number(self, citation_list):
        assert citation_list.get_by_source_number(99) is None


# ===========================================================================
# citation_id determinism
# ===========================================================================

class TestCitationIdDeterminism:
    def test_same_doc_id_same_citation_id(self):
        engine = CitationEngine()
        doc_id = "aabbccdd1122" * 5
        r1 = _make_result(chunk_id="c01", doc_id=doc_id)
        cl = engine.build([r1], "answer")
        assert cl.citations[0].citation_id == f"cite_{doc_id[:12]}"

    def test_different_doc_ids_different_citation_ids(self):
        engine = CitationEngine()
        r1 = _make_result(chunk_id="c01", doc_id="doc_aaaa" * 10)
        r2 = _make_result(chunk_id="c02", doc_id="doc_bbbb" * 10,
                          display_name="Other SOP", score=0.80, rerank_score=None,
                          department="Other")
        cl = engine.build([r1, r2], "answer")
        ids = [c.citation_id for c in cl.citations]
        assert len(set(ids)) == 2


# ===========================================================================
# Singleton
# ===========================================================================

class TestSingleton:
    def test_same_instance(self):
        a = get_citation_engine()
        b = get_citation_engine()
        assert a is b

    def test_reset_creates_new(self):
        a = get_citation_engine()
        reset_citation_engine()
        b = get_citation_engine()
        assert a is not b
