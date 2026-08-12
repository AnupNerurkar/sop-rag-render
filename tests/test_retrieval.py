"""
tests/test_retrieval.py
------------------------
Phase 6: Unit and integration tests for the retrieval layer.

Fast unit tests (no model, no the vector store) are unmarked — they run in < 1 second.
Integration tests that hit the live vector store are marked @pytest.mark.slow.
"""

from __future__ import annotations

import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from retrieval.retrieval_schema import (
    RetrievalFilter,
    RetrievalQuery,
    RetrievalResponse,
    RetrievalResult,
    SourceCitation,
    VALID_ROLES,
    DEFAULT_TOP_K,
    MAX_TOP_K,
)
from retrieval.filters import FilterBuilder, build_where_clause, ACCESS_HIERARCHY
from retrieval.hybrid_search import (
    RawSearchResult,
    DenseSearchBackend,
    BM25SearchBackend,
)
from retrieval.retriever import QueryPreprocessor, _build_citation


# ===========================================================================
# retrieval_schema.py
# ===========================================================================

class TestRetrievalFilter:
    def test_empty_filter_is_empty(self):
        assert RetrievalFilter().is_empty

    def test_filter_with_department_not_empty(self):
        assert not RetrievalFilter(department="Admissions").is_empty

    def test_strips_whitespace(self):
        f = RetrievalFilter(department="  Admissions  ")
        assert f.department == "Admissions"

    def test_all_fields_none_by_default(self):
        f = RetrievalFilter()
        assert f.department is None
        assert f.category   is None
        assert f.doc_id     is None
        assert f.version    is None


class TestRetrievalQuery:
    def test_valid_query(self):
        q = RetrievalQuery(text="What is the admissions process?")
        assert q.text == "What is the admissions process?"
        assert q.role == "Public"
        assert q.top_k == DEFAULT_TOP_K

    def test_too_short_text_raises(self):
        with pytest.raises(Exception):
            RetrievalQuery(text="a")

    def test_empty_text_raises(self):
        with pytest.raises(Exception):
            RetrievalQuery(text="")

    def test_unknown_role_defaults_to_public(self):
        q = RetrievalQuery(text="test query", role="Hacker")
        assert q.role == "Public"

    def test_role_normalized_to_title_case(self):
        q = RetrievalQuery(text="test query", role="faculty")
        assert q.role == "Faculty"

    def test_top_k_clamped_to_max(self):
        with pytest.raises(Exception):
            RetrievalQuery(text="test query", top_k=MAX_TOP_K + 1)

    def test_top_k_minimum_one(self):
        with pytest.raises(Exception):
            RetrievalQuery(text="test query", top_k=0)

    def test_custom_filters_accepted(self):
        q = RetrievalQuery(
            text="fee payment",
            filters=RetrievalFilter(department="Finance", category="SOP"),
        )
        assert q.filters.department == "Finance"
        assert q.filters.category   == "SOP"

    def test_remove_stopwords_default_false(self):
        q = RetrievalQuery(text="test query")
        assert q.remove_stopwords is False


class TestSourceCitation:
    @pytest.fixture
    def citation(self):
        return SourceCitation(
            doc_id          = "abc123",
            source_file     = "data/staging/1. VIT Admissions.docx",
            display_name    = "VIT Admissions",
            department      = "Admissions",
            category        = "SOP",
            version         = "1.0",
            section_heading = "Name: - Student Admission Process",
            chunk_index     = 2,
            total_chunks    = 27,
        )

    def test_inline_citation_contains_version(self, citation):
        s = citation.to_inline_citation()
        assert "v1.0" in s

    def test_inline_citation_contains_section(self, citation):
        s = citation.to_inline_citation()
        assert "Student Admission Process" in s

    def test_inline_citation_contains_chunk_position(self, citation):
        s = citation.to_inline_citation()
        assert "3/27" in s      # chunk_index is 0-based → displayed as 1-based

    def test_display_citation_contains_department_info(self, citation):
        s = citation.to_display_citation()
        assert "VIT Admissions" in s
        assert "v1.0" in s

    def test_citation_no_section_heading(self):
        c = SourceCitation(
            doc_id="x", source_file="f.docx", display_name="Doc",
            section_heading="", chunk_index=0, total_chunks=5,
        )
        s = c.to_inline_citation()
        assert "§" not in s


class TestRetrievalResult:
    @pytest.fixture
    def result(self):
        return RetrievalResult(
            rank     = 1,
            chunk_id = "abc12345",
            content  = "The admission process begins with...",
            score    = 0.87,
            distance = 0.13,
            citation = SourceCitation(
                doc_id="x", source_file="f.docx", display_name="Admissions SOP",
                section_heading="Admission Sub-Process",
            ),
        )

    def test_context_block_has_source_header(self, result):
        block = result.to_context_block()
        assert block.startswith("[SOURCE:")

    def test_context_block_contains_content(self, result):
        block = result.to_context_block()
        assert "The admission process begins with" in block


class TestRetrievalResponse:
    @pytest.fixture
    def response(self):
        return RetrievalResponse(
            query_text       = "What is the fee structure?",
            clean_query_text = "what is the fee structure?",
            role             = "Student",
            results          = [],
            total_results    = 0,
            top_k_requested  = 10,
        )

    def test_has_results_false_when_empty(self, response):
        assert not response.has_results

    def test_context_window_empty_when_no_results(self, response):
        assert response.to_context_window() == ""

    def test_summary_contains_query(self, response):
        assert "fee structure" in response.summary()


# ===========================================================================
# filters.py
# ===========================================================================

class TestAccessHierarchy:
    def test_admin_sees_all(self):
        assert set(ACCESS_HIERARCHY["Admin"]) == {"Public", "Student", "Faculty", "Admin"}

    def test_student_excludes_faculty_admin(self):
        levels = ACCESS_HIERARCHY["Student"]
        assert "Faculty" not in levels
        assert "Admin" not in levels

    def test_public_sees_only_public(self):
        assert ACCESS_HIERARCHY["Public"] == ["Public"]


class TestFilterBuilder:
    def test_public_role_single_eq(self):
        # Phase 6: doc_status="active" is now an unconditional second
        # condition alongside RBAC, so even a single-role query produces
        # $and rather than a bare condition.
        where = FilterBuilder(role="Public").build()
        assert "$and" in where
        keys = {list(c.keys())[0] for c in where["$and"]}
        assert keys == {"access_level", "doc_status"}
        for c in where["$and"]:
            if "access_level" in c:
                assert c["access_level"] == {"$eq": "Public"}
            if "doc_status" in c:
                assert c["doc_status"] == {"$eq": "active"}

    def test_admin_no_filters_still_excludes_superseded(self):
        # Previously Admin-with-no-filters returned None (unfiltered).
        # Phase 6: superseded documents are excluded for every role,
        # Admin included -- there is no longer a truly unfiltered case.
        where = FilterBuilder(role="Admin").build()
        assert where == {"doc_status": {"$eq": "active"}}

    def test_admin_with_department_returns_condition(self):
        where = FilterBuilder(
            role="Admin",
            filters=RetrievalFilter(department="Admissions")
        ).build()
        assert "$and" in where
        keys = {list(c.keys())[0] for c in where["$and"]}
        assert keys == {"doc_status", "department"}

    def test_faculty_uses_dollar_in(self):
        where = FilterBuilder(role="Faculty").build()
        access_cond = next(c["access_level"] for c in where["$and"] if "access_level" in c)
        assert "$in" in access_cond
        assert "Admin" not in access_cond["$in"]

    def test_department_filter_combined_with_rbac(self):
        where = FilterBuilder(
            role="Student",
            filters=RetrievalFilter(department="Finance")
        ).build()
        assert "$and" in where
        keys = {list(c.keys())[0] for c in where["$and"]}
        assert "access_level" in keys
        assert "department"   in keys

    def test_all_four_filters_combined(self):
        where = FilterBuilder(
            role="Faculty",
            filters=RetrievalFilter(
                department="Admissions", category="SOP",
                version="1.0", doc_id="abc" * 21 + "a"
            )
        ).build()
        assert "$and" in where
        cond_keys = {list(c.keys())[0] for c in where["$and"]}
        assert "access_level" in cond_keys
        assert "department"   in cond_keys
        assert "category"     in cond_keys
        assert "version"      in cond_keys
        assert "doc_id"       in cond_keys

    def test_describe_includes_role(self):
        desc = FilterBuilder(role="Student").describe()
        assert desc["role"] == "Student"

    def test_describe_truncates_doc_id(self):
        desc = FilterBuilder(
            role="Admin",
            filters=RetrievalFilter(doc_id="a" * 64)
        ).describe()
        assert "doc_id" in desc
        assert len(desc["doc_id"]) < 64

    def test_convenience_function(self):
        where = build_where_clause(role="Student", department="Finance")
        assert "$and" in where


# ===========================================================================
# hybrid_search.py
# ===========================================================================

class TestRawSearchResult:
    def test_defaults(self):
        r = RawSearchResult(chunk_id="c1", content="text", score=0.9, distance=0.1)
        assert r.backend == "dense"
        assert r.metadata == {}


class TestBM25SearchBackend:
    def test_backend_name(self):
        assert BM25SearchBackend().backend_name == "bm25"


# HybridSearchEngine (a second, duplicate RRF implementation, never
# instantiated anywhere in the app -- retriever.py fuses via retrieval/
# fusion.py's RecipRankFusion instead) was removed in Phase 8, along with
# its dedicated tests that lived here.


# ===========================================================================
# retriever.py — QueryPreprocessor
# ===========================================================================

class TestQueryPreprocessor:
    @pytest.fixture
    def pp(self):
        return QueryPreprocessor()

    def test_preserves_case(self, pp):
        # Phase 7: casing is no longer destroyed before embedding -- BGE is
        # cased and the corpus embeddings are built from cased text, so
        # lowercasing the query widened the query/document distance for
        # acronyms and proper nouns (MMS, VIT, NAAC) the document side
        # never lost.
        assert pp.preprocess("ADMISSION PROCESS") == "ADMISSION PROCESS"
        assert pp.preprocess("What is the MMS admission process?") == \
            "What is the MMS admission process?"

    def test_strips_whitespace(self, pp):
        assert pp.preprocess("  hello world  ") == "hello world"

    def test_collapses_multiple_spaces(self, pp):
        assert pp.preprocess("what   is   the  fee") == "what is the fee"

    def test_collapses_tabs_and_newlines(self, pp):
        assert pp.preprocess("what\tis\nthe fee") == "what is the fee"

    def test_strips_control_characters(self, pp):
        result = pp.preprocess("hello\x00world")
        assert "\x00" not in result
        assert "hello" in result and "world" in result

    def test_stopword_removal_on_long_query(self, pp):
        text = "what is the process for admissions and how does it work"
        result = pp.preprocess(text, remove_stopwords=True)
        assert "the" not in result.split()
        assert "admissions" in result

    def test_stopword_removal_preserves_short_query(self, pp):
        # Short queries (≤5 words) are NOT stopword-filtered to preserve intent
        result = pp.preprocess("what is admissions", remove_stopwords=True)
        # "what is admissions" is 3 words — stopwords kept
        assert "is" in result

    def test_empty_after_strip_returns_empty(self, pp):
        result = pp.preprocess("   ")
        assert result == ""

    def test_runon_typo_split_preserves_case(self, pp):
        # The split inserts a space into the matched text itself rather
        # than reconstructing from the (lowercased) term list, so casing
        # survives regardless of how the typo was capitalized.
        assert pp.preprocess("theattendance policy") == "the attendance policy"
        assert pp.preprocess("TheAttendance policy") == "The Attendance policy"
        assert pp.preprocess("THEMMS process") == "THE MMS process"


class TestBuildCitation:
    def test_title_used_when_present(self):
        meta = {
            "doc_id": "abc", "source_file": "data/staging/1. VIT Admissions.docx",
            "title": "Admissions SOP", "department": "Admissions",
            "category": "SOP", "version": "1.0",
            "section_heading": "", "chunk_index": 0, "total_chunks": 20,
        }
        citation = _build_citation(meta)
        assert citation.display_name == "Admissions SOP"

    def test_filename_fallback_when_title_empty(self):
        meta = {
            "doc_id": "abc", "source_file": "data/staging/1. VIT Admissions.docx",
            "title": "", "department": "Admissions",
            "category": "SOP", "version": "1.0",
            "section_heading": "", "chunk_index": 0, "total_chunks": 20,
        }
        citation = _build_citation(meta)
        # Should fall back to "VIT Admissions" (number prefix stripped)
        assert "VIT Admissions" in citation.display_name

    def test_leading_number_stripped_from_filename(self):
        meta = {
            "doc_id": "abc", "source_file": "data/staging/1. VIT Admissions.docx",
            "title": "", "department": "Admissions",
            "category": "SOP", "version": "1.0",
            "section_heading": "Sub Process 1", "chunk_index": 2, "total_chunks": 27,
        }
        citation = _build_citation(meta)
        assert not citation.display_name.startswith("1.")
        assert not citation.display_name.startswith("1 ")

    def test_section_heading_propagated(self):
        meta = {
            "doc_id": "abc", "source_file": "f.docx", "title": "",
            "department": "Finance", "category": "SOP", "version": "1.0",
            "section_heading": "Name: Fee Collection", "chunk_index": 0, "total_chunks": 5,
        }
        citation = _build_citation(meta)
        assert citation.section_heading == "Name: Fee Collection"


# ===========================================================================
# Integration tests — require a live vector store and BGE model
# ===========================================================================

@pytest.mark.slow
class TestRetrieverIntegration:
    """End-to-end retrieval tests against the live production the vector store index."""

    @pytest.fixture(scope="class")
    def retriever(self):
        from retrieval.retriever import get_retriever
        return get_retriever()

    def test_basic_retrieve_returns_results(self, retriever):
        response = retriever.retrieve_by_text("admission process")
        assert response.has_results
        assert response.total_results > 0

    def test_retrieve_returns_retrieval_response_type(self, retriever):
        from retrieval.retrieval_schema import RetrievalResponse
        response = retriever.retrieve_by_text("fee payment")
        assert isinstance(response, RetrievalResponse)

    def test_results_have_valid_scores(self, retriever):
        response = retriever.retrieve_by_text("examination procedure")
        for r in response.results:
            assert 0.0 <= r.score <= 1.0
            assert r.distance >= 0.0

    def test_results_are_ranked_by_score_descending(self, retriever):
        response = retriever.retrieve_by_text("library book issue")
        scores = [r.score for r in response.results]
        assert scores == sorted(scores, reverse=True)

    def test_results_have_citations(self, retriever):
        response = retriever.retrieve_by_text("student placement")
        for r in response.results:
            assert r.citation.doc_id
            assert r.citation.source_file
            assert r.citation.display_name

    def test_top_k_respected(self, retriever):
        response = retriever.retrieve_by_text("faculty recruitment", top_k=3)
        assert len(response.results) <= 3
        assert response.top_k_requested == 3

    def test_department_filter_applied(self, retriever):
        response = retriever.retrieve_by_text(
            "sub process",
            department="Examination",
            top_k=5,
        )
        for r in response.results:
            assert r.metadata["department"] == "Examination"

    def test_role_public_returns_only_public_chunks(self, retriever):
        response = retriever.retrieve_by_text("process", role="Public", top_k=20)
        for r in response.results:
            assert r.metadata["access_level"] == "Public"

    def test_context_window_non_empty(self, retriever):
        response = retriever.retrieve_by_text("admissions committee")
        assert len(response.to_context_window()) > 0

    def test_inline_citations_renderable(self, retriever):
        response = retriever.retrieve_by_text("fee collection process")
        for r in response.results:
            citation_str = r.citation.to_inline_citation()
            assert citation_str.startswith("[")
            assert citation_str.endswith("]")

    def test_latency_recorded(self, retriever):
        response = retriever.retrieve_by_text("security management")
        assert response.latency_ms > 0

    def test_applied_filters_echoed(self, retriever):
        response = retriever.retrieve_by_text(
            "budget approval", role="Admin", department="Budgeting"
        )
        assert "department" in response.applied_filters

    def test_retrieve_by_query_object(self, retriever):
        query = RetrievalQuery(
            text    = "student activities and clubs",
            role    = "Student",
            top_k   = 5,
            filters = RetrievalFilter(department="Student Activities"),
        )
        response = retriever.retrieve(query)
        assert isinstance(response, RetrievalResponse)
        for r in response.results:
            assert r.metadata["department"] == "Student Activities"
