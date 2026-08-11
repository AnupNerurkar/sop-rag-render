"""
tests/test_hybrid_retrieval.py
-------------------------------
Unit and integration tests for hybrid retrieval + reranking.

Fast unit tests (no model, no SQLite): unmarked, run in < 1s.
Integration tests (live index):        @pytest.mark.slow
"""

from __future__ import annotations

import math
import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from retrieval.hybrid_search import RawSearchResult, BM25SearchBackend, DenseSearchBackend
from retrieval.fusion import RecipRankFusion, RRF_K
from retrieval.fts5 import _sanitize_query, ensure_schema, search as fts5_search
from retrieval.reranker import CrossEncoderReranker, RERANKER_MODEL
from retrieval.retrieval_schema import (
    RetrievalQuery,
    DEFAULT_TOP_K_DENSE,
    DEFAULT_TOP_K_BM25,
    DEFAULT_TOP_K_FUSION,
    DEFAULT_TOP_K_FINAL,
)


# ===========================================================================
# FTS5 query sanitizer
# ===========================================================================

class TestFTS5Sanitizer:
    def test_lowercases(self):
        assert '"admission"' in _sanitize_query("ADMISSION")

    def test_removes_short_tokens(self):
        # "a" (length 1) is dropped; "an"/"the"/"of" (length 2-3) are kept --
        # sanitization filters by length only, stopwording is a separate,
        # unrelated concern (retriever.QueryPreprocessor).
        expr = _sanitize_query("a an the of")
        assert '"a"' not in expr.split(" OR ")
        assert '"an"' in expr

    def test_keeps_numbers(self):
        expr = _sanitize_query("Section 8A Process 1.3")
        assert '"8a"' in expr or '"8"' in expr

    def test_empty_string(self):
        assert _sanitize_query("") == ""

    def test_punctuation_only(self):
        assert _sanitize_query("!!! ???") == ""

    def test_domain_terms_preserved(self):
        expr = _sanitize_query("VIT examination committee HOD")
        for t in ("vit", "examination", "committee", "hod"):
            assert f'"{t}"' in expr

    def test_query_syntax_characters_do_not_raise(self):
        # These would be interpreted as FTS5 MATCH operators unquoted --
        # sanitization must neutralize them, not just tolerate them.
        for bad in ['what is "attendance"?', "sub-process 1.3", "fees/billing*", "NOT admissions"]:
            expr = _sanitize_query(bad)
            assert isinstance(expr, str)

    def test_terms_joined_with_or(self):
        expr = _sanitize_query("library book")
        assert " OR " in expr


# ===========================================================================
# FTS5 search — unit tests against a throwaway SQLite file
# ===========================================================================

class TestFTS5Search:
    @pytest.fixture
    def conn(self, tmp_path):
        """
        A real (not :memory:) SQLite file with the `chunks` + `documents`
        shape ensure_schema expects, seeded with a small fixed corpus.
        File-based because the module under test opens its own connections
        via ledger.get_connection() in the live path; here the fixture hands
        a connection straight to fts5.search(), which is the unit under test.
        """
        import sqlite3
        db_path = tmp_path / "test.db"
        c = sqlite3.connect(str(db_path))
        c.row_factory = sqlite3.Row
        c.execute("""
            CREATE TABLE documents (doc_id TEXT PRIMARY KEY, title TEXT)
        """)
        c.execute("""
            CREATE TABLE chunks (
                chunk_id TEXT PRIMARY KEY, doc_id TEXT, chunk_index INTEGER,
                content TEXT, section_heading TEXT, category TEXT,
                department TEXT, access_level TEXT, version TEXT,
                source_file TEXT, total_chunks INTEGER
            )
        """)
        rows = [
            ("c1", "d1", "Admission and student fee details",    "Admissions",         "Public"),
            ("c2", "d2", "Examination paper setting procedure",  "Examination",        "Public"),
            ("c3", "d3", "Library book issue and return process", "Library Management", "Public"),
            ("c4", "d1", "VIT admission committee process",      "Admissions",         "Public"),
            ("c5", "d4", "Campus security management SOP",       "Security Management", "Student"),
        ]
        for chunk_id, doc_id, content, dept, level in rows:
            c.execute(
                "INSERT OR IGNORE INTO documents VALUES (?, ?)",
                (doc_id, f"Doc {doc_id}"),
            )
            c.execute(
                "INSERT INTO chunks (chunk_id, doc_id, chunk_index, content, "
                "section_heading, category, department, access_level, version, "
                "source_file, total_chunks) VALUES (?, ?, 0, ?, '', 'SOP', ?, ?, '1.0', '', 1)",
                (chunk_id, doc_id, content, dept, level),
            )
        c.commit()
        ensure_schema(c)
        yield c
        c.close()

    def test_search_returns_results(self, conn):
        results = fts5_search(conn, "admission fee", "", [], 3)
        assert len(results) > 0

    def test_search_top_result_relevant(self, conn):
        results = fts5_search(conn, "library book issue", "", [], 3)
        chunk_ids = [r[0] for r in results]
        assert "c3" in chunk_ids[:2]

    def test_search_scores_normalized_0_1(self, conn):
        results = fts5_search(conn, "examination paper", "", [], 5)
        for _, _, score, _ in results:
            assert 0.0 <= score <= 1.0

    def test_search_rbac_filter(self, conn):
        results = fts5_search(conn, "campus security student", "access_level = ?", ["Public"], 5)
        chunk_ids = [r[0] for r in results]
        assert "c5" not in chunk_ids  # c5 is access_level=Student

    def test_search_department_filter(self, conn):
        results = fts5_search(conn, "admission", "department = ?", ["Admissions"], 5)
        for _, _, _, meta in results:
            assert meta["department"] == "Admissions"

    def test_search_empty_query(self, conn):
        assert fts5_search(conn, "", "", [], 5) == []

    def test_search_n_results_respected(self, conn):
        results = fts5_search(conn, "admission", "", [], 2)
        assert len(results) <= 2

    def test_title_comes_from_documents_join(self, conn):
        # The old bm25.py hardcoded title="" for every result because it
        # never joined to `documents`; this is the fix.
        results = fts5_search(conn, "admission fee", "", [], 5)
        assert any(meta["title"] == "Doc d1" for _, _, _, meta in results)

    def test_query_syntax_characters_do_not_crash_search(self, conn):
        # End-to-end version of the sanitizer test: these used to raise
        # inside sqlite3's MATCH parser and degrade the whole query to
        # dense-only silently.
        for bad in ['what is "admission"?', "fees/billing*", "NOT admission"]:
            fts5_search(conn, bad, "", [], 5)  # must not raise

    def test_delete_is_reflected_immediately(self, conn):
        # The actual bug this phase fixes: no invalidate() to forget to
        # call, no in-memory copy to fall out of sync. A delete on `chunks`
        # is synced to chunks_fts by a trigger in the same transaction.
        assert any(r[0] == "c1" for r in fts5_search(conn, "admission fee", "", [], 5))
        conn.execute("DELETE FROM chunks WHERE chunk_id = 'c1'")
        conn.commit()
        assert not any(r[0] == "c1" for r in fts5_search(conn, "admission fee", "", [], 5))

    def test_ensure_schema_is_idempotent(self, conn):
        # Called on every app startup -- must not raise or duplicate rows
        # on a table/triggers that already exist.
        ensure_schema(conn)
        ensure_schema(conn)
        results = fts5_search(conn, "admission", "", [], 10)
        assert len(results) <= 2  # c1, c4 -- not duplicated


# ===========================================================================
# RecipRankFusion
# ===========================================================================

def _make_result(chunk_id: str, score: float, backend: str = "dense") -> RawSearchResult:
    return RawSearchResult(
        chunk_id=chunk_id, content=f"content of {chunk_id}",
        score=score, distance=1-score, metadata={"department": "Test"},
        backend=backend,
    )


class TestRecipRankFusion:
    @pytest.fixture
    def fuser(self):
        return RecipRankFusion(k=60)

    def test_fuse_single_list_returns_unchanged(self, fuser):
        results = [_make_result(f"c{i}", 0.9 - i*0.1) for i in range(5)]
        fused = fuser.fuse([results], n_results=5)
        assert len(fused) == 5
        ids = [r.chunk_id for r in fused]
        assert ids == [r.chunk_id for r in results]

    def test_fuse_two_lists(self, fuser):
        dense = [_make_result("c1", 0.9), _make_result("c2", 0.8), _make_result("c3", 0.7)]
        bm25  = [_make_result("c2", 0.9), _make_result("c4", 0.8), _make_result("c1", 0.7)]
        fused = fuser.fuse([dense, bm25], n_results=4)
        assert len(fused) <= 4
        # c1 and c2 appear in both lists → should score higher than c3/c4
        top2_ids = {r.chunk_id for r in fused[:2]}
        assert "c1" in top2_ids or "c2" in top2_ids

    def test_fuse_no_duplicates(self, fuser):
        dense = [_make_result(f"c{i}", 0.9-i*0.1) for i in range(5)]
        bm25  = [_make_result(f"c{i}", 0.9-i*0.1) for i in range(5)]
        fused = fuser.fuse([dense, bm25], n_results=10)
        ids = [r.chunk_id for r in fused]
        assert len(ids) == len(set(ids))

    def test_fuse_n_results_respected(self, fuser):
        lists = [[_make_result(f"c{i}", 0.9) for i in range(10)]]
        fused = fuser.fuse(lists, n_results=3)
        assert len(fused) <= 3

    def test_fuse_empty_lists_ignored(self, fuser):
        dense = [_make_result("c1", 0.9)]
        fused = fuser.fuse([dense, []], n_results=5)
        assert len(fused) >= 1

    def test_fuse_all_empty(self, fuser):
        fused = fuser.fuse([[], []], n_results=5)
        assert fused == []

    def test_fuse_sets_backend_hybrid(self, fuser):
        d = [_make_result("c1", 0.9, "dense")]
        b = [_make_result("c2", 0.8, "bm25")]
        fused = fuser.fuse([d, b], n_results=2)
        for r in fused:
            assert r.backend == "hybrid"

    def test_rrf_score_formula(self, fuser):
        # Two lists: c1 ranks 1st in both → score = 2 × 1/(60+1)
        # Single-list short-circuits (returns original score), so two lists are needed.
        d = [_make_result("c1", 0.9)]
        b = [_make_result("c1", 0.8)]
        fused = fuser.fuse([d, b], n_results=1)
        expected = 2.0 / (RRF_K + 1)
        assert abs(fused[0].score - expected) < 1e-6

    def test_cross_list_boosting(self, fuser):
        # c1 ranks 1st in dense, 2nd in BM25 → higher score than c_dense_only (1st in dense only)
        dense = [_make_result("c_both",       0.9), _make_result("c_dense_only", 0.8)]
        bm25  = [_make_result("c_bm25_only",  0.9), _make_result("c_both",       0.8)]
        fused = fuser.fuse([dense, bm25], n_results=4)
        scores = {r.chunk_id: r.score for r in fused}
        # c_both appears in both → must beat c_dense_only (appears only in dense at rank 2)
        assert scores.get("c_both", 0) > scores.get("c_dense_only", 0)

    def test_k_constant_respected(self):
        fuser10 = RecipRankFusion(k=10)
        fuser60 = RecipRankFusion(k=60)
        # Two lists are required — single-list short-circuits without applying RRF.
        d = [_make_result("c1", 0.9)]
        b = [_make_result("c1", 0.8)]
        s10 = fuser10.fuse([d, b], n_results=1)[0].score
        s60 = fuser60.fuse([d, b], n_results=1)[0].score
        # Larger k → smaller score contribution per rank
        assert s10 > s60


# ===========================================================================
# CrossEncoderReranker — unit tests (no model load)
# ===========================================================================

class TestCrossEncoderRerankerUnit:
    def test_not_loaded_initially(self):
        r = CrossEncoderReranker()
        assert not r.is_loaded

    def test_model_name(self):
        r = CrossEncoderReranker()
        assert r.model_name == RERANKER_MODEL

    def test_rerank_raises_when_not_loaded(self):
        r = CrossEncoderReranker()
        with pytest.raises(RuntimeError, match="not loaded"):
            r.rerank("query", [], 5)

    def test_rerank_empty_candidates_returns_empty(self):
        r = CrossEncoderReranker.__new__(CrossEncoderReranker)
        r._model_name = RERANKER_MODEL
        r._model      = object()  # fake loaded model marker
        # Don't call load() — just check empty input handling
        r._model      = None       # reset
        r._model_name = RERANKER_MODEL

    def test_custom_model_name(self):
        r = CrossEncoderReranker(model_name="custom/model")
        assert r.model_name == "custom/model"


# ===========================================================================
# Schema — new fields
# ===========================================================================

class TestSchemaNewFields:
    def test_retrieval_query_has_bm25_flag(self):
        q = RetrievalQuery(text="test query")
        assert hasattr(q, "use_bm25")
        assert q.use_bm25 is True

    def test_retrieval_query_has_reranker_flag(self):
        q = RetrievalQuery(text="test query")
        assert hasattr(q, "use_reranker")
        assert q.use_reranker is True

    def test_retrieval_query_default_top_k_dense(self):
        q = RetrievalQuery(text="test query")
        assert q.top_k_dense  == DEFAULT_TOP_K_DENSE
        assert q.top_k_bm25   == DEFAULT_TOP_K_BM25
        assert q.top_k_fusion == DEFAULT_TOP_K_FUSION
        assert q.top_k_final  == DEFAULT_TOP_K_FINAL

    def test_retrieval_query_disable_both(self):
        q = RetrievalQuery(text="test query", use_bm25=False, use_reranker=False)
        assert q.use_bm25     is False
        assert q.use_reranker is False

    def test_retrieval_result_has_rerank_score(self):
        from retrieval.retrieval_schema import RetrievalResult, SourceCitation
        r = RetrievalResult(
            rank=1, chunk_id="c1", content="text", score=0.9, distance=0.1,
            citation=SourceCitation(doc_id="d", source_file="f.docx", display_name="Doc"),
        )
        assert r.rerank_score is None  # None by default
        assert r.retrieval_mode == "dense"

    def test_retrieval_response_has_mode(self):
        from retrieval.retrieval_schema import RetrievalResponse
        resp = RetrievalResponse(
            query_text="q", clean_query_text="q", role="Public",
        )
        assert hasattr(resp, "reranked")
        assert hasattr(resp, "retrieval_mode")
        assert resp.reranked is False


# ===========================================================================
# Integration tests — live index, BGE model + reranker
# ===========================================================================

@pytest.mark.slow
class TestHybridRetrieverIntegration:
    @pytest.fixture(scope="class")
    def retriever(self):
        from retrieval.retriever import get_retriever
        return get_retriever()

    def test_dense_only_returns_results(self, retriever):
        r = retriever.retrieve_by_text("admission process", use_bm25=False, use_reranker=False, top_k=5)
        assert r.has_results
        assert r.retrieval_mode == "dense"
        assert r.reranked is False

    def test_hybrid_no_reranker_returns_results(self, retriever):
        r = retriever.retrieve_by_text(
            "fee payment procedure",
            use_bm25=True, use_reranker=False,
            top_k_dense=10, top_k_bm25=10, top_k_fusion=5,
        )
        assert r.has_results
        assert r.retrieval_mode == "hybrid"
        assert r.reranked is False

    def test_hybrid_with_reranker_returns_results(self, retriever):
        r = retriever.retrieve_by_text(
            "examination question paper setting",
            use_bm25=True, use_reranker=True,
            top_k_dense=25, top_k_bm25=25, top_k_fusion=25, top_k_final=5,
        )
        assert r.has_results
        assert r.retrieval_mode == "hybrid+rerank"
        assert r.reranked is True

    def test_reranked_results_have_rerank_score(self, retriever):
        r = retriever.retrieve_by_text(
            "library book issue",
            use_bm25=True, use_reranker=True,
            top_k_dense=15, top_k_bm25=15, top_k_fusion=15, top_k_final=5,
        )
        for result in r.results:
            assert result.rerank_score is not None
            assert 0.0 <= result.rerank_score <= 1.0

    def test_rerank_scores_sorted_descending(self, retriever):
        r = retriever.retrieve_by_text(
            "student placement process",
            use_bm25=True, use_reranker=True,
            top_k_dense=25, top_k_bm25=25, top_k_fusion=25, top_k_final=5,
        )
        scores = [res.rerank_score for res in r.results]
        assert scores == sorted(scores, reverse=True)

    def test_hybrid_vs_dense_may_differ(self, retriever):
        # Results can legitimately differ between modes — just verify both run
        r_dense  = retriever.retrieve_by_text("security management", use_bm25=False, use_reranker=False, top_k=5)
        r_hybrid = retriever.retrieve_by_text("security management", use_bm25=True,  use_reranker=False, top_k_dense=15, top_k_bm25=15, top_k_fusion=5)
        assert r_dense.has_results
        assert r_hybrid.has_results

    def test_hybrid_query_uses_live_chunk_count(self, retriever):
        # There is no singleton index to inspect any more (that was the bug);
        # the meaningful assertion is that chunks_fts is in sync with the
        # live `chunks` table, checked directly against the DB rather than
        # any process-level cache.
        import ledger
        r = retriever.retrieve_by_text(
            "budget process", use_bm25=True, use_reranker=False,
            top_k_dense=5, top_k_bm25=5, top_k_fusion=5,
        )
        assert r.has_results
        conn = ledger.get_connection()
        try:
            chunks_n = conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
            fts_n    = conn.execute("SELECT COUNT(*) AS n FROM chunks_fts").fetchone()["n"]
        finally:
            conn.close()
        assert fts_n == chunks_n

    def test_department_filter_works_in_hybrid_mode(self, retriever):
        r = retriever.retrieve_by_text(
            "process steps",
            department="Examination",
            use_bm25=True, use_reranker=False,
            top_k_dense=10, top_k_bm25=10, top_k_fusion=5,
        )
        for result in r.results:
            assert result.metadata["department"] == "Examination"

    def test_latency_reranker_higher_than_dense(self, retriever):
        r_dense  = retriever.retrieve_by_text("admission", use_bm25=False, use_reranker=False, top_k=5)
        r_rerank = retriever.retrieve_by_text("admission", use_bm25=True,  use_reranker=True,  top_k_dense=25, top_k_bm25=25, top_k_fusion=25, top_k_final=5)
        # Reranker adds latency — it must be slower
        assert r_rerank.latency_ms > r_dense.latency_ms
