"""
tests/test_multi_agent.py
--------------------------
Tests for the multi-agent LangGraph workflow (agents/multi_agent_graph.py).

Groups:
  TestQueryAnalyzer    — intent, follow-up, filters, ambiguity        (no LLM)
  TestPlanner          — decomposition of complex questions           (no LLM)
  TestContextValidator — dedup, relevance filter, citation quality    (no LLM)
  TestConfidence       — confidence + hallucination + reflect routing (no LLM)
  TestReflection       — query broadening / retry bookkeeping         (no LLM)
  TestFinalResponse    — memory update + trimming                     (no LLM)
  TestGraphStructure   — compile + routing helpers                    (no LLM)
  TestIntegration      — full graph + FastAPI endpoint           (mocked / slow)

Run only fast unit tests:   pytest tests/test_multi_agent.py -m "not slow"
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents import multi_agent_graph as M
from agents.agent_state import DEFAULT_AGENT_CONFIG as CFG


# ── Test doubles ────────────────────────────────────────────────────────────

class _FakeCitation:
    def __init__(self, doc_id="d1", display_name="VIT Academics", department="Academics",
                 version="1.0", page_number=1, chunk_index=0, total_chunks=5,
                 source_file="vit.docx"):
        self.doc_id = doc_id
        self.display_name = display_name
        self.department = department
        self.version = version
        self.page_number = page_number
        self.chunk_index = chunk_index
        self.total_chunks = total_chunks
        self.source_file = source_file


class _FakeResult:
    """Mimics retrieval.retrieval_schema.RetrievalResult for the pure nodes."""
    def __init__(self, chunk_id, content, score, doc_id="d1", display_name="VIT Academics",
                 rerank_score=None, distance=None):
        self.chunk_id = chunk_id
        self.content = content
        self.score = score
        self.rerank_score = rerank_score
        # response_schema.compute_confidence (Phase 4) reads distance, not
        # score/rerank_score directly. Default derives it the same way
        # _make_result does in test_rag_pipeline.py, so a caller that
        # doesn't care about confidence can still get a sane value; pass
        # distance= explicitly to control compute_confidence's output.
        self.distance = distance if distance is not None else (1.0 - score)
        self.citation = _FakeCitation(doc_id=doc_id, display_name=display_name)


# ── Query Analyzer ──────────────────────────────────────────────────────────

class TestQueryAnalyzer:
    def test_greeting_short_circuits(self):
        out = M.node_query_analyzer({"question": "hello", "memory": []})
        assert out["intent"] == "greeting"
        assert out["answer"]                      # canned answer present
        assert out["confidence"] == "HIGH"

    def test_out_of_scope(self):
        out = M.node_query_analyzer({"question": "what is the weather today?", "memory": []})
        assert out["intent"] == "out_of_scope"
        assert "VIT" in out["answer"]

    def test_rag_intent(self):
        out = M.node_query_analyzer({"question": "What is the attendance requirement?", "memory": []})
        assert out["intent"] == "rag"

    def test_filter_detection_category_dept_version(self):
        out = M.node_query_analyzer(
            {"question": "Show the examination circular policy v2.0", "memory": []}
        )
        f = out["filters"]
        assert f.get("category") in ("Circular", "Policy")  # first keyword wins
        assert f.get("department") == "Examination"
        assert f.get("version") == "2.0"

    def test_followup_resolved_from_memory(self):
        out = M.node_query_analyzer({
            "question": "tell me more about it",
            "memory": [{"role": "user", "content": "attendance policy"}],
        })
        assert out["is_followup"] is True
        assert "attendance policy" in out["effective_query"]

    def test_ambiguous_short_query(self):
        out = M.node_query_analyzer({"question": "fees", "memory": []})
        assert out["ambiguous"] is True


# ── Planner ─────────────────────────────────────────────────────────────────

class TestPlanner:
    def test_complex_decomposition(self):
        out = M.node_planner(
            {"effective_query": "What are the exam guidelines and how do I apply for a scholarship?"}
        )
        assert out["is_complex"] is True
        assert len(out["plan"]) >= 2

    def test_simple_single_step(self):
        out = M.node_planner({"effective_query": "What is the attendance requirement?"})
        assert out["is_complex"] is False
        assert out["plan"] == ["What is the attendance requirement?"]

    def test_plan_capped(self):
        q = "a policy and b policy and c policy and d policy and e policy"
        out = M.node_planner({"effective_query": q})
        assert len(out["plan"]) <= CFG.max_plan_steps


# ── Context Validator ───────────────────────────────────────────────────────

class TestContextValidator:
    def test_dedup_by_chunk_id(self):
        r = _FakeResult("c1", "same chunk text here", 0.8)
        dup = _FakeResult("c1", "same chunk text here", 0.8)
        out = M.node_context_validator({"raw_results": [r, dup]})
        assert len(out["validated_results"]) == 1

    def test_dedup_by_content(self):
        r1 = _FakeResult("c1", "identical body content", 0.8)
        r2 = _FakeResult("c2", "identical body content", 0.7)
        out = M.node_context_validator({"raw_results": [r1, r2]})
        assert len(out["validated_results"]) == 1

    def test_drops_low_score_but_keeps_at_least_one(self):
        low1 = _FakeResult("c1", "low relevance a", 0.05)
        low2 = _FakeResult("c2", "low relevance b", 0.02)
        out = M.node_context_validator({"raw_results": [low1, low2]})
        # all below floor → keep deduped set rather than empty
        assert len(out["validated_results"]) >= 1

    def test_drops_results_without_citation_name(self):
        good = _FakeResult("c1", "good chunk", 0.9)
        bad = _FakeResult("c2", "bad chunk", 0.9, display_name="")
        out = M.node_context_validator({"raw_results": [good, bad]})
        names = [getattr(r.citation, "display_name") for r in out["validated_results"]]
        assert "" not in names
        assert len(out["validated_results"]) == 1

    def test_caps_context_size(self):
        results = [_FakeResult(f"c{i}", f"chunk {i}", 0.9 - i * 0.01) for i in range(20)]
        out = M.node_context_validator({"raw_results": results})
        assert len(out["validated_results"]) <= CFG.max_context_chunks

    def test_empty_raw(self):
        out = M.node_context_validator({"raw_results": []})
        assert out["validated_results"] == []
        assert out["dropped_count"] == 0


# ── Confidence Evaluator ────────────────────────────────────────────────────

class TestConfidence:
    def test_high_confidence_finalizes(self):
        # compute_confidence (Phase 4) blends top1 with mean(top3) rather
        # than reading a single value: top1=0.90, mean_top3=(0.90+0.70)/2
        # =0.80, blend = 0.7*0.90 + 0.3*0.80 = 0.87. Chosen to land clear of
        # any float-rounding boundary rather than exactly on one.
        results = [
            _FakeResult("c1", "x", 0.90, doc_id="d1"),
            _FakeResult("c2", "y", 0.70, doc_id="d2"),
        ]
        out = M.node_confidence_evaluator(
            {"validated_results": results, "answer": "A detailed grounded answer.", "retry_count": 0}
        )
        assert out["confidence"] == "87%"
        assert out["control"] == "finalize"
        assert out["hallucination_risk"] == "low"

    def test_low_confidence_triggers_reflect(self):
        results = [_FakeResult("c1", "x", 0.2, doc_id="d1", rerank_score=0.2)]
        out = M.node_confidence_evaluator(
            {"validated_results": results, "answer": "Maybe.", "retry_count": 0}
        )
        assert out["confidence"] == "20%"
        assert out["control"] == "reflect"

    def test_no_reflect_when_retries_exhausted(self):
        results = [_FakeResult("c1", "x", 0.2, doc_id="d1", rerank_score=0.2)]
        out = M.node_confidence_evaluator(
            {"validated_results": results, "answer": "Maybe.",
             "retry_count": CFG.max_reflection_retries}
        )
        assert out["control"] == "finalize"

    def test_insufficient_context_flags_hallucination(self):
        out = M.node_confidence_evaluator(
            {"validated_results": [], "answer": "Some confident-sounding answer.", "retry_count": 5}
        )
        assert out["insufficient_context"] is True
        assert out["hallucination_risk"] == "high"

    def test_fallback_answer_does_not_reflect(self):
        from agents.agent_state import FALLBACK_ANSWER
        out = M.node_confidence_evaluator(
            {"validated_results": [], "answer": FALLBACK_ANSWER, "retry_count": 0}
        )
        assert out["control"] == "finalize"   # no point retrying a fallback


# ── Reflection ──────────────────────────────────────────────────────────────

class TestReflection:
    def test_broadens_and_bumps_retry(self):
        out = M.node_reflection_agent({
            "question": "What is the detailed attendance requirement?",
            "filters": {"department": "Examination"},
            "confidence": "LOW", "confidence_score": 0.2, "retry_count": 0,
        })
        assert out["retry_count"] == 1
        assert out["filters"] == {}                       # filters dropped
        assert out["plan"] == [out["effective_query"]]    # fresh single-step plan
        assert out["reflection_note"]


# ── Final Response ──────────────────────────────────────────────────────────

class TestFinalResponse:
    def test_appends_memory_pair(self):
        out = M.node_final_response(
            {"question": "Q?", "answer": "A.", "memory": []}
        )
        assert len(out["memory"]) == 2
        assert out["memory"][0]["role"] == "user"
        assert out["memory"][1]["role"] == "assistant"
        assert out["finished"] is True

    def test_memory_trimmed(self):
        existing = [{"role": "user", "content": f"q{i}"} for i in range(CFG.memory_max_turns)]
        out = M.node_final_response({"question": "new", "answer": "a", "memory": existing})
        assert len(out["memory"]) <= CFG.memory_max_turns


# ── Graph structure ─────────────────────────────────────────────────────────

class TestGraphStructure:
    def test_graph_compiles(self):
        M.reset_multi_agent()
        g = M.get_multi_agent()
        assert g is not None
        assert M.get_multi_agent() is g          # singleton

    def test_route_after_analyzer(self):
        assert M._route_after_analyzer({"intent": "rag"}) == "rag"
        assert M._route_after_analyzer({"intent": "greeting"}) == "direct"
        assert M._route_after_analyzer({"intent": "out_of_scope"}) == "direct"

    def test_route_after_confidence(self):
        assert M._route_after_confidence({"control": "reflect"}) == "reflect"
        assert M._route_after_confidence({"control": "finalize"}) == "finalize"
        assert M._route_after_confidence({}) == "finalize"   # safe default


# ── Integration ─────────────────────────────────────────────────────────────

class TestIntegration:
    def test_greeting_runs_full_graph_no_llm(self):
        """A greeting traverses analyzer → final_response without any retrieval/LLM."""
        out = M.run_multi_agent(question="hi", role="Public", memory=[])
        assert out["intent"] == "greeting"
        assert out["answer"]
        assert out["citations"] == []
        assert len(out["memory"]) == 2
        assert "processing_time_ms" in out

    def test_response_shape_backward_compatible(self):
        """run_multi_agent returns every key the legacy agent endpoint relies on."""
        out = M.run_multi_agent(question="hello", role="Student", memory=[])
        for key in ("answer", "answer_with_refs", "source_documents", "citations",
                    "confidence", "confidence_score", "retrieval_mode",
                    "processing_time_ms", "reflection_note", "memory"):
            assert key in out, f"missing back-compat key: {key}"

    def test_endpoint_uses_multi_agent(self, monkeypatch):
        """The FastAPI /api/agent/chat endpoint routes through the new graph."""
        fastapi = pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient
        from backend.database import Base, engine, SessionLocal, User
        from backend.auth import hash_password
        Base.metadata.create_all(bind=engine)

        db = SessionLocal()
        if not db.query(User).filter(User.username == "agent_student").first():
            db.add(User(username="agent_student",
                        hashed_password=hash_password("Student@123"), role="Student"))
            db.commit()
        db.close()

        # Patch the agent so the test is deterministic and LLM-free.
        import agents.multi_agent_graph as mag
        monkeypatch.setattr(mag, "run_multi_agent", lambda **kw: {
            "answer": "Mocked agent answer.", "answer_with_refs": "Mocked agent answer.",
            "source_documents": ["VIT Academics"], "citations": [{"display_name": "VIT Academics"}],
            "confidence": "HIGH", "confidence_score": 0.8, "retrieval_mode": "hybrid+rerank",
            "processing_time_ms": 12.0, "reflection_note": "", "memory": [],
            "intent": "rag", "hallucination_risk": "low", "insufficient_context": False,
            "retrieval_iterations": 1,
        })

        from backend.app import app
        client = TestClient(app)
        tok = client.post("/api/auth/login",
                          json={"username": "agent_student", "password": "Student@123"}).json()["access_token"]
        r = client.post("/api/agent/chat", json={"query": "What is the attendance policy?"},
                        headers={"Authorization": f"Bearer {tok}"})
        assert r.status_code == 200
        data = r.json()
        assert data["answer"] == "Mocked agent answer."
        assert data["confidence"] == "HIGH"
        assert "session_id" in data

    @pytest.mark.slow
    def test_full_rag_pipeline_live(self):
        """End-to-end with real retrieval + reranker + Ollama. Requires running stack."""
        out = M.run_multi_agent(question="What is the attendance requirement?",
                                role="Student", memory=[])
        assert out["intent"] == "rag"
        assert out["answer"] and len(out["answer"]) > 20
        assert out["retrieval_iterations"] >= 1
        assert out["confidence"] in ("High", "Medium", "Low", "Unknown")
