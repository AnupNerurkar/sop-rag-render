"""
tests/test_integration.py
--------------------------
Integration tests for the FastAPI backend.

Tests are grouped:
  TestAuth          — login, token, RBAC
  TestChatEndpoints — public + authenticated chat (mocked RAG)
  TestAdminAPI      — stats, documents, logs (requires DB)
  TestDocUpload     — upload validation logic
  TestLangGraphAgent — agent nodes unit tests (no LLM)

Slow markers:
  @pytest.mark.slow — skip with: pytest -m "not slow"
"""

import json
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def app():
    """Create the FastAPI app with a clean test database."""
    import os, sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    # Use a test-specific SQLite DB
    os.environ["TEST_DB"] = "1"
    from backend.database import Base, engine
    Base.metadata.create_all(bind=engine)

    from backend.app import app as _app
    return _app


@pytest.fixture(scope="session")
def client(app):
    return TestClient(app)


@pytest.fixture(scope="session")
def seeded_db(app):
    """Ensure test users exist in the test DB."""
    from backend.database import SessionLocal, User
    from backend.auth import hash_password
    db = SessionLocal()
    if not db.query(User).filter(User.username == "admin_test").first():
        users = [
            ("admin_test",   "Admin@123",   "Admin"),
            ("student_test", "Student@123", "Student"),
            ("faculty_test", "Faculty@123", "Faculty"),
            ("public_user",  "Public@123",  "Public"),
        ]
        for username, password, role in users:
            db.add(User(
                username=username,
                hashed_password=hash_password(password),
                role=role,
            ))
        db.commit()
    db.close()


@pytest.fixture
def admin_token(client, seeded_db):
    r = client.post("/api/auth/login", json={"username": "admin_test", "password": "Admin@123"})
    return r.json()["access_token"]


@pytest.fixture
def student_token(client, seeded_db):
    r = client.post("/api/auth/login", json={"username": "student_test", "password": "Student@123"})
    return r.json()["access_token"]


# ── Auth tests ────────────────────────────────────────────────────────────────

class TestAuth:
    def test_login_success_admin(self, client, seeded_db):
        r = client.post("/api/auth/login", json={"username": "admin_test", "password": "Admin@123"})
        assert r.status_code == 200
        data = r.json()
        assert "access_token" in data
        assert data["role"] == "Admin"
        assert data["username"] == "admin_test"

    def test_login_success_student(self, client, seeded_db):
        r = client.post("/api/auth/login", json={"username": "student_test", "password": "Student@123"})
        assert r.status_code == 200
        assert r.json()["role"] == "Student"

    def test_login_wrong_password(self, client, seeded_db):
        r = client.post("/api/auth/login", json={"username": "admin_test", "password": "wrong"})
        assert r.status_code == 401

    def test_login_unknown_user(self, client, seeded_db):
        r = client.post("/api/auth/login", json={"username": "nobody", "password": "x"})
        assert r.status_code == 401

    def test_protected_endpoint_no_token(self, client, seeded_db):
        r = client.get("/api/users")
        assert r.status_code in (401, 403)  # HTTPBearer returns 401 when no credentials

    def test_protected_endpoint_invalid_token(self, client, seeded_db):
        r = client.get("/api/users", headers={"Authorization": "Bearer invalid.jwt.token"})
        assert r.status_code == 401


# ── Chat endpoint tests ────────────────────────────────────────────────────────

_MOCK_RAG_RESULT = {
    "answer":             "Minimum 75% attendance is required.",
    "answer_with_refs":   "Minimum 75% attendance is required. [SOURCE 1]",
    "formatted_answer":   "Minimum 75% attendance is required.",
    "source_documents":   ["VIT Academics"],
    "citations":          [{"display_name": "VIT Academics", "version": "1.0", "score": 0.72}],
    "confidence":         "HIGH",
    "confidence_score":   0.72,
    "retrieval_mode":     "hybrid+rerank",
    "processing_time_ms": 1500.0,
    "retrieval_time_ms":  300.0,
    "generation_time_ms": 1200.0,
    "chunks_in_context":  5,
    "has_conflicts":      False,
    "model_name":         "qwen2.5:7b",
    "template_used":      "default",
}


class TestChatEndpoints:
    @patch("backend.rag_integration.query", return_value=_MOCK_RAG_RESULT)
    def test_public_chat(self, mock_query, client):
        r = client.post("/api/chat", json={"query": "What is the attendance requirement?"})
        assert r.status_code == 200
        data = r.json()
        assert "answer" in data
        assert "session_id" in data
        assert data["confidence"] == "HIGH"
        assert data["source_documents"] == ["VIT Academics"]
        mock_query.assert_called_once_with(question="What is the attendance requirement?", role="Public")

    @patch("backend.rag_integration.query", return_value=_MOCK_RAG_RESULT)
    def test_auth_chat(self, mock_query, client, student_token):
        r = client.post(
            "/api/chat/auth",
            json={"query": "What is the attendance requirement?"},
            headers={"Authorization": f"Bearer {student_token}"},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["answer"] == "Minimum 75% attendance is required."
        # RAG was called with Student role
        mock_query.assert_called_once_with(
            question="What is the attendance requirement?", role="Student"
        )

    @patch("backend.rag_integration.query", return_value=_MOCK_RAG_RESULT)
    def test_auth_chat_saves_session_id(self, mock_query, client, student_token):
        sid = "test-session-123"
        r = client.post(
            "/api/chat/auth",
            json={"query": "Test?", "session_id": sid},
            headers={"Authorization": f"Bearer {student_token}"},
        )
        assert r.status_code == 200
        assert r.json()["session_id"] == sid

    def test_auth_chat_requires_token(self, client):
        r = client.post("/api/chat/auth", json={"query": "Test"})
        assert r.status_code in (401, 403)


# ── Admin API tests ───────────────────────────────────────────────────────────

class TestAdminAPI:
    def test_stats_requires_admin(self, client, student_token):
        r = client.get("/api/admin/stats", headers={"Authorization": f"Bearer {student_token}"})
        assert r.status_code == 403

    def test_stats_admin_success(self, client, admin_token):
        r = client.get("/api/admin/stats", headers={"Authorization": f"Bearer {admin_token}"})
        assert r.status_code == 200
        data = r.json()
        assert "total_documents" in data
        assert "total_chunks" in data
        assert "embedded_vectors" in data

    def test_documents_admin_success(self, client, admin_token):
        r = client.get("/api/admin/documents", headers={"Authorization": f"Bearer {admin_token}"})
        assert r.status_code == 200
        docs = r.json()
        assert isinstance(docs, list)

    def test_logs_admin_success(self, client, admin_token):
        r = client.get("/api/admin/logs", headers={"Authorization": f"Bearer {admin_token}"})
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_users_requires_admin(self, client, student_token):
        r = client.get("/api/users", headers={"Authorization": f"Bearer {student_token}"})
        assert r.status_code == 403

    def test_users_admin_success(self, client, admin_token):
        r = client.get("/api/users", headers={"Authorization": f"Bearer {admin_token}"})
        assert r.status_code == 200
        users = r.json()
        assert any(u["username"] == "admin_test" for u in users)


# ── Upload tests ───────────────────────────────────────────────────────────────

class TestDocUpload:
    def test_upload_requires_admin(self, client, student_token):
        r = client.post(
            "/api/upload",
            files={"file": ("test.docx", b"content", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
            headers={"Authorization": f"Bearer {student_token}"},
        )
        assert r.status_code == 403

    def test_upload_invalid_extension(self, client, admin_token):
        with patch("backend.document_manager.ingest_uploaded_file") as mock_ingest:
            from backend.document_manager import IngestionResult
            mock_ingest.return_value = IngestionResult(
                filename="test.pdf", status="failed",
                error="Unsupported file type '.pdf'."
            )
            r = client.post(
                "/api/upload",
                files={"file": ("test.pdf", b"fake pdf content", "application/pdf")},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
            assert r.status_code == 400

    def test_upload_success(self, client, admin_token):
        with patch("backend.document_manager.ingest_uploaded_file") as mock_ingest:
            from backend.document_manager import IngestionResult
            mock_ingest.return_value = IngestionResult(
                filename="test.docx", status="ingested",
                doc_id="abc123", department="Academics", version="1.0",
                chunks_created=10, vectors_added=10, processing_ms=1500.0,
            )
            r = client.post(
                "/api/upload",
                files={"file": ("test.docx", b"fake docx content", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
            assert r.status_code == 200
            data = r.json()
            assert data["status"] == "ingested"
            assert data["chunks_created"] == 10
            assert data["vectors_added"] == 10

    def test_upload_duplicate(self, client, admin_token):
        with patch("backend.document_manager.ingest_uploaded_file") as mock_ingest:
            from backend.document_manager import IngestionResult
            mock_ingest.return_value = IngestionResult(
                filename="test.docx", status="duplicate",
                error="Identical document already indexed (SHA-256 match).",
            )
            r = client.post(
                "/api/upload",
                files={"file": ("test.docx", b"duplicate", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
            assert r.status_code == 409


# ── History tests ──────────────────────────────────────────────────────────────

class TestHistory:
    @patch("backend.rag_integration.query", return_value=_MOCK_RAG_RESULT)
    def test_history_after_chat(self, mock_query, client, student_token):
        # Send a chat message
        r = client.post(
            "/api/chat/auth",
            json={"query": "Test history?", "session_id": "hist-test-001"},
            headers={"Authorization": f"Bearer {student_token}"},
        )
        assert r.status_code == 200

        # Check history
        r2 = client.get("/api/history", headers={"Authorization": f"Bearer {student_token}"})
        assert r2.status_code == 200
        sessions = r2.json()
        session_ids = [s["session_id"] for s in sessions]
        assert "hist-test-001" in session_ids

    @patch("backend.rag_integration.query", return_value=_MOCK_RAG_RESULT)
    def test_get_conversation(self, mock_query, client, student_token):
        r = client.post(
            "/api/chat/auth",
            json={"query": "Get conv test?", "session_id": "conv-test-002"},
            headers={"Authorization": f"Bearer {student_token}"},
        )
        assert r.status_code == 200

        r2 = client.get(
            "/api/history/conv-test-002",
            headers={"Authorization": f"Bearer {student_token}"},
        )
        assert r2.status_code == 200
        msgs = r2.json()
        assert len(msgs) >= 2
        # Most recent Q&A pair
        assert any(m["is_user"] is True for m in msgs)
        assert any(m["is_user"] is False for m in msgs)


# agents/langgraph_agent.py (the older single-agent graph, superseded by
# agents/multi_agent_graph.py which is what /api/agent/chat actually runs)
# was removed in Phase 8 -- grep-confirmed its only references anywhere in
# the codebase were the tests that used to live in this class.
