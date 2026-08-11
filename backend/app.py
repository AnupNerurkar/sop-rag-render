"""
backend/app.py
---------------
FastAPI application — EduMind Institutional Knowledge Engine v3.0

Endpoints:
  GET  /                          — serve frontend index.html
  GET  /styles.css, /app.js       — serve static assets

  GET  /api/auth/departments      — department dropdown choices
  POST /api/auth/login            — JWT login
  POST /api/auth/signup           — self-service Student/Faculty signup (active immediately)
  GET  /api/auth/signup-domains   — enabled signup email-domain allowlist (public)
  POST /api/chat                  — public chat (no auth, role=Public)
  POST /api/chat/auth             — authenticated chat (saves history)
  GET  /api/chat/stream           — SSE streaming, public
  GET  /api/chat/auth/stream      — SSE streaming, authenticated

  GET  /api/history               — list sessions (auth)
  GET  /api/history/{id}          — get messages in session (auth)
  DELETE /api/history/{id}        — delete session (auth)

  POST /api/upload                — Admin doc upload → real ingestion pipeline
  GET  /api/users                 — list users (Admin)

  GET  /api/admin/stats           — doc + vector stats (Admin)
  GET  /api/admin/documents       — full document list (Admin)
  GET  /api/admin/logs            — ingestion audit log (Admin)
  GET  /api/admin/vectors         — vector count (Admin)

  POST /api/agent/chat            — LangGraph agentic chat with memory (auth)
"""

import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend import signup_policy
from backend.auth import create_access_token, get_current_user, hash_password, verify_password
from backend.constants import DEPARTMENTS, SIGNUP_ROLES
from backend.database import ChatMessage, SignupDomain, User, get_db, init_db

logger = logging.getLogger(__name__)

app = FastAPI(title="EduMind AI", version="3.0.0", description="Institutional Knowledge Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Populated by the startup Groq model probe (see on_startup); read by
# /api/health. Public and never containing an error message with request
# content -- it's model-selection metadata, not diagnostics.
_GROQ_MODEL_STATUS: dict = {}


@app.on_event("startup")
def on_startup():
    init_db()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    # Autoseed default users if table is empty
    from backend.database import SessionLocal, User
    db = SessionLocal()
    try:
        if db.query(User).count() == 0:
            logging.info("[STARTUP] Database is empty. Seeding default accounts...")
            seed_users = [
                {"username": "public_user",  "password": "Public@123",  "role": "Public"},
                {"username": "student_test", "password": "Student@123", "role": "Student"},
                {"username": "faculty_test", "password": "Faculty@123", "role": "Faculty"},
                {"username": "admin_test",   "password": "Admin@123",   "role": "Admin"},
            ]
            for u in seed_users:
                db.add(User(
                    username=u["username"],
                    hashed_password=hash_password(u["password"]),
                    role=u["role"]
                ))
            db.commit()
            logging.info("[STARTUP] Successfully seeded default accounts.")
        else:
            logging.info("[STARTUP] Database already initialized with users.")
    except Exception as e:
        logging.error(f"[STARTUP] Failed to seed database: {e}")
    finally:
        db.close()

    # Phase 3: Startup Validation Report
    try:
        import ledger
        from vector_store.sqlite_store import get_vector_store

        ledger.initialize_db()

        from retrieval.fts5 import ensure_schema as ensure_fts5_schema
        _fts5_conn = ledger.get_connection()
        try:
            ensure_fts5_schema(_fts5_conn)
        finally:
            _fts5_conn.close()
        docs = ledger.get_all_documents()
        total_docs = len(docs)
        existing_files = 0
        missing_files = 0
        
        for d in docs:
            fp = d.get("source_file") or d.get("filepath")
            if fp:
                full_path = Path(__file__).parent.parent / fp
                if full_path.is_file():
                    existing_files += 1
                else:
                    missing_files += 1
            else:
                missing_files += 1
                
        store = get_vector_store()
        vector_count = store.get_collection_stats().get("vector_count", 0)
        ledger_chunk_count = ledger.get_chunk_count()

        # Orphans are vectors whose chunk row is gone from the ledger -- the
        # signature of a delete that didn't fully commit (see
        # document_manager.delete_document, now transactional). Swept, not
        # just counted: an orphan is otherwise still fully searchable and can
        # surface as a citation for a document that was supposedly deleted.
        orphaned_count = store.sweep_orphaned_embeddings() if vector_count > 0 else 0
        if orphaned_count > 0:
            # Swept rows change the vector count reported above; keep it honest.
            vector_count = store.get_collection_stats().get("vector_count", 0)

        logging.info(
            f"[STARTUP] [VALIDATION] Ingestion Integrity Report:\n"
            f"  - Total Documents in Ledger: {total_docs}\n"
            f"  - Existing Physical Files:   {existing_files}\n"
            f"  - Missing Physical Files:    {missing_files}\n"
            f"  - Chunks in SQLite Ledger:   {ledger_chunk_count}\n"
            f"  - Vectors in store:          {vector_count}\n"
            f"  - Orphaned vectors:          {orphaned_count}"
        )
    except Exception as exc:
        logging.error(f"[STARTUP] Ingestion validation failed: {exc}")

    # Resolve and probe the active Groq model BEFORE anything constructs a
    # GroqConfig (get_rag_engine() below does exactly that) -- GroqConfig
    # reads GROQ_MODEL as a field default at construction time, so a
    # fallback decided after that point would never take effect.
    if os.environ.get("LLM_BACKEND", "ollama").strip().lower() == "groq":
        try:
            from rag.groq_client import resolve_active_model
            global _GROQ_MODEL_STATUS
            _GROQ_MODEL_STATUS = resolve_active_model()
            logging.info(f"[STARTUP] Groq model: {_GROQ_MODEL_STATUS}")
        except Exception as exc:
            logging.error(f"[STARTUP] Groq model probe failed: {exc}")

    # Pre-load retrieval models (embedder & BM25) to avoid slow first queries
    try:
        logging.info("[STARTUP] Pre-loading retrieval models (embedder, BM25)...")
        from retrieval.retriever import get_retriever
        from embeddings.embedder import get_embedder
        get_embedder()
        get_retriever()
        logging.info("[STARTUP] Retrieval models pre-loaded.")
    except Exception as exc:
        logging.error(f"[STARTUP] Failed to pre-load retrieval models: {exc}")



# ── Serve frontend ─────────────────────────────────────────────────────────────
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"


# Static frontend assets are served with `Cache-Control: no-cache` so the browser
# always revalidates (via ETag) and picks up new JS/CSS immediately. Without this,
# browsers heuristically cache app.js and can keep running a stale version — which
# is how inline-handler functions like openDoc end up "undefined" after an update.
_NO_CACHE = {"Cache-Control": "no-cache"}


@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    return HTMLResponse(
        content=(FRONTEND_DIR / "index.html").read_text(encoding="utf-8"),
        headers=_NO_CACHE,
    )


@app.get("/styles.css")
def serve_css():
    return FileResponse(FRONTEND_DIR / "styles.css", media_type="text/css", headers=_NO_CACHE)


@app.get("/app.js")
def serve_js():
    return FileResponse(FRONTEND_DIR / "app.js", media_type="application/javascript", headers=_NO_CACHE)


# ── Schemas ────────────────────────────────────────────────────────────────────
class LoginRequest(BaseModel):
    username: str
    password: str


class SignupRequest(BaseModel):
    username: str
    email: str
    password: str
    role: str
    department: str


class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    use_agent: bool = False        # set True to route through LangGraph agent


class AgentChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    memory: Optional[list] = None  # client may pass conversation memory


# ── Auth ───────────────────────────────────────────────────────────────────────
@app.get("/api/auth/departments")
def auth_departments():
    """Department choices for the signup dropdown."""
    return {"departments": DEPARTMENTS}


@app.post("/api/auth/login")
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == payload.username).first()
    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )
    if user.approval_status == "pending":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Your account is awaiting admin approval. Please check back later.",
        )
    if user.approval_status == "rejected":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Your signup was rejected: {user.rejection_reason or 'no reason given'}.",
        )
    token = create_access_token({"sub": user.username, "role": user.role})
    return {
        "access_token": token,
        "token_type":   "bearer",
        "role":         user.role,
        "username":     user.username,
        "department":   user.department,
        "is_committee_head": user.is_committee_head,
        "committee_name":    user.committee_name,
    }


@app.get("/api/auth/signup-domains")
def auth_signup_domains(db: Session = Depends(get_db)):
    """Public: which email domains (if any) signup is currently restricted to.
    Used by the signup form to show a hint and pre-validate the email. When
    `restricted` is false, any email may register."""
    active = signup_policy.enabled_domains(db)
    return {"restricted": bool(active), "domains": active}


@app.post("/api/auth/signup")
def signup(payload: SignupRequest, db: Session = Depends(get_db)):
    """Self-service registration for Students and Faculty. Accounts are active
    immediately — there is no admin approval step. If the admin has enabled one
    or more allowed email domains, only emails on that allowlist may register;
    otherwise any email is accepted. Admin/Committee Head are assigned by an
    Admin, never self-selected."""
    username = payload.username.strip()
    email = payload.email.strip().lower()
    password = payload.password

    if not (3 <= len(username) <= 32) or not all(c.isalnum() or c in "_-" for c in username):
        raise HTTPException(
            status_code=400,
            detail="Username must be 3-32 characters (letters, numbers, _ or -).",
        )
    if not signup_policy.is_valid_email(email):
        raise HTTPException(status_code=400, detail="Please enter a valid email address.")
    if len(password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters.")
    if payload.role not in SIGNUP_ROLES:
        raise HTTPException(status_code=400, detail=f"Role must be one of: {', '.join(SIGNUP_ROLES)}.")
    if payload.department not in DEPARTMENTS:
        raise HTTPException(status_code=400, detail="Please select a valid department.")

    if not signup_policy.is_email_allowed(db, email):
        allowed = ", ".join("@" + d for d in signup_policy.enabled_domains(db))
        raise HTTPException(
            status_code=403,
            detail=f"Sign-ups are limited to these email domains: {allowed}.",
        )

    if db.query(User).filter(User.username == username).first():
        raise HTTPException(status_code=409, detail="That username is already taken.")
    if db.query(User).filter(User.email == email).first():
        raise HTTPException(status_code=409, detail="An account with that email already exists.")

    user = User(
        username=username,
        email=email,
        hashed_password=hash_password(password),
        role=payload.role,
        department=payload.department,
        approval_status="approved",
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    # Account is active immediately — issue a token so the user is signed straight in.
    token = create_access_token({"sub": user.username, "role": user.role})
    return {
        "status":            "approved",
        "access_token":      token,
        "token_type":        "bearer",
        "role":              user.role,
        "username":          user.username,
        "department":        user.department,
        "is_committee_head": user.is_committee_head,
        "committee_name":    user.committee_name,
    }


# ── Indexing status (public — all users see when the KB is updating) ───────────
@app.get("/api/indexing-status")
def indexing_status():
    """Whether a document is currently being ingested/embedded into the KB."""
    from backend.document_manager import get_indexing_state
    return get_indexing_state()


@app.get("/api/health")
def health():
    """Liveness + active-model info. Public: no request content, just which
    generation backend/model actually answered the startup probe -- useful
    to confirm a Groq model fallback happened without grepping container logs."""
    backend = os.environ.get("LLM_BACKEND", "ollama").strip().lower()
    result = {"status": "ok", "llm_backend": backend}
    if backend == "groq":
        result["groq_model"] = _GROQ_MODEL_STATUS
    return result


# ── Public chat (no auth) ──────────────────────────────────────────────────────
@app.post("/api/chat")
def chat(payload: ChatRequest, db: Session = Depends(get_db)):
    """Public chat — always uses role=Public."""
    from backend.rag_integration import query as rag_query
    sid    = payload.session_id or str(uuid.uuid4())
    result = rag_query(question=payload.query, role="Public")
    return {**result, "session_id": sid}


# ── Authenticated chat (saves history) ────────────────────────────────────────
@app.post("/api/chat/auth")
def chat_authenticated(
    payload: ChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Authenticated chat — uses the user's RBAC role, saves history."""
    from backend.rag_integration import query as rag_query
    sid = payload.session_id or str(uuid.uuid4())

    db.add(ChatMessage(
        username=current_user.username, session_id=sid,
        content=payload.query, is_user=True, sources=None,
    ))

    result = rag_query(question=payload.query, role=current_user.role)

    db.add(ChatMessage(
        username=current_user.username, session_id=sid,
        content=result["answer"], is_user=False,
        sources=json.dumps(result["source_documents"]),
    ))
    db.commit()

    return {**result, "session_id": sid}


# ── SSE Streaming chat ─────────────────────────────────────────────────────────
def _sse_public_generator(question: str, role: str):
    """
    True SSE streaming for public (unauthenticated) chat — no history saved.

    Streams tokens live and emits a final [META] event with citations so the
    frontend can render sources for guest sessions exactly as it does for
    authenticated users. Uses the same structured stream as the auth path.
    """
    from backend.rag_integration import stream_structured

    final_meta: dict = {}
    try:
        for kind, payload in stream_structured(question=question, role=role):
            if kind == "token":
                safe = payload.replace("\n", "\\n")
                yield f"data: {safe}\n\n"
            elif kind == "meta":
                final_meta = payload
    except Exception as exc:
        yield f"data: [ERROR: {str(exc)[:200]}]\n\n"
        yield "data: [DONE]\n\n"
        return

    meta = json.dumps({
        "source_documents":   final_meta.get("source_documents", []),
        "citations":          final_meta.get("citations", []),
        "confidence":         final_meta.get("confidence", "UNKNOWN"),
        "confidence_score":   final_meta.get("confidence_score", 0.0),
        "retrieval_mode":     final_meta.get("retrieval_mode", ""),
        "processing_time_ms": final_meta.get("processing_time_ms", 0.0),
    })
    yield f"data: [META]{meta}\n\n"
    yield "data: [DONE]\n\n"


def _sse_auth_generator(question: str, role: str, username: str, session_id: str):
    """
    Authenticated SSE: streams Qwen tokens LIVE, then saves history and emits
    citations/confidence as a final metadata event.

    Previously this blocked on a full non-streaming pipeline run before sending
    anything — on slow hardware that exceeded Ollama's read timeout, so the
    client only ever saw a blinking cursor that never resolved.  We now consume
    the pipeline's structured stream so the first token paints in seconds and
    the read timeout is reset by every chunk.

    Protocol (unchanged):
      data: <text chunk>              — tokens, as they arrive
      data: [META]{"answer":...}      — final metadata JSON (citations, confidence)
      data: [DONE]                    — stream complete
    """
    from backend.rag_integration import stream_structured
    from backend.database import SessionLocal, ChatMessage

    final_meta: dict = {}
    try:
        for kind, payload in stream_structured(question=question, role=role):
            if kind == "token":
                safe = payload.replace("\n", "\\n")
                yield f"data: {safe}\n\n"
            elif kind == "meta":
                final_meta = payload
    except Exception as exc:
        yield f"data: [ERROR: {str(exc)[:200]}]\n\n"
        yield "data: [DONE]\n\n"
        return

    # Persist to history DB (full accumulated answer from the meta payload)
    try:
        db = SessionLocal()
        db.add(ChatMessage(
            username=username, session_id=session_id,
            content=question, is_user=True, sources=None,
        ))
        db.add(ChatMessage(
            username=username, session_id=session_id,
            content=final_meta.get("answer", ""), is_user=False,
            sources=json.dumps(final_meta.get("source_documents", [])),
        ))
        db.commit()
        db.close()
    except Exception as exc:
        logger.error("[SSE] history save failed: %s", exc)

    # Emit metadata
    meta = json.dumps({
        "source_documents":   final_meta.get("source_documents", []),
        "citations":          final_meta.get("citations", []),
        "confidence":         final_meta.get("confidence", "UNKNOWN"),
        "confidence_score":   final_meta.get("confidence_score", 0.0),
        "retrieval_mode":     final_meta.get("retrieval_mode", ""),
        "processing_time_ms": final_meta.get("processing_time_ms", 0.0),
        "session_id":         session_id,
    })
    yield f"data: [META]{meta}\n\n"
    yield "data: [DONE]\n\n"


@app.get("/api/chat/stream")
def chat_stream(q: str = Query(..., description="Query text"), role: str = Query("Public")):
    """Public SSE streaming endpoint — no auth, no history saved."""
    return StreamingResponse(
        _sse_public_generator(question=q, role=role),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/chat/auth/stream")
def chat_stream_auth(
    q: str = Query(..., description="Query text"),
    session_id: str = Query(default=""),
    current_user: User = Depends(get_current_user),
):
    """Authenticated SSE streaming — saves history, emits metadata at end."""
    sid = session_id or str(uuid.uuid4())
    return StreamingResponse(
        _sse_auth_generator(
            question=q, role=current_user.role,
            username=current_user.username, session_id=sid,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── LangGraph agent chat ───────────────────────────────────────────────────────
@app.post("/api/agent/chat")
def agent_chat(
    payload: AgentChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Agentic chat via the multi-agent LangGraph workflow:
    Query Analyzer → Planner → Retrieval Agent → Context Validator →
    Response Generator → Citation Formatter → Confidence Evaluator →
    Reflection Agent (conditional retry) → Final Response.

    Response shape is a superset of the legacy agent, so existing clients keep
    working unchanged.
    """
    from agents.multi_agent_graph import run_multi_agent
    sid = payload.session_id or str(uuid.uuid4())

    db.add(ChatMessage(
        username=current_user.username, session_id=sid,
        content=payload.query, is_user=True, sources=None,
    ))

    result = run_multi_agent(
        question=payload.query,
        role=current_user.role,
        memory=payload.memory or [],
    )

    db.add(ChatMessage(
        username=current_user.username, session_id=sid,
        content=result["answer"], is_user=False,
        sources=json.dumps(result["source_documents"]),
    ))
    db.commit()

    return {**result, "session_id": sid}


# ── Chat history ───────────────────────────────────────────────────────────────
@app.get("/api/history")
def get_history(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    rows = (
        db.query(ChatMessage)
        .filter(ChatMessage.username == current_user.username)
        .order_by(ChatMessage.timestamp.desc())
        .all()
    )

    sessions: dict[str, dict] = {}
    for row in rows:
        if row.session_id not in sessions:
            sessions[row.session_id] = {
                "session_id":    row.session_id,
                "preview":       "",
                "message_count": 0,
                "timestamp":     row.timestamp.isoformat(),
            }
        sessions[row.session_id]["message_count"] += 1
        if row.is_user and not sessions[row.session_id]["preview"]:
            sessions[row.session_id]["preview"] = row.content[:80]

    return sorted(sessions.values(), key=lambda x: x["timestamp"], reverse=True)


@app.get("/api/history/{session_id}")
def get_conversation(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    rows = (
        db.query(ChatMessage)
        .filter(
            ChatMessage.username   == current_user.username,
            ChatMessage.session_id == session_id,
        )
        .order_by(ChatMessage.timestamp.asc())
        .all()
    )
    if not rows:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return [
        {
            "content":   r.content,
            "is_user":   r.is_user,
            "sources":   json.loads(r.sources) if r.sources else [],
            "timestamp": r.timestamp.isoformat(),
        }
        for r in rows
    ]


@app.delete("/api/history/{session_id}")
def delete_conversation(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    db.query(ChatMessage).filter(
        ChatMessage.username   == current_user.username,
        ChatMessage.session_id == session_id,
    ).delete()
    db.commit()
    return {"status": "deleted"}


# ── Document upload (Admin) ────────────────────────────────────────────────────
# Ingestion (extraction, chunking, embedding, vector upsert) can take well over
# a minute depending on the embedding provider's latency. Running it inline
# held the HTTP connection open for the whole duration, which on constrained
# hosting (e.g. Render's free tier) can exceed the platform's request/health
# check timeout and take the whole service down mid-request. So this endpoint
# only validates the upload synchronously and hands the actual pipeline off to
# a background task — the client polls GET /api/indexing-status for progress
# and the final result (see document_manager.get_indexing_state()).
@app.post("/api/upload")
def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
):
    """Admin-only: upload a .docx document → triggers full ingestion pipeline
    in the background. Returns immediately; poll GET /api/indexing-status."""
    if current_user.role != "Admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only Admin users can upload documents",
        )

    content = file.file.read()
    if not content:
        raise HTTPException(status_code=400, detail="File is empty.")

    from backend.document_manager import run_ingestion_background
    background_tasks.add_task(
        run_ingestion_background, file.filename, content, None, current_user.username
    )

    return {"status": "processing", "filename": file.filename}


# ── Committee Head SOP submissions ──────────────────────────────────────────────
def _require_committee_head(current_user: User = Depends(get_current_user)) -> User:
    if current_user.role != "Student" or not current_user.is_committee_head:
        raise HTTPException(status_code=403, detail="Committee Head access required")
    return current_user


@app.post("/api/committee/upload")
def committee_upload(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    department: Optional[str] = Form(None),
    current_user: User = Depends(_require_committee_head),
):
    """Committee Head: stage an SOP submission pending admin approval."""
    content = file.file.read()

    from backend.committee_manager import save_pending_upload
    try:
        row = save_pending_upload(
            username=current_user.username,
            committee_name=current_user.committee_name,
            filename=file.filename,
            content_bytes=content,
            title=title,
            department=department,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return row


@app.get("/api/committee/my-uploads")
def committee_my_uploads(current_user: User = Depends(_require_committee_head)):
    """Committee Head: track approval status of own SOP submissions."""
    from backend.committee_manager import list_uploads_for_user
    return list_uploads_for_user(current_user.username)


# ── Users list (Admin) ─────────────────────────────────────────────────────────
@app.get("/api/users")
def list_users(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if current_user.role != "Admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    users = db.query(User).all()
    return [
        {
            "id": u.id,
            "username": u.username,
            "email": u.email,
            "role": u.role,
            "department": u.department,
            "approval_status": u.approval_status,
            "rejection_reason": u.rejection_reason,
            "is_committee_head": u.is_committee_head,
            "committee_name": u.committee_name,
        }
        for u in users
    ]


# ── Citation document viewer ────────────────────────────────────────────────────
def _authorize_document(doc_id: str, current_user: User):
    """
    Shared RBAC + resolution for the document endpoints.

    Enforces, in order:
      1. Public role can never open documents          → 403
      2. Document must exist and have a servable file   → 404
      3. Role must satisfy the document's access level  → 403

    Returns the resolved DocumentRef. Filesystem paths are never exposed.
    """
    from backend import document_service as ds

    if not ds.role_can_open(current_user.role):
        raise HTTPException(
            status_code=403,
            detail="Document viewing is not available for public users. Please sign in.",
        )

    docref = ds.load_document(doc_id)
    if docref is None:
        raise HTTPException(status_code=404, detail="Document not found.")

    if not ds.can_access(current_user.role, docref.access_level):
        raise HTTPException(
            status_code=403,
            detail=f"Your role ({current_user.role}) cannot access this document.",
        )
    return docref


@app.get("/api/documents/{doc_id}/meta")
def document_meta(doc_id: str, current_user: User = Depends(get_current_user)):
    """Lightweight metadata for a document the user is allowed to open."""
    docref = _authorize_document(doc_id, current_user)
    return {
        "doc_id":       docref.doc_id,
        "title":        docref.title,
        "department":   docref.department,
        "version":      docref.version,
        "category":     docref.category,
        "access_level": docref.access_level,
        "kind":         docref.kind,
    }


@app.get("/api/documents/{doc_id}/view", response_class=HTMLResponse)
def document_view(
    doc_id: str,
    chunk_index: Optional[int] = Query(default=None),
    current_user: User = Depends(get_current_user),
):
    """
    Returns a self-contained HTML viewer for the document with the source chunk
    highlighted. RBAC-enforced; public users are rejected.
    """
    from backend import document_service as ds
    docref = _authorize_document(doc_id, current_user)
    logger.info(
        "[DOCVIEW] view doc_id=%s chunk=%s by=%s(%s)",
        doc_id[:12], chunk_index, current_user.username, current_user.role,
    )
    return HTMLResponse(content=ds.render_viewer_html(docref, chunk_index))


@app.get("/api/documents/{doc_id}/file")
@app.get("/documents/{doc_id}/file")
def document_file(doc_id: str, current_user: User = Depends(get_current_user)):
    """
    Streams the original document bytes (RBAC-enforced) for inline display.
    """
    from fastapi.responses import JSONResponse
    from backend import document_service as ds
    docref = _authorize_document(doc_id, current_user)
    
    if docref.path is None or not docref.path.is_file():
        logger.warning(
            "[DOCVIEW] file not available inline doc_id=%s path=%s by=%s(%s)",
            doc_id[:12], docref.path, current_user.username, current_user.role
        )
        return JSONResponse(
            status_code=404,
            content={
                "status": "missing",
                "doc_id": doc_id,
                "filepath": str(docref.path) if docref.path else "",
                "message": "Original source document is not available."
            }
        )

    filename = docref.path.name
    logger.info(
        "[DOCVIEW] file doc_id=%s resolved=%s by=%s(%s)",
        doc_id[:12], docref.path, current_user.username, current_user.role,
    )
    return StreamingResponse(
        ds.iter_file_bytes(docref.path),
        media_type=ds.media_type_for(docref.path),
        headers={"Content-Disposition": f'inline; filename="{filename}"'},
    )


@app.get("/api/documents/{doc_id}")
@app.get("/documents/{doc_id}")
def get_document_details(doc_id: str, current_user: User = Depends(get_current_user)):
    """Retrieves document metadata. RBAC-enforced."""
    docref = _authorize_document(doc_id, current_user)
    logger.info("[DOCMETA] details doc_id=%s by=%s(%s)", doc_id[:12], current_user.username, current_user.role)
    return {
        "doc_id":       docref.doc_id,
        "title":        docref.title,
        "department":   docref.department,
        "version":      docref.version,
        "category":     docref.category,
        "access_level": docref.access_level,
        "kind":         docref.kind,
        "filepath":     str(docref.path) if docref.path else "",
        "exists":       docref.path is not None and docref.path.is_file()
    }


@app.get("/api/documents/{doc_id}/download")
@app.get("/documents/{doc_id}/download")
def document_download(doc_id: str, current_user: User = Depends(get_current_user)):
    """Downloads the original document if it exists, otherwise returns a missing error JSON."""
    from fastapi.responses import JSONResponse
    from backend import document_service as ds
    docref = _authorize_document(doc_id, current_user)
    
    if docref.path is None or not docref.path.is_file():
        logger.warning(
            "[DOWNLOAD] file not available doc_id=%s path=%s by=%s(%s)",
            doc_id[:12], docref.path, current_user.username, current_user.role
        )
        return JSONResponse(
            status_code=404,
            content={
                "status": "missing",
                "doc_id": doc_id,
                "filepath": str(docref.path) if docref.path else "",
                "message": "Original source document is not available."
            }
        )
        
    filename = docref.path.name
    logger.info(
        "[DOWNLOAD] file doc_id=%s resolved=%s by=%s(%s)",
        doc_id[:12], docref.path, current_user.username, current_user.role,
    )
    return StreamingResponse(
        ds.iter_file_bytes(docref.path),
        media_type=ds.media_type_for(docref.path),
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/chunks/{chunk_id}")
@app.get("/chunks/{chunk_id}")
def get_chunk_details(chunk_id: str, current_user: User = Depends(get_current_user)):
    """Retrieves specific chunk details from the ledger. RBAC-enforced."""
    from backend import document_service as ds
    if not ds.role_can_open(current_user.role):
        raise HTTPException(status_code=403, detail="Public users cannot access chunk details.")
        
    chunk = ds.get_chunk_by_id(chunk_id)
    if not chunk:
        raise HTTPException(status_code=404, detail="Chunk not found.")
        
    if not ds.can_access(current_user.role, chunk.get("access_level", "Public")):
        raise HTTPException(status_code=403, detail="Your role cannot access this chunk.")
        
    logger.info("[CHUNK] details chunk_id=%s by=%s(%s)", chunk_id[:12], current_user.username, current_user.role)
    return chunk



# ── Admin Dashboard APIs ───────────────────────────────────────────────────────
def _require_admin(current_user: User = Depends(get_current_user)) -> User:
    if current_user.role != "Admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user


@app.get("/api/admin/stats")
def admin_stats(_: User = Depends(_require_admin)):
    """Document counts, chunk counts, vector counts from ledger + vector store."""
    from backend.document_manager import get_document_stats, get_vector_count
    stats = get_document_stats()
    stats["vector_count"] = get_vector_count()
    return stats


@app.get("/api/admin/documents")
def admin_documents(_: User = Depends(_require_admin)):
    """Full document registry from SQLite ledger."""
    from backend.document_manager import get_document_list
    docs = get_document_list()
    # Return a clean subset of fields
    return [
        {
            "doc_id":       d.get("doc_id", ""),
            "source_file":  d.get("source_file", ""),
            "title":        d.get("title", ""),
            "department":   d.get("department", ""),
            "version":      d.get("version", ""),
            "category":     d.get("category", ""),
            "status":       d.get("status", ""),
            "access_level": d.get("access_level", ""),
            "uploaded_by":  d.get("uploaded_by") or "—",
            "total_chunks": d.get("total_chunks", 0),
            "upload_date":  d.get("upload_date", ""),
            "ingested_at":  d.get("ingested_at", ""),
        }
        for d in docs
    ]


@app.delete("/api/admin/documents/{doc_id}")
def admin_delete_document(doc_id: str, current_user: User = Depends(_require_admin)):
    """Removes a document from the knowledge base (ledger + vectors)."""
    from backend.document_manager import delete_document
    from backend.committee_manager import mark_removed_by_doc_id
    result = delete_document(doc_id)
    if not result["removed"]:
        raise HTTPException(status_code=404, detail="Document not found.")
    mark_removed_by_doc_id(doc_id, current_user.username)
    return result


@app.get("/api/admin/logs")
def admin_logs(
    limit: int = Query(50, ge=1, le=500),
    _: User = Depends(_require_admin),
):
    """Recent ingestion audit log events."""
    from backend.document_manager import get_ingestion_logs
    return get_ingestion_logs(limit=limit)


@app.get("/api/admin/vectors")
def admin_vectors(_: User = Depends(_require_admin)):
    """Local vector store info."""
    from backend.document_manager import get_vector_count
    from backend.document_manager import get_document_stats
    stats = get_document_stats()
    return {
        "collection":    "vit_institutional_kb",
        "vector_count":  get_vector_count(),
        "total_chunks":  stats["total_chunks"],
        "embedded":      stats["embedded_vectors"],
        "coverage_pct":  round(
            100 * stats["embedded_vectors"] / max(stats["total_chunks"], 1), 1
        ),
    }


# ── Committee Head approval workflow (Admin) ────────────────────────────────────
class RejectRequest(BaseModel):
    reason: str


class CommitteeHeadRequest(BaseModel):
    is_committee_head: bool
    committee_name: Optional[str] = None


@app.get("/api/admin/pending-approvals")
def admin_pending_approvals(_: User = Depends(_require_admin)):
    """List committee-head SOP submissions awaiting review."""
    from backend.committee_manager import list_pending_approvals
    return list_pending_approvals()


def _resolve_pending_file(upload_id: int) -> tuple[dict, Path]:
    from backend.committee_manager import get_upload
    upload = get_upload(upload_id)
    if upload is None:
        raise HTTPException(status_code=404, detail="Submission not found.")
    path = Path(upload["stored_path"])
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Submission file is no longer available.")
    return upload, path


@app.get("/api/admin/pending-approvals/{upload_id}/preview")
def admin_preview_pending_upload(upload_id: int, _: User = Depends(_require_admin)):
    """Renders a pending SOP submission in-browser so an admin can review it before approving."""
    from backend.document_service import render_pending_preview_html
    upload, path = _resolve_pending_file(upload_id)
    html_doc = render_pending_preview_html(path, upload["original_filename"])
    return HTMLResponse(content=html_doc)


@app.get("/api/admin/pending-approvals/{upload_id}/file")
def admin_download_pending_upload(upload_id: int, _: User = Depends(_require_admin)):
    """Streams the original bytes of a pending SOP submission (download)."""
    from backend.document_service import media_type_for
    upload, path = _resolve_pending_file(upload_id)
    return FileResponse(
        path=str(path),
        media_type=media_type_for(path),
        filename=upload["original_filename"],
    )


@app.post("/api/admin/approvals/{upload_id}/approve")
def admin_approve_upload(
    upload_id: int,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(_require_admin),
):
    """Approve a committee-head SOP submission — validates synchronously, then
    runs ingestion + embedding in the background (see /api/upload's docstring
    for why). Returns immediately; poll GET /api/indexing-status."""
    from backend.committee_manager import approve_upload, get_upload

    row = get_upload(upload_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Submission not found.")
    if row["approval_status"] != "pending":
        raise HTTPException(status_code=400, detail=f"Submission already {row['approval_status']}.")

    background_tasks.add_task(approve_upload, upload_id, current_user.username)
    return {"status": "processing", "upload_id": upload_id, "filename": row["original_filename"]}


@app.post("/api/admin/approvals/{upload_id}/reject")
def admin_reject_upload(
    upload_id: int,
    payload: RejectRequest,
    current_user: User = Depends(_require_admin),
):
    """Reject a committee-head SOP submission with a reason."""
    if not payload.reason.strip():
        raise HTTPException(status_code=400, detail="Rejection reason is required.")

    from backend.committee_manager import reject_upload
    try:
        return reject_upload(upload_id, current_user.username, payload.reason.strip())
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


# ── Signup domain allowlist (Admin) ─────────────────────────────────────────────
# Self-service signup is open by default. Admins can restrict it to specific
# email domains: enabling one or more domains blocks every other domain, while
# emails on the allowlist register (pre-approved) as normal.
class SignupDomainRequest(BaseModel):
    domain: str


class SignupDomainToggle(BaseModel):
    enabled: bool


def _domain_dict(row: SignupDomain) -> dict:
    return {"id": row.id, "domain": row.domain, "enabled": row.enabled, "is_default": row.is_default}


@app.get("/api/admin/signup-domains")
def admin_list_signup_domains(
    _: User = Depends(_require_admin),
    db: Session = Depends(get_db),
):
    """List every configured signup domain and whether it is currently enabled."""
    rows = (
        db.query(SignupDomain)
        .order_by(SignupDomain.is_default.desc(), SignupDomain.domain.asc())
        .all()
    )
    return [_domain_dict(r) for r in rows]


@app.post("/api/admin/signup-domains")
def admin_add_signup_domain(
    payload: SignupDomainRequest,
    _: User = Depends(_require_admin),
    db: Session = Depends(get_db),
):
    """Add a custom allowed domain. New custom domains are enabled on creation,
    which (like enabling any domain) restricts signup to the allowlist."""
    domain = signup_policy.normalize_domain(payload.domain)
    if not signup_policy.is_valid_domain(domain):
        raise HTTPException(status_code=400, detail="Enter a valid domain, e.g. college.edu.in.")
    if db.query(SignupDomain).filter(SignupDomain.domain == domain).first():
        raise HTTPException(status_code=409, detail="That domain is already in the list.")

    row = SignupDomain(domain=domain, enabled=True, is_default=False)
    db.add(row)
    db.commit()
    db.refresh(row)
    return _domain_dict(row)


@app.patch("/api/admin/signup-domains/{domain_id}")
def admin_toggle_signup_domain(
    domain_id: int,
    payload: SignupDomainToggle,
    _: User = Depends(_require_admin),
    db: Session = Depends(get_db),
):
    """Turn an allowed domain on or off. Turning every domain off reopens signup
    to any email."""
    row = db.query(SignupDomain).filter(SignupDomain.id == domain_id).first()
    if row is None:
        raise HTTPException(status_code=404, detail="Domain not found.")
    row.enabled = payload.enabled
    db.commit()
    db.refresh(row)
    return _domain_dict(row)


@app.delete("/api/admin/signup-domains/{domain_id}")
def admin_delete_signup_domain(
    domain_id: int,
    _: User = Depends(_require_admin),
    db: Session = Depends(get_db),
):
    """Remove a custom domain. The two seeded defaults can be turned off but not
    deleted, so they always remain available as one-click options."""
    row = db.query(SignupDomain).filter(SignupDomain.id == domain_id).first()
    if row is None:
        raise HTTPException(status_code=404, detail="Domain not found.")
    if row.is_default:
        raise HTTPException(status_code=400, detail="Default domains can be turned off but not removed.")
    domain = row.domain
    db.delete(row)
    db.commit()
    return {"status": "deleted", "id": domain_id, "domain": domain}


@app.patch("/api/admin/users/{user_id}/committee-head")
def admin_set_committee_head(
    user_id: int,
    payload: CommitteeHeadRequest,
    _: User = Depends(_require_admin),
    db: Session = Depends(get_db),
):
    """Designate or revoke a Student user's Committee Head status."""
    target = db.query(User).filter(User.id == user_id).first()
    if target is None:
        raise HTTPException(status_code=404, detail="User not found.")
    if target.role != "Student":
        raise HTTPException(status_code=400, detail="Committee Head status only applies to Student users.")

    target.is_committee_head = payload.is_committee_head
    target.committee_name = payload.committee_name if payload.is_committee_head else None
    db.commit()
    db.refresh(target)

    return {
        "id": target.id,
        "username": target.username,
        "role": target.role,
        "is_committee_head": target.is_committee_head,
        "committee_name": target.committee_name,
    }


@app.delete("/api/users/{user_id}")
def delete_user(
    user_id: int,
    current_user: User = Depends(_require_admin),
    db: Session = Depends(get_db),
):
    """Deletes a user account. Admins cannot delete themselves or the last
    remaining Admin account, to avoid ever locking every admin out."""
    target = db.query(User).filter(User.id == user_id).first()
    if target is None:
        raise HTTPException(status_code=404, detail="User not found.")
    if target.id == current_user.id:
        raise HTTPException(status_code=400, detail="You cannot delete your own account.")
    if target.role == "Admin":
        admin_count = db.query(User).filter(User.role == "Admin").count()
        if admin_count <= 1:
            raise HTTPException(status_code=400, detail="Cannot delete the last remaining Admin account.")

    db.delete(target)
    db.commit()
    return {"status": "deleted", "id": user_id, "username": target.username}
