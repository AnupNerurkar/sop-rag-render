import os
from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Integer, String, Text, create_engine, event
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./institutional.db")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

IS_SQLITE = DATABASE_URL.startswith("sqlite")

if IS_SQLITE:
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
else:
    engine = create_engine(DATABASE_URL)

if IS_SQLITE:
    @event.listens_for(engine, "connect")
    def _sqlite_pragmas(dbapi_conn, _connection_record):
        """
        FastAPI serves `def` endpoints from a threadpool, so this engine has
        concurrent writers. SQLite's default busy timeout is zero, which turns
        ordinary contention into "database is locked" 500s. WAL also lets
        readers proceed while a write is in flight. journal_mode persists with
        the file, but busy_timeout is per-connection and must be set on each.
        Mirrors the PRAGMAs in ledger.get_connection().
        """
        cursor = dbapi_conn.cursor()
        try:
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA busy_timeout=5000")
            cursor.execute("PRAGMA synchronous=NORMAL")
        finally:
            cursor.close()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


DEFAULT_SIGNUP_DOMAINS = ["spit.ac.in", "vit.edu.in"]


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, nullable=False, index=True)
    email = Column(String, unique=True, nullable=True, index=True)
    hashed_password = Column(String, nullable=False)
    role = Column(String, nullable=False)  # 'Public','Student','Faculty','Admin'
    is_committee_head = Column(Boolean, nullable=False, default=False)
    committee_name = Column(String, nullable=True)
    department = Column(String, nullable=True)
    approval_status = Column(String, nullable=False, default="approved")  # 'pending'|'approved'|'rejected'
    rejection_reason = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class CommitteeUpload(Base):
    __tablename__ = "committee_uploads"
    id = Column(Integer, primary_key=True, index=True)
    uploaded_by = Column(String, nullable=False, index=True)
    committee_name = Column(String, nullable=True)
    original_filename = Column(String, nullable=False)
    stored_path = Column(String, nullable=False)
    sha256_hash = Column(String, nullable=False, index=True)
    title = Column(String, nullable=True)
    department = Column(String, nullable=True)
    approval_status = Column(String, nullable=False, default="pending")  # 'pending'|'approved'|'rejected'
    rejection_reason = Column(Text, nullable=True)
    reviewed_by = Column(String, nullable=True)
    reviewed_at = Column(DateTime, nullable=True)
    doc_id = Column(String, nullable=True)
    submitted_at = Column(DateTime, default=datetime.utcnow)


class SignupDomain(Base):
    """Admin-managed email-domain allowlist for self-service signup.

    When no rows are `enabled`, signup is open to any email. As soon as one or
    more domains are enabled, only emails on that allowlist may register — every
    other domain is blocked. The two seeded defaults start disabled so signup
    stays open until an admin deliberately turns one on."""
    __tablename__ = "signup_domains"
    id = Column(Integer, primary_key=True, index=True)
    domain = Column(String, unique=True, nullable=False, index=True)
    enabled = Column(Boolean, nullable=False, default=False)
    is_default = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class ChatMessage(Base):
    __tablename__ = "chat_messages"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, nullable=False, index=True)
    session_id = Column(String, nullable=False, index=True)
    content = Column(Text, nullable=False)
    is_user = Column(Boolean, nullable=False)
    sources = Column(Text, nullable=True)   # JSON list stored as string
    timestamp = Column(DateTime, default=datetime.utcnow)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _ensure_column(cursor, table: str, column: str, definition: str) -> None:
    """Adds a column to an existing table if it is not already present."""
    if engine.dialect.name == "postgresql":
        cursor.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = %s",
            (table.lower(),)
        )
        existing = {row[0].lower() for row in cursor.fetchall()}
    else:
        cursor.execute(f"PRAGMA table_info({table})")
        existing = {row[1] for row in cursor.fetchall()}
        
    if column.lower() not in existing:
        if engine.dialect.name == "postgresql":
            if "DEFAULT 0" in definition.upper() and "BOOLEAN" in definition.upper():
                definition = definition.replace("DEFAULT 0", "DEFAULT FALSE")
        cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def migrate_user_columns():
    """Extends a pre-existing users table with committee-head columns."""
    conn = engine.raw_connection()
    try:
        cursor = conn.cursor()
        _ensure_column(cursor, "users", "email", "TEXT")
        _ensure_column(cursor, "users", "is_committee_head", "BOOLEAN NOT NULL DEFAULT 0")
        _ensure_column(cursor, "users", "committee_name", "TEXT")
        _ensure_column(cursor, "users", "department", "TEXT")
        _ensure_column(cursor, "users", "approval_status", "TEXT NOT NULL DEFAULT 'approved'")
        _ensure_column(cursor, "users", "rejection_reason", "TEXT")
        _ensure_column(cursor, "users", "created_at", "TIMESTAMP")
        conn.commit()
    finally:
        conn.close()


def seed_default_domains():
    """Ensures the two canonical college domains exist as (disabled) defaults."""
    db = SessionLocal()
    try:
        for domain in DEFAULT_SIGNUP_DOMAINS:
            if not db.query(SignupDomain).filter(SignupDomain.domain == domain).first():
                db.add(SignupDomain(domain=domain, enabled=False, is_default=True))
        db.commit()
    finally:
        db.close()


def auto_approve_pending_users():
    """Signup approval was removed — free any accounts still stuck as 'pending'."""
    db = SessionLocal()
    try:
        db.query(User).filter(User.approval_status == "pending").update(
            {User.approval_status: "approved", User.rejection_reason: None},
            synchronize_session=False,
        )
        db.commit()
    finally:
        db.close()


def init_db():
    Base.metadata.create_all(bind=engine)
    migrate_user_columns()
    seed_default_domains()
    auto_approve_pending_users()
