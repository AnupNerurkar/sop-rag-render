"""
backend/committee_manager.py
-----------------------------
Committee Head SOP submission workflow: upload → pending → admin review → ingest.

Committee-head uploads are held in data/pending_uploads/ and tracked in the
CommitteeUpload table until an admin approves or rejects them. Approval hands
the file off to document_manager.ingest_uploaded_file() (the same pipeline
used for admin uploads) with access_level forced to "Student". Nothing is
ingested, chunked, or embedded until approval.

Public API:
  save_pending_upload(username, committee_name, filename, content_bytes, title, department) -> CommitteeUpload
  list_uploads_for_user(username)   -> list[dict]
  list_pending_approvals()          -> list[dict]
  approve_upload(upload_id, admin_username) -> dict
  reject_upload(upload_id, admin_username, reason) -> dict
"""

from __future__ import annotations

import hashlib
import logging
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from backend.database import CommitteeUpload, SessionLocal
from backend.document_manager import ALLOWED_EXTENSIONS

PENDING_DIR = Path(_PROJECT_ROOT) / "data" / "pending_uploads"


def _discard_pending_file(stored_path: str) -> None:
    """Removes a reviewed submission's staged copy from data/pending_uploads/.

    Once a submission is reviewed the raw bytes are no longer needed: approved
    files live in data/staging/ + the vector store, and the CommitteeUpload row retains
    all submission metadata. This keeps data/pending_uploads/ holding only files
    still awaiting review.
    """
    try:
        os.remove(stored_path)
    except OSError:
        pass


def _row_to_dict(row: CommitteeUpload) -> dict:
    return {
        "id": row.id,
        "uploaded_by": row.uploaded_by,
        "committee_name": row.committee_name,
        "original_filename": row.original_filename,
        "title": row.title,
        "department": row.department,
        "approval_status": row.approval_status,
        "rejection_reason": row.rejection_reason,
        "reviewed_by": row.reviewed_by,
        "reviewed_at": row.reviewed_at.isoformat() if row.reviewed_at else None,
        "doc_id": row.doc_id,
        "submitted_at": row.submitted_at.isoformat() if row.submitted_at else None,
    }


def save_pending_upload(
    username: str,
    committee_name: Optional[str],
    filename: str,
    content_bytes: bytes,
    title: Optional[str] = None,
    department: Optional[str] = None,
) -> dict:
    """Validates and stages a committee-head SOP submission as 'pending'."""
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file type '{ext}'. Accepted: {', '.join(ALLOWED_EXTENSIONS)}")
    if not content_bytes:
        raise ValueError("File is empty.")

    file_hash = hashlib.sha256(content_bytes).hexdigest()

    PENDING_DIR.mkdir(parents=True, exist_ok=True)
    stored_name = f"{uuid.uuid4().hex}_{filename}"
    stored_path = PENDING_DIR / stored_name
    with open(stored_path, "wb") as f:
        f.write(content_bytes)

    db = SessionLocal()
    try:
        row = CommitteeUpload(
            uploaded_by=username,
            committee_name=committee_name,
            original_filename=filename,
            stored_path=str(stored_path),
            sha256_hash=file_hash,
            title=title,
            department=department,
            approval_status="pending",
            submitted_at=datetime.utcnow(),
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        logger.info("[COMMITTEE] '%s' staged pending submission '%s'", username, filename)
        return _row_to_dict(row)
    finally:
        db.close()


def mark_removed_by_doc_id(doc_id: str, admin_username: str) -> int:
    """Flags any approved submission whose document was deleted from the
    knowledge base, so the committee head sees the removal in their tracking
    view. Returns the number of rows updated."""
    db = SessionLocal()
    try:
        rows = (
            db.query(CommitteeUpload)
            .filter(CommitteeUpload.doc_id == doc_id, CommitteeUpload.approval_status == "approved")
            .all()
        )
        for row in rows:
            row.approval_status = "removed"
            row.reviewed_by = admin_username
            row.reviewed_at = datetime.utcnow()
        db.commit()
        return len(rows)
    finally:
        db.close()


def list_uploads_for_user(username: str) -> list[dict]:
    db = SessionLocal()
    try:
        rows = (
            db.query(CommitteeUpload)
            .filter(CommitteeUpload.uploaded_by == username)
            .order_by(CommitteeUpload.submitted_at.desc())
            .all()
        )
        return [_row_to_dict(r) for r in rows]
    finally:
        db.close()


def get_upload(upload_id: int) -> Optional[dict]:
    """Returns a submission's tracking fields plus its staged file path, for
    admin preview/download before a decision is made."""
    db = SessionLocal()
    try:
        row = db.query(CommitteeUpload).filter(CommitteeUpload.id == upload_id).first()
        if row is None:
            return None
        data = _row_to_dict(row)
        data["stored_path"] = row.stored_path
        return data
    finally:
        db.close()


def list_pending_approvals() -> list[dict]:
    db = SessionLocal()
    try:
        rows = (
            db.query(CommitteeUpload)
            .filter(CommitteeUpload.approval_status == "pending")
            .order_by(CommitteeUpload.submitted_at.asc())
            .all()
        )
        return [_row_to_dict(r) for r in rows]
    finally:
        db.close()


def approve_upload(upload_id: int, admin_username: str) -> dict:
    from backend.document_manager import ingest_uploaded_file

    db = SessionLocal()
    try:
        row = db.query(CommitteeUpload).filter(CommitteeUpload.id == upload_id).first()
        if row is None:
            raise ValueError("Submission not found.")
        if row.approval_status != "pending":
            raise ValueError(f"Submission already {row.approval_status}.")

        with open(row.stored_path, "rb") as f:
            content_bytes = f.read()

        result = ingest_uploaded_file(
            row.original_filename,
            content_bytes,
            forced_access_level="Student",
            uploaded_by=row.uploaded_by,
        )

        if not result.success:
            logger.error(
                "[COMMITTEE] Approval ingestion failed for upload %d: %s", upload_id, result.error
            )
            return {
                "success": False,
                "error": result.error or f"Ingestion status: {result.status}",
                "upload": _row_to_dict(row),
            }

        row.approval_status = "approved"
        row.reviewed_by = admin_username
        row.reviewed_at = datetime.utcnow()
        row.doc_id = result.doc_id
        db.commit()
        db.refresh(row)
        _discard_pending_file(row.stored_path)

        logger.info("[COMMITTEE] Upload %d approved by '%s' -> doc_id=%s", upload_id, admin_username, result.doc_id)
        return {"success": True, "upload": _row_to_dict(row), "ingestion": result.__dict__}
    finally:
        db.close()


def reject_upload(upload_id: int, admin_username: str, reason: str) -> dict:
    db = SessionLocal()
    try:
        row = db.query(CommitteeUpload).filter(CommitteeUpload.id == upload_id).first()
        if row is None:
            raise ValueError("Submission not found.")
        if row.approval_status != "pending":
            raise ValueError(f"Submission already {row.approval_status}.")

        row.approval_status = "rejected"
        row.rejection_reason = reason
        row.reviewed_by = admin_username
        row.reviewed_at = datetime.utcnow()
        db.commit()
        db.refresh(row)
        _discard_pending_file(row.stored_path)

        logger.info("[COMMITTEE] Upload %d rejected by '%s': %s", upload_id, admin_username, reason)
        return _row_to_dict(row)
    finally:
        db.close()
