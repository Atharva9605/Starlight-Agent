"""Email attachment persistence — files + metadata for CRM threads."""
from __future__ import annotations

import logging
import os
import re
import uuid
from pathlib import Path
from typing import Any, Optional

from vector_store import use_postgres, _pg_conn
from tenant import get_organization_id

log = logging.getLogger("attachment_store")

_BASE = Path(__file__).parent
ATTACH_ROOT = Path(os.getenv("ATTACHMENT_DIR", str(_BASE / "static" / "attachments")))


def _org(organization_id: str | None = None) -> str:
    return organization_id or get_organization_id()


def _new_id() -> str:
    return str(uuid.uuid4())


def _safe_filename(name: str) -> str:
    name = (name or "file").strip().replace("\\", "/").split("/")[-1]
    name = re.sub(r"[^\w.\-()+ ]+", "_", name)
    return (name[:180] or "file").strip()


def init_attachment_tables() -> None:
    if not use_postgres():
        return
    conn = _pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS email_attachments (
                    id TEXT PRIMARY KEY,
                    organization_id TEXT NOT NULL,
                    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    message_id TEXT,
                    filename TEXT NOT NULL,
                    mime_type TEXT NOT NULL DEFAULT 'application/octet-stream',
                    size_bytes INTEGER NOT NULL DEFAULT 0,
                    storage_path TEXT NOT NULL DEFAULT '',
                    download_url TEXT NOT NULL DEFAULT '',
                    source TEXT NOT NULL DEFAULT 'inbound',
                    gmail_attachment_id TEXT,
                    extracted_text TEXT NOT NULL DEFAULT '',
                    rag_ingested BOOLEAN NOT NULL DEFAULT FALSE,
                    include_on_send BOOLEAN NOT NULL DEFAULT FALSE,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_email_attachments_conv "
                "ON email_attachments (conversation_id)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_email_attachments_msg "
                "ON email_attachments (message_id)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_email_attachments_org "
                "ON email_attachments (organization_id)"
            )
        conn.commit()
        log.info("email_attachments table ready")
    finally:
        conn.close()


def _row(r) -> dict:
    return {
        "id": r[0],
        "organization_id": r[1],
        "conversation_id": r[2],
        "message_id": r[3],
        "filename": r[4],
        "mime_type": r[5],
        "size_bytes": int(r[6] or 0),
        "storage_path": r[7],
        "download_url": r[8],
        "source": r[9],
        "gmail_attachment_id": r[10],
        "extracted_text": (r[11] or "")[:5000],
        "rag_ingested": bool(r[12]),
        "include_on_send": bool(r[13]),
        "created_at": r[14].isoformat() if r[14] else None,
    }


_SELECT = """
    SELECT id, organization_id, conversation_id, message_id, filename, mime_type,
           size_bytes, storage_path, download_url, source, gmail_attachment_id,
           extracted_text, rag_ingested, include_on_send, created_at
    FROM email_attachments
"""


def write_bytes(
    organization_id: str,
    attachment_id: str,
    filename: str,
    data: bytes,
) -> str:
    """Persist file to disk; return absolute storage path."""
    ATTACH_ROOT.mkdir(parents=True, exist_ok=True)
    org_dir = ATTACH_ROOT / organization_id
    org_dir.mkdir(parents=True, exist_ok=True)
    safe = _safe_filename(filename)
    path = org_dir / f"{attachment_id}_{safe}"
    path.write_bytes(data)
    return str(path)


def read_bytes(storage_path: str) -> bytes:
    return Path(storage_path).read_bytes()


def create_attachment(
    *,
    conversation_id: str,
    filename: str,
    mime_type: str,
    data: bytes,
    source: str = "inbound",
    message_id: str | None = None,
    gmail_attachment_id: str | None = None,
    extracted_text: str = "",
    include_on_send: bool = False,
    organization_id: str | None = None,
) -> dict:
    from vector_store import use_postgres
    if not use_postgres():
        raise RuntimeError("Attachments require DATABASE_URL")

    org = _org(organization_id)
    att_id = _new_id()
    path = write_bytes(org, att_id, filename, data)
    download_url = f"/api/attachments/{att_id}/download"

    conn = _pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO email_attachments (
                    id, organization_id, conversation_id, message_id, filename, mime_type,
                    size_bytes, storage_path, download_url, source, gmail_attachment_id,
                    extracted_text, rag_ingested, include_on_send
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,FALSE,%s)
                RETURNING id, organization_id, conversation_id, message_id, filename, mime_type,
                          size_bytes, storage_path, download_url, source, gmail_attachment_id,
                          extracted_text, rag_ingested, include_on_send, created_at
                """,
                (
                    att_id,
                    org,
                    conversation_id,
                    message_id,
                    _safe_filename(filename),
                    mime_type or "application/octet-stream",
                    len(data),
                    path,
                    download_url,
                    source,
                    gmail_attachment_id,
                    extracted_text or "",
                    include_on_send,
                ),
            )
            row = cur.fetchone()
        conn.commit()
        return _row(row)
    finally:
        conn.close()


def get_attachment(attachment_id: str, organization_id: str | None = None) -> Optional[dict]:
    org = _org(organization_id)
    conn = _pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                _SELECT + " WHERE id = %s AND organization_id = %s",
                (attachment_id, org),
            )
            row = cur.fetchone()
        return _row(row) if row else None
    finally:
        conn.close()


def list_attachments(
    conversation_id: str,
    organization_id: str | None = None,
) -> list[dict]:
    org = _org(organization_id)
    conn = _pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                _SELECT
                + " WHERE conversation_id = %s AND organization_id = %s ORDER BY created_at ASC",
                (conversation_id, org),
            )
            return [_row(r) for r in cur.fetchall()]
    finally:
        conn.close()


def list_outbound_pending(
    conversation_id: str,
    message_id: str | None = None,
    organization_id: str | None = None,
) -> list[dict]:
    """Attachments that should go out with the next send."""
    org = _org(organization_id)
    conn = _pg_conn()
    try:
        with conn.cursor() as cur:
            if message_id:
                cur.execute(
                    _SELECT
                    + """
                    WHERE conversation_id = %s AND organization_id = %s
                      AND include_on_send = TRUE
                      AND (message_id IS NULL OR message_id = %s)
                      AND source IN ('outbound', 'generated')
                    ORDER BY created_at ASC
                    """,
                    (conversation_id, org, message_id),
                )
            else:
                cur.execute(
                    _SELECT
                    + """
                    WHERE conversation_id = %s AND organization_id = %s
                      AND include_on_send = TRUE
                      AND source IN ('outbound', 'generated')
                    ORDER BY created_at ASC
                    """,
                    (conversation_id, org),
                )
            return [_row(r) for r in cur.fetchall()]
    finally:
        conn.close()


def mark_rag_ingested(attachment_id: str, extracted_text: str = "") -> None:
    conn = _pg_conn()
    try:
        with conn.cursor() as cur:
            if extracted_text:
                cur.execute(
                    """
                    UPDATE email_attachments
                    SET rag_ingested = TRUE, extracted_text = %s
                    WHERE id = %s
                    """,
                    (extracted_text[:50000], attachment_id),
                )
            else:
                cur.execute(
                    "UPDATE email_attachments SET rag_ingested = TRUE WHERE id = %s",
                    (attachment_id,),
                )
        conn.commit()
    finally:
        conn.close()


def link_to_message(attachment_ids: list[str], message_id: str) -> None:
    if not attachment_ids:
        return
    conn = _pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE email_attachments SET message_id = %s
                WHERE id = ANY(%s)
                """,
                (message_id, attachment_ids),
            )
        conn.commit()
    finally:
        conn.close()


def delete_attachment(attachment_id: str, organization_id: str | None = None) -> bool:
    att = get_attachment(attachment_id, organization_id)
    if not att:
        return False
    try:
        p = Path(att["storage_path"])
        if p.exists():
            p.unlink()
    except OSError:
        pass
    conn = _pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM email_attachments WHERE id = %s AND organization_id = %s",
                (attachment_id, _org(organization_id)),
            )
        conn.commit()
        return True
    finally:
        conn.close()


def find_by_gmail_id(
    conversation_id: str,
    gmail_attachment_id: str,
    organization_id: str | None = None,
) -> Optional[dict]:
    if not gmail_attachment_id:
        return None
    org = _org(organization_id)
    conn = _pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                _SELECT
                + " WHERE conversation_id = %s AND organization_id = %s AND gmail_attachment_id = %s",
                (conversation_id, org, gmail_attachment_id),
            )
            row = cur.fetchone()
        return _row(row) if row else None
    finally:
        conn.close()
