"""
Persistent CRM conversation storage (PostgreSQL via DATABASE_URL).

Requires the same Neon/pgvector DATABASE_URL as the knowledge base.
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from vector_store import use_postgres, _pg_conn
from tenant import get_organization_id

log = logging.getLogger("conversation_store")


def _org(organization_id: str | None = None) -> str:
    return organization_id or get_organization_id()


def _require_postgres() -> None:
    if not use_postgres():
        raise RuntimeError(
            "Conversations require DATABASE_URL (PostgreSQL). "
            "Set DATABASE_URL on your HuggingFace Space secrets."
        )


def _new_id() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


def init_conversation_tables() -> None:
    """Create CRM tables. Called from vector_store.init_db()."""
    if not use_postgres():
        return

    conn = _pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS clients (
                    id TEXT PRIMARY KEY,
                    email TEXT NOT NULL UNIQUE,
                    company TEXT NOT NULL DEFAULT '',
                    website TEXT NOT NULL DEFAULT '',
                    profile_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    client_id TEXT NOT NULL REFERENCES clients(id) ON DELETE CASCADE,
                    status TEXT NOT NULL DEFAULT 'active',
                    gmail_thread_id TEXT,
                    subject TEXT NOT NULL DEFAULT '',
                    template_name TEXT NOT NULL DEFAULT 'email_template.html',
                    conversation_summary TEXT NOT NULL DEFAULT '',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_conversations_client ON conversations (client_id)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_conversations_thread ON conversations (gmail_thread_id)"
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    direction TEXT NOT NULL,
                    sender_email TEXT NOT NULL DEFAULT '',
                    recipient_email TEXT NOT NULL DEFAULT '',
                    subject TEXT NOT NULL DEFAULT '',
                    body_text TEXT NOT NULL DEFAULT '',
                    body_html TEXT NOT NULL DEFAULT '',
                    banner_html TEXT NOT NULL DEFAULT '',
                    gmail_message_id TEXT,
                    in_reply_to TEXT,
                    references_header TEXT,
                    message_id_header TEXT,
                    status TEXT NOT NULL DEFAULT 'received',
                    ai_generated BOOLEAN NOT NULL DEFAULT FALSE,
                    internal_note TEXT NOT NULL DEFAULT '',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages (conversation_id)"
            )
            cur.execute(
                "ALTER TABLE messages ADD COLUMN IF NOT EXISTS message_id_header TEXT"
            )
            cur.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_messages_gmail_id "
                "ON messages (gmail_message_id) WHERE gmail_message_id IS NOT NULL"
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS timeline_events (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    event_type TEXT NOT NULL,
                    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_timeline_conversation ON timeline_events (conversation_id)"
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS gmail_sync_state (
                    id TEXT PRIMARY KEY DEFAULT 'default',
                    history_id TEXT,
                    last_sync_at TIMESTAMPTZ,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
        conn.commit()
        log.info("Conversation CRM tables ready")
        try:
            from attachment_store import init_attachment_tables
            init_attachment_tables()
        except Exception as exc:
            log.warning("Attachment tables init failed: %s", exc)
    finally:
        conn.close()


def upsert_client(
    email: str,
    company: str = "",
    website: str = "",
    profile_json: Optional[dict] = None,
) -> dict:
    _require_postgres()
    email = email.strip().lower()
    if not email:
        raise ValueError("Client email is required")

    conn = _pg_conn()
    org = _org()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM clients WHERE email = %s AND organization_id = %s",
                (email, org),
            )
            row = cur.fetchone()
            if row:
                client_id = row[0]
                cur.execute(
                    """
                    UPDATE clients SET
                        company = COALESCE(NULLIF(%s, ''), company),
                        website = COALESCE(NULLIF(%s, ''), website),
                        profile_json = COALESCE(%s::jsonb, profile_json)
                    WHERE id = %s AND organization_id = %s
                    RETURNING id, email, company, website, profile_json, created_at
                    """,
                    (company, website, json.dumps(profile_json or {}), client_id, org),
                )
            else:
                client_id = _new_id()
                cur.execute(
                    """
                    INSERT INTO clients (id, email, company, website, profile_json, organization_id)
                    VALUES (%s, %s, %s, %s, %s::jsonb, %s)
                    RETURNING id, email, company, website, profile_json, created_at
                    """,
                    (client_id, email, company, website, json.dumps(profile_json or {}), org),
                )
            row = cur.fetchone()
        conn.commit()
        return _client_row(row)
    finally:
        conn.close()


def create_conversation(
    client_id: str,
    subject: str = "",
    template_name: str = "email_template.html",
    gmail_thread_id: Optional[str] = None,
    status: str = "active",
) -> dict:
    _require_postgres()
    conv_id = _new_id()
    org = _org()
    conn = _pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO conversations (id, client_id, subject, template_name, gmail_thread_id, status, organization_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id, client_id, status, gmail_thread_id, subject, template_name,
                          conversation_summary, created_at, updated_at
                """,
                (conv_id, client_id, subject, template_name, gmail_thread_id, status, org),
            )
            row = cur.fetchone()
        conn.commit()
        return _conversation_row(row)
    finally:
        conn.close()


def add_message(
    conversation_id: str,
    direction: str,
    *,
    sender_email: str = "",
    recipient_email: str = "",
    subject: str = "",
    body_text: str = "",
    body_html: str = "",
    banner_html: str = "",
    gmail_message_id: Optional[str] = None,
    in_reply_to: str = "",
    references_header: str = "",
    message_id_header: str = "",
    status: str = "received",
    ai_generated: bool = False,
    internal_note: str = "",
) -> dict:
    _require_postgres()
    msg_id = _new_id()
    org = _org()
    conn = _pg_conn()
    try:
        with conn.cursor() as cur:
            if gmail_message_id:
                cur.execute(
                    "SELECT id FROM messages WHERE gmail_message_id = %s AND organization_id = %s",
                    (gmail_message_id, org),
                )
                existing = cur.fetchone()
                if existing:
                    return get_message(existing[0])

            cur.execute(
                """
                INSERT INTO messages (
                    id, conversation_id, direction, sender_email, recipient_email,
                    subject, body_text, body_html, banner_html, gmail_message_id,
                    in_reply_to, references_header, message_id_header, status, ai_generated, internal_note,
                    organization_id
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id, conversation_id, direction, sender_email, recipient_email,
                          subject, body_text, body_html, banner_html, gmail_message_id,
                          in_reply_to, references_header, message_id_header, status, ai_generated, internal_note, created_at
                """,
                (
                    msg_id, conversation_id, direction, sender_email, recipient_email,
                    subject, body_text, body_html, banner_html, gmail_message_id,
                    in_reply_to, references_header, message_id_header, status, ai_generated, internal_note,
                    org,
                ),
            )
            row = cur.fetchone()
            cur.execute(
                "UPDATE conversations SET updated_at = NOW() WHERE id = %s",
                (conversation_id,),
            )
        conn.commit()
        return _message_row(row)
    finally:
        conn.close()


def add_timeline_event(
    conversation_id: str,
    event_type: str,
    payload: Optional[dict] = None,
) -> dict:
    _require_postgres()
    event_id = _new_id()
    org = _org()
    conn = _pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO timeline_events (id, conversation_id, event_type, payload, organization_id)
                VALUES (%s, %s, %s, %s::jsonb, %s)
                RETURNING id, conversation_id, event_type, payload, created_at
                """,
                (event_id, conversation_id, event_type, json.dumps(payload or {}), org),
            )
            row = cur.fetchone()
        conn.commit()
        return _timeline_row(row)
    finally:
        conn.close()


def list_conversations(
    status: Optional[str] = None,
    search: Optional[str] = None,
    mailbox_only: bool = False,
    limit: int = 100,
) -> list[dict]:
    _require_postgres()
    org = _org()
    conn = _pg_conn()
    try:
        clauses = ["c.organization_id = %s"]
        params: list[Any] = [org]
        if status:
            clauses.append("c.status = %s")
            params.append(status)
        if mailbox_only:
            # Inbox = any thread with a real inbound message (client/vendor email).
            # Previously required outbound+inbound, which hid almost every synced reply.
            clauses.append(
                """EXISTS (
                    SELECT 1 FROM messages m
                    WHERE m.conversation_id = c.id
                      AND m.direction = 'inbound'
                      AND m.status != 'draft'
                )"""
            )
        if search:
            clauses.append(
                "(cl.email ILIKE %s OR cl.company ILIKE %s OR c.subject ILIKE %s)"
            )
            like = f"%{search}%"
            params.extend([like, like, like])

        params.append(limit)
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT c.id, c.client_id, c.status, c.gmail_thread_id, c.subject,
                       c.template_name, c.conversation_summary, c.created_at, c.updated_at,
                       cl.email, cl.company, cl.website,
                       (SELECT body_text FROM messages m
                        WHERE m.conversation_id = c.id AND m.status != 'draft'
                        ORDER BY m.created_at DESC LIMIT 1) AS last_message,
                       (SELECT COUNT(*) FROM messages m
                        WHERE m.conversation_id = c.id AND m.status = 'draft') AS draft_count,
                       (SELECT COUNT(*) FROM messages m
                        WHERE m.conversation_id = c.id
                          AND m.direction = 'inbound' AND m.status != 'draft') AS inbound_count,
                       (SELECT direction FROM messages m
                        WHERE m.conversation_id = c.id AND m.status != 'draft'
                        ORDER BY m.created_at DESC LIMIT 1) AS last_direction,
                       (SELECT COUNT(*) FROM messages m
                        WHERE m.conversation_id = c.id AND m.status != 'rejected') AS message_count,
                       (SELECT COUNT(*) FROM messages m
                        WHERE m.conversation_id = c.id
                          AND m.direction = 'outbound' AND m.status = 'sent') AS outbound_sent_count
                FROM conversations c
                JOIN clients cl ON cl.id = c.client_id
                WHERE {' AND '.join(clauses)}
                ORDER BY
                  CASE WHEN (SELECT COUNT(*) FROM messages m
                             WHERE m.conversation_id = c.id AND m.status = 'draft') > 0
                       THEN 0 ELSE 1 END,
                  CASE WHEN (SELECT direction FROM messages m
                             WHERE m.conversation_id = c.id AND m.status != 'draft'
                             ORDER BY m.created_at DESC LIMIT 1) = 'inbound'
                       THEN 0 ELSE 1 END,
                  c.updated_at DESC
                LIMIT %s
                """,
                params,
            )
            rows = cur.fetchall()
        results = [_list_conversation_row(r) for r in rows]
        if mailbox_only:
            import os
            from email_filters import is_noise_email
            from send_eml_gsuite import DELEGATED_USER

            own = {
                (DELEGATED_USER or "").lower(),
                (os.getenv("SENDER_EMAIL") or "").lower(),
                (os.getenv("GSUITE_DELEGATED_USER") or "").lower(),
            }
            own.discard("")
            results = [
                r
                for r in results
                if r.get("client_email")
                and not is_noise_email(r["client_email"])
                and r["client_email"].lower() not in own
            ]
        return results
    finally:
        conn.close()


def get_conversation(conversation_id: str, organization_id: str | None = None) -> Optional[dict]:
    _require_postgres()
    org = _org(organization_id)
    conn = _pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT c.id, c.client_id, c.status, c.gmail_thread_id, c.subject,
                       c.template_name, c.conversation_summary, c.created_at, c.updated_at,
                       cl.id, cl.email, cl.company, cl.website, cl.profile_json, cl.created_at
                FROM conversations c
                JOIN clients cl ON cl.id = c.client_id
                WHERE c.id = %s AND c.organization_id = %s
                """,
                (conversation_id, org),
            )
            row = cur.fetchone()
            if not row:
                return None

            cur.execute(
                """
                SELECT id, conversation_id, direction, sender_email, recipient_email,
                       subject, body_text, body_html, banner_html, gmail_message_id,
                       in_reply_to, references_header, message_id_header, status,
                       ai_generated, internal_note, created_at
                FROM messages WHERE conversation_id = %s ORDER BY created_at ASC
                """,
                (conversation_id,),
            )
            messages = [_message_row(m) for m in cur.fetchall()]

            cur.execute(
                """
                SELECT id, conversation_id, event_type, payload, created_at
                FROM timeline_events WHERE conversation_id = %s ORDER BY created_at ASC
                """,
                (conversation_id,),
            )
            timeline = [_timeline_row(t) for t in cur.fetchall()]

        conv = _conversation_detail_row(row)
        conv["client"] = {
            "id": row[9],
            "email": row[10],
            "company": row[11],
            "website": row[12],
            "profile_json": row[13] if isinstance(row[13], dict) else json.loads(row[13] or "{}"),
            "created_at": row[14].isoformat() if row[14] else None,
        }
        conv["messages"] = messages
        conv["timeline"] = timeline
        try:
            from attachment_store import list_attachments
            conv["attachments"] = list_attachments(conversation_id, org)
        except Exception:
            conv["attachments"] = []
        return conv
    finally:
        conn.close()


def get_message(message_id: str) -> Optional[dict]:
    _require_postgres()
    conn = _pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, conversation_id, direction, sender_email, recipient_email,
                       subject, body_text, body_html, banner_html, gmail_message_id,
                       in_reply_to, references_header, message_id_header, status,
                       ai_generated, internal_note, created_at
                FROM messages WHERE id = %s
                """,
                (message_id,),
            )
            row = cur.fetchone()
        return _message_row(row) if row else None
    finally:
        conn.close()


def update_message(message_id: str, **fields) -> Optional[dict]:
    _require_postgres()
    allowed = {
        "subject", "body_text", "body_html", "banner_html",
        "status", "internal_note", "gmail_message_id",
    }
    updates = {k: v for k, v in fields.items() if k in allowed and v is not None}
    if not updates:
        return get_message(message_id)

    set_clause = ", ".join(f"{k} = %s" for k in updates)
    params = list(updates.values()) + [message_id]

    conn = _pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                UPDATE messages SET {set_clause}
                WHERE id = %s
                RETURNING id, conversation_id, direction, sender_email, recipient_email,
                          subject, body_text, body_html, banner_html, gmail_message_id,
                          in_reply_to, references_header, message_id_header, status,
                          ai_generated, internal_note, created_at
                """,
                params,
            )
            row = cur.fetchone()
        conn.commit()
        return _message_row(row) if row else None
    finally:
        conn.close()


def update_conversation(conversation_id: str, **fields) -> Optional[dict]:
    _require_postgres()
    allowed = {"status", "gmail_thread_id", "subject", "template_name", "conversation_summary"}
    updates = {k: v for k, v in fields.items() if k in allowed and v is not None}
    if not updates:
        conv = get_conversation(conversation_id)
        return conv

    updates["updated_at"] = _now()
    set_clause = ", ".join(f"{k} = %s" for k in updates)
    params = list(updates.values()) + [conversation_id]

    conn = _pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                UPDATE conversations SET {set_clause}
                WHERE id = %s AND organization_id = %s
                RETURNING id, client_id, status, gmail_thread_id, subject, template_name,
                          conversation_summary, created_at, updated_at
                """,
                params + [_org()],
            )
            row = cur.fetchone()
        conn.commit()
        return _conversation_row(row) if row else None
    finally:
        conn.close()


def find_conversation_by_thread(gmail_thread_id: str, organization_id: str | None = None) -> Optional[dict]:
    _require_postgres()
    if not gmail_thread_id:
        return None
    org = _org(organization_id)
    conn = _pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, client_id, status, gmail_thread_id, subject, template_name,
                       conversation_summary, created_at, updated_at
                FROM conversations WHERE gmail_thread_id = %s AND organization_id = %s LIMIT 1
                """,
                (gmail_thread_id, org),
            )
            row = cur.fetchone()
        return _conversation_row(row) if row else None
    finally:
        conn.close()


def find_conversation_by_client_email(email: str, organization_id: str | None = None) -> Optional[dict]:
    _require_postgres()
    email = email.strip().lower()
    org = _org(organization_id)
    conn = _pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT c.id, c.client_id, c.status, c.gmail_thread_id, c.subject,
                       c.template_name, c.conversation_summary, c.created_at, c.updated_at
                FROM conversations c
                JOIN clients cl ON cl.id = c.client_id
                WHERE cl.email = %s AND c.organization_id = %s
                ORDER BY c.updated_at DESC LIMIT 1
                """,
                (email, org),
            )
            row = cur.fetchone()
        return _conversation_row(row) if row else None
    finally:
        conn.close()


def get_gmail_sync_state(organization_id: str | None = None) -> dict:
    _require_postgres()
    org = _org(organization_id)
    conn = _pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, history_id, last_sync_at, updated_at FROM gmail_sync_state
                WHERE organization_id = %s OR id = %s
                ORDER BY updated_at DESC LIMIT 1
                """,
                (org, org),
            )
            row = cur.fetchone()
            if not row:
                return {"history_id": None, "last_sync_at": None}
            return {
                "history_id": row[1],
                "last_sync_at": row[2].isoformat() if row[2] else None,
            }
    finally:
        conn.close()


def set_gmail_sync_state(history_id: Optional[str] = None, organization_id: str | None = None) -> None:
    _require_postgres()
    org = _org(organization_id)
    conn = _pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO gmail_sync_state (id, organization_id, history_id, last_sync_at, updated_at)
                VALUES (%s, %s, %s, NOW(), NOW())
                ON CONFLICT (id) DO UPDATE SET
                    history_id = COALESCE(EXCLUDED.history_id, gmail_sync_state.history_id),
                    last_sync_at = NOW(),
                    updated_at = NOW(),
                    organization_id = EXCLUDED.organization_id
                """,
                (org, org, history_id),
            )
        conn.commit()
    finally:
        conn.close()


def delete_draft(message_id: str) -> bool:
    _require_postgres()
    conn = _pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM messages WHERE id = %s AND status = 'draft'",
                (message_id,),
            )
            deleted = cur.rowcount > 0
        conn.commit()
        return deleted
    finally:
        conn.close()


def _iso(dt) -> Optional[str]:
    return dt.isoformat() if dt else None


def _client_row(row) -> dict:
    profile = row[4]
    if profile is not None and not isinstance(profile, dict):
        profile = json.loads(profile)
    return {
        "id": row[0],
        "email": row[1],
        "company": row[2],
        "website": row[3],
        "profile_json": profile or {},
        "created_at": _iso(row[5]),
    }


def _conversation_row(row) -> dict:
    return {
        "id": row[0],
        "client_id": row[1],
        "status": row[2],
        "gmail_thread_id": row[3],
        "subject": row[4],
        "template_name": row[5],
        "conversation_summary": row[6] if len(row) > 6 else "",
        "created_at": _iso(row[7] if len(row) > 7 else row[6]),
        "updated_at": _iso(row[8] if len(row) > 8 else row[7]),
    }


def _list_conversation_row(row) -> dict:
    draft_count = int(row[13] or 0)
    inbound_count = int(row[14] or 0)
    last_direction = row[15] if len(row) > 15 else None
    message_count = int(row[16] or 0) if len(row) > 16 else 0
    outbound_sent_count = int(row[17] or 0) if len(row) > 17 else 0
    # Needs reply: AI draft waiting, or client spoke last.
    needs_action = draft_count > 0 or last_direction == "inbound"
    return {
        "id": row[0],
        "client_id": row[1],
        "status": row[2],
        "gmail_thread_id": row[3],
        "subject": row[4],
        "template_name": row[5],
        "conversation_summary": row[6],
        "created_at": _iso(row[7]),
        "updated_at": _iso(row[8]),
        "client_email": row[9],
        "client_company": row[10],
        "client_website": row[11],
        "last_message_preview": (row[12] or "")[:200],
        "draft_count": draft_count,
        "inbound_count": inbound_count,
        "needs_action": needs_action,
        "last_direction": last_direction,
        "message_count": message_count,
        "outbound_sent_count": outbound_sent_count,
        "is_campaign_reply": outbound_sent_count > 0 and inbound_count > 0,
    }


def _conversation_detail_row(row) -> dict:
    return {
        "id": row[0],
        "client_id": row[1],
        "status": row[2],
        "gmail_thread_id": row[3],
        "subject": row[4],
        "template_name": row[5],
        "conversation_summary": row[6],
        "created_at": _iso(row[7]),
        "updated_at": _iso(row[8]),
    }


def _message_row(row) -> dict:
    return {
        "id": row[0],
        "conversation_id": row[1],
        "direction": row[2],
        "sender_email": row[3],
        "recipient_email": row[4],
        "subject": row[5],
        "body_text": row[6],
        "body_html": row[7],
        "banner_html": row[8],
        "gmail_message_id": row[9],
        "in_reply_to": row[10],
        "references_header": row[11],
        "message_id_header": row[12],
        "status": row[13],
        "ai_generated": row[14],
        "internal_note": row[15],
        "created_at": _iso(row[16]),
    }


def _timeline_row(row) -> dict:
    payload = row[3]
    if payload is not None and not isinstance(payload, dict):
        payload = json.loads(payload)
    return {
        "id": row[0],
        "conversation_id": row[1],
        "event_type": row[2],
        "payload": payload or {},
        "created_at": _iso(row[4]),
    }
