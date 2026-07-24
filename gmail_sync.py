"""
Background Gmail sync — polls inbox and triggers conversation agent drafts.
"""
from __future__ import annotations

import asyncio
import logging
import os

from send_eml_gsuite import poll_history_changes, DELEGATED_USER, download_gmail_attachment
import conversation_store as store
from conversation_agent import generate_draft_for_conversation
from email_filters import extract_email, is_noise_email
import base64

log = logging.getLogger("gmail_sync")

POLL_INTERVAL = int(os.getenv("GMAIL_POLL_INTERVAL_SEC", "120"))


def _extract_email(addr: str) -> str:
    return extract_email(addr)


def _is_from_client(from_email: str) -> bool:
    """Ignore our own sends and automated/system mailers."""
    email = _extract_email(from_email)
    if not email:
        return False
    delegated = _extract_email(DELEGATED_USER)
    if email == delegated:
        return False
    if is_noise_email(email):
        return False
    return True


def _ingest_inbound_attachment(
    conversation_id: str,
    message_id: str,
    gmail_message_id: str | None,
    att_meta: dict,
) -> None:
    from attachment_store import find_by_gmail_id
    from attachment_service import save_and_ingest, MAX_ATTACHMENT_BYTES

    gmail_att_id = att_meta.get("attachment_id")
    if gmail_att_id and find_by_gmail_id(conversation_id, gmail_att_id):
        return

    data = None
    if att_meta.get("inline_data"):
        try:
            data = base64.urlsafe_b64decode(att_meta["inline_data"])
        except Exception:
            data = None
    if data is None and gmail_att_id and gmail_message_id:
        data = download_gmail_attachment(gmail_message_id, gmail_att_id)
    if not data:
        return
    if len(data) > MAX_ATTACHMENT_BYTES:
        log.warning("Skipping oversized attachment %s", att_meta.get("filename"))
        return

    save_and_ingest(
        conversation_id=conversation_id,
        filename=att_meta.get("filename") or "attachment",
        data=data,
        mime_type=att_meta.get("mime_type") or "application/octet-stream",
        source="inbound",
        message_id=message_id,
        gmail_attachment_id=gmail_att_id,
        include_on_send=False,
    )


def process_inbound_message(gmail_msg: dict) -> dict | None:
    """
    Match inbound Gmail message to a conversation, store it, and generate draft.
    Returns summary dict or None if skipped.
    """
    from_email = _extract_email(gmail_msg.get("from_email", ""))
    if not from_email or not _is_from_client(from_email):
        if from_email and is_noise_email(from_email):
            log.debug("Skipping noise/system email from %s", from_email)
        return None

    gmail_id = gmail_msg.get("gmail_message_id")
    thread_id = gmail_msg.get("thread_id")

    conv = None
    if thread_id:
        conv = store.find_conversation_by_thread(thread_id)
    if not conv:
        conv = store.find_conversation_by_client_email(from_email)

    if not conv:
        log.debug("Skipping inbox message from %s — no matching client thread", from_email)
        return None

    if thread_id and not conv.get("gmail_thread_id"):
        store.update_conversation(conv["id"], gmail_thread_id=thread_id)

    msg = store.add_message(
        conv["id"],
        direction="inbound",
        sender_email=from_email,
        recipient_email=_extract_email(gmail_msg.get("to_email", "")),
        subject=gmail_msg.get("subject", ""),
        body_text=gmail_msg.get("body_text", ""),
        body_html=gmail_msg.get("body_html", ""),
        gmail_message_id=gmail_id,
        in_reply_to=gmail_msg.get("in_reply_to", ""),
        references_header=gmail_msg.get("references", ""),
        message_id_header=gmail_msg.get("message_id_header", ""),
        status="received",
    )

    # Ingest file attachments into storage + RAG
    for att_meta in gmail_msg.get("attachments") or []:
        try:
            _ingest_inbound_attachment(conv["id"], msg["id"], gmail_id, att_meta)
        except Exception as exc:
            log.warning("Attachment ingest failed: %s", exc)

    store.add_timeline_event(
        conv["id"],
        "inbound_received",
        {
            "message_id": msg["id"],
            "from": from_email,
            "subject": gmail_msg.get("subject", ""),
            "preview": (gmail_msg.get("body_text", "") or "")[:200],
        },
    )

    try:
        draft = generate_draft_for_conversation(conv["id"])
        return {
            "conversation_id": conv["id"],
            "inbound_message_id": msg["id"],
            "draft_id": draft["id"],
        }
    except ValueError as exc:
        # Draft gate skips are expected — not errors
        if str(exc).startswith("Draft skipped:"):
            log.info("Draft skipped for %s: %s", conv["id"], exc)
            return {
                "conversation_id": conv["id"],
                "inbound_message_id": msg["id"],
                "draft_skipped": str(exc),
            }
        raise
    except Exception as exc:
        log.error("Draft generation failed for %s: %s", conv["id"], exc)
        store.add_timeline_event(
            conv["id"],
            "error",
            {"stage": "draft_generation", "error": str(exc)},
        )
        return {
            "conversation_id": conv["id"],
            "inbound_message_id": msg["id"],
            "error": str(exc),
        }


def run_gmail_sync(organization_id: str | None = None) -> dict:
    """Single sync pass. Returns stats."""
    from tenant import DEFAULT_ORG_ID, TenantContext, set_tenant_context

    org = organization_id or DEFAULT_ORG_ID
    set_tenant_context(TenantContext(user_id="system", organization_id=org, role="owner"))
    try:
        store._require_postgres()
    except RuntimeError as exc:
        return {"error": str(exc), "processed": 0}

    state = store.get_gmail_sync_state()
    start_id = state.get("history_id")

    new_messages, latest_id = poll_history_changes(start_id)
    processed = 0
    results = []

    for gmail_msg in new_messages:
        result = process_inbound_message(gmail_msg)
        if result:
            processed += 1
            results.append(result)

    if latest_id:
        store.set_gmail_sync_state(latest_id)

    return {
        "processed": processed,
        "new_messages_seen": len(new_messages),
        "results": results,
        "history_id": latest_id,
    }


def _orgs_to_sync() -> list[str]:
    """All orgs with Gmail OAuth, plus default SA path if not already listed."""
    from tenant import DEFAULT_ORG_ID

    org_ids: list[str] = []
    try:
        from org_store import list_orgs_with_gmail

        org_ids = list(list_orgs_with_gmail())
    except Exception as exc:
        log.warning("Could not list Gmail-connected orgs: %s", exc)

    if DEFAULT_ORG_ID not in org_ids:
        org_ids.append(DEFAULT_ORG_ID)
    return org_ids


async def gmail_sync_loop() -> None:
    """Background asyncio loop — syncs every org with Gmail (not only default)."""
    log.info("Gmail sync loop starting (interval=%ds)", POLL_INTERVAL)
    await asyncio.sleep(10)

    while True:
        try:
            org_ids = await asyncio.to_thread(_orgs_to_sync)
            for org_id in org_ids:
                try:
                    result = await asyncio.to_thread(run_gmail_sync, org_id)
                    if result.get("processed"):
                        log.info(
                            "Gmail sync [%s]: processed %d inbound messages",
                            org_id,
                            result["processed"],
                        )
                    if result.get("error"):
                        log.debug("Gmail sync [%s]: %s", org_id, result["error"])
                except Exception as exc:
                    log.error("Gmail sync failed for org %s: %s", org_id, exc)
        except Exception as exc:
            log.error("Gmail sync loop error: %s", exc)
        await asyncio.sleep(POLL_INTERVAL)
