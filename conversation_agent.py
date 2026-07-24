"""
AI conversation agent — tool-calling reply drafter with HITL (never auto-sends).

Tools (max 3 steps): search_catalogue, get_attachment_text, get_thread_summary, propose_draft.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

from azure_client import azure_manager
from config_manager import get_prompt, get_sender
from generator_v2 import query_rag_with_trace
from schemas import ConversationDraft, parse_with_retry
from ai_events import timed_ai_event
from draft_gate import should_auto_draft
import conversation_store as store

# Re-export for callers/tests
__all__ = [
    "generate_draft_for_conversation",
    "generate_banner",
    "record_outbound_from_pipeline",
    "should_auto_draft",
]

log = logging.getLogger("conversation_agent")

MAX_HISTORY_MESSAGES = 20
SUMMARY_THRESHOLD = 12
MAX_AGENT_STEPS = 3

REPLY_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_catalogue",
            "description": "Search the product catalogue knowledge base for grounded product facts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_attachment_text",
            "description": "Get extracted text from an email attachment by id, or list attachments if id omitted.",
            "parameters": {
                "type": "object",
                "properties": {
                    "attachment_id": {"type": "string"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_thread_summary",
            "description": "Return the stored conversation summary plus recent message count.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "propose_draft",
            "description": "Propose the final email draft for human approval. Never sends.",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string"},
                    "body_text": {"type": "string"},
                    "body_html": {"type": "string"},
                    "banner_html": {"type": "string"},
                    "internal_note": {"type": "string"},
                },
                "required": ["subject", "body_text"],
            },
        },
    },
]


def _strip_html(html: str) -> str:
    if not html:
        return ""
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.I | re.S)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"</p>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _format_message_history(messages: list[dict], sender_company: str = "US") -> str:
    lines = []
    for msg in messages:
        if msg.get("status") == "draft":
            continue
        direction = msg.get("direction", "unknown")
        role = "CLIENT" if direction == "inbound" else sender_company.upper()
        subject = msg.get("subject", "")
        body = msg.get("body_text") or _strip_html(msg.get("body_html", ""))
        if not body:
            continue
        ts = msg.get("created_at", "")
        lines.append(f"[{ts}] {role} ({subject}):\n{body}\n")
    return "\n---\n".join(lines) if lines else "(No prior messages)"


def _build_rag_context(
    client_json: dict,
    latest_reply: str,
    conversation_id: str,
) -> tuple[str, dict]:
    client_desc = json.dumps(client_json, ensure_ascii=False)
    combined = f"{client_desc}\n\nLatest client message:\n{latest_reply}"
    # Catalogue + this conversation's attachment chunks only
    trace = query_rag_with_trace(
        combined,
        k=5,
        where={"conversation_id": conversation_id},
    )
    return trace.get("context_str", ""), trace


def _clear_existing_drafts(conversation_id: str, messages: list[dict]) -> None:
    for msg in messages:
        if msg.get("status") == "draft":
            store.delete_draft(msg["id"])


def maybe_summarize_conversation(conversation_id: str, messages: list[dict]) -> str:
    """When history is long, compress older turns into conversation_summary."""
    conv = store.get_conversation(conversation_id) or {}
    existing = (conv.get("conversation_summary") or "").strip()
    real = [m for m in messages if m.get("status") != "draft"]
    if len(real) <= SUMMARY_THRESHOLD:
        return existing

    older = real[:-MAX_HISTORY_MESSAGES]
    if not older:
        return existing

    sender = get_sender()
    transcript = _format_message_history(older, sender.get("sender_company", "US"))
    raw = azure_manager.chat_completion(
        [
            {
                "role": "system",
                "content": (
                    "Summarize this B2B email thread for a sales CRM. "
                    "Keep facts, open questions, commitments, and product interests. "
                    "Return plain text under 250 words."
                ),
            },
            {
                "role": "user",
                "content": f"Prior summary:\n{existing or '(none)'}\n\nOlder messages:\n{transcript}",
            },
        ],
        temperature=0.2,
        max_tokens=512,
    )
    summary = (raw or existing).strip()
    if summary:
        store.update_conversation(conversation_id, conversation_summary=summary)
    return summary


def _run_tool(name: str, args: dict, *, conversation_id: str, client_profile: dict) -> str:
    if name == "search_catalogue":
        q = args.get("query") or json.dumps(client_profile)
        trace = query_rag_with_trace(q, k=4, where={"catalogue_only": True})
        return json.dumps(
            {
                "context": trace.get("context_str", ""),
                "empty_rag": trace.get("empty_rag", False),
                "chunks": len(trace.get("chunks") or []),
            },
            ensure_ascii=False,
        )

    if name == "get_attachment_text":
        try:
            from attachment_store import list_attachments, get_attachment
        except Exception as exc:
            return json.dumps({"error": str(exc)})
        att_id = (args.get("attachment_id") or "").strip()
        if not att_id:
            atts = list_attachments(conversation_id)
            return json.dumps(
                [
                    {
                        "id": a.get("id"),
                        "filename": a.get("filename"),
                        "rag_ingested": a.get("rag_ingested"),
                    }
                    for a in atts
                ]
            )
        att = get_attachment(att_id)
        if not att or att.get("conversation_id") != conversation_id:
            return json.dumps({"error": "attachment not found"})
        text = (att.get("extracted_text") or "")[:6000]
        return json.dumps({"filename": att.get("filename"), "text": text})

    if name == "get_thread_summary":
        conv = store.get_conversation(conversation_id) or {}
        return json.dumps(
            {
                "summary": conv.get("conversation_summary") or "",
                "subject": conv.get("subject") or "",
                "message_count": len(conv.get("messages") or []),
            }
        )

    if name == "propose_draft":
        return json.dumps({"accepted": True, "draft": args})

    return json.dumps({"error": f"Unknown tool: {name}"})


def _agent_loop(
    *,
    conversation_id: str,
    system_prompt: str,
    user_prompt: str,
    client_profile: dict,
) -> ConversationDraft:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    proposed: dict | None = None

    for step in range(MAX_AGENT_STEPS):
        msg = azure_manager.chat_completion_message(
            messages,
            temperature=0.4,
            max_tokens=4096,
            tools=REPLY_TOOLS,
            tool_choice="auto",
        )
        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            # Fallback: treat content as JSON draft
            content = msg.get("content") or ""
            return parse_with_retry(
                ConversationDraft,
                content,
                retry_fn=lambda: azure_manager.chat_completion(
                    messages
                    + [
                        {
                            "role": "user",
                            "content": (
                                "Return ONLY JSON with keys: subject, body_text, "
                                "body_html, banner_html, internal_note"
                            ),
                        }
                    ],
                    temperature=0.2,
                    max_tokens=4096,
                    json_mode=True,
                ),
            )

        messages.append(
            {
                "role": "assistant",
                "content": msg.get("content") or None,
                "tool_calls": tool_calls,
            }
        )

        for tc in tool_calls:
            fn = tc["function"]["name"]
            try:
                args = json.loads(tc["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {}
            result = _run_tool(
                fn, args, conversation_id=conversation_id, client_profile=client_profile
            )
            if fn == "propose_draft":
                try:
                    proposed = json.loads(result).get("draft") or args
                except json.JSONDecodeError:
                    proposed = args
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result,
                }
            )

        if proposed is not None:
            break

    if proposed is None:
        # Force a final propose without tools
        raw = azure_manager.chat_completion(
            messages
            + [
                {
                    "role": "user",
                    "content": (
                        "Call complete. Now output ONLY the final draft JSON with keys: "
                        "subject, body_text, body_html, banner_html, internal_note."
                    ),
                }
            ],
            temperature=0.3,
            max_tokens=4096,
            json_mode=True,
        )
        return parse_with_retry(ConversationDraft, raw)

    return parse_with_retry(ConversationDraft, proposed)


def generate_draft_for_conversation(
    conversation_id: str,
    instructions: str = "",
    *,
    skip_gate: bool = False,
) -> dict:
    """
    Generate an AI draft reply for a conversation (tool-calling agent, HITL).
    Saves draft message + timeline events. Returns the draft message dict.
    """
    conv = store.get_conversation(conversation_id)
    if not conv:
        raise ValueError(f"Conversation not found: {conversation_id}")

    messages = conv.get("messages", [])
    inbound = [m for m in messages if m["direction"] == "inbound" and m["status"] != "draft"]
    if not inbound:
        raise ValueError("No inbound messages to reply to")

    latest = inbound[-1]
    client = conv.get("client", {})
    profile = client.get("profile_json", {})
    latest_reply = latest.get("body_text") or _strip_html(latest.get("body_html", ""))

    if not skip_gate:
        gate = should_auto_draft(latest_reply, latest.get("subject", ""))
        if not gate.should_draft:
            store.add_timeline_event(
                conversation_id,
                "draft_skipped",
                {"reason": gate.reason, "message_id": latest.get("id")},
            )
            raise ValueError(f"Draft skipped: {gate.reason}")

    summary = maybe_summarize_conversation(conversation_id, messages)
    sender = get_sender()
    history_messages = [m for m in messages if m["status"] != "draft"][-MAX_HISTORY_MESSAGES:]
    message_history = _format_message_history(
        history_messages, sender.get("sender_company", "US")
    )

    rag_context, rag_trace = _build_rag_context(profile, latest_reply, conversation_id)
    sender_info = json.dumps(sender, ensure_ascii=False)

    try:
        from attachment_service import attachment_context_for_conversation
        attachment_context = attachment_context_for_conversation(conversation_id)
    except Exception:
        attachment_context = "(No email attachments)"

    system_prompt = get_prompt("conversation_system")
    # Encourage tools without auto-send
    system_prompt += (
        "\n\nYou may use tools to search the catalogue, read attachments, "
        "or read the thread summary. Always finish by calling propose_draft. "
        "Never claim the email was sent — a human must approve."
    )

    user_prompt = get_prompt("conversation_user").format(
        message_history=message_history,
        client_json=json.dumps(profile, ensure_ascii=False),
        rag_context=rag_context,
        latest_reply=latest_reply,
        sender_info=sender_info,
    )
    if summary:
        user_prompt = f"THREAD SUMMARY:\n{summary}\n\n" + user_prompt
    user_prompt += (
        "\n\nEMAIL ATTACHMENTS (use facts from these when relevant):\n"
        f"{attachment_context}\n"
    )
    if instructions.strip():
        user_prompt += (
            "\n\nAdditional instructions from the reviewer:\n"
            f"{instructions.strip()}"
        )

    _clear_existing_drafts(conversation_id, messages)

    with timed_ai_event(
        "conversation_draft",
        prompt_key="conversation_system",
        meta={"conversation_id": conversation_id},
    ) as bag:
        bag["chunk_ids"] = [
            str((c.get("metadata") or {}).get("source", ""))
            for c in (rag_trace.get("chunks") or [])
        ]
        parsed = _agent_loop(
            conversation_id=conversation_id,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            client_profile=profile if isinstance(profile, dict) else {},
        )

    subject = parsed.subject or f"Re: {conv.get('subject', 'Your inquiry')}"
    body_text = parsed.body_text
    body_html = parsed.body_html
    banner_html = parsed.banner_html
    internal_note = parsed.internal_note

    if banner_html and body_html and banner_html not in body_html:
        body_html = banner_html + "\n" + body_html

    if not body_html and body_text:
        body_html = f"<html><body><p>{body_text.replace(chr(10), '</p><p>')}</p></body></html>"

    draft = store.add_message(
        conversation_id,
        direction="outbound",
        sender_email=sender.get("sender_email", ""),
        recipient_email=client.get("email", ""),
        subject=subject,
        body_text=body_text,
        body_html=body_html,
        banner_html=banner_html,
        in_reply_to=latest.get("message_id_header", ""),
        references_header=latest.get("references_header", ""),
        status="draft",
        ai_generated=True,
        internal_note=internal_note,
    )

    store.add_timeline_event(
        conversation_id,
        "draft_generated",
        {
            "message_id": draft["id"],
            "subject": subject,
            "internal_note": internal_note,
            "rag_chunks": len(rag_trace.get("chunks", [])),
            "agent": "tool_loop",
        },
    )

    if banner_html:
        store.add_timeline_event(
            conversation_id,
            "banner_created",
            {"message_id": draft["id"], "banner_preview": banner_html[:500]},
        )

    return draft


def generate_banner(
    conversation_id: str,
    message_id: Optional[str] = None,
) -> dict:
    """Generate an HTML banner for a conversation draft."""
    conv = store.get_conversation(conversation_id)
    if not conv:
        raise ValueError(f"Conversation not found: {conversation_id}")

    messages = conv.get("messages", [])
    target = None
    if message_id:
        target = next((m for m in messages if m["id"] == message_id), None)
    if not target:
        drafts = [m for m in messages if m.get("status") == "draft"]
        target = drafts[-1] if drafts else None
    if not target:
        raise ValueError("No draft message to attach banner to")

    client = conv.get("client", {})
    profile = client.get("profile_json", {})
    sender = get_sender()

    raw = azure_manager.chat_completion(
        [
            {"role": "system", "content": get_prompt("banner_system")},
            {
                "role": "user",
                "content": get_prompt("banner_user").format(
                    instructions=f"Create a banner for subject: {target.get('subject', '')}",
                    client_json=json.dumps(profile, ensure_ascii=False),
                    rag_context=sender.get("sender_company", ""),
                ),
            },
        ],
        temperature=0.5,
        max_tokens=1024,
        json_mode=True,
    )
    from schemas import extract_json_object

    parsed = extract_json_object(raw) or {}
    if isinstance(parsed, dict):
        banner_html = parsed.get("banner_html", raw)
    else:
        banner_html = str(raw)

    store.update_message(target["id"], banner_html=banner_html)
    store.add_timeline_event(
        conversation_id,
        "banner_created",
        {"message_id": target["id"], "banner_preview": banner_html[:500]},
    )
    return {"message_id": target["id"], "banner_html": banner_html}


def record_outbound_from_pipeline(
    *,
    client_email: str,
    client_company: str = "",
    client_website: str = "",
    subject: str = "",
    body_html: str = "",
    body_text: str = "",
    profile_json: dict | None = None,
    template_name: str = "email_template.html",
    gmail_message_id: str = "",
    gmail_thread_id: str = "",
) -> dict:
    """Record an outbound campaign email into CRM conversations."""
    client = store.upsert_client(
        email=client_email,
        company=client_company,
        website=client_website,
        profile_json=profile_json or {},
    )
    conv = store.find_conversation_by_client_email(client_email)
    if not conv:
        conv = store.create_conversation(
            client_id=client["id"],
            subject=subject,
            template_name=template_name,
            gmail_thread_id=gmail_thread_id or None,
        )
    elif gmail_thread_id and not conv.get("gmail_thread_id"):
        store.update_conversation(conv["id"], gmail_thread_id=gmail_thread_id)

    msg = store.add_message(
        conv["id"],
        direction="outbound",
        sender_email=get_sender().get("sender_email", ""),
        recipient_email=client_email,
        subject=subject,
        body_text=body_text,
        body_html=body_html,
        gmail_message_id=gmail_message_id or None,
        status="sent",
        ai_generated=True,
    )
    store.add_timeline_event(
        conv["id"],
        "outbound_sent",
        {"message_id": msg["id"], "subject": subject, "source": "pipeline"},
    )
    return {"conversation_id": conv["id"], "message_id": msg["id"]}
