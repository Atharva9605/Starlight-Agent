"""Campaign draft generate / revise / send helpers."""
from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from email import message_from_string
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

from azure_client import azure_manager

log = logging.getLogger("campaign_drafts")

_lock = threading.Lock()
_DRAFTS: dict[str, dict[str, Any]] = {}


def store_draft(draft: dict[str, Any]) -> str:
    draft_id = draft.get("id") or str(uuid.uuid4())
    draft["id"] = draft_id
    with _lock:
        _DRAFTS[draft_id] = draft
    return draft_id


def get_draft(draft_id: str) -> Optional[dict[str, Any]]:
    with _lock:
        return _DRAFTS.get(draft_id)


def pop_draft(draft_id: str) -> Optional[dict[str, Any]]:
    with _lock:
        return _DRAFTS.pop(draft_id, None)


def rewrite_eml(draft: dict[str, Any]) -> str:
    """
    Rebuild the .eml after a chat edit.

    Only called when the draft is dirty — an untouched draft keeps the .eml the
    generator produced. Uses the same builder as generation so the inline logo
    part and multipart/related structure survive the rewrite.
    """
    from config_manager import get_sender
    from email_html import build_eml_message, inline_css

    eml_path = draft.get("eml_path") or ""
    if not eml_path:
        outdir = draft.get("outdir") or "out_emails_api"
        os.makedirs(outdir, exist_ok=True)
        eml_path = os.path.join(outdir, f"{draft['id']}.eml")
        draft["eml_path"] = eml_path

    html = inline_css(draft.get("html") or "")
    draft["html"] = html

    sender = get_sender()
    logo_path = os.path.join(os.path.dirname(__file__), "starlight.jpg")

    msg = build_eml_message(
        subject=draft.get("subject") or "",
        html=html,
        from_addr=draft.get("from") or "",
        to_addr=draft.get("to") or "",
        logo_path=logo_path,
        embed_logo=sender.get("company_logo_url", "cid:company_logo") == "cid:company_logo",
    )

    Path(eml_path).write_text(msg.as_string(), encoding="utf-8")
    html_path = Path(eml_path).with_suffix(".html")
    html_path.write_text(html, encoding="utf-8")
    draft["html_path"] = str(html_path)
    draft["dirty"] = False
    return eml_path


def revise_draft_content(
    subject: str,
    html: str,
    instruction: str,
    company: str = "",
    website: str = "",
) -> dict[str, str]:
    """Ask Azure to revise subject + HTML per a chat instruction. Returns {subject, html}."""
    instruction = (instruction or "").strip()
    if not instruction:
        raise ValueError("Instruction is required")

    system = (
        "You are an expert B2B email editor for Starlight Linear LED. "
        "Revise the outbound HTML email based on the reviewer's instruction. "
        "Keep the overall HTML structure and inline styles intact unless the "
        "instruction requires a layout change. Do not invent fake product specs. "
        "Return ONLY a JSON object with keys: subject, html."
    )
    user = (
        f"Company: {company or '—'}\n"
        f"Website: {website or '—'}\n\n"
        f"Current subject:\n{subject}\n\n"
        f"Current HTML:\n{html[:45000]}\n\n"
        f"Reviewer instruction:\n{instruction}\n"
    )
    raw = azure_manager.chat_completion(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.3,
        max_tokens=4096,
        json_mode=True,
    )
    if not raw:
        raise RuntimeError("Empty revision response from Azure")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Could not parse revision JSON: {e}") from e

    new_subject = (data.get("subject") or subject).strip()
    new_html = (data.get("html") or html).strip()
    if not new_html:
        raise RuntimeError("Revision returned empty HTML")
    return {"subject": new_subject, "html": new_html}


def company_from_website(website: str, fallback: str = "") -> str:
    if fallback:
        return fallback
    try:
        return urlparse(website).netloc.replace("www.", "") or website
    except Exception:
        return website or ""


def body_text_from_eml(eml_path: str) -> str:
    try:
        raw = Path(eml_path).read_text(encoding="utf-8")
        msg = message_from_string(raw)
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                payload = part.get_payload(decode=True)
                if isinstance(payload, bytes):
                    return payload.decode("utf-8", errors="replace")
                return str(payload or "")
    except Exception:
        pass
    return ""
