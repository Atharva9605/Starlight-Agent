"""Cheap pre-LLM gates for conversation drafting."""
from __future__ import annotations

import re

from schemas import DraftGateResult


def should_auto_draft(latest_body: str, subject: str = "") -> DraftGateResult:
    """Skip thanks / OOO / empty noise before spending tokens."""
    text = f"{subject}\n{latest_body}".strip().lower()
    if len(text) < 8:
        return DraftGateResult(should_draft=False, reason="Message too short")

    ooo_markers = (
        "out of office",
        "automatic reply",
        "auto-reply",
        "away from the office",
        "on vacation",
        "maternity leave",
    )
    if any(m in text for m in ooo_markers):
        return DraftGateResult(should_draft=False, reason="Out-of-office / auto-reply")

    thanks_only = re.sub(r"[^a-z\s]", "", text).strip()
    if thanks_only in {"thanks", "thank you", "thanks a lot", "ok thanks", "ok", "okay", "noted"}:
        return DraftGateResult(should_draft=False, reason="Low-signal acknowledgement")

    if len(text) < 40 and "thank" in text and "?" not in text:
        return DraftGateResult(should_draft=False, reason="Short thanks without question")

    return DraftGateResult(should_draft=True, reason="ok")
