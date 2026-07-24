"""
Structured AI event logging for observability.

Logs org_id, prompt_key, latency, tokens (when available), chunk ids, deployment.
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from contextlib import contextmanager
from typing import Any, Iterator, Optional

log = logging.getLogger("ai_events")

_EVENTS_DIR = os.path.join(os.path.dirname(__file__), "data", "ai_events")


def _org_id() -> str:
    try:
        from tenant import get_organization_id
        return get_organization_id()
    except Exception:
        return "unknown"


def _ensure_dir() -> str:
    os.makedirs(_EVENTS_DIR, exist_ok=True)
    return _EVENTS_DIR


def log_ai_event(
    event_type: str,
    *,
    prompt_key: str = "",
    latency_ms: float | None = None,
    chunk_ids: list[str] | None = None,
    deployment: str = "",
    success: bool = True,
    error: str = "",
    meta: dict[str, Any] | None = None,
) -> dict:
    from azure_client import azure_manager

    event = {
        "id": f"evt_{uuid.uuid4().hex[:12]}",
        "ts": time.time(),
        "event_type": event_type,
        "organization_id": _org_id(),
        "prompt_key": prompt_key,
        "latency_ms": latency_ms,
        "chunk_ids": chunk_ids or [],
        "deployment": deployment or azure_manager.get_chat_deployment(),
        "success": success,
        "error": error,
        "meta": meta or {},
    }

    try:
        path = os.path.join(_ensure_dir(), f"{_org_id()}.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception as exc:
        log.debug("Failed to persist AI event: %s", exc)

    log.info(
        "ai_event type=%s org=%s prompt=%s latency_ms=%s success=%s",
        event_type,
        event["organization_id"],
        prompt_key,
        latency_ms,
        success,
    )
    return event


@contextmanager
def timed_ai_event(event_type: str, **kwargs) -> Iterator[dict]:
    """Context manager that records latency automatically."""
    start = time.perf_counter()
    bag: dict[str, Any] = {"chunk_ids": [], "meta": {}}
    try:
        yield bag
        latency = (time.perf_counter() - start) * 1000
        log_ai_event(
            event_type,
            latency_ms=latency,
            chunk_ids=bag.get("chunk_ids") or kwargs.get("chunk_ids"),
            prompt_key=kwargs.get("prompt_key", ""),
            deployment=kwargs.get("deployment", ""),
            success=True,
            meta={**(kwargs.get("meta") or {}), **(bag.get("meta") or {})},
        )
    except Exception as exc:
        latency = (time.perf_counter() - start) * 1000
        log_ai_event(
            event_type,
            latency_ms=latency,
            prompt_key=kwargs.get("prompt_key", ""),
            success=False,
            error=str(exc),
            meta=kwargs.get("meta"),
        )
        raise


def list_recent_events(limit: int = 50, organization_id: str | None = None) -> list[dict]:
    org = organization_id or _org_id()
    path = os.path.join(_EVENTS_DIR, f"{org}.jsonl")
    if not os.path.exists(path):
        return []
    rows: list[dict] = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    return rows[-limit:]
