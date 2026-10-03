"""
Durable, server-side campaign runs.

A run lives in Postgres, not in the browser tab. The HTTP request that starts a
campaign only records it and hands it to a worker thread; the worker keeps
scraping, drafting and sending after the tab is gone. Clients attach with a
snapshot plus an event cursor, so reopening the tab (or opening another device)
picks the live view back up, and a finished run stays readable forever as the
"full run" record.

Why a dedicated thread per run: tenant context is a threading.local set by the
auth middleware on the event-loop thread, so a background asyncio task would
read whichever org handled the most recent request. The worker therefore owns
its own thread, its own event loop and its own tenant context.
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Optional

from tenant import TenantContext, set_tenant_context
from vector_store import _pg_conn, use_postgres

log = logging.getLogger("campaign_runs")

# Statuses a run can hold. 'running' means a worker should be alive for it.
ACTIVE_STATUSES = ("running", "paused")
TERMINAL_STATUSES = ("done", "stopped", "failed")

# A run whose heartbeat is older than this is considered orphaned (API restart)
# and may be adopted by another worker.
HEARTBEAT_STALE_SECONDS = 120

# Pipeline generator, injected by api.py to avoid a circular import.
_pipeline: Optional[Callable[..., Any]] = None

# run_id -> worker thread, so a resume never double-starts a live run.
_workers: dict[str, threading.Thread] = {}
_workers_lock = threading.Lock()


def set_pipeline(factory: Callable[..., Any]) -> None:
    """Register the async generator that yields campaign event payloads."""
    global _pipeline
    _pipeline = factory


def _require_postgres() -> None:
    if not use_postgres():
        raise RuntimeError(
            "Background campaigns require DATABASE_URL (PostgreSQL). "
            "Without it a run cannot outlive the browser tab."
        )


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: Any) -> Optional[str]:
    if isinstance(value, datetime):
        return value.isoformat()
    return value if value is None else str(value)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def init_campaign_run_tables() -> None:
    """Create the campaign run tables. Called from vector_store.init_db()."""
    if not use_postgres():
        return

    conn = _pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS campaign_runs (
                    id TEXT PRIMARY KEY,
                    organization_id TEXT NOT NULL,
                    created_by TEXT NOT NULL DEFAULT '',
                    created_by_email TEXT NOT NULL DEFAULT '',
                    file_name TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'running',
                    control TEXT NOT NULL DEFAULT 'run',
                    options JSONB NOT NULL DEFAULT '{}'::jsonb,
                    sender_email TEXT NOT NULL DEFAULT '',
                    total INTEGER NOT NULL DEFAULT 0,
                    error TEXT NOT NULL DEFAULT '',
                    worker_token TEXT NOT NULL DEFAULT '',
                    heartbeat_at TIMESTAMPTZ,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    finished_at TIMESTAMPTZ
                )
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_campaign_runs_org "
                "ON campaign_runs (organization_id, created_at DESC)"
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS campaign_run_leads (
                    run_id TEXT NOT NULL REFERENCES campaign_runs(id) ON DELETE CASCADE,
                    row_index INTEGER NOT NULL,
                    lead JSONB NOT NULL DEFAULT '{}'::jsonb,
                    status TEXT NOT NULL DEFAULT '',
                    state TEXT NOT NULL DEFAULT 'pending',
                    company TEXT NOT NULL DEFAULT '',
                    website TEXT NOT NULL DEFAULT '',
                    to_email TEXT NOT NULL DEFAULT '',
                    subject TEXT NOT NULL DEFAULT '',
                    html TEXT NOT NULL DEFAULT '',
                    product_count INTEGER NOT NULL DEFAULT 0,
                    product_sheet TEXT NOT NULL DEFAULT '',
                    stages JSONB NOT NULL DEFAULT '[]'::jsonb,
                    error TEXT NOT NULL DEFAULT '',
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (run_id, row_index)
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS campaign_run_events (
                    id BIGSERIAL PRIMARY KEY,
                    run_id TEXT NOT NULL REFERENCES campaign_runs(id) ON DELETE CASCADE,
                    payload JSONB NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_campaign_run_events_run "
                "ON campaign_run_events (run_id, id)"
            )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Lead state helpers
# ---------------------------------------------------------------------------

def _state_from_status(status: str) -> str:
    s = (status or "").lower()
    if "sent" in s or "✅" in s:
        return "sent"
    if "skip" in s or "discard" in s or "⏭" in s:
        return "skipped"
    if "fail" in s or "error" in s or "❌" in s:
        return "failed"
    if s:
        return "processing"
    return "pending"


def _settled_state(state: str) -> bool:
    """Sent or deliberately skipped. A failure is not settled — resume retries it."""
    return state in ("sent", "skipped")


# ---------------------------------------------------------------------------
# Run CRUD
# ---------------------------------------------------------------------------

def create_run(
    *,
    organization_id: str,
    leads: list[dict],
    options: dict[str, Any],
    file_name: str = "",
    created_by: str = "",
    created_by_email: str = "",
) -> str:
    _require_postgres()
    run_id = str(uuid.uuid4())
    conn = _pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO campaign_runs
                    (id, organization_id, created_by, created_by_email, file_name,
                     status, control, options, sender_email, total)
                VALUES (%s, %s, %s, %s, %s, 'running', 'run', %s, %s, %s)
                """,
                (
                    run_id,
                    organization_id,
                    created_by,
                    created_by_email,
                    file_name,
                    json.dumps(options or {}),
                    str(options.get("sender_email") or ""),
                    len(leads),
                ),
            )
            for idx, lead in enumerate(leads):
                row = dict(lead or {})
                cur.execute(
                    """
                    INSERT INTO campaign_run_leads
                        (run_id, row_index, lead, status, state, company, website)
                    VALUES (%s, %s, %s, '', 'pending', %s, %s)
                    """,
                    (
                        run_id,
                        idx,
                        json.dumps(row),
                        str(row.get("company") or row.get("name") or ""),
                        str(row.get("website") or ""),
                    ),
                )
        conn.commit()
    finally:
        conn.close()
    return run_id


_RUN_COLUMNS = """
    r.id, r.organization_id, r.created_by, r.created_by_email, r.file_name,
    r.status, r.control, r.options, r.sender_email, r.total, r.error,
    r.created_at, r.updated_at, r.finished_at, r.heartbeat_at
"""


def _run_row_to_dict(row: tuple, counts: dict[str, int]) -> dict[str, Any]:
    options = row[7] or {}
    if isinstance(options, str):
        options = json.loads(options)
    total = int(row[9] or 0)
    sent = counts.get("sent", 0)
    failed = counts.get("failed", 0)
    skipped = counts.get("skipped", 0)
    processed = sent + failed + skipped
    return {
        "id": row[0],
        "organization_id": row[1],
        "created_by": row[2],
        "created_by_email": row[3],
        "file_name": row[4],
        "status": row[5],
        "control": row[6],
        "options": options,
        "sender_email": row[8],
        "total": total,
        "error": row[10],
        "created_at": _iso(row[11]),
        "updated_at": _iso(row[12]),
        "finished_at": _iso(row[13]),
        "heartbeat_at": _iso(row[14]),
        "counts": {
            "total": total,
            "sent": sent,
            "failed": failed,
            "skipped": skipped,
            "processed": processed,
            "pending": max(0, total - processed),
            # What a resume would still work through: never-reached plus failed.
            "retriable": max(0, total - sent - skipped),
        },
    }


def _counts_for(cur, run_ids: Iterable[str]) -> dict[str, dict[str, int]]:
    ids = list(run_ids)
    if not ids:
        return {}
    cur.execute(
        """
        SELECT run_id, state, COUNT(*)
        FROM campaign_run_leads
        WHERE run_id = ANY(%s)
        GROUP BY run_id, state
        """,
        (ids,),
    )
    out: dict[str, dict[str, int]] = {rid: {} for rid in ids}
    for run_id, state, count in cur.fetchall():
        out.setdefault(run_id, {})[state] = int(count)
    return out


def get_run(run_id: str, organization_id: Optional[str] = None) -> Optional[dict[str, Any]]:
    if not use_postgres():
        return None
    conn = _pg_conn()
    try:
        with conn.cursor() as cur:
            params: list[Any] = [run_id]
            sql = f"SELECT {_RUN_COLUMNS} FROM campaign_runs r WHERE r.id = %s"
            if organization_id:
                sql += " AND r.organization_id = %s"
                params.append(organization_id)
            cur.execute(sql, params)
            row = cur.fetchone()
            if not row:
                return None
            counts = _counts_for(cur, [run_id]).get(run_id, {})
            return _run_row_to_dict(row, counts)
    finally:
        conn.close()


def list_runs(organization_id: str, limit: int = 50, status: Optional[str] = None) -> list[dict[str, Any]]:
    if not use_postgres():
        return []
    conn = _pg_conn()
    try:
        with conn.cursor() as cur:
            params: list[Any] = [organization_id]
            sql = f"SELECT {_RUN_COLUMNS} FROM campaign_runs r WHERE r.organization_id = %s"
            if status == "active":
                sql += " AND r.status = ANY(%s)"
                params.append(list(ACTIVE_STATUSES))
            elif status:
                sql += " AND r.status = %s"
                params.append(status)
            sql += " ORDER BY r.created_at DESC LIMIT %s"
            params.append(limit)
            cur.execute(sql, params)
            rows = cur.fetchall()
            counts = _counts_for(cur, [r[0] for r in rows])
            return [_run_row_to_dict(r, counts.get(r[0], {})) for r in rows]
    finally:
        conn.close()


def get_run_leads(run_id: str, include_html: bool | int = True) -> list[dict[str, Any]]:
    """
    Leads shaped the way the frontend campaign state expects them.

    `include_html` may be a row index, which ships the body for that lead only —
    a long run holds megabytes of markup, and clients fetch the rest on demand.
    """
    if not use_postgres():
        return []
    conn = _pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT row_index, lead, status, state, company, website, to_email,
                       subject, html, product_count, product_sheet, stages, error
                FROM campaign_run_leads
                WHERE run_id = %s
                ORDER BY row_index
                """,
                (run_id,),
            )
            out: list[dict[str, Any]] = []
            for row in cur.fetchall():
                lead = row[1] or {}
                if isinstance(lead, str):
                    lead = json.loads(lead)
                stages = row[11] or []
                if isinstance(stages, str):
                    stages = json.loads(stages)
                html = row[8] or ""
                if include_html is True:
                    send_html = html
                elif isinstance(include_html, int) and not isinstance(include_html, bool):
                    send_html = html if row[0] == include_html else ""
                else:
                    send_html = ""
                merged = dict(lead)
                merged.update(
                    {
                        "company": row[4] or lead.get("company") or "",
                        "website": row[5] or lead.get("website") or "",
                        "_row_index": row[0],
                        "_status": row[2] or "",
                        "_state": row[3] or "pending",
                        "_to": row[6] or "",
                        "_subject": row[7] or "",
                        "_preview_html": send_html,
                        "_has_preview": bool(html),
                        "_product_count": int(row[9] or 0),
                        "_product_sheet": row[10] or "",
                        "_stages": stages,
                        "_error": row[12] or "",
                    }
                )
                out.append(merged)
            return out
    finally:
        conn.close()


def run_snapshot(
    run_id: str,
    organization_id: Optional[str] = None,
    log_limit: int = 400,
) -> Optional[dict[str, Any]]:
    """Everything a client needs to render a run without replaying its events."""
    run = get_run(run_id, organization_id)
    if not run:
        return None
    conn = _pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT row_index FROM campaign_run_leads
                WHERE run_id = %s AND html <> ''
                ORDER BY row_index DESC LIMIT 1
                """,
                (run_id,),
            )
            row = cur.fetchone()
            newest_preview = int(row[0]) if row else -1
            cur.execute(
                "SELECT COALESCE(MAX(id), 0) FROM campaign_run_events WHERE run_id = %s",
                (run_id,),
            )
            cursor = int((cur.fetchone() or [0])[0] or 0)
            cur.execute(
                """
                SELECT payload FROM campaign_run_events
                WHERE run_id = %s AND payload->>'type' = 'log'
                ORDER BY id DESC LIMIT %s
                """,
                (run_id, log_limit),
            )
            logs = []
            for (payload,) in cur.fetchall():
                if isinstance(payload, str):
                    payload = json.loads(payload)
                logs.append(str(payload.get("message") or ""))
            logs.reverse()
    finally:
        conn.close()
    leads = get_run_leads(run_id, include_html=newest_preview)
    return {"run": run, "leads": leads, "logs": logs, "cursor": cursor}


def delete_run(run_id: str, organization_id: str) -> bool:
    if not use_postgres():
        return False
    conn = _pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM campaign_runs WHERE id = %s AND organization_id = %s",
                (run_id, organization_id),
            )
            deleted = cur.rowcount > 0
        conn.commit()
        return deleted
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Status / control
# ---------------------------------------------------------------------------

def set_status(run_id: str, status: str, error: str = "") -> None:
    if not use_postgres():
        return
    finished = status in TERMINAL_STATUSES
    conn = _pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE campaign_runs
                SET status = %s,
                    error = %s,
                    updated_at = NOW(),
                    finished_at = CASE WHEN %s THEN NOW() ELSE finished_at END
                WHERE id = %s
                """,
                (status, error or "", finished, run_id),
            )
        conn.commit()
    finally:
        conn.close()


def set_control(run_id: str, control: str, organization_id: Optional[str] = None) -> bool:
    if not use_postgres():
        return False
    conn = _pg_conn()
    try:
        with conn.cursor() as cur:
            params: list[Any] = [control, run_id]
            sql = "UPDATE campaign_runs SET control = %s, updated_at = NOW() WHERE id = %s"
            if organization_id:
                sql += " AND organization_id = %s"
                params.append(organization_id)
            cur.execute(sql, params)
            ok = cur.rowcount > 0
        conn.commit()
        return ok
    finally:
        conn.close()


def get_control(run_id: str) -> str:
    if not use_postgres():
        return "run"
    conn = _pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT control FROM campaign_runs WHERE id = %s", (run_id,))
            row = cur.fetchone()
            return (row[0] if row else "stop") or "run"
    finally:
        conn.close()


def heartbeat(run_id: str, token: str) -> None:
    if not use_postgres():
        return
    conn = _pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE campaign_runs SET heartbeat_at = NOW() WHERE id = %s AND worker_token = %s",
                (run_id, token),
            )
        conn.commit()
    finally:
        conn.close()


def claim_run(run_id: str) -> Optional[str]:
    """
    Atomically take ownership of a run, so two processes never send the same
    campaign twice. Returns a worker token, or None if someone else owns it.
    """
    if not use_postgres():
        return None
    token = str(uuid.uuid4())
    conn = _pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                UPDATE campaign_runs
                SET worker_token = %s,
                    heartbeat_at = NOW(),
                    status = 'running',
                    control = 'run',
                    updated_at = NOW()
                WHERE id = %s
                  AND (status <> 'running'
                       OR heartbeat_at IS NULL
                       OR heartbeat_at < NOW() - INTERVAL '{HEARTBEAT_STALE_SECONDS} seconds')
                """,
                (token, run_id),
            )
            claimed = cur.rowcount > 0
        conn.commit()
        return token if claimed else None
    finally:
        conn.close()


def orphaned_run_ids() -> list[tuple[str, str, str]]:
    """Runs left 'running' by a crashed/restarted API: (run_id, org_id, user_id)."""
    if not use_postgres():
        return []
    conn = _pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT id, organization_id, created_by
                FROM campaign_runs
                WHERE status = 'running'
                  AND (heartbeat_at IS NULL
                       OR heartbeat_at < NOW() - INTERVAL '{HEARTBEAT_STALE_SECONDS} seconds')
                ORDER BY created_at
                """
            )
            return [(r[0], r[1], r[2]) for r in cur.fetchall()]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

def _apply_event_to_lead(cur, run_id: str, payload: dict[str, Any]) -> None:
    etype = payload.get("type")
    idx = payload.get("row_index")
    if idx is None:
        return

    if etype == "status_update":
        status = str(payload.get("status") or "")
        cur.execute(
            """
            UPDATE campaign_run_leads
            SET status = %s, state = %s, updated_at = NOW()
            WHERE run_id = %s AND row_index = %s
            """,
            (status, _state_from_status(status), run_id, idx),
        )
    elif etype == "lead_resolved":
        cur.execute(
            """
            UPDATE campaign_run_leads
            SET website = COALESCE(NULLIF(%s, ''), website),
                company = COALESCE(NULLIF(%s, ''), company),
                updated_at = NOW()
            WHERE run_id = %s AND row_index = %s
            """,
            (
                str(payload.get("website") or ""),
                str(payload.get("company") or ""),
                run_id,
                idx,
            ),
        )
    elif etype == "preview_html":
        cur.execute(
            """
            UPDATE campaign_run_leads
            SET subject = %s,
                html = %s,
                to_email = COALESCE(NULLIF(%s, ''), to_email),
                company = COALESCE(NULLIF(%s, ''), company),
                website = COALESCE(NULLIF(%s, ''), website),
                product_count = %s,
                product_sheet = %s,
                updated_at = NOW()
            WHERE run_id = %s AND row_index = %s
            """,
            (
                str(payload.get("subject") or ""),
                str(payload.get("html") or ""),
                str(payload.get("to") or ""),
                str(payload.get("company") or ""),
                str(payload.get("website") or ""),
                int(payload.get("product_count") or 0),
                str(payload.get("product_sheet") or ""),
                run_id,
                idx,
            ),
        )
    elif etype == "stage":
        stage = {
            "stage": payload.get("stage"),
            "label": payload.get("label"),
            "state": payload.get("state") or "active",
        }
        # Replace the entry for this stage so the timeline stays one row per stage.
        cur.execute(
            "SELECT stages FROM campaign_run_leads WHERE run_id = %s AND row_index = %s",
            (run_id, idx),
        )
        row = cur.fetchone()
        stages = row[0] if row else []
        if isinstance(stages, str):
            stages = json.loads(stages)
        stages = [s for s in (stages or []) if s.get("stage") != stage["stage"]]
        stages.append(stage)
        error = str(payload.get("label") or "") if payload.get("state") == "error" else None
        cur.execute(
            """
            UPDATE campaign_run_leads
            SET stages = %s,
                error = COALESCE(%s, error),
                updated_at = NOW()
            WHERE run_id = %s AND row_index = %s
            """,
            (json.dumps(stages), error, run_id, idx),
        )


def append_event(run_id: str, payload: dict[str, Any]) -> int:
    """
    Persist one pipeline event and fold it into run/lead state.

    The stored copy of a preview drops the HTML body — the latest preview lives
    on the lead row, so replaying a long run never ships megabytes of markup.
    """
    if not use_postgres():
        return 0
    stored = dict(payload)
    if stored.get("type") == "preview_html":
        stored.pop("html", None)
        stored["html_in_snapshot"] = True

    conn = _pg_conn()
    try:
        with conn.cursor() as cur:
            _apply_event_to_lead(cur, run_id, payload)
            if payload.get("type") == "run_meta" and payload.get("sender_email"):
                cur.execute(
                    "UPDATE campaign_runs SET sender_email = %s, updated_at = NOW() WHERE id = %s",
                    (str(payload["sender_email"]), run_id),
                )
            cur.execute(
                "INSERT INTO campaign_run_events (run_id, payload) VALUES (%s, %s) RETURNING id",
                (run_id, json.dumps(stored)),
            )
            event_id = int(cur.fetchone()[0])
        conn.commit()
        return event_id
    finally:
        conn.close()


def events_since(run_id: str, cursor: int = 0, limit: int = 400) -> list[dict[str, Any]]:
    if not use_postgres():
        return []
    conn = _pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, payload FROM campaign_run_events
                WHERE run_id = %s AND id > %s
                ORDER BY id LIMIT %s
                """,
                (run_id, cursor, limit),
            )
            out = []
            for event_id, payload in cur.fetchall():
                if isinstance(payload, str):
                    payload = json.loads(payload)
                out.append({"id": int(event_id), "payload": payload})
            return out
    finally:
        conn.close()


def lead_html(run_id: str, row_index: int) -> str:
    if not use_postgres():
        return ""
    conn = _pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT html FROM campaign_run_leads WHERE run_id = %s AND row_index = %s",
                (run_id, row_index),
            )
            row = cur.fetchone()
            return (row[0] if row else "") or ""
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def is_worker_live(run_id: str) -> bool:
    with _workers_lock:
        thread = _workers.get(run_id)
    return bool(thread and thread.is_alive())


async def _drive_run(run_id: str, token: str) -> None:
    if _pipeline is None:
        raise RuntimeError("Campaign pipeline not registered")

    run = get_run(run_id)
    if not run:
        return
    options = run.get("options") or {}

    # Leads carry their persisted _status, which is what makes a resumed run
    # pass over everything already sent, skipped or discarded.
    leads = get_run_leads(run_id, include_html=False)

    pending = [l for l in leads if not _settled_state(str(l.get("_state") or "pending"))]
    if not pending:
        append_event(run_id, {"type": "log", "message": "Nothing left to send — run already complete."})
        append_event(run_id, {"type": "done", "message": "Processing complete!"})
        set_status(run_id, "done")
        return

    last_beat = 0.0
    final_status = "done"
    ended_early = False

    def control() -> str:
        return get_control(run_id)

    try:
        stream = _pipeline(
            leads,
            template=str(options.get("template") or "email_template.html"),
            delay=int(options.get("delay") or 0),
            sender_email=str(options.get("sender_email") or ""),
            recipient_override=str(options.get("recipient_override") or ""),
            attach_product_sheet=bool(options.get("attach_product_sheet", True)),
            autosend=True,
            control=control,
        )
        async for payload in stream:
            if payload.get("type") == "control":
                signal = payload.get("signal")
                final_status = "paused" if signal == "pause" else "stopped"
                ended_early = True
                await asyncio.to_thread(
                    append_event,
                    run_id,
                    {
                        "type": "log",
                        "message": "Run paused — resume picks up at the next unsent lead."
                        if signal == "pause"
                        else "Run stopped.",
                    },
                )
                break
            if payload.get("type") == "done":
                continue
            await asyncio.to_thread(append_event, run_id, payload)

            now = asyncio.get_running_loop().time()
            if now - last_beat > 20:
                last_beat = now
                await asyncio.to_thread(heartbeat, run_id, token)

        if not ended_early:
            await asyncio.to_thread(
                append_event, run_id, {"type": "done", "message": "Processing complete!"}
            )
        await asyncio.to_thread(set_status, run_id, final_status)
    except Exception as e:  # noqa: BLE001 - a run failure must be recorded, not raised
        log.exception("Campaign run %s failed", run_id)
        await asyncio.to_thread(
            append_event, run_id, {"type": "log", "message": f"Run failed: {e}"}
        )
        await asyncio.to_thread(set_status, run_id, "failed", str(e))


def start_worker(run_id: str, ctx: TenantContext) -> bool:
    """
    Start (or adopt) the worker for a run. Returns False when another worker
    already owns it.
    """
    _require_postgres()
    if is_worker_live(run_id):
        return False
    token = claim_run(run_id)
    if not token:
        return False

    def runner() -> None:
        set_tenant_context(ctx)
        try:
            asyncio.run(_drive_run(run_id, token))
        except Exception:  # noqa: BLE001
            log.exception("Campaign worker for %s crashed", run_id)
            try:
                set_status(run_id, "failed", "worker crashed")
            except Exception:  # noqa: BLE001
                pass
        finally:
            with _workers_lock:
                _workers.pop(run_id, None)

    thread = threading.Thread(target=runner, name=f"campaign-run-{run_id[:8]}", daemon=True)
    with _workers_lock:
        _workers[run_id] = thread
    thread.start()
    return True


def adopt_orphaned_runs() -> int:
    """
    On startup, pick up runs that were mid-flight when the API went down. The
    heartbeat claim keeps this safe if more than one process boots at once.
    """
    if not use_postgres():
        return 0
    adopted = 0
    for run_id, org_id, user_id in orphaned_run_ids():
        ctx = TenantContext(
            user_id=user_id or "system",
            organization_id=org_id,
            role="owner",
        )
        try:
            if start_worker(run_id, ctx):
                adopted += 1
                log.info("Adopted orphaned campaign run %s", run_id)
        except Exception:  # noqa: BLE001
            log.exception("Could not adopt campaign run %s", run_id)
    return adopted
