"""
Background catalogue ingestion.

Vision ingestion of a scanned catalogue runs GPT-4o over every page and takes
around five minutes per file. Holding the upload request open for that long is
fragile: any proxy or browser idle timeout surfaces to the user as "Failed to
fetch" even though the server went on to finish the work successfully. So the
upload endpoint stages the files, hands them to a worker thread and returns
immediately; the client polls for progress.
"""
from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from typing import Any, Optional

from tenant import TenantContext, set_tenant_context

log = logging.getLogger("ingest_jobs")

# Finished jobs stay queryable for a while so a client that reconnects late
# still sees the outcome.
_JOB_TTL_SECONDS = 60 * 60

_lock = threading.Lock()
_JOBS: dict[str, dict[str, Any]] = {}


def get_job(job_id: str) -> Optional[dict[str, Any]]:
    with _lock:
        job = _JOBS.get(job_id)
        return dict(job) if job else None


def _update(job_id: str, **fields: Any) -> None:
    with _lock:
        job = _JOBS.get(job_id)
        if job is None:
            return
        job.update(fields)
        job["updated_at"] = time.time()


def _purge_expired() -> None:
    cutoff = time.time() - _JOB_TTL_SECONDS
    with _lock:
        for job_id in [
            j for j, v in _JOBS.items()
            if v["status"] != "running" and v["updated_at"] < cutoff
        ]:
            _JOBS.pop(job_id, None)


def start_ingest_job(
    staged_files: list[tuple[str, str]],
    ctx: TenantContext,
) -> str:
    """
    Queue ingestion of already-staged uploads.

    staged_files is a list of (original_filename, temp_path). The worker owns
    those temp files and deletes them when it is done.
    """
    _purge_expired()

    job_id = str(uuid.uuid4())
    now = time.time()
    with _lock:
        _JOBS[job_id] = {
            "id": job_id,
            "status": "running",
            "progress": 0.0,
            "message": "Queued…",
            "current_file": staged_files[0][0] if staged_files else "",
            "total_files": len(staged_files),
            "completed_files": 0,
            "results": [],
            "chunks": 0,
            "catalogues": [],
            "created_at": now,
            "updated_at": now,
        }

    thread = threading.Thread(
        target=_run_job,
        args=(job_id, staged_files, ctx),
        name=f"ingest-{job_id[:8]}",
        daemon=True,
    )
    thread.start()
    return job_id


def _run_job(
    job_id: str,
    staged_files: list[tuple[str, str]],
    ctx: TenantContext,
) -> None:
    # The worker runs outside the request, so it has to re-establish the tenant
    # context that add_chunks() and get_status() scope their rows by.
    set_tenant_context(ctx)

    from rag_uploader import process_pdf_to_chroma
    from vector_store import get_status

    total = len(staged_files) or 1
    results: list[dict[str, Any]] = []

    try:
        for index, (filename, temp_path) in enumerate(staged_files):
            _update(
                job_id,
                current_file=filename,
                completed_files=index,
                message=f"Reading {filename}…",
            )

            def on_progress(pct: float, msg: str, _i: int = index) -> None:
                _update(
                    job_id,
                    progress=(_i + max(0.0, min(1.0, pct))) / total,
                    message=msg,
                )

            try:
                class StagedFile:
                    name = filename

                    def read(self) -> bytes:
                        with open(temp_path, "rb") as fh:
                            return fh.read()

                success, message = process_pdf_to_chroma(
                    StagedFile(), progress_callback=on_progress
                )
            except Exception as exc:
                log.exception("Ingestion failed for %s", filename)
                success, message = False, f"Error processing catalogue: {exc}"
            finally:
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

            results.append(
                {"filename": filename, "success": success, "message": message}
            )
            _update(job_id, results=list(results), completed_files=index + 1)

        chunks, catalogues = get_status()
        succeeded = [r for r in results if r["success"]]
        _update(
            job_id,
            status="done" if succeeded else "error",
            progress=1.0,
            chunks=chunks,
            catalogues=catalogues,
            message=(
                f"Indexed {len(succeeded)} of {len(results)}"
                if succeeded
                else "; ".join(r["message"] for r in results) or "Ingestion failed"
            ),
        )
    except Exception as exc:
        log.exception("Ingestion job %s crashed", job_id)
        _update(job_id, status="error", message=str(exc), results=list(results))
    finally:
        # Nothing should leak if the loop exited early.
        for _, temp_path in staged_files:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
