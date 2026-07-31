import os
import json
import shutil
import tempfile
import asyncio
import re
from functools import partial
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Query
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse, HTMLResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd
import zipfile

from auth import (
    create_access_token,
    saas_auth_middleware,
    decode_token,
)
from org_store import (
    add_user_to_organization,
    authenticate_user,
    create_user_with_org,
    get_user_by_id,
    list_organization_members,
)
from tenant import get_organization_id, get_tenant_context, run_in_thread
from ingest_jobs import get_job, start_ingest_job
import gmail_oauth

# Import project modules
from scraper import scrape_and_process
from generator_v2 import generate_eml_from_record, query_rag_with_trace
from template_generator import generate_template_html
from send_eml_gsuite import send_email_gsuite, send_threaded_reply, check_gmail_health
from vector_store import init_db, get_status, clear_all, use_postgres
import conversation_store as conv_store
from conversation_agent import (
    generate_draft_for_conversation,
    generate_banner,
    record_outbound_from_pipeline,
)
from campaign_drafts import (
    body_text_from_eml,
    company_from_website,
    get_draft,
    pop_draft,
    revise_draft_content,
    rewrite_eml,
    store_draft,
)
from gmail_sync import run_gmail_sync, gmail_sync_loop
from config_manager import (
    get_config,
    save_config,
    reset_config,
    list_prompts,
    update_prompt,
    update_sender,
    get_sender,
    list_templates,
    get_template_content,
    save_template,
    delete_template,
    reset_template,
    get_template_variables,
    get_sample_preview_data,
    preview_template,
    preview_overrides,
)

app = FastAPI(title="Starlight AI-CRM Mailer API")

_cors_origins = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _cors_origins if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.middleware("http")(saas_auth_middleware)

_static_dir = Path(__file__).resolve().parent / "static"
_static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/media", StaticFiles(directory=str(_static_dir)), name="media")


@app.on_event("startup")
async def startup():
    init_db()
    asyncio.create_task(gmail_sync_loop())

# Legacy CORS block removed — configured above

@app.get("/")
async def root():
    return {
        "message": "Starlight AI-CRM Mailer API is Live!",
        "docs": "/docs",
        "status": "healthy",
        "version": "1.0.0",
        "release_date": "2026-07-25",
    }

class SignupRequest(BaseModel):
    email: str
    password: str
    name: str = ""
    organization_name: str = "My Organization"


class LoginRequest(BaseModel):
    email: str
    password: str


class SwitchOrgRequest(BaseModel):
    organization_id: str


class AddOrgMemberRequest(BaseModel):
    email: str
    password: str = ""
    name: str = ""
    role: str = "member"


_ADMIN_ROLES = frozenset({"owner", "admin"})


def _require_org_admin():
    """Raise 403 unless the current tenant is an org owner/admin."""
    from tenant import get_tenant_context

    ctx = get_tenant_context()
    if ctx.role not in _ADMIN_ROLES:
        raise HTTPException(
            status_code=403,
            detail="Only organization owners and admins can manage users",
        )
    return ctx


_auth_attempts: dict[str, list[float]] = {}
_AUTH_RATE_LIMIT = 20
_AUTH_WINDOW_SEC = 300


def _check_auth_rate(ip: str) -> None:
    import time
    now = time.time()
    attempts = _auth_attempts.get(ip, [])
    attempts = [t for t in attempts if now - t < _AUTH_WINDOW_SEC]
    if len(attempts) >= _AUTH_RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Too many attempts. Try again later.")
    attempts.append(now)
    _auth_attempts[ip] = attempts


@app.post("/api/auth/signup")
async def signup(body: SignupRequest, request: Request):
    _check_auth_rate(request.client.host if request.client else "unknown")
    try:
        user = create_user_with_org(
            body.email, body.password, body.name, body.organization_name
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    token = create_access_token(
        user["user_id"],
        user["email"],
        user["name"],
        user["organization_id"],
        user["role"],
    )
    return {
        "token": token,
        "user": {
            "id": user["user_id"],
            "email": user["email"],
            "name": user["name"],
        },
        "organization": {
            "id": user["organization_id"],
            "name": user["organization_name"],
            "role": user["role"],
        },
    }


@app.post("/api/auth/login")
async def login(body: LoginRequest, request: Request):
    _check_auth_rate(request.client.host if request.client else "unknown")
    user = authenticate_user(body.email, body.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = create_access_token(
        user["user_id"],
        user["email"],
        user["name"],
        user["organization_id"],
        user["role"],
    )
    return {
        "token": token,
        "user": {
            "id": user["user_id"],
            "email": user["email"],
            "name": user["name"],
        },
        "organization": {
            "id": user["organization_id"],
            "name": user["organization_name"],
            "role": user["role"],
        },
        "organizations": user.get("organizations", []),
    }


@app.get("/api/auth/me")
async def auth_me():
    from tenant import get_tenant_context

    ctx = get_tenant_context()
    user = get_user_by_id(ctx.user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    org = next(
        (o for o in user["organizations"] if o["organization_id"] == ctx.organization_id),
        None,
    )
    return {
        "user": {"id": user["user_id"], "email": user["email"], "name": user["name"]},
        "organization": org,
        "organizations": user["organizations"],
    }


@app.post("/api/auth/switch-org")
async def switch_org(body: SwitchOrgRequest):
    from tenant import get_tenant_context

    ctx = get_tenant_context()
    user = get_user_by_id(ctx.user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    membership = next(
        (o for o in user["organizations"] if o["organization_id"] == body.organization_id),
        None,
    )
    if not membership:
        raise HTTPException(status_code=403, detail="Not a member of this organization")
    token = create_access_token(
        ctx.user_id,
        ctx.email,
        ctx.name,
        membership["organization_id"],
        membership["role"],
    )
    return {
        "token": token,
        "organization": membership,
    }


@app.get("/api/org/members")
async def org_members():
    ctx = _require_org_admin()
    members = list_organization_members(ctx.organization_id)
    return {"members": members, "count": len(members)}


@app.post("/api/org/members")
async def org_add_member(body: AddOrgMemberRequest):
    ctx = _require_org_admin()
    try:
        member = add_user_to_organization(
            organization_id=ctx.organization_id,
            email=body.email,
            password=body.password,
            name=body.name,
            role=body.role,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {"member": member}


@app.get("/api/integrations/gmail/authorize")
async def gmail_authorize():
    org_id = get_organization_id()
    return gmail_oauth.get_authorize_url(org_id)


@app.get("/api/integrations/gmail/callback")
async def gmail_callback(code: str = "", state: str = "", error: str = ""):
    frontend = os.getenv("FRONTEND_URL", "http://localhost:5173").rstrip("/")
    if error:
        return RedirectResponse(url=f"{frontend}/settings?gmail_error={error}")
    try:
        result = gmail_oauth.handle_oauth_callback(code, state)
        return RedirectResponse(
            url=f"{frontend}/settings?gmail_connected={result.get('connected_email', '')}"
        )
    except Exception as e:
        return RedirectResponse(url=f"{frontend}/settings?gmail_error={str(e)[:100]}")


@app.get("/api/integrations/gmail/status")
async def gmail_integration_status():
    return gmail_oauth.get_integration_status(get_organization_id())


@app.delete("/api/integrations/gmail")
async def gmail_disconnect():
    gmail_oauth.disconnect(get_organization_id())
    return {"status": "disconnected"}


# ---------------------------------------------------------------------------
# Business Configuration — Prompts, Sender, Templates
# ---------------------------------------------------------------------------

class PromptUpdate(BaseModel):
    content: str

class PromptBulkUpdate(BaseModel):
    prompts: dict  # key -> content string

class SenderUpdate(BaseModel):
    sender_name: Optional[str] = None
    sender_company: Optional[str] = None
    sender_phone: Optional[str] = None
    sender_website: Optional[str] = None
    sender_email: Optional[str] = None
    company_logo_url: Optional[str] = None
    subject_fallback: Optional[str] = None

class TemplateSave(BaseModel):
    content: str
    label: Optional[str] = None

class TemplatePreviewRequest(BaseModel):
    template_name: Optional[str] = None
    template_content: Optional[str] = None
    sample_data: Optional[dict] = None


class EmailPreviewRequest(BaseModel):
    website: str
    template_name: Optional[str] = "email_template.html"
    template_content: Optional[str] = None
    prompt_overrides: Optional[dict] = None
    recipient_override: Optional[str] = None
    sender_email: Optional[str] = None
    sender_name: Optional[str] = None
    compare_saved: bool = True


class TemplateGenerateRequest(BaseModel):
    instructions: str
    style: Optional[str] = "modern"
    reference_template: Optional[str] = None


class RagQueryRequest(BaseModel):
    client_desc: str
    k: int = 5


class CreateConversationRequest(BaseModel):
    email: str
    company: Optional[str] = ""
    website: Optional[str] = ""
    subject: Optional[str] = ""
    template_name: Optional[str] = "email_template.html"
    profile_json: Optional[dict] = None


class AddMessageRequest(BaseModel):
    direction: str = "inbound"
    body_text: str = ""
    body_html: Optional[str] = ""
    subject: Optional[str] = ""
    sender_email: Optional[str] = ""


class DraftUpdateRequest(BaseModel):
    subject: Optional[str] = None
    body_text: Optional[str] = None
    body_html: Optional[str] = None
    banner_html: Optional[str] = None


class BannerGenerateRequest(BaseModel):
    instructions: str


class GenerateDraftRequest(BaseModel):
    instructions: str = ""


def _require_conversations_db():
    if not use_postgres():
        raise HTTPException(
            status_code=503,
            detail="Conversations require DATABASE_URL (PostgreSQL). Set it in HuggingFace Space secrets.",
        )


@app.get("/api/config")
async def get_full_config():
    """Return full business config (prompts + sender)."""
    return get_config()


@app.get("/api/config/sender")
async def get_sender_info():
    return get_sender()


@app.put("/api/config/sender")
async def put_sender_info(body: SenderUpdate):
    updates = {k: v for k, v in body.model_dump().items() if v is not None}
    config = update_sender(updates)
    return {"status": "success", "sender": config.get("sender", {})}


class BrandingUpdate(BaseModel):
    display_name: Optional[str] = None
    logo_url: Optional[str] = None
    accent_color: Optional[str] = None


@app.get("/api/config/branding")
async def get_branding_info():
    from config_manager import get_branding
    return get_branding()


@app.put("/api/config/branding")
async def put_branding_info(body: BrandingUpdate):
    from config_manager import update_branding, get_branding
    updates = {k: v for k, v in body.model_dump().items() if v is not None}
    update_branding(updates)
    return {"status": "success", "branding": get_branding()}


@app.get("/api/ai/events")
async def ai_events(limit: int = Query(50, ge=1, le=200)):
    from ai_events import list_recent_events
    return {"events": list_recent_events(limit=limit)}


@app.get("/api/config/prompts")
async def get_all_prompts():
    """List all editable prompts with metadata and current content."""
    return {"prompts": list_prompts()}


@app.get("/api/config/prompts/{key}")
async def get_single_prompt(key: str):
    prompts = {p["key"]: p for p in list_prompts()}
    if key not in prompts:
        raise HTTPException(status_code=404, detail=f"Unknown prompt: {key}")
    return prompts[key]


@app.put("/api/config/prompts/{key}")
async def put_single_prompt(key: str, body: PromptUpdate):
    try:
        update_prompt(key, body.content)
        return {"status": "success", "key": key}
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.put("/api/config/prompts")
async def put_bulk_prompts(body: PromptBulkUpdate):
    for key, content in body.prompts.items():
        try:
            update_prompt(key, content)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))
    return {"status": "success", "updated": list(body.prompts.keys())}


@app.post("/api/config/reset")
async def reset_all_config():
    config = reset_config()
    return {"status": "success", "message": "All prompts and sender info reset to defaults.", "config": config}


@app.get("/api/templates")
async def get_templates():
    return {"templates": list_templates()}


@app.get("/api/templates/variables")
async def get_variables():
    return {
        "variables": get_template_variables(),
        "sample_data": get_sample_preview_data(),
    }


@app.post("/api/templates/preview")
async def preview_email_template(body: TemplatePreviewRequest):
    try:
        html = preview_template(
            template_content=body.template_content,
            template_name=body.template_name,
            sample_data=body.sample_data,
        )
        return HTMLResponse(content=html)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/templates/preview/json")
async def preview_email_template_json(body: TemplatePreviewRequest):
    try:
        html = preview_template(
            template_content=body.template_content,
            template_name=body.template_name,
            sample_data=body.sample_data,
        )
        return {"html": html}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


def _strip_html(html: str) -> str:
    text = re.sub(r"<[^>]+>", " ", html or "")
    return re.sub(r"\s+", " ", text).strip()


def _build_preview_result(trace_info: dict) -> dict:
    return {
        "subject": trace_info.get("subject", ""),
        "html": trace_info.get("html", ""),
        "from": trace_info.get("from", ""),
        "to": trace_info.get("to", ""),
        "company": trace_info.get("company", ""),
        "website": trace_info.get("website", ""),
        "rag_trace": {
            "hyde_doc": trace_info.get("hyde_doc", ""),
            "combined_query": trace_info.get("combined_query", ""),
            "chunks": trace_info.get("chunks", []),
            "product_refs": trace_info.get("product_refs", []),
            "creative_draft": trace_info.get("creative_draft", ""),
        },
    }


def _compare_previews(saved: dict | None, draft: dict) -> dict:
    if not saved:
        return {"has_changes": False, "summary": []}
    summary = []
    if saved.get("subject") != draft.get("subject"):
        summary.append({
            "field": "subject",
            "saved": saved.get("subject", ""),
            "draft": draft.get("subject", ""),
        })
    saved_text = _strip_html(saved.get("html", ""))
    draft_text = _strip_html(draft.get("html", ""))
    if saved_text != draft_text:
        summary.append({
            "field": "body",
            "saved_preview": saved_text[:600],
            "draft_preview": draft_text[:600],
        })
    if saved.get("from") != draft.get("from"):
        summary.append({
            "field": "from",
            "saved": saved.get("from", ""),
            "draft": draft.get("from", ""),
        })
    return {"has_changes": bool(summary), "summary": summary}


async def _run_email_preview(
    website: str,
    template_name: str,
    template_content: str | None,
    prompt_overrides: dict | None,
    sender_overrides: dict | None,
    recipient_override: str | None,
) -> dict:
    outdir = tempfile.mkdtemp(prefix="starlight_preview_")
    try:
        with preview_overrides(prompts=prompt_overrides, sender=sender_overrides):
            scraped_data = await run_in_thread(scrape_and_process, website)
            if not scraped_data:
                raise HTTPException(status_code=400, detail="Scraper failed for this website.")
            if recipient_override:
                scraped_data["emails"] = recipient_override

            gen_fn = partial(
                generate_eml_from_record,
                scraped_data,
                1,
                outdir,
                template_name,
                template_content=template_content,
            )
            result = await run_in_thread(gen_fn)
            if not result:
                raise HTTPException(status_code=500, detail="Email generation failed.")
            _, trace_info = result
            return _build_preview_result(trace_info)
    finally:
        shutil.rmtree(outdir, ignore_errors=True)


@app.post("/api/preview-email")
async def preview_email(body: EmailPreviewRequest):
    """Generate a full pipeline email preview without sending."""
    website = (body.website or "").strip()
    if not website:
        raise HTTPException(status_code=400, detail="website is required")

    template_name = body.template_name or "email_template.html"
    saved_sender = get_sender()
    sender_overrides: dict[str, str] = {}
    if body.sender_email and body.sender_email.strip() != saved_sender.get("sender_email", ""):
        sender_overrides["sender_email"] = body.sender_email.strip()
    if body.sender_name and body.sender_name.strip() != saved_sender.get("sender_name", ""):
        sender_overrides["sender_name"] = body.sender_name.strip()
    sender_overrides = sender_overrides or None

    has_draft_changes = (
        bool(body.prompt_overrides)
        or bool(body.template_content)
        or bool(sender_overrides)
        or bool(body.recipient_override and body.recipient_override.strip())
    )
    compare_saved = body.compare_saved and has_draft_changes

    try:
        saved_result = None
        if compare_saved:
            saved_result = await _run_email_preview(
                website,
                template_name,
                None,
                None,
                None,
                body.recipient_override,
            )

        draft_result = await _run_email_preview(
            website,
            template_name,
            body.template_content,
            body.prompt_overrides,
            sender_overrides,
            body.recipient_override,
        )

        return {
            "saved": saved_result,
            "draft": draft_result,
            "changes": _compare_previews(saved_result, draft_result),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/templates/{name}")
async def get_template(name: str):
    try:
        content = get_template_content(name)
        meta = next((t for t in list_templates() if t["name"] == name), {"name": name})
        return {**meta, "content": content}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Template not found: {name}")


@app.put("/api/templates/{name}")
async def put_template(name: str, body: TemplateSave):
    try:
        result = save_template(name, body.content, body.label)
        return {"status": "success", **result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/api/templates/{name}")
async def remove_template(name: str):
    try:
        delete_template(name)
        return {"status": "success", "message": f"Template '{name}' removed."}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Template not found: {name}")


@app.post("/api/templates/{name}/reset")
async def reset_builtin_template(name: str):
    try:
        content = reset_template(name)
        return {"status": "success", "name": name, "content": content}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Template not found: {name}")


@app.post("/api/templates/generate")
async def generate_template_ai(body: TemplateGenerateRequest):
    try:
        content = await run_in_thread(
            generate_template_html,
            body.instructions,
            body.style,
            body.reference_template,
        )
        return {"content": content}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/rag/query")
async def rag_query(body: RagQueryRequest):
    try:
        result = await asyncio.to_thread(query_rag_with_trace, body.client_desc, body.k)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/kb-status")
async def get_kb():
    chunks, catalogues = get_status()
    from vector_store import store_info
    return {
        "chunks": chunks,
        "catalogues": catalogues,
        "store": store_info(),
    }

@app.post("/api/upload-catalogue")
@app.post("/api/upload-catalogues")
async def upload_catalogues(files: List[UploadFile] = File(...)):
    """
    Stage the uploads and hand them to a background worker.

    A scanned catalogue takes minutes of GPT-4o Vision per file. Returning the
    job id straight away keeps the request short, so a proxy or browser idle
    timeout can no longer report "Failed to fetch" for work that is actually
    still running. The client polls /api/catalogue-jobs/{id} for progress.
    """
    staged: list[tuple[str, str]] = []
    for file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            shutil.copyfileobj(file.file, tmp)
            staged.append((file.filename or "catalogue.pdf", tmp.name))

    job_id = start_ingest_job(staged, get_tenant_context())
    return JSONResponse(
        status_code=202,
        content={
            "job_id": job_id,
            "total_files": len(staged),
            "status": "running",
        },
    )


@app.get("/api/catalogue-jobs/{job_id}")
async def catalogue_job_status(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown or expired job")
    return job


@app.post("/api/clear-kb")
async def clear_kb():
    try:
        clear_all()
        return {"status": "success", "message": "Knowledge base cleared."}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/api/stop-processing")
async def stop_processing():
    # Implementation depends on how you want to handle cancellation.
    # For now, return success as a placeholder.
    return {"status": "success", "message": "Stop signal received."}

@app.post("/api/reset-queue")
async def reset_queue():
    outdir = os.path.join("out_emails_api", get_organization_id())
    if os.path.exists(outdir):
        shutil.rmtree(outdir)
    os.makedirs(outdir, exist_ok=True)
    return {"status": "success", "message": "Queue and output directory reset."}

@app.get("/api/download-emls")
async def download_emls():
    outdir = os.path.join("out_emails_api", get_organization_id())
    if not os.path.exists(outdir) or not os.listdir(outdir):
        return JSONResponse(status_code=404, content={"error": "No emails found to download."})
    
    zip_path = "starlight_emls.zip"
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for root, dirs, files in os.walk(outdir):
            for file in files:
                zipf.write(os.path.join(root, file), file)
    
    return FileResponse(zip_path, media_type='application/zip', filename="starlight_emls.zip")

def _normalize_lead_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Map common Excel headers onto website / email / company."""
    rename: dict = {}
    for col in df.columns:
        key = str(col).strip().lower().replace("-", "_").replace(" ", "_")
        if key in ("website", "web_site", "url", "site", "website_url"):
            rename[col] = "website"
        elif key in ("email", "e_mail", "mail", "emails", "email_address", "contact_email"):
            rename[col] = "email"
        elif key in ("company", "company_name", "organisation", "organization", "org"):
            rename[col] = "company"
    if rename:
        df = df.rename(columns=rename)
    # Keep a single canonical column if duplicates appear after rename
    return df.loc[:, ~df.columns.duplicated()]


def _lead_email_from_row(lead: dict | None) -> str:
    """Prefer an explicit email from the uploaded sheet when present."""
    if not lead:
        return ""
    for key, val in lead.items():
        if str(key).strip().lower() not in (
            "email",
            "emails",
            "e_mail",
            "mail",
            "email_address",
            "contact_email",
        ):
            continue
        if val is None:
            continue
        text = str(val).strip()
        if not text or text.lower() in ("nan", "none", "null"):
            continue
        first = text.split(",")[0].strip()
        if "@" in first:
            return first
    return ""


def _apply_recipient(scraped_data: dict, *, recipient_override: str | None = None, lead: dict | None = None) -> None:
    """
    Resolve who the email goes to:
    1) global recipient_override (test sends)
    2) email column on the lead row
    3) otherwise leave scraped emails alone
    """
    override = (recipient_override or "").strip()
    if override:
        scraped_data["emails"] = override
        return
    lead_email = _lead_email_from_row(lead)
    if lead_email:
        scraped_data["emails"] = lead_email


@app.post("/api/upload-leads")
async def upload_leads(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    
    try:
        df = _normalize_lead_dataframe(pd.read_excel(tmp_path))
        if "website" not in df.columns:
            return JSONResponse(status_code=400, content={"error": "Excel file must have a 'website' column."})
        
        df['status'] = "⏳ Pending"
        df['notes'] = ""
        records = df.fillna("").to_dict(orient="records")
        return {"leads": records}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    finally:
        os.remove(tmp_path)


class ProcessRequest(BaseModel):
    leads: List[dict]
    sender_email: str = ""
    recipient_override: Optional[str] = ""
    template: str = "email_template.html"
    delay: int = 3
    autosend: bool = True


class CampaignGenerateRequest(BaseModel):
    lead: dict
    template: str = "email_template.html"
    recipient_override: Optional[str] = ""
    sender_email: Optional[str] = ""
    row_index: int = 0


class CampaignReviseRequest(BaseModel):
    message: str


@app.post("/api/campaign/generate")
async def campaign_generate(req: CampaignGenerateRequest):
    """Scrape + generate one outbound draft. Does not send."""
    website = (req.lead or {}).get("website", "").strip()
    if not website:
        raise HTTPException(status_code=400, detail="Lead must include a website")

    org_id = get_organization_id()
    outdir = os.path.join("out_emails_api", org_id, "drafts")
    os.makedirs(outdir, exist_ok=True)

    scraped_data = await run_in_thread(scrape_and_process, website)
    if not scraped_data:
        raise HTTPException(status_code=400, detail="Scraper failed for this website.")

    _apply_recipient(scraped_data, recipient_override=req.recipient_override, lead=req.lead)
    if req.lead.get("company"):
        scraped_data.setdefault("company", req.lead["company"])
    scraped_data["website"] = website

    result = await run_in_thread(
        generate_eml_from_record, scraped_data, req.row_index + 1, outdir, req.template
    )
    if not result:
        raise HTTPException(status_code=500, detail="Email generation failed.")
    eml_path, trace_info = result
    html = (trace_info or {}).get("html") or ""
    if not html:
        html_path = Path(eml_path).with_suffix(".html")
        if html_path.exists():
            html = html_path.read_text(encoding="utf-8")

    to_email = str(scraped_data.get("emails", "")).split(",")[0].strip()
    company = company_from_website(
        website, req.lead.get("company") or scraped_data.get("company") or ""
    )
    draft = {
        "organization_id": org_id,
        "outdir": outdir,
        "eml_path": eml_path,
        "html_path": str(Path(eml_path).with_suffix(".html")),
        "subject": (trace_info or {}).get("subject") or "",
        "html": html,
        "from": (trace_info or {}).get("from") or "",
        "to": to_email,
        "website": website,
        "company": company,
        "template": req.template,
        "sender_email": req.sender_email or "",
        "scraped_data": scraped_data,
        "row_index": req.row_index,
        "chat": [],
    }
    draft_id = store_draft(draft)
    return {
        "draft_id": draft_id,
        "subject": draft["subject"],
        "html": draft["html"],
        "to": draft["to"],
        "from": draft["from"],
        "website": website,
        "company": company,
        "row_index": req.row_index,
    }


@app.post("/api/campaign/drafts/{draft_id}/revise")
async def campaign_revise(draft_id: str, body: CampaignReviseRequest):
    draft = get_draft(draft_id)
    if not draft or draft.get("organization_id") != get_organization_id():
        raise HTTPException(status_code=404, detail="Draft not found")
    try:
        revised = await run_in_thread(
            revise_draft_content,
            draft["subject"],
            draft["html"],
            body.message,
            draft.get("company") or "",
            draft.get("website") or "",
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    draft["subject"] = revised["subject"]
    draft["html"] = revised["html"]
    draft["dirty"] = True
    draft.setdefault("chat", []).append({"role": "user", "content": body.message})
    draft["chat"].append({"role": "assistant", "content": "Updated the email."})
    await run_in_thread(rewrite_eml, draft)
    return {
        "draft_id": draft_id,
        "subject": draft["subject"],
        "html": draft["html"],
        "to": draft["to"],
        "chat": draft["chat"],
    }


@app.post("/api/campaign/drafts/{draft_id}/send")
async def campaign_send(draft_id: str):
    draft = get_draft(draft_id)
    if not draft or draft.get("organization_id") != get_organization_id():
        raise HTTPException(status_code=404, detail="Draft not found")

    # An unedited draft already has a correctly built .eml from the generator.
    if draft.get("dirty"):
        await run_in_thread(rewrite_eml, draft)

    send_result = await run_in_thread(
        send_email_gsuite,
        draft["eml_path"],
        draft.get("sender_email") or "",
        get_organization_id(),
    )
    if not send_result.get("success"):
        raise HTTPException(
            status_code=500,
            detail=send_result.get("error") or "GSuite sending failed",
        )

    recipient_email = (draft.get("to") or "").strip()
    if use_postgres() and recipient_email:
        try:
            await asyncio.to_thread(
                record_outbound_from_pipeline,
                client_email=recipient_email,
                client_company=draft.get("company") or "",
                client_website=draft.get("website") or "",
                subject=draft.get("subject") or "",
                body_html=draft.get("html") or "",
                body_text=body_text_from_eml(draft["eml_path"])
                or _strip_html(draft.get("html") or ""),
                profile_json=draft.get("scraped_data") or {},
                template_name=draft.get("template") or "email_template.html",
                gmail_message_id=send_result.get("message_id") or "",
                gmail_thread_id=send_result.get("thread_id") or "",
            )
        except Exception as conv_err:
            # Send succeeded — don't fail the request on CRM write.
            log = __import__("logging").getLogger("api")
            log.warning("conversation not saved after send: %s", conv_err)

    pop_draft(draft_id)
    return {
        "status": "sent",
        "message_id": send_result.get("message_id"),
        "thread_id": send_result.get("thread_id"),
        "to": recipient_email,
    }


@app.delete("/api/campaign/drafts/{draft_id}")
async def campaign_discard(draft_id: str):
    draft = get_draft(draft_id)
    if not draft or draft.get("organization_id") != get_organization_id():
        raise HTTPException(status_code=404, detail="Draft not found")
    pop_draft(draft_id)
    return {"status": "discarded"}


@app.post("/api/process-stream")
async def process_leads(req: ProcessRequest):
    """
    Initiates processing of leads.
    Returns a Server-Sent Events (SSE) stream.
    When autosend=false, drafts are generated and emitted but not sent
    (prefer /api/campaign/generate for interactive review).
    """
    async def event_stream():
        org_id = get_organization_id()
        outdir = os.path.join("out_emails_api", org_id)
        os.makedirs(outdir, exist_ok=True)
        
        for idx, row in enumerate(req.leads):
            website = row.get('website', '')
            if not website:
                continue
                
            yield f"data: {json.dumps({'type': 'status_update', 'row_index': idx, 'status': '⚙️ Processing...'})}\n\n"
            
            try:
                yield f"data: {json.dumps({'type': 'log', 'message': f'Scraping: {website}'})}\n\n"
                
                # Run sync scraper in a thread to not block asyncio
                scraped_data = await run_in_thread(scrape_and_process, website)
                if not scraped_data:
                    raise Exception("Scraper failed to return data.")
                
                _apply_recipient(
                    scraped_data,
                    recipient_override=req.recipient_override,
                    lead=row,
                )
                
                yield f"data: {json.dumps({'type': 'log', 'message': f'Generating EML for: {website}'})}\n\n"
                eml_path, trace_info = await run_in_thread(
                    generate_eml_from_record, scraped_data, idx + 1, outdir, req.template
                )
                
                if not eml_path or not os.path.exists(eml_path):
                    raise Exception("EML generation failed.")
                
                # Load HTML preview
                html_content = ""
                html_path = Path(eml_path).with_suffix(".html")
                if html_path.exists():
                    with open(html_path, 'r', encoding='utf-8') as f:
                        html_content = f.read()
                    
                    yield f"data: {json.dumps({'type': 'preview_html', 'row_index': idx, 'subject': (trace_info or {}).get('subject', ''), 'html': html_content})}\n\n"
                
                # Send RAG Trace Info
                yield f"data: {json.dumps({'type': 'rag_trace', 'row_index': idx, 'data': {'company': scraped_data.get('company', website), 'website': website, 'trace_info': trace_info}})}\n\n"

                if not req.autosend:
                    to_email = str(scraped_data.get("emails", "")).split(",")[0].strip()
                    draft_id = store_draft({
                        "organization_id": org_id,
                        "outdir": outdir,
                        "eml_path": eml_path,
                        "html_path": str(html_path),
                        "subject": (trace_info or {}).get("subject") or "",
                        "html": html_content,
                        "from": (trace_info or {}).get("from") or "",
                        "to": to_email,
                        "website": website,
                        "company": company_from_website(
                            website, row.get("company") or scraped_data.get("company") or ""
                        ),
                        "template": req.template,
                        "sender_email": req.sender_email or "",
                        "scraped_data": scraped_data,
                        "row_index": idx,
                        "chat": [],
                    })
                    yield f"data: {json.dumps({'type': 'draft_ready', 'row_index': idx, 'draft_id': draft_id, 'subject': (trace_info or {}).get('subject', ''), 'html': html_content, 'to': to_email})}\n\n"
                    yield f"data: {json.dumps({'type': 'status_update', 'row_index': idx, 'status': '📝 Ready for review'})}\n\n"
                    continue
                
                yield f"data: {json.dumps({'type': 'log', 'message': f'Sending email via GSuite for {website}'})}\n\n"
                send_result = await run_in_thread(
                    send_email_gsuite, eml_path, req.sender_email, org_id
                )

                if not send_result.get("success"):
                    raise Exception(send_result.get("error", "GSuite sending failed."))

                recipient_email = str(scraped_data.get("emails", "")).split(",")[0].strip()
                subject = ""
                body_text = ""
                if html_path.exists():
                    from email import message_from_string
                    with open(eml_path, "r", encoding="utf-8") as f:
                        eml_raw = f.read()
                    eml_msg = message_from_string(eml_raw)
                    subject = eml_msg.get("Subject", "")
                    for part in eml_msg.walk():
                        if part.get_content_type() == "text/plain":
                            body_text = part.get_payload(decode=True).decode("utf-8", errors="replace")
                            break

                if use_postgres() and recipient_email:
                    try:
                        await asyncio.to_thread(
                            record_outbound_from_pipeline,
                            client_email=recipient_email,
                            client_company=row.get("company")
                            or scraped_data.get("company")
                            or urlparse(website).netloc.replace("www.", ""),
                            client_website=website,
                            subject=subject,
                            body_html=html_content,
                            body_text=body_text,
                            profile_json=scraped_data,
                            template_name=req.template,
                            gmail_message_id=send_result.get("message_id") or "",
                            gmail_thread_id=send_result.get("thread_id") or "",
                        )
                        yield f"data: {json.dumps({'type': 'log', 'message': f'Conversation created for {recipient_email}'})}\n\n"
                    except Exception as conv_err:
                        yield f"data: {json.dumps({'type': 'log', 'message': f'Warning: conversation not saved: {conv_err}'})}\n\n"

                yield f"data: {json.dumps({'type': 'log', 'message': f'Email sent successfully for: {website}'})}\n\n"
                yield f"data: {json.dumps({'type': 'status_update', 'row_index': idx, 'status': '✅ Sent'})}\n\n"
                
            except Exception as e:
                err_msg = str(e)
                if "Content filter triggered" in err_msg:
                    log_msg = f"Policy Violation: Azure's safety filters flagged the content for {website}. This often happens if the website content or our prompt looks suspicious to the AI."
                    status_msg = "❌ Safety Filter Triggered"
                else:
                    log_msg = f"Error processing {website}: {err_msg}"
                    status_msg = f"❌ Failed: {err_msg[:50]}..."
                
                yield f"data: {json.dumps({'type': 'log', 'message': log_msg})}\n\n"
                yield f"data: {json.dumps({'type': 'status_update', 'row_index': idx, 'status': status_msg})}\n\n"
            
            await asyncio.sleep(req.delay)
            
        yield f"data: {json.dumps({'type': 'done', 'message': 'Processing complete!'})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Conversations CRM
# ---------------------------------------------------------------------------

@app.get("/api/conversations")
async def list_conversations_api(
    status: Optional[str] = None,
    search: Optional[str] = None,
    mailbox_only: bool = False,
):
    _require_conversations_db()
    try:
        conversations = await asyncio.to_thread(
            conv_store.list_conversations, status, search, mailbox_only
        )
        return {"conversations": conversations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/conversations/{conversation_id}")
async def get_conversation_api(conversation_id: str):
    _require_conversations_db()
    conv = await asyncio.to_thread(conv_store.get_conversation, conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv


@app.post("/api/conversations")
async def create_conversation_api(body: CreateConversationRequest):
    _require_conversations_db()
    try:
        client = await asyncio.to_thread(
            conv_store.upsert_client,
            body.email,
            body.company or "",
            body.website or "",
            body.profile_json,
        )
        conv = await asyncio.to_thread(
            conv_store.create_conversation,
            client["id"],
            body.subject or "",
            body.template_name or "email_template.html",
        )
        await asyncio.to_thread(
            conv_store.add_timeline_event,
            conv["id"],
            "agent_note",
            {"note": "Manual conversation created"},
        )
        return {"client": client, "conversation": conv}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/conversations/{conversation_id}/messages")
async def add_message_api(conversation_id: str, body: AddMessageRequest):
    _require_conversations_db()
    conv = await asyncio.to_thread(conv_store.get_conversation, conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    msg = await asyncio.to_thread(
        conv_store.add_message,
        conversation_id,
        body.direction,
        sender_email=body.sender_email or conv["client"]["email"],
        recipient_email=get_sender().get("sender_email", ""),
        subject=body.subject or conv.get("subject", ""),
        body_text=body.body_text,
        body_html=body.body_html or "",
        status="received" if body.direction == "inbound" else "sent",
    )

    event_type = "inbound_received" if body.direction == "inbound" else "outbound_sent"
    await asyncio.to_thread(
        conv_store.add_timeline_event,
        conversation_id,
        event_type,
        {"message_id": msg["id"], "manual": True},
    )

    if body.direction == "inbound":
        try:
            draft = await asyncio.to_thread(generate_draft_for_conversation, conversation_id)
            return {"message": msg, "draft": draft}
        except Exception as e:
            return {"message": msg, "draft_error": str(e)}

    return {"message": msg}


@app.post("/api/conversations/{conversation_id}/generate-draft")
async def generate_draft_api(
    conversation_id: str,
    body: GenerateDraftRequest = GenerateDraftRequest(),
):
    _require_conversations_db()
    try:
        draft = await asyncio.to_thread(
            lambda: generate_draft_for_conversation(
                conversation_id, body.instructions, skip_gate=True
            )
        )
        return {"draft": draft}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.put("/api/conversations/{conversation_id}/drafts/{message_id}")
async def update_draft_api(conversation_id: str, message_id: str, body: DraftUpdateRequest):
    _require_conversations_db()
    msg = await asyncio.to_thread(conv_store.get_message, message_id)
    if not msg or msg["conversation_id"] != conversation_id or msg["status"] != "draft":
        raise HTTPException(status_code=404, detail="Draft not found")

    updates = {k: v for k, v in body.model_dump().items() if v is not None}
    updated = await asyncio.to_thread(conv_store.update_message, message_id, **updates)
    return {"message": updated}


@app.post("/api/conversations/{conversation_id}/drafts/{message_id}/approve")
async def approve_draft_api(conversation_id: str, message_id: str):
    _require_conversations_db()
    conv = await asyncio.to_thread(conv_store.get_conversation, conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    draft = await asyncio.to_thread(conv_store.get_message, message_id)
    if not draft or draft["conversation_id"] != conversation_id or draft["status"] != "draft":
        raise HTTPException(status_code=404, detail="Draft not found")

    inbound_msgs = [m for m in conv["messages"] if m["direction"] == "inbound" and m["status"] != "draft"]
    in_reply_to = draft.get("in_reply_to", "")
    references = draft.get("references_header", "")
    if inbound_msgs:
        latest_inbound = inbound_msgs[-1]
        if not in_reply_to:
            in_reply_to = latest_inbound.get("message_id_header", "")
        if not references:
            refs = latest_inbound.get("references_header", "")
            mid = latest_inbound.get("message_id_header", "")
            references = f"{refs} {mid}".strip() if mid else refs

    sender = get_sender()

    # Collect outbound / generated attachments for this send
    outbound_files = []
    try:
        import attachment_store as att_store

        pending = await asyncio.to_thread(
            att_store.list_outbound_pending, conversation_id, message_id
        )
        for a in pending:
            try:
                data = att_store.read_bytes(a["storage_path"])
                outbound_files.append(
                    {
                        "filename": a["filename"],
                        "mime_type": a["mime_type"],
                        "data": data,
                    }
                )
            except Exception:
                continue
    except Exception:
        pending = []

    send_result = await asyncio.to_thread(
        send_threaded_reply,
        conv["client"]["email"],
        draft["subject"],
        draft["body_html"] or draft["body_text"],
        draft["body_text"],
        in_reply_to,
        references,
        conv.get("gmail_thread_id"),
        sender.get("sender_name", ""),
        sender.get("sender_email", ""),
        get_organization_id(),
        outbound_files,
    )

    if not send_result.get("success"):
        raise HTTPException(status_code=500, detail=send_result.get("error", "Send failed"))

    await asyncio.to_thread(
        conv_store.update_message,
        message_id,
        status="sent",
        gmail_message_id=send_result.get("message_id"),
    )
    if pending:
        try:
            import attachment_store as att_store
            await asyncio.to_thread(
                att_store.link_to_message,
                [a["id"] for a in pending],
                message_id,
            )
        except Exception:
            pass
    if send_result.get("thread_id") and not conv.get("gmail_thread_id"):
        await asyncio.to_thread(
            conv_store.update_conversation,
            conversation_id,
            gmail_thread_id=send_result["thread_id"],
        )

    await asyncio.to_thread(
        conv_store.add_timeline_event,
        conversation_id,
        "draft_approved",
        {
            "message_id": message_id,
            "gmail_message_id": send_result.get("message_id"),
            "attachments": len(outbound_files),
        },
    )
    await asyncio.to_thread(
        conv_store.add_timeline_event,
        conversation_id,
        "outbound_sent",
        {"message_id": message_id, "gmail_message_id": send_result.get("message_id")},
    )

    updated = await asyncio.to_thread(conv_store.get_conversation, conversation_id)
    return {"status": "sent", "send_result": send_result, "conversation": updated}


@app.post("/api/conversations/{conversation_id}/drafts/{message_id}/reject")
async def reject_draft_api(conversation_id: str, message_id: str):
    _require_conversations_db()
    msg = await asyncio.to_thread(conv_store.get_message, message_id)
    if not msg or msg["conversation_id"] != conversation_id or msg["status"] != "draft":
        raise HTTPException(status_code=404, detail="Draft not found")

    await asyncio.to_thread(conv_store.delete_draft, message_id)
    await asyncio.to_thread(
        conv_store.add_timeline_event,
        conversation_id,
        "draft_rejected",
        {"message_id": message_id},
    )
    return {"status": "rejected"}


@app.post("/api/conversations/{conversation_id}/generate-banner")
async def generate_banner_api(conversation_id: str, body: BannerGenerateRequest):
    _require_conversations_db()
    try:
        result = await asyncio.to_thread(generate_banner, conversation_id, body.instructions)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


class GenerateAttachmentRequest(BaseModel):
    instructions: str
    title: str = ""
    format: str = "pdf"


@app.get("/api/conversations/{conversation_id}/attachments")
async def list_attachments_api(conversation_id: str):
    _require_conversations_db()
    import attachment_store as att_store

    conv = await asyncio.to_thread(conv_store.get_conversation, conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    atts = await asyncio.to_thread(att_store.list_attachments, conversation_id)
    # Don't leak storage paths
    safe = [
        {k: v for k, v in a.items() if k not in ("storage_path", "organization_id")}
        for a in atts
    ]
    return {"attachments": safe}


@app.post("/api/conversations/{conversation_id}/attachments")
async def upload_attachment_api(
    conversation_id: str,
    file: UploadFile = File(...),
    include_on_send: bool = Query(True),
):
    _require_conversations_db()
    from attachment_service import save_and_ingest, MAX_ATTACHMENT_BYTES

    conv = await asyncio.to_thread(conv_store.get_conversation, conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    data = await file.read()
    if len(data) > MAX_ATTACHMENT_BYTES:
        raise HTTPException(status_code=400, detail="File too large (max 15MB)")
    try:
        att = await asyncio.to_thread(
            save_and_ingest,
            conversation_id=conversation_id,
            filename=file.filename or "upload.bin",
            data=data,
            mime_type=file.content_type or "",
            source="outbound",
            include_on_send=include_on_send,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    await asyncio.to_thread(
        conv_store.add_timeline_event,
        conversation_id,
        "attachment_uploaded",
        {"attachment_id": att["id"], "filename": att["filename"]},
    )
    return {
        "attachment": {
            k: v for k, v in att.items() if k not in ("storage_path", "organization_id")
        }
    }


@app.post("/api/conversations/{conversation_id}/attachments/generate")
async def generate_attachment_api(conversation_id: str, body: GenerateAttachmentRequest):
    _require_conversations_db()
    from attachment_service import generate_document

    if not body.instructions.strip():
        raise HTTPException(status_code=400, detail="instructions required")
    try:
        att = await asyncio.to_thread(
            generate_document,
            conversation_id,
            instructions=body.instructions,
            title=body.title,
            fmt=body.format,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    await asyncio.to_thread(
        conv_store.add_timeline_event,
        conversation_id,
        "attachment_generated",
        {"attachment_id": att["id"], "filename": att["filename"]},
    )
    return {
        "attachment": {
            k: v for k, v in att.items() if k not in ("storage_path", "organization_id")
        }
    }


@app.delete("/api/attachments/{attachment_id}")
async def delete_attachment_api(attachment_id: str):
    _require_conversations_db()
    import attachment_store as att_store
    from vector_store import delete_by_source

    att = await asyncio.to_thread(att_store.get_attachment, attachment_id)
    if not att:
        raise HTTPException(status_code=404, detail="Attachment not found")
    await asyncio.to_thread(delete_by_source, f"email_attachment:{attachment_id}")
    ok = await asyncio.to_thread(att_store.delete_attachment, attachment_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Attachment not found")
    return {"status": "deleted"}


@app.get("/api/attachments/{attachment_id}/download")
async def download_attachment_api(attachment_id: str):
    _require_conversations_db()
    import attachment_store as att_store

    att = await asyncio.to_thread(att_store.get_attachment, attachment_id)
    if not att:
        raise HTTPException(status_code=404, detail="Attachment not found")
    path = Path(att["storage_path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="File missing on disk")
    return FileResponse(
        path,
        media_type=att.get("mime_type") or "application/octet-stream",
        filename=att["filename"],
    )


@app.post("/api/gmail/sync")
async def gmail_sync_api():
    _require_conversations_db()
    result = await asyncio.to_thread(run_gmail_sync)
    return result


@app.get("/api/gmail/status")
async def gmail_status_api():
    """Diagnose Gmail API credentials and domain-wide delegation."""
    return await asyncio.to_thread(check_gmail_health)
