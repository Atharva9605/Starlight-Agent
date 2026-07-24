import os
import json
import base64
import re
import logging
import threading
from email import message_from_string
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.utils import parseaddr
from typing import Optional

from google.oauth2 import service_account
from googleapiclient.discovery import build

# --- Configuration ---
_here = os.path.dirname(os.path.abspath(__file__))
SERVICE_ACCOUNT_FILE = os.getenv(
    "GSUITE_SERVICE_ACCOUNT_JSON", os.path.join(_here, "service_account.json")
)
DELEGATED_USER = os.getenv("GSUITE_DELEGATED_USER", "vivek@starlightlinearled.com")
SCOPES = [
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.readonly",
]

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("gmail")

_service_lock = threading.Lock()
_service = None


def _fix_private_key(info: dict) -> dict:
    """Normalize private_key newlines — common issue when JSON is pasted into HF secrets."""
    pk = info.get("private_key")
    if not pk or not isinstance(pk, str):
        return info
    # Literal \n sequences instead of real newlines
    if "\\n" in pk and "-----BEGIN PRIVATE KEY-----" in pk:
        info = {**info, "private_key": pk.replace("\\n", "\n")}
    return info


def _load_service_account_info() -> dict:
    """
    Load service account JSON from env or file.
    Supports GSUITE_SERVICE_ACCOUNT_JSON_CONTENT as raw JSON or base64-encoded JSON.
    """
    raw = os.getenv("GSUITE_SERVICE_ACCOUNT_JSON_CONTENT", "").strip()
    if raw:
        try:
            if raw.startswith("{"):
                info = json.loads(raw)
            else:
                decoded = base64.b64decode(raw).decode("utf-8")
                info = json.loads(decoded)
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(
                "GSUITE_SERVICE_ACCOUNT_JSON_CONTENT is not valid JSON. "
                "Paste the full service account key file, or base64-encode it. "
                f"Parse error: {e}"
            ) from e
        return _fix_private_key(info)

    if not os.path.exists(SERVICE_ACCOUNT_FILE):
        raise FileNotFoundError(
            f"Gmail credentials not found. Set GSUITE_SERVICE_ACCOUNT_JSON_CONTENT "
            f"or place a key file at {SERVICE_ACCOUNT_FILE}"
        )
    with open(SERVICE_ACCOUNT_FILE, encoding="utf-8") as f:
        return _fix_private_key(json.load(f))


def _build_credentials():
    info = _load_service_account_info()
    delegated = os.getenv("GSUITE_DELEGATED_USER", DELEGATED_USER)
    return service_account.Credentials.from_service_account_info(
        info, scopes=SCOPES, subject=delegated
    )


def _invalidate_service():
    global _service
    with _service_lock:
        _service = None


def get_gmail_service(force_refresh: bool = False, organization_id: str | None = None):
    """Authenticate and return Gmail API service (lazy, thread-safe)."""
    if organization_id:
        from gmail_oauth import build_oauth_credentials
        oauth_creds = build_oauth_credentials(organization_id)
        if oauth_creds:
            try:
                if oauth_creds.expired and oauth_creds.refresh_token:
                    from google.auth.transport.requests import Request
                    oauth_creds.refresh(Request())
                return build("gmail", "v1", credentials=oauth_creds, cache_discovery=False)
            except Exception as e:
                log.error("OAuth Gmail service failed for org %s: %s", organization_id, e)

    global _service
    if force_refresh:
        _invalidate_service()

    with _service_lock:
        if _service is not None:
            return _service
        try:
            credentials = _build_credentials()
            _service = build("gmail", "v1", credentials=credentials, cache_discovery=False)
            return _service
        except Exception as e:
            log.error("Failed to create Gmail service: %s", e)
            return None


def _friendly_auth_error(exc: Exception) -> str:
    msg = str(exc)
    if "SignatureException" in msg or "invalid_grant" in msg:
        sa_email = ""
        try:
            sa_email = _load_service_account_info().get("client_email", "")
        except Exception:
            pass
        return (
            "Gmail authentication failed: invalid service account signature. "
            "This usually means GSUITE_SERVICE_ACCOUNT_JSON_CONTENT has a corrupted "
            "private_key (broken newlines when pasted into HuggingFace Secrets). "
            "Fix: re-download the JSON key from Google Cloud Console → IAM → Service "
            "Accounts → Keys, paste the entire file as the secret value, or base64-encode "
            "the file and paste that instead. "
            f"Service account: {sa_email or 'unknown'}. "
            "Also verify domain-wide delegation Client ID matches this service account."
        )
    if "unauthorized_client" in msg.lower():
        return (
            "Gmail unauthorized_client: add domain-wide delegation in Google Admin for "
            "scopes gmail.send and gmail.readonly, using the service account's numeric Client ID."
        )
    return msg


def check_gmail_health() -> dict:
    """Test Gmail API auth — useful for /api/gmail/status diagnostics."""
    try:
        info = _load_service_account_info()
        svc = get_gmail_service(force_refresh=True)
        if not svc:
            return {
                "ok": False,
                "error": "Could not build Gmail service — check credentials env var",
            }
        profile = svc.users().getProfile(userId="me").execute()
        return {
            "ok": True,
            "delegated_user": os.getenv("GSUITE_DELEGATED_USER", DELEGATED_USER),
            "service_account": info.get("client_email"),
            "client_id": info.get("client_id"),
            "email_address": profile.get("emailAddress"),
            "messages_total": profile.get("messagesTotal"),
            "scopes": SCOPES,
        }
    except Exception as e:
        return {"ok": False, "error": _friendly_auth_error(e)}


def _with_service(fn, organization_id: str | None = None):
    """Run Gmail API call; refresh credentials once on auth failure."""
    svc = get_gmail_service(organization_id=organization_id)
    if not svc:
        return None, "Gmail service unavailable — credentials not configured"
    try:
        return fn(svc), None
    except Exception as e:
        err = str(e)
        if organization_id:
            svc = get_gmail_service(force_refresh=True, organization_id=organization_id)
            if svc:
                try:
                    return fn(svc), None
                except Exception as e2:
                    return None, _friendly_auth_error(e2)
        if "invalid_grant" in err or "SignatureException" in err:
            log.warning("Gmail auth failed, refreshing credentials: %s", err)
            svc = get_gmail_service(force_refresh=True, organization_id=organization_id)
            if svc:
                try:
                    return fn(svc), None
                except Exception as e2:
                    return None, _friendly_auth_error(e2)
        return None, _friendly_auth_error(e)


def _decode_body_data(data: str) -> str:
    if not data:
        return ""
    try:
        return base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")
    except Exception:
        return ""


def _strip_html(html: str) -> str:
    if not html:
        return ""
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.I | re.S)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"</p>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _extract_parts(
    payload: dict,
    text_parts: list,
    html_parts: list,
    attachment_parts: list | None = None,
) -> None:
    mime = payload.get("mimeType", "")
    body = payload.get("body", {})
    data = body.get("data")
    filename = payload.get("filename") or ""
    headers = {h["name"].lower(): h["value"] for h in payload.get("headers", [])}
    disposition = headers.get("content-disposition", "")

    is_attachment = bool(filename) or "attachment" in disposition.lower()
    if is_attachment and attachment_parts is not None:
        att_id = body.get("attachmentId")
        if att_id or data:
            attachment_parts.append(
                {
                    "filename": filename or "attachment",
                    "mime_type": mime or "application/octet-stream",
                    "size": int(body.get("size") or 0),
                    "attachment_id": att_id,
                    "inline_data": data,  # present for small inline parts
                }
            )
        # Still continue into multipart children if any
    elif mime == "text/plain" and data and not is_attachment:
        text_parts.append(_decode_body_data(data))
    elif mime == "text/html" and data and not is_attachment:
        html_parts.append(_decode_body_data(data))

    if mime.startswith("multipart/") or payload.get("parts"):
        for part in payload.get("parts", []):
            _extract_parts(part, text_parts, html_parts, attachment_parts)


def parse_message_payload(payload: dict, headers: list) -> dict:
    """Extract structured fields from a Gmail API message payload."""
    header_map = {h["name"].lower(): h["value"] for h in headers}
    text_parts: list[str] = []
    html_parts: list[str] = []
    attachment_parts: list[dict] = []
    _extract_parts(payload, text_parts, html_parts, attachment_parts)

    body_text = "\n".join(text_parts).strip()
    body_html = "\n".join(html_parts).strip()
    if not body_text and body_html:
        body_text = _strip_html(body_html)

    from_raw = header_map.get("from", "")
    to_raw = header_map.get("to", "")
    from_email = parseaddr(from_raw)[1]
    to_email = parseaddr(to_raw)[1]

    return {
        "from_email": from_email,
        "from_raw": from_raw,
        "to_email": to_email,
        "to_raw": to_raw,
        "subject": header_map.get("subject", ""),
        "body_text": body_text,
        "body_html": body_html,
        "in_reply_to": header_map.get("in-reply-to", ""),
        "references": header_map.get("references", ""),
        "message_id_header": header_map.get("message-id", ""),
        "date": header_map.get("date", ""),
        "attachments": attachment_parts,
    }


def download_gmail_attachment(message_id: str, attachment_id: str, organization_id: str | None = None) -> bytes | None:
    """Download raw attachment bytes from Gmail."""
    if not message_id or not attachment_id:
        return None

    def _fetch(svc):
        att = (
            svc.users()
            .messages()
            .attachments()
            .get(userId="me", messageId=message_id, id=attachment_id)
            .execute()
        )
        data = att.get("data") or ""
        return base64.urlsafe_b64decode(data)

    result, err = _with_service(_fetch, organization_id=organization_id)
    if err:
        log.error("Failed to download attachment %s: %s", attachment_id, err)
        return None
    return result


def get_message_detail(message_id: str) -> Optional[dict]:
    def _fetch(svc):
        msg = svc.users().messages().get(userId="me", id=message_id, format="full").execute()
        parsed = parse_message_payload(
            msg.get("payload", {}), msg.get("payload", {}).get("headers", [])
        )
        parsed["gmail_message_id"] = message_id
        parsed["thread_id"] = msg.get("threadId", "")
        parsed["label_ids"] = msg.get("labelIds", [])
        return parsed

    result, err = _with_service(_fetch)
    if err:
        log.error("Failed to get message %s: %s", message_id, err)
    return result


def fetch_thread_messages(thread_id: str) -> list[dict]:
    if not thread_id:
        return []

    def _fetch(svc):
        thread = svc.users().threads().get(userId="me", id=thread_id, format="full").execute()
        results = []
        for msg in thread.get("messages", []):
            parsed = parse_message_payload(
                msg.get("payload", {}), msg.get("payload", {}).get("headers", [])
            )
            parsed["gmail_message_id"] = msg["id"]
            parsed["thread_id"] = thread_id
            results.append(parsed)
        return results

    result, err = _with_service(_fetch)
    if err:
        log.error("Failed to fetch thread %s: %s", thread_id, err)
    return result or []


def list_recent_inbox_messages(max_results: int = 50) -> list[dict]:
    def _fetch(svc):
        resp = (
            svc.users()
            .messages()
            .list(userId="me", q="in:inbox newer_than:7d", maxResults=max_results)
            .execute()
        )
        messages = []
        for item in resp.get("messages", []):
            detail = get_message_detail(item["id"])
            if detail:
                messages.append(detail)
        return messages

    result, err = _with_service(_fetch)
    if err:
        log.error("Failed to list inbox: %s", err)
    return result or []


def poll_history_changes(start_history_id: Optional[str]) -> tuple[list[dict], Optional[str]]:
    """Incremental sync via users.history.list."""

    def _fetch(svc):
        profile = svc.users().getProfile(userId="me").execute()
        current_history_id = profile.get("historyId")

        if not start_history_id:
            return [], str(current_history_id)

        new_messages: list[dict] = []
        page_token = None
        while True:
            kwargs = {
                "userId": "me",
                "startHistoryId": start_history_id,
                "historyTypes": ["messageAdded"],
            }
            if page_token:
                kwargs["pageToken"] = page_token

            history = svc.users().history().list(**kwargs).execute()
            for record in history.get("history", []):
                for added in record.get("messagesAdded", []):
                    msg_meta = added.get("message", {})
                    msg_id = msg_meta.get("id")
                    labels = msg_meta.get("labelIds", [])
                    if msg_id and "INBOX" in labels:
                        detail = get_message_detail(msg_id)
                        if detail:
                            new_messages.append(detail)

            page_token = history.get("nextPageToken")
            if not page_token:
                break

        return new_messages, str(current_history_id)

    result, err = _with_service(_fetch)
    if err:
        log.warning("History poll failed (%s), falling back to inbox list", err)
        return list_recent_inbox_messages(), start_history_id
    return result if result else ([], start_history_id)


def send_email_gsuite(
    eml_path: str,
    sender_email: str = None,
    organization_id: str | None = None,
) -> dict:
    """
    Sends an email using Gmail API from a .eml file.
    Returns {success, message_id, thread_id, error}.
    """
    if not os.path.exists(eml_path):
        return {"success": False, "error": f".eml file not found: {eml_path}"}

    with open(eml_path, "r", encoding="utf-8") as f:
        raw_email = f.read()

    msg = message_from_string(raw_email)
    if sender_email:
        if msg.get("From"):
            msg.replace_header("From", sender_email)
        else:
            msg["From"] = sender_email

    encoded_message = base64.urlsafe_b64encode(msg.as_bytes()).decode("utf-8")

    def _send(svc):
        return (
            svc.users()
            .messages()
            .send(userId="me", body={"raw": encoded_message})
            .execute()
        )

    send_result, err = _with_service(_send, organization_id=organization_id)
    if err:
        log.error("Failed to send email: %s", err)
        return {"success": False, "error": err}

    log.info("Email sent: %s to %s", send_result.get("id"), msg.get("To"))
    return {
        "success": True,
        "message_id": send_result.get("id"),
        "thread_id": send_result.get("threadId"),
    }


def send_threaded_reply(
    to_email: str,
    subject: str,
    body_html: str,
    body_text: str = "",
    in_reply_to: str = "",
    references: str = "",
    thread_id: Optional[str] = None,
    sender_name: str = "",
    sender_email: str = "",
    organization_id: str | None = None,
    attachments: list | None = None,
) -> dict:
    """Send a threaded HTML reply via Gmail API, optionally with file attachments."""
    from email.mime.base import MIMEBase
    from email import encoders

    delegated = os.getenv("GSUITE_DELEGATED_USER", DELEGATED_USER)
    from_addr = (
        f'"{sender_name}" <{sender_email}>'
        if sender_name
        else sender_email or delegated
    )

    file_atts = [a for a in (attachments or []) if a.get("data") or a.get("path")]
    if file_atts:
        msg = MIMEMultipart("mixed")
        if body_text:
            alt = MIMEMultipart("alternative")
            alt.attach(MIMEText(body_text, "plain", "utf-8"))
            alt.attach(MIMEText(body_html, "html", "utf-8"))
            msg.attach(alt)
        else:
            msg.attach(MIMEText(body_html, "html", "utf-8"))
        for att in file_atts:
            raw = att.get("data")
            if raw is None and att.get("path"):
                with open(att["path"], "rb") as f:
                    raw = f.read()
            if not raw:
                continue
            maintype, _, subtype = (att.get("mime_type") or "application/octet-stream").partition("/")
            if not subtype:
                maintype, subtype = "application", "octet-stream"
            part = MIMEBase(maintype, subtype)
            part.set_payload(raw)
            encoders.encode_base64(part)
            filename = att.get("filename") or "attachment"
            part.add_header("Content-Disposition", "attachment", filename=filename)
            msg.attach(part)
    elif body_text:
        msg = MIMEMultipart("alternative")
        msg.attach(MIMEText(body_text, "plain", "utf-8"))
        msg.attach(MIMEText(body_html, "html", "utf-8"))
    else:
        msg = MIMEText(body_html, "html", "utf-8")

    msg["To"] = to_email
    msg["From"] = from_addr
    msg["Subject"] = subject
    if in_reply_to:
        msg["In-Reply-To"] = in_reply_to
    if references:
        msg["References"] = references

    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode("utf-8")
    body: dict = {"raw": raw}
    if thread_id:
        body["threadId"] = thread_id

    def _send(svc):
        return svc.users().messages().send(userId="me", body=body).execute()

    send_result, err = _with_service(_send, organization_id=organization_id)
    if err:
        log.error("Failed to send threaded reply: %s", err)
        return {"success": False, "error": err}

    return {
        "success": True,
        "message_id": send_result.get("id"),
        "thread_id": send_result.get("threadId"),
    }
