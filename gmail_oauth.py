"""Google OAuth — Sign in with Google + per-org Gmail send/sync.

Any Google account (personal Gmail or Workspace) can sign in. Tokens are stored
per organization so outbound mail is sent as that signed-in inbox.
"""
from __future__ import annotations

import json
import os
import secrets
import time
from typing import Any, Optional
from urllib.parse import urlencode

from google_auth_oauthlib.flow import Flow

from org_store import (
    delete_gmail_integration,
    get_gmail_integration,
    get_or_create_google_user,
    save_gmail_integration,
)

# openid/email/profile for identity; Gmail scopes so the same consent powers sending.
SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.readonly",
]

# state -> {purpose, org_id?, created_at}
_pending_states: dict[str, dict[str, Any]] = {}
_STATE_TTL_SEC = 600


def _oauth_configured() -> bool:
    return bool(
        os.getenv("GOOGLE_OAUTH_CLIENT_ID")
        and os.getenv("GOOGLE_OAUTH_CLIENT_SECRET")
        and os.getenv("GOOGLE_OAUTH_REDIRECT_URI")
    )


def _purge_expired_states() -> None:
    now = time.time()
    dead = [k for k, v in _pending_states.items() if now - v.get("created_at", 0) > _STATE_TTL_SEC]
    for k in dead:
        _pending_states.pop(k, None)


def _flow() -> Flow:
    client_config = {
        "web": {
            "client_id": os.environ["GOOGLE_OAUTH_CLIENT_ID"],
            "client_secret": os.environ["GOOGLE_OAUTH_CLIENT_SECRET"],
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": [os.environ["GOOGLE_OAUTH_REDIRECT_URI"]],
        }
    }
    return Flow.from_client_config(
        client_config,
        scopes=SCOPES,
        redirect_uri=os.environ["GOOGLE_OAUTH_REDIRECT_URI"],
    )


def _authorization_url(state: str) -> str:
    """Build consent URL. No hd= — any Google account is allowed."""
    flow = _flow()
    url, _ = flow.authorization_url(
        access_type="offline",
        prompt="consent",
        state=state,
        # Do not pass include_granted_scopes — it causes intermittent scope mismatches.
    )
    return url


def get_authorize_url(organization_id: str) -> dict:
    """Connect / reconnect Gmail for an already signed-in org."""
    if not _oauth_configured():
        return {
            "configured": False,
            "url": None,
            "message": "Google OAuth is not configured on this server.",
        }
    _purge_expired_states()
    state = secrets.token_urlsafe(24)
    _pending_states[state] = {
        "purpose": "connect",
        "org_id": organization_id,
        "created_at": time.time(),
    }
    return {"configured": True, "url": _authorization_url(state), "state": state}


def get_login_authorize_url() -> dict:
    """Sign in with Google (any account). Also grants Gmail send/read for that inbox."""
    if not _oauth_configured():
        return {
            "configured": False,
            "url": None,
            "message": "Google OAuth is not configured on this server. Set GOOGLE_OAUTH_CLIENT_ID, GOOGLE_OAUTH_CLIENT_SECRET, and GOOGLE_OAUTH_REDIRECT_URI.",
        }
    _purge_expired_states()
    state = secrets.token_urlsafe(24)
    _pending_states[state] = {
        "purpose": "login",
        "org_id": None,
        "created_at": time.time(),
    }
    return {"configured": True, "url": _authorization_url(state), "state": state}


def _token_payload(creds) -> dict:
    return {
        "token": creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri": creds.token_uri,
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        "scopes": list(creds.scopes or SCOPES),
    }


def _profile_from_creds(creds) -> tuple[str, str]:
    """Return (email, name) from Gmail profile / userinfo."""
    email = ""
    name = ""
    try:
        from googleapiclient.discovery import build

        svc = build("gmail", "v1", credentials=creds, cache_discovery=False)
        profile = svc.users().getProfile(userId="me").execute()
        email = (profile.get("emailAddress") or "").strip().lower()
    except Exception:
        pass
    try:
        from google.oauth2 import id_token
        from google.auth.transport import requests as google_requests

        if creds.id_token:
            info = id_token.verify_oauth2_token(
                creds.id_token,
                google_requests.Request(),
                os.environ["GOOGLE_OAUTH_CLIENT_ID"],
            )
            email = email or (info.get("email") or "").strip().lower()
            name = (info.get("name") or info.get("given_name") or "").strip()
    except Exception:
        pass
    if not name and email:
        name = email.split("@")[0]
    return email, name


def handle_oauth_callback(code: str, state: str) -> dict:
    """
    Finish OAuth.
    Returns either:
      {purpose: 'connect', organization_id, connected_email}
      {purpose: 'login', user, organization, token fields via caller}
    """
    _purge_expired_states()
    meta = _pending_states.get(state)
    if not meta:
        raise ValueError("Invalid or expired OAuth state — please try again.")

    flow = _flow()
    flow.fetch_token(code=code)
    creds = flow.credentials
    email, name = _profile_from_creds(creds)
    if not email:
        raise ValueError("Could not read Google account email from OAuth response.")

    token_data = _token_payload(creds)
    purpose = meta.get("purpose") or "connect"

    if purpose == "login":
        user = get_or_create_google_user(email, name)
        save_gmail_integration(user["organization_id"], email, token_data)
        _pending_states.pop(state, None)
        return {
            "purpose": "login",
            "connected_email": email,
            "user": user,
        }

    org_id = meta.get("org_id")
    if not org_id:
        raise ValueError("Missing organization for Gmail connect.")
    save_gmail_integration(org_id, email, token_data)
    _pending_states.pop(state, None)
    return {
        "purpose": "connect",
        "organization_id": org_id,
        "connected_email": email,
    }


def pending_purpose(state: str) -> Optional[str]:
    _purge_expired_states()
    meta = _pending_states.get(state) or {}
    return meta.get("purpose")



def get_integration_status(organization_id: str) -> dict:
    row = get_gmail_integration(organization_id)
    if row:
        return {
            "connected": True,
            "email": row.get("connected_email", ""),
            "connected_email": row.get("connected_email", ""),
            "mode": "oauth",
            "message": f"Sending as {row.get('connected_email', '')}",
        }
    return {
        "connected": False,
        "email": "",
        "connected_email": "",
        "mode": "none",
        "message": "Sign in with Google or connect Gmail to send from your inbox.",
    }


def disconnect(organization_id: str) -> None:
    delete_gmail_integration(organization_id)


def build_oauth_credentials(organization_id: str):
    from google.oauth2.credentials import Credentials

    row = get_gmail_integration(organization_id)
    if not row or not row.get("credentials"):
        return None
    data = row["credentials"]
    return Credentials(
        token=data.get("token"),
        refresh_token=data.get("refresh_token"),
        token_uri=data.get("token_uri", "https://oauth2.googleapis.com/token"),
        client_id=data.get("client_id", os.getenv("GOOGLE_OAUTH_CLIENT_ID")),
        client_secret=data.get("client_secret", os.getenv("GOOGLE_OAUTH_CLIENT_SECRET")),
        scopes=data.get("scopes", SCOPES),
    )


def frontend_login_redirect(token: str, email: str = "") -> str:
    frontend = os.getenv("FRONTEND_URL", "http://localhost:5173").rstrip("/")
    qs = urlencode({"google_token": token, "google_email": email})
    return f"{frontend}/login?{qs}"
