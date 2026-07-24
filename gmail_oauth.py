"""Per-organization Gmail OAuth integration."""
from __future__ import annotations

import json
import os
import secrets
import urllib.parse
from typing import Optional

from google_auth_oauthlib.flow import Flow

from org_store import delete_gmail_integration, get_gmail_integration, save_gmail_integration

SCOPES = [
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.readonly",
    "openid",
    "email",
]

_pending_states: dict[str, str] = {}


def _oauth_configured() -> bool:
    return bool(
        os.getenv("GOOGLE_OAUTH_CLIENT_ID")
        and os.getenv("GOOGLE_OAUTH_CLIENT_SECRET")
        and os.getenv("GOOGLE_OAUTH_REDIRECT_URI")
    )


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


def get_authorize_url(organization_id: str) -> dict:
    if not _oauth_configured():
        return {
            "configured": False,
            "url": None,
            "message": "Google OAuth is not configured on this server.",
        }
    state = secrets.token_urlsafe(24)
    _pending_states[state] = organization_id
    flow = _flow()
    url, _ = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
        state=state,
    )
    return {"configured": True, "url": url, "state": state}


def handle_oauth_callback(code: str, state: str) -> dict:
    org_id = _pending_states.pop(state, None)
    if not org_id:
        raise ValueError("Invalid or expired OAuth state")
    flow = _flow()
    flow.fetch_token(code=code)
    creds = flow.credentials
    email = ""
    try:
        from googleapiclient.discovery import build
        svc = build("gmail", "v1", credentials=creds, cache_discovery=False)
        profile = svc.users().getProfile(userId="me").execute()
        email = profile.get("emailAddress", "")
    except Exception:
        pass
    token_data = {
        "token": creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri": creds.token_uri,
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        "scopes": list(creds.scopes or SCOPES),
    }
    save_gmail_integration(org_id, email, token_data)
    return {"organization_id": org_id, "connected_email": email}


def get_integration_status(organization_id: str) -> dict:
    row = get_gmail_integration(organization_id)
    if row:
        return {
            "connected": True,
            "email": row.get("connected_email", ""),
            "mode": "oauth",
        }
    return {
        "connected": False,
        "email": os.getenv("GSUITE_DELEGATED_USER", ""),
        "mode": "platform",
        "message": "Using platform sender (connect Gmail to send from your inbox)",
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
