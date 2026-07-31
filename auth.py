"""JWT authentication and password hashing."""
from __future__ import annotations

import os
import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import bcrypt
import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from tenant import TenantContext, set_tenant_context

_bearer = HTTPBearer(auto_error=False)

JWT_SECRET = os.getenv("JWT_SECRET", "change-me-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_HOURS = int(os.getenv("JWT_EXPIRY_HOURS", "168"))


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode(), password_hash.encode())
    except Exception:
        return False


def create_access_token(
    user_id: str,
    email: str,
    name: str,
    organization_id: str,
    role: str,
) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "email": email,
        "name": name,
        "org_id": organization_id,
        "role": role,
        "iat": now,
        "exp": now + timedelta(hours=JWT_EXPIRY_HOURS),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> dict[str, Any]:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.PyJWTError as e:
        raise HTTPException(status_code=401, detail="Invalid or expired token") from e


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug[:60] or str(uuid.uuid4())[:8]


async def get_current_tenant(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> TenantContext:
    if not credentials or not credentials.credentials:
        raise HTTPException(status_code=401, detail="Authentication required")
    payload = decode_token(credentials.credentials)
    ctx = TenantContext(
        user_id=payload["sub"],
        email=payload.get("email", ""),
        name=payload.get("name", ""),
        organization_id=payload.get("org_id", ""),
        role=payload.get("role", "member"),
    )
    if not ctx.organization_id:
        raise HTTPException(status_code=401, detail="Token missing organization")
    set_tenant_context(ctx)
    return ctx


def saas_mode_enabled() -> bool:
    return os.getenv("SAAS_MODE", "true").lower() in ("1", "true", "yes")


PUBLIC_PATHS = {
    "/",
    "/api/auth/signup",
    "/api/auth/login",
    "/api/integrations/gmail/callback",
    "/docs",
    "/openapi.json",
    "/redoc",
}


def _is_public_path(path: str) -> bool:
    if path in PUBLIC_PATHS:
        return True
    if path.startswith("/docs") or path.startswith("/redoc"):
        return True
    # Catalogue page images + attachments must be fetchable by email clients
    # (no JWT). Only expose the static media tree, never /api/*.
    if path.startswith("/media/"):
        return True
    return False


async def saas_auth_middleware(request, call_next):
    from starlette.responses import JSONResponse

    if not saas_mode_enabled():
        return await call_next(request)

    if request.method == "OPTIONS" or _is_public_path(request.url.path):
        return await call_next(request)

    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return JSONResponse(status_code=401, content={"detail": "Authentication required"})

    token = auth_header[7:].strip()
    try:
        payload = decode_token(token)
        ctx = TenantContext(
            user_id=payload["sub"],
            email=payload.get("email", ""),
            name=payload.get("name", ""),
            organization_id=payload.get("org_id", ""),
            role=payload.get("role", "member"),
        )
        if not ctx.organization_id:
            return JSONResponse(status_code=401, content={"detail": "Token missing organization"})
        set_tenant_context(ctx)
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"detail": e.detail})

    return await call_next(request)
