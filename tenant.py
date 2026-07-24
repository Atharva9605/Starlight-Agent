"""Request-scoped tenant context for multi-tenant data access."""
from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from typing import Any, Callable, Optional

_tenant_local = threading.local()

DEFAULT_ORG_ID = "org_default_starlight"


@dataclass
class TenantContext:
    user_id: str
    organization_id: str
    role: str
    email: str = ""
    name: str = ""


def set_tenant_context(ctx: TenantContext | None) -> None:
    _tenant_local.ctx = ctx


def get_tenant_context() -> TenantContext:
    ctx = getattr(_tenant_local, "ctx", None)
    if ctx is None:
        return TenantContext(
            user_id="system",
            organization_id=DEFAULT_ORG_ID,
            role="owner",
        )
    return ctx


def get_organization_id() -> str:
    return get_tenant_context().organization_id


def require_organization_id() -> str:
    org_id = get_organization_id()
    if not org_id:
        raise RuntimeError("No organization context")
    return org_id


async def run_in_thread(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Run sync function in thread pool preserving tenant context."""
    ctx = get_tenant_context()

    def wrapper() -> Any:
        set_tenant_context(ctx)
        return fn(*args, **kwargs)

    return await asyncio.to_thread(wrapper)
