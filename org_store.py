"""Users, organizations, memberships, integrations — PostgreSQL."""
from __future__ import annotations

import json
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Any, Optional

from auth import hash_password, slugify, verify_password
from crypto_utils import decrypt_text, encrypt_text
from tenant import DEFAULT_ORG_ID

log = logging.getLogger("org_store")

_BASE_DIR = Path(__file__).parent
_DEFAULTS_PATH = _BASE_DIR / "defaults" / "business_config.json"


def _pg_conn():
    from vector_store import _pg_conn as conn_fn
    return conn_fn()


def _new_id(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4()}" if prefix else str(uuid.uuid4())


def _run_sql_file(conn, filename: str) -> None:
    path = _BASE_DIR / "migrations" / filename
    sql = path.read_text(encoding="utf-8")
    with conn.cursor() as cur:
        cur.execute(sql)


def _column_exists(cur, table: str, column: str) -> bool:
    cur.execute(
        """
        SELECT 1 FROM information_schema.columns
        WHERE table_name = %s AND column_name = %s
        """,
        (table, column),
    )
    return cur.fetchone() is not None


def _backfill_organization_id(conn, default_org_id: str) -> None:
    tables = [
        "catalogue_chunks",
        "clients",
        "conversations",
        "messages",
        "timeline_events",
    ]
    with conn.cursor() as cur:
        for table in tables:
            if _column_exists(cur, table, "organization_id"):
                cur.execute(
                    f"UPDATE {table} SET organization_id = %s WHERE organization_id IS NULL",
                    (default_org_id,),
                )
        if _column_exists(cur, "gmail_sync_state", "organization_id"):
            cur.execute(
                "UPDATE gmail_sync_state SET organization_id = %s WHERE organization_id IS NULL",
                (default_org_id,),
            )
            cur.execute(
                """
                INSERT INTO gmail_sync_state (id, organization_id, history_id, last_sync_at, updated_at)
                VALUES (%s, %s, NULL, NULL, NOW())
                ON CONFLICT (id) DO UPDATE SET
                    organization_id = EXCLUDED.organization_id,
                    updated_at = NOW()
                """,
                (default_org_id, default_org_id),
            )


def _fix_clients_unique_constraint(conn) -> None:
    with conn.cursor() as cur:
        # Drop legacy UNIQUE(email) — constraint name from original schema
        cur.execute("ALTER TABLE clients DROP CONSTRAINT IF EXISTS clients_email_key")

        cur.execute(
            """
            SELECT c.conname
            FROM pg_constraint c
            JOIN pg_class t ON c.conrelid = t.oid
            WHERE t.relname = 'clients' AND c.contype = 'u'
              AND c.conname != 'clients_pkey'
            """
        )
        for (conname,) in cur.fetchall():
            if conname == "idx_clients_org_email":
                continue
            cur.execute(f'ALTER TABLE clients DROP CONSTRAINT IF EXISTS "{conname}"')

        cur.execute("DROP INDEX IF EXISTS idx_clients_org_email")

        # Remove duplicate (org, email) rows before creating composite unique index
        cur.execute(
            """
            DELETE FROM clients c1
            USING clients c2
            WHERE c1.organization_id IS NOT DISTINCT FROM c2.organization_id
              AND lower(c1.email) = lower(c2.email)
              AND c1.ctid > c2.ctid
            """
        )

        cur.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_clients_org_email
            ON clients (organization_id, lower(email))
            """
        )


def _load_file_config() -> dict:
    paths = [
        Path("/data/business_config.json"),
        _BASE_DIR / "data" / "business_config.json",
        _DEFAULTS_PATH,
    ]
    for p in paths:
        if p.exists():
            with open(p, encoding="utf-8") as f:
                return json.load(f)
    with open(_DEFAULTS_PATH, encoding="utf-8") as f:
        return json.load(f)


def run_saas_migrations() -> None:
    from vector_store import use_postgres
    if not use_postgres():
        log.warning("SaaS migrations skipped — DATABASE_URL not set")
        return

    conn = _pg_conn()
    try:
        _run_sql_file(conn, "001_saas.sql")
        conn.commit()

        default_org_id = ensure_default_organization(conn)
        conn.commit()

        _backfill_organization_id(conn, default_org_id)
        conn.commit()

        _fix_clients_unique_constraint(conn)
        conn.commit()

        bootstrap_admin_user(conn, default_org_id)
        conn.commit()

        seed_default_org_config(conn, default_org_id)
        conn.commit()

        log.info("SaaS migrations complete (default org: %s)", default_org_id)
    except Exception as exc:
        conn.rollback()
        log.exception("SaaS migration failed: %s", exc)
        raise
    finally:
        conn.close()


def ensure_default_organization(conn) -> str:
    org_id = DEFAULT_ORG_ID
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM organizations WHERE id = %s", (org_id,))
        if not cur.fetchone():
            cur.execute(
                """
                INSERT INTO organizations (id, name, slug)
                VALUES (%s, %s, %s)
                ON CONFLICT (id) DO NOTHING
                """,
                (org_id, "Default Organization", "default"),
            )
    return org_id


def seed_default_org_config(conn, org_id: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM organization_config WHERE organization_id = %s",
            (org_id,),
        )
        if cur.fetchone():
            return
        config = _load_file_config()
        cur.execute(
            """
            INSERT INTO organization_config (organization_id, config_json)
            VALUES (%s, %s::jsonb)
            ON CONFLICT (organization_id) DO NOTHING
            """,
            (org_id, json.dumps(config)),
        )


def bootstrap_admin_user(conn, default_org_id: str) -> None:
    email = os.getenv("SAAS_BOOTSTRAP_EMAIL", "").strip().lower()
    password = os.getenv("SAAS_BOOTSTRAP_PASSWORD", "").strip()
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM users")
        if int(cur.fetchone()[0]) > 0:
            return
        if not email or not password:
            log.info("No SAAS_BOOTSTRAP_* — skipping admin user creation")
            return
        user_id = _new_id()
        cur.execute(
            """
            INSERT INTO users (id, email, password_hash, name)
            VALUES (%s, %s, %s, %s)
            """,
            (user_id, email, hash_password(password), "Admin"),
        )
        cur.execute(
            """
            INSERT INTO organization_members (organization_id, user_id, role)
            VALUES (%s, %s, 'owner')
            ON CONFLICT DO NOTHING
            """,
            (default_org_id, user_id),
        )
        log.info("Bootstrap admin created: %s", email)


def create_user_with_org(
    email: str,
    password: str,
    name: str,
    org_name: str,
) -> dict:
    email = email.strip().lower()
    if not email or not password:
        raise ValueError("Email and password are required")
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters")

    conn = _pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM users WHERE email = %s", (email,))
            if cur.fetchone():
                raise ValueError("Email already registered")

            user_id = _new_id()
            org_id = _new_id("org_")
            slug = slugify(org_name)
            cur.execute("SELECT id FROM organizations WHERE slug = %s", (slug,))
            if cur.fetchone():
                slug = f"{slug}-{uuid.uuid4().hex[:6]}"

            cur.execute(
                "INSERT INTO users (id, email, password_hash, name) VALUES (%s, %s, %s, %s)",
                (user_id, email, hash_password(password), name.strip() or email.split("@")[0]),
            )
            cur.execute(
                "INSERT INTO organizations (id, name, slug) VALUES (%s, %s, %s)",
                (org_id, org_name.strip() or "My Organization", slug),
            )
            cur.execute(
                "INSERT INTO organization_members (organization_id, user_id, role) VALUES (%s, %s, 'owner')",
                (org_id, user_id),
            )
            defaults = _load_file_config()
            cur.execute(
                "INSERT INTO organization_config (organization_id, config_json) VALUES (%s, %s::jsonb)",
                (org_id, json.dumps(defaults)),
            )
        conn.commit()
        return {
            "user_id": user_id,
            "organization_id": org_id,
            "email": email,
            "name": name.strip() or email.split("@")[0],
            "role": "owner",
            "organization_name": org_name.strip() or "My Organization",
        }
    finally:
        conn.close()


def authenticate_user(email: str, password: str) -> Optional[dict]:
    email = email.strip().lower()
    conn = _pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, email, password_hash, name FROM users WHERE email = %s",
                (email,),
            )
            row = cur.fetchone()
            if not row or not verify_password(password, row[2]):
                return None
            user_id, user_email, _, user_name = row
            cur.execute(
                """
                SELECT om.organization_id, om.role, o.name, o.slug
                FROM organization_members om
                JOIN organizations o ON o.id = om.organization_id
                WHERE om.user_id = %s
                ORDER BY om.created_at ASC
                """,
                (user_id,),
            )
            memberships = [
                {
                    "organization_id": r[0],
                    "role": r[1],
                    "name": r[2],
                    "slug": r[3],
                }
                for r in cur.fetchall()
            ]
            if not memberships:
                return None
            primary = memberships[0]
            return {
                "user_id": user_id,
                "email": user_email,
                "name": user_name,
                "organization_id": primary["organization_id"],
                "organization_name": primary["name"],
                "role": primary["role"],
                "organizations": memberships,
            }
    finally:
        conn.close()


def get_user_by_id(user_id: str) -> Optional[dict]:
    conn = _pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, email, name FROM users WHERE id = %s",
                (user_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            cur.execute(
                """
                SELECT om.organization_id, om.role, o.name, o.slug
                FROM organization_members om
                JOIN organizations o ON o.id = om.organization_id
                WHERE om.user_id = %s
                """,
                (user_id,),
            )
            memberships = [
                {"organization_id": r[0], "role": r[1], "name": r[2], "slug": r[3]}
                for r in cur.fetchall()
            ]
            return {
                "user_id": row[0],
                "email": row[1],
                "name": row[2],
                "organizations": memberships,
            }
    finally:
        conn.close()


def get_org_config(organization_id: str) -> dict:
    conn = _pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT config_json FROM organization_config WHERE organization_id = %s",
                (organization_id,),
            )
            row = cur.fetchone()
            if row:
                cfg = row[0]
                return cfg if isinstance(cfg, dict) else json.loads(cfg)
            defaults = _load_file_config()
            cur.execute(
                """
                INSERT INTO organization_config (organization_id, config_json)
                VALUES (%s, %s::jsonb)
                ON CONFLICT (organization_id) DO NOTHING
                """,
                (organization_id, json.dumps(defaults)),
            )
        conn.commit()
        return defaults
    finally:
        conn.close()


def save_org_config(organization_id: str, config: dict) -> dict:
    conn = _pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO organization_config (organization_id, config_json, updated_at)
                VALUES (%s, %s::jsonb, NOW())
                ON CONFLICT (organization_id) DO UPDATE SET
                    config_json = EXCLUDED.config_json,
                    updated_at = NOW()
                """,
                (organization_id, json.dumps(config)),
            )
        conn.commit()
        return config
    finally:
        conn.close()


def get_gmail_integration(organization_id: str) -> Optional[dict]:
    conn = _pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, credentials_json, connected_email, status
                FROM organization_integrations
                WHERE organization_id = %s AND provider = 'gmail' AND status = 'active'
                """,
                (organization_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            creds = {}
            if row[1]:
                try:
                    creds = json.loads(decrypt_text(row[1]))
                except Exception:
                    creds = {}
            return {
                "id": row[0],
                "connected_email": row[2],
                "credentials": creds,
                "status": row[3],
            }
    finally:
        conn.close()


def save_gmail_integration(
    organization_id: str,
    connected_email: str,
    credentials: dict,
) -> None:
    conn = _pg_conn()
    try:
        int_id = _new_id("int_")
        encrypted = encrypt_text(json.dumps(credentials))
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO organization_integrations
                    (id, organization_id, provider, credentials_json, connected_email, status)
                VALUES (%s, %s, 'gmail', %s, %s, 'active')
                ON CONFLICT (organization_id, provider) DO UPDATE SET
                    credentials_json = EXCLUDED.credentials_json,
                    connected_email = EXCLUDED.connected_email,
                    status = 'active',
                    updated_at = NOW()
                """,
                (int_id, organization_id, encrypted, connected_email),
            )
        conn.commit()
    finally:
        conn.close()


def delete_gmail_integration(organization_id: str) -> None:
    conn = _pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM organization_integrations WHERE organization_id = %s AND provider = 'gmail'",
                (organization_id,),
            )
        conn.commit()
    finally:
        conn.close()


def list_orgs_with_gmail() -> list[str]:
    conn = _pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT organization_id FROM organization_integrations
                WHERE provider = 'gmail' AND status = 'active'
                """
            )
            return [r[0] for r in cur.fetchall()]
    finally:
        conn.close()
