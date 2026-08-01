"""
Structured digital catalogue library (Postgres).

Products are seeded by PDF Vision ingest and power:
  • public web catalogue pages (/c/{org_slug}/{catalogue_slug})
  • outbound email product cards + catalogue links

No-ops when DATABASE_URL is unset (Chroma-only local mode).
"""
from __future__ import annotations

import json
import logging
import os
import re
import uuid
from typing import Any, Optional

log = logging.getLogger("catalogue_library")


def _pg_conn():
    from vector_store import _pg_conn as conn_fn
    return conn_fn()


def _use_postgres() -> bool:
    from vector_store import use_postgres
    return use_postgres()


def _org_id(organization_id: str | None = None) -> str:
    if organization_id:
        return organization_id
    from tenant import get_organization_id
    return get_organization_id()


def _new_id() -> str:
    return str(uuid.uuid4())


def frontend_base_url() -> str:
    return os.getenv("FRONTEND_URL", "http://localhost:5173").rstrip("/")


def share_url(org_slug: str, catalogue_slug: str) -> str:
    return f"{frontend_base_url()}/c/{org_slug}/{catalogue_slug}"


def get_org_by_slug(slug: str) -> Optional[dict]:
    if not _use_postgres() or not slug:
        return None
    conn = _pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, name, slug FROM organizations WHERE slug = %s",
                (slug,),
            )
            row = cur.fetchone()
            if not row:
                return None
            return {"id": row[0], "name": row[1], "slug": row[2]}
    finally:
        conn.close()


def get_org_slug(organization_id: str | None = None) -> str:
    oid = _org_id(organization_id)
    if not _use_postgres():
        return "default"
    conn = _pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT slug FROM organizations WHERE id = %s", (oid,))
            row = cur.fetchone()
            return row[0] if row else "default"
    finally:
        conn.close()


def upsert_catalogue(
    *,
    slug: str,
    name: str,
    source_filename: str,
    page_count: int = 0,
    cover_image_url: str = "",
    organization_id: str | None = None,
    share_enabled: bool = True,
) -> Optional[dict]:
    """Create or update a catalogue row keyed by (org, slug). Returns the row dict."""
    if not _use_postgres():
        return None

    oid = _org_id(organization_id)
    conn = _pg_conn()
    try:
        with conn.cursor() as cur:
            # Prefer existing row by slug, else by source filename (re-upload)
            cur.execute(
                """
                SELECT id FROM catalogues
                WHERE organization_id = %s AND slug = %s
                """,
                (oid, slug),
            )
            existing = cur.fetchone()
            cat_id = existing[0] if existing else None

            if not cat_id and source_filename:
                cur.execute(
                    """
                    SELECT id FROM catalogues
                    WHERE organization_id = %s AND source_filename = %s
                    """,
                    (oid, source_filename),
                )
                by_source = cur.fetchone()
                if by_source:
                    cat_id = by_source[0]

            if cat_id:
                cur.execute(
                    """
                    UPDATE catalogues SET
                        slug = %s,
                        name = %s,
                        source_filename = %s,
                        page_count = %s,
                        cover_image_url = COALESCE(NULLIF(%s, ''), cover_image_url),
                        share_enabled = %s,
                        updated_at = NOW()
                    WHERE id = %s
                    RETURNING id, organization_id, slug, name, source_filename,
                              page_count, cover_image_url, share_enabled
                    """,
                    (
                        slug,
                        name,
                        source_filename,
                        page_count,
                        cover_image_url or "",
                        share_enabled,
                        cat_id,
                    ),
                )
            else:
                cat_id = _new_id()
                cur.execute(
                    """
                    INSERT INTO catalogues (
                        id, organization_id, slug, name, source_filename,
                        page_count, cover_image_url, share_enabled, updated_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
                    ON CONFLICT (organization_id, slug) DO UPDATE SET
                        name = EXCLUDED.name,
                        source_filename = EXCLUDED.source_filename,
                        page_count = EXCLUDED.page_count,
                        cover_image_url = COALESCE(
                            NULLIF(EXCLUDED.cover_image_url, ''),
                            catalogues.cover_image_url
                        ),
                        share_enabled = EXCLUDED.share_enabled,
                        updated_at = NOW()
                    RETURNING id, organization_id, slug, name, source_filename,
                              page_count, cover_image_url, share_enabled
                    """,
                    (
                        cat_id,
                        oid,
                        slug,
                        name,
                        source_filename,
                        page_count,
                        cover_image_url or "",
                        share_enabled,
                    ),
                )
            row = cur.fetchone()
        conn.commit()
        if not row:
            return None
        return {
            "id": row[0],
            "organization_id": row[1],
            "slug": row[2],
            "name": row[3],
            "source_filename": row[4],
            "page_count": row[5],
            "cover_image_url": row[6],
            "share_enabled": row[7],
        }
    except Exception:
        conn.rollback()
        log.exception("upsert_catalogue failed for slug=%s", slug)
        raise
    finally:
        conn.close()


def replace_products(
    catalogue_id: str,
    products: list[dict[str, Any]],
    organization_id: str | None = None,
) -> int:
    """Delete existing products for a catalogue and insert the new set."""
    if not _use_postgres() or not catalogue_id:
        return 0

    oid = _org_id(organization_id)
    conn = _pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM catalogue_products WHERE catalogue_id = %s AND organization_id = %s",
                (catalogue_id, oid),
            )
            for i, p in enumerate(products):
                pid = str(p.get("id") or p.get("chunk_id") or _new_id())
                features = p.get("features") or []
                specs = p.get("specs") or {}
                variants = p.get("variants") or []
                if isinstance(features, (str, bytes)):
                    features = [features]
                if not isinstance(specs, dict):
                    specs = {}
                if isinstance(variants, (str, bytes)):
                    variants = [variants]

                cur.execute(
                    """
                    INSERT INTO catalogue_products (
                        id, organization_id, catalogue_id, product_name, category,
                        description, features, specs, variants, page_number,
                        image_url, specs_preview, sort_order, chunk_id
                    )
                    VALUES (
                        %s, %s, %s, %s, %s,
                        %s, %s::jsonb, %s::jsonb, %s::jsonb, %s,
                        %s, %s, %s, %s
                    )
                    """,
                    (
                        pid,
                        oid,
                        catalogue_id,
                        str(p.get("product_name") or "Unknown Product"),
                        str(p.get("category") or "other"),
                        str(p.get("description") or ""),
                        json.dumps(features),
                        json.dumps(specs),
                        json.dumps(variants),
                        int(p.get("page_number") or 0),
                        str(p.get("image_url") or p.get("blob_url") or ""),
                        str(p.get("specs_preview") or ""),
                        int(p.get("sort_order") if p.get("sort_order") is not None else i),
                        str(p.get("chunk_id") or pid),
                    ),
                )
        conn.commit()
        return len(products)
    except Exception:
        conn.rollback()
        log.exception("replace_products failed for catalogue_id=%s", catalogue_id)
        raise
    finally:
        conn.close()


def delete_catalogue_by_source(
    source_filename: str,
    organization_id: str | None = None,
) -> int:
    """Remove catalogue + products for a source PDF (CASCADE deletes products)."""
    if not _use_postgres() or not source_filename:
        return 0
    oid = _org_id(organization_id)
    conn = _pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM catalogues WHERE organization_id = %s AND source_filename = %s",
                (oid, source_filename),
            )
            deleted = cur.rowcount
        conn.commit()
        return deleted
    finally:
        conn.close()


def clear_library(organization_id: str | None = None) -> None:
    if not _use_postgres():
        return
    oid = _org_id(organization_id)
    conn = _pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM catalogue_products WHERE organization_id = %s",
                (oid,),
            )
            cur.execute(
                "DELETE FROM catalogues WHERE organization_id = %s",
                (oid,),
            )
        conn.commit()
    finally:
        conn.close()


def _row_to_catalogue(row, org_slug: str | None = None) -> dict:
    cat = {
        "id": row[0],
        "organization_id": row[1],
        "slug": row[2],
        "name": row[3],
        "source_filename": row[4],
        "page_count": row[5],
        "cover_image_url": row[6] or "",
        "share_enabled": bool(row[7]),
        "created_at": row[8].isoformat() if row[8] else None,
        "updated_at": row[9].isoformat() if row[9] else None,
        "product_count": row[10] if len(row) > 10 else 0,
    }
    if org_slug:
        cat["org_slug"] = org_slug
        cat["share_url"] = share_url(org_slug, cat["slug"])
    return cat


def _row_to_product(row) -> dict:
    return {
        "id": row[0],
        "organization_id": row[1],
        "catalogue_id": row[2],
        "product_name": row[3],
        "category": row[4],
        "description": row[5] or "",
        "features": row[6] if isinstance(row[6], list) else (row[6] or []),
        "specs": row[7] if isinstance(row[7], dict) else (row[7] or {}),
        "variants": row[8] if isinstance(row[8], list) else (row[8] or []),
        "page_number": row[9],
        "image_url": row[10] or "",
        "blob_url": row[10] or "",  # alias for email templates
        "specs_preview": row[11] or "",
        "sort_order": row[12],
        "chunk_id": row[13],
    }


_CAT_SELECT = """
    SELECT c.id, c.organization_id, c.slug, c.name, c.source_filename,
           c.page_count, c.cover_image_url, c.share_enabled,
           c.created_at, c.updated_at,
           (SELECT COUNT(*) FROM catalogue_products p WHERE p.catalogue_id = c.id) AS product_count
    FROM catalogues c
"""

_PROD_SELECT = """
    SELECT id, organization_id, catalogue_id, product_name, category,
           description, features, specs, variants, page_number,
           image_url, specs_preview, sort_order, chunk_id
    FROM catalogue_products
"""


def list_catalogues(organization_id: str | None = None) -> list[dict]:
    if not _use_postgres():
        return []
    oid = _org_id(organization_id)
    org_slug = get_org_slug(oid)
    conn = _pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                _CAT_SELECT + " WHERE c.organization_id = %s ORDER BY c.updated_at DESC",
                (oid,),
            )
            return [_row_to_catalogue(r, org_slug) for r in cur.fetchall()]
    finally:
        conn.close()


def get_catalogue(
    catalogue_id: str,
    organization_id: str | None = None,
    *,
    include_products: bool = True,
) -> Optional[dict]:
    if not _use_postgres() or not catalogue_id:
        return None
    oid = _org_id(organization_id)
    org_slug = get_org_slug(oid)
    conn = _pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                _CAT_SELECT + " WHERE c.id = %s AND c.organization_id = %s",
                (catalogue_id, oid),
            )
            row = cur.fetchone()
            if not row:
                return None
            cat = _row_to_catalogue(row, org_slug)
            if include_products:
                cur.execute(
                    _PROD_SELECT
                    + " WHERE catalogue_id = %s ORDER BY sort_order, page_number, product_name",
                    (catalogue_id,),
                )
                cat["products"] = [_row_to_product(r) for r in cur.fetchall()]
            return cat
    finally:
        conn.close()


def get_catalogue_by_slug(
    org_slug: str,
    catalogue_slug: str,
    *,
    public_only: bool = False,
) -> Optional[dict]:
    """Resolve public or admin catalogue by org slug + catalogue slug."""
    if not _use_postgres() or not org_slug or not catalogue_slug:
        return None
    org = get_org_by_slug(org_slug)
    if not org:
        return None
    conn = _pg_conn()
    try:
        with conn.cursor() as cur:
            sql = _CAT_SELECT + " WHERE c.organization_id = %s AND c.slug = %s"
            params: list[Any] = [org["id"], catalogue_slug]
            if public_only:
                sql += " AND c.share_enabled = TRUE"
            cur.execute(sql, params)
            row = cur.fetchone()
            if not row:
                return None
            cat = _row_to_catalogue(row, org["slug"])
            cat["organization_name"] = org["name"]
            cur.execute(
                _PROD_SELECT
                + " WHERE catalogue_id = %s ORDER BY sort_order, page_number, product_name",
                (cat["id"],),
            )
            cat["products"] = [_row_to_product(r) for r in cur.fetchall()]
            return cat
    finally:
        conn.close()


def get_products_by_chunk_ids(
    chunk_ids: list[str],
    organization_id: str | None = None,
) -> list[dict]:
    if not _use_postgres() or not chunk_ids:
        return []
    oid = _org_id(organization_id)
    conn = _pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                _PROD_SELECT
                + " WHERE organization_id = %s AND chunk_id = ANY(%s)",
                (oid, list(chunk_ids)),
            )
            by_chunk = {r[13]: _row_to_product(r) for r in cur.fetchall()}
            # Preserve request order
            return [by_chunk[cid] for cid in chunk_ids if cid in by_chunk]
    finally:
        conn.close()


def resolve_products_from_rag_metas(
    metas: list[dict],
    chunk_ids: list[str] | None = None,
    organization_id: str | None = None,
) -> list[dict]:
    """
    Hydrate product refs from library rows using chunk_id first, then
    (catalogue_slug, product_name, page_number) fallback.
    """
    if not _use_postgres():
        return []

    oid = _org_id(organization_id)
    ids = chunk_ids or []
    found: dict[str, dict] = {}

    if ids:
        for p in get_products_by_chunk_ids(ids, oid):
            key = p.get("chunk_id") or p["id"]
            found[key] = p

    # Fallback lookups for metas without a library hit
    need_fallback: list[tuple[int, dict]] = []
    for i, meta in enumerate(metas):
        cid = ids[i] if i < len(ids) else ""
        if cid and cid in found:
            continue
        need_fallback.append((i, meta))

    if need_fallback:
        conn = _pg_conn()
        try:
            with conn.cursor() as cur:
                for i, meta in need_fallback:
                    slug = str(meta.get("catalogue_slug") or "")
                    name = str(meta.get("product_name") or "")
                    page = meta.get("page_number")
                    if not slug or not name:
                        continue
                    cur.execute(
                        """
                        SELECT p.id, p.organization_id, p.catalogue_id, p.product_name,
                               p.category, p.description, p.features, p.specs, p.variants,
                               p.page_number, p.image_url, p.specs_preview, p.sort_order,
                               p.chunk_id
                        FROM catalogue_products p
                        JOIN catalogues c ON c.id = p.catalogue_id
                        WHERE p.organization_id = %s
                          AND c.slug = %s
                          AND p.product_name = %s
                          AND p.page_number = %s
                        LIMIT 1
                        """,
                        (oid, slug, name, int(page or 0)),
                    )
                    row = cur.fetchone()
                    if row:
                        found[f"fb:{i}"] = _row_to_product(row)
        finally:
            conn.close()

    # Build ordered unique list matching RAG order
    out: list[dict] = []
    seen: set[str] = set()
    for i, meta in enumerate(metas):
        cid = ids[i] if i < len(ids) else ""
        prod = found.get(cid) or found.get(f"fb:{i}")
        if not prod:
            continue
        key = prod["id"]
        if key in seen:
            continue
        seen.add(key)
        enriched = {
            **prod,
            "catalogue_name": prod.get("catalogue_name")
            or meta.get("catalogue_name", ""),
            "catalogue_slug": prod.get("catalogue_slug")
            or meta.get("catalogue_slug", ""),
        }
        out.append(enriched)
    return out


def catalogue_url_for_products(
    products: list[dict],
    organization_id: str | None = None,
) -> str:
    """Build public share URL from the first product's catalogue."""
    if not products or not _use_postgres():
        return ""
    oid = _org_id(organization_id)
    org_slug = get_org_slug(oid)
    cat_id = products[0].get("catalogue_id")
    if not cat_id:
        return ""
    conn = _pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT slug FROM catalogues WHERE id = %s AND organization_id = %s",
                (cat_id, oid),
            )
            row = cur.fetchone()
            if not row:
                return ""
            return share_url(org_slug, row[0])
    finally:
        conn.close()


def _slug_from_filename(filename: str) -> str:
    import re
    from pathlib import Path

    stem = Path(filename or "catalogue").stem
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", stem).strip("_").lower()
    return (slug or "catalogue")[:60]


def _display_name_from_filename(filename: str) -> str:
    import re
    from pathlib import Path

    stem = Path(filename or "Catalogue").stem
    name = re.sub(r"[_\-]+", " ", stem).strip()
    name = re.sub(r"\s*\(\d+\)\s*$", "", name).strip()
    name = re.sub(r"\.pdf$", "", name, flags=re.I).strip()
    return name.title() or "Catalogue"


_EMPTY_SPEC = re.compile(r'^["\']?(not stated|n/?a|none|unknown|-|—|–|\.?)["\']?$', re.I)
_SKIP_SPEC_KEYS = {"anomaly", "anomalies", "raw", "page_context"}


def _unwrap_document(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        return ""
    if text.startswith("[") or text.startswith('"'):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                text = "\n".join(str(x) for x in parsed)
            elif isinstance(parsed, str):
                text = parsed
        except Exception:
            text = text.strip("[]\"' \n")
    return text.strip()


def _parse_pipe_product(document: str, meta: dict) -> dict[str, Any]:
    """Turn text-extract pipe lines into name + specs (drop 'not stated')."""
    text = _unwrap_document(document)
    primary = next((ln.strip() for ln in text.splitlines() if ln.strip()), text)
    parts = [p.strip() for p in primary.split("|") if p.strip()]

    name = str(meta.get("product_name") or "").strip()
    specs: dict[str, str] = {}
    if isinstance(meta.get("specs"), dict):
        for k, v in meta["specs"].items():
            sv = str(v or "").strip()
            if sv and not _EMPTY_SPEC.match(sv):
                specs[str(k)] = sv

    for i, part in enumerate(parts):
        if ":" in part:
            label, _, val = part.partition(":")
            key = re.sub(r"[^a-z0-9]+", "_", label.strip().lower()).strip("_")
            val = val.strip()
            if key and val and not _EMPTY_SPEC.match(val) and key not in _SKIP_SPEC_KEYS:
                specs[key] = val
            continue
        if i == 0 and (not name or "|" in name or name.startswith("[")):
            name = part
            continue
        if re.match(r"^\d+(\.\d+)?\s*x\s*\d+", part, re.I) or re.search(r"\d+\s*mm", part, re.I):
            specs.setdefault("dimensions", part)

    if not name:
        name = "Product"

    preview_keys = ("wattage", "cct", "cri", "ip", "ip_rating", "dimensions", "beam")
    preview = " · ".join(specs[k] for k in preview_keys if k in specs)[:120]

    # Short description: applications/notes only — never the raw pipe dump
    description = ""
    for key in ("applications", "notes"):
        if specs.get(key):
            description = specs.pop(key)
            break

    return {
        "product_name": name[:120],
        "specs": specs,
        "specs_preview": preview or str(meta.get("specs_preview") or ""),
        "description": description,
    }


def seed_library_from_chunks(
    source: str | None = None,
    organization_id: str | None = None,
) -> list[dict]:
    """
    Backfill catalogues + products from existing catalogue_chunks rows.

    Used for text-PDF ingest and one-shot repair when RAG exists but the
    structured library is empty.
    """
    if not _use_postgres():
        return []

    oid = _org_id(organization_id)
    conn = _pg_conn()
    seeded: list[dict] = []
    try:
        with conn.cursor() as cur:
            if source:
                cur.execute(
                    """
                    SELECT id, source, catalogue_name, document, metadata
                    FROM catalogue_chunks
                    WHERE organization_id = %s AND source = %s
                    ORDER BY id
                    """,
                    (oid, source),
                )
            else:
                cur.execute(
                    """
                    SELECT id, source, catalogue_name, document, metadata
                    FROM catalogue_chunks
                    WHERE organization_id = %s
                    ORDER BY source, id
                    """,
                    (oid,),
                )
            rows = cur.fetchall()
        conn.commit()
    finally:
        conn.close()

    by_source: dict[str, list[tuple]] = {}
    for row in rows:
        by_source.setdefault(str(row[1] or ""), []).append(row)

    for src, chunk_rows in by_source.items():
        if not src:
            continue
        meta0 = chunk_rows[0][4]
        if isinstance(meta0, str):
            meta0 = json.loads(meta0)
        if not isinstance(meta0, dict):
            meta0 = {}

        slug = str(meta0.get("catalogue_slug") or "").strip() or _slug_from_filename(src)
        name = _display_name_from_filename(
            str(meta0.get("catalogue_name") or chunk_rows[0][2] or src)
        )
        cover = ""
        for r in chunk_rows:
            m = r[4] if isinstance(r[4], dict) else (json.loads(r[4]) if r[4] else {})
            url = str((m or {}).get("blob_url") or "")
            if url and not url.startswith("["):
                cover = url
                break

        cat = upsert_catalogue(
            slug=slug,
            name=name,
            source_filename=src,
            page_count=0,
            cover_image_url=cover,
            organization_id=oid,
        )
        if not cat:
            continue

        products: list[dict[str, Any]] = []
        for i, (chunk_id, _src, _cname, document, metadata) in enumerate(chunk_rows):
            meta = metadata if isinstance(metadata, dict) else (json.loads(metadata) if metadata else {})
            if not isinstance(meta, dict):
                meta = {}
            parsed = _parse_pipe_product(str(document or ""), meta)
            products.append({
                "id": chunk_id,
                "chunk_id": chunk_id,
                "product_name": parsed["product_name"],
                "category": str(meta.get("category") or "other"),
                "description": parsed["description"] or "",
                "features": [],
                "specs": parsed["specs"],
                "variants": [],
                "page_number": int(meta.get("page_number") or 0),
                "image_url": str(meta.get("blob_url") or ""),
                "specs_preview": parsed["specs_preview"],
                "sort_order": i,
            })

        replace_products(cat["id"], products, organization_id=oid)
        seeded.append({**cat, "product_count": len(products)})
        log.info(
            "seed_library_from_chunks: %s → %s (%d products)",
            src,
            cat["slug"],
            len(products),
        )

    return seeded
