"""
Vector store — persistent catalogue chunks.

When DATABASE_URL is set (Neon PostgreSQL + pgvector), embeddings are stored in
hosted Postgres and survive container rebuilds / HF Space redeploys.

When DATABASE_URL is unset, falls back to local ChromaDB (dev only).
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

from embedding_config import get_embedding_dimension, validate_embedding

log = logging.getLogger("vector_store")

COLLECTION_NAME = "starlight_vision"

if os.path.exists("/data") and os.path.isdir("/data"):
    CHROMA_DB_DIR = "/data/chroma_db"
else:
    CHROMA_DB_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")


def use_postgres() -> bool:
    return bool(os.getenv("DATABASE_URL", "").strip())


def saas_mode() -> bool:
    return os.getenv("SAAS_MODE", "true").strip().lower() in {"1", "true", "yes"}


def require_scoped_backend() -> None:
    """SaaS multi-tenant mode requires Postgres/pgvector — Chroma has no org isolation."""
    if saas_mode() and not use_postgres():
        raise RuntimeError(
            "SAAS_MODE=true requires DATABASE_URL (Postgres + pgvector). "
            "Local ChromaDB is not org-scoped and cannot be used in SaaS mode."
        )


def backend_name() -> str:
    return "neon_pgvector" if use_postgres() else "chromadb_local"


def _pg_conn():
    import psycopg
    from pgvector.psycopg import register_vector

    conn = psycopg.connect(os.environ["DATABASE_URL"])
    # Extension must exist before register_vector() looks up the type OID.
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.commit()
    register_vector(conn)
    return conn


def _pg_embedding_column_dim(cur) -> int | None:
    """Read vector(n) dimension from existing table, or None if table missing."""
    cur.execute(
        """
        SELECT format_type(a.atttypid, a.atttypmod)
        FROM pg_attribute a
        JOIN pg_class c ON a.attrelid = c.oid
        JOIN pg_namespace n ON c.relnamespace = n.oid
        WHERE c.relname = 'catalogue_chunks'
          AND n.nspname = current_schema()
          AND a.attname = 'embedding'
          AND NOT a.attisdropped
        """
    )
    row = cur.fetchone()
    if not row or not row[0]:
        return None
    # format_type returns e.g. 'vector(1536)'
    typ = str(row[0])
    if typ.startswith("vector(") and typ.endswith(")"):
        try:
            return int(typ[7:-1])
        except ValueError:
            return None
    return None


def init_db() -> None:
    """Create pgvector table/indexes. No-op when using Chroma fallback."""
    if not use_postgres():
        log.info("DATABASE_URL not set — using local ChromaDB at %s", CHROMA_DB_DIR)
        return

    dim = get_embedding_dimension()
    conn = _pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

            existing_dim = _pg_embedding_column_dim(cur)
            if existing_dim is not None and existing_dim != dim:
                raise RuntimeError(
                    f"catalogue_chunks.embedding is vector({existing_dim}) but "
                    f"current embedding model expects vector({dim}). "
                    f"Either set EMBEDDING_DIMENSION={existing_dim}, or run "
                    f"TRUNCATE catalogue_chunks / drop the table and re-upload catalogues."
                )

            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS catalogue_chunks (
                    id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    catalogue_name TEXT NOT NULL DEFAULT '',
                    document TEXT NOT NULL,
                    embedding vector({dim}),
                    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_catalogue_chunks_source "
                "ON catalogue_chunks (source)"
            )
            # HNSW cosine index — matches Chroma hnsw:space=cosine
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_catalogue_chunks_embedding
                ON catalogue_chunks USING hnsw (embedding vector_cosine_ops)
                """
            )
        conn.commit()
        log.info("PostgreSQL vector store ready (dim=%s, cosine HNSW)", dim)

        from conversation_store import init_conversation_tables
        init_conversation_tables()

        from org_store import run_saas_migrations
        run_saas_migrations()
    finally:
        conn.close()


def _chroma_collection():
    import chromadb

    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    return client.get_or_create_collection(
        COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )


def add_chunks(
    ids: list[str],
    embeddings: list[list[float]],
    documents: list[str],
    metadatas: list[dict[str, Any]],
    organization_id: str | None = None,
) -> None:
    if not ids:
        return

    require_scoped_backend()

    if organization_id is None:
        from tenant import get_organization_id
        organization_id = get_organization_id()

    if len({len(ids), len(embeddings), len(documents), len(metadatas)}) != 1:
        raise ValueError(
            f"add_chunks length mismatch: ids={len(ids)}, embeddings={len(embeddings)}, "
            f"documents={len(documents)}, metadatas={len(metadatas)}"
        )

    for i, emb in enumerate(embeddings):
        validate_embedding(emb, context=f"chunk index {i}")

    if use_postgres():
        conn = _pg_conn()
        try:
            with conn.cursor() as cur:
                for chunk_id, emb, doc, meta in zip(ids, embeddings, documents, metadatas):
                    source = str(meta.get("source", ""))
                    cat_name = str(meta.get("catalogue_name", ""))
                    cur.execute(
                        """
                        INSERT INTO catalogue_chunks
                            (id, source, catalogue_name, document, embedding, metadata, organization_id)
                        VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s)
                        ON CONFLICT (id) DO UPDATE SET
                            source = EXCLUDED.source,
                            catalogue_name = EXCLUDED.catalogue_name,
                            document = EXCLUDED.document,
                            embedding = EXCLUDED.embedding,
                            metadata = EXCLUDED.metadata,
                            organization_id = EXCLUDED.organization_id
                        """,
                        (chunk_id, source, cat_name, doc, emb, json.dumps(meta), organization_id),
                    )
            conn.commit()
        finally:
            conn.close()
        return

    collection = _chroma_collection()
    collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)


def delete_by_source(source: str, organization_id: str | None = None) -> int:
    if not source:
        return 0

    require_scoped_backend()

    if organization_id is None:
        from tenant import get_organization_id
        organization_id = get_organization_id()

    if use_postgres():
        conn = _pg_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM catalogue_chunks WHERE source = %s AND organization_id = %s",
                    (source, organization_id),
                )
                deleted = cur.rowcount
            conn.commit()
            return deleted
        finally:
            conn.close()

    collection = _chroma_collection()
    try:
        existing = collection.get(where={"source": source})
        ids = existing.get("ids") or []
        if ids:
            collection.delete(ids=ids)
        return len(ids)
    except Exception:
        return 0


def clear_all(organization_id: str | None = None) -> None:
    require_scoped_backend()

    if organization_id is None:
        from tenant import get_organization_id
        organization_id = get_organization_id()

    if use_postgres():
        conn = _pg_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM catalogue_chunks WHERE organization_id = %s",
                    (organization_id,),
                )
            conn.commit()
        finally:
            conn.close()
        return

    import chromadb

    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    client.create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})


def _build_where_sql(where: dict[str, Any] | None) -> tuple[str, list[Any]]:
    """
    Build extra AND clauses from a filter dict.

    Supported keys:
      exclude_kinds: list[str] — drop chunks whose metadata.kind is in this list
      kinds: list[str] — only include these kinds (NULL kind treated as catalogue)
      conversation_id: str — include email_attachment chunks for this thread only
      catalogue_only: bool — shorthand: exclude kind=email_attachment
    """
    if not where:
        return "", []

    clauses: list[str] = []
    params: list[Any] = []

    if where.get("catalogue_only"):
        clauses.append(
            "(metadata->>'kind' IS NULL OR metadata->>'kind' <> 'email_attachment')"
        )

    exclude_kinds = where.get("exclude_kinds") or []
    if exclude_kinds:
        clauses.append(
            "(metadata->>'kind' IS NULL OR NOT (metadata->>'kind' = ANY(%s)))"
        )
        params.append(list(exclude_kinds))

    kinds = where.get("kinds")
    if kinds:
        clauses.append(
            "(COALESCE(metadata->>'kind', 'catalogue') = ANY(%s))"
        )
        params.append(list(kinds))

    conversation_id = where.get("conversation_id")
    if conversation_id:
        # Catalogue chunks OR attachments for this conversation
        clauses.append(
            """(
                metadata->>'kind' IS NULL
                OR metadata->>'kind' <> 'email_attachment'
                OR metadata->>'conversation_id' = %s
            )"""
        )
        params.append(conversation_id)

    if not clauses:
        return "", []
    return " AND " + " AND ".join(clauses), params


def query(
    query_embedding: list[float],
    k: int = 5,
    organization_id: str | None = None,
    where: dict[str, Any] | None = None,
) -> dict[str, list]:
    """Return Chroma-compatible shape: documents, metadatas, distances (each a list of lists)."""
    validate_embedding(query_embedding, context="query")
    require_scoped_backend()

    if organization_id is None:
        from tenant import get_organization_id
        organization_id = get_organization_id()

    if use_postgres():
        extra_sql, extra_params = _build_where_sql(where)
        conn = _pg_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT document, metadata, (embedding <=> %s::vector) AS distance
                    FROM catalogue_chunks
                    WHERE organization_id = %s{extra_sql}
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (query_embedding, organization_id, *extra_params, query_embedding, k),
                )
                rows = cur.fetchall()
        finally:
            conn.close()

        if not rows:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

        docs = [r[0] for r in rows]
        metas = [r[1] if isinstance(r[1], dict) else json.loads(r[1]) for r in rows]
        dists = [float(r[2]) for r in rows]
        return {"documents": [docs], "metadatas": [metas], "distances": [dists]}

    # Dev-only Chroma path (blocked when SAAS_MODE=true via require_scoped_backend)
    collection = _chroma_collection()
    chroma_where = None
    if where and where.get("catalogue_only"):
        chroma_where = {"kind": {"$ne": "email_attachment"}}
    kwargs: dict[str, Any] = {
        "query_embeddings": [query_embedding],
        "n_results": k,
        "include": ["documents", "metadatas", "distances"],
    }
    if chroma_where:
        kwargs["where"] = chroma_where
    try:
        return collection.query(**kwargs)
    except Exception:
        # Older chroma metadata may lack "kind" — fall back unfiltered in local only
        return collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )


def get_status(organization_id: str | None = None) -> tuple[int, list[str]]:
    if organization_id is None:
        from tenant import get_organization_id
        organization_id = get_organization_id()

    if use_postgres():
        conn = _pg_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM catalogue_chunks WHERE organization_id = %s",
                    (organization_id,),
                )
                count = int(cur.fetchone()[0])
                cur.execute(
                    "SELECT DISTINCT catalogue_name FROM catalogue_chunks "
                    "WHERE organization_id = %s AND catalogue_name <> '' ORDER BY catalogue_name",
                    (organization_id,),
                )
                catalogues = [row[0] for row in cur.fetchall()]
            return count, catalogues
        except Exception as exc:
            log.warning("KB status query failed: %s", exc)
            return 0, []
        finally:
            conn.close()

    try:
        collection = _chroma_collection()
        count = collection.count()
        if count == 0:
            return 0, []
        results = collection.get(limit=min(count, 500), include=["metadatas"])
        catalogues: set[str] = set()
        for m in results.get("metadatas") or []:
            name = m.get("catalogue_name", "")
            if name:
                catalogues.add(name)
        return count, sorted(catalogues)
    except Exception:
        return 0, []


def store_info() -> dict[str, Any]:
    """Diagnostics for /api/kb-status."""
    info: dict[str, Any] = {
        "backend": backend_name(),
        "embedding_dimension": get_embedding_dimension(),
        "collection": COLLECTION_NAME,
    }
    deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
    info["embedding_deployment"] = deployment
    if use_postgres():
        info["database"] = "postgresql (Neon/pgvector)"
    else:
        info["database"] = f"chromadb ({CHROMA_DB_DIR})"
    return info
