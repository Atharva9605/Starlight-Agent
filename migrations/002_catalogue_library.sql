-- Structured digital catalogue library (idempotent)

CREATE TABLE IF NOT EXISTS catalogues (
    id TEXT PRIMARY KEY,
    organization_id TEXT NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    slug TEXT NOT NULL,
    name TEXT NOT NULL DEFAULT '',
    source_filename TEXT NOT NULL DEFAULT '',
    page_count INTEGER NOT NULL DEFAULT 0,
    cover_image_url TEXT NOT NULL DEFAULT '',
    share_enabled BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (organization_id, slug)
);

CREATE INDEX IF NOT EXISTS idx_catalogues_org ON catalogues (organization_id);
CREATE INDEX IF NOT EXISTS idx_catalogues_source
    ON catalogues (organization_id, source_filename);

CREATE TABLE IF NOT EXISTS catalogue_products (
    id TEXT PRIMARY KEY,
    organization_id TEXT NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    catalogue_id TEXT NOT NULL REFERENCES catalogues(id) ON DELETE CASCADE,
    product_name TEXT NOT NULL DEFAULT '',
    category TEXT NOT NULL DEFAULT 'other',
    description TEXT NOT NULL DEFAULT '',
    features JSONB NOT NULL DEFAULT '[]'::jsonb,
    specs JSONB NOT NULL DEFAULT '{}'::jsonb,
    variants JSONB NOT NULL DEFAULT '[]'::jsonb,
    page_number INTEGER NOT NULL DEFAULT 0,
    image_url TEXT NOT NULL DEFAULT '',
    specs_preview TEXT NOT NULL DEFAULT '',
    sort_order INTEGER NOT NULL DEFAULT 0,
    chunk_id TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_catalogue_products_catalogue_page
    ON catalogue_products (catalogue_id, page_number);
CREATE INDEX IF NOT EXISTS idx_catalogue_products_org_category
    ON catalogue_products (organization_id, category);
CREATE INDEX IF NOT EXISTS idx_catalogue_products_chunk
    ON catalogue_products (chunk_id);
CREATE INDEX IF NOT EXISTS idx_catalogue_products_org
    ON catalogue_products (organization_id);
