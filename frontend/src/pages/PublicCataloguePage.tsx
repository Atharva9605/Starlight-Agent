import { useEffect, useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useParams } from 'react-router-dom'
import {
  DismissRegular,
  DocumentSearchRegular,
  ErrorCircleRegular,
  OpenRegular,
  SearchRegular,
} from '@fluentui/react-icons'
import { api } from '../api/client'
import { EmptyState, Spinner, useDocumentTitle } from '../components/ui'
import {
  cleanCatalogueTitle,
  toDisplayProduct,
  type DisplayProduct,
} from '../catalogue/productDisplay'

function dedupeFeatures(product: DisplayProduct): string[] {
  const specVals = new Set(
    product.specs.map((s) => s.value.toLowerCase()).concat(
      product.specsPreview.toLowerCase().split(/[·|]/).map((x) => x.trim()),
    ),
  )
  return product.features.filter((f) => {
    const lower = f.toLowerCase()
    // Drop features that only restate IP/IK already in the table
    if (/^temporary immersion|^impact resistant/i.test(f) && (specVals.has('ip67') || /ip67|ik10/.test(lower))) {
      return !/ip67|ik10/i.test(f) || product.features.length <= 2
    }
    return true
  })
}

function ProductCard({
  product,
  onOpen,
}: {
  product: DisplayProduct
  onOpen: (p: DisplayProduct) => void
}) {
  const initial = (product.name || 'P').slice(0, 1).toUpperCase()
  const highlight = product.specs
    .filter((s) => ['dimensions', 'wattage', 'ip_rating', 'voltage'].includes(s.key))
    .slice(0, 3)

  return (
    <article className="pc-card">
      <button
        type="button"
        className={`pc-media${product.imageUrl ? '' : ' is-empty'}`}
        onClick={() => onOpen(product)}
        aria-label={`Open ${product.name}`}
      >
        {product.imageUrl ? (
          <img src={product.imageUrl} alt="" loading="lazy" />
        ) : (
          <span className="pc-monogram" aria-hidden>
            {initial}
          </span>
        )}
        {product.pageNumber ? <span className="pc-page-badge">p. {product.pageNumber}</span> : null}
      </button>

      <div className="pc-body">
        <div className="pc-meta-row">
          {product.code ? <span className="pc-code">{product.code}</span> : null}
          {product.category ? (
            <span className="pc-cat">{product.category.replace(/_/g, ' ')}</span>
          ) : null}
        </div>

        <h2 className="pc-title">{product.name}</h2>

        {highlight.length ? (
          <dl className="pc-specs">
            {highlight.map((s) => (
              <div key={s.key} className="pc-spec">
                <dt>{s.label}</dt>
                <dd>{s.value}</dd>
              </div>
            ))}
          </dl>
        ) : null}

        <button type="button" className="pc-link" onClick={() => onOpen(product)}>
          View details
        </button>
      </div>
    </article>
  )
}

function ProductModal({
  product,
  onClose,
}: {
  product: DisplayProduct
  onClose: () => void
}) {
  const features = dedupeFeatures(product)

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    const prev = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    return () => {
      window.removeEventListener('keydown', onKey)
      document.body.style.overflow = prev
    }
  }, [onClose])

  return (
    <div className="pc-modal" role="dialog" aria-modal="true" aria-label={product.name}>
      <button type="button" className="pc-modal-backdrop" aria-label="Close" onClick={onClose} />
      <div className="pc-modal-panel">
        <button type="button" className="btn subtle icon-only pc-modal-close" aria-label="Close" title="Close" onClick={onClose}>
          <DismissRegular />
        </button>

        {product.imageUrl ? (
          <div className="pc-modal-preview">
            <img src={product.imageUrl} alt="" />
          </div>
        ) : (
          <div className="pc-modal-preview is-empty" aria-hidden>
            <span className="pc-monogram">{(product.name || 'P').slice(0, 1)}</span>
          </div>
        )}

        <div className="pc-modal-body">
          <div className="pc-meta-row">
            {product.code ? <span className="pc-code">{product.code}</span> : null}
            {product.category ? (
              <span className="pc-cat">{product.category.replace(/_/g, ' ')}</span>
            ) : null}
            {product.pageNumber ? (
              <span className="pc-page-chip">p. {product.pageNumber}</span>
            ) : null}
          </div>
          <h2>{product.name}</h2>
          {product.description ? <p className="pc-modal-tagline">{product.description}</p> : null}

          <section className="pc-modal-section">
            <h3 className="pc-aside-label">Specifications</h3>
            {product.specs.length ? (
              <div className="pc-spec-grid">
                {product.specs.map((s) => (
                  <div key={s.key} className="pc-spec-tile">
                    <span className="pc-spec-tile-label">{s.label}</span>
                    <span className="pc-spec-tile-value">{s.value}</span>
                  </div>
                ))}
              </div>
            ) : (
              <p className="pc-status">No structured specs for this product yet.</p>
            )}
          </section>

          {features.length ? (
            <section className="pc-modal-section">
              <h3 className="pc-aside-label">Highlights</h3>
              <ul className="pc-features">
                {features.map((f) => (
                  <li key={f}>{f}</li>
                ))}
              </ul>
            </section>
          ) : null}

          {product.imageUrl ? (
            <a className="btn" href={product.imageUrl} target="_blank" rel="noreferrer">
              <OpenRegular /> Open full catalogue page
              {product.pageNumber ? ` (p. ${product.pageNumber})` : ''}
            </a>
          ) : null}
        </div>
      </div>
    </div>
  )
}

export function PublicCataloguePage() {
  const { orgSlug = '', catalogueSlug = '' } = useParams()
  const [category, setCategory] = useState('all')
  const [query, setQuery] = useState('')
  const [active, setActive] = useState<DisplayProduct | null>(null)

  const q = useQuery({
    queryKey: ['public-catalogue', orgSlug, catalogueSlug],
    queryFn: () => api.publicCatalogue(orgSlug, catalogueSlug),
    enabled: Boolean(orgSlug && catalogueSlug),
    retry: false,
  })
  useDocumentTitle(q.data ? cleanCatalogueTitle(q.data.name) : 'Digital catalogue')

  const displayProducts = useMemo(
    () => (q.data?.products || []).map(toDisplayProduct),
    [q.data?.products],
  )

  const categories = useMemo(() => {
    const set = new Set<string>()
    for (const p of displayProducts) {
      if (p.category) set.add(p.category)
    }
    const list = Array.from(set).sort()
    const useful = list.filter((c) => c.length <= 28)
    return useful.length ? ['all', ...useful] : []
  }, [displayProducts])

  const products = useMemo(() => {
    const qLower = query.trim().toLowerCase()
    return displayProducts.filter((p) => {
      if (category !== 'all' && p.category !== category) return false
      if (!qLower) return true
      const hay = [p.name, p.code, p.specsPreview, p.description, ...p.specs.map((s) => s.value)]
        .join(' ')
        .toLowerCase()
      return hay.includes(qLower)
    })
  }, [displayProducts, category, query])

  if (q.isLoading) {
    return (
      <div className="pc-page">
        <div className="pc-shell center-fill" style={{ minHeight: '60vh' }}>
          <Spinner size="lg" label="Loading catalogue…" />
        </div>
      </div>
    )
  }

  if (q.isError || !q.data) {
    return (
      <div className="pc-page">
        <div className="pc-shell center-fill" style={{ minHeight: '60vh' }}>
          <EmptyState
            icon={<ErrorCircleRegular />}
            title="Catalogue unavailable"
            description={(q.error as Error)?.message || 'This catalogue link may have expired or been turned off.'}
          />
        </div>
      </div>
    )
  }

  const cat = q.data
  const title = cleanCatalogueTitle(cat.name)
  const withImages = displayProducts.filter((p) => p.imageUrl).length

  return (
    <div className="pc-page">
      <header className="pc-hero">
        {cat.cover_image_url ? (
          <img className="pc-hero-bg" src={cat.cover_image_url} alt="" />
        ) : null}
        <div className="pc-hero-veil" />
        <div className="pc-hero-copy">
          <p className="pc-brand">{cat.organization_name || 'Starlight Linear LED'}</p>
          <h1>{title}</h1>
          <p className="pc-hero-sub">
            {displayProducts.length} products
            {withImages ? ` · ${withImages} with page art` : ''}
            {cat.page_count ? ` · ${cat.page_count} pages` : ''}
          </p>
        </div>
      </header>

      <div className="pc-shell">
        <div className="pc-toolbar">
          <div className="input-wrap pc-search">
            <SearchRegular />
            <input
              className="input"
              type="search"
              aria-label="Search products"
              placeholder="Search name, code, wattage…"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
          </div>
          {categories.length ? (
            <div className="pc-filters" role="tablist" aria-label="Categories">
              {categories.map((c) => (
                <button
                  key={c}
                  type="button"
                  role="tab"
                  aria-selected={category === c}
                  className={`pc-filter${category === c ? ' is-active' : ''}`}
                  onClick={() => setCategory(c)}
                >
                  {c === 'all' ? 'All' : c.replace(/_/g, ' ')}
                </button>
              ))}
            </div>
          ) : null}
        </div>

        {!products.length ? (
          <EmptyState
            icon={<DocumentSearchRegular />}
            title="No matching products"
            description="Try another search or category."
          />
        ) : (
          <div className="pc-grid">
            {products.map((p) => (
              <ProductCard key={p.id} product={p} onOpen={setActive} />
            ))}
          </div>
        )}

        <footer className="pc-footer">
          <span>{cat.organization_name || 'Starlight Linear LED'}</span>
          <span>Digital product catalogue</span>
        </footer>
      </div>

      {active ? <ProductModal product={active} onClose={() => setActive(null)} /> : null}
    </div>
  )
}
