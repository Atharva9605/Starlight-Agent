import { useEffect, useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useParams } from 'react-router-dom'
import { api } from '../api/client'
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
  const topSpecs = product.specs.filter((s) => s.key !== 'code').slice(0, 4)

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

        {topSpecs.length ? (
          <p className="pc-preview">
            {topSpecs.map((s) => s.value).join(' · ')}
          </p>
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
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  return (
    <div className="pc-modal" role="dialog" aria-modal="true" aria-label={product.name}>
      <button type="button" className="pc-modal-backdrop" aria-label="Close" onClick={onClose} />
      <div className="pc-modal-panel">
        <header className="pc-modal-head">
          <div className="pc-modal-titleblock">
            <div className="pc-meta-row">
              {product.code ? <span className="pc-code">{product.code}</span> : null}
              {product.category ? (
                <span className="pc-cat">{product.category.replace(/_/g, ' ')}</span>
              ) : null}
              {product.pageNumber ? <span className="pc-page">Catalogue p. {product.pageNumber}</span> : null}
            </div>
            <h2>{product.name}</h2>
            {product.description ? <p className="pc-modal-tagline">{product.description}</p> : null}
          </div>
          <button type="button" className="pc-modal-close" onClick={onClose}>
            Close
          </button>
        </header>

        <div className="pc-modal-grid">
          <div className="pc-modal-media">
            {product.imageUrl ? (
              <img src={product.imageUrl} alt={`${product.name} catalogue page`} />
            ) : (
              <div className="pc-monogram" style={{ minHeight: 320 }}>
                {(product.name || 'P').slice(0, 1)}
              </div>
            )}
          </div>

          <aside className="pc-modal-info">
            <h3 className="pc-aside-label">Specifications</h3>
            {product.specs.length ? (
              <dl className="pc-specs pc-specs-modal">
                {product.specs.map((s) => (
                  <div key={s.key} className="pc-spec">
                    <dt>{s.label}</dt>
                    <dd>{s.value}</dd>
                  </div>
                ))}
              </dl>
            ) : (
              <p className="pc-status">No structured specs for this product yet.</p>
            )}

            {features.length ? (
              <>
                <h3 className="pc-aside-label">Highlights</h3>
                <ul className="pc-features">
                  {features.map((f) => (
                    <li key={f}>{f}</li>
                  ))}
                </ul>
              </>
            ) : null}

            {product.imageUrl ? (
              <a className="pc-btn" href={product.imageUrl} target="_blank" rel="noreferrer">
                Open full catalogue page
              </a>
            ) : null}
          </aside>
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
        <div className="pc-shell">
          <p className="pc-status">Loading catalogue…</p>
        </div>
      </div>
    )
  }

  if (q.isError || !q.data) {
    return (
      <div className="pc-page">
        <div className="pc-shell">
          <h1 className="pc-error-title">Catalogue unavailable</h1>
          <p className="pc-status">{(q.error as Error)?.message || 'Not found'}</p>
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
          <label className="pc-search">
            <span className="sr-only">Search products</span>
            <input
              type="search"
              placeholder="Search name, code, wattage…"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
          </label>
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
          <div className="pc-empty">
            <strong>No matching products</strong>
            <p>Try another search or category.</p>
          </div>
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
