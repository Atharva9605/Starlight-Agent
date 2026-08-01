import { useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useParams } from 'react-router-dom'
import { api } from '../api/client'
import {
  cleanCatalogueTitle,
  toDisplayProduct,
  type DisplayProduct,
} from '../catalogue/productDisplay'

function ProductCard({ product }: { product: DisplayProduct }) {
  const initial = (product.name || 'P').slice(0, 1).toUpperCase()

  return (
    <article className="pc-card">
      <div className={`pc-media${product.imageUrl ? '' : ' is-empty'}`}>
        {product.imageUrl ? (
          <a href={product.imageUrl} target="_blank" rel="noreferrer">
            <img src={product.imageUrl} alt={product.name} loading="lazy" />
          </a>
        ) : (
          <div className="pc-monogram" aria-hidden>
            <span>{initial}</span>
          </div>
        )}
      </div>

      <div className="pc-body">
        <div className="pc-meta-row">
          {product.code ? <span className="pc-code">{product.code}</span> : null}
          {product.category ? (
            <span className="pc-cat">{product.category.replace(/_/g, ' ')}</span>
          ) : null}
          {product.pageNumber ? <span className="pc-page">p. {product.pageNumber}</span> : null}
        </div>

        <h2 className="pc-title">{product.name}</h2>

        {product.specsPreview ? (
          <p className="pc-preview">{product.specsPreview}</p>
        ) : null}

        {product.description ? (
          <p className="pc-desc">{product.description}</p>
        ) : null}

        {product.specs.length ? (
          <dl className="pc-specs">
            {product.specs.map((s) => (
              <div key={s.key} className="pc-spec">
                <dt>{s.label}</dt>
                <dd>{s.value}</dd>
              </div>
            ))}
          </dl>
        ) : null}

        {product.features.length ? (
          <ul className="pc-features">
            {product.features.slice(0, 4).map((f) => (
              <li key={f}>{f}</li>
            ))}
          </ul>
        ) : null}
      </div>
    </article>
  )
}

export function PublicCataloguePage() {
  const { orgSlug = '', catalogueSlug = '' } = useParams()
  const [category, setCategory] = useState('all')
  const [query, setQuery] = useState('')

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
    return list.length ? ['all', ...list] : []
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
              <ProductCard key={p.id} product={p} />
            ))}
          </div>
        )}

        <footer className="pc-footer">
          <span>{cat.organization_name || 'Starlight Linear LED'}</span>
          <span>Digital product catalogue</span>
        </footer>
      </div>
    </div>
  )
}
