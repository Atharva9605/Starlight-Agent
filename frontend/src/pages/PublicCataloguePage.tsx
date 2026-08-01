import { useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useParams } from 'react-router-dom'
import { api } from '../api/client'

export function PublicCataloguePage() {
  const { orgSlug = '', catalogueSlug = '' } = useParams()
  const [category, setCategory] = useState('all')

  const q = useQuery({
    queryKey: ['public-catalogue', orgSlug, catalogueSlug],
    queryFn: () => api.publicCatalogue(orgSlug, catalogueSlug),
    enabled: Boolean(orgSlug && catalogueSlug),
    retry: false,
  })

  const categories = useMemo(() => {
    const set = new Set<string>()
    for (const p of q.data?.products || []) {
      if (p.category) set.add(p.category)
    }
    return ['all', ...Array.from(set).sort()]
  }, [q.data?.products])

  const products = useMemo(() => {
    const list = q.data?.products || []
    if (category === 'all') return list
    return list.filter((p) => p.category === category)
  }, [q.data?.products, category])

  if (q.isLoading) {
    return (
      <div className="public-catalogue">
        <div className="public-catalogue-inner">
          <p className="muted">Loading catalogue…</p>
        </div>
      </div>
    )
  }

  if (q.isError || !q.data) {
    return (
      <div className="public-catalogue">
        <div className="public-catalogue-inner">
          <h1>Catalogue unavailable</h1>
          <p className="muted">{(q.error as Error)?.message || 'Not found'}</p>
        </div>
      </div>
    )
  }

  const cat = q.data

  return (
    <div className="public-catalogue">
      <header className="public-catalogue-hero">
        {cat.cover_image_url ? (
          <img className="public-catalogue-cover" src={cat.cover_image_url} alt="" />
        ) : null}
        <div className="public-catalogue-hero-copy">
          <p className="public-catalogue-brand">{cat.organization_name || 'Starlight'}</p>
          <h1>{cat.name}</h1>
          <p className="muted">
            {cat.product_count ?? products.length} products
            {cat.page_count ? ` · ${cat.page_count} pages` : ''}
          </p>
        </div>
      </header>

      <div className="public-catalogue-inner">
        <div className="public-catalogue-filters">
          {categories.map((c) => (
            <button
              key={c}
              type="button"
              className={`pill ${category === c ? 'ok' : ''}`}
              onClick={() => setCategory(c)}
            >
              {c === 'all' ? 'All' : c.replace(/_/g, ' ')}
            </button>
          ))}
        </div>

        {!products.length ? (
          <div className="empty-state">
            <strong>No products in this category</strong>
          </div>
        ) : (
          <div className="public-product-grid">
            {products.map((p) => (
              <article key={p.id} className="public-product-card">
                {p.image_url ? (
                  <a href={p.image_url} target="_blank" rel="noreferrer">
                    <img src={p.image_url} alt={p.product_name} loading="lazy" />
                  </a>
                ) : (
                  <div className="public-product-placeholder" />
                )}
                <div className="public-product-body">
                  <span className="pill">{(p.category || 'other').replace(/_/g, ' ')}</span>
                  <h2>{p.product_name}</h2>
                  {p.specs_preview ? <p className="ref-specs">{p.specs_preview}</p> : null}
                  {p.description ? <p className="muted">{p.description}</p> : null}
                  {p.page_number ? (
                    <p className="muted" style={{ fontSize: 12 }}>Page {p.page_number}</p>
                  ) : null}
                </div>
              </article>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
