import { useState } from 'react'
import {
  BeakerRegular,
  CodeRegular,
  DocumentSearchRegular,
  PlayRegular,
  SparkleRegular,
} from '@fluentui/react-icons'
import { api } from '../api/client'
import { CardHeader, EmptyState, MessageBar, PageHeader, Spinner } from '../components/ui'

const SAMPLE_QUERIES = [
  'boutique hotel lobby linear LED lighting',
  'office open-plan suspended profiles',
  'retail store shelf lighting',
]

/** Map cosine distance (0 = identical) to a 0–100 relevance bar. */
function relevance(distance: number) {
  return Math.max(0, Math.min(100, Math.round((1 - distance) * 100)))
}

export function RagLabPage() {
  const [query, setQuery] = useState(SAMPLE_QUERIES[0])
  const [result, setResult] = useState<any>(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [showRaw, setShowRaw] = useState(false)

  const runRag = async () => {
    if (!query.trim()) return
    setBusy(true)
    setError('')
    try {
      const res = await api.ragQuery(query)
      setResult(res)
    } catch (e: any) {
      setError(e.message || 'Query failed')
    } finally {
      setBusy(false)
    }
  }

  const chunks: any[] = result?.chunks || []

  return (
    <div>
      <PageHeader
        breadcrumb={[{ label: 'Admin' }, { label: 'RAG lab' }]}
        title="RAG lab"
        subtitle="Check what the catalogues return for a client description before it reaches an email."
      />

      <div className="studio">
        <section className="card studio-nav" style={{ padding: 16 }}>
          <CardHeader icon={<BeakerRegular />} title="Query" />
          <div className="stack" style={{ marginTop: 12 }}>
            <label className="field">
              <span>Client description</span>
              <textarea
                className="textarea"
                rows={5}
                value={query}
                disabled={busy}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => {
                  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') void runRag()
                }}
              />
              <span className="field-hint">Expanded with HyDE, then matched against indexed chunks.</span>
            </label>
            <div className="suggestions">
              {SAMPLE_QUERIES.map((s) => (
                <button key={s} type="button" className="suggestion" onClick={() => setQuery(s)} disabled={busy}>
                  {s}
                </button>
              ))}
            </div>
            <button className="btn" type="button" disabled={busy || !query.trim()} onClick={runRag}>
              {busy ? <Spinner size="sm" /> : <PlayRegular />}
              {busy ? 'Running…' : 'Run query'}
            </button>
            {error ? (
              <MessageBar intent="error" title="Query failed" onDismiss={() => setError('')}>
                {error}
              </MessageBar>
            ) : null}
          </div>
        </section>

        <section className="card studio-editor">
          {!result ? (
            <EmptyState
              icon={<DocumentSearchRegular />}
              title={busy ? 'Searching catalogues…' : 'No results yet'}
              description="Run a query to see the HyDE expansion and the catalogue chunks that ground the email."
            />
          ) : (
            <>
              <CardHeader
                title="Results"
                subtitle={`${chunks.length} chunk${chunks.length === 1 ? '' : 's'} returned`}
                actions={
                  <button type="button" className="btn subtle sm" onClick={() => setShowRaw((v) => !v)}>
                    <CodeRegular /> {showRaw ? 'Hide raw JSON' : 'View raw JSON'}
                  </button>
                }
              />

              {result.empty_rag ? (
                <MessageBar intent="warning" title="Weak grounding">
                  Few or no relevant chunks — drafts for this client should avoid specific product claims.
                </MessageBar>
              ) : null}

              <div className="hyde-box">
                <span className="ai-label"><SparkleRegular /> HyDE expansion</span>
                <p>{result.hyde_doc || '—'}</p>
              </div>

              {chunks.length === 0 ? (
                <EmptyState compact title="No chunks returned" description="Upload catalogues or try a broader description." />
              ) : (
                <div className="stack">
                  {chunks.map((c: any, i: number) => {
                    const meta = c.metadata || {}
                    const doc = String(c.document || '')
                    return (
                      <article key={i} className="chunk-card">
                        <div className="chunk-head">
                          <span className="chunk-rank">{c.rank || i + 1}</span>
                          <strong className="grow truncate">
                            {meta.product_name || meta.catalogue_name || 'Chunk'}
                            {meta.product_name && meta.catalogue_name ? (
                              <span className="muted" style={{ fontWeight: 400 }}> · {meta.catalogue_name}</span>
                            ) : null}
                          </strong>
                          {c.distance != null ? (
                            <span className="chunk-score" title={`Distance ${Number(c.distance).toFixed(3)}`}>
                              <span className="progress">
                                <span
                                  className="progress-fill"
                                  style={{ display: 'block', width: `${relevance(Number(c.distance))}%` }}
                                />
                              </span>
                              {Number(c.distance).toFixed(3)}
                            </span>
                          ) : null}
                        </div>
                        <p className="chunk-text">
                          {doc.slice(0, 420)}
                          {doc.length > 420 ? '…' : ''}
                        </p>
                      </article>
                    )
                  })}
                </div>
              )}

              {showRaw ? <pre className="terminal" style={{ maxHeight: 320 }}>{JSON.stringify(result, null, 2)}</pre> : null}
            </>
          )}
        </section>
      </div>
    </div>
  )
}
