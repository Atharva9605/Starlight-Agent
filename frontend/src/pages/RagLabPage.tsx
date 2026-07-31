import { useState } from 'react'
import { api } from '../api/client'

export function RagLabPage() {
  const [query, setQuery] = useState('boutique hotel lobby linear LED lighting')
  const [result, setResult] = useState<any>(null)
  const [msg, setMsg] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [showRaw, setShowRaw] = useState(false)

  const runRag = async () => {
    setBusy(true)
    setMsg('Running HyDE…')
    setError('')
    try {
      const res = await api.ragQuery(query)
      setResult(res)
      setMsg('Done')
    } catch (e: any) {
      setError(e.message || 'Query failed')
      setMsg('')
    } finally {
      setBusy(false)
    }
  }

  const chunks: any[] = result?.chunks || []

  return (
    <div className="studio-screen">
      <div className="page-hero">
        <div>
          <h1>RAG Lab</h1>
          <p>Admin grounding check — readable chunks, not a JSON wall.</p>
        </div>
        {msg ? <span className="pill ok">{msg}</span> : null}
      </div>

      <div className="studio-body">
        <aside className="studio-nav panel stack">
          <div className="studio-nav-label muted">Query</div>
          <label className="field">
            <span>HyDE prompt</span>
            <textarea
              className="textarea"
              rows={5}
              value={query}
              disabled={busy}
              onChange={(e) => setQuery(e.target.value)}
            />
          </label>
          <button className="btn" type="button" disabled={busy || !query.trim()} onClick={runRag}>
            {busy ? 'Running…' : 'Run HyDE query'}
          </button>
          {result?.empty_rag ? (
            <span className="pill warn">Empty / weak RAG — drafts should avoid product claims</span>
          ) : null}
          {error ? (
            <div className="alert danger">
              <strong>Query failed</strong>
              <div style={{ marginTop: 4 }}>{error}</div>
            </div>
          ) : null}
        </aside>

        <div className="studio-editor panel stack">
          {!result ? (
            <div className="empty-state" style={{ margin: 'auto', textAlign: 'center' }}>
              <strong style={{ fontFamily: 'var(--display)' }}>No results yet</strong>
              <p className="muted" style={{ margin: '0.5rem 0 0' }}>
                Run a HyDE query to inspect grounding against indexed catalogues.
              </p>
            </div>
          ) : (
            <>
              <div className="studio-editor-head">
                <div>
                  <h2>Results</h2>
                  <p>{chunks.length} chunk{chunks.length === 1 ? '' : 's'} returned</p>
                </div>
                <button type="button" className="btn secondary" onClick={() => setShowRaw((v) => !v)}>
                  {showRaw ? 'Hide raw' : 'View raw'}
                </button>
              </div>

              <div className="panel tint-cyan">
                <strong style={{ fontFamily: 'var(--display)' }}>HyDE expansion</strong>
                <p style={{ whiteSpace: 'pre-wrap', margin: '0.5rem 0 0' }}>{result.hyde_doc || '—'}</p>
              </div>

              {chunks.length === 0 ? (
                <div className="muted">No chunks returned.</div>
              ) : (
                <div className="lab-chunks">
                  {chunks.map((c: any, i: number) => (
                    <div key={i} className="chunk-card">
                      <div className="row" style={{ justifyContent: 'space-between' }}>
                        <strong>
                          #{c.rank || i + 1}{' '}
                          {(c.metadata || {}).catalogue_name || (c.metadata || {}).product_name || 'Chunk'}
                        </strong>
                        {c.distance != null ? (
                          <span className="dist">distance {Number(c.distance).toFixed(3)}</span>
                        ) : null}
                      </div>
                      <p className="muted" style={{ margin: '0.55rem 0 0', fontSize: 14, color: 'var(--text)' }}>
                        {(c.document || '').slice(0, 420)}
                        {(c.document || '').length > 420 ? '…' : ''}
                      </p>
                    </div>
                  ))}
                </div>
              )}

              {showRaw ? (
                <pre className="terminal" style={{ maxHeight: 280 }}>
                  {JSON.stringify(result, null, 2)}
                </pre>
              ) : null}
            </>
          )}
        </div>
      </div>
    </div>
  )
}
