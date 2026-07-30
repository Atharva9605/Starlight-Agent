import { useState } from 'react'
import { api } from '../api/client'

export function RagLabPage() {
  const [query, setQuery] = useState('boutique hotel lobby linear LED lighting')
  const [result, setResult] = useState<any>(null)
  const [msg, setMsg] = useState('')
  const [showRaw, setShowRaw] = useState(false)

  const runRag = async () => {
    setMsg('Running HyDE…')
    try {
      const res = await api.ragQuery(query)
      setResult(res)
      setMsg('Done')
    } catch (e: any) {
      setMsg(e.message)
    }
  }

  const chunks: any[] = result?.chunks || []

  return (
    <div>
      <div className="page-hero">
        <div>
          <h1>RAG Lab</h1>
          <p>Admin grounding check — readable chunks, not a JSON wall.</p>
        </div>
        {msg ? <span className="pill ok">{msg}</span> : null}
      </div>

      <div className="panel stack" style={{ marginBottom: '1rem' }}>
        <textarea className="textarea" rows={3} value={query} onChange={(e) => setQuery(e.target.value)} />
        <button className="btn" onClick={runRag}>Run HyDE query</button>
        {result?.empty_rag ? <span className="pill warn">Empty / weak RAG — drafts should avoid product claims</span> : null}
      </div>

      {result ? (
        <div className="stack" style={{ marginBottom: '1rem' }}>
          <div className="panel tint-cyan">
            <strong style={{ fontFamily: 'var(--display)' }}>HyDE expansion</strong>
            <p style={{ whiteSpace: 'pre-wrap', margin: '0.5rem 0 0' }}>{result.hyde_doc || '—'}</p>
          </div>
          {chunks.length === 0 ? (
            <div className="panel muted">No chunks returned.</div>
          ) : (
            chunks.map((c: any, i: number) => (
              <div key={i} className="chunk-card">
                <div className="row" style={{ justifyContent: 'space-between' }}>
                  <strong>
                    #{c.rank || i + 1}{' '}
                    {(c.metadata || {}).catalogue_name || (c.metadata || {}).product_name || 'Chunk'}
                  </strong>
                  {c.distance != null ? <span className="dist">distance {Number(c.distance).toFixed(3)}</span> : null}
                </div>
                <p style={{ margin: '0.55rem 0 0', fontSize: 14, color: '#334155' }}>
                  {(c.document || '').slice(0, 420)}
                  {(c.document || '').length > 420 ? '…' : ''}
                </p>
              </div>
            ))
          )}
          <button type="button" className="btn secondary" onClick={() => setShowRaw((v) => !v)}>
            {showRaw ? 'Hide raw' : 'View raw'}
          </button>
          {showRaw ? (
            <pre className="terminal" style={{ maxHeight: 320 }}>
              {JSON.stringify(result, null, 2)}
            </pre>
          ) : null}
        </div>
      ) : null}
    </div>
  )
}
