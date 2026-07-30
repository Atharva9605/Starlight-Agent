import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'
import { api } from '../api/client'
import { Link } from 'react-router-dom'

/** Sales-facing catalogues page — no JSON / RAG dump. */
export function CataloguesPage() {
  const kb = useQuery({ queryKey: ['kb'], queryFn: api.kbStatus })
  const [msg, setMsg] = useState('')

  const upload = async (files?: FileList | null) => {
    if (!files?.length) return
    setMsg('Ingesting…')
    try {
      await api.uploadCatalogues(files)
      setMsg('Catalogue ready')
      await kb.refetch()
    } catch (e: any) {
      setMsg(e.message)
    }
  }

  const catalogues = kb.data?.catalogues || []

  return (
    <div>
      <div className="page-hero">
        <div>
          <h1>Catalogues</h1>
          <p>Upload Starlight product PDFs that ground every outbound email.</p>
        </div>
        <button className="btn danger" onClick={async () => { await api.clearKb(); kb.refetch() }}>
          Clear all
        </button>
      </div>

      <div className="stat-row">
        <div className="stat blue"><div className="label">Chunks</div><div className="value">{kb.data?.chunk_count ?? '—'}</div></div>
        <div className="stat cyan"><div className="label">Catalogues</div><div className="value">{catalogues.length}</div></div>
        <div className="stat amber"><div className="label">Status</div><div className="value" style={{ fontSize: '1rem', marginTop: 8 }}>{msg || 'Ready'}</div></div>
      </div>

      <div className="panel tint-cyan stack" style={{ marginBottom: '1rem' }}>
        <strong style={{ fontFamily: 'var(--display)' }}>Upload catalogues</strong>
        <input type="file" multiple accept=".pdf,.txt,.docx" onChange={(e) => upload(e.target.files)} />
        {!catalogues.length ? (
          <div className="muted">No catalogues yet — upload a PDF to get started.</div>
        ) : (
          <div className="stack">
            {catalogues.map((name: string) => (
              <div key={name} className="list-row">
                <span style={{ fontWeight: 700 }}>{name}</span>
                <span className="pill ok">Indexed</span>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="panel">
        <p className="muted" style={{ margin: 0 }}>
          Need to test retrieval quality? Open{' '}
          <Link to="/admin/rag" style={{ color: 'var(--blue)', fontWeight: 700 }}>Admin → RAG Lab</Link>.
        </p>
      </div>
    </div>
  )
}
