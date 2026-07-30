import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'
import { api } from '../api/client'

export function KnowledgePage() {
  const kb = useQuery({ queryKey: ['kb'], queryFn: api.kbStatus })
  const [query, setQuery] = useState('boutique hotel lobby linear LED lighting')
  const [result, setResult] = useState<any>(null)
  const [msg, setMsg] = useState('')

  const upload = async (files?: FileList | null) => {
    if (!files?.length) return
    setMsg('Ingesting catalogues…')
    try {
      await api.uploadCatalogues(files)
      setMsg('Catalogue ready')
      await kb.refetch()
    } catch (e: any) {
      setMsg(e.message)
    }
  }

  const runRag = async () => {
    setMsg('Running HyDE…')
    try {
      const res = await api.ragQuery(query)
      setResult(res)
      setMsg('Grounding check complete')
    } catch (e: any) {
      setMsg(e.message)
    }
  }

  return (
    <div>
      <div className="page-hero">
        <div>
          <h1>Knowledge</h1>
          <p>Starlight product catalogues power grounded emails — upload PDFs and validate retrieval.</p>
        </div>
        <button className="btn danger" onClick={async () => { await api.clearKb(); kb.refetch() }}>Clear KB</button>
      </div>

      <div className="stat-row">
        <div className="stat blue"><div className="label">Chunks</div><div className="value">{kb.data?.chunk_count ?? '—'}</div></div>
        <div className="stat cyan"><div className="label">Catalogues</div><div className="value">{(kb.data?.catalogues || []).length}</div></div>
        <div className="stat amber"><div className="label">Status</div><div className="value" style={{ fontSize: '1rem', marginTop: 8 }}>{msg || 'Idle'}</div></div>
      </div>

      <div className="panel tint-cyan stack" style={{ marginBottom: '1rem' }}>
        <strong style={{ fontFamily: 'var(--display)' }}>Upload catalogues</strong>
        <input type="file" multiple accept=".pdf,.txt,.docx" onChange={(e) => upload(e.target.files)} />
        <div className="muted">{(kb.data?.catalogues || []).join(' · ') || 'No catalogues yet'}</div>
      </div>

      <div className="panel stack">
        <strong style={{ fontFamily: 'var(--display)' }}>RAG playground</strong>
        <textarea className="textarea" rows={3} value={query} onChange={(e) => setQuery(e.target.value)} />
        <button className="btn" onClick={runRag}>Run HyDE query</button>
        {result ? (
          <pre style={{ whiteSpace: 'pre-wrap', fontSize: 12, background: '#0f172a', color: '#e2e8f0', padding: 14, borderRadius: 14, overflow: 'auto', maxHeight: 360 }}>
            {JSON.stringify(result, null, 2)}
          </pre>
        ) : null}
      </div>
    </div>
  )
}
