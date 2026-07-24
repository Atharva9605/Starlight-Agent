import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'
import { api } from '../api/client'

export function KnowledgePage() {
  const kb = useQuery({ queryKey: ['kb'], queryFn: api.kbStatus })
  const [query, setQuery] = useState('boutique hotel lobby lighting')
  const [result, setResult] = useState<any>(null)
  const [msg, setMsg] = useState('')

  const upload = async (files?: FileList | null) => {
    if (!files?.length) return
    setMsg('Uploading…')
    try {
      await api.uploadCatalogues(files)
      setMsg('Catalogue ingested')
      await kb.refetch()
    } catch (e: any) {
      setMsg(e.message)
    }
  }

  const runRag = async () => {
    setMsg('Querying…')
    try {
      const res = await api.ragQuery(query)
      setResult(res)
      setMsg('Done')
    } catch (e: any) {
      setMsg(e.message)
    }
  }

  return (
    <div className="stack" style={{ gap: '1.25rem' }}>
      <div>
        <h1 style={{ margin: 0, fontFamily: 'var(--font-display)', fontWeight: 500 }}>Knowledge</h1>
        <p className="muted">Catalogue ingest + RAG playground for grounding checks.</p>
      </div>

      <div className="panel stack">
        <div className="row" style={{ justifyContent: 'space-between' }}>
          <div>
            <div><strong>{kb.data?.chunk_count ?? '—'} chunks</strong></div>
            <div className="muted">{(kb.data?.catalogues || []).join(', ') || 'No catalogues'}</div>
          </div>
          <div className="row">
            <input type="file" multiple accept=".pdf,.txt,.docx" onChange={(e) => upload(e.target.files)} />
            <button className="btn danger" onClick={async () => { await api.clearKb(); kb.refetch() }}>Clear KB</button>
          </div>
        </div>
        {msg ? <div className="muted">{msg}</div> : null}
      </div>

      <div className="panel stack">
        <strong>RAG playground</strong>
        <textarea className="textarea" rows={3} value={query} onChange={(e) => setQuery(e.target.value)} />
        <button className="btn" onClick={runRag}>Run HyDE query</button>
        {result ? (
          <pre style={{ whiteSpace: 'pre-wrap', fontSize: 12, background: '#020617', padding: 12, borderRadius: 12, overflow: 'auto', maxHeight: 360 }}>
            {JSON.stringify(result, null, 2)}
          </pre>
        ) : null}
      </div>
    </div>
  )
}
