import { useRef, useState } from 'react'
import { api, getApiBase } from '../api/client'

type Lead = Record<string, any>

export function CampaignsPage() {
  const [leads, setLeads] = useState<Lead[]>([])
  const [logs, setLogs] = useState<string[]>([])
  const [running, setRunning] = useState(false)
  const [template, setTemplate] = useState('email_template.html')
  const [preview, setPreview] = useState('')
  const abortRef = useRef<AbortController | null>(null)

  const onUpload = async (file?: File) => {
    if (!file) return
    const res = await api.uploadLeads(file)
    setLeads(res.leads || [])
  }

  const start = async () => {
    setRunning(true)
    setLogs([])
    const ctrl = new AbortController()
    abortRef.current = ctrl
    const token = localStorage.getItem('token')
    try {
      const res = await fetch(`${getApiBase()}/api/process-stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        body: JSON.stringify({
          leads,
          template,
          delay: 2,
          sender_email: '',
          recipient_override: '',
        }),
        signal: ctrl.signal,
      })
      if (!res.ok || !res.body) throw new Error(await res.text())
      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const chunks = buffer.split('\n\n')
        buffer = chunks.pop() || ''
        for (const chunk of chunks) {
          const line = chunk.split('\n').find((l) => l.startsWith('data:'))
          if (!line) continue
          try {
            const evt = JSON.parse(line.slice(5).trim())
            if (evt.type === 'log') setLogs((prev) => [...prev, evt.message || JSON.stringify(evt)])
            if (evt.type === 'preview_html') setPreview(evt.html || '')
            if (evt.type === 'status_update') {
              setLeads((prev) =>
                prev.map((l, i) => (i === evt.row_index ? { ...l, _status: evt.status } : l)),
              )
            }
            if (evt.type === 'done') setLogs((prev) => [...prev, 'Pipeline complete'])
          } catch {
            /* ignore parse errors */
          }
        }
      }
    } catch (e: any) {
      if (e.name !== 'AbortError') setLogs((prev) => [...prev, e.message])
    } finally {
      setRunning(false)
    }
  }

  return (
    <div className="stack" style={{ gap: '1.25rem' }}>
      <div>
        <h1 style={{ margin: 0, fontFamily: 'var(--font-display)', fontWeight: 500 }}>Campaigns</h1>
        <p className="muted">Upload leads, stream the outbound AI pipeline, preview emails live.</p>
      </div>

      <div className="panel row">
        <input type="file" accept=".xlsx,.xls,.csv" onChange={(e) => onUpload(e.target.files?.[0])} />
        <select className="select" style={{ maxWidth: 260 }} value={template} onChange={(e) => setTemplate(e.target.value)}>
          <option value="email_template.html">Modern Soft</option>
          <option value="email_template_minimalist.html">Minimalist</option>
          <option value="email_template_bold.html">Bold</option>
        </select>
        <button className="btn" disabled={!leads.length || running} onClick={start}>
          {running ? 'Running…' : 'Start pipeline'}
        </button>
        {running ? (
          <button className="btn secondary" onClick={() => abortRef.current?.abort()}>Stop</button>
        ) : null}
      </div>

      <div className="grid-2">
        <div className="panel stack">
          <strong>Leads ({leads.length})</strong>
          <div style={{ maxHeight: 320, overflow: 'auto' }}>
            {leads.map((l, i) => (
              <div key={i} className="row" style={{ justifyContent: 'space-between', borderBottom: '1px solid var(--border)', padding: '0.45rem 0' }}>
                <span>{l.website || l.company || `Lead ${i + 1}`}</span>
                <span className="muted">{l._status || 'pending'}</span>
              </div>
            ))}
          </div>
          <pre style={{ background: '#020617', color: '#86efac', padding: 12, borderRadius: 12, maxHeight: 220, overflow: 'auto', fontSize: 12 }}>
            {logs.join('\n') || 'Logs appear here…'}
          </pre>
        </div>
        <div className="panel">
          <strong>Live preview</strong>
          {preview ? (
            <iframe title="email" sandbox="" srcDoc={preview} style={{ width: '100%', minHeight: 420, border: 0, borderRadius: 12, background: 'white', marginTop: 12 }} />
          ) : (
            <p className="muted">Preview HTML streams in as each email is generated.</p>
          )}
        </div>
      </div>
    </div>
  )
}
