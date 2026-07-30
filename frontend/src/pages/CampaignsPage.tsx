import { useRef, useState } from 'react'
import { api, getApiBase } from '../api/client'
import { EmailPreviewFrame } from '../components/EmailPreviewFrame'

type Lead = Record<string, any>

const TEMPLATES = [
  { value: 'email_template.html', label: 'Modern Soft' },
  { value: 'email_template_minimalist.html', label: 'Minimalist' },
  { value: 'email_template_bold.html', label: 'Bold & Vibrant' },
]

export function CampaignsPage() {
  const [leads, setLeads] = useState<Lead[]>([])
  const [logs, setLogs] = useState<string[]>([])
  const [showLog, setShowLog] = useState(false)
  const [running, setRunning] = useState(false)
  const [template, setTemplate] = useState(TEMPLATES[0].value)
  const [preview, setPreview] = useState('')
  const [previewSubject, setPreviewSubject] = useState('Starlight outreach')
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
            if (evt.type === 'log') setLogs((prev) => [...prev, evt.message || String(evt)])
            if (evt.type === 'preview_html') {
              setPreview(evt.html || '')
              if (evt.subject) setPreviewSubject(evt.subject)
            }
            if (evt.type === 'status_update') {
              setLeads((prev) =>
                prev.map((l, i) => (i === evt.row_index ? { ...l, _status: evt.status } : l)),
              )
            }
            if (evt.type === 'done') setLogs((prev) => [...prev, 'Pipeline complete'])
          } catch {
            /* ignore */
          }
        }
      }
    } catch (e: any) {
      if (e.name !== 'AbortError') setLogs((prev) => [...prev, e.message])
    } finally {
      setRunning(false)
    }
  }

  const sent = leads.filter((l) => String(l._status || '').toLowerCase().includes('sent')).length

  return (
    <div>
      <div className="page-hero">
        <div>
          <h1>Campaigns</h1>
          <p>Upload leads and watch live customer-ready previews — not code.</p>
        </div>
        <div className="row">
          <button className="btn" disabled={!leads.length || running} onClick={start}>
            {running ? 'Running…' : 'Start pipeline'}
          </button>
          {running ? (
            <button className="btn secondary" onClick={() => abortRef.current?.abort()}>Stop</button>
          ) : null}
        </div>
      </div>

      <div className="stat-row">
        <div className="stat blue"><div className="label">Leads</div><div className="value">{leads.length}</div></div>
        <div className="stat cyan"><div className="label">Sent</div><div className="value">{sent}</div></div>
        <div className="stat amber"><div className="label">Status</div><div className="value" style={{ fontSize: '1.05rem', marginTop: 8 }}>{running ? 'Live' : 'Idle'}</div></div>
      </div>

      <div className="panel tint-cyan row" style={{ marginBottom: '1rem' }}>
        <input type="file" accept=".xlsx,.xls,.csv" onChange={(e) => onUpload(e.target.files?.[0])} />
        <select className="select" style={{ maxWidth: 280 }} value={template} onChange={(e) => setTemplate(e.target.value)}>
          {TEMPLATES.map((t) => (
            <option key={t.value} value={t.value}>{t.label}</option>
          ))}
        </select>
      </div>

      <div className="grid-2">
        <div className="panel stack">
          <strong style={{ fontFamily: 'var(--display)' }}>Lead queue</strong>
          <div style={{ maxHeight: 300, overflow: 'auto' }}>
            {leads.map((l, i) => (
              <div key={i} className="list-row" style={{ padding: '0.55rem 0.4rem' }}>
                <span style={{ fontWeight: 600 }}>{l.website || l.company || `Lead ${i + 1}`}</span>
                <span className="pill">{l._status || 'pending'}</span>
              </div>
            ))}
            {!leads.length ? <div className="muted">Upload Excel with a website column.</div> : null}
          </div>
          <button type="button" className="btn secondary" onClick={() => setShowLog((v) => !v)}>
            {showLog ? 'Hide technical log' : 'Show technical log'}
          </button>
          {showLog ? <pre className="terminal">{logs.join('\n') || 'No log lines yet…'}</pre> : null}
        </div>
        <div className="panel tint-blue stack">
          <strong style={{ fontFamily: 'var(--display)' }}>Live customer preview</strong>
          {preview ? (
            <EmailPreviewFrame html={preview} subject={previewSubject} />
          ) : (
            <p className="muted">Preview appears as each Starlight email is generated.</p>
          )}
        </div>
      </div>
    </div>
  )
}
