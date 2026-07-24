import { useEffect, useState } from 'react'
import { api } from '../api/client'

export function TemplatesPage() {
  const [names, setNames] = useState<string[]>([
    'email_template.html',
    'email_template_minimalist.html',
    'email_template_bold.html',
  ])
  const [selected, setSelected] = useState(names[0])
  const [content, setContent] = useState('')
  const [msg, setMsg] = useState('')

  useEffect(() => {
    api.templates().then((res: any) => {
      const list = Array.isArray(res) ? res : res.templates || []
      const mapped = list.map((t: any) => (typeof t === 'string' ? t : t.name)).filter(Boolean)
      if (mapped.length) {
        setNames(mapped)
        setSelected(mapped[0])
      }
    }).catch(() => undefined)
  }, [])

  useEffect(() => {
    if (!selected) return
    api.getTemplate(selected).then((t) => setContent(t.content || '')).catch((e) => setMsg(e.message))
  }, [selected])

  const save = async () => {
    await api.saveTemplate(selected, content)
    setMsg('Template saved')
  }

  return (
    <div className="stack" style={{ gap: '1.25rem' }}>
      <div>
        <h1 style={{ margin: 0, fontFamily: 'var(--font-display)', fontWeight: 500 }}>Templates</h1>
        <p className="muted">Edit HTML email templates and preview branding.</p>
      </div>
      <div className="panel stack">
        <select className="select" value={selected} onChange={(e) => setSelected(e.target.value)}>
          {names.map((n) => <option key={n} value={n}>{n}</option>)}
        </select>
        <textarea className="textarea" rows={18} value={content} onChange={(e) => setContent(e.target.value)} />
        <div className="row">
          <button className="btn" onClick={save}>Save</button>
          {msg ? <span className="muted">{msg}</span> : null}
        </div>
        <iframe title="tpl" sandbox="" srcDoc={content} style={{ width: '100%', minHeight: 280, borderRadius: 12, border: '1px solid var(--border)', background: 'white' }} />
      </div>
    </div>
  )
}
