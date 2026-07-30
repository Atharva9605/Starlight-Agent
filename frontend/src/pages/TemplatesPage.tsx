import { useEffect, useState } from 'react'
import { api } from '../api/client'
import { EmailPreviewFrame } from '../components/EmailPreviewFrame'

export function TemplatesPage() {
  const [names, setNames] = useState<string[]>([
    'email_template.html',
    'email_template_minimalist.html',
    'email_template_bold.html',
  ])
  const [selected, setSelected] = useState(names[0])
  const [content, setContent] = useState('')
  const [msg, setMsg] = useState('')
  const [showHtml, setShowHtml] = useState(false)

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

  const label = selected.replace('email_template_', '').replace('.html', '').replace(/_/g, ' ')

  return (
    <div>
      <div className="page-hero">
        <div>
          <h1>Email Design</h1>
          <p>Preview-first Starlight templates. HTML is advanced-only.</p>
        </div>
        <button className="btn" onClick={save}>Save template</button>
      </div>

      <div className="panel stack" style={{ marginBottom: '1rem' }}>
        <select className="select" value={selected} onChange={(e) => setSelected(e.target.value)}>
          {names.map((n) => (
            <option key={n} value={n}>
              {n.includes('minimalist') ? 'Minimalist' : n.includes('bold') ? 'Bold & Vibrant' : 'Modern Soft'} ({n})
            </option>
          ))}
        </select>
        {msg ? <span className="pill ok">{msg}</span> : null}
      </div>

      <div className="grid-2">
        <div className="panel tint-blue stack">
          <strong style={{ fontFamily: 'var(--display)' }}>Customer preview — {label}</strong>
          <EmailPreviewFrame html={content} subject="Sample Starlight email" />
        </div>
        <div className="panel stack">
          <button type="button" className="btn secondary" onClick={() => setShowHtml((v) => !v)}>
            {showHtml ? 'Hide HTML source' : 'Advanced: Edit HTML'}
          </button>
          {showHtml ? (
            <div className="advanced-box stack">
              <p className="muted" style={{ margin: 0, fontSize: 13 }}>
                For designers / ops only. Sales users never see this on Inbox.
              </p>
              <textarea className="textarea" rows={18} value={content} onChange={(e) => setContent(e.target.value)} />
            </div>
          ) : (
            <div className="muted">
              Preview updates as you edit HTML in Advanced. Keep Jinja variables like{' '}
              <code>{'{{ subject }}'}</code> intact.
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
