import { useCallback, useEffect, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { api } from '../api/client'
import { EmailPreviewFrame } from '../components/EmailPreviewFrame'

type TemplateMeta = {
  name: string
  label: string
  builtin?: boolean
  is_custom?: boolean
}

function slugify(label: string): string {
  const base = label
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_|_$/g, '')
    .slice(0, 40)
  return base ? `email_template_${base}.html` : `email_template_custom_${Date.now()}.html`
}

function displayLabel(t: TemplateMeta): string {
  if (t.label && t.label !== t.name) return t.label
  if (t.name.includes('minimalist')) return 'Minimalist'
  if (t.name.includes('bold')) return 'Bold & Vibrant'
  if (t.name === 'email_template.html') return 'Modern Soft'
  return t.name.replace(/^email_template_?/, '').replace(/\.html$/, '').replace(/_/g, ' ') || t.name
}

/** Admin page — pick, preview, and save email HTML templates. */
export function TemplatesPage() {
  const [searchParams] = useSearchParams()
  const preferSelected = searchParams.get('selected') || ''

  const [templates, setTemplates] = useState<TemplateMeta[]>([])
  const [selected, setSelected] = useState('')
  const [content, setContent] = useState('')
  const [previewHtml, setPreviewHtml] = useState('')
  const [msg, setMsg] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [previewing, setPreviewing] = useState(false)
  const [saveName, setSaveName] = useState('')
  const [saveLabel, setSaveLabel] = useState('')

  const refreshList = useCallback(async (prefer?: string) => {
    const list = await api.templates()
    setTemplates(list)
    const want = prefer || preferSelected
    const next = want && list.some((t) => t.name === want) ? want : list[0]?.name || ''
    setSelected((prev) => (prefer || preferSelected ? next : prev || next))
    return list
  }, [preferSelected])

  useEffect(() => {
    refreshList().catch((e) => setError(e.message || 'Could not load templates'))
  }, [refreshList])

  useEffect(() => {
    if (!selected) return
    let cancelled = false
    setError('')
    api
      .getTemplate(selected)
      .then((t) => {
        if (cancelled) return
        setContent(t.content || '')
        const meta = templates.find((x) => x.name === selected)
        setSaveName(selected)
        setSaveLabel(meta ? displayLabel(meta) : selected)
      })
      .catch((e) => {
        if (!cancelled) setError(e.message)
      })
    return () => {
      cancelled = true
    }
  }, [selected, templates])

  useEffect(() => {
    if (!content.trim()) {
      setPreviewHtml('')
      return
    }
    let cancelled = false
    const timer = window.setTimeout(async () => {
      setPreviewing(true)
      try {
        const res = await api.previewTemplate({ template_content: content })
        if (!cancelled) setPreviewHtml(res.html || '')
      } catch {
        if (!cancelled) setPreviewHtml(content)
      } finally {
        if (!cancelled) setPreviewing(false)
      }
    }, 450)
    return () => {
      cancelled = true
      window.clearTimeout(timer)
    }
  }, [content])

  const save = async () => {
    const name = (saveName.trim() || selected || slugify(saveLabel || 'custom')).replace(/\s+/g, '_')
    const finalName = name.endsWith('.html') ? name : `${name}.html`
    if (!content.trim()) {
      setError('Nothing to save yet.')
      return
    }
    setBusy(true)
    setError('')
    try {
      const res = await api.saveTemplate(finalName, content, saveLabel.trim() || undefined)
      await refreshList(res.name || finalName)
      setSelected(res.name || finalName)
      setMsg(`Saved as ${res.label || finalName}`)
    } catch (e: any) {
      setError(e.message || 'Save failed')
    } finally {
      setBusy(false)
    }
  }

  const remove = async () => {
    const meta = templates.find((t) => t.name === selected)
    if (!meta?.is_custom) {
      setError('Built-in templates cannot be deleted. Save a custom copy first.')
      return
    }
    if (!window.confirm(`Delete custom template “${displayLabel(meta)}”?`)) return
    setBusy(true)
    try {
      await api.deleteTemplate(selected)
      await refreshList()
      setMsg('Template deleted')
    } catch (e: any) {
      setError(e.message || 'Delete failed')
    } finally {
      setBusy(false)
    }
  }

  const current = templates.find((t) => t.name === selected)

  return (
    <div className="design-screen">
      <div className="page-hero">
        <div>
          <h1>Email Design</h1>
          <p>Preview how customers see the email — pick a saved template or create a new one with AI.</p>
        </div>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          <Link to="/admin/templates/create" className="btn secondary">
            Create with AI
          </Link>
          <button className="btn" type="button" disabled={busy} onClick={save}>
            Save template
          </button>
        </div>
      </div>

      <div className="design-body">
        <div className="design-mail">
          <div className="design-preview-label muted">
            Customer preview{previewing ? ' · refreshing…' : ''}
          </div>
          <EmailPreviewFrame
            html={previewHtml || content}
            subject="Sample Starlight email"
            fullscreen
          />
        </div>

        <aside className="design-side panel stack">
          <strong style={{ fontFamily: 'var(--display)' }}>Template</strong>
          <label className="field">
            <span>Choose design</span>
            <select
              className="select"
              value={selected}
              disabled={busy}
              onChange={(e) => setSelected(e.target.value)}
            >
              {templates.map((t) => (
                <option key={t.name} value={t.name}>
                  {displayLabel(t)}
                  {t.is_custom ? ' · custom' : ''}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            <span>Display label</span>
            <input
              className="input"
              value={saveLabel}
              disabled={busy}
              onChange={(e) => setSaveLabel(e.target.value)}
              placeholder="My design"
            />
          </label>
          <label className="field">
            <span>Save as filename</span>
            <input
              className="input"
              value={saveName}
              disabled={busy}
              onChange={(e) => setSaveName(e.target.value)}
              placeholder="email_template_my_design.html"
            />
          </label>
          {msg ? <span className="pill ok">{msg}</span> : null}
          {error ? (
            <div className="alert danger">
              <strong>Template error</strong>
              <div style={{ marginTop: 4 }}>{error}</div>
            </div>
          ) : null}
          {current?.is_custom ? (
            <button className="btn danger" type="button" disabled={busy} onClick={remove}>
              Delete custom
            </button>
          ) : (
            <p className="muted" style={{ margin: 0, fontSize: 13 }}>
              Built-in templates stay read-only — save under a new name to keep edits.
            </p>
          )}
        </aside>
      </div>
    </div>
  )
}
