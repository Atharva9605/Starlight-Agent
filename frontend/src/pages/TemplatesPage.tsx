import { useCallback, useEffect, useState } from 'react'
import { api } from '../api/client'
import { EmailPreviewFrame } from '../components/EmailPreviewFrame'

type TemplateMeta = {
  name: string
  label: string
  builtin?: boolean
  is_custom?: boolean
}

const STYLES = [
  { value: 'modern', label: 'Modern Soft' },
  { value: 'minimal', label: 'Minimalist' },
  { value: 'bold', label: 'Bold & Vibrant' },
]

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

/** Admin page — pick, preview, AI-create, and save email HTML templates. */
export function TemplatesPage() {
  const [templates, setTemplates] = useState<TemplateMeta[]>([])
  const [selected, setSelected] = useState('')
  const [content, setContent] = useState('')
  const [previewHtml, setPreviewHtml] = useState('')
  const [msg, setMsg] = useState('')
  const [error, setError] = useState('')
  const [showAi, setShowAi] = useState(false)
  const [busy, setBusy] = useState(false)
  const [previewing, setPreviewing] = useState(false)

  const [instructions, setInstructions] = useState(
    'Clean Starlight LED outreach email with logo header, short personalized intro, feature highlights, use cases, and a clear CTA. Keep it mobile-friendly.',
  )
  const [style, setStyle] = useState('modern')
  const [useReference, setUseReference] = useState(true)
  const [saveName, setSaveName] = useState('')
  const [saveLabel, setSaveLabel] = useState('')

  const refreshList = useCallback(async (prefer?: string) => {
    const list = await api.templates()
    setTemplates(list)
    const next = prefer && list.some((t) => t.name === prefer) ? prefer : list[0]?.name || ''
    setSelected((prev) => (prefer ? next : prev || next))
    return list
  }, [])

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

  const generate = async () => {
    if (!instructions.trim()) {
      setError('Describe the template you want.')
      return
    }
    setBusy(true)
    setError('')
    setMsg('Generating with AI…')
    try {
      const res = await api.generateTemplate({
        instructions: instructions.trim(),
        style,
        reference_template: useReference && selected ? selected : null,
      })
      setContent(res.content)
      setShowAi(false)
      const stamp = new Date().toISOString().slice(0, 10).replace(/-/g, '')
      setSaveName(`email_template_ai_${stamp}.html`)
      setSaveLabel(`AI ${STYLES.find((s) => s.value === style)?.label || 'Custom'}`)
      setMsg('Draft ready — review the preview, then save under a new name.')
    } catch (e: any) {
      setError(e.message || 'Generate failed')
      setMsg('')
    } finally {
      setBusy(false)
    }
  }

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
    <div>
      <div className="page-hero">
        <div>
          <h1>Email Design</h1>
          <p>Preview-first Starlight templates — or let AI draft a new one.</p>
        </div>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          <button className="btn secondary" type="button" disabled={busy} onClick={() => setShowAi((v) => !v)}>
            {showAi ? 'Hide AI create' : 'AI create template'}
          </button>
          <button className="btn" type="button" disabled={busy} onClick={save}>
            Save template
          </button>
        </div>
      </div>

      {showAi ? (
        <div className="panel tint-cyan stack" style={{ marginBottom: '1rem' }}>
          <strong style={{ fontFamily: 'var(--display)' }}>Create with AI</strong>
          <p className="muted" style={{ margin: 0, fontSize: 13 }}>
            Describe the layout and tone. AI returns a full Jinja HTML template using Starlight variables —
            then preview and save it as a custom design.
          </p>
          <label className="field">
            <span>Instructions</span>
            <textarea
              className="textarea"
              rows={4}
              value={instructions}
              disabled={busy}
              onChange={(e) => setInstructions(e.target.value)}
              placeholder="e.g. Dark header with lime accent, product cards for feature_highlights, soft CTA…"
            />
          </label>
          <div className="grid-2" style={{ gap: '0.75rem' }}>
            <label className="field">
              <span>Style</span>
              <select className="select" value={style} disabled={busy} onChange={(e) => setStyle(e.target.value)}>
                {STYLES.map((s) => (
                  <option key={s.value} value={s.value}>
                    {s.label}
                  </option>
                ))}
              </select>
            </label>
            <label className="toggle-row" style={{ alignSelf: 'end' }}>
              <input
                type="checkbox"
                checked={useReference}
                disabled={busy || !selected}
                onChange={(e) => setUseReference(e.target.checked)}
              />
              <span>
                <strong>Use current as reference</strong>
                <span className="muted" style={{ display: 'block', fontSize: 13 }}>
                  Borrow structure from {current ? displayLabel(current) : 'the selected template'}.
                </span>
              </span>
            </label>
          </div>
          <button className="btn" type="button" disabled={busy} onClick={generate}>
            {busy ? 'Generating…' : 'Generate template'}
          </button>
        </div>
      ) : null}

      <div className="panel stack" style={{ marginBottom: '1rem' }}>
        <div className="grid-2" style={{ gap: '0.75rem' }}>
          <label className="field">
            <span>Template</span>
            <select
              className="select"
              value={selected}
              disabled={busy}
              onChange={(e) => setSelected(e.target.value)}
            >
              {templates.map((t) => (
                <option key={t.name} value={t.name}>
                  {displayLabel(t)}
                  {t.is_custom ? ' · custom' : ''} ({t.name})
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            <span>Save as name</span>
            <input
              className="input"
              value={saveName}
              disabled={busy}
              onChange={(e) => setSaveName(e.target.value)}
              placeholder="email_template_my_design.html"
            />
          </label>
        </div>
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
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
          {msg ? <span className="pill ok">{msg}</span> : null}
          {current?.is_custom ? (
            <button className="btn danger" type="button" disabled={busy} onClick={remove}>
              Delete custom
            </button>
          ) : null}
        </div>
        {error ? (
          <div className="alert danger">
            <strong>Template error</strong>
            <div style={{ marginTop: 4 }}>{error}</div>
          </div>
        ) : null}
      </div>

      <div className="panel tint-blue stack">
        <strong style={{ fontFamily: 'var(--display)' }}>
          Customer preview{previewing ? ' · refreshing…' : ''}
        </strong>
        <EmailPreviewFrame html={previewHtml || content} subject="Sample Starlight email" />
      </div>
    </div>
  )
}
