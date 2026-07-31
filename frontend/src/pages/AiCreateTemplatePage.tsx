import { useCallback, useEffect, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { api } from '../api/client'
import { EmailPreviewFrame } from '../components/EmailPreviewFrame'

type TemplateMeta = {
  name: string
  label: string
  builtin?: boolean
  is_custom?: boolean
}

function displayLabel(t: TemplateMeta): string {
  if (t.label && t.label !== t.name) return t.label
  if (t.name.includes('minimalist')) return 'Minimalist'
  if (t.name.includes('bold')) return 'Bold & Vibrant'
  if (t.name === 'email_template.html') return 'Modern Soft'
  return t.name.replace(/^email_template_?/, '').replace(/\.html$/, '').replace(/_/g, ' ') || t.name
}

/** Admin page — generate a new email HTML template with AI, preview, and save. */
export function AiCreateTemplatePage() {
  const navigate = useNavigate()
  const [templates, setTemplates] = useState<TemplateMeta[]>([])
  const [reference, setReference] = useState('')
  const [content, setContent] = useState('')
  const [previewHtml, setPreviewHtml] = useState('')
  const [msg, setMsg] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [previewing, setPreviewing] = useState(false)

  const [instructions, setInstructions] = useState(
    'Clean Starlight LED outreach email with logo header, short personalized intro, feature highlights, use cases, product catalogue cards, and a clear CTA. Keep it mobile-friendly. Invent a fresh layout — do not copy the stock Modern Soft / Minimalist / Bold shells.',
  )
  const [saveName, setSaveName] = useState('')
  const [saveLabel, setSaveLabel] = useState('')

  const loadTemplates = useCallback(async () => {
    const list = await api.templates()
    setTemplates(list)
    return list
  }, [])

  useEffect(() => {
    loadTemplates().catch((e) => setError(e.message || 'Could not load templates'))
  }, [loadTemplates])

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

  const refMeta = templates.find((t) => t.name === reference)

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
        style: null,
        reference_template: reference || null,
      })
      setContent(res.content)
      const stamp = new Date().toISOString().slice(0, 10).replace(/-/g, '')
      setSaveName(`email_template_ai_${stamp}.html`)
      setSaveLabel('AI Custom')
      setMsg('Draft ready — review the preview, then save.')
    } catch (e: any) {
      setError(e.message || 'Generate failed')
      setMsg('')
    } finally {
      setBusy(false)
    }
  }

  const save = async () => {
    const name = (saveName.trim() || `email_template_ai_${Date.now()}`).replace(/\s+/g, '_')
    const finalName = name.endsWith('.html') ? name : `${name}.html`
    if (!content.trim()) {
      setError('Generate a template before saving.')
      return
    }
    setBusy(true)
    setError('')
    try {
      const res = await api.saveTemplate(finalName, content, saveLabel.trim() || undefined)
      setMsg(`Saved as ${res.label || finalName}`)
      navigate(`/admin/templates?selected=${encodeURIComponent(res.name || finalName)}`)
    } catch (e: any) {
      setError(e.message || 'Save failed')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="design-screen">
      <div className="page-hero">
        <div>
          <h1>Create with AI</h1>
          <p>Describe any layout you want — AI drafts a fresh Jinja HTML template. Old stock styles are optional inspiration only.</p>
        </div>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          <Link to="/admin/templates" className="btn secondary">
            Back to Email Design
          </Link>
          <button className="btn" type="button" disabled={busy || !content.trim()} onClick={save}>
            Save template
          </button>
        </div>
      </div>

      <div className="design-body">
        <div className="design-mail">
          <div className="design-preview-label muted">
            Customer preview{previewing ? ' · refreshing…' : ''}
            {!content.trim() ? ' · waiting for a draft' : ''}
          </div>
          {content.trim() ? (
            <EmailPreviewFrame
              html={previewHtml || content}
              subject="Sample Starlight email"
              fullscreen
            />
          ) : (
            <div className="panel empty-state" style={{ flex: 1, display: 'grid', placeItems: 'center' }}>
              <div style={{ textAlign: 'center', maxWidth: 360 }}>
                <strong style={{ fontFamily: 'var(--display)' }}>No draft yet</strong>
                <p className="muted" style={{ margin: '0.5rem 0 0' }}>
                  Describe the design on the right and hit Generate — no need to pick Modern Soft or any stock shell.
                </p>
              </div>
            </div>
          )}
        </div>

        <aside className="design-side panel tint-cyan stack">
          <strong style={{ fontFamily: 'var(--display)' }}>Prompt</strong>
          <label className="field">
            <span>Instructions</span>
            <textarea
              className="textarea"
              rows={7}
              value={instructions}
              disabled={busy}
              onChange={(e) => setInstructions(e.target.value)}
              placeholder="Describe layout, colors, sections, tone… e.g. dark navy header, lime accents, product cards for referenced_products, soft CTA"
            />
          </label>
          <label className="field">
            <span>Reference template (optional)</span>
            <select
              className="select"
              value={reference}
              disabled={busy}
              onChange={(e) => setReference(e.target.value)}
            >
              <option value="">None — invent a new design</option>
              {templates.map((t) => (
                <option key={t.name} value={t.name}>
                  {displayLabel(t)}
                  {t.is_custom ? ' · custom' : ''}
                </option>
              ))}
            </select>
            <span className="muted" style={{ fontSize: 12 }}>
              {reference
                ? `May borrow structure from ${refMeta ? displayLabel(refMeta) : reference}, but should still feel original.`
                : 'Leave as None so AI is not steered toward the old Modern Soft / Minimalist / Bold layouts.'}
            </span>
          </label>
          <button className="btn" type="button" disabled={busy} onClick={generate}>
            {busy && msg.startsWith('Generating') ? 'Generating…' : 'Generate template'}
          </button>

          {content.trim() ? (
            <>
              <hr style={{ border: 0, borderTop: '1px solid var(--border)', margin: '0.25rem 0' }} />
              <label className="field">
                <span>Display label</span>
                <input
                  className="input"
                  value={saveLabel}
                  disabled={busy}
                  onChange={(e) => setSaveLabel(e.target.value)}
                  placeholder="My AI design"
                />
              </label>
              <label className="field">
                <span>Save as filename</span>
                <input
                  className="input"
                  value={saveName}
                  disabled={busy}
                  onChange={(e) => setSaveName(e.target.value)}
                  placeholder="email_template_ai_design.html"
                />
              </label>
            </>
          ) : null}

          {msg ? <span className="pill ok">{msg}</span> : null}
          {error ? (
            <div className="alert danger">
              <strong>Create error</strong>
              <div style={{ marginTop: 4 }}>{error}</div>
            </div>
          ) : null}
        </aside>
      </div>
    </div>
  )
}
