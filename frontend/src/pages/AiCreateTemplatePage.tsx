import { useCallback, useEffect, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { ArrowLeftRegular, PaintBrushRegular, SaveRegular, SparkleRegular } from '@fluentui/react-icons'
import { api } from '../api/client'
import { EmailPreviewFrame } from '../components/EmailPreviewFrame'
import { CardHeader, MessageBar, PageHeader, Spinner, useToast } from '../components/ui'

type TemplateMeta = {
  name: string
  label: string
  builtin?: boolean
  is_custom?: boolean
}

const SUGGESTIONS = [
  'Dark navy header with lime accents',
  'Product cards for referenced_products',
  'Minimal, text-first, soft CTA',
  'Full-width hero image',
]

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
  const toast = useToast()
  const [templates, setTemplates] = useState<TemplateMeta[]>([])
  const [reference, setReference] = useState('')
  const [content, setContent] = useState('')
  const [previewHtml, setPreviewHtml] = useState('')
  const [error, setError] = useState('')
  const [generating, setGenerating] = useState(false)
  const [saving, setSaving] = useState(false)
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
  const busy = generating || saving

  const generate = async () => {
    if (!instructions.trim()) {
      setError('Describe the template you want.')
      return
    }
    setGenerating(true)
    setError('')
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
      toast.success('Draft ready', 'Review the preview, then save it.')
    } catch (e: any) {
      setError(e.message || 'Generate failed')
    } finally {
      setGenerating(false)
    }
  }

  const save = async () => {
    const name = (saveName.trim() || `email_template_ai_${Date.now()}`).replace(/\s+/g, '_')
    const finalName = name.endsWith('.html') ? name : `${name}.html`
    if (!content.trim()) {
      setError('Generate a template before saving.')
      return
    }
    setSaving(true)
    setError('')
    try {
      const res = await api.saveTemplate(finalName, content, saveLabel.trim() || undefined)
      toast.success('Template saved', res.label || finalName)
      navigate(`/admin/templates?selected=${encodeURIComponent(res.name || finalName)}`)
    } catch (e: any) {
      setError(e.message || 'Save failed')
    } finally {
      setSaving(false)
    }
  }

  const addSuggestion = (s: string) => {
    setInstructions((v) => (v.trim() ? `${v.trim().replace(/[.]$/, '')}. ${s}.` : `${s}.`))
  }

  return (
    <div>
      <PageHeader
        breadcrumb={[
          { label: 'Admin' },
          { label: 'Email design', to: '/admin/templates' },
          { label: 'Create with AI' },
        ]}
        title="Create with AI"
        subtitle="Describe any layout — AI drafts a fresh Jinja HTML template you can preview and save."
        actions={
          <>
            <Link to="/admin/templates" className="btn secondary">
              <ArrowLeftRegular /> Email design
            </Link>
            <button className="btn" type="button" disabled={busy || !content.trim()} onClick={save}>
              {saving ? <Spinner size="sm" /> : <SaveRegular />} Save template
            </button>
          </>
        }
      />

      <div className="design-layout">
        <section className="card design-preview">
          <div className="design-preview-bar">
            <strong>Customer preview</strong>
            {previewing ? (
              <Spinner size="sm" label="Refreshing…" />
            ) : (
              <span className="muted text-sm">{content.trim() ? 'Sample data' : 'Waiting for a draft'}</span>
            )}
          </div>
          {content.trim() ? (
            <EmailPreviewFrame html={previewHtml || content} subject="Sample Starlight email" fullscreen deviceToggle />
          ) : (
            <div className="preview-empty">
              {generating ? <Spinner size="lg" /> : <PaintBrushRegular className="preview-empty-icon" />}
              <strong>{generating ? 'Designing your template…' : 'No draft yet'}</strong>
              <p>
                {generating
                  ? 'This usually takes 20–40 seconds.'
                  : 'Describe the design and select Generate. There’s no need to start from a stock layout.'}
              </p>
            </div>
          )}
        </section>

        <aside className="card design-side">
          <CardHeader icon={<SparkleRegular />} title="Describe your design" />
          <div className="stack" style={{ marginTop: 16 }}>
            <label className="field">
              <span>Instructions</span>
              <textarea
                className="textarea"
                rows={7}
                value={instructions}
                disabled={busy}
                onChange={(e) => setInstructions(e.target.value)}
                placeholder="Layout, colours, sections, tone…"
              />
            </label>
            <div className="suggestions">
              {SUGGESTIONS.map((s) => (
                <button key={s} type="button" className="suggestion" disabled={busy} onClick={() => addSuggestion(s)}>
                  + {s}
                </button>
              ))}
            </div>
            <label className="field">
              <span>Reference template</span>
              <select className="select" value={reference} disabled={busy} onChange={(e) => setReference(e.target.value)}>
                <option value="">None — invent a new design</option>
                {templates.map((t) => (
                  <option key={t.name} value={t.name}>
                    {displayLabel(t)}
                    {t.is_custom ? ' · custom' : ''}
                  </option>
                ))}
              </select>
              <span className="field-hint">
                {reference
                  ? `May borrow structure from ${refMeta ? displayLabel(refMeta) : reference}, but should still feel original.`
                  : 'Leave as None so the AI isn’t steered toward the stock layouts.'}
              </span>
            </label>
            <button className={`btn${content.trim() ? ' secondary' : ''}`} type="button" disabled={busy} onClick={generate}>
              {generating ? <Spinner size="sm" /> : <SparkleRegular />}
              {generating ? 'Generating…' : content.trim() ? 'Regenerate' : 'Generate template'}
            </button>

            {content.trim() ? (
              <>
                <hr className="divider" />
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
                    className="input mono"
                    value={saveName}
                    disabled={busy}
                    onChange={(e) => setSaveName(e.target.value)}
                    placeholder="email_template_ai_design.html"
                  />
                </label>
              </>
            ) : null}

            {error ? (
              <MessageBar intent="error" title="Something went wrong" onDismiss={() => setError('')}>
                {error}
              </MessageBar>
            ) : null}
          </div>
        </aside>
      </div>
    </div>
  )
}
