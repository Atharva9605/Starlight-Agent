import { useCallback, useEffect, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import {
  DeleteRegular,
  LockClosedRegular,
  PaintBrushRegular,
  SaveRegular,
  WandRegular,
} from '@fluentui/react-icons'
import { api } from '../api/client'
import { EmailPreviewFrame } from '../components/EmailPreviewFrame'
import { CardHeader, MessageBar, PageHeader, Spinner, useConfirm, useToast } from '../components/ui'

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
  const toast = useToast()
  const confirm = useConfirm()
  const [searchParams] = useSearchParams()
  const preferSelected = searchParams.get('selected') || ''

  const [templates, setTemplates] = useState<TemplateMeta[]>([])
  const [selected, setSelected] = useState('')
  const [content, setContent] = useState('')
  const [previewHtml, setPreviewHtml] = useState('')
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
      toast.success('Template saved', res.label || finalName)
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
    const ok = await confirm({
      title: 'Delete this template?',
      body: `“${displayLabel(meta)}” will be removed. Campaigns that use it will fall back to the default design.`,
      confirmLabel: 'Delete',
      danger: true,
    })
    if (!ok) return
    setBusy(true)
    try {
      await api.deleteTemplate(selected)
      await refreshList()
      toast.info('Template deleted')
    } catch (e: any) {
      setError(e.message || 'Delete failed')
    } finally {
      setBusy(false)
    }
  }

  const current = templates.find((t) => t.name === selected)

  return (
    <div>
      <PageHeader
        breadcrumb={[{ label: 'Admin' }, { label: 'Email design' }]}
        title="Email design"
        subtitle="Preview exactly what customers see — pick a saved template or create a new one with AI."
        actions={
          <>
            <Link to="/admin/templates/create" className="btn secondary">
              <WandRegular /> Create with AI
            </Link>
            <button className="btn" type="button" disabled={busy} onClick={save}>
              {busy ? <Spinner size="sm" /> : <SaveRegular />} Save template
            </button>
          </>
        }
      />

      <div className="design-layout">
        <section className="card design-preview">
          <div className="design-preview-bar">
            <strong>Customer preview</strong>
            {previewing ? <Spinner size="sm" label="Refreshing…" /> : <span className="muted text-sm">Sample data</span>}
          </div>
          <EmailPreviewFrame html={previewHtml || content} subject="Sample Starlight email" fullscreen deviceToggle />
        </section>

        <aside className="card design-side">
          <CardHeader icon={<PaintBrushRegular />} title="Template" />
          <div className="stack" style={{ marginTop: 16 }}>
            <div className="vnav" role="listbox" aria-label="Templates" style={{ maxHeight: 280, overflow: 'auto' }}>
              {templates.map((t) => (
                <button
                  key={t.name}
                  type="button"
                  role="option"
                  aria-selected={selected === t.name}
                  className={`vnav-item${selected === t.name ? ' active' : ''}`}
                  disabled={busy}
                  onClick={() => setSelected(t.name)}
                >
                  <span className="vnav-copy">
                    <span className="vnav-title">{displayLabel(t)}</span>
                    <span className="vnav-sub mono">{t.name}</span>
                  </span>
                  {t.is_custom ? <span className="badge teal">Custom</span> : <LockClosedRegular title="Built-in" />}
                </button>
              ))}
            </div>

            <hr className="divider" />

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
                className="input mono"
                value={saveName}
                disabled={busy}
                onChange={(e) => setSaveName(e.target.value)}
                placeholder="email_template_my_design.html"
              />
              <span className="field-hint">Use a new filename to keep a built-in design intact.</span>
            </label>

            {error ? (
              <MessageBar intent="error" title="Template error" onDismiss={() => setError('')}>
                {error}
              </MessageBar>
            ) : null}

            {current?.is_custom ? (
              <button className="btn danger-outline" type="button" disabled={busy} onClick={remove}>
                <DeleteRegular /> Delete template
              </button>
            ) : (
              <MessageBar intent="info">Built-in templates are read-only — save under a new filename to keep edits.</MessageBar>
            )}
          </div>
        </aside>
      </div>
    </div>
  )
}
