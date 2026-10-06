import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { CheckmarkCircleFilled, WandRegular } from '@fluentui/react-icons'
import { api } from '../../../api/client'
import { useAuth } from '../../../auth/AuthContext'
import { useCampaign } from '../../../campaign/CampaignContext'
import { EmailPreviewFrame } from '../../../components/EmailPreviewFrame'
import { CardHeader, Spinner } from '../../../components/ui'
import { WizardNav, useWizard } from './CampaignWizard'

/** Step 2 — pick the layout, with a live sample of each. */
export function DesignStep() {
  const { template, setTemplate } = useCampaign()
  const { templateOptions } = useWizard()
  const { me } = useAuth()
  const isAdmin = me?.role === 'owner' || me?.role === 'admin'
  const [html, setHtml] = useState('')
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (!template) return
    let cancelled = false
    setLoading(true)
    api
      .previewTemplate({ template_name: template })
      .then((res) => {
        if (!cancelled) setHtml(res.html || '')
      })
      .catch(() => {
        if (!cancelled) setHtml('')
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [template])

  const current = templateOptions.find((t) => t.value === template)

  return (
    <>
      <div className="wizard-body design">
        <section className="card">
          <CardHeader
            title="Choose a design"
            subtitle="The layout every email in this campaign uses. The words are written per lead."
            actions={
              isAdmin ? (
                <Link to="/admin/templates/create" className="btn subtle sm">
                  <WandRegular /> Create with AI
                </Link>
              ) : null
            }
          />
          <div className="stack tight" role="radiogroup" aria-label="Email design" style={{ marginTop: 16 }}>
            {templateOptions.map((t) => (
              <button
                key={t.value}
                type="button"
                role="radio"
                aria-checked={template === t.value}
                className={`choice-card${template === t.value ? ' selected' : ''}`}
                onClick={() => setTemplate(t.value)}
              >
                <span className="swatch" style={{ background: `linear-gradient(135deg, ${t.swatch[0]}, ${t.swatch[1]})` }} />
                <span className="choice-card-copy">
                  <strong>
                    {t.label} {t.custom ? <span className="badge teal">Custom</span> : null}
                  </strong>
                  <span>{t.blurb}</span>
                </span>
                {template === t.value ? <CheckmarkCircleFilled className="choice-card-check" /> : <span className="choice-card-ring" />}
              </button>
            ))}
          </div>
        </section>
        <section className="card design-preview">
          <div className="design-preview-bar">
            <strong>{current?.label || 'Preview'}</strong>
            {loading ? <Spinner size="sm" label="Loading…" /> : <span className="muted text-sm">Shown with sample text</span>}
          </div>
          {html ? (
            <EmailPreviewFrame html={html} subject="Sample Starlight email" fullscreen deviceToggle />
          ) : (
            <div className="preview-empty">{loading ? <Spinner size="lg" /> : <strong>No preview available</strong>}</div>
          )}
        </section>
      </div>
      <WizardNav step={1} canNext={Boolean(template)} />
    </>
  )
}
