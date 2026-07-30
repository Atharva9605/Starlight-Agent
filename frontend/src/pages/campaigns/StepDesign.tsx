import { useQuery } from '@tanstack/react-query'
import { Navigate, useNavigate } from 'react-router-dom'
import { api } from '../../api/client'
import { EmailPreviewFrame } from '../../components/EmailPreviewFrame'
import { TEMPLATES, useCampaign } from '../../campaign/CampaignContext'

export function StepDesign() {
  const nav = useNavigate()
  const { leads, template, setTemplate } = useCampaign()

  const preview = useQuery({
    queryKey: ['template-preview', template],
    queryFn: () => api.previewTemplate(template),
    retry: false,
  })

  if (!leads.length) return <Navigate to="/campaigns/new/leads" replace />

  return (
    <div className="step-body">
      <div className="design-grid">
        <div className="stack">
          <div className="panel stack">
            <div>
              <strong style={{ fontFamily: 'var(--display)' }}>Choose a template</strong>
              <p className="muted" style={{ margin: '0.25rem 0 0' }}>
                Copy is written per lead by the AI — this controls the layout and styling.
              </p>
            </div>

            <div className="stack" style={{ gap: '0.6rem' }}>
              {TEMPLATES.map((t) => (
                <button
                  key={t.value}
                  type="button"
                  className={`template-card${template === t.value ? ' selected' : ''}`}
                  onClick={() => setTemplate(t.value)}
                >
                  <span
                    className="template-swatch"
                    style={{ background: `linear-gradient(135deg, ${t.swatch[0]}, ${t.swatch[1]})` }}
                  />
                  <span className="template-copy">
                    <strong>{t.label}</strong>
                    <span className="muted">{t.blurb}</span>
                  </span>
                  <span className="template-check">{template === t.value ? '✓' : ''}</span>
                </button>
              ))}
            </div>
          </div>

          <div className="panel tint-cyan">
            <strong style={{ fontFamily: 'var(--display)' }}>Where the words come from</strong>
            <p className="muted" style={{ margin: '0.5rem 0 0' }}>
              Each email is generated from the lead's website plus your indexed catalogues, then
              poured into this template. Wording lives under Admin → Prompt Studio.
            </p>
          </div>
        </div>

        <div className="panel tint-blue stack preview-panel">
          <div className="row" style={{ justifyContent: 'space-between' }}>
            <strong style={{ fontFamily: 'var(--display)' }}>Sample preview</strong>
            <span className="pill">Sample data</span>
          </div>

          {preview.isLoading ? (
            <div className="skeleton-frame">Rendering preview…</div>
          ) : preview.isError ? (
            <div className="alert danger">
              Could not render this template. It may have a syntax error — check Admin → Email Design.
            </div>
          ) : (
            <EmailPreviewFrame
              html={preview.data?.html}
              subject="Precision lighting for your next project"
            />
          )}
        </div>
      </div>

      <div className="step-actions">
        <button className="btn secondary" onClick={() => nav('/campaigns/new/leads')}>
          ← Back to leads
        </button>
        <button className="btn" onClick={() => nav('/campaigns/new/review')}>
          Continue to review →
        </button>
      </div>
    </div>
  )
}
