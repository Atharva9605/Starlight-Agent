import { useState } from 'react'
import { Navigate, useNavigate } from 'react-router-dom'
import { TEMPLATES, useCampaign } from '../../campaign/CampaignContext'

export function StepReview() {
  const nav = useNavigate()
  const {
    leads,
    fileName,
    template,
    delay,
    setDelay,
    recipientOverride,
    setRecipientOverride,
    senderEmail,
    setSenderEmail,
    start,
  } = useCampaign()
  const [advanced, setAdvanced] = useState(false)

  if (!leads.length) return <Navigate to="/campaigns/new/leads" replace />

  const templateLabel = TEMPLATES.find((t) => t.value === template)?.label || template
  const testMode = Boolean(recipientOverride.trim())
  const eta = Math.round((leads.length * (delay + 18)) / 60)

  const launch = async () => {
    nav('/campaigns/live')
    void start()
  }

  return (
    <div className="step-body">
      <div className="review-grid">
        <div className="panel stack">
          <strong style={{ fontFamily: 'var(--display)' }}>Summary</strong>
          <dl className="summary-list">
            <div>
              <dt>Leads</dt>
              <dd>{leads.length} from {fileName || 'your upload'}</dd>
            </div>
            <div>
              <dt>Template</dt>
              <dd>{templateLabel}</dd>
            </div>
            <div>
              <dt>Pause between sends</dt>
              <dd>{delay}s</dd>
            </div>
            <div>
              <dt>Estimated time</dt>
              <dd>~{Math.max(eta, 1)} min</dd>
            </div>
            <div>
              <dt>Recipients</dt>
              <dd>
                {testMode ? (
                  <span className="pill warn">Test mode → {recipientOverride}</span>
                ) : (
                  'Real contacts scraped from each site'
                )}
              </dd>
            </div>
          </dl>
        </div>

        <div className="panel stack">
          <strong style={{ fontFamily: 'var(--display)' }}>Sending options</strong>

          <label className="field">
            <span>Pause between emails</span>
            <div className="row" style={{ gap: '0.75rem' }}>
              <input
                type="range"
                min={1}
                max={15}
                value={delay}
                onChange={(e) => setDelay(Number(e.target.value))}
                style={{ flex: 1 }}
              />
              <span className="pill">{delay}s</span>
            </div>
            <span className="muted" style={{ fontSize: 12 }}>
              Slower sending looks more human and protects deliverability.
            </span>
          </label>

          <label className="field">
            <span>Send everything to one test inbox</span>
            <input
              className="input"
              placeholder="you@starlightlinearled.com (optional)"
              value={recipientOverride}
              onChange={(e) => setRecipientOverride(e.target.value)}
            />
            <span className="muted" style={{ fontSize: 12 }}>
              Recommended for your first run — no real prospect is contacted.
            </span>
          </label>

          <button className="btn secondary" type="button" onClick={() => setAdvanced((v) => !v)}>
            {advanced ? 'Hide advanced' : 'Advanced options'}
          </button>

          {advanced ? (
            <label className="field advanced-box">
              <span>Send from a specific mailbox</span>
              <input
                className="input"
                placeholder="Leave blank to use the connected Gmail"
                value={senderEmail}
                onChange={(e) => setSenderEmail(e.target.value)}
              />
            </label>
          ) : null}
        </div>
      </div>

      {!testMode ? (
        <div className="alert warn">
          <strong>This sends real emails.</strong> Add a test inbox above if you just want to see
          how it looks.
        </div>
      ) : null}

      <div className="step-actions">
        <button className="btn secondary" onClick={() => nav('/campaigns/new/design')}>
          ← Back to design
        </button>
        <button className="btn" onClick={launch}>
          🚀 Launch to {leads.length} {leads.length === 1 ? 'lead' : 'leads'}
        </button>
      </div>
    </div>
  )
}
