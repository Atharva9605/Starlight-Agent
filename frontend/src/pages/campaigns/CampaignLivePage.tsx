import { Link, Navigate, useNavigate } from 'react-router-dom'
import { EmailPreviewFrame } from '../../components/EmailPreviewFrame'
import { leadState, useCampaign, type StageEvent } from '../../campaign/CampaignContext'

const STAGE_ORDER = ['queued', 'scrape', 'analyze', 'retrieve', 'draft', 'render', 'send'] as const

const STAGE_LABELS: Record<string, string> = {
  queued: 'Queued',
  scrape: 'Scrape',
  analyze: 'Analyze',
  retrieve: 'Catalogue',
  draft: 'Draft',
  render: 'Preview',
  send: 'Send',
  error: 'Failed',
}

function mergeTimeline(events: StageEvent[] | undefined): { id: string; label: string; state: string }[] {
  const byStage = new Map((events || []).map((e) => [e.stage, e]))
  const rows: { id: string; label: string; state: string }[] = STAGE_ORDER.map((id) => {
    const ev = byStage.get(id)
    return {
      id,
      label: ev?.label || STAGE_LABELS[id],
      state: ev?.state || 'pending',
    }
  })
  if (byStage.has('error')) {
    const err = byStage.get('error')!
    rows.push({ id: 'error', label: err.label || 'Failed', state: 'error' })
  }
  return rows
}

/** Active stage, else last completed, else Queued — for the lead queue column. */
function currentStageLabel(events: StageEvent[] | undefined): string {
  const timeline = mergeTimeline(events)
  const active = timeline.find((s) => s.state === 'active')
  if (active) return active.label
  const err = timeline.find((s) => s.state === 'error')
  if (err) return err.label
  const done = [...timeline].reverse().find((s) => s.state === 'done')
  if (done) return done.label
  return STAGE_LABELS.queued
}

/** Live progress logs — preview + per-lead pipeline stages. */
export function CampaignLivePage() {
  const nav = useNavigate()
  const {
    leads,
    status,
    logs,
    counts,
    stop,
    reset,
    livePreview,
    stagesByLead,
    runSender,
    currentIndex,
  } = useCampaign()

  if (!leads.length) return <Navigate to="/campaigns" replace />

  const running = status === 'running'
  const pct = counts.progressPct

  const activeIdx = leads.findIndex((l) => {
    const s = leadState(l)
    return s === 'processing' || s === 'pending'
  })
  const focusIdx =
    livePreview?.rowIndex ??
    (currentIndex >= 0 ? currentIndex : activeIdx >= 0 ? activeIdx : Math.max(0, counts.processed - 1))
  const focus = leads[focusIdx]
  const focusState = focus ? leadState(focus) : 'pending'
  const timeline = mergeTimeline(stagesByLead[focusIdx])
  const previewHtml = livePreview?.html || focus?._preview_html || ''
  const previewSubject = livePreview?.subject || focus?._subject || 'Starlight outreach'
  const fromLabel = livePreview?.from || runSender || 'Starlight Linear LED'
  const productSheet = livePreview?.productSheet || focus?._product_sheet || ''
  const productCount = livePreview?.productCount ?? focus?._product_count ?? 0
  const activeStep = timeline.find((s) => s.state === 'active') || timeline.find((s) => s.state === 'pending')

  return (
    <div className="dash-screen">
      <header className="dash-hero">
        <div>
          <div className="dash-kicker">
            {running ? <span className="live-dot" /> : null}
            Live progress logs
          </div>
          <h1>
            {running
              ? `Working lead ${(focusIdx || 0) + 1} of ${counts.total}`
              : status === 'done'
                ? 'Campaign complete'
                : status === 'stopped'
                  ? 'Campaign stopped'
                  : 'Live progress logs'}
          </h1>
          <p className="muted">
            {counts.sent} sent · {counts.failed} failed · {counts.processing} in progress
            {runSender ? ` · from ${runSender}` : ''}
          </p>
        </div>
        <div className="row">
          {running ? (
            <button className="btn danger" type="button" onClick={stop}>Stop</button>
          ) : (
            <>
              <button
                className="btn secondary"
                type="button"
                onClick={() => {
                  reset()
                  nav('/campaigns')
                }}
              >
                New campaign
              </button>
              <Link to="/inbox" className="btn">Inbox</Link>
            </>
          )}
        </div>
      </header>

      <div className="dash-progress panel">
        <div className="row" style={{ justifyContent: 'space-between', marginBottom: 10 }}>
          <div>
            <strong style={{ fontSize: 15 }}>{pct}% complete</strong>
            <span className="muted" style={{ marginLeft: 10, fontSize: 13 }}>
              {counts.processed} finished · {Math.max(0, counts.total - counts.processed - counts.processing)} waiting
            </span>
          </div>
          {activeStep ? (
            <span className="pill warn">{activeStep.label}</span>
          ) : null}
        </div>
        <div className="progress thick">
          <div className={`progress-fill${running ? ' animated' : ''}`} style={{ width: `${pct}%` }} />
        </div>
        <ol className="dash-rail">
          {timeline.filter((s) => s.id !== 'error').map((step) => (
            <li key={step.id} className={`dash-rail-step ${step.state}`}>
              <span className="dash-rail-dot" />
              <span className="dash-rail-label">{STAGE_LABELS[step.id] || step.id}</span>
            </li>
          ))}
        </ol>
      </div>

      <div className="dash-grid">
        <section className="dash-preview panel stack">
          <div className="row" style={{ justifyContent: 'space-between', alignItems: 'flex-start' }}>
            <div>
              <div className="design-preview-label muted">Live email preview</div>
              <h2 style={{ margin: '0.2rem 0 0', fontFamily: 'var(--display)', fontSize: '1.25rem' }}>
                {focus?.company || focus?.website || `Lead ${(focusIdx || 0) + 1}`}
              </h2>
              {focus?.website && focus?.company ? (
                <p className="muted" style={{ margin: '0.25rem 0 0', fontSize: 13 }}>{focus.website}</p>
              ) : null}
            </div>
            <div className="row" style={{ gap: 6 }}>
              {productSheet ? <span className="pill ok">PDF sheet attached</span> : null}
              {productCount ? <span className="pill">{productCount} products</span> : null}
              <span className={`pill ${focusState === 'sent' ? 'ok' : focusState === 'failed' ? 'pink' : 'warn'}`}>
                {focus?._status || 'Queued'}
              </span>
            </div>
          </div>

          {previewHtml ? (
            <div className="dash-preview-frame">
              <EmailPreviewFrame
                html={previewHtml}
                subject={previewSubject}
                fromLabel={fromLabel}
                fullscreen
              />
            </div>
          ) : (
            <div className="skeleton-frame tall live-preview-empty">
              <div className="live-pulse-ring" />
              <strong>{running ? 'Generating this email…' : 'Preview appears here as each email is written'}</strong>
              <p className="muted" style={{ margin: '0.4rem 0 0', maxWidth: '40ch' }}>
                Live progress logs — open anytime from Campaigns → View Live progress Logs while leads are loaded.
              </p>
            </div>
          )}
        </section>

        <aside className="dash-side">
          <div className="panel stack">
            <strong style={{ fontFamily: 'var(--display)' }}>This email</strong>
            <ul className="dash-detail-list">
              {timeline.map((step) => (
                <li key={step.id} className={`dash-detail ${step.state}`}>
                  <span className="dash-detail-mark" />
                  <div>
                    <div className="dash-detail-title">{step.label}</div>
                    <div className="muted" style={{ fontSize: 12 }}>
                      {step.state === 'done'
                        ? 'Done'
                        : step.state === 'active'
                          ? 'Running now'
                          : step.state === 'error'
                            ? 'Error'
                            : 'Waiting'}
                    </div>
                  </div>
                </li>
              ))}
            </ul>
          </div>

          <div className="panel stack">
            <strong style={{ fontFamily: 'var(--display)' }}>Lead queue</strong>
            <div className="queue dash-queue">
              {leads.map((l, i) => {
                const st = leadState(l)
                const stage = currentStageLabel(stagesByLead[i])
                return (
                  <div key={i} className={`queue-row${i === focusIdx ? ' active' : ''}`}>
                    <span className={`queue-index ${st}`}>{i + 1}</span>
                    <span className="queue-name">
                      {l.website || l.company || `Lead ${i + 1}`}
                      <span className="muted" style={{ display: 'block', fontSize: 12, marginTop: 2 }}>
                        {stage}
                      </span>
                    </span>
                    <span className={`pill ${st === 'sent' ? 'ok' : st === 'failed' ? 'pink' : st === 'processing' ? 'warn' : ''}`}>
                      {st === 'pending' ? 'Queued' : (l._status || st)}
                    </span>
                  </div>
                )
              })}
            </div>
          </div>

          <div className="stat-row dash-stats">
            <div className="stat blue"><div className="label">Sent</div><div className="value">{counts.sent}</div></div>
            <div className="stat amber"><div className="label">Failed</div><div className="value">{counts.failed}</div></div>
            <div className="stat cyan"><div className="label">Left</div><div className="value">{Math.max(0, counts.total - counts.processed)}</div></div>
          </div>

          {logs.length ? (
            <div className="panel stack">
              <strong style={{ fontFamily: 'var(--display)' }}>Activity</strong>
              <pre className="terminal">{logs.slice(-30).join('\n')}</pre>
            </div>
          ) : null}
        </aside>
      </div>
    </div>
  )
}
