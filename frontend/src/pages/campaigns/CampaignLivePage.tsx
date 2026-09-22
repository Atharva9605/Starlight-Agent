import { Link, Navigate, useNavigate } from 'react-router-dom'
import { EmailPreviewFrame } from '../../components/EmailPreviewFrame'
import { leadState, useCampaign, type StageEvent } from '../../campaign/CampaignContext'

const STAGE_ORDER = ['queued', 'scrape', 'analyze', 'retrieve', 'draft', 'render', 'send'] as const

const STAGE_LABELS: Record<string, string> = {
  queued: 'Queued',
  scrape: 'Scrape website',
  analyze: 'Analyze company',
  retrieve: 'Match catalogue',
  draft: 'Write email',
  render: 'Render preview',
  send: 'Send via Gmail',
  error: 'Failed',
}

function mergeTimeline(events: StageEvent[] | undefined): { id: string; label: string; state: string }[] {
  const byStage = new Map((events || []).map((e) => [e.stage, e]))
  const rows = STAGE_ORDER.map((id) => {
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

/** Autosend live monitor — live preview + per-email progress timeline. */
export function CampaignLivePage() {
  const nav = useNavigate()
  const {
    leads,
    status,
    logs,
    counts,
    stop,
    reset,
    autosend,
    livePreview,
    stagesByLead,
    runSender,
    currentIndex,
  } = useCampaign()

  if (!leads.length) return <Navigate to="/campaigns" replace />
  if (!autosend && status !== 'running' && status !== 'done' && status !== 'stopped' && status !== 'failed') {
    return <Navigate to="/campaigns/review" replace />
  }

  const running = status === 'running'
  const pct = counts.total ? Math.round((counts.processed / counts.total) * 100) : 0

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

  return (
    <div className="live-screen">
      <div className="page-hero">
        <div>
          <h1>
            {running ? <span className="live-dot" /> : null}
            {running
              ? 'Live generation'
              : status === 'done'
                ? 'Campaign complete'
                : status === 'stopped'
                  ? 'Stopped'
                  : 'Autosend'}
          </h1>
          <p>
            {counts.sent} sent · {counts.failed} failed · {counts.total} total
            {runSender ? ` · sending as ${runSender}` : ''}
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
      </div>

      <div className="panel progress-panel">
        <div className="row" style={{ justifyContent: 'space-between', marginBottom: 10 }}>
          <strong>{counts.processed} of {counts.total} leads</strong>
          <span className="muted">{pct}%</span>
        </div>
        <div className="progress">
          <div className={`progress-fill${running ? ' animated' : ''}`} style={{ width: `${pct}%` }} />
        </div>
      </div>

      <div className="live-dashboard">
        <section className="live-preview-pane panel stack">
          <div className="row" style={{ justifyContent: 'space-between', alignItems: 'baseline' }}>
            <div>
              <div className="design-preview-label muted">Live customer preview</div>
              <h2 style={{ margin: '0.15rem 0 0', fontFamily: 'var(--display)', fontSize: '1.2rem' }}>
                {focus?.website || focus?.company || `Lead ${(focusIdx || 0) + 1}`}
              </h2>
            </div>
            <span className={`pill ${focusState === 'sent' ? 'ok' : focusState === 'failed' ? 'pink' : 'warn'}`}>
              {focus?._status || 'Queued'}
            </span>
          </div>

          {previewHtml ? (
            <div className="live-preview-frame">
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
              <strong>{running ? 'Generating this email…' : 'Preview appears as each email is written'}</strong>
              <p className="muted" style={{ margin: '0.4rem 0 0', maxWidth: '36ch' }}>
                Scraping, catalogue match, and GPT drafting stream into this pane the moment HTML is ready.
              </p>
            </div>
          )}
        </section>

        <aside className="live-side">
          <div className="panel stack live-timeline-panel">
            <strong style={{ fontFamily: 'var(--display)' }}>Generation timeline</strong>
            <p className="muted" style={{ margin: 0, fontSize: 13 }}>
              Per-email pipeline for lead {(focusIdx || 0) + 1}
            </p>
            <ol className="gen-timeline">
              {timeline.map((step) => (
                <li key={step.id} className={`gen-step ${step.state}`}>
                  <span className="gen-step-marker" aria-hidden />
                  <div className="gen-step-body">
                    <div className="gen-step-title">{step.label}</div>
                    <div className="gen-step-meta muted">
                      {step.state === 'done'
                        ? 'Complete'
                        : step.state === 'active'
                          ? 'In progress'
                          : step.state === 'error'
                            ? 'Error'
                            : 'Waiting'}
                    </div>
                  </div>
                </li>
              ))}
            </ol>
            <div className="stat-row" style={{ marginTop: 4 }}>
              <div className="stat blue"><div className="label">Sent</div><div className="value">{counts.sent}</div></div>
              <div className="stat amber"><div className="label">Failed</div><div className="value">{counts.failed}</div></div>
              <div className="stat cyan"><div className="label">Left</div><div className="value">{Math.max(0, counts.total - counts.processed)}</div></div>
            </div>
          </div>

          <div className="panel stack">
            <strong style={{ fontFamily: 'var(--display)' }}>Lead queue</strong>
            <div className="queue">
              {leads.map((l, i) => {
                const st = leadState(l)
                return (
                  <div key={i} className={`queue-row${i === focusIdx ? ' active' : ''}`}>
                    <span className={`queue-index ${st}`}>{i + 1}</span>
                    <span className="queue-name">{l.website || l.company || `Lead ${i + 1}`}</span>
                    <span className="pill">{l._status || 'Queued'}</span>
                  </div>
                )
              })}
            </div>
          </div>

          {logs.length ? (
            <div className="panel stack">
              <strong style={{ fontFamily: 'var(--display)' }}>Log</strong>
              <pre className="terminal">{logs.slice(-40).join('\n')}</pre>
            </div>
          ) : null}
        </aside>
      </div>
    </div>
  )
}
