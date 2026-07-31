import { Link, Navigate, useNavigate } from 'react-router-dom'
import { leadState, useCampaign } from '../../campaign/CampaignContext'

/** Autosend live monitor — only used when autosend is on. */
export function CampaignLivePage() {
  const nav = useNavigate()
  const { leads, status, logs, counts, stop, reset, autosend } = useCampaign()

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
  const focusIdx = activeIdx >= 0 ? activeIdx : Math.max(0, counts.processed - 1)
  const focus = leads[focusIdx]
  const focusState = focus ? leadState(focus) : 'pending'

  return (
    <div className="live-screen">
      <div className="page-hero">
        <div>
          <h1>
            {running ? <span className="live-dot" /> : null}
            {running ? 'Autosending' : status === 'done' ? 'Autosend complete' : status === 'stopped' ? 'Stopped' : 'Autosend'}
          </h1>
          <p>
            {counts.sent} sent · {counts.failed} failed · {counts.total} total
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
          <strong>{counts.processed} of {counts.total}</strong>
          <span className="muted">{pct}%</span>
        </div>
        <div className="progress">
          <div className={`progress-fill${running ? ' animated' : ''}`} style={{ width: `${pct}%` }} />
        </div>
      </div>

      <div className="live-body">
        <div className="live-focus panel stack">
          <div className="design-preview-label muted">Current lead</div>
          {focus ? (
            <>
              <h2 style={{ margin: 0, fontFamily: 'var(--display)', fontSize: '1.35rem' }}>
                {focus.website || focus.company || `Lead ${focusIdx + 1}`}
              </h2>
              <div className="row">
                <span className={`pill ${focusState === 'sent' ? 'ok' : focusState === 'failed' ? 'pink' : 'warn'}`}>
                  {focus._status || 'Queued'}
                </span>
                {focus.email ? <span className="muted" style={{ fontSize: 13 }}>{focus.email}</span> : null}
                {focus.company ? <span className="muted" style={{ fontSize: 13 }}>{focus.company}</span> : null}
              </div>
              <p className="muted" style={{ margin: 0, fontSize: 14, maxWidth: '52ch' }}>
                {running
                  ? 'Scraping, writing, and sending automatically. Watch the queue and log for progress.'
                  : status === 'done'
                    ? 'All leads processed. Open Inbox for replies, or start a new campaign.'
                    : 'Autosend is idle. Resume from Campaigns or start fresh.'}
              </p>
              <div className="stat-row" style={{ marginTop: 'auto' }}>
                <div className="stat blue"><div className="label">Sent</div><div className="value">{counts.sent}</div></div>
                <div className="stat amber"><div className="label">Failed</div><div className="value">{counts.failed}</div></div>
                <div className="stat cyan"><div className="label">Left</div><div className="value">{Math.max(0, counts.total - counts.processed)}</div></div>
              </div>
            </>
          ) : (
            <div className="empty-state">No leads in this run.</div>
          )}
        </div>

        <aside className="live-side">
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
              <pre className="terminal">{logs.join('\n')}</pre>
            </div>
          ) : null}
        </aside>
      </div>
    </div>
  )
}
