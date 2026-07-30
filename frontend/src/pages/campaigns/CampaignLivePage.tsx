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

  return (
    <div>
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
            <button className="btn danger" onClick={stop}>Stop</button>
          ) : (
            <>
              <button className="btn secondary" onClick={() => { reset(); nav('/campaigns') }}>
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

      <div className="panel stack">
        <strong style={{ fontFamily: 'var(--display)' }}>Lead queue</strong>
        <div className="queue">
          {leads.map((l, i) => {
            const st = leadState(l)
            return (
              <div key={i} className="queue-row">
                <span className={`queue-index ${st}`}>{i + 1}</span>
                <span className="queue-name">{l.website || l.company || `Lead ${i + 1}`}</span>
                <span className="pill">{l._status || 'Queued'}</span>
              </div>
            )
          })}
        </div>
      </div>

      {logs.length ? (
        <div className="panel stack" style={{ marginTop: '1rem' }}>
          <strong style={{ fontFamily: 'var(--display)' }}>Log</strong>
          <pre className="terminal">{logs.join('\n')}</pre>
        </div>
      ) : null}
    </div>
  )
}
