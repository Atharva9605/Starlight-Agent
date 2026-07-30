import { useEffect, useState } from 'react'
import { Link, Navigate, useNavigate } from 'react-router-dom'
import { EmailPreviewFrame } from '../../components/EmailPreviewFrame'
import { leadState, useCampaign } from '../../campaign/CampaignContext'

const STATE_LABEL: Record<string, { label: string; cls: string }> = {
  sent: { label: 'Sent', cls: 'ok' },
  failed: { label: 'Failed', cls: 'pink' },
  processing: { label: 'Working…', cls: 'warn' },
  pending: { label: 'Queued', cls: '' },
}

export function LiveRunPage() {
  const nav = useNavigate()
  const { leads, status, logs, previews, currentRow, counts, stop, reset } = useCampaign()
  const [showLog, setShowLog] = useState(false)
  const [selected, setSelected] = useState(0)

  // Follow the newest preview while the run is live.
  useEffect(() => {
    if (status === 'running' && previews.length) setSelected(previews.length - 1)
  }, [previews.length, status])

  if (!leads.length) return <Navigate to="/campaigns/new/leads" replace />

  const pct = counts.total ? Math.round((counts.processed / counts.total) * 100) : 0
  const active = previews[Math.min(selected, previews.length - 1)]
  const running = status === 'running'

  const headline =
    running ? 'Sending now'
    : status === 'done' ? 'Campaign complete'
    : status === 'stopped' ? 'Campaign stopped'
    : status === 'failed' ? 'Campaign hit an error'
    : 'Ready to send'

  return (
    <div>
      <div className="page-hero">
        <div>
          <h1>
            {running ? <span className="live-dot" /> : null}
            {headline}
          </h1>
          <p>
            {running
              ? 'Each lead is scraped, written, and sent. You can leave this page — it keeps running.'
              : `${counts.sent} sent · ${counts.failed} failed · ${counts.total} total`}
          </p>
        </div>
        <div className="row">
          {running ? (
            <button className="btn danger" onClick={stop}>Stop</button>
          ) : (
            <>
              <button className="btn secondary" onClick={() => { reset(); nav('/campaigns/new/leads') }}>
                New campaign
              </button>
              <Link to="/inbox" className="btn">View inbox</Link>
            </>
          )}
        </div>
      </div>

      <div className="panel progress-panel">
        <div className="row" style={{ justifyContent: 'space-between', marginBottom: 10 }}>
          <strong>{counts.processed} of {counts.total} processed</strong>
          <span className="muted">{pct}%</span>
        </div>
        <div className="progress">
          <div className={`progress-fill${running ? ' animated' : ''}`} style={{ width: `${pct}%` }} />
        </div>
        <div className="mini-stats" style={{ marginTop: 12 }}>
          <div><span className="muted">Sent</span><strong style={{ color: 'var(--ok)' }}>{counts.sent}</strong></div>
          <div><span className="muted">Failed</span><strong style={{ color: 'var(--danger)' }}>{counts.failed}</strong></div>
          <div><span className="muted">Queued</span><strong>{counts.pending}</strong></div>
        </div>
      </div>

      <div className="grid-2">
        <div className="panel stack">
          <strong style={{ fontFamily: 'var(--display)' }}>Lead queue</strong>
          <div className="queue">
            {leads.map((l, i) => {
              const st = leadState(l)
              const meta = STATE_LABEL[st]
              return (
                <div key={i} className={`queue-row${currentRow === i ? ' active' : ''}`}>
                  <span className={`queue-index ${st}`}>{i + 1}</span>
                  <span className="queue-name">{l.website || l.company || `Lead ${i + 1}`}</span>
                  <span className={`pill ${meta.cls}`}>{meta.label}</span>
                </div>
              )
            })}
          </div>
        </div>

        <div className="panel tint-blue stack preview-panel">
          <div className="row" style={{ justifyContent: 'space-between' }}>
            <strong style={{ fontFamily: 'var(--display)' }}>Customer preview</strong>
            {previews.length > 1 ? (
              <div className="row" style={{ gap: 6 }}>
                <button
                  className="icon-btn"
                  disabled={selected <= 0}
                  onClick={() => setSelected((s) => Math.max(0, s - 1))}
                >
                  ‹
                </button>
                <span className="muted" style={{ fontSize: 12 }}>
                  {Math.min(selected + 1, previews.length)}/{previews.length}
                </span>
                <button
                  className="icon-btn"
                  disabled={selected >= previews.length - 1}
                  onClick={() => setSelected((s) => Math.min(previews.length - 1, s + 1))}
                >
                  ›
                </button>
              </div>
            ) : null}
          </div>

          {active ? (
            <>
              <span className="muted" style={{ fontSize: 12 }}>{active.website}</span>
              <EmailPreviewFrame html={active.html} subject={active.subject} />
            </>
          ) : (
            <div className="skeleton-frame">
              {running ? 'Writing the first email…' : 'Previews appear here as emails are generated.'}
            </div>
          )}
        </div>
      </div>

      <div className="panel stack" style={{ marginTop: '1rem' }}>
        <button className="log-toggle" onClick={() => setShowLog((v) => !v)}>
          <span>Technical log</span>
          <span className="muted">{showLog ? 'Hide ▲' : `Show ▼ (${logs.length})`}</span>
        </button>
        {showLog ? (
          <pre className="terminal">{logs.join('\n') || 'No log lines yet…'}</pre>
        ) : null}
      </div>
    </div>
  )
}
