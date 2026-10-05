import { useCallback, useEffect, useMemo, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { api, type CampaignRunSnapshot } from '../../api/client'
import { EmailPreviewFrame } from '../../components/EmailPreviewFrame'
import { useCampaign } from '../../campaign/CampaignContext'
import { formatWhen, runLabel } from './CampaignRunsPage'

const STATE_PILL: Record<string, string> = {
  sent: 'ok',
  failed: 'pink',
  skipped: '',
  processing: 'warn',
  pending: '',
}

/** The whole run, after the fact: every lead, what it sent, and the activity log. */
export function CampaignRunPage() {
  const { id = '' } = useParams()
  const nav = useNavigate()
  const { attachRun } = useCampaign()
  const [snap, setSnap] = useState<CampaignRunSnapshot | null>(null)
  const [error, setError] = useState('')
  const [selected, setSelected] = useState(0)
  const [busy, setBusy] = useState(false)
  // Bodies are not shipped with the snapshot; they are pulled per lead.
  const [bodies, setBodies] = useState<Record<number, string>>({})

  const load = useCallback(async () => {
    try {
      const res = await api.campaignRun(id)
      setSnap(res)
      setError('')
    } catch (e: any) {
      setError(e?.message || 'Could not load that run')
    }
  }, [id])

  useEffect(() => {
    void load()
  }, [load])

  // A run still sending is better watched live, but keep this view fresh anyway.
  useEffect(() => {
    if (snap?.run.status !== 'running') return
    const t = setInterval(() => void load(), 5000)
    return () => clearInterval(t)
  }, [snap?.run.status, load])

  const leads = snap?.leads || []
  const lead = leads[selected]
  const counts = snap?.run.counts
  const body = lead?._preview_html || bodies[selected] || ''

  useEffect(() => {
    if (!snap || !lead || body || !lead._has_preview) return
    let cancelled = false
    void api
      .campaignRunLeadHtml(id, selected)
      .then((res) => {
        if (!cancelled && res.html) setBodies((prev) => ({ ...prev, [selected]: res.html }))
      })
      .catch(() => undefined)
    return () => {
      cancelled = true
    }
  }, [id, selected, snap, lead, body])

  const sentLeads = useMemo(() => leads.filter((l) => l._state === 'sent'), [leads])

  const resume = async () => {
    setBusy(true)
    try {
      await api.resumeCampaignRun(id)
      await attachRun(id)
      nav('/campaigns/live')
    } catch (e: any) {
      setError(e?.message || 'Could not resume that run')
    } finally {
      setBusy(false)
    }
  }

  if (error && !snap) {
    return (
      <div className="dash-screen">
        <div className="panel stack">
          <strong>{error}</strong>
          <Link to="/campaigns/runs" className="btn secondary">
            Back to runs
          </Link>
        </div>
      </div>
    )
  }

  if (!snap) {
    return (
      <div className="dash-screen">
        <div className="panel stack">
          <p className="muted">Loading the run…</p>
        </div>
      </div>
    )
  }

  const run = snap.run
  // Failures count as resumable — a resume retries them along with the untouched.
  const canResume = run.counts.retriable > 0 && run.status !== 'running'

  return (
    <div className="dash-screen">
      <header className="dash-hero">
        <div>
          <div className="dash-kicker">
            {run.status === 'running' ? <span className="live-dot" /> : null}
            Full run
          </div>
          <h1>{runLabel(run)}</h1>
          <p className="muted">
            {formatWhen(run.created_at)}
            {run.finished_at ? ` → ${formatWhen(run.finished_at)}` : ''} · {run.counts.sent} sent ·{' '}
            {run.counts.failed} failed · {run.counts.skipped} skipped
            {run.counts.pending ? ` · ${run.counts.pending} never sent` : ''}
            {run.sender_email ? ` · from ${run.sender_email}` : ''}
          </p>
          {run.status === 'scheduled' ? (
            <p className="muted">Scheduled to start {formatWhen(run.scheduled_at)}</p>
          ) : null}
          {run.error ? <p className="muted">Run error: {run.error}</p> : null}
        </div>
        <div className="row">
          {run.status === 'running' ? (
            <button className="btn" type="button" onClick={() => void attachRun(id).then(() => nav('/campaigns/live'))}>
              Watch live
            </button>
          ) : canResume ? (
            <button className="btn" type="button" disabled={busy} onClick={() => void resume()}>
              {busy
                ? run.status === 'scheduled'
                  ? 'Starting…'
                  : 'Resuming…'
                : run.status === 'scheduled'
                  ? 'Start now'
                  : `Resume ${run.counts.retriable} left`}
            </button>
          ) : null}
          <Link to="/campaigns/runs" className="btn secondary">
            All runs
          </Link>
        </div>
      </header>

      <div className="stat-row dash-stats">
        <div className="stat blue">
          <div className="label">Sent</div>
          <div className="value">{counts?.sent ?? 0}</div>
        </div>
        <div className="stat amber">
          <div className="label">Failed</div>
          <div className="value">{counts?.failed ?? 0}</div>
        </div>
        <div className="stat cyan">
          <div className="label">Skipped</div>
          <div className="value">{counts?.skipped ?? 0}</div>
        </div>
        <div className="stat">
          <div className="label">Never sent</div>
          <div className="value">{counts?.pending ?? 0}</div>
        </div>
      </div>

      <div className="dash-grid">
        <section className="dash-preview panel stack">
          <div className="row" style={{ justifyContent: 'space-between', alignItems: 'flex-start' }}>
            <div>
              <div className="design-preview-label muted">Email that went out</div>
              <h2 style={{ margin: '0.2rem 0 0', fontFamily: 'var(--display)', fontSize: '1.25rem' }}>
                {lead?.company || lead?.name || lead?.website || `Lead ${selected + 1}`}
              </h2>
              <p className="muted" style={{ margin: '0.25rem 0 0', fontSize: 13 }}>
                {lead?.website || '—'}
                {lead?._to ? ` · to ${lead._to}` : ''}
              </p>
            </div>
            <span className={`pill ${STATE_PILL[lead?._state] ?? ''}`}>
              {lead?._status || lead?._state || 'pending'}
            </span>
          </div>

          {body ? (
            <div className="dash-preview-frame">
              <EmailPreviewFrame
                html={body}
                subject={lead._subject || 'Starlight outreach'}
                fromLabel={run.sender_email || 'Starlight Linear LED'}
                fullscreen
              />
            </div>
          ) : (
            <div className="skeleton-frame tall live-preview-empty">
              <strong>No email was written for this lead</strong>
              <p className="muted" style={{ margin: '0.4rem 0 0', maxWidth: '42ch' }}>
                {lead?._error || lead?._status || 'It was skipped or never reached.'}
              </p>
            </div>
          )}

          {lead?._stages?.length ? (
            <ul className="dash-detail-list">
              {lead._stages.map((s: any, i: number) => (
                <li key={i} className={`dash-detail ${s.state}`}>
                  <span className="dash-detail-mark" />
                  <div>
                    <div className="dash-detail-title">{s.label || s.stage}</div>
                    <div className="muted" style={{ fontSize: 12 }}>
                      {s.stage}
                    </div>
                  </div>
                </li>
              ))}
            </ul>
          ) : null}
        </section>

        <aside className="dash-side">
          <div className="panel stack">
            <strong style={{ fontFamily: 'var(--display)' }}>
              Leads ({sentLeads.length}/{leads.length} sent)
            </strong>
            <div className="queue dash-queue">
              {leads.map((l, i) => (
                <button
                  key={i}
                  type="button"
                  className={`queue-row queue-row-btn${i === selected ? ' active' : ''}`}
                  onClick={() => setSelected(i)}
                >
                  <span className={`queue-index ${l._state}`}>{i + 1}</span>
                  <span className="queue-name">
                    {l.company || l.name || l.website || `Lead ${i + 1}`}
                    <span className="muted" style={{ display: 'block', fontSize: 12, marginTop: 2 }}>
                      {l._to || l.website || '—'}
                    </span>
                  </span>
                  <span className={`pill ${STATE_PILL[l._state] ?? ''}`}>{l._state}</span>
                </button>
              ))}
            </div>
          </div>

          {snap.logs.length ? (
            <div className="panel stack">
              <strong style={{ fontFamily: 'var(--display)' }}>Activity</strong>
              <pre className="terminal">{snap.logs.join('\n')}</pre>
            </div>
          ) : null}
        </aside>
      </div>
    </div>
  )
}
