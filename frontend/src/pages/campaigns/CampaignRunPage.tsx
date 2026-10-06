import { useCallback, useEffect, useMemo, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import {
  ArrowLeftRegular,
  CheckmarkCircleFilled,
  CircleRegular,
  DismissCircleFilled,
  ErrorCircleRegular,
  EyeRegular,
  MailOffRegular,
  PlayRegular,
  SendRegular,
  SkipForwardTabRegular,
  ClockRegular,
} from '@fluentui/react-icons'
import { api, type CampaignRunSnapshot } from '../../api/client'
import { EmailPreviewFrame } from '../../components/EmailPreviewFrame'
import { useCampaign } from '../../campaign/CampaignContext'
import { CardHeader, EmptyState, MessageBar, PageHeader, Spinner } from '../../components/ui'
import { formatWhen, runLabel } from './CampaignRunsPage'

const STATE_BADGE: Record<string, string> = {
  sent: 'success',
  failed: 'danger',
  skipped: '',
  processing: 'brand',
  pending: '',
}

const RUN_BADGE: Record<string, { label: string; cls: string }> = {
  scheduled: { label: 'Scheduled', cls: 'purple' },
  running: { label: 'Sending', cls: 'brand' },
  paused: { label: 'Paused', cls: 'warning' },
  done: { label: 'Completed', cls: 'success' },
  stopped: { label: 'Stopped', cls: '' },
  failed: { label: 'Failed', cls: 'danger' },
}

function StageIcon({ state }: { state: string }) {
  if (state === 'done') return <CheckmarkCircleFilled className="step-item-icon" />
  if (state === 'error') return <DismissCircleFilled className="step-item-icon" />
  if (state === 'active') return <Spinner size="sm" />
  return <CircleRegular className="step-item-icon" />
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

  const crumbs = [{ label: 'Campaign runs', to: '/campaigns/runs' }]

  if (error && !snap) {
    return (
      <div>
        <PageHeader breadcrumb={[...crumbs, { label: 'Run' }]} title="Run not available" />
        <section className="card">
          <EmptyState
            icon={<ErrorCircleRegular />}
            title="We couldn't load this run"
            description={error}
            actions={
              <Link to="/campaigns/runs" className="btn secondary">
                <ArrowLeftRegular /> Back to runs
              </Link>
            }
          />
        </section>
      </div>
    )
  }

  if (!snap) {
    return (
      <div className="center-fill" style={{ minHeight: '50vh' }}>
        <Spinner size="lg" label="Loading the run…" />
      </div>
    )
  }

  const run = snap.run
  // Failures count as resumable — a resume retries them along with the untouched.
  const canResume = run.counts.retriable > 0 && run.status !== 'running'
  const runBadge = RUN_BADGE[run.status] || { label: run.status, cls: '' }

  return (
    <div>
      <PageHeader
        breadcrumb={[...crumbs, { label: runLabel(run) }]}
        title={runLabel(run)}
        subtitle={
          <>
            {formatWhen(run.created_at)}
            {run.finished_at ? ` – ${formatWhen(run.finished_at)}` : ''}
            {run.sender_email ? ` · from ${run.sender_email}` : ''}
          </>
        }
        kicker={
          <span className={`badge ${runBadge.cls}`}>
            {run.status === 'running' ? <span className="live-dot" /> : null}
            {runBadge.label}
          </span>
        }
        actions={
          run.status === 'running' ? (
            <button className="btn" type="button" onClick={() => void attachRun(id).then(() => nav('/campaigns/live'))}>
              <EyeRegular /> Watch live
            </button>
          ) : canResume ? (
            <button className="btn" type="button" disabled={busy} onClick={() => void resume()}>
              {busy ? <Spinner size="sm" /> : <PlayRegular />}
              {busy
                ? run.status === 'scheduled'
                  ? 'Starting…'
                  : 'Resuming…'
                : run.status === 'scheduled'
                  ? 'Start now'
                  : `Resume ${run.counts.retriable} left`}
            </button>
          ) : null
        }
      />

      <div className="page-alerts">
        {run.status === 'scheduled' ? (
          <MessageBar intent="info" title="Scheduled">Starts {formatWhen(run.scheduled_at)}.</MessageBar>
        ) : null}
        {run.error ? <MessageBar intent="error" title="Run error">{run.error}</MessageBar> : null}
        {error ? <MessageBar intent="error" onDismiss={() => setError('')}>{error}</MessageBar> : null}
      </div>

      <div className="kpi-grid">
        <div className="kpi">
          <span className="kpi-icon success"><SendRegular /></span>
          <div className="kpi-copy">
            <div className="kpi-label">Sent</div>
            <div className="kpi-value">{counts?.sent ?? 0}</div>
          </div>
        </div>
        <div className="kpi">
          <span className="kpi-icon danger"><ErrorCircleRegular /></span>
          <div className="kpi-copy">
            <div className="kpi-label">Failed</div>
            <div className="kpi-value">{counts?.failed ?? 0}</div>
          </div>
        </div>
        <div className="kpi">
          <span className="kpi-icon neutral"><SkipForwardTabRegular /></span>
          <div className="kpi-copy">
            <div className="kpi-label">Skipped</div>
            <div className="kpi-value">{counts?.skipped ?? 0}</div>
          </div>
        </div>
        <div className="kpi">
          <span className="kpi-icon warning"><ClockRegular /></span>
          <div className="kpi-copy">
            <div className="kpi-label">Never sent</div>
            <div className="kpi-value">{counts?.pending ?? 0}</div>
          </div>
        </div>
      </div>

      <div className="dash-grid">
        <section className="card dash-preview">
          <CardHeader
            title={lead?.company || lead?.name || lead?.website || `Lead ${selected + 1}`}
            subtitle={
              <>
                {lead?.website || '—'}
                {lead?._to ? ` · to ${lead._to}` : ''}
              </>
            }
            actions={
              <span className={`badge ${STATE_BADGE[lead?._state] ?? ''}`}>
                {lead?._status || lead?._state || 'pending'}
              </span>
            }
          />

          {body ? (
            <div className="dash-preview-frame">
              <EmailPreviewFrame
                html={body}
                subject={lead._subject || 'Starlight outreach'}
                fromLabel={run.sender_email || 'Starlight Linear LED'}
                fullscreen
                deviceToggle
              />
            </div>
          ) : (
            <div className="preview-empty">
              <MailOffRegular className="preview-empty-icon" />
              <strong>No email was written for this lead</strong>
              <p>{lead?._error || lead?._status || 'It was skipped or never reached.'}</p>
            </div>
          )}

          {lead?._stages?.length ? (
            <ul className="step-list">
              {lead._stages.map((s: any, i: number) => (
                <li key={i} className={`step-item ${s.state}`}>
                  <StageIcon state={s.state} />
                  <div>
                    <div className="step-item-title">{s.label || s.stage}</div>
                    <div className="step-item-sub">{s.stage}</div>
                  </div>
                </li>
              ))}
            </ul>
          ) : null}
        </section>

        <aside className="dash-side">
          <section className="card">
            <CardHeader title="Leads" subtitle={`${sentLeads.length} of ${leads.length} sent`} />
            <div className="queue" style={{ marginTop: 12 }}>
              {leads.map((l, i) => (
                <button
                  key={i}
                  type="button"
                  className={`queue-row${i === selected ? ' active' : ''}`}
                  aria-current={i === selected ? 'true' : undefined}
                  onClick={() => setSelected(i)}
                >
                  <span className={`queue-index ${l._state}`}>{i + 1}</span>
                  <span className="queue-name">
                    <strong>{l.company || l.name || l.website || `Lead ${i + 1}`}</strong>
                    <span>{l._to || l.website || '—'}</span>
                  </span>
                  <span className={`badge ${STATE_BADGE[l._state] ?? ''}`}>{l._state}</span>
                </button>
              ))}
            </div>
          </section>

          {snap.logs.length ? (
            <section className="card">
              <CardHeader title="Activity" />
              <pre className="terminal" style={{ marginTop: 12 }}>{snap.logs.join('\n')}</pre>
            </section>
          ) : null}
        </aside>
      </div>
    </div>
  )
}
