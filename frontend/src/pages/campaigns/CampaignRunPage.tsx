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
  ArrowDownloadRegular,
  TaskListSquareLtrRegular,
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
  reviewing: { label: 'In review', cls: 'brand' },
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

type LeadFilter = 'all' | 'sent' | 'failed' | 'other'

const LEAD_FILTERS: { id: LeadFilter; label: string; match: (l: Record<string, any>) => boolean }[] = [
  { id: 'all', label: 'All', match: () => true },
  { id: 'sent', label: 'Sent', match: (l) => l._state === 'sent' },
  { id: 'failed', label: 'Failed', match: (l) => l._state === 'failed' },
  { id: 'other', label: 'Not sent', match: (l) => l._state !== 'sent' && l._state !== 'failed' },
]

function csvCell(v: unknown): string {
  const t = String(v ?? '')
  return /[",\n]/.test(t) ? `"${t.replace(/"/g, '""')}"` : t
}

/** Results as a spreadsheet, so sales can follow up outside the app. */
function downloadCsv(name: string, leads: Record<string, any>[]) {
  const header = ['#', 'Company', 'Website', 'Sent to', 'Status', 'Subject', 'Error']
  const rows = leads.map((l, i) => [i + 1, l.company || l.name || '', l.website || '', l._to || '', l._state || '', l._subject || '', l._error || ''])
  const csv = [header, ...rows].map((r) => r.map(csvCell).join(',')).join('\r\n')
  const url = URL.createObjectURL(new Blob(['﻿' + csv], { type: 'text/csv;charset=utf-8' }))
  const a = document.createElement('a')
  a.href = url
  a.download = `${name.replace(/\.[a-z0-9]+$/i, '') || 'campaign'}-results.csv`
  a.click()
  URL.revokeObjectURL(url)
}

/** The whole run, after the fact: every lead, what it sent, and the activity log. */
export function CampaignRunPage() {
  const { id = '' } = useParams()
  const nav = useNavigate()
  const { attachRun, attachReviewRun } = useCampaign()
  const [snap, setSnap] = useState<CampaignRunSnapshot | null>(null)
  const [error, setError] = useState('')
  const [selected, setSelected] = useState(0)
  const [busy, setBusy] = useState(false)
  const [leadFilter, setLeadFilter] = useState<LeadFilter>('all')
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
    if (snap?.run.status !== 'running' && snap?.run.status !== 'reviewing') return
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

  const continueReview = async () => {
    setBusy(true)
    try {
      await attachReviewRun(id)
      nav('/campaigns/review')
    } catch (e: any) {
      setError(e?.message || 'Could not reopen this review')
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
  const canResume =
    run.counts.retriable > 0 && run.status !== 'running' && run.status !== 'reviewing' && !run.options?.review
  const canReview = Boolean(run.options?.review) && (run.status === 'reviewing' || run.counts.retriable > 0)
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
          <>
          {leads.length ? (
            <button className="btn secondary" type="button" onClick={() => downloadCsv(runLabel(run), leads)}>
              <ArrowDownloadRegular /> Export CSV
            </button>
          ) : null}
          {run.status === 'running' ? (
            <button className="btn" type="button" onClick={() => void attachRun(id).then(() => nav('/campaigns/live'))}>
              <EyeRegular /> Watch live
            </button>
          ) : canReview ? (
            <button className="btn" type="button" disabled={busy} onClick={() => void continueReview()}>
              {busy ? <Spinner size="sm" /> : <TaskListSquareLtrRegular />}
              {busy ? 'Opening…' : 'Continue review'}
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
          ) : null}
          </>
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
            <div className="tablist compact" role="tablist" aria-label="Filter leads" style={{ marginTop: 8 }}>
              {LEAD_FILTERS.map((f) => (
                <button
                  key={f.id}
                  type="button"
                  role="tab"
                  aria-selected={leadFilter === f.id}
                  className={`tab${leadFilter === f.id ? ' active' : ''}`}
                  onClick={() => setLeadFilter(f.id)}
                >
                  {f.label}
                  <span className={`tab-count${f.id === 'failed' && leads.some(f.match) ? ' danger' : ''}`}>{leads.filter(f.match).length}</span>
                </button>
              ))}
            </div>
            <div className="queue" style={{ marginTop: 12 }}>
              {leads.map((l, i) => !LEAD_FILTERS.find((f) => f.id === leadFilter)!.match(l) ? null : (
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
                    <span className={l._state === 'failed' && l._error ? 'text-danger' : undefined}>
                      {l._state === 'failed' && l._error ? l._error : l._to || l.website || '—'}
                    </span>
                  </span>
                  <span className={`badge ${STATE_BADGE[l._state] ?? ''}`}>
                    {l._state === 'processing' && /ready for review/i.test(l._status || '') ? 'ready' : l._state}
                  </span>
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
