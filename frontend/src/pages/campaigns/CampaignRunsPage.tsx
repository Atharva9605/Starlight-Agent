import { useCallback, useEffect, useMemo, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import {
  AddRegular,
  DeleteRegular,
  DismissCircleRegular,
  HistoryRegular,
  OpenRegular,
  PlayRegular,
  ArrowEnterRegular,
  SearchRegular,
} from '@fluentui/react-icons'
import { api, type CampaignRun } from '../../api/client'
import { useCampaign } from '../../campaign/CampaignContext'
import { EmptyState, MessageBar, PageHeader, Spinner, useConfirm, useToast } from '../../components/ui'

const STATUS_BADGE: Record<string, { label: string; cls: string }> = {
  scheduled: { label: 'Scheduled', cls: 'purple' },
  running: { label: 'Sending', cls: 'brand' },
  paused: { label: 'Paused', cls: 'warning' },
  reviewing: { label: 'In review', cls: 'brand' },
  done: { label: 'Completed', cls: 'success' },
  stopped: { label: 'Stopped', cls: '' },
  failed: { label: 'Failed', cls: 'danger' },
}

export function runLabel(run: CampaignRun): string {
  return run.file_name || `${run.total} lead${run.total === 1 ? '' : 's'}`
}

export function formatWhen(value: string | null | undefined): string {
  if (!value) return '—'
  const d = new Date(value)
  if (Number.isNaN(d.getTime())) return '—'
  return d.toLocaleString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

type RunFilter = 'all' | 'active' | 'scheduled' | 'finished'

const FILTERS: { id: RunFilter; label: string; match: (r: CampaignRun) => boolean }[] = [
  { id: 'all', label: 'All', match: () => true },
  { id: 'active', label: 'Sending', match: (r) => r.status === 'running' || r.status === 'paused' || r.status === 'reviewing' },
  { id: 'scheduled', label: 'Scheduled', match: (r) => r.status === 'scheduled' },
  { id: 'finished', label: 'Finished', match: (r) => r.status === 'done' || r.status === 'stopped' || r.status === 'failed' },
]

/** Every campaign this org has run — live ones to rejoin, finished ones to read. */
export function CampaignRunsPage() {
  const nav = useNavigate()
  const toast = useToast()
  const confirm = useConfirm()
  const { attachRun } = useCampaign()
  const [runs, setRuns] = useState<CampaignRun[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [busyId, setBusyId] = useState('')
  const [filter, setFilter] = useState<RunFilter>('all')
  const [search, setSearch] = useState('')

  const visible = useMemo(() => {
    const f = FILTERS.find((x) => x.id === filter)!
    const term = search.trim().toLowerCase()
    return runs.filter(
      (r) => f.match(r) && (!term || `${r.file_name} ${r.sender_email} ${r.created_by_email}`.toLowerCase().includes(term)),
    )
  }, [runs, filter, search])

  const load = useCallback(async () => {
    try {
      const res = await api.campaignRuns()
      setRuns(res.runs || [])
      setError('')
    } catch (e: any) {
      setError(e?.message || 'Could not load campaign runs')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void load()
    // Keep the list honest while something is still sending.
    const t = setInterval(() => void load(), 10000)
    return () => clearInterval(t)
  }, [load])

  const rejoin = async (run: CampaignRun) => {
    setBusyId(run.id)
    try {
      await attachRun(run.id)
      nav('/campaigns/live')
    } catch (e: any) {
      setError(e?.message || 'Could not open that run')
    } finally {
      setBusyId('')
    }
  }

  const startNow = async (run: CampaignRun) => {
    setBusyId(run.id)
    try {
      await api.resumeCampaignRun(run.id)
      await attachRun(run.id)
      nav('/campaigns/live')
    } catch (e: any) {
      setError(e?.message || 'Could not start that run')
    } finally {
      setBusyId('')
    }
  }

  const cancel = async (run: CampaignRun) => {
    const ok = await confirm({
      title: 'Cancel this scheduled campaign?',
      body: `“${runLabel(run)}” won't be sent. This can't be undone.`,
      confirmLabel: 'Cancel campaign',
      cancelLabel: 'Keep it',
      danger: true,
    })
    if (!ok) return
    setBusyId(run.id)
    try {
      await api.stopCampaignRun(run.id)
      toast.info('Scheduled campaign cancelled')
      await load()
    } catch (e: any) {
      setError(e?.message || 'Could not cancel that run')
    } finally {
      setBusyId('')
    }
  }

  const remove = async (run: CampaignRun) => {
    const ok = await confirm({
      title: 'Delete this run?',
      body: `The history for “${runLabel(run)}” will be removed. Emails that were already sent are not affected.`,
      confirmLabel: 'Delete',
      danger: true,
    })
    if (!ok) return
    setBusyId(run.id)
    try {
      await api.deleteCampaignRun(run.id)
      setRuns((prev) => prev.filter((r) => r.id !== run.id))
      toast.info('Run deleted')
    } catch (e: any) {
      setError(e?.message || 'Could not delete that run')
    } finally {
      setBusyId('')
    }
  }

  return (
    <div>
      <PageHeader
        title="Campaign runs"
        subtitle="Every campaign runs on the server, so closing the tab never stops one. Rejoin a live run or open a finished one."
        actions={
          <Link to="/campaigns/new" className="btn">
            <AddRegular /> New campaign
          </Link>
        }
      />

      {error ? (
        <div className="page-alerts">
          <MessageBar intent="error" onDismiss={() => setError('')}>{error}</MessageBar>
        </div>
      ) : null}

      <section className="card flush">
        {loading ? (
          <div className="center-fill" style={{ minHeight: 200 }}>
            <Spinner label="Loading runs…" />
          </div>
        ) : !runs.length ? (
          <EmptyState
            icon={<HistoryRegular />}
            title="No campaigns yet"
            description="Runs appear here as soon as you start or schedule a campaign."
            actions={
              <Link to="/campaigns/new" className="btn">
                <AddRegular /> New campaign
              </Link>
            }
          />
        ) : (
          <>
          <div className="inbox-toolbar">
            <div className="tablist" role="tablist" aria-label="Filter runs">
              {FILTERS.map((f) => (
                <button
                  key={f.id}
                  type="button"
                  role="tab"
                  aria-selected={filter === f.id}
                  className={`tab${filter === f.id ? ' active' : ''}`}
                  onClick={() => setFilter(f.id)}
                >
                  {f.label}
                  <span className="tab-count">{runs.filter(f.match).length}</span>
                </button>
              ))}
            </div>
            <div className="input-wrap inbox-search">
              <SearchRegular />
              <input
                className="input"
                type="search"
                placeholder="Search by file or sender"
                aria-label="Search runs"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
              />
            </div>
          </div>
          {visible.length === 0 ? (
            <EmptyState compact icon={<SearchRegular />} title="No runs match" description="Try another tab or search." />
          ) : (
          <div className="table-wrap">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Campaign</th>
                  <th>Status</th>
                  <th className="hide-sm">Progress</th>
                  <th className="hide-sm">When</th>
                  <th className="actions"><span className="sr-only">Actions</span></th>
                </tr>
              </thead>
              <tbody>
                {visible.map((run) => {
                  const live = run.status === 'running' || run.status === 'paused'
                  const scheduled = run.status === 'scheduled'
                  const badge = STATUS_BADGE[run.status] || { label: run.status, cls: '' }
                  const pct = run.total ? Math.round((run.counts.processed / run.total) * 100) : 0
                  const rowBusy = busyId === run.id
                  return (
                    <tr key={run.id}>
                      <td>
                        <div style={{ minWidth: 0 }}>
                          {scheduled || live ? (
                            <strong className="truncate" style={{ display: 'block', maxWidth: 320 }}>{runLabel(run)}</strong>
                          ) : (
                            <Link to={`/campaigns/runs/${run.id}`} className="link" style={{ fontWeight: 600 }}>
                              {runLabel(run)}
                            </Link>
                          )}
                          <span className="cell-sub">
                            {run.total} lead{run.total === 1 ? '' : 's'}
                            {run.sender_email ? ` · from ${run.sender_email}` : ''}
                            {run.options?.review ? ' · reviewed' : ''}
                            {run.options?.recipient_override ? ' · test run' : ''}
                          </span>
                        </div>
                      </td>
                      <td>
                        <span className={`badge ${badge.cls}`}>
                          {run.status === 'running' ? <span className="live-dot" /> : null}
                          {badge.label}
                        </span>
                      </td>
                      <td className="hide-sm" style={{ minWidth: 180 }}>
                        {scheduled ? (
                          <span className="muted">Not started</span>
                        ) : (
                          <div className="stack tight">
                            <span className="text-sm">
                              {run.counts.sent} sent · {run.counts.failed} failed
                              {run.counts.pending ? ` · ${run.counts.pending} left` : ''}
                            </span>
                            <div className="progress">
                              <div className={`progress-fill${run.status === 'done' ? ' success' : ''}`} style={{ width: `${pct}%` }} />
                            </div>
                          </div>
                        )}
                      </td>
                      <td className="hide-sm muted" style={{ whiteSpace: 'nowrap' }}>
                        {scheduled ? `Sends ${formatWhen(run.scheduled_at)}` : formatWhen(run.created_at)}
                      </td>
                      <td className="actions">
                        <div className="row nowrap" style={{ justifyContent: 'flex-end', gap: 4 }}>
                          {scheduled ? (
                            <>
                              <button className="btn sm" type="button" disabled={rowBusy} onClick={() => void startNow(run)}>
                                <PlayRegular /> Start now
                              </button>
                              <button className="btn subtle sm" type="button" disabled={rowBusy} onClick={() => void cancel(run)}>
                                <DismissCircleRegular /> Cancel
                              </button>
                            </>
                          ) : live ? (
                            <button className="btn sm" type="button" disabled={rowBusy} onClick={() => void rejoin(run)}>
                              {rowBusy ? <Spinner size="sm" /> : <ArrowEnterRegular />}
                              {rowBusy ? 'Opening…' : 'Rejoin'}
                            </button>
                          ) : (
                            <>
                              <Link className="btn secondary sm" to={`/campaigns/runs/${run.id}`}>
                                <OpenRegular /> Open
                              </Link>
                              <button
                                className="btn subtle icon-only sm"
                                type="button"
                                aria-label={`Delete ${runLabel(run)}`}
                                title="Delete run"
                                disabled={rowBusy}
                                onClick={() => void remove(run)}
                              >
                                <DeleteRegular />
                              </button>
                            </>
                          )}
                        </div>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
          )}
          </>
        )}
      </section>
    </div>
  )
}
