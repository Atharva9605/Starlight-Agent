import { useCallback, useEffect, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { api, type CampaignRun } from '../../api/client'
import { useCampaign } from '../../campaign/CampaignContext'

const STATUS_PILL: Record<string, string> = {
  running: 'warn',
  paused: 'warn',
  done: 'ok',
  stopped: '',
  failed: 'pink',
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

/** Every campaign this org has run — live ones to rejoin, finished ones to read. */
export function CampaignRunsPage() {
  const nav = useNavigate()
  const { attachRun } = useCampaign()
  const [runs, setRuns] = useState<CampaignRun[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [busyId, setBusyId] = useState('')

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

  const remove = async (run: CampaignRun) => {
    setBusyId(run.id)
    try {
      await api.deleteCampaignRun(run.id)
      setRuns((prev) => prev.filter((r) => r.id !== run.id))
    } catch (e: any) {
      setError(e?.message || 'Could not delete that run')
    } finally {
      setBusyId('')
    }
  }

  return (
    <div className="dash-screen">
      <header className="dash-hero">
        <div>
          <div className="dash-kicker">Campaign history</div>
          <h1>Runs</h1>
          <p className="muted">
            Every campaign runs on the server, so closing the tab never stops one. Rejoin a live
            run or open a finished one to read the whole thing.
          </p>
        </div>
        <Link to="/campaigns" className="btn">
          New campaign
        </Link>
      </header>

      {error ? <div className="panel stack notice pink">{error}</div> : null}

      <div className="panel stack">
        {loading ? (
          <p className="muted">Loading runs…</p>
        ) : !runs.length ? (
          <p className="muted">No campaigns yet. Start one from Campaigns.</p>
        ) : (
          <div className="queue">
            {runs.map((run) => {
              const live = run.status === 'running' || run.status === 'paused'
              return (
                <div key={run.id} className="queue-row">
                  <span className={`queue-index ${run.status === 'done' ? 'sent' : run.status}`}>
                    {run.counts.sent}
                  </span>
                  <span className="queue-name">
                    {runLabel(run)}
                    <span className="muted" style={{ display: 'block', fontSize: 12, marginTop: 2 }}>
                      {formatWhen(run.created_at)} · {run.counts.sent} sent · {run.counts.failed} failed
                      {run.counts.pending ? ` · ${run.counts.pending} left` : ''}
                      {run.sender_email ? ` · from ${run.sender_email}` : ''}
                    </span>
                  </span>
                  <span className={`pill ${STATUS_PILL[run.status] ?? ''}`}>
                    {run.status === 'running' ? 'Sending' : run.status}
                  </span>
                  <span className="row" style={{ gap: 6 }}>
                    {live ? (
                      <button
                        className="btn"
                        type="button"
                        disabled={busyId === run.id}
                        onClick={() => void rejoin(run)}
                      >
                        {busyId === run.id ? 'Opening…' : 'Rejoin'}
                      </button>
                    ) : (
                      <Link className="btn secondary" to={`/campaigns/runs/${run.id}`}>
                        View full run
                      </Link>
                    )}
                    {!live ? (
                      <button
                        className="btn secondary"
                        type="button"
                        disabled={busyId === run.id}
                        onClick={() => void remove(run)}
                      >
                        Delete
                      </button>
                    ) : null}
                  </span>
                </div>
              )
            })}
          </div>
        )}
      </div>
    </div>
  )
}
