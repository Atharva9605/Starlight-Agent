import { useEffect, useState } from 'react'
import {
  CheckmarkCircleFilled,
  CircleRegular,
  CopyRegular,
  DismissCircleFilled,
  DismissRegular,
  InfoRegular,
} from '@fluentui/react-icons'
import { api, type LeadLogEntry } from '../api/client'
import { MessageBar, Spinner } from './ui'

type LocalStage = { stage: string; label: string; state: string; at?: number }

function entryClass(e: LeadLogEntry): string {
  if (e.state === 'error' || e.state === 'failed') return 'error'
  if (e.state === 'done' || e.state === 'sent') return 'done'
  if (e.state === 'active' || e.state === 'processing') return 'active'
  return 'pending'
}

function EntryIcon({ cls, type }: { cls: string; type: string }) {
  if (cls === 'error') return <DismissCircleFilled className="step-item-icon" />
  if (cls === 'done') return <CheckmarkCircleFilled className="step-item-icon" />
  if (type === 'log' || type === 'lead_resolved') return <InfoRegular className="step-item-icon" />
  return <CircleRegular className="step-item-icon" />
}

function clock(at: string | null): string {
  if (!at) return ''
  const d = new Date(at)
  return Number.isNaN(d.getTime()) ? '' : d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
}

function duration(ms: number): string {
  const s = Math.max(0, Math.round(ms / 1000))
  if (s < 60) return `${s}s`
  const m = Math.floor(s / 60)
  return m < 60 ? `${m}m ${s % 60}s` : `${Math.floor(m / 60)}h ${m % 60}m`
}

/** Everything that happened to one lead: lookups, scraping, writing, edits, sends and errors. */
export function LeadLogDrawer({
  runId,
  rowIndex,
  lead,
  live,
  localStages,
  onClose,
}: {
  runId: string | null
  rowIndex: number
  lead: Record<string, any> | undefined
  /** Still being worked on — keep the log fresh. */
  live: boolean
  /** Shown when the run has no server record to read from. */
  localStages?: LocalStage[]
  onClose: () => void
}) {
  const [entries, setEntries] = useState<LeadLogEntry[] | null>(null)
  const [error, setError] = useState('')
  const [now, setNow] = useState(Date.now())
  const [copied, setCopied] = useState(false)

  useEffect(() => {
    setEntries(null)
    setError('')
    if (!runId) return
    let cancelled = false
    const load = () =>
      api
        .campaignRunLeadLog(runId, rowIndex)
        .then((res) => {
          if (!cancelled) setEntries(res.entries)
        })
        .catch((e) => {
          if (!cancelled) setError(e?.message || 'Could not load the log')
        })
    void load()
    const t = live ? window.setInterval(load, 4000) : undefined
    return () => {
      cancelled = true
      if (t) window.clearInterval(t)
    }
  }, [runId, rowIndex, live])

  useEffect(() => {
    if (!live) return
    const t = window.setInterval(() => setNow(Date.now()), 1000)
    return () => window.clearInterval(t)
  }, [live])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  const fallback: LeadLogEntry[] = (localStages || []).map((s) => ({
    at: s.at ? new Date(s.at).toISOString() : null,
    type: 'stage',
    text: s.label || s.stage,
    state: s.state,
  }))
  const list = runId ? entries || [] : fallback
  const loading = Boolean(runId) && entries === null && !error

  const title = lead?.company || lead?.name || lead?.website || `Lead ${rowIndex + 1}`
  const tried: string[] = Array.from(
    new Set([...(Array.isArray(lead?._exclude_websites) ? lead!._exclude_websites : []), lead?._failed_website].filter(Boolean)),
  )
  const lastAt = [...list].reverse().find((e) => e.at)?.at
  const quietFor = live && lastAt ? now - new Date(lastAt).getTime() : 0

  const copy = async () => {
    const text = [
      `${title} (lead ${rowIndex + 1})`,
      lead?.website ? `Website: ${lead.website}` : '',
      lead?._status ? `Status: ${lead._status}` : '',
      lead?._error ? `Error: ${lead._error}` : '',
      '',
      ...list.map((e) => `${clock(e.at)}  ${e.text}`),
    ]
      .filter((l, i) => l || i > 3)
      .join('\n')
    try {
      await navigator.clipboard.writeText(text)
      setCopied(true)
      window.setTimeout(() => setCopied(false), 1500)
    } catch {
      /* clipboard blocked — nothing to do */
    }
  }

  return (
    <div className="drawer-layer">
      <div className="drawer-backdrop" onClick={onClose} />
      <div className="drawer" role="dialog" aria-modal="true" aria-labelledby="lead-log-title">
        <div className="drawer-head">
          <div style={{ minWidth: 0 }}>
            <h2 id="lead-log-title" className="truncate">{title}</h2>
            <span className="muted text-sm">Lead {rowIndex + 1}{lead?.website ? ` · ${lead.website}` : ''}</span>
          </div>
          <button type="button" className="btn subtle icon-only" aria-label="Close" onClick={onClose}>
            <DismissRegular />
          </button>
        </div>
        <div className="drawer-body">
          <dl className="facts">
            <div>
              <dt>Status</dt>
              <dd>{String(lead?._status || 'Not started').replace(/^[^\p{L}\p{N}]+/u, '') || '—'}</dd>
            </div>
            {lead?._to ? (
              <div>
                <dt>Recipient</dt>
                <dd>{lead._to}</dd>
              </div>
            ) : null}
            {tried.length ? (
              <div>
                <dt>Websites tried</dt>
                <dd>{tried.join(', ')}</dd>
              </div>
            ) : null}
          </dl>

          {lead?._error ? <MessageBar intent="error" title="Last error">{lead._error}</MessageBar> : null}
          {error ? <MessageBar intent="warning" title="Couldn't load the full log">{error}</MessageBar> : null}

          {loading ? (
            <Spinner label="Loading the log…" />
          ) : list.length ? (
            <ol className="step-list lead-log">
              {list.map((e, i) => {
                const cls = entryClass(e)
                const prev = i > 0 ? list[i - 1].at : null
                const gap = prev && e.at ? new Date(e.at).getTime() - new Date(prev).getTime() : 0
                return (
                  <li key={i} className={`step-item ${cls}`}>
                    <EntryIcon cls={cls} type={e.type} />
                    <div style={{ minWidth: 0 }}>
                      <div className={e.type === 'log' ? 'lead-log-text' : 'step-item-title'}>{e.text}</div>
                      <div className="step-item-sub">
                        {clock(e.at)}
                        {gap >= 1000 ? ` · +${duration(gap)}` : ''}
                      </div>
                    </div>
                  </li>
                )
              })}
            </ol>
          ) : (
            <p className="muted">Nothing has happened to this lead yet.</p>
          )}

          {live && !loading ? (
            <div className="row muted text-sm" style={{ gap: 8 }}>
              <Spinner size="sm" />
              {quietFor ? `Still working · last update ${duration(quietFor)} ago` : 'Still working…'}
            </div>
          ) : null}
        </div>
        <div className="drawer-foot">
          <button type="button" className="btn secondary" onClick={() => void copy()} disabled={!list.length}>
            <CopyRegular /> {copied ? 'Copied' : 'Copy log'}
          </button>
          <span className="grow" />
          <button type="button" className="btn" onClick={onClose}>
            Close
          </button>
        </div>
      </div>
    </div>
  )
}
