import { useEffect, useRef, useState, type FormEvent } from 'react'
import { Link, Navigate, useNavigate } from 'react-router-dom'
import {
  AddRegular,
  ArrowClockwiseRegular,
  ArrowRightRegular,
  CheckmarkCircleFilled,
  CheckmarkRegular,
  CircleRegular,
  DismissCircleFilled,
  DocumentPdfRegular,
  ErrorCircleRegular,
  GlobeSearchRegular,
  MailInboxRegular,
  MailRegular,
  PauseRegular,
  PlayRegular,
  SendRegular,
  SkipForwardTabRegular,
  SparkleRegular,
  StopRegular,
  ClockRegular,
} from '@fluentui/react-icons'
import { EmailPreviewFrame } from '../../components/EmailPreviewFrame'
import { leadState, useCampaign, type StageEvent } from '../../campaign/CampaignContext'
import { CardHeader, MessageBar, PageHeader, Spinner, useConfirm } from '../../components/ui'

const STAGE_ORDER = ['queued', 'discover', 'scrape', 'analyze', 'retrieve', 'draft', 'render', 'send'] as const

const STAGE_LABELS: Record<string, string> = {
  queued: 'Queued',
  discover: 'Find website',
  scrape: 'Scrape',
  analyze: 'Analyze',
  retrieve: 'Catalogue',
  draft: 'Draft',
  render: 'Preview',
  send: 'Send',
  error: 'Failed',
}

function leadNeedsDiscover(lead: Record<string, any> | undefined, events: StageEvent[] | undefined): boolean {
  if (events?.some((e) => e.stage === 'discover')) return true
  if (lead?._discovered) return true
  return !String(lead?.website || '').trim() && Boolean(String(lead?.company || lead?.name || '').trim())
}

function mergeTimeline(
  events: StageEvent[] | undefined,
  opts?: { includeDiscover?: boolean },
): { id: string; label: string; state: string }[] {
  const byStage = new Map((events || []).map((e) => [e.stage, e]))
  const order = opts?.includeDiscover
    ? STAGE_ORDER
    : STAGE_ORDER.filter((id) => id !== 'discover')
  const rows: { id: string; label: string; state: string }[] = order.map((id) => {
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

function currentStageLabel(
  events: StageEvent[] | undefined,
  includeDiscover: boolean,
): string {
  const timeline = mergeTimeline(events, { includeDiscover })
  const active = timeline.find((s) => s.state === 'active')
  if (active) return active.label
  const err = timeline.find((s) => s.state === 'error')
  if (err) return err.label
  const done = [...timeline].reverse().find((s) => s.state === 'done')
  if (done) return done.label
  return STAGE_LABELS.queued
}

function leadTitle(lead: Record<string, any> | undefined, fallback: string): string {
  return lead?.company || lead?.name || lead?.website || fallback
}

function StageIcon({ state }: { state: string }) {
  if (state === 'done') return <CheckmarkCircleFilled className="step-item-icon" />
  if (state === 'error') return <DismissCircleFilled className="step-item-icon" />
  if (state === 'active') return <Spinner size="sm" />
  return <CircleRegular className="step-item-icon" />
}

function stateBadge(st: string, showDiscover: boolean) {
  if (st === 'sent') return 'success'
  if (st === 'failed') return 'danger'
  if (st === 'processing' || st === 'ready') return 'brand'
  if (showDiscover && st === 'pending') return 'teal'
  return ''
}

/** Live progress — full journey including website lookup, review & send. */
export function CampaignLivePage() {
  const nav = useNavigate()
  const confirm = useConfirm()
  const {
    leads,
    status,
    logs,
    counts,
    stop,
    pause,
    resume,
    reset,
    livePreview,
    stagesByLead,
    runSender,
    currentIndex,
    draft,
    chat,
    generating,
    revising,
    sending,
    sendCurrent,
    skipCurrent,
    reviseCurrent,
    autosend,
    runId,
    attaching,
    loadLeadPreview,
  } = useCampaign()

  const [input, setInput] = useState('')
  const [focusOverride, setFocusOverride] = useState<number | null>(null)
  const chatEnd = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    chatEnd.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
  }, [chat.length, revising])

  useEffect(() => {
    setFocusOverride(null)
  }, [draft?.rowIndex, livePreview?.rowIndex, currentIndex, status])

  const activeIdx = leads.findIndex((l) => {
    const s = leadState(l)
    return s === 'processing' || s === 'pending' || s === 'ready'
  })
  const focusIdx =
    focusOverride ??
    draft?.rowIndex ??
    livePreview?.rowIndex ??
    (activeIdx >= 0
      ? activeIdx
      : currentIndex >= 0
        ? currentIndex
        : Math.max(0, counts.processed - 1))

  // Email bodies live in the run record, not in state — fetch the one on screen.
  useEffect(() => {
    const lead = leads[focusIdx]
    if (runId && lead && !lead._preview_html && lead._has_preview) {
      void loadLeadPreview(focusIdx)
    }
  }, [focusIdx, leads, runId, loadLeadPreview])

  if (attaching) {
    return (
      <div className="center-fill" style={{ minHeight: '50vh' }}>
        <div className="empty-state">
          <Spinner size="lg" />
          <strong className="empty-title" style={{ marginTop: 12 }}>Rejoining your campaign…</strong>
          <p className="empty-desc">It has been running on the server the whole time. Picking up where it is now.</p>
        </div>
      </div>
    )
  }
  if (!leads.length) return <Navigate to="/campaigns" replace />
  if (status === 'reviewing' || (status === 'paused' && !autosend)) {
    return <Navigate to="/campaigns/review" replace />
  }

  const running = status === 'running'
  const paused = status === 'paused'
  const activeRun = running || generating
  const pct = counts.progressPct
  const busy = generating || revising || sending
  const left = Math.max(0, counts.total - counts.processed)

  const focus = leads[focusIdx]
  const focusState = focus ? leadState(focus) : 'pending'
  const includeDiscover = leadNeedsDiscover(focus, stagesByLead[focusIdx])
  const timeline = mergeTimeline(stagesByLead[focusIdx], { includeDiscover })
  const previewHtml = draft?.html || livePreview?.html || focus?._preview_html || ''
  const previewSubject = draft?.subject || livePreview?.subject || focus?._subject || 'Starlight outreach'
  const fromLabel = livePreview?.from || runSender || 'Starlight Linear LED'
  const productSheet = livePreview?.productSheet || focus?._product_sheet || ''
  const productCount = livePreview?.productCount ?? focus?._product_count ?? 0
  const canReview = !!draft && draft.rowIndex === focusIdx && focusState === 'ready'
  const showReview =
    canReview || (generating && draft?.rowIndex === focusIdx) || (generating && currentIndex === focusIdx)

  const onChat = async (e: FormEvent) => {
    e.preventDefault()
    if (!input.trim() || busy || !draft) return
    const msg = input
    setInput('')
    await reviseCurrent(msg)
  }

  const onStop = async () => {
    const ok = await confirm({
      title: 'Stop this campaign?',
      body: `Emails already sent stay sent. The ${left} lead${left === 1 ? '' : 's'} left won't be contacted unless you resume later from Campaign runs.`,
      confirmLabel: 'Stop campaign',
      danger: true,
    })
    if (ok) stop()
  }

  const finished = status === 'done' || status === 'stopped' || status === 'failed'
  // A resume picks up the untouched leads and retries the failures.
  const retriable = left + counts.failed

  const headline = (() => {
    if (running || generating) return `Working on lead ${focusIdx + 1} of ${counts.total}`
    if (paused) return 'Campaign paused'
    if (canReview) return `Review lead ${focusIdx + 1} of ${counts.total}`
    if (status === 'stopped') return 'Campaign stopped'
    if (status === 'done' && left > 0) return `${left} lead${left === 1 ? '' : 's'} still waiting`
    if (status === 'done') return 'Campaign complete'
    if (status === 'failed') return 'Campaign failed'
    return 'Live progress'
  })()

  const railSteps = timeline.filter((s) => s.id !== 'error')

  return (
    <div>
      <PageHeader
        breadcrumb={[{ label: 'Campaigns', to: '/campaigns' }, { label: 'Live progress' }]}
        kicker={
          activeRun ? (
            <><span className="live-dot" /> Live</>
          ) : paused ? (
            <span className="badge warning">Paused</span>
          ) : null
        }
        title={headline}
        subtitle={
          <>
            {counts.sent} sent · {counts.failed} failed · {counts.processing + counts.ready} in progress
            {left ? ` · ${left} left` : ''}
            {draft?.to ? ` · to ${draft.to}` : runSender ? ` · from ${runSender}` : ''}
          </>
        }
        actions={
          paused ? (
            <>
              <button className="btn danger-outline" type="button" onClick={onStop} disabled={sending}>
                <StopRegular /> Stop
              </button>
              <button className="btn" type="button" onClick={() => void resume()} disabled={sending}>
                <PlayRegular /> Resume
              </button>
            </>
          ) : activeRun || canReview ? (
            <>
              <button className="btn danger-outline" type="button" onClick={onStop} disabled={sending}>
                <StopRegular /> Stop
              </button>
              <button className="btn secondary" type="button" onClick={pause} disabled={sending}>
                <PauseRegular /> Pause
              </button>
            </>
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
                <AddRegular /> New campaign
              </button>
              {runId && finished ? (
                <Link to={`/campaigns/runs/${runId}`} className="btn">
                  See the whole run <ArrowRightRegular />
                </Link>
              ) : (
                <Link to="/inbox" className="btn">
                  <MailInboxRegular /> Inbox
                </Link>
              )}
            </>
          )
        }
      />

      <div className="page-alerts">
        {runId && activeRun ? (
          <MessageBar intent="info">Running on the server — you can close this tab and it keeps sending.</MessageBar>
        ) : null}
        {runId && finished ? (
          <MessageBar
            intent={status === 'done' && !retriable ? 'success' : status === 'failed' ? 'error' : 'warning'}
            title={
              status === 'done'
                ? `Campaign finished — ${counts.sent} email${counts.sent === 1 ? '' : 's'} sent`
                : status === 'stopped'
                  ? 'Campaign stopped'
                  : 'Campaign failed'
            }
            actions={
              <>
                {retriable ? (
                  <button className="btn secondary sm" type="button" onClick={() => void resume()}>
                    <ArrowClockwiseRegular /> Resume {retriable} left
                  </button>
                ) : null}
                <Link to="/campaigns/runs" className="btn subtle sm">All runs</Link>
              </>
            }
          >
            Every lead, the email it received and the activity log are kept.
            {retriable ? ` ${retriable} lead${retriable === 1 ? ' was' : 's were'} not sent.` : ''}
          </MessageBar>
        ) : null}
      </div>

      <section className="card" style={{ marginBottom: 20 }}>
        <div className="row between" style={{ marginBottom: 10 }}>
          <div className="row" style={{ gap: 10 }}>
            <strong style={{ fontSize: 20, lineHeight: '28px' }}>{pct}%</strong>
            <span className="muted text-sm">
              {counts.processed} of {counts.total} finished · {Math.max(0, counts.pending)} waiting
            </span>
          </div>
          {includeDiscover ? (
            <span className="badge teal"><GlobeSearchRegular /> Website lookup</span>
          ) : (
            <span className="badge">Website provided</span>
          )}
        </div>
        <div className={`progress thick${activeRun && pct === 0 ? ' indeterminate' : ''}`}>
          <div className={`progress-fill${status === 'done' ? ' success' : ''}`} style={activeRun && pct === 0 ? undefined : { width: `${pct}%` }} />
        </div>
        <ol className="stage-rail" style={{ marginTop: 20 }} aria-label={`Stages for ${leadTitle(focus, `lead ${focusIdx + 1}`)}`}>
          {railSteps.map((step) => (
            <li key={step.id} className={`stage ${step.state}`}>
              <span className="stage-dot">
                {step.state === 'done' ? <CheckmarkRegular /> : step.state === 'error' ? <DismissCircleFilled /> : null}
              </span>
              <span className="stage-label">{STAGE_LABELS[step.id] || step.id}</span>
            </li>
          ))}
        </ol>
      </section>

      <div className="dash-grid">
        <section className="card dash-preview">
          <CardHeader
            icon={<MailRegular />}
            title={leadTitle(draft || focus, `Lead ${focusIdx + 1}`)}
            subtitle={
              draft?.website || focus?.website
                ? `${draft?.website || focus?.website}${focus?._discovered || includeDiscover ? ' · found by website lookup' : ''}`
                : !focus?.website && (focus?.company || focus?.name) && generating
                  ? `Looking up the website for ${focus.company || focus.name}…`
                  : undefined
            }
            actions={
              <>
                {productSheet ? <span className="badge success"><DocumentPdfRegular /> PDF attached</span> : null}
                {productCount ? <span className="badge">{productCount} products</span> : null}
                <span className={`badge ${stateBadge(focusState, false) || 'warning'}`}>{focus?._status || 'Queued'}</span>
              </>
            }
          />

          {previewHtml ? (
            <div className="dash-preview-frame">
              <EmailPreviewFrame html={previewHtml} subject={previewSubject} fromLabel={fromLabel} fullscreen deviceToggle />
            </div>
          ) : (
            <div className="preview-empty">
              {generating || running ? <Spinner size="lg" /> : <MailRegular className="preview-empty-icon" />}
              <strong>
                {generating && includeDiscover
                  ? 'Finding the company website…'
                  : generating || running
                    ? 'Writing this email…'
                    : status === 'idle'
                      ? 'Start the campaign to begin'
                      : 'The preview appears as each email is written'}
              </strong>
              <p>
                With a website we scrape it directly; with only a company name we look the website up first, then
                continue the same journey.
              </p>
            </div>
          )}
        </section>

        <aside className="dash-side">
          {showReview ? (
            <section className="card">
              <CardHeader icon={<SparkleRegular />} title="Review & send" subtitle="Ask for changes, then send or skip." />
              <div className="stack" style={{ marginTop: 12 }}>
                {chat.length || revising ? (
                  <div className="chat-log" style={{ maxHeight: 180 }}>
                    {chat.map((m, i) => (
                      <div key={i} className={`chat-bubble ${m.role}`}>
                        {m.content}
                      </div>
                    ))}
                    {revising ? <div className="chat-bubble assistant"><Spinner size="sm" label="Updating…" /></div> : null}
                    <div ref={chatEnd} />
                  </div>
                ) : null}
                <form className="chat-compose" onSubmit={onChat}>
                  <input
                    className="input"
                    placeholder={draft ? 'e.g. Make the call to action softer' : 'Waiting for the draft…'}
                    aria-label="Ask for a change"
                    value={input}
                    disabled={!draft || busy}
                    onChange={(e) => setInput(e.target.value)}
                  />
                  <button
                    className="btn secondary icon-only"
                    type="submit"
                    aria-label="Update email"
                    title="Update email"
                    disabled={!draft || busy || !input.trim()}
                  >
                    <SparkleRegular />
                  </button>
                </form>
                <div className="row nowrap">
                  <button className="btn secondary grow" type="button" disabled={!draft || busy} onClick={() => void skipCurrent()}>
                    <SkipForwardTabRegular /> Skip
                  </button>
                  <button className="btn grow" type="button" disabled={!draft || busy} onClick={() => void sendCurrent()}>
                    {sending ? <Spinner size="sm" /> : <SendRegular />}
                    {sending ? 'Sending…' : 'Send'}
                  </button>
                </div>
              </div>
            </section>
          ) : null}

          <section className="card">
            <CardHeader title="This email" />
            <ul className="step-list" style={{ marginTop: 8 }}>
              {timeline.map((step) => (
                <li key={step.id} className={`step-item ${step.state}`}>
                  <StageIcon state={step.state} />
                  <div>
                    <div className="step-item-title">{step.label}</div>
                    <div className="step-item-sub">
                      {step.state === 'done'
                        ? 'Done'
                        : step.state === 'active'
                          ? step.id === 'send' && canReview
                            ? 'Ready — select Send'
                            : step.id === 'discover'
                              ? 'Searching the web…'
                              : 'Running now'
                          : step.state === 'error'
                            ? 'Error'
                            : 'Waiting'}
                    </div>
                  </div>
                </li>
              ))}
            </ul>
          </section>

          <section className="card">
            <CardHeader title="Lead queue" subtitle={`${counts.total} leads`} />
            <div className="kpi-grid compact" style={{ margin: '12px 0' }}>
              <div className="kpi">
                <span className="kpi-icon success"><SendRegular /></span>
                <div className="kpi-copy"><div className="kpi-label">Sent</div><div className="kpi-value">{counts.sent}</div></div>
              </div>
              <div className="kpi">
                <span className="kpi-icon danger"><ErrorCircleRegular /></span>
                <div className="kpi-copy"><div className="kpi-label">Failed</div><div className="kpi-value">{counts.failed}</div></div>
              </div>
              <div className="kpi">
                <span className="kpi-icon neutral"><ClockRegular /></span>
                <div className="kpi-copy"><div className="kpi-label">Left</div><div className="kpi-value">{left}</div></div>
              </div>
            </div>
            <div className="queue">
              {leads.map((l, i) => {
                const st = leadState(l)
                const showDiscover = leadNeedsDiscover(l, stagesByLead[i])
                const stage = currentStageLabel(stagesByLead[i], showDiscover)
                return (
                  <button
                    key={i}
                    type="button"
                    className={`queue-row${i === focusIdx ? ' active' : ''}`}
                    aria-current={i === focusIdx ? 'true' : undefined}
                    onClick={() => setFocusOverride(i)}
                  >
                    <span className={`queue-index ${st}`}>{i + 1}</span>
                    <span className="queue-name">
                      <strong>{leadTitle(l, `Lead ${i + 1}`)}</strong>
                      <span>
                        {l.website ? l.website.replace(/^https?:\/\//, '') : showDiscover ? 'Website lookup' : stage}
                        {l.website && stage ? ` · ${stage}` : !l.website && stage && showDiscover ? ` · ${stage}` : ''}
                      </span>
                    </span>
                    <span className={`badge ${stateBadge(st, showDiscover)}`}>
                      {st === 'pending' ? (showDiscover ? 'Lookup' : 'Queued') : l._status || st}
                    </span>
                  </button>
                )
              })}
            </div>
          </section>

          {logs.length ? (
            <section className="card">
              <CardHeader title="Activity" />
              <pre className="terminal" style={{ marginTop: 12 }}>{logs.slice(-30).join('\n')}</pre>
            </section>
          ) : null}
        </aside>
      </div>
    </div>
  )
}
