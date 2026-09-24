import { useEffect, useRef, useState, type FormEvent } from 'react'
import { Link, Navigate, useNavigate } from 'react-router-dom'
import { EmailPreviewFrame } from '../../components/EmailPreviewFrame'
import { leadState, useCampaign, type StageEvent } from '../../campaign/CampaignContext'

const STAGE_ORDER = ['queued', 'discover', 'scrape', 'analyze', 'retrieve', 'draft', 'render', 'send'] as const

const STAGE_LABELS: Record<string, string> = {
  queued: 'Queued',
  discover: 'OpenSERP',
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

/** Live progress logs — full journey including OpenSERP, review & send. */
export function CampaignLivePage() {
  const nav = useNavigate()
  const {
    leads,
    status,
    logs,
    counts,
    stop,
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
  } = useCampaign()

  const [input, setInput] = useState('')
  const [focusOverride, setFocusOverride] = useState<number | null>(null)
  const chatEnd = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    chatEnd.current?.scrollIntoView({ behavior: 'smooth' })
  }, [chat.length, revising])

  useEffect(() => {
    setFocusOverride(null)
  }, [draft?.rowIndex, livePreview?.rowIndex, currentIndex, status])

  if (!leads.length) return <Navigate to="/campaigns" replace />

  const running = status === 'running'
  const reviewing = status === 'reviewing'
  const activeRun = running || reviewing || generating
  const pct = counts.progressPct
  const busy = generating || revising || sending
  const left = Math.max(0, counts.total - counts.processed)

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
  const focus = leads[focusIdx]
  const focusState = focus ? leadState(focus) : 'pending'
  const includeDiscover = leadNeedsDiscover(focus, stagesByLead[focusIdx])
  const timeline = mergeTimeline(stagesByLead[focusIdx], { includeDiscover })
  const previewHtml = draft?.html || livePreview?.html || focus?._preview_html || ''
  const previewSubject = draft?.subject || livePreview?.subject || focus?._subject || 'Starlight outreach'
  const fromLabel = livePreview?.from || runSender || 'Starlight Linear LED'
  const productSheet = livePreview?.productSheet || focus?._product_sheet || ''
  const productCount = livePreview?.productCount ?? focus?._product_count ?? 0
  const activeStep = timeline.find((s) => s.state === 'active') || timeline.find((s) => s.state === 'pending')
  const canReview = Boolean(draft) && draft!.rowIndex === focusIdx && (reviewing || focusState === 'ready')
  const pathHint = includeDiscover
    ? 'Company → OpenSERP → scrape → draft → send'
    : 'Website → scrape → draft → send'

  const onChat = async (e: FormEvent) => {
    e.preventDefault()
    if (!input.trim() || busy || !draft) return
    const msg = input
    setInput('')
    await reviseCurrent(msg)
  }

  const headline = (() => {
    if (running || generating) return `Working lead ${focusIdx + 1} of ${counts.total}`
    if (canReview) return `Review lead ${focusIdx + 1} of ${counts.total}`
    if (status === 'stopped') return 'Campaign stopped'
    if (status === 'done' && left > 0) return `${left} lead${left === 1 ? '' : 's'} still waiting`
    if (status === 'done') return 'Campaign complete'
    if (status === 'failed') return 'Campaign failed'
    return 'Live progress logs'
  })()

  return (
    <div className="dash-screen">
      <header className="dash-hero">
        <div>
          <div className="dash-kicker">
            {activeRun ? <span className="live-dot" /> : null}
            Live progress logs
          </div>
          <h1>{headline}</h1>
          <p className="muted">
            {counts.sent} sent · {counts.failed} failed · {counts.processing + counts.ready} in progress
            {left ? ` · ${left} left` : ''}
            {draft?.to ? ` · to ${draft.to}` : runSender ? ` · from ${runSender}` : ''}
          </p>
        </div>
        <div className="row">
          {activeRun || canReview ? (
            <button className="btn danger" type="button" onClick={stop} disabled={sending}>
              Stop
            </button>
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
              <Link to="/inbox" className="btn">
                Inbox
              </Link>
            </>
          )}
        </div>
      </header>

      <div className="dash-progress panel">
        <div className="row" style={{ justifyContent: 'space-between', marginBottom: 10, gap: 12 }}>
          <div>
            <strong style={{ fontSize: 15 }}>{pct}% complete</strong>
            <span className="muted" style={{ marginLeft: 10, fontSize: 13 }}>
              {counts.processed} finished · {Math.max(0, counts.pending)} waiting
            </span>
          </div>
          <div className="row" style={{ gap: 6 }}>
            {includeDiscover ? <span className="pill discover">OpenSERP path</span> : <span className="pill">Website path</span>}
            {activeStep ? <span className="pill warn">{activeStep.label}</span> : null}
          </div>
        </div>
        <div className="progress thick">
          <div className={`progress-fill${activeRun ? ' animated' : ''}`} style={{ width: `${pct}%` }} />
        </div>
        <p className="muted" style={{ margin: '0.65rem 0 0', fontSize: 12 }}>
          Journey for this lead: {pathHint}
        </p>
        <ol className="dash-rail" style={{ ['--rail-cols' as string]: timeline.filter((s) => s.id !== 'error').length }}>
          {timeline
            .filter((s) => s.id !== 'error')
            .map((step) => (
              <li key={step.id} className={`dash-rail-step ${step.state}${step.id === 'discover' ? ' discover' : ''}`}>
                <span className="dash-rail-dot" />
                <span className="dash-rail-label">{STAGE_LABELS[step.id] || step.id}</span>
              </li>
            ))}
        </ol>
      </div>

      <div className="dash-grid">
        <section className="dash-preview panel stack">
          <div className="row" style={{ justifyContent: 'space-between', alignItems: 'flex-start' }}>
            <div>
              <div className="design-preview-label muted">Live email preview</div>
              <h2 style={{ margin: '0.2rem 0 0', fontFamily: 'var(--display)', fontSize: '1.25rem' }}>
                {leadTitle(draft || focus, `Lead ${focusIdx + 1}`)}
              </h2>
              {(draft?.website || focus?.website) && (
                <p className="muted" style={{ margin: '0.25rem 0 0', fontSize: 13 }}>
                  {draft?.website || focus?.website}
                  {focus?._discovered || includeDiscover ? (
                    <span className="pill discover" style={{ marginLeft: 8, verticalAlign: 'middle' }}>
                      via OpenSERP
                    </span>
                  ) : null}
                </p>
              )}
              {!focus?.website && (focus?.company || focus?.name) && generating ? (
                <p className="muted" style={{ margin: '0.25rem 0 0', fontSize: 13 }}>
                  Looking up website for <strong>{focus.company || focus.name}</strong>…
                </p>
              ) : null}
            </div>
            <div className="row" style={{ gap: 6 }}>
              {productSheet ? <span className="pill ok">PDF sheet attached</span> : null}
              {productCount ? <span className="pill">{productCount} products</span> : null}
              <span
                className={`pill ${
                  focusState === 'sent' ? 'ok' : focusState === 'failed' ? 'pink' : 'warn'
                }`}
              >
                {focus?._status || 'Queued'}
              </span>
            </div>
          </div>

          {previewHtml ? (
            <div className="dash-preview-frame">
              <EmailPreviewFrame
                html={previewHtml}
                subject={previewSubject}
                fromLabel={fromLabel}
                fullscreen
              />
            </div>
          ) : (
            <div className="skeleton-frame tall live-preview-empty">
              <div className="live-pulse-ring" />
              <strong>
                {generating && includeDiscover
                  ? 'OpenSERP is finding the company website…'
                  : generating || running
                    ? 'Generating this email…'
                    : status === 'idle'
                      ? 'Start the campaign from Campaigns to begin'
                      : 'Preview appears here as each email is written'}
              </strong>
              <p className="muted" style={{ margin: '0.4rem 0 0', maxWidth: '42ch' }}>
                If the sheet has a website we scrape it; if only a company name is present we resolve it with OpenSERP, then continue the same journey.
              </p>
            </div>
          )}
        </section>

        <aside className="dash-side">
          {canReview || (generating && draft?.rowIndex === focusIdx) || (generating && currentIndex === focusIdx) ? (
            <div className="panel stack">
              <strong style={{ fontFamily: 'var(--display)' }}>Review & send</strong>
              <div className="chat-log" style={{ maxHeight: 160 }}>
                {chat.map((m, i) => (
                  <div key={i} className={`chat-bubble ${m.role}`}>
                    {m.content}
                  </div>
                ))}
                {revising ? <div className="chat-bubble assistant">Updating…</div> : null}
                <div ref={chatEnd} />
              </div>
              <form className="chat-compose" onSubmit={onChat}>
                <input
                  className="input"
                  placeholder={draft ? 'e.g. Make the CTA softer…' : 'Waiting for draft…'}
                  value={input}
                  disabled={!draft || busy}
                  onChange={(e) => setInput(e.target.value)}
                />
                <button className="btn secondary" type="submit" disabled={!draft || busy || !input.trim()}>
                  Update
                </button>
              </form>
              <div className="row" style={{ gap: 8 }}>
                <button
                  className="btn secondary"
                  type="button"
                  style={{ flex: 1 }}
                  disabled={!draft || busy}
                  onClick={() => void skipCurrent()}
                >
                  Skip
                </button>
                <button
                  className="btn"
                  type="button"
                  style={{ flex: 1.4 }}
                  disabled={!draft || busy}
                  onClick={() => void sendCurrent()}
                >
                  {sending ? 'Sending…' : 'Send'}
                </button>
              </div>
            </div>
          ) : null}

          <div className="panel stack">
            <strong style={{ fontFamily: 'var(--display)' }}>This email</strong>
            <ul className="dash-detail-list">
              {timeline.map((step) => (
                <li key={step.id} className={`dash-detail ${step.state}${step.id === 'discover' ? ' discover' : ''}`}>
                  <span className="dash-detail-mark" />
                  <div>
                    <div className="dash-detail-title">{step.label}</div>
                    <div className="muted" style={{ fontSize: 12 }}>
                      {step.state === 'done'
                        ? 'Done'
                        : step.state === 'active'
                          ? step.id === 'send' && canReview
                            ? 'Ready — hit Send'
                            : step.id === 'discover'
                              ? 'Searching OpenSERP…'
                              : 'Running now'
                          : step.state === 'error'
                            ? 'Error'
                            : 'Waiting'}
                    </div>
                  </div>
                </li>
              ))}
            </ul>
          </div>

          <div className="panel stack">
            <strong style={{ fontFamily: 'var(--display)' }}>Lead queue</strong>
            <div className="queue dash-queue">
              {leads.map((l, i) => {
                const st = leadState(l)
                const showDiscover = leadNeedsDiscover(l, stagesByLead[i])
                const stage = currentStageLabel(stagesByLead[i], showDiscover)
                return (
                  <button
                    key={i}
                    type="button"
                    className={`queue-row queue-row-btn${i === focusIdx ? ' active' : ''}`}
                    onClick={() => setFocusOverride(i)}
                  >
                    <span className={`queue-index ${st}`}>{i + 1}</span>
                    <span className="queue-name">
                      {leadTitle(l, `Lead ${i + 1}`)}
                      <span className="muted" style={{ display: 'block', fontSize: 12, marginTop: 2 }}>
                        {l.website
                          ? l.website.replace(/^https?:\/\//, '')
                          : showDiscover
                            ? 'OpenSERP lookup'
                            : stage}
                        {l.website && stage ? ` · ${stage}` : !l.website && stage && showDiscover ? ` · ${stage}` : ''}
                      </span>
                    </span>
                    <span
                      className={`pill ${
                        st === 'sent'
                          ? 'ok'
                          : st === 'failed'
                            ? 'pink'
                            : st === 'processing' || st === 'ready'
                              ? 'warn'
                              : showDiscover && st === 'pending'
                                ? 'discover'
                                : ''
                      }`}
                    >
                      {st === 'pending' ? (showDiscover ? 'OpenSERP' : 'Queued') : l._status || st}
                    </span>
                  </button>
                )
              })}
            </div>
          </div>

          <div className="stat-row dash-stats">
            <div className="stat blue">
              <div className="label">Sent</div>
              <div className="value">{counts.sent}</div>
            </div>
            <div className="stat amber">
              <div className="label">Failed</div>
              <div className="value">{counts.failed}</div>
            </div>
            <div className="stat cyan">
              <div className="label">Left</div>
              <div className="value">{left}</div>
            </div>
          </div>

          {logs.length ? (
            <div className="panel stack">
              <strong style={{ fontFamily: 'var(--display)' }}>Activity</strong>
              <pre className="terminal">{logs.slice(-30).join('\n')}</pre>
            </div>
          ) : null}
        </aside>
      </div>
    </div>
  )
}
