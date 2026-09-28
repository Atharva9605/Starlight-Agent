import { useEffect, useRef, useState, type FormEvent } from 'react'
import { Link, Navigate, useNavigate } from 'react-router-dom'
import { EmailPreviewFrame } from '../../components/EmailPreviewFrame'
import { leadState, useCampaign, type Lead } from '../../campaign/CampaignContext'

function leadTitle(lead: Lead | undefined, fallback: string) {
  return lead?.company || lead?.name || lead?.website || fallback
}

function snippet(html?: string) {
  if (!html) return ''
  return html
    .replace(/<style[\s\S]*?<\/style>/gi, ' ')
    .replace(/<[^>]+>/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .slice(0, 140)
}

function stateLabel(lead: Lead) {
  const st = leadState(lead)
  if (st === 'queued') return 'In bulk'
  if (st === 'ready') return 'Ready'
  if (st === 'processing') return lead._status || 'Writing'
  if (st === 'sent') return 'Sent'
  if (st === 'skipped') return 'Discarded'
  if (st === 'failed') return 'Failed'
  return lead._status || 'Waiting'
}

function pillClass(st: ReturnType<typeof leadState>) {
  if (st === 'sent') return 'ok'
  if (st === 'failed') return 'pink'
  if (st === 'queued' || st === 'processing' || st === 'ready') return 'warn'
  return ''
}

/** Review every generated email — send now, queue for bulk, or discard. */
export function CampaignReviewPage() {
  const nav = useNavigate()
  const {
    leads,
    draft,
    chat,
    generating,
    revising,
    sendingIndex,
    bulkSending,
    status,
    currentIndex,
    counts,
    livePreview,
    draftsByLead,
    runSender,
    sendLead,
    discardLead,
    queueLead,
    sendBulk,
    reviseCurrent,
    selectLead,
    stop,
    reset,
  } = useCampaign()

  const [input, setInput] = useState('')
  const chatEnd = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    chatEnd.current?.scrollIntoView({ behavior: 'smooth' })
  }, [chat.length, revising, currentIndex])

  if (!leads.length) return <Navigate to="/campaigns" replace />
  if (status === 'running') return <Navigate to="/campaigns/live" replace />

  if (status === 'idle') {
    return (
      <div className="panel empty-state">
        <strong>Nothing to review yet</strong>
        <p className="muted">Start a campaign from the setup page to generate every email at once.</p>
        <Link to="/campaigns" className="btn">
          Back to setup
        </Link>
      </div>
    )
  }

  const done = status === 'done' || status === 'stopped'
  const sendingThis = sendingIndex === currentIndex
  const acting = revising || sendingThis || bulkSending
  const focus = leads[currentIndex]
  const focusState = focus ? leadState(focus) : 'pending'
  const focusDraft = draftsByLead[currentIndex] || (draft?.rowIndex === currentIndex ? draft : null)
  const previewHtml = focusDraft?.html || livePreview?.html || focus?._preview_html || ''
  const previewSubject = focusDraft?.subject || livePreview?.subject || focus?._subject || 'Starlight outreach'
  const fromLabel = focusDraft?.from || livePreview?.from || runSender || 'Starlight Linear LED'
  const canAct = Boolean(focusDraft) && (focusState === 'ready' || focusState === 'queued')
  const queued = counts.queued
  const generated = counts.ready + counts.queued + counts.sent + counts.failed + counts.skipped

  const onChat = async (e: FormEvent) => {
    e.preventDefault()
    if (!input.trim() || acting || !focusDraft) return
    const msg = input
    setInput('')
    await reviseCurrent(msg)
  }

  const headline = generating
    ? `Writing ${generated} of ${counts.total} emails`
    : bulkSending
      ? `Sending bulk queue…`
      : done
        ? status === 'stopped'
          ? 'Stopped'
          : 'Campaign complete'
        : `Review ${counts.total} email${counts.total === 1 ? '' : 's'}`

  return (
    <div className="mail-board-screen">
      <header className="review-top">
        <div>
          <div className="dash-kicker">
            {generating || bulkSending ? <span className="live-dot" /> : null}
            Review all emails
          </div>
          <h1>{headline}</h1>
          <p className="muted" style={{ margin: '0.2rem 0 0' }}>
            {counts.ready} ready · {queued} in bulk · {counts.sent} sent · {counts.skipped} discarded
            {counts.failed ? ` · ${counts.failed} failed` : ''}
            {counts.processing ? ` · ${counts.processing} writing` : ''}
          </p>
        </div>
        <div className="row" style={{ flexWrap: 'wrap' }}>
          {!done ? (
            <button className="btn secondary" type="button" onClick={stop} disabled={sendingThis}>
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
          <button
            className="btn amber"
            type="button"
            disabled={!queued || bulkSending}
            onClick={() => void sendBulk()}
          >
            {bulkSending ? 'Sending bulk…' : `Bulk send ${queued || ''}`}
          </button>
        </div>
      </header>

      {generating ? (
        <div className="progress thick" style={{ margin: 0 }}>
          <div className="progress-fill animated" style={{ width: `${counts.progressPct}%` }} />
        </div>
      ) : null}

      <div className="mail-board">
        <aside className="mail-inbox panel">
          <div className="row" style={{ justifyContent: 'space-between', marginBottom: 8 }}>
            <strong style={{ fontFamily: 'var(--display)', fontSize: 15 }}>All emails</strong>
            <button
              className="btn secondary"
              type="button"
              style={{ padding: '0.35rem 0.75rem', fontSize: 12 }}
              disabled={!counts.ready || bulkSending}
              onClick={() => {
                leads.forEach((_, i) => {
                  if (leadState(leads[i]) === 'ready' && draftsByLead[i]) queueLead(i, true)
                })
              }}
            >
              Add all to bulk
            </button>
          </div>
          <div className="mail-inbox-list">
            {leads.map((l, i) => {
              const st = leadState(l)
              const d = draftsByLead[i]
              const subject = d?.subject || l._subject || (st === 'processing' ? 'Writing…' : 'Waiting…')
              const to = d?.to || l._to
              return (
                <div
                  key={i}
                  role="button"
                  tabIndex={0}
                  className={`mail-card${i === currentIndex ? ' active' : ''}${st === 'queued' ? ' queued' : ''}${st === 'sent' ? ' sent' : ''}${st === 'skipped' ? ' discarded' : ''}`}
                  onClick={() => selectLead(i)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault()
                      selectLead(i)
                    }
                  }}
                >
                  <span
                    className="mail-card-check"
                    onClick={(e) => e.stopPropagation()}
                  >
                    <input
                      type="checkbox"
                      checked={st === 'queued'}
                      disabled={!d || (st !== 'ready' && st !== 'queued')}
                      onChange={() => queueLead(i, st !== 'queued')}
                      title="Add to bulk send"
                    />
                  </span>
                  <span className={`queue-index ${st}`}>{i + 1}</span>
                  <span className="mail-card-body">
                    <span className="mail-card-top">
                      <strong>{leadTitle(l, `Lead ${i + 1}`)}</strong>
                      <span className={`pill ${pillClass(st)}`}>{stateLabel(l)}</span>
                    </span>
                    <span className="mail-card-subject">{subject}</span>
                    <span className="muted mail-card-meta">
                      {to ? `To ${to}` : l.website || 'No recipient yet'}
                      {snippet(d?.html || l._preview_html) ? ` · ${snippet(d?.html || l._preview_html)}` : ''}
                    </span>
                  </span>
                </div>
              )
            })}
          </div>
        </aside>

        <section className="mail-reader">
          <div className="mail-reader-main panel stack">
            <div className="row" style={{ justifyContent: 'space-between', alignItems: 'flex-start' }}>
              <div>
                <div className="design-preview-label muted">Selected email</div>
                <h2 style={{ margin: '0.2rem 0 0', fontFamily: 'var(--display)', fontSize: '1.25rem' }}>
                  {leadTitle(focusDraft || focus, `Lead ${currentIndex + 1}`)}
                </h2>
                {(focusDraft?.to || focus?._to) && (
                  <p className="muted" style={{ margin: '0.25rem 0 0', fontSize: 13 }}>
                    To {focusDraft?.to || focus?._to}
                    {focusDraft?.website || focus?.website
                      ? ` · ${(focusDraft?.website || focus?.website || '').replace(/^https?:\/\//, '')}`
                      : ''}
                  </p>
                )}
              </div>
              <span className={`pill ${pillClass(focusState)}`}>{focus ? stateLabel(focus) : '—'}</span>
            </div>

            {previewHtml ? (
              <div className="mail-reader-preview">
                <EmailPreviewFrame html={previewHtml} subject={previewSubject} fromLabel={fromLabel} fullscreen />
              </div>
            ) : (
              <div className="skeleton-frame tall live-preview-empty">
                <div className="live-pulse-ring" />
                <strong>
                  {focusState === 'processing' || generating
                    ? `Writing email ${currentIndex + 1} of ${counts.total}…`
                    : focusState === 'failed'
                      ? 'This email failed to generate'
                      : focusState === 'skipped'
                        ? 'This email was discarded'
                        : focusState === 'sent'
                          ? 'This email was sent'
                          : 'Preview appears here when the draft is ready'}
                </strong>
                <p className="muted" style={{ margin: '0.4rem 0 0', maxWidth: '42ch' }}>
                  Every lead is generated together. Open any card on the left as soon as it is ready.
                </p>
              </div>
            )}

            <div className="mail-actions">
              <button
                className="btn danger"
                type="button"
                disabled={!canAct || acting}
                onClick={() => void discardLead(currentIndex)}
              >
                Discard
              </button>
              <button
                className="btn secondary"
                type="button"
                disabled={!canAct || acting}
                onClick={() => queueLead(currentIndex)}
              >
                {focusState === 'queued' ? 'Remove from bulk' : 'Add to bulk send'}
              </button>
              <button
                className="btn"
                type="button"
                disabled={!canAct || acting}
                onClick={() => void sendLead(currentIndex)}
              >
                {sendingThis ? 'Sending…' : 'Send now'}
              </button>
            </div>
          </div>

          <aside className="review-chat panel">
            <strong style={{ fontFamily: 'var(--display)' }}>Ask for changes</strong>
            <div className="chat-log">
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
                placeholder={focusDraft ? 'e.g. Make the CTA softer…' : 'Waiting for draft…'}
                value={input}
                disabled={!focusDraft || acting || !canAct}
                onChange={(e) => setInput(e.target.value)}
              />
              <button className="btn secondary" type="submit" disabled={!focusDraft || acting || !canAct || !input.trim()}>
                Update
              </button>
            </form>
          </aside>
        </section>
      </div>
    </div>
  )
}
