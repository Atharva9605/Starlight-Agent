import { useEffect, useRef, useState, type FormEvent } from 'react'
import { Link, Navigate, useNavigate } from 'react-router-dom'
import { EmailPreviewFrame } from '../../components/EmailPreviewFrame'
import { leadState, useCampaign } from '../../campaign/CampaignContext'

/** Page 2 — full-screen email + chat edits. Send only on click. */
export function CampaignReviewPage() {
  const nav = useNavigate()
  const {
    leads,
    draft,
    chat,
    generating,
    revising,
    sending,
    status,
    currentIndex,
    counts,
    sendCurrent,
    skipCurrent,
    reviseCurrent,
    stop,
    reset,
  } = useCampaign()

  const [input, setInput] = useState('')
  const chatEnd = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    chatEnd.current?.scrollIntoView({ behavior: 'smooth' })
  }, [chat.length, revising])

  if (!leads.length) return <Navigate to="/campaigns" replace />

  if (status === 'idle') {
    return (
      <div className="panel empty-state">
        <strong>Nothing to review yet</strong>
        <p className="muted">Start a campaign from the setup page.</p>
        <Link to="/campaigns" className="btn">Back to setup</Link>
      </div>
    )
  }

  const done = status === 'done' || status === 'stopped'
  const busy = generating || revising || sending

  const onChat = async (e: FormEvent) => {
    e.preventDefault()
    if (!input.trim() || busy || !draft) return
    const msg = input
    setInput('')
    await reviseCurrent(msg)
  }

  return (
    <div className="review-screen">
      <header className="review-top">
        <div>
          <div className="muted" style={{ fontSize: 12, fontWeight: 700, letterSpacing: '0.06em' }}>
            LEAD {Math.min(currentIndex + 1, leads.length)} / {leads.length}
            {counts.sent ? ` · ${counts.sent} sent` : ''}
            {counts.skipped ? ` · ${counts.skipped} skipped` : ''}
          </div>
          <h1>
            {done
              ? status === 'stopped'
                ? 'Stopped'
                : 'Campaign complete'
              : generating
                ? 'Writing email…'
                : draft?.company || draft?.website || 'Review email'}
          </h1>
          {draft?.to ? (
            <p className="muted" style={{ margin: '0.2rem 0 0' }}>To {draft.to}</p>
          ) : null}
        </div>
        <div className="row">
          {!done ? (
            <button className="btn secondary" type="button" onClick={stop} disabled={sending}>
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
              <Link to="/inbox" className="btn">Inbox</Link>
            </>
          )}
        </div>
      </header>

      {done && !draft ? (
        <div className="panel empty-state" style={{ marginTop: '1rem' }}>
          <strong>
            {counts.sent} sent · {counts.failed} failed · {counts.skipped} skipped
          </strong>
          <p className="muted">All leads processed.</p>
        </div>
      ) : (
        <div className="review-body">
          <div className="review-mail">
            {generating && !draft ? (
              <div className="skeleton-frame tall live-preview-empty">
                <div className="live-pulse-ring" />
                <strong>Writing email for lead {currentIndex + 1}…</strong>
                <ol className="gen-timeline" style={{ width: 'min(360px, 100%)', textAlign: 'left', marginTop: '1.25rem' }}>
                  {[
                    'Scraping website',
                    'Analyzing company',
                    'Matching catalogue',
                    'Drafting personalized email',
                    'Rendering preview',
                  ].map((label, i) => (
                    <li key={label} className={`gen-step ${i < 2 ? 'done' : i === 2 ? 'active' : 'pending'}`}>
                      <span className="gen-step-marker" aria-hidden />
                      <div className="gen-step-body">
                        <div className="gen-step-title">{label}</div>
                      </div>
                    </li>
                  ))}
                </ol>
              </div>
            ) : draft ? (
              <EmailPreviewFrame html={draft.html} subject={draft.subject} fullscreen />
            ) : (
              <div className="skeleton-frame tall">Waiting for the next draft…</div>
            )}
          </div>

          <aside className="review-side">
            <div className="review-queue">
              <strong style={{ fontFamily: 'var(--display)', fontSize: 14 }}>Queue</strong>
              <div className="queue compact">
                {leads.map((l, i) => {
                  const st = leadState(l)
                  return (
                    <div key={i} className={`queue-row${i === currentIndex ? ' active' : ''}`}>
                      <span className={`queue-index ${st}`}>{i + 1}</span>
                      <span className="queue-name">{l.website || l.company || `Lead ${i + 1}`}</span>
                    </div>
                  )
                })}
              </div>
            </div>

            <div className="review-chat panel">
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
          </aside>
        </div>
      )}
    </div>
  )
}
