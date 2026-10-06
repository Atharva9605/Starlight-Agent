import { useEffect, useRef, useState, type FormEvent } from 'react'
import { Link, Navigate, useNavigate } from 'react-router-dom'
import {
  AddRegular,
  ArrowClockwiseRegular,
  CheckmarkCircleRegular,
  DeleteRegular,
  DismissRegular,
  ErrorCircleRegular,
  MailInboxRegular,
  MailRegular,
  PauseRegular,
  PlayRegular,
  SendRegular,
  SparkleRegular,
  StopRegular,
  TaskListAddRegular,
  TaskListSquareLtrRegular,
} from '@fluentui/react-icons'
import { EmailPreviewFrame } from '../../components/EmailPreviewFrame'
import {
  firstPreviewIndex,
  isFailedLead,
  leadState,
  useCampaign,
  type Lead,
} from '../../campaign/CampaignContext'
import { Avatar, EmptyState, PageHeader, Spinner, useConfirm } from '../../components/ui'

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
  if (st === 'processing') return 'Writing'
  if (st === 'sent') return 'Sent'
  if (st === 'skipped') return 'Discarded'
  if (st === 'failed') return 'Failed'
  return 'Waiting'
}

function badgeClass(st: ReturnType<typeof leadState>) {
  if (st === 'sent') return 'success'
  if (st === 'failed') return 'danger'
  if (st === 'queued') return 'purple'
  if (st === 'ready') return 'brand'
  if (st === 'processing') return 'warning'
  return ''
}

function cleanStatus(status?: string) {
  return String(status || '')
    .replace(/^[❌⏭⚙️🔎📝📦⏳🔁✅\s]+/u, '')
    .trim()
}

function cardSubject(lead: Lead, subject: string | undefined, st: ReturnType<typeof leadState>) {
  if (subject) return subject
  if (st === 'processing') return cleanStatus(lead._status) || 'Writing…'
  if (st === 'failed') return cleanStatus(lead._error || lead._status) || 'Could not generate'
  if (st === 'sent') return 'Sent'
  if (st === 'skipped') return 'Discarded'
  return 'Waiting…'
}

type ListTab = 'review' | 'failed'

/** Review every generated email — send now, queue for bulk, or discard. */
export function CampaignReviewPage() {
  const nav = useNavigate()
  const confirm = useConfirm()
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
    recipientOverride,
    sendLead,
    discardLead,
    queueLead,
    sendBulk,
    retryLead,
    retryFailed,
    reviseCurrent,
    applyBulkEdit,
    removeBulkEdit,
    clearBulkEdits,
    bulkEdits,
    bulkRevising,
    bulkReviseProgress,
    selectLead,
    stop,
    pause,
    resume,
    reset,
  } = useCampaign()

  const [input, setInput] = useState('')
  const [bulkInput, setBulkInput] = useState('')
  const [tab, setTab] = useState<ListTab>('review')
  const [bulkOpen, setBulkOpen] = useState(false)
  // After a send or discard, move on to the next email that is ready to review.
  const [advanceFrom, setAdvanceFrom] = useState<number | null>(null)
  const chatEnd = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    chatEnd.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
  }, [chat.length, revising, currentIndex])

  useEffect(() => {
    if (!leads.length) return
    if (!isFailedLead(leads[currentIndex])) return
    const next = firstPreviewIndex(leads, currentIndex)
    if (next >= 0 && next !== currentIndex) selectLead(next)
  }, [leads, currentIndex, selectLead])

  useEffect(() => {
    if (advanceFrom === null) return
    const st = leads[advanceFrom] ? leadState(leads[advanceFrom]) : 'pending'
    if (st === 'failed') {
      setAdvanceFrom(null)
      return
    }
    if (st !== 'sent' && st !== 'skipped') return
    setAdvanceFrom(null)
    if (currentIndex !== advanceFrom) return
    const order = [...leads.keys()].map((k) => (advanceFrom + 1 + k) % leads.length)
    const next = order.find((i) => leadState(leads[i]) === 'ready')
    if (next !== undefined) selectLead(next)
  }, [leads, advanceFrom, currentIndex, selectLead])

  if (!leads.length) return <Navigate to="/campaigns/new" replace />
  if (status === 'running') return <Navigate to="/campaigns/live" replace />

  if (status === 'idle') {
    return (
      <section className="card">
        <EmptyState
          icon={<MailRegular />}
          title="Nothing to review yet"
          description="Start a campaign from the setup page to generate every email at once."
          actions={
            <Link to="/campaigns/new" className="btn">
              Back to setup
            </Link>
          }
        />
      </section>
    )
  }

  const done = status === 'done' || status === 'stopped'
  const paused = status === 'paused'
  const sendingThis = sendingIndex === currentIndex
  const acting = revising || sendingThis || bulkSending || bulkRevising
  const focus = leads[currentIndex]
  const focusState = focus ? leadState(focus) : 'pending'
  const focusFailed = isFailedLead(focus)
  const focusDraft = draftsByLead[currentIndex] || (draft?.rowIndex === currentIndex ? draft : null)
  const previewHtml = focusDraft?.html || livePreview?.html || focus?._preview_html || ''
  const previewSubject = focusDraft?.subject || livePreview?.subject || focus?._subject || 'Starlight outreach'
  const fromLabel = focusDraft?.from || livePreview?.from || runSender || 'Starlight Linear LED'
  const canAct = Boolean(focusDraft) && (focusState === 'ready' || focusState === 'queued')
  const queued = counts.queued
  const generated = counts.ready + counts.queued + counts.sent + counts.failed + counts.skipped
  const reviewable = leads.filter((l) => !isFailedLead(l)).length
  const busyBanner = generating || bulkSending || bulkRevising
  const progressPct =
    bulkRevising && bulkReviseProgress.total
      ? Math.round((bulkReviseProgress.done / bulkReviseProgress.total) * 100)
      : counts.progressPct

  const onChat = async (e: FormEvent) => {
    e.preventDefault()
    if (!input.trim() || acting || !focusDraft) return
    const msg = input
    setInput('')
    await reviseCurrent(msg)
  }

  const onStop = async () => {
    const ok = await confirm({
      title: 'Stop this campaign?',
      body: 'Emails that are still being written will be abandoned. Anything already sent stays sent.',
      confirmLabel: 'Stop campaign',
      danger: true,
    })
    if (ok) stop()
  }

  const onBulkSend = async () => {
    const ok = await confirm({
      title: `Send ${queued} email${queued === 1 ? '' : 's'}?`,
      body: 'Every email in the bulk queue will be sent now, one after another with the pause you chose.',
      confirmLabel: 'Send all',
    })
    if (ok) void sendBulk()
  }

  const headline = generating
    ? `Writing ${generated} of ${counts.total} emails`
    : bulkSending
      ? 'Sending the bulk queue…'
      : bulkRevising
        ? `Applying AI edit ${bulkReviseProgress.done}/${bulkReviseProgress.total}`
        : paused
          ? 'Campaign paused'
          : done
            ? status === 'stopped'
              ? 'Campaign stopped'
              : 'Campaign complete'
            : `Review ${counts.total} email${counts.total === 1 ? '' : 's'}`

  const stats: { label: string; value: number; cls: string; onClick?: () => void }[] = [
    { label: 'Ready', value: counts.ready, cls: 'brand' },
    { label: 'In bulk', value: queued, cls: 'purple' },
    { label: 'Sent', value: counts.sent, cls: 'success' },
    { label: 'Discarded', value: counts.skipped, cls: '' },
    { label: 'Failed', value: counts.failed, cls: 'danger', onClick: () => setTab('failed') },
  ]
  const focusTo = focusDraft?.to || focus?._to
  const focusSite = (focusDraft?.website || focus?.website || '').replace(/^https?:\/\//, '')
  const focusName = leadTitle(focusDraft || focus, `Lead ${currentIndex + 1}`)

  return (
    <div className="review-page">
      <PageHeader
        breadcrumb={[{ label: 'Campaigns', to: '/campaigns/runs' }, { label: 'Review' }]}
        kicker={
          busyBanner ? (
            <><span className="live-dot" /> Working</>
          ) : paused ? (
            <span className="badge warning">Paused</span>
          ) : recipientOverride.trim() ? (
            <span className="badge warning">Test mode · all to {recipientOverride.trim()}</span>
          ) : null
        }
        title={headline}
        subtitle={done ? 'Every email has been handled.' : 'Read each email, ask for changes if needed, then send it now or add it to the bulk send.'}
        actions={
          done ? (
            <>
              <button
                className="btn secondary"
                type="button"
                onClick={() => {
                  reset()
                  nav('/campaigns/new')
                }}
              >
                <AddRegular /> New campaign
              </button>
              <Link to="/inbox" className="btn">
                <MailInboxRegular /> Inbox
              </Link>
            </>
          ) : (
            <>
              {paused ? (
                <button className="btn secondary" type="button" onClick={() => void resume()} disabled={sendingThis}>
                  <PlayRegular /> Resume
                </button>
              ) : (
                <button className="btn secondary" type="button" onClick={pause} disabled={sendingThis}>
                  <PauseRegular /> Pause
                </button>
              )}
              <button className="btn danger-outline" type="button" onClick={onStop} disabled={sendingThis}>
                <StopRegular /> Stop
              </button>
            </>
          )
        }
      />

      {busyBanner ? (
        <div className="progress" style={{ marginBottom: 16 }}>
          <div className="progress-fill" style={{ width: `${progressPct}%` }} />
        </div>
      ) : null}

      <div className="stat-strip" role="list">
        {stats.map((s) => (
          <button
            key={s.label}
            type="button"
            role="listitem"
            className={`stat-chip ${s.cls}${s.value ? '' : ' zero'}`}
            onClick={s.onClick}
            disabled={!s.onClick}
          >
            <strong>{s.value}</strong>
            <span>{s.label}</span>
          </button>
        ))}
      </div>

      <section className="card review-toolbar">
        <div className="review-toolbar-row">
          <button
            className="btn subtle"
            type="button"
            disabled={!counts.ready || bulkSending || done}
            onClick={() => {
              leads.forEach((_, i) => {
                if (leadState(leads[i]) === 'ready' && draftsByLead[i]) queueLead(i, true)
              })
            }}
          >
            <TaskListAddRegular /> Add all ready to bulk
          </button>
          <button className="btn" type="button" disabled={!queued || bulkSending || done} onClick={onBulkSend}>
            {bulkSending ? <Spinner size="sm" /> : <SendRegular />}
            {bulkSending ? 'Sending…' : `Send bulk${queued ? ` (${queued})` : ''}`}
          </button>
          <span className="review-toolbar-sep" />
          <button
            type="button"
            className={`btn ${bulkOpen || bulkRevising ? 'secondary' : 'subtle'}`}
            aria-expanded={bulkOpen || bulkRevising}
            disabled={done}
            onClick={() => setBulkOpen((v) => !v)}
          >
            <SparkleRegular /> Edit all with AI
            {bulkEdits.length ? <span className="tab-count">{bulkEdits.length}</span> : null}
          </button>
        </div>
        {bulkOpen || bulkRevising ? (
          <form
            className="review-bulk"
            onSubmit={(e) => {
              e.preventDefault()
              if (!bulkInput.trim() || bulkRevising) return
              const msg = bulkInput.trim()
              setBulkInput('')
              void applyBulkEdit(msg)
            }}
          >
            <span className="muted text-sm">One instruction, applied to every email already written and every one still to come.</span>
            {bulkEdits.length ? (
              <div className="row" style={{ gap: 6 }}>
                {bulkEdits.map((msg, i) => (
                  <span key={`${i}-${msg.slice(0, 24)}`} className="badge brand text-wrap">
                    <span className="badge-text">{msg}</span>
                    <button type="button" className="chip-x" onClick={() => removeBulkEdit(i)} aria-label={`Remove “${msg}”`}>
                      <DismissRegular />
                    </button>
                  </span>
                ))}
                <button type="button" className="btn subtle sm" onClick={clearBulkEdits}>
                  Clear upcoming
                </button>
              </div>
            ) : null}
            <div className="chat-compose">
              <input
                className="input"
                placeholder="e.g. Make the tone warmer and shorten the opening"
                aria-label="Edit instruction for every email"
                value={bulkInput}
                disabled={bulkRevising}
                onChange={(e) => setBulkInput(e.target.value)}
              />
              <button className="btn" type="submit" disabled={bulkRevising || !bulkInput.trim()}>
                {bulkRevising ? <Spinner size="sm" /> : <SparkleRegular />}
                {bulkRevising ? `Updating ${bulkReviseProgress.done}/${bulkReviseProgress.total}` : 'Apply to all'}
              </button>
            </div>
          </form>
        ) : null}
      </section>

      <div className="board">
        <aside className="card flush board-list">
          <div className="tablist" role="tablist">
            <button
              type="button"
              role="tab"
              aria-selected={tab === 'review'}
              className={`tab${tab === 'review' ? ' active' : ''}`}
              onClick={() => setTab('review')}
            >
              <MailRegular /> Emails <span className="tab-count">{reviewable}</span>
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={tab === 'failed'}
              className={`tab${tab === 'failed' ? ' active' : ''}`}
              onClick={() => setTab('failed')}
            >
              <ErrorCircleRegular /> Failed
              <span className={`tab-count${counts.failed ? ' danger' : ''}`}>{counts.failed}</span>
            </button>
          </div>

          {tab === 'review' ? (
            <div className="board-list-items">
              {reviewable === 0 ? (
                <EmptyState
                  compact
                  icon={<ErrorCircleRegular />}
                  title="No emails to review"
                  description="Every lead failed. Check the Failed tab to retry them."
                />
              ) : (
                leads.map((l, i) => {
                  const st = leadState(l)
                  if (isFailedLead(l)) return null
                  const d = draftsByLead[i]
                  const subject = cardSubject(l, d?.subject || l._subject, st)
                  const to = d?.to || l._to
                  const snip = snippet(d?.html || l._preview_html)
                  return (
                    <div
                      key={i}
                      role="button"
                      tabIndex={0}
                      aria-current={i === currentIndex ? 'true' : undefined}
                      className={`mail-card${i === currentIndex ? ' active' : ''}${st === 'sent' || st === 'skipped' ? ' dim' : ''}`}
                      onClick={() => selectLead(i)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' || e.key === ' ') {
                          e.preventDefault()
                          selectLead(i)
                        }
                      }}
                    >
                      <span className="mail-card-check" onClick={(e) => e.stopPropagation()}>
                        <input
                          type="checkbox"
                          checked={st === 'queued'}
                          disabled={!d || (st !== 'ready' && st !== 'queued')}
                          onChange={() => queueLead(i, st !== 'queued')}
                          aria-label={`Add ${leadTitle(l, `lead ${i + 1}`)} to bulk send`}
                          title="Add to bulk send"
                        />
                      </span>
                      <span className="mail-card-body">
                        <span className="mail-card-top">
                          <strong>{leadTitle(l, `Lead ${i + 1}`)}</strong>
                          <span className={`badge ${badgeClass(st)}`}>
                            {st === 'processing' ? <Spinner size="sm" /> : null}
                            {stateLabel(l)}
                          </span>
                        </span>
                        <span className="mail-card-subject">{subject}</span>
                        <span className="mail-card-meta">{snip || (to ? `To ${to}` : l.website || 'Waiting…')}</span>
                      </span>
                    </div>
                  )
                })
              )}
            </div>
          ) : (
            <>
              <div className="board-list-head">
                <span>Retry finds a working website, then writes the email.</span>
                <button className="btn subtle sm" type="button" disabled={!counts.failed || bulkSending} onClick={() => void retryFailed()}>
                  <ArrowClockwiseRegular /> Retry all
                </button>
              </div>
              <div className="board-list-items">
                {counts.failed === 0 && !leads.some((l) => leadState(l) === 'processing' && l._failed_website) ? (
                  <EmptyState compact icon={<CheckmarkCircleRegular />} title="No failures" description="Every lead has been written so far." />
                ) : (
                  leads.map((l, i) => {
                    const st = leadState(l)
                    const retrying = st === 'processing' && Boolean(l._failed_website)
                    if (st !== 'failed' && !retrying) return null
                    const err = cleanStatus(l._error || l._status)
                    return (
                      <div key={`fail-${i}`} className="mail-card no-check">
                        <span className="mail-card-body">
                          <span className="mail-card-top">
                            <strong>{leadTitle(l, `Lead ${i + 1}`)}</strong>
                          </span>
                          <span className="mail-card-subject" style={{ color: retrying ? undefined : 'var(--danger-fg)' }}>
                            {retrying ? cleanStatus(l._status) || 'Looking up website…' : err || 'Could not generate'}
                          </span>
                          <span className="mail-card-meta">{l._failed_website || l.website || 'No website'}</span>
                        </span>
                        {retrying ? (
                          <Spinner size="sm" />
                        ) : (
                          <button className="btn secondary sm" type="button" disabled={acting} onClick={() => void retryLead(i)}>
                            <ArrowClockwiseRegular /> Retry
                          </button>
                        )}
                      </div>
                    )
                  })
                )}
              </div>
            </>
          )}
        </aside>

        <section className="card flush reader">
          <div className="reader-head">
            <Avatar name={focusName} size={40} />
            <div className="reader-head-copy">
              <div className="row nowrap" style={{ gap: 8 }}>
                <strong className="truncate">{focusName}</strong>
                {focus ? <span className={`badge ${badgeClass(focusState)}`}>{stateLabel(focus)}</span> : null}
              </div>
              <span className="muted text-sm truncate">
                {focusTo ? `To ${focusTo}` : 'No recipient yet'}
                {focusSite ? ` · ${focusSite}` : ''}
              </span>
            </div>
            {focusFailed ? null : (
              <div className="reader-head-actions">
                <button className="btn subtle" type="button" disabled={!canAct || acting} onClick={() => {
                  setAdvanceFrom(currentIndex)
                  void discardLead(currentIndex)
                }}>
                  <DeleteRegular /> Discard
                </button>
                <button className="btn secondary" type="button" disabled={!canAct || acting} onClick={() => queueLead(currentIndex)}>
                  {focusState === 'queued' ? <TaskListSquareLtrRegular /> : <TaskListAddRegular />}
                  {focusState === 'queued' ? 'Remove from bulk' : 'Add to bulk'}
                </button>
                <button className="btn" type="button" disabled={!canAct || acting} onClick={() => {
                  setAdvanceFrom(currentIndex)
                  void sendLead(currentIndex)
                }}>
                  {sendingThis ? <Spinner size="sm" /> : <SendRegular />}
                  {sendingThis ? 'Sending…' : 'Send now'}
                </button>
              </div>
            )}
          </div>

          {previewHtml && !focusFailed ? (
            <div className="reader-preview">
              <EmailPreviewFrame html={previewHtml} subject={previewSubject} fromLabel={fromLabel} fullscreen deviceToggle />
            </div>
          ) : (
            <div className="preview-empty">
              {focusState === 'processing' && !focusFailed ? <Spinner size="lg" /> : <MailRegular className="preview-empty-icon" />}
              <strong>
                {focusFailed
                  ? 'Select an email to preview'
                  : focusState === 'processing'
                    ? `Writing email ${currentIndex + 1} of ${counts.total}…`
                    : focusState === 'skipped'
                      ? 'This email was discarded'
                      : focusState === 'sent'
                        ? 'This email was sent'
                        : 'The preview appears when the draft is ready'}
              </strong>
              <p>
                {focusFailed
                  ? 'Failed leads are listed in the Failed tab.'
                  : focusState === 'processing'
                    ? 'This lead is being researched and written. You can review other emails meanwhile.'
                    : 'Pick any ready email from the list to preview, send or add it to bulk.'}
              </p>
            </div>
          )}

          {focusFailed ? null : (
            <div className="reader-chat">
              {chat.length || revising ? (
                <div className="chat-log">
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
                <SparkleRegular className="reader-chat-icon" />
                <input
                  className="input"
                  placeholder={focusDraft ? 'Ask for a change to this email, e.g. “Shorter, warmer opening”' : 'Waiting for the draft…'}
                  aria-label="Describe a change to this email"
                  value={input}
                  disabled={!focusDraft || acting || !canAct}
                  onChange={(e) => setInput(e.target.value)}
                />
                <button className="btn secondary" type="submit" disabled={!focusDraft || acting || !canAct || !input.trim()}>
                  Update
                </button>
              </form>
            </div>
          )}
        </section>
      </div>
    </div>
  )
}
