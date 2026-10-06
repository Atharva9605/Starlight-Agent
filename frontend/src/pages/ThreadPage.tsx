import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useParams } from 'react-router-dom'
import { useEffect, useMemo, useState } from 'react'
import {
  ChatMultipleRegular,
  CheckmarkRegular,
  DeleteRegular,
  MailRegular,
  SaveRegular,
  SparkleRegular,
} from '@fluentui/react-icons'
import { api } from '../api/client'
import { EmailComposer } from '../components/EmailComposer'
import { EmailPreviewFrame } from '../components/EmailPreviewFrame'
import { MessageBubble } from '../components/MessageBubble'
import {
  CardHeader,
  EmptyState,
  MessageBar,
  PageHeader,
  Spinner,
  useConfirm,
  useToast,
} from '../components/ui'

type Pane = 'draft' | 'thread'

export function ThreadPage() {
  const { id = '' } = useParams()
  const qc = useQueryClient()
  const toast = useToast()
  const confirm = useConfirm()
  const [instructions, setInstructions] = useState('')
  const [subject, setSubject] = useState('')
  const [bodyHtml, setBodyHtml] = useState('')
  const [bodyText, setBodyText] = useState('')
  const [composerKey, setComposerKey] = useState(0)
  const [pane, setPane] = useState<Pane>('draft')

  const q = useQuery({
    queryKey: ['conversation', id],
    queryFn: () => api.conversation(id),
    enabled: !!id,
  })

  const draft = useMemo(() => {
    const messages = q.data?.messages || []
    return [...messages].reverse().find((m: any) => m.status === 'draft')
  }, [q.data])

  const timeline = useMemo(() => {
    const messages = q.data?.messages || []
    return messages.filter((m: any) => m.status !== 'draft')
  }, [q.data])

  useEffect(() => {
    if (!draft) {
      setSubject('')
      setBodyHtml('')
      setBodyText('')
      return
    }
    setSubject(draft.subject || '')
    setBodyHtml(draft.body_html || '')
    setBodyText(draft.body_text || '')
    setComposerKey((k) => k + 1)
  }, [draft?.id])

  const generate = useMutation({
    mutationFn: () => api.generateDraft(id, instructions),
    onSuccess: async () => {
      toast.success('Draft ready', 'Review the preview, then approve.')
      setPane('draft')
      await qc.invalidateQueries({ queryKey: ['conversation', id] })
    },
    onError: (e: any) => toast.error("Couldn't write a draft", e.message),
  })

  const save = useMutation({
    mutationFn: () =>
      api.updateDraft(id, draft.id, {
        subject,
        body_html: bodyHtml,
        body_text: bodyText,
      }),
    onSuccess: async () => {
      toast.success('Draft saved')
      await qc.invalidateQueries({ queryKey: ['conversation', id] })
    },
    onError: (e: any) => toast.error("Couldn't save the draft", e.message),
  })

  const approve = useMutation({
    mutationFn: async () => {
      if (draft) {
        await api.updateDraft(id, draft.id, {
          subject,
          body_html: bodyHtml,
          body_text: bodyText,
        })
      }
      return api.approveDraft(id, draft.id)
    },
    onSuccess: async () => {
      toast.success('Reply sent', 'The approved draft is on its way to the client.')
      await qc.invalidateQueries({ queryKey: ['conversation', id] })
    },
    onError: (e: any) => toast.error("Couldn't send the reply", e.message),
  })

  const reject = useMutation({
    mutationFn: () => api.rejectDraft(id, draft.id),
    onSuccess: async () => {
      toast.info('Draft discarded')
      setSubject('')
      setBodyHtml('')
      setBodyText('')
      await qc.invalidateQueries({ queryKey: ['conversation', id] })
    },
    onError: (e: any) => toast.error("Couldn't discard the draft", e.message),
  })

  const onApprove = async () => {
    const ok = await confirm({
      title: 'Send this reply?',
      body: `The draft will be sent to ${clientLabel} from your connected mailbox.`,
      confirmLabel: 'Approve & send',
    })
    if (ok) approve.mutate()
  }

  const onReject = async () => {
    const ok = await confirm({
      title: 'Discard this draft?',
      body: 'The AI draft will be removed. You can generate a new one at any time.',
      confirmLabel: 'Discard',
      danger: true,
    })
    if (ok) reject.mutate()
  }

  const busy = generate.isPending || save.isPending || approve.isPending || reject.isPending
  const clientLabel = q.data?.client?.company || q.data?.client?.email || 'Client'
  const hasDraftBody = Boolean(draft && (bodyHtml || bodyText))

  return (
    <div>
      <PageHeader
        breadcrumb={[{ label: 'Inbox', to: '/inbox' }, { label: q.data?.subject || 'Conversation' }]}
        title={q.data?.subject || 'Conversation'}
        subtitle={clientLabel}
        actions={<span className="badge warning lg">Human approval required</span>}
      />

      {q.isError ? (
        <div className="page-alerts">
          <MessageBar intent="error" title="Couldn't load this conversation">
            {(q.error as Error)?.message}
          </MessageBar>
        </div>
      ) : null}

      <div className="thread-layout">
        <section className="card flush thread-main">
          <div className="tablist" role="tablist">
            <button
              type="button"
              role="tab"
              aria-selected={pane === 'draft'}
              className={`tab${pane === 'draft' ? ' active' : ''}`}
              onClick={() => setPane('draft')}
            >
              <MailRegular /> Draft preview
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={pane === 'thread'}
              className={`tab${pane === 'thread' ? ' active' : ''}`}
              onClick={() => setPane('thread')}
            >
              <ChatMultipleRegular /> Conversation
              <span className="tab-count">{timeline.length}</span>
            </button>
          </div>

          <div className="thread-pane">
            {q.isLoading ? (
              <div className="center-fill">
                <Spinner label="Loading conversation…" />
              </div>
            ) : pane === 'draft' ? (
              hasDraftBody ? (
                <EmailPreviewFrame
                  html={bodyHtml}
                  text={bodyText}
                  subject={subject}
                  fullscreen
                  deviceToggle
                />
              ) : (
                <div className="preview-empty">
                  {generate.isPending ? (
                    <Spinner size="lg" />
                  ) : (
                    <MailRegular className="preview-empty-icon" />
                  )}
                  <strong>{generate.isPending ? 'Writing draft…' : 'No draft yet'}</strong>
                  <p>
                    {generate.isPending
                      ? 'Starlight is composing a reply for this thread.'
                      : 'Use the AI draft panel to generate a reply once the client has written back.'}
                  </p>
                </div>
              )
            ) : timeline.length === 0 ? (
              <EmptyState
                compact
                icon={<ChatMultipleRegular />}
                title="No messages yet"
                description="Messages in this thread will appear here."
              />
            ) : (
              <div className="timeline">
                {timeline.map((m: any, i: number) => (
                  <MessageBubble
                    key={m.id}
                    message={m}
                    clientLabel={clientLabel}
                    defaultOpen={i === timeline.length - 1}
                  />
                ))}
              </div>
            )}
          </div>
        </section>

        <aside className="card thread-side">
          <CardHeader
            icon={<SparkleRegular />}
            title="AI draft"
            subtitle="Edit it like an email — the preview is exactly what the client receives."
          />

          <div className="stack" style={{ marginTop: 16 }}>
            <label className="field">
              <span>Refine with AI</span>
              <input
                className="input"
                placeholder="Tone, products, call to action…"
                value={instructions}
                onChange={(e) => setInstructions(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !busy) generate.mutate()
                }}
              />
            </label>
            <button
              className={`btn ${draft ? 'secondary' : ''}`}
              type="button"
              onClick={() => generate.mutate()}
              disabled={busy}
            >
              {generate.isPending ? <Spinner size="sm" /> : <SparkleRegular />}
              {generate.isPending ? 'Writing draft…' : draft ? 'Regenerate draft' : 'Generate draft'}
            </button>

            {draft ? (
              <>
                <hr className="divider" />
                {draft.internal_note ? (
                  <MessageBar intent="warning" title="Sales note">
                    {draft.internal_note}
                  </MessageBar>
                ) : null}
                <label className="field">
                  <span>Subject</span>
                  <input className="input" value={subject} onChange={(e) => setSubject(e.target.value)} />
                </label>
                <div className="field">
                  <span className="field-label">Body</span>
                  <EmailComposer
                    key={composerKey}
                    html={bodyHtml}
                    onChange={(h, t) => {
                      setBodyHtml(h)
                      setBodyText(t)
                    }}
                  />
                </div>
                <div className="row between">
                  <button className="btn danger-outline" type="button" onClick={onReject} disabled={busy}>
                    <DeleteRegular /> Discard
                  </button>
                  <div className="row">
                    <button className="btn secondary" type="button" onClick={() => save.mutate()} disabled={busy}>
                      <SaveRegular /> Save
                    </button>
                    <button className="btn" type="button" onClick={onApprove} disabled={busy}>
                      <CheckmarkRegular /> Approve & send
                    </button>
                  </div>
                </div>
              </>
            ) : (
              <p className="muted text-sm">No draft yet — generate one after a client reply.</p>
            )}
          </div>
        </aside>
      </div>
    </div>
  )
}
