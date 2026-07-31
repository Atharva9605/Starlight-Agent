import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Link, useParams } from 'react-router-dom'
import { api } from '../api/client'
import { useEffect, useMemo, useState } from 'react'
import { EmailComposer } from '../components/EmailComposer'
import { EmailPreviewFrame } from '../components/EmailPreviewFrame'
import { MessageBubble } from '../components/MessageBubble'

export function ThreadPage() {
  const { id = '' } = useParams()
  const qc = useQueryClient()
  const [instructions, setInstructions] = useState('')
  const [subject, setSubject] = useState('')
  const [bodyHtml, setBodyHtml] = useState('')
  const [bodyText, setBodyText] = useState('')
  const [notice, setNotice] = useState('')
  const [composerKey, setComposerKey] = useState(0)

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
      setNotice('Draft ready — review the preview, then approve')
      await qc.invalidateQueries({ queryKey: ['conversation', id] })
    },
    onError: (e: any) => setNotice(e.message),
  })

  const save = useMutation({
    mutationFn: () =>
      api.updateDraft(id, draft.id, {
        subject,
        body_html: bodyHtml,
        body_text: bodyText,
      }),
    onSuccess: async () => {
      setNotice('Saved')
      await qc.invalidateQueries({ queryKey: ['conversation', id] })
    },
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
      setNotice('Approved & sent')
      await qc.invalidateQueries({ queryKey: ['conversation', id] })
    },
    onError: (e: any) => setNotice(e.message),
  })

  const reject = useMutation({
    mutationFn: () => api.rejectDraft(id, draft.id),
    onSuccess: async () => {
      setNotice('Draft discarded')
      setSubject('')
      setBodyHtml('')
      setBodyText('')
      await qc.invalidateQueries({ queryKey: ['conversation', id] })
    },
  })

  const busy = generate.isPending || save.isPending || approve.isPending || reject.isPending
  const clientLabel = q.data?.client?.company || q.data?.client?.email || 'Client thread'

  return (
    <div className="review-screen">
      <header className="review-top">
        <div>
          <Link to="/inbox" className="muted" style={{ fontWeight: 600, fontSize: 12, letterSpacing: '0.04em' }}>
            ← INBOX
          </Link>
          <h1>{q.data?.subject || 'Conversation'}</h1>
          <p className="muted" style={{ margin: '0.2rem 0 0' }}>{clientLabel}</p>
        </div>
        <div className="row">
          {notice ? <span className="pill ok">{notice}</span> : null}
          <span className="pill pink">Human approve required</span>
        </div>
      </header>

      <div className="review-body">
        <div className="review-mail">
          {draft && (bodyHtml || bodyText) ? (
            <EmailPreviewFrame html={bodyHtml} text={bodyText} subject={subject} fullscreen />
          ) : (
            <div className="panel empty-state" style={{ flex: 1, display: 'grid', placeItems: 'center' }}>
              <div style={{ textAlign: 'center', maxWidth: 360 }}>
                <strong style={{ fontFamily: 'var(--display)' }}>
                  {generate.isPending ? 'Writing draft…' : 'No draft yet'}
                </strong>
                <p className="muted" style={{ margin: '0.5rem 0 0' }}>
                  {generate.isPending
                    ? 'Starlight is composing a reply for this thread.'
                    : 'Generate a draft from the side panel after a client reply.'}
                </p>
              </div>
            </div>
          )}
        </div>

        <aside className={`review-side${generate.isPending ? ' is-generating' : ''}`}>
          <div className="review-queue stack" style={{ maxHeight: 200, overflow: 'auto' }}>
            <strong style={{ fontFamily: 'var(--display)', fontSize: 14 }}>Timeline</strong>
            {timeline.length === 0 ? (
              <div className="muted" style={{ fontSize: 13 }}>No messages yet in this thread.</div>
            ) : (
              timeline.map((m: any) => <MessageBubble key={m.id} message={m} />)
            )}
          </div>

          <div className="panel tint-amber stack" style={{ flex: 1 }}>
            <div className="studio-editor-head">
              <div>
                <h2 style={{ fontSize: '1.05rem' }}>AI draft</h2>
                <p className="muted" style={{ margin: '0.25rem 0 0', fontSize: 13 }}>
                  Edit like email — preview is what the client gets.
                </p>
              </div>
            </div>

            <label className="field">
              <span>Refine with AI</span>
              <input
                className="input"
                placeholder="Tone, products, CTA…"
                value={instructions}
                onChange={(e) => setInstructions(e.target.value)}
              />
            </label>
            <button className="btn amber" type="button" onClick={() => generate.mutate()} disabled={busy}>
              {generate.isPending ? 'Writing draft…' : draft ? 'Regenerate draft' : 'Generate draft'}
            </button>

            {draft ? (
              <>
                {draft.internal_note ? (
                  <div className="pill warn">Sales note: {draft.internal_note}</div>
                ) : null}
                <label className="field">
                  <span>Subject</span>
                  <input className="input" value={subject} onChange={(e) => setSubject(e.target.value)} />
                </label>
                <label className="field">
                  <span>Body</span>
                  <EmailComposer
                    key={composerKey}
                    html={bodyHtml}
                    onChange={(h, t) => {
                      setBodyHtml(h)
                      setBodyText(t)
                    }}
                  />
                </label>
                <div className="row">
                  <button className="btn secondary" type="button" onClick={() => save.mutate()} disabled={busy}>
                    Save
                  </button>
                  <button className="btn" type="button" onClick={() => approve.mutate()} disabled={busy}>
                    Approve & send
                  </button>
                  <button className="btn danger" type="button" onClick={() => reject.mutate()} disabled={busy}>
                    Reject
                  </button>
                </div>
              </>
            ) : (
              <p className="muted" style={{ margin: 0, fontSize: 13 }}>
                No draft yet — generate after a client reply.
              </p>
            )}
          </div>
        </aside>
      </div>
    </div>
  )
}
