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

  return (
    <div>
      <div className="page-hero">
        <div>
          <Link to="/inbox" className="muted" style={{ fontWeight: 600, fontSize: 13 }}>← Inbox</Link>
          <h1 style={{ marginTop: 6 }}>{q.data?.subject || 'Conversation'}</h1>
          <p>{q.data?.client?.company || q.data?.client?.email || 'Client thread'}</p>
        </div>
        <span className="pill pink">Human approve required</span>
      </div>

      <div className="grid-2 thread-layout">
        <div className="panel stack">
          <strong style={{ fontFamily: 'var(--display)' }}>Conversation</strong>
          {timeline.length === 0 ? (
            <div className="muted">No messages yet in this thread.</div>
          ) : null}
          {timeline.map((m: any) => (
            <MessageBubble key={m.id} message={m} />
          ))}
        </div>

        <div className={`panel tint-amber stack ${generate.isPending ? 'is-generating' : ''}`}>
          <strong style={{ fontFamily: 'var(--display)' }}>AI draft for customer</strong>
          <p className="muted" style={{ margin: 0, fontSize: 13 }}>
            Edit like email — you never need to touch HTML. Preview is exactly what the client gets.
          </p>

          <input
            className="input"
            placeholder="Refine with AI (tone, products, CTA…)"
            value={instructions}
            onChange={(e) => setInstructions(e.target.value)}
          />
          <button className="btn amber" onClick={() => generate.mutate()} disabled={generate.isPending}>
            {generate.isPending ? 'Writing draft…' : draft ? 'Regenerate draft' : 'Generate draft'}
          </button>

          {draft ? (
            <>
              {draft.internal_note ? (
                <div className="pill warn">Sales note: {draft.internal_note}</div>
              ) : null}
              <label className="muted" style={{ fontSize: 12, fontWeight: 700 }}>Subject</label>
              <input className="input" value={subject} onChange={(e) => setSubject(e.target.value)} />

              <label className="muted" style={{ fontSize: 12, fontWeight: 700 }}>Body</label>
              <EmailComposer
                key={composerKey}
                html={bodyHtml}
                onChange={(h, t) => {
                  setBodyHtml(h)
                  setBodyText(t)
                }}
              />

              <label className="muted" style={{ fontSize: 12, fontWeight: 700 }}>Customer preview</label>
              <EmailPreviewFrame html={bodyHtml} text={bodyText} subject={subject} />

              <div className="row">
                <button className="btn secondary" onClick={() => save.mutate()} disabled={save.isPending}>Save</button>
                <button className="btn" onClick={() => approve.mutate()} disabled={approve.isPending}>Approve & send</button>
                <button className="btn danger" onClick={() => reject.mutate()} disabled={reject.isPending}>Reject</button>
              </div>
            </>
          ) : (
            <div className="muted">No draft yet — generate after a client reply.</div>
          )}
          {notice ? <div className="pill ok">{notice}</div> : null}
        </div>
      </div>
    </div>
  )
}
