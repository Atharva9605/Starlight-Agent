import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useParams } from 'react-router-dom'
import { api } from '../api/client'
import { useEffect, useMemo, useState } from 'react'

export function ThreadPage() {
  const { id = '' } = useParams()
  const qc = useQueryClient()
  const [instructions, setInstructions] = useState('')
  const [edit, setEdit] = useState({ subject: '', body_html: '', body_text: '' })
  const [notice, setNotice] = useState('')

  const q = useQuery({
    queryKey: ['conversation', id],
    queryFn: () => api.conversation(id),
    enabled: !!id,
  })

  const draft = useMemo(() => {
    const messages = q.data?.messages || []
    return [...messages].reverse().find((m: any) => m.status === 'draft')
  }, [q.data])

  useEffect(() => {
    if (!draft) {
      setEdit({ subject: '', body_html: '', body_text: '' })
      return
    }
    setEdit({
      subject: draft.subject || '',
      body_html: draft.body_html || '',
      body_text: draft.body_text || '',
    })
  }, [draft?.id])

  const generate = useMutation({
    mutationFn: () => api.generateDraft(id, instructions),
    onSuccess: async () => {
      setNotice('Draft generated')
      await qc.invalidateQueries({ queryKey: ['conversation', id] })
    },
    onError: (e: any) => setNotice(e.message),
  })

  const save = useMutation({
    mutationFn: () => api.updateDraft(id, draft.id, edit),
    onSuccess: async () => {
      setNotice('Draft saved')
      await qc.invalidateQueries({ queryKey: ['conversation', id] })
    },
  })

  const approve = useMutation({
    mutationFn: () => api.approveDraft(id, draft.id),
    onSuccess: async () => {
      setNotice('Draft approved & sent')
      await qc.invalidateQueries({ queryKey: ['conversation', id] })
    },
    onError: (e: any) => setNotice(e.message),
  })

  const reject = useMutation({
    mutationFn: () => api.rejectDraft(id, draft.id),
    onSuccess: async () => {
      setNotice('Draft rejected')
      setEdit({ subject: '', body_html: '', body_text: '' })
      await qc.invalidateQueries({ queryKey: ['conversation', id] })
    },
  })

  const messages = q.data?.messages || []

  return (
    <div className="stack" style={{ gap: '1.25rem' }}>
      <div>
        <h1 style={{ margin: 0, fontFamily: 'var(--font-display)', fontWeight: 500 }}>
          {q.data?.subject || 'Thread'}
        </h1>
        <p className="muted">{q.data?.client?.email}</p>
      </div>

      <div className="grid-2">
        <div className="panel stack">
          <strong>Timeline</strong>
          {messages.map((m: any) => (
            <div key={m.id} style={{ borderLeft: '2px solid var(--border)', paddingLeft: 12 }}>
              <div className="muted" style={{ fontSize: 12 }}>
                {m.direction} · {m.status} {m.ai_generated ? '· AI' : ''}
              </div>
              <div style={{ fontWeight: 600 }}>{m.subject}</div>
              <div style={{ whiteSpace: 'pre-wrap', fontSize: 14 }}>
                {m.body_text || '(html body)'}
              </div>
            </div>
          ))}
        </div>

        <div className="panel stack">
          <strong>AI draft (human approve)</strong>
          <textarea
            className="textarea"
            rows={2}
            placeholder="Optional refinement instructions"
            value={instructions}
            onChange={(e) => setInstructions(e.target.value)}
          />
          <button className="btn" onClick={() => generate.mutate()} disabled={generate.isPending}>
            {generate.isPending ? 'Generating…' : 'Generate draft'}
          </button>

          {draft ? (
            <>
              <input className="input" value={edit.subject} onChange={(e) => setEdit({ ...edit, subject: e.target.value })} placeholder="Subject" />
              <textarea className="textarea" rows={10} value={edit.body_html || edit.body_text} onChange={(e) => setEdit({ ...edit, body_html: e.target.value, body_text: e.target.value })} />
              <div className="row">
                <button className="btn secondary" onClick={() => save.mutate()} disabled={save.isPending}>Save</button>
                <button className="btn" onClick={() => approve.mutate()} disabled={approve.isPending}>Approve & send</button>
                <button className="btn danger" onClick={() => reject.mutate()} disabled={reject.isPending}>Reject</button>
              </div>
              {edit.body_html ? (
                <iframe title="preview" sandbox="" srcDoc={edit.body_html} style={{ width: '100%', minHeight: 220, border: '1px solid var(--border)', borderRadius: 12, background: 'white' }} />
              ) : null}
            </>
          ) : (
            <div className="muted">No draft yet. Generate one after an inbound message.</div>
          )}
          {notice ? <div className="muted">{notice}</div> : null}
        </div>
      </div>
    </div>
  )
}
