import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { api } from '../api/client'
import { useState } from 'react'

function statusPill(c: any) {
  if (c.has_draft || c.pending_draft) return { label: 'Needs review', cls: 'warn' }
  if (c.status === 'closed' || c.status === 'sent') return { label: 'Done', cls: 'ok' }
  return { label: c.status || 'Open', cls: '' }
}

export function InboxPage() {
  const [syncing, setSyncing] = useState(false)
  const [msg, setMsg] = useState('')
  const [error, setError] = useState('')
  const q = useQuery({
    queryKey: ['conversations'],
    queryFn: () => api.conversations(true),
  })

  const sync = async () => {
    setSyncing(true)
    setMsg('')
    setError('')
    try {
      const res: any = await api.gmailSync()
      setMsg(`Synced · ${res.processed ?? 0} new`)
      await q.refetch()
    } catch (e: any) {
      setError(e.message)
    } finally {
      setSyncing(false)
    }
  }

  const conversations = q.data?.conversations || []
  const needsReview = conversations.filter((c: any) => c.has_draft || c.pending_draft).length

  return (
    <div className="inbox-screen">
      <div className="page-hero">
        <div>
          <h1>Inbox</h1>
          <p>Client threads with Starlight AI drafts ready for approval.</p>
        </div>
        <div className="row">
          {msg ? <span className="pill ok">{msg}</span> : null}
          {needsReview ? <span className="pill warn">{needsReview} need review</span> : null}
          <button className="btn" type="button" onClick={sync} disabled={syncing}>
            {syncing ? 'Syncing…' : 'Sync Gmail'}
          </button>
        </div>
      </div>

      {error ? (
        <div className="alert danger">
          <strong>Sync failed</strong>
          <div style={{ marginTop: 4 }}>{error}</div>
        </div>
      ) : null}

      <div className="panel tint-blue inbox-list">
        {q.isLoading ? <div className="muted" style={{ padding: '1rem' }}>Loading mailbox…</div> : null}

        {!q.isLoading && conversations.length === 0 ? (
          <div className="empty-state stack" style={{ textAlign: 'center' }}>
            <strong style={{ fontFamily: 'var(--display)' }}>No client threads yet</strong>
            <p className="muted" style={{ margin: 0 }}>
              Connect Gmail, run a campaign, then replies land here with AI drafts.
            </p>
            <div className="row" style={{ justifyContent: 'center' }}>
              <Link to="/campaigns" className="btn">Start a campaign</Link>
              <Link to="/settings" className="btn secondary">Open settings</Link>
            </div>
          </div>
        ) : null}

        {conversations.map((c: any) => {
          const company = c.client?.company || c.client?.email || c.client_email || 'Client'
          const initial = String(company).trim().charAt(0).toUpperCase()
          const pill = statusPill(c)
          const snippet = c.last_message_preview || c.snippet || c.subject || 'Open thread'
          return (
            <Link key={c.id} to={`/inbox/${c.id}`} className="inbox-item">
              <div className="avatar">{initial}</div>
              <div style={{ minWidth: 0 }}>
                <div style={{ fontWeight: 700 }}>{c.subject || '(no subject)'}</div>
                <div className="muted" style={{ fontSize: 13 }}>{company}</div>
                <div className="inbox-snippet">{snippet}</div>
              </div>
              <div style={{ textAlign: 'right' }}>
                <span className={`pill ${pill.cls}`}>{pill.label}</span>
                {c.updated_at || c.created_at ? (
                  <div className="muted" style={{ fontSize: 11, marginTop: 6 }}>
                    {new Date(c.updated_at || c.created_at).toLocaleDateString()}
                  </div>
                ) : null}
              </div>
            </Link>
          )
        })}
      </div>
    </div>
  )
}
