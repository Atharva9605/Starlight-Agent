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
  const q = useQuery({
    queryKey: ['conversations'],
    queryFn: () => api.conversations(true),
  })

  const sync = async () => {
    setSyncing(true)
    setMsg('')
    try {
      const res: any = await api.gmailSync()
      setMsg(`Synced · ${res.processed ?? 0} new`)
      await q.refetch()
    } catch (e: any) {
      setMsg(e.message)
    } finally {
      setSyncing(false)
    }
  }

  const conversations = q.data?.conversations || []

  return (
    <div>
      <div className="page-hero">
        <div>
          <h1>Inbox</h1>
          <p>Client threads with Starlight AI drafts ready for approval.</p>
        </div>
        <button className="btn" onClick={sync} disabled={syncing}>
          {syncing ? 'Syncing…' : 'Sync Gmail'}
        </button>
      </div>

      {msg ? <div className="pill ok" style={{ marginBottom: 12 }}>{msg}</div> : null}

      <div className="panel tint-blue stack">
        {q.isLoading ? <div className="muted">Loading mailbox…</div> : null}
        {!q.isLoading && conversations.length === 0 ? (
          <div className="stack" style={{ padding: '1.5rem 0.5rem', textAlign: 'center' }}>
            <div style={{ fontSize: 40 }}>💡</div>
            <strong>No client threads yet</strong>
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
