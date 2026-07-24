import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { api } from '../api/client'
import { useState } from 'react'

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
      setMsg(`Synced — processed ${res.processed ?? 0}`)
      await q.refetch()
    } catch (e: any) {
      setMsg(e.message)
    } finally {
      setSyncing(false)
    }
  }

  const conversations = q.data?.conversations || []

  return (
    <div className="stack" style={{ gap: '1.25rem' }}>
      <div className="row" style={{ justifyContent: 'space-between' }}>
        <div>
          <h1 style={{ margin: 0, fontFamily: 'var(--font-display)', fontWeight: 500 }}>Inbox</h1>
          <p className="muted" style={{ margin: '0.35rem 0 0' }}>Client threads with AI drafts awaiting your approval.</p>
        </div>
        <button className="btn secondary" onClick={sync} disabled={syncing}>
          {syncing ? 'Syncing…' : 'Sync Gmail'}
        </button>
      </div>
      {msg ? <div className="muted">{msg}</div> : null}

      <div className="panel stack">
        {q.isLoading ? <div className="muted">Loading conversations…</div> : null}
        {!q.isLoading && conversations.length === 0 ? (
          <div className="muted">No mailbox threads yet. Connect Gmail in Settings, run a campaign, or wait for replies.</div>
        ) : null}
        {conversations.map((c: any) => (
          <Link key={c.id} to={`/inbox/${c.id}`} className="row" style={{
            justifyContent: 'space-between',
            padding: '0.85rem 0.4rem',
            borderBottom: '1px solid var(--border)',
          }}>
            <div>
              <div style={{ fontWeight: 600 }}>{c.subject || '(no subject)'}</div>
              <div className="muted" style={{ fontSize: 13 }}>
                {c.client?.company || c.client?.email || c.client_email || 'Client'}
              </div>
            </div>
            <div className="muted" style={{ fontSize: 12 }}>{c.status}</div>
          </Link>
        ))}
      </div>
    </div>
  )
}
