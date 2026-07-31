import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { api } from '../api/client'
import { useAuth } from '../auth/AuthContext'

export function HomePage() {
  const { me } = useAuth()
  const conv = useQuery({ queryKey: ['conversations'], queryFn: () => api.conversations(true) })
  const kb = useQuery({ queryKey: ['kb'], queryFn: api.kbStatus })

  const conversations = conv.data?.conversations || []
  const pending = conversations.filter((c: any) =>
    String(c.status || '').includes('draft') || c.has_draft || c.pending_draft,
  ).length

  return (
    <div>
      <div className="page-hero">
        <div>
          <h1>Hi {me?.name?.split(' ')[0] || 'there'}</h1>
          <p>Starlight AI Mailer — review replies, run outreach, keep catalogues grounded.</p>
        </div>
        <div className="row">
          <Link to="/inbox" className="btn secondary">Open inbox</Link>
          <Link to="/campaigns" className="btn">New campaign</Link>
        </div>
      </div>

      <div className="stat-row">
        <div className="stat blue">
          <div className="label">Inbox threads</div>
          <div className="value">{conversations.length}</div>
        </div>
        <div className="stat amber">
          <div className="label">Needs attention</div>
          <div className="value">{pending}</div>
        </div>
        <div className="stat cyan">
          <div className="label">Catalogue chunks</div>
          <div className="value">{kb.data?.chunk_count ?? '—'}</div>
        </div>
      </div>

      <div className="home-cta">
        <Link to="/inbox">
          <h3>Review inbox</h3>
          <p className="muted" style={{ margin: 0 }}>Approve AI drafts before they leave as Starlight.</p>
        </Link>
        <Link to="/campaigns">
          <h3>New campaign</h3>
          <p className="muted" style={{ margin: 0 }}>Upload leads and stream personalized outreach.</p>
        </Link>
        <Link to="/catalogues">
          <h3>Catalogues</h3>
          <p className="muted" style={{ margin: 0 }}>Keep product PDFs indexed so emails stay grounded.</p>
        </Link>
      </div>
    </div>
  )
}
