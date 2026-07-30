import { Link, useNavigate } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { api } from '../../api/client'
import { useCampaign } from '../../campaign/CampaignContext'

/** Landing page for outreach: explains the flow and launches the wizard. */
export function CampaignsOverviewPage() {
  const nav = useNavigate()
  const { leads, status, counts, fileName } = useCampaign()
  const kb = useQuery({ queryKey: ['kb'], queryFn: api.kbStatus })

  const hasCatalogue = (kb.data?.chunk_count ?? 0) > 0
  const inFlight = status === 'running'
  const hasDraftRun = leads.length > 0

  return (
    <div>
      <div className="page-hero">
        <div>
          <h1>Campaigns</h1>
          <p>Upload a lead list, pick a look, and send grounded Starlight outreach.</p>
        </div>
        <button className="btn" onClick={() => nav('/campaigns/new/leads')}>
          {hasDraftRun ? 'Continue setup' : 'New campaign'}
        </button>
      </div>

      {inFlight ? (
        <Link to="/campaigns/live" className="banner live">
          <span className="dot" />
          <div>
            <strong>Campaign running</strong>
            <div className="muted">
              {counts.processed} of {counts.total} leads processed — open live view
            </div>
          </div>
          <span className="banner-arrow">→</span>
        </Link>
      ) : null}

      {!hasCatalogue && !kb.isLoading ? (
        <div className="banner warn">
          <span className="banner-icon">📚</span>
          <div>
            <strong>No catalogue indexed yet</strong>
            <div className="muted">
              Emails are grounded in your product PDFs. Upload one for better copy.
            </div>
          </div>
          <Link to="/catalogues" className="btn secondary">Upload</Link>
        </div>
      ) : null}

      <div className="flow-cards">
        <FlowCard
          step="1"
          title="Add leads"
          body="Drop an Excel or CSV with a website column. Review and prune the list."
          to="/campaigns/new/leads"
          state={leads.length ? `${leads.length} leads · ${fileName || 'uploaded'}` : 'Not started'}
          done={leads.length > 0}
        />
        <FlowCard
          step="2"
          title="Pick the design"
          body="Choose a template and see exactly what lands in the customer's inbox."
          to="/campaigns/new/design"
          state={leads.length ? 'Ready' : 'Needs leads'}
          done={false}
        />
        <FlowCard
          step="3"
          title="Review & send"
          body="Confirm sending speed and safety options, then launch with live monitoring."
          to="/campaigns/new/review"
          state={leads.length ? 'Ready' : 'Needs leads'}
          done={false}
        />
      </div>

      {hasDraftRun ? (
        <div className="panel stack" style={{ marginTop: '1rem' }}>
          <div className="row" style={{ justifyContent: 'space-between' }}>
            <strong style={{ fontFamily: 'var(--display)' }}>Last list</strong>
            <Link to="/campaigns/live" className="pill">
              {status === 'idle' ? 'Not launched' : status}
            </Link>
          </div>
          <div className="mini-stats">
            <div><span className="muted">Leads</span><strong>{counts.total}</strong></div>
            <div><span className="muted">Sent</span><strong>{counts.sent}</strong></div>
            <div><span className="muted">Failed</span><strong>{counts.failed}</strong></div>
          </div>
        </div>
      ) : null}
    </div>
  )
}

function FlowCard({
  step,
  title,
  body,
  to,
  state,
  done,
}: {
  step: string
  title: string
  body: string
  to: string
  state: string
  done: boolean
}) {
  return (
    <Link to={to} className="flow-card">
      <div className={`flow-step${done ? ' done' : ''}`}>{done ? '✓' : step}</div>
      <div>
        <h3>{title}</h3>
        <p className="muted">{body}</p>
        <span className="pill">{state}</span>
      </div>
    </Link>
  )
}
