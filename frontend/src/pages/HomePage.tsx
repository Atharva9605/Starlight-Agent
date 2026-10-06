import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import {
  AddRegular,
  ArrowRightRegular,
  BoxRegular,
  CheckmarkCircleFilled,
  CircleRegular,
  HistoryRegular,
  MailInboxRegular,
  MailWarningRegular,
  PersonChatRegular,
  SendRegular,
} from '@fluentui/react-icons'
import { api, type CampaignRun } from '../api/client'
import { useAuth } from '../auth/AuthContext'
import { Avatar, CardHeader, EmptyState, PageHeader, relativeTime } from '../components/ui'
import { convClientWaiting, convCompany, convNeedsReview, convSnippet } from '../inbox/conversation'

const RUN_BADGE: Record<string, { label: string; cls: string }> = {
  scheduled: { label: 'Scheduled', cls: 'purple' },
  running: { label: 'Sending', cls: 'brand' },
  paused: { label: 'Paused', cls: 'warning' },
  done: { label: 'Completed', cls: 'success' },
  stopped: { label: 'Stopped', cls: '' },
  failed: { label: 'Failed', cls: 'danger' },
}

const MONTH_MS = 30 * 24 * 60 * 60 * 1000

function greeting() {
  const h = new Date().getHours()
  if (h < 12) return 'Good morning'
  if (h < 18) return 'Good afternoon'
  return 'Good evening'
}

/** Start page: what needs you now, how outreach is going, and what's left to set up. */
export function HomePage() {
  const { me } = useAuth()
  const conv = useQuery({ queryKey: ['conversations'], queryFn: () => api.conversations(true) })
  const runs = useQuery({ queryKey: ['campaign-runs'], queryFn: () => api.campaignRuns() })
  const gmail = useQuery({ queryKey: ['gmail-status'], queryFn: api.gmailStatus })
  const library = useQuery({ queryKey: ['library-catalogues'], queryFn: api.listCatalogues })
  const kb = useQuery({ queryKey: ['kb'], queryFn: api.kbStatus })
  const sender = useQuery({ queryKey: ['sender'], queryFn: api.sender })

  const conversations: any[] = conv.data?.conversations || []
  const toReview = conversations.filter(convNeedsReview)
  const waiting = conversations.filter(convClientWaiting)
  const allRuns: CampaignRun[] = runs.data?.runs || []
  const liveRuns = allRuns.filter((r) => r.status === 'running' || r.status === 'paused' || r.status === 'scheduled')
  const recentRuns = allRuns.filter((r) => !liveRuns.includes(r)).slice(0, 5)
  const sentThisMonth = allRuns
    .filter((r) => Date.now() - new Date(r.created_at).getTime() < MONTH_MS)
    .reduce((n, r) => n + (r.counts?.sent || 0), 0)
  const catalogues = library.data?.catalogues || []
  const productCount = catalogues.reduce((n, c) => n + (c.product_count || 0), 0)
  const hasCatalogue = catalogues.length > 0 || (kb.data?.chunk_count || 0) > 0
  const firstName = me?.name?.split(' ')[0] || 'there'

  const setupLoaded = gmail.isSuccess && runs.isSuccess && library.isSuccess && sender.isSuccess
  const setup = [
    {
      done: Boolean(gmail.data?.connected || gmail.data?.mode === 'platform'),
      title: 'Connect Gmail',
      body: 'Campaigns send from this inbox, and client replies sync back into the Inbox.',
      to: '/settings',
      cta: 'Connect',
    },
    {
      done: Boolean(sender.data?.sender_email && sender.data?.sender_name),
      title: 'Fill in your sender profile',
      body: 'Your name, phone and logo appear in every email signature.',
      to: '/settings',
      cta: 'Open settings',
    },
    {
      done: hasCatalogue,
      title: 'Upload a product catalogue',
      body: 'Emails only mention products that are in your catalogues.',
      to: '/catalogues',
      cta: 'Upload',
    },
    {
      done: allRuns.length > 0,
      title: 'Send your first campaign',
      body: 'Upload a lead sheet, check the drafts, then send.',
      to: '/campaigns',
      cta: 'Start',
    },
  ]
  const setupDone = setup.filter((s) => s.done).length
  const showSetup = setupLoaded && setupDone < setup.length

  const attention = [...toReview, ...waiting].slice(0, 6)
  const dash = (loading: boolean, v: number | string) => (loading ? '—' : v)

  return (
    <div>
      <PageHeader
        title={`${greeting()}, ${firstName}`}
        subtitle={
          conv.isLoading
            ? 'Here’s what’s happening with your outreach.'
            : toReview.length
              ? `${toReview.length} AI draft${toReview.length === 1 ? ' is' : 's are'} waiting for your approval.`
              : waiting.length
                ? `${waiting.length} client${waiting.length === 1 ? ' is' : 's are'} waiting for a reply.`
                : 'You’re all caught up. Start a campaign to reach new clients.'
        }
        actions={
          <>
            <Link to="/inbox" className="btn secondary">
              <MailInboxRegular /> Open inbox
            </Link>
            <Link to="/campaigns" className="btn">
              <AddRegular /> New campaign
            </Link>
          </>
        }
      />

      {showSetup ? (
        <section className="card" style={{ marginBottom: 20 }}>
          <CardHeader
            title="Finish setting up"
            subtitle={`${setupDone} of ${setup.length} done. Complete these so campaigns send well.`}
          />
          <div className="progress" style={{ margin: '12px 0 4px' }}>
            <div className="progress-fill success" style={{ width: `${(setupDone / setup.length) * 100}%` }} />
          </div>
          <ol className="checklist">
            {setup.map((s) => (
              <li key={s.title} className={`checklist-item${s.done ? ' done' : ''}`}>
                {s.done ? <CheckmarkCircleFilled className="checklist-icon" /> : <CircleRegular className="checklist-icon" />}
                <div className="grow">
                  <strong>{s.title}</strong>
                  <span>{s.body}</span>
                </div>
                {s.done ? null : (
                  <Link to={s.to} className="btn secondary sm">
                    {s.cta}
                  </Link>
                )}
              </li>
            ))}
          </ol>
        </section>
      ) : null}

      {liveRuns.map((r) => {
        const pct = r.total ? Math.round((r.counts.processed / r.total) * 100) : 0
        const badge = RUN_BADGE[r.status]
        return (
          <section key={r.id} className="card live-run" style={{ marginBottom: 20 }}>
            <div className="row between" style={{ gap: 16 }}>
              <div className="grow" style={{ minWidth: 0 }}>
                <div className="row" style={{ gap: 8 }}>
                  <span className={`badge ${badge.cls}`}>
                    {r.status === 'running' ? <span className="live-dot" /> : null}
                    {badge.label}
                  </span>
                  <strong className="truncate">{r.file_name || `${r.total} leads`}</strong>
                </div>
                <div className="muted text-sm" style={{ marginTop: 4 }}>
                  {r.status === 'scheduled'
                    ? `Starts ${new Date(r.scheduled_at || '').toLocaleString(undefined, { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' })} · ${r.total} leads`
                    : `${r.counts.sent} sent · ${r.counts.failed} failed · ${r.counts.pending} left`}
                </div>
              </div>
              <Link to={r.status === 'scheduled' ? '/campaigns/runs' : `/campaigns/runs/${r.id}`} className="btn secondary">
                Open <ArrowRightRegular />
              </Link>
            </div>
            {r.status !== 'scheduled' ? (
              <div className="progress" style={{ marginTop: 12 }}>
                <div className="progress-fill" style={{ width: `${pct}%` }} />
              </div>
            ) : null}
          </section>
        )
      })}

      <div className="kpi-grid">
        <Link to="/inbox" className="kpi">
          <span className={`kpi-icon ${toReview.length ? 'warning' : 'success'}`}><MailWarningRegular /></span>
          <div className="kpi-copy">
            <div className="kpi-label">Drafts to review</div>
            <div className="kpi-value">{dash(conv.isLoading, toReview.length)}</div>
          </div>
        </Link>
        <Link to="/inbox" className="kpi">
          <span className={`kpi-icon ${waiting.length ? 'danger' : 'neutral'}`}><PersonChatRegular /></span>
          <div className="kpi-copy">
            <div className="kpi-label">Clients waiting</div>
            <div className="kpi-value">{dash(conv.isLoading, waiting.length)}</div>
          </div>
        </Link>
        <Link to="/campaigns/runs" className="kpi">
          <span className="kpi-icon teal"><SendRegular /></span>
          <div className="kpi-copy">
            <div className="kpi-label">Emails sent · 30 days</div>
            <div className="kpi-value">{dash(runs.isLoading, sentThisMonth)}</div>
          </div>
        </Link>
        <Link to="/catalogues" className="kpi">
          <span className="kpi-icon neutral"><BoxRegular /></span>
          <div className="kpi-copy">
            <div className="kpi-label">Products indexed</div>
            <div className="kpi-value">{dash(library.isLoading, productCount || (hasCatalogue ? '✓' : 0))}</div>
          </div>
        </Link>
      </div>

      <div className="home-grid">
        <section className="card">
          <CardHeader
            icon={<MailInboxRegular />}
            title="Needs your attention"
            subtitle="Drafts to approve first, then clients who wrote back."
            actions={
              <Link to="/inbox" className="btn subtle sm">
                Inbox <ArrowRightRegular />
              </Link>
            }
          />
          <div className="list" style={{ marginTop: 12 }}>
            {attention.length === 0 && !conv.isLoading ? (
              <EmptyState
                compact
                icon={<CheckmarkCircleFilled />}
                title="Nothing waiting on you"
                description="When a client replies, Starlight drafts an answer and it shows up here for approval."
              />
            ) : null}
            {attention.map((c: any) => {
              const company = convCompany(c)
              const review = convNeedsReview(c)
              return (
                <Link key={c.id} to={`/inbox/${c.id}`} className="list-item">
                  <Avatar name={company} size={32} />
                  <div className="list-item-copy">
                    <span className="list-item-title">{company}</span>
                    <span className="list-item-sub">{c.subject || convSnippet(c) || '(no subject)'}</span>
                  </div>
                  <div className="list-item-meta">
                    <span>{relativeTime(c.updated_at || c.created_at)}</span>
                    <span className={`badge ${review ? 'warning' : 'danger'}`}>{review ? 'Draft to review' : 'Client waiting'}</span>
                  </div>
                </Link>
              )
            })}
          </div>
        </section>

        <section className="card">
          <CardHeader
            icon={<HistoryRegular />}
            title="Recent campaigns"
            actions={
              <Link to="/campaigns/runs" className="btn subtle sm">
                All runs <ArrowRightRegular />
              </Link>
            }
          />
          <div className="list" style={{ marginTop: 12 }}>
            {recentRuns.length === 0 && !runs.isLoading ? (
              <EmptyState
                compact
                icon={<SendRegular />}
                title={liveRuns.length ? 'No finished campaigns yet' : 'No campaigns yet'}
                description="Upload a lead sheet to send your first personalised campaign."
                actions={
                  liveRuns.length ? null : (
                    <Link to="/campaigns" className="btn">
                      <AddRegular /> New campaign
                    </Link>
                  )
                }
              />
            ) : null}
            {recentRuns.map((r) => {
              const badge = RUN_BADGE[r.status] || { label: r.status, cls: '' }
              const pct = r.total ? Math.round((r.counts.sent / r.total) * 100) : 0
              return (
                <Link key={r.id} to={`/campaigns/runs/${r.id}`} className="list-item">
                  <div className="list-item-copy">
                    <span className="list-item-title">{r.file_name || 'Campaign'}</span>
                    <span className="list-item-sub">
                      {r.counts.sent} of {r.total} sent · {relativeTime(r.created_at)}
                    </span>
                    <span className="progress mini">
                      <span className={`progress-fill${r.status === 'done' ? ' success' : ''}`} style={{ display: 'block', width: `${pct}%` }} />
                    </span>
                  </div>
                  <span className={`badge ${badge.cls}`}>{badge.label}</span>
                </Link>
              )
            })}
          </div>
        </section>
      </div>
    </div>
  )
}
