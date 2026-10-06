import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import {
  AddRegular,
  ArrowRightRegular,
  DatabaseRegular,
  HistoryRegular,
  LibraryRegular,
  MailInboxRegular,
  MailWarningRegular,
  PlugDisconnectedRegular,
  SendRegular,
} from '@fluentui/react-icons'
import { api, type CampaignRun } from '../api/client'
import { useAuth } from '../auth/AuthContext'
import { Avatar, CardHeader, EmptyState, MessageBar, PageHeader, relativeTime } from '../components/ui'

const RUN_BADGE: Record<string, { label: string; cls: string }> = {
  scheduled: { label: 'Scheduled', cls: 'purple' },
  running: { label: 'Running', cls: 'brand' },
  paused: { label: 'Paused', cls: 'warning' },
  done: { label: 'Completed', cls: 'success' },
  stopped: { label: 'Stopped', cls: '' },
  failed: { label: 'Failed', cls: 'danger' },
}

function greeting() {
  const h = new Date().getHours()
  if (h < 12) return 'Good morning'
  if (h < 18) return 'Good afternoon'
  return 'Good evening'
}

export function HomePage() {
  const { me } = useAuth()
  const conv = useQuery({ queryKey: ['conversations'], queryFn: () => api.conversations(true) })
  const kb = useQuery({ queryKey: ['kb'], queryFn: api.kbStatus })
  const runs = useQuery({ queryKey: ['campaign-runs'], queryFn: () => api.campaignRuns() })
  const gmail = useQuery({ queryKey: ['gmail-status'], queryFn: api.gmailStatus })

  const conversations = conv.data?.conversations || []
  const pending = conversations.filter((c: any) =>
    String(c.status || '').includes('draft') || c.has_draft || c.pending_draft,
  ).length
  const recentRuns: CampaignRun[] = (runs.data?.runs || []).slice(0, 5)
  const recentThreads = conversations.slice(0, 5)
  const firstName = me?.name?.split(' ')[0] || 'there'

  return (
    <div>
      <PageHeader
        title={`${greeting()}, ${firstName}`}
        subtitle="Review replies, run outreach and keep your catalogues grounded — all in one place."
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

      {gmail.data && !gmail.data.connected ? (
        <div className="page-alerts">
          <MessageBar
            intent="warning"
            title="Gmail isn't connected"
            actions={
              <Link to="/settings" className="btn secondary sm">
                <PlugDisconnectedRegular /> Connect Gmail
              </Link>
            }
          >
            Connect a mailbox so campaigns can send and client replies can sync into the inbox.
          </MessageBar>
        </div>
      ) : null}

      <div className="kpi-grid">
        <Link to="/inbox" className="kpi">
          <span className="kpi-icon"><MailInboxRegular /></span>
          <div className="kpi-copy">
            <div className="kpi-label">Inbox threads</div>
            <div className="kpi-value">{conv.isLoading ? '—' : conversations.length}</div>
          </div>
        </Link>
        <Link to="/inbox" className="kpi">
          <span className={`kpi-icon ${pending ? 'warning' : 'success'}`}><MailWarningRegular /></span>
          <div className="kpi-copy">
            <div className="kpi-label">Needs your review</div>
            <div className="kpi-value">{conv.isLoading ? '—' : pending}</div>
          </div>
        </Link>
        <Link to="/campaigns/runs" className="kpi">
          <span className="kpi-icon teal"><SendRegular /></span>
          <div className="kpi-copy">
            <div className="kpi-label">Campaign runs</div>
            <div className="kpi-value">{runs.isLoading ? '—' : runs.data?.runs.length ?? 0}</div>
          </div>
        </Link>
        <Link to="/catalogues" className="kpi">
          <span className="kpi-icon neutral"><DatabaseRegular /></span>
          <div className="kpi-copy">
            <div className="kpi-label">Catalogue chunks</div>
            <div className="kpi-value">{kb.data?.chunk_count ?? '—'}</div>
          </div>
        </Link>
      </div>

      <div className="home-grid">
        <div className="stack loose">
          <section className="card">
            <CardHeader title="Get started" subtitle="Jump straight into the most common tasks." />
            <div className="quick-actions" style={{ marginTop: 16 }}>
              <Link to="/inbox" className="quick-action">
                <MailInboxRegular className="quick-action-icon" />
                <div>
                  <strong>Review inbox</strong>
                  <span>Approve AI drafts before they leave as Starlight.</span>
                </div>
              </Link>
              <Link to="/campaigns" className="quick-action">
                <SendRegular className="quick-action-icon" />
                <div>
                  <strong>New campaign</strong>
                  <span>Upload leads and stream personalised outreach.</span>
                </div>
              </Link>
              <Link to="/catalogues" className="quick-action">
                <LibraryRegular className="quick-action-icon" />
                <div>
                  <strong>Catalogues</strong>
                  <span>Keep product PDFs indexed so emails stay grounded.</span>
                </div>
              </Link>
            </div>
          </section>

          <section className="card">
            <CardHeader
              title="Recent conversations"
              subtitle="Latest client threads from your mailbox."
              actions={
                <Link to="/inbox" className="btn subtle sm">
                  View all <ArrowRightRegular />
                </Link>
              }
            />
            <div className="list" style={{ marginTop: 12 }}>
              {recentThreads.length === 0 && !conv.isLoading ? (
                <EmptyState
                  compact
                  icon={<MailInboxRegular />}
                  title="No client threads yet"
                  description="Replies to your campaigns will show up here with AI drafts ready to review."
                />
              ) : null}
              {recentThreads.map((c: any) => {
                const company = c.client?.company || c.client?.email || c.client_email || 'Client'
                const needsReview = c.has_draft || c.pending_draft
                return (
                  <Link key={c.id} to={`/inbox/${c.id}`} className="list-item">
                    <Avatar name={String(company)} size={32} />
                    <div className="list-item-copy">
                      <span className="list-item-title">{c.subject || '(no subject)'}</span>
                      <span className="list-item-sub">{company}</span>
                    </div>
                    <div className="list-item-meta">
                      <span>{relativeTime(c.updated_at || c.created_at)}</span>
                      {needsReview ? <span className="badge warning">Needs review</span> : null}
                    </div>
                  </Link>
                )
              })}
            </div>
          </section>
        </div>

        <section className="card">
          <CardHeader
            icon={<HistoryRegular />}
            title="Recent campaign runs"
            actions={
              <Link to="/campaigns/runs" className="btn subtle sm">
                View all <ArrowRightRegular />
              </Link>
            }
          />
          <div className="list" style={{ marginTop: 12 }}>
            {recentRuns.length === 0 && !runs.isLoading ? (
              <EmptyState
                compact
                icon={<SendRegular />}
                title="No campaigns yet"
                description="Upload a lead sheet to send your first personalised campaign."
                actions={
                  <Link to="/campaigns" className="btn">
                    <AddRegular /> New campaign
                  </Link>
                }
              />
            ) : null}
            {recentRuns.map((r) => {
              const badge = RUN_BADGE[r.status] || { label: r.status, cls: '' }
              return (
                <Link key={r.id} to={`/campaigns/runs/${r.id}`} className="list-item">
                  <div className="list-item-copy">
                    <span className="list-item-title">{r.file_name || 'Campaign'}</span>
                    <span className="list-item-sub">
                      {r.counts.sent} of {r.total} sent · {relativeTime(r.scheduled_at || r.created_at)}
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
