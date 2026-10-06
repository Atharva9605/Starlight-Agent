import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { useEffect, useMemo, useState } from 'react'
import {
  ArrowSyncRegular,
  MailInboxRegular,
  PlugDisconnectedRegular,
  SearchRegular,
  SendRegular,
  SettingsRegular,
} from '@fluentui/react-icons'
import { api } from '../api/client'
import { Avatar, EmptyState, MessageBar, PageHeader, relativeTime, useToast } from '../components/ui'
import { convBadge, convClientWaiting, convCompany, convEmail, convNeedsReview, convSnippet } from '../inbox/conversation'

type Filter = 'review' | 'waiting' | 'all'

export function InboxPage() {
  const toast = useToast()
  const [syncing, setSyncing] = useState(false)
  const [error, setError] = useState('')
  const [filter, setFilter] = useState<Filter | null>(null)
  const [search, setSearch] = useState('')
  const q = useQuery({
    queryKey: ['conversations'],
    queryFn: () => api.conversations(true),
  })
  const gmail = useQuery({ queryKey: ['gmail-status'], queryFn: api.gmailStatus })

  const sync = async () => {
    setSyncing(true)
    setError('')
    try {
      const res: any = await api.gmailSync()
      toast.success('Mailbox synced', `${res.processed ?? 0} new message${res.processed === 1 ? '' : 's'}`)
      await q.refetch()
    } catch (e: any) {
      setError(e.message)
    } finally {
      setSyncing(false)
    }
  }

  const conversations = q.data?.conversations || []
  const reviewCount = conversations.filter(convNeedsReview).length
  const waitingCount = conversations.filter(convClientWaiting).length

  // Land on whatever needs action, once the list first arrives.
  useEffect(() => {
    if (filter || !q.data) return
    setFilter(reviewCount ? 'review' : waitingCount ? 'waiting' : 'all')
  }, [q.data, filter, reviewCount, waitingCount])
  const active: Filter = filter || 'all'

  const visible = useMemo(() => {
    const term = search.trim().toLowerCase()
    return conversations.filter((c: any) => {
      if (active === 'review' && !convNeedsReview(c)) return false
      if (active === 'waiting' && !convClientWaiting(c)) return false
      if (!term) return true
      const hay = [c.subject, convCompany(c), convEmail(c), convSnippet(c)]
        .filter(Boolean)
        .join(' ')
        .toLowerCase()
      return hay.includes(term)
    })
  }, [conversations, active, search])

  const tabs: { id: Filter; label: string; count: number }[] = [
    { id: 'review', label: 'Drafts to review', count: reviewCount },
    { id: 'waiting', label: 'Client waiting', count: waitingCount },
    { id: 'all', label: 'All', count: conversations.length },
  ]

  return (
    <div>
      <PageHeader
        title="Inbox"
        subtitle="Client replies to your campaigns. Starlight drafts an answer; nothing is sent until you approve it."
        actions={
          <button className="btn secondary" type="button" onClick={sync} disabled={syncing || (gmail.data?.connected === false && gmail.data?.mode !== 'platform')}>
            <ArrowSyncRegular /> {syncing ? 'Syncing…' : 'Sync Gmail'}
          </button>
        }
      />

      {gmail.data && !gmail.data.connected && gmail.data.mode !== 'platform' ? (
        <div className="page-alerts">
          <MessageBar
            intent="warning"
            title="Gmail isn't connected"
            actions={
              <Link to="/settings/gmail" className="btn secondary sm">
                <PlugDisconnectedRegular /> Connect Gmail
              </Link>
            }
          >
            Client replies can't sync until a mailbox is connected.
          </MessageBar>
        </div>
      ) : null}

      {error ? (
        <div className="page-alerts">
          <MessageBar intent="error" title="Sync failed" onDismiss={() => setError('')}>
            {error}
          </MessageBar>
        </div>
      ) : null}

      <section className="card flush">
        <div className="inbox-toolbar">
          <div className="tablist" role="tablist" aria-label="Filter conversations">
            {tabs.map((t) => (
              <button
                key={t.id}
                type="button"
                role="tab"
                aria-selected={active === t.id}
                className={`tab${active === t.id ? ' active' : ''}`}
                onClick={() => setFilter(t.id)}
              >
                {t.label}
                <span className="tab-count">{t.count}</span>
              </button>
            ))}
          </div>
          <div className="input-wrap inbox-search">
            <SearchRegular />
            <input
              className="input"
              type="search"
              placeholder="Search subject, company or email"
              aria-label="Search conversations"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />
          </div>
        </div>

        <div className="inbox-list">
          {q.isLoading
            ? Array.from({ length: 5 }).map((_, i) => (
                <div key={i} className="inbox-item" aria-hidden>
                  <span className="skeleton circle" style={{ width: 40, height: 40 }} />
                  <div className="stack tight">
                    <span className="skeleton line" style={{ width: '40%' }} />
                    <span className="skeleton line" style={{ width: '70%' }} />
                  </div>
                  <span className="skeleton line" style={{ width: 48 }} />
                </div>
              ))
            : null}

          {!q.isLoading && conversations.length === 0 ? (
            <EmptyState
              icon={<MailInboxRegular />}
              title="No client threads yet"
              description="Connect Gmail and run a campaign — replies land here with AI drafts ready to review."
              actions={
                <>
                  <Link to="/campaigns/new" className="btn">
                    <SendRegular /> Start a campaign
                  </Link>
                  <Link to="/settings" className="btn secondary">
                    <SettingsRegular /> Open settings
                  </Link>
                </>
              }
            />
          ) : null}

          {!q.isLoading && conversations.length > 0 && visible.length === 0 ? (
            <EmptyState
              compact
              icon={<SearchRegular />}
              title={search ? 'Nothing matches' : active === 'review' ? 'No drafts waiting' : 'No clients waiting'}
              description={search ? 'Try a different search or switch to another tab.' : "You're all caught up here. Check the All tab for older threads."}
            />
          ) : null}

          {visible.map((c: any) => {
            const company = convCompany(c)
            const badge = convBadge(c)
            const snippet = convSnippet(c)
            return (
              <Link
                key={c.id}
                to={`/inbox/${c.id}`}
                className={`inbox-item${convNeedsReview(c) || convClientWaiting(c) ? ' unread' : ''}`}
              >
                <Avatar name={String(company)} size={40} />
                <div style={{ minWidth: 0 }}>
                  <div className="inbox-from">{company}</div>
                  <div className="inbox-subject">{c.subject || '(no subject)'}</div>
                  {snippet ? <div className="inbox-snippet">{snippet}</div> : null}
                </div>
                <div className="inbox-meta">
                  <span>{relativeTime(c.updated_at || c.created_at)}</span>
                  <span className={`badge ${badge.cls}`}>{badge.label}</span>
                </div>
              </Link>
            )
          })}
        </div>
      </section>
    </div>
  )
}
