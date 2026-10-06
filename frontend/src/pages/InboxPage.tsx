import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { useMemo, useState } from 'react'
import {
  ArrowSyncRegular,
  MailInboxRegular,
  SearchRegular,
  SendRegular,
  SettingsRegular,
} from '@fluentui/react-icons'
import { api } from '../api/client'
import { Avatar, EmptyState, MessageBar, PageHeader, relativeTime, useToast } from '../components/ui'

type Filter = 'all' | 'review' | 'done'

function needsReview(c: any) {
  return Boolean(c.has_draft || c.pending_draft)
}

function isDone(c: any) {
  return c.status === 'closed' || c.status === 'sent'
}

function statusBadge(c: any) {
  if (needsReview(c)) return { label: 'Needs review', cls: 'warning' }
  if (isDone(c)) return { label: 'Done', cls: 'success' }
  return { label: c.status || 'Open', cls: '' }
}

export function InboxPage() {
  const toast = useToast()
  const [syncing, setSyncing] = useState(false)
  const [error, setError] = useState('')
  const [filter, setFilter] = useState<Filter>('all')
  const [search, setSearch] = useState('')
  const q = useQuery({
    queryKey: ['conversations'],
    queryFn: () => api.conversations(true),
  })

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
  const reviewCount = conversations.filter(needsReview).length
  const doneCount = conversations.filter(isDone).length

  const visible = useMemo(() => {
    const term = search.trim().toLowerCase()
    return conversations.filter((c: any) => {
      if (filter === 'review' && !needsReview(c)) return false
      if (filter === 'done' && !isDone(c)) return false
      if (!term) return true
      const hay = [c.subject, c.client?.company, c.client?.email, c.client_email, c.last_message_preview, c.snippet]
        .filter(Boolean)
        .join(' ')
        .toLowerCase()
      return hay.includes(term)
    })
  }, [conversations, filter, search])

  const tabs: { id: Filter; label: string; count: number }[] = [
    { id: 'all', label: 'All', count: conversations.length },
    { id: 'review', label: 'Needs review', count: reviewCount },
    { id: 'done', label: 'Done', count: doneCount },
  ]

  return (
    <div>
      <PageHeader
        title="Inbox"
        subtitle="Client threads with Starlight AI drafts ready for your approval."
        actions={
          <button className="btn" type="button" onClick={sync} disabled={syncing}>
            <ArrowSyncRegular /> {syncing ? 'Syncing…' : 'Sync Gmail'}
          </button>
        }
      />

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
                aria-selected={filter === t.id}
                className={`tab${filter === t.id ? ' active' : ''}`}
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
                  <Link to="/campaigns" className="btn">
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
              title="Nothing matches"
              description="Try a different search or switch to another tab."
            />
          ) : null}

          {visible.map((c: any) => {
            const company = c.client?.company || c.client?.email || c.client_email || 'Client'
            const badge = statusBadge(c)
            const snippet = c.last_message_preview || c.snippet || ''
            return (
              <Link
                key={c.id}
                to={`/inbox/${c.id}`}
                className={`inbox-item${needsReview(c) ? ' unread' : ''}`}
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
