import { useEffect, useMemo, useState, type FormEvent } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Navigate } from 'react-router-dom'
import {
  CheckmarkCircleFilled,
  DismissRegular,
  PeopleRegular,
  PersonAddRegular,
  SearchRegular,
} from '@fluentui/react-icons'
import { api, type OrgMember } from '../api/client'
import { useAuth } from '../auth/AuthContext'
import { Avatar, EmptyState, MessageBar, PageHeader, Spinner, relativeTime, useToast } from '../components/ui'

const ROLE_OPTIONS = [
  { value: 'member', label: 'Member', blurb: 'Inbox, campaigns and catalogues' },
  { value: 'admin', label: 'Admin', blurb: 'Everything members can do, plus users and admin tools' },
]

function roleBadge(role: string) {
  if (role === 'owner') return 'purple'
  if (role === 'admin') return 'brand'
  return ''
}

function AddUserDrawer({ onClose }: { onClose: () => void }) {
  const qc = useQueryClient()
  const toast = useToast()
  const [email, setEmail] = useState('')
  const [name, setName] = useState('')
  const [password, setPassword] = useState('')
  const [role, setRole] = useState('member')
  const [error, setError] = useState('')

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  const add = useMutation({
    mutationFn: (body: { email: string; password?: string; name?: string; role?: string }) =>
      api.addOrgMember(body),
    onSuccess: (res) => {
      const m = res.member
      toast.success(
        m.created ? 'User added' : m.action === 'role_updated' ? 'Role updated' : 'User added',
        m.created
          ? `Created ${m.email} as ${m.role}.`
          : m.action === 'role_updated'
            ? `${m.email} is now ${m.role}.`
            : `${m.email} joined as ${m.role}.`,
      )
      qc.invalidateQueries({ queryKey: ['org-members'] })
      onClose()
    },
    onError: (e: Error) => setError(e.message),
  })

  const onSubmit = (e: FormEvent) => {
    e.preventDefault()
    setError('')
    add.mutate({ email, name, password, role })
  }

  return (
    <div className="drawer-layer">
      <div className="drawer-backdrop" onClick={onClose} />
      <form className="drawer" role="dialog" aria-modal="true" aria-labelledby="add-user-title" onSubmit={onSubmit}>
        <div className="drawer-head">
          <h2 id="add-user-title">Add a user</h2>
          <button type="button" className="btn subtle icon-only" aria-label="Close" onClick={onClose}>
            <DismissRegular />
          </button>
        </div>
        <div className="drawer-body">
          <p className="muted">
            They can sign in straight away with the password you set. Share it with them privately.
          </p>
          <label className="field">
            <span>Email <span className="req" style={{ color: 'var(--danger-fg)' }}>*</span></span>
            <input
              className="input"
              type="email"
              required
              autoFocus
              autoComplete="off"
              placeholder="colleague@starlightlinearled.com"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
            />
          </label>
          <label className="field">
            <span>Display name</span>
            <input
              className="input"
              type="text"
              placeholder="Optional"
              value={name}
              onChange={(e) => setName(e.target.value)}
            />
          </label>
          <label className="field">
            <span>Temporary password <span style={{ color: 'var(--danger-fg)' }}>*</span></span>
            <input
              className="input"
              type="text"
              autoComplete="new-password"
              placeholder="At least 8 characters"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              minLength={8}
              required
            />
            <span className="field-hint">Required for new accounts. Ignored if this email already has an account.</span>
          </label>
          <fieldset className="field" style={{ border: 0, padding: 0, margin: 0 }}>
            <legend className="field-label" style={{ marginBottom: 6 }}>Role</legend>
            <div className="stack tight" role="radiogroup">
              {ROLE_OPTIONS.map((opt) => (
                <button
                  key={opt.value}
                  type="button"
                  role="radio"
                  aria-checked={role === opt.value}
                  className={`choice-card${role === opt.value ? ' selected' : ''}`}
                  onClick={() => setRole(opt.value)}
                >
                  <span className="choice-card-copy">
                    <strong>{opt.label}</strong>
                    <span>{opt.blurb}</span>
                  </span>
                  {role === opt.value ? (
                    <CheckmarkCircleFilled className="choice-card-check" />
                  ) : (
                    <span className="choice-card-ring" />
                  )}
                </button>
              ))}
            </div>
          </fieldset>
          {error ? <MessageBar intent="error" title="Couldn't add user">{error}</MessageBar> : null}
        </div>
        <div className="drawer-foot">
          <button className="btn" type="submit" disabled={add.isPending}>
            {add.isPending ? <Spinner size="sm" /> : <PersonAddRegular />}
            {add.isPending ? 'Adding…' : 'Add user'}
          </button>
          <button className="btn secondary" type="button" onClick={onClose}>
            Cancel
          </button>
        </div>
      </form>
    </div>
  )
}

export function UsersPage() {
  const { me } = useAuth()
  const canManage = me?.role === 'owner' || me?.role === 'admin'
  const [adding, setAdding] = useState(false)
  const [search, setSearch] = useState('')

  const members = useQuery({
    queryKey: ['org-members'],
    queryFn: api.orgMembers,
    enabled: canManage,
  })

  const list = members.data?.members || []
  const visible = useMemo(() => {
    const term = search.trim().toLowerCase()
    if (!term) return list
    return list.filter((m: OrgMember) => `${m.name || ''} ${m.email} ${m.role}`.toLowerCase().includes(term))
  }, [list, search])

  if (!me) {
    return (
      <div className="center-fill" style={{ minHeight: '40vh' }}>
        <Spinner label="Loading…" />
      </div>
    )
  }
  if (!canManage) {
    return <Navigate to="/" replace />
  }

  return (
    <div>
      <PageHeader
        breadcrumb={[{ label: 'Admin' }, { label: 'Users' }]}
        title="Users"
        subtitle="People who can sign in to this Starlight workspace."
        actions={
          <button className="btn" type="button" onClick={() => setAdding(true)}>
            <PersonAddRegular /> Add user
          </button>
        }
      />

      {members.isError ? (
        <div className="page-alerts">
          <MessageBar intent="error" title="Couldn't load users">{(members.error as Error).message}</MessageBar>
        </div>
      ) : null}

      <section className="card flush">
        <div className="card-section row between">
          <div className="row" style={{ gap: 8 }}>
            <strong>Team</strong>
            <span className="badge">{list.length} {list.length === 1 ? 'user' : 'users'}</span>
            {(['owner', 'admin', 'member'] as const).map((r) => {
              const n = list.filter((m: OrgMember) => m.role === r).length
              return n ? <span key={r} className="muted text-sm hide-sm">{n} {r}{n === 1 ? '' : 's'}</span> : null
            })}
          </div>
          <div className="input-wrap" style={{ width: 'min(280px, 100%)' }}>
            <SearchRegular />
            <input
              className="input"
              type="search"
              placeholder="Search users"
              aria-label="Search users"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />
          </div>
        </div>

        {members.isLoading ? (
          <div className="center-fill" style={{ minHeight: 160 }}>
            <Spinner label="Loading users…" />
          </div>
        ) : list.length === 0 ? (
          <EmptyState
            icon={<PeopleRegular />}
            title="No users yet"
            description="Add teammates so they can review replies and run campaigns."
            actions={
              <button className="btn" type="button" onClick={() => setAdding(true)}>
                <PersonAddRegular /> Add user
              </button>
            }
          />
        ) : (
          <div className="table-wrap" style={{ borderTop: '1px solid var(--stroke-2)' }}>
            <table className="data-table">
              <thead>
                <tr>
                  <th>Name</th>
                  <th>Role</th>
                  <th className="hide-sm">Can do</th>
                  <th className="hide-sm">Joined</th>
                </tr>
              </thead>
              <tbody>
                {visible.map((m: OrgMember) => (
                  <tr key={m.id}>
                    <td>
                      <div className="cell-primary">
                        <Avatar name={m.name || m.email} size={32} />
                        <div style={{ minWidth: 0 }}>
                          <strong className="truncate" style={{ display: 'block' }}>
                            {m.name || m.email.split('@')[0]}
                            {m.email === me?.email ? <span className="muted" style={{ fontWeight: 400 }}> (you)</span> : null}
                          </strong>
                          <span className="cell-sub truncate">{m.email}</span>
                        </div>
                      </div>
                    </td>
                    <td>
                      <span className={`badge ${roleBadge(m.role)}`} style={{ textTransform: 'capitalize' }}>
                        {m.role}
                      </span>
                    </td>
                    <td className="hide-sm muted text-sm">
                      {m.role === 'owner' || m.role === 'admin' ? 'Everything, plus users and admin tools' : 'Inbox, campaigns and catalogues'}
                    </td>
                    <td className="hide-sm muted text-sm" style={{ whiteSpace: 'nowrap' }}>
                      {relativeTime(m.joined_at || m.created_at) || '—'}
                    </td>
                  </tr>
                ))}
                {visible.length === 0 ? (
                  <tr>
                    <td colSpan={4} className="muted" style={{ textAlign: 'center' }}>
                      No users match “{search}”
                    </td>
                  </tr>
                ) : null}
              </tbody>
            </table>
          </div>
        )}
      </section>

      {adding ? <AddUserDrawer onClose={() => setAdding(false)} /> : null}
    </div>
  )
}
