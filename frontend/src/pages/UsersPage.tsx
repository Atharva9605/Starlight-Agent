import { useState, type FormEvent } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Navigate } from 'react-router-dom'
import { api, type OrgMember } from '../api/client'
import { useAuth } from '../auth/AuthContext'

const ROLE_OPTIONS = [
  { value: 'member', label: 'Member', blurb: 'Inbox, campaigns, catalogues' },
  { value: 'admin', label: 'Admin', blurb: 'Everything members can do, plus users & admin tools' },
]

function rolePill(role: string) {
  if (role === 'owner') return 'ok'
  if (role === 'admin') return 'warn'
  return ''
}

export function UsersPage() {
  const { me } = useAuth()
  const qc = useQueryClient()
  const canManage = me?.role === 'owner' || me?.role === 'admin'

  const [email, setEmail] = useState('')
  const [name, setName] = useState('')
  const [password, setPassword] = useState('')
  const [role, setRole] = useState('member')
  const [formMsg, setFormMsg] = useState('')

  const members = useQuery({
    queryKey: ['org-members'],
    queryFn: api.orgMembers,
    enabled: canManage,
  })

  const add = useMutation({
    mutationFn: (body: {
      email: string
      password?: string
      name?: string
      role?: string
    }) => api.addOrgMember(body),
    onSuccess: (res) => {
      const m = res.member
      setFormMsg(
        m.created
          ? `Created ${m.email} and added as ${m.role}`
          : m.action === 'role_updated'
            ? `Updated ${m.email} to ${m.role}`
            : `Added ${m.email} as ${m.role}`,
      )
      setEmail('')
      setName('')
      setPassword('')
      setRole('member')
      qc.invalidateQueries({ queryKey: ['org-members'] })
    },
    onError: (e: Error) => setFormMsg(e.message),
  })

  if (!canManage) {
    return <Navigate to="/" replace />
  }

  const onSubmit = (e: FormEvent) => {
    e.preventDefault()
    setFormMsg('')
    add.mutate({ email, name, password, role })
  }

  const list = members.data?.members || []

  return (
    <div className="studio-screen">
      <div className="page-hero">
        <div>
          <h1>Users</h1>
          <p>People who can sign in to this Starlight workspace.</p>
        </div>
        <span className="pill">{list.length} {list.length === 1 ? 'user' : 'users'}</span>
      </div>

      <div className="studio-body">
        <aside className="studio-nav panel stack">
          <div className="studio-nav-label muted">Team</div>
          {members.isLoading ? <div className="muted">Loading…</div> : null}
          {members.isError ? (
            <div className="alert danger">{(members.error as Error).message}</div>
          ) : null}
          {!members.isLoading && list.length === 0 ? (
            <div className="empty-state" style={{ padding: '1rem 0.25rem' }}>
              <strong>No members yet</strong>
              <p className="muted" style={{ margin: '0.35rem 0 0', fontSize: 13 }}>
                Add someone with the form on the right.
              </p>
            </div>
          ) : null}
          <div className="studio-nav-list">
            {list.map((m: OrgMember) => {
              const initial = (m.name || m.email).trim().charAt(0).toUpperCase()
              return (
                <div key={m.id} className="list-row" style={{ alignItems: 'center', padding: '0.55rem 0.45rem' }}>
                  <div className="avatar" style={{ width: 34, height: 34, fontSize: 13 }}>
                    {initial}
                  </div>
                  <div style={{ minWidth: 0, flex: 1 }}>
                    <div style={{ fontWeight: 700, fontSize: 13.5 }}>{m.name || m.email}</div>
                    <div className="muted" style={{ fontSize: 12 }}>{m.email}</div>
                  </div>
                  <span className={`pill ${rolePill(m.role)}`}>{m.role}</span>
                </div>
              )
            })}
          </div>
        </aside>

        <form className="studio-editor panel stack" onSubmit={onSubmit}>
          <div className="studio-editor-head">
            <div>
              <h2>Add a user</h2>
              <p>They can sign in immediately with the password you set. Share it out of band.</p>
            </div>
            <button className="btn" type="submit" disabled={add.isPending}>
              {add.isPending ? 'Adding…' : 'Add user'}
            </button>
          </div>

          <label className="field">
            <span>Email</span>
            <input
              className="input"
              type="email"
              required
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
            <span>Temporary password</span>
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
            <span className="muted" style={{ fontSize: 12 }}>
              Required for new accounts. Ignored if this email already exists.
            </span>
          </label>

          <fieldset className="field" style={{ border: 0, padding: 0, margin: 0 }}>
            <span>Role</span>
            <div className="stack" style={{ gap: 6, marginTop: 6 }}>
              {ROLE_OPTIONS.map((opt) => (
                <button
                  key={opt.value}
                  type="button"
                  className={`role-option${role === opt.value ? ' selected' : ''}`}
                  onClick={() => setRole(opt.value)}
                >
                  <span>
                    <strong>{opt.label}</strong>
                    <span className="muted" style={{ display: 'block' }}>{opt.blurb}</span>
                  </span>
                  <span className="role-check">{role === opt.value ? '✓' : ''}</span>
                </button>
              ))}
            </div>
          </fieldset>

          {formMsg ? (
            <div className={`alert ${add.isError ? 'danger' : 'warn'}`}>{formMsg}</div>
          ) : null}
        </form>
      </div>
    </div>
  )
}
