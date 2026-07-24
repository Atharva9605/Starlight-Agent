import { NavLink, Outlet, useNavigate } from 'react-router-dom'
import { useAuth } from '../auth/AuthContext'
import { api } from '../api/client'

const links = [
  { to: '/inbox', label: 'Inbox' },
  { to: '/campaigns', label: 'Campaigns' },
  { to: '/knowledge', label: 'Knowledge' },
  { to: '/prompts', label: 'Prompts' },
  { to: '/templates', label: 'Templates' },
  { to: '/settings', label: 'Settings' },
]

export function AppShell() {
  const { me, branding, logout, setToken, refresh } = useAuth()
  const navigate = useNavigate()

  const onSwitch = async (organization_id: string) => {
    const res = await api.switchOrg(organization_id)
    setToken(res.token)
    await refresh()
    navigate('/inbox')
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="row">
          {branding?.logo_url ? (
            <img src={branding.logo_url} alt="" style={{ width: 36, height: 36, borderRadius: 8, objectFit: 'cover' }} />
          ) : null}
          <div>
            <div className="brand">{branding?.display_name || 'CRM Agent'}</div>
            <div className="muted" style={{ fontSize: 12 }}>All-in-one AI CRM</div>
          </div>
        </div>

        <nav className="stack" style={{ gap: 4 }}>
          {links.map((l) => (
            <NavLink
              key={l.to}
              to={l.to}
              className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}
            >
              {l.label}
            </NavLink>
          ))}
        </nav>

        <div style={{ marginTop: 'auto' }} className="stack">
          <select
            className="select"
            value={me?.organization_id || ''}
            onChange={(e) => onSwitch(e.target.value)}
          >
            {(me?.organizations || []).map((o: any) => (
              <option key={o.organization_id} value={o.organization_id}>
                {o.name || o.organization_id}
              </option>
            ))}
          </select>
          <div className="muted" style={{ fontSize: 12 }}>{me?.email}</div>
          <button className="btn secondary" onClick={logout}>Sign out</button>
        </div>
      </aside>
      <main className="main">
        <Outlet />
      </main>
    </div>
  )
}
