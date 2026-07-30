import { NavLink, Outlet } from 'react-router-dom'
import { useAuth } from '../auth/AuthContext'
import { useState } from 'react'

const salesLinks = [
  { to: '/', label: 'Home', ico: '⌂', end: true },
  { to: '/inbox', label: 'Inbox', ico: '✉' },
  { to: '/campaigns', label: 'Campaigns', ico: '🚀' },
  { to: '/catalogues', label: 'Catalogues', ico: '📚' },
  { to: '/settings', label: 'Settings', ico: '⚙' },
]

const adminLinks = [
  { to: '/admin/prompts', label: 'Prompt Studio', ico: '✨' },
  { to: '/admin/templates', label: 'Email Design', ico: '🖌' },
  { to: '/admin/rag', label: 'RAG Lab', ico: '🔬' },
]

export function AppShell() {
  const { me, logout } = useAuth()
  const [adminOpen, setAdminOpen] = useState(false)
  const initial = (me?.name || me?.email || 'S').trim().charAt(0).toUpperCase()

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand-block">
          <div className="brand-mark">S</div>
          <div>
            <div className="brand"><span>Starlight</span></div>
            <div className="muted" style={{ fontSize: 12, fontWeight: 600 }}>AI Mailer · Linear LED</div>
          </div>
        </div>

        <div>
          <div className="nav-section-label">Workspace</div>
          <nav className="stack" style={{ gap: 4 }}>
            {salesLinks.map((l) => (
              <NavLink
                key={l.to}
                to={l.to}
                end={l.end}
                className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}
              >
                <span className="ico">{l.ico}</span>
                {l.label}
              </NavLink>
            ))}
          </nav>
        </div>

        <div>
          <button
            type="button"
            className="nav-section-label"
            style={{ background: 'none', border: 0, cursor: 'pointer', paddingLeft: '0.75rem' }}
            onClick={() => setAdminOpen((v) => !v)}
          >
            Admin {adminOpen ? '▾' : '▸'}
          </button>
          {adminOpen ? (
            <nav className="stack" style={{ gap: 4 }}>
              {adminLinks.map((l) => (
                <NavLink
                  key={l.to}
                  to={l.to}
                  className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}
                >
                  <span className="ico">{l.ico}</span>
                  {l.label}
                </NavLink>
              ))}
            </nav>
          ) : null}
        </div>

        <div style={{ marginTop: 'auto' }} className="stack">
          <div className="user-chip">
            <div className="avatar">{initial}</div>
            <div style={{ minWidth: 0 }}>
              <div style={{ fontWeight: 700, fontSize: 13, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                {me?.name || 'Starlight user'}
              </div>
              <div className="muted" style={{ fontSize: 11, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                {me?.email}
              </div>
            </div>
          </div>
          <button className="btn secondary" onClick={logout}>Sign out</button>
        </div>
      </aside>
      <main className="main">
        <Outlet />
      </main>
    </div>
  )
}
