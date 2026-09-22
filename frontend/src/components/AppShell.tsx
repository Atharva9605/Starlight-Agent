import { NavLink, Outlet } from 'react-router-dom'
import { useAuth } from '../auth/AuthContext'
import { useState, type ReactNode } from 'react'

type NavItem = { to: string; label: string; icon: ReactNode; end?: boolean }

function Icon({ children }: { children: ReactNode }) {
  return (
    <svg className="nav-ico" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      {children}
    </svg>
  )
}

const I = {
  home: (
    <Icon>
      <path d="M3 10.5 12 3l9 7.5" />
      <path d="M5 9.5V20h14V9.5" />
    </Icon>
  ),
  inbox: (
    <Icon>
      <path d="M4 6h16v12H4z" />
      <path d="m4 7 8 6 8-6" />
    </Icon>
  ),
  campaigns: (
    <Icon>
      <path d="M5 19V5l14 7-14 7z" />
    </Icon>
  ),
  catalogues: (
    <Icon>
      <path d="M4 5h7v14H4z" />
      <path d="M13 5h7v14h-7z" />
      <path d="M8 8h1M8 12h1M17 8h1M17 12h1" />
    </Icon>
  ),
  settings: (
    <Icon>
      <circle cx="12" cy="12" r="3" />
      <path d="M12 3v2.2M12 18.8V21M3 12h2.2M18.8 12H21M5.6 5.6l1.6 1.6M16.8 16.8l1.6 1.6M18.4 5.6l-1.6 1.6M7.2 16.8l-1.6 1.6" />
    </Icon>
  ),
  users: (
    <Icon>
      <circle cx="9" cy="8" r="3" />
      <path d="M3 19c0-3 2.7-5 6-5s6 2 6 5" />
      <circle cx="17" cy="9" r="2.5" />
      <path d="M21 19c0-2.2-1.8-4-4-4" />
    </Icon>
  ),
  prompts: (
    <Icon>
      <path d="M12 3l1.4 4.2L18 9l-4.6 1.8L12 15l-1.4-4.2L6 9l4.6-1.8L12 3z" />
      <path d="M18 14l.7 2.1L21 17l-2.3.9L18 20l-.7-2.1L15 17l2.3-.9L18 14z" />
    </Icon>
  ),
  design: (
    <Icon>
      <path d="M12 20h9" />
      <path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L8 18l-4 1 1-4 11.5-11.5z" />
    </Icon>
  ),
  create: (
    <Icon>
      <path d="M12 5v14M5 12h14" />
    </Icon>
  ),
  rag: (
    <Icon>
      <circle cx="11" cy="11" r="6" />
      <path d="m16 16 4 4" />
    </Icon>
  ),
  chevron: (
    <svg className="nav-chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden>
      <path d="m6 9 6 6 6-6" />
    </svg>
  ),
}

const salesLinks: NavItem[] = [
  { to: '/', label: 'Home', icon: I.home, end: true },
  { to: '/inbox', label: 'Inbox', icon: I.inbox },
  { to: '/campaigns', label: 'Campaigns', icon: I.campaigns },
  { to: '/catalogues', label: 'Catalogues', icon: I.catalogues },
  { to: '/settings', label: 'Settings', icon: I.settings },
]

const adminLinks: NavItem[] = [
  { to: '/admin/users', label: 'Users', icon: I.users },
  { to: '/admin/prompts', label: 'Prompt Studio', icon: I.prompts },
  { to: '/admin/templates', label: 'Email Design', icon: I.design, end: true },
  { to: '/admin/templates/create', label: 'Create with AI', icon: I.create },
  { to: '/admin/rag', label: 'RAG Lab', icon: I.rag },
]

function NavGroup({ items }: { items: NavItem[] }) {
  return (
    <nav className="nav-list">
      {items.map((l) => (
        <NavLink
          key={l.to}
          to={l.to}
          end={l.end}
          className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}
        >
          <span className="nav-ico-wrap">{l.icon}</span>
          <span className="nav-label">{l.label}</span>
        </NavLink>
      ))}
    </nav>
  )
}

export function AppShell() {
  const { me, logout } = useAuth()
  const [adminOpen, setAdminOpen] = useState(true)
  const initial = (me?.name || me?.email || 'S').trim().charAt(0).toUpperCase()
  const isAdmin = me?.role === 'owner' || me?.role === 'admin'

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand-block">
          <div className="brand-mark">S</div>
          <div className="brand-copy">
            <div className="brand"><span>Starlight</span></div>
            <div className="brand-sub">AI Mailer</div>
          </div>
        </div>

        <div className="sidebar-scroll">
          <div className="nav-group">
            <div className="nav-section-label">Workspace</div>
            <NavGroup items={salesLinks} />
          </div>

          {isAdmin ? (
            <div className="nav-group">
              <button
                type="button"
                className={`nav-section-toggle${adminOpen ? ' open' : ''}`}
                onClick={() => setAdminOpen((v) => !v)}
                aria-expanded={adminOpen}
              >
                <span>Admin</span>
                {I.chevron}
              </button>
              {adminOpen ? <NavGroup items={adminLinks} /> : null}
            </div>
          ) : null}
        </div>

        <div className="sidebar-foot">
          <div className="user-chip">
            <div className="avatar" aria-hidden>{initial}</div>
            <div className="user-meta">
              <div className="user-name">{me?.name || 'Starlight user'}</div>
              <div className="user-email">{me?.email}</div>
              {me?.role ? <div className="user-role">{me.role}</div> : null}
            </div>
          </div>
          <button type="button" className="btn-signout" onClick={logout}>
            Sign out
          </button>
          <div className="sidebar-legal">
            <NavLink to="/privacy">Privacy</NavLink>
            <span aria-hidden>·</span>
            <NavLink to="/terms">Terms</NavLink>
          </div>
        </div>
      </aside>
      <main className="main">
        <Outlet />
      </main>
    </div>
  )
}
