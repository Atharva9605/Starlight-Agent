import { Link, NavLink, Outlet, useLocation, useNavigate } from 'react-router-dom'
import { useEffect, useMemo, useRef, useState } from 'react'
import {
  bundleIcon,
  HomeFilled,
  HomeRegular,
  MailInboxFilled,
  MailInboxRegular,
  SendFilled,
  SendRegular,
  HistoryFilled,
  HistoryRegular,
  LibraryFilled,
  LibraryRegular,
  SettingsFilled,
  SettingsRegular,
  PeopleFilled,
  PeopleRegular,
  SparkleFilled,
  SparkleRegular,
  PaintBrushFilled,
  PaintBrushRegular,
  WandFilled,
  WandRegular,
  DocumentSearchFilled,
  DocumentSearchRegular,
  NavigationRegular,
  SearchRegular,
  SignOutRegular,
  ShieldRegular,
  DocumentTextRegular,
  ChevronDownRegular,
  QuestionCircleRegular,
} from '@fluentui/react-icons'
import { useAuth } from '../auth/AuthContext'
import { useCampaign } from '../campaign/CampaignContext'
import { Avatar, BrandMark } from './ui'

const Home = bundleIcon(HomeFilled, HomeRegular)
const Inbox = bundleIcon(MailInboxFilled, MailInboxRegular)
const Campaigns = bundleIcon(SendFilled, SendRegular)
const Runs = bundleIcon(HistoryFilled, HistoryRegular)
const Catalogues = bundleIcon(LibraryFilled, LibraryRegular)
const Settings = bundleIcon(SettingsFilled, SettingsRegular)
const Users = bundleIcon(PeopleFilled, PeopleRegular)
const Prompts = bundleIcon(SparkleFilled, SparkleRegular)
const Design = bundleIcon(PaintBrushFilled, PaintBrushRegular)
const Create = bundleIcon(WandFilled, WandRegular)
const Rag = bundleIcon(DocumentSearchFilled, DocumentSearchRegular)

type NavItem = {
  to: string
  label: string
  icon: typeof Home
  end?: boolean
  keywords?: string
}

const workspaceLinks: NavItem[] = [
  { to: '/', label: 'Home', icon: Home, end: true, keywords: 'dashboard overview' },
  { to: '/inbox', label: 'Inbox', icon: Inbox, keywords: 'replies threads drafts mail' },
  { to: '/campaigns', label: 'Campaigns', icon: Campaigns, end: true, keywords: 'new outreach leads send' },
  { to: '/campaigns/runs', label: 'Campaign runs', icon: Runs, keywords: 'history scheduled' },
  { to: '/catalogues', label: 'Catalogues', icon: Catalogues, keywords: 'pdf products library upload' },
  { to: '/settings', label: 'Settings', icon: Settings, keywords: 'gmail sender profile' },
]

const adminLinks: NavItem[] = [
  { to: '/admin/users', label: 'Users', icon: Users, keywords: 'team members invite' },
  { to: '/admin/prompts', label: 'Prompt studio', icon: Prompts, keywords: 'ai prompts tone' },
  { to: '/admin/templates', label: 'Email design', icon: Design, end: true, keywords: 'templates html' },
  { to: '/admin/templates/create', label: 'Create with AI', icon: Create, keywords: 'template generate' },
  { to: '/admin/rag', label: 'RAG lab', icon: Rag, keywords: 'grounding search chunks' },
]

function NavGroup({ items, collapsed, onNavigate }: { items: NavItem[]; collapsed: boolean; onNavigate: () => void }) {
  return (
    <ul className="nav-list">
      {items.map((l) => {
        const Icon = l.icon
        return (
          <li key={l.to}>
            <NavLink
              to={l.to}
              end={l.end}
              title={collapsed ? l.label : undefined}
              onClick={onNavigate}
              className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}
            >
              {({ isActive }) => (
                <>
                  <Icon className="nav-ico" filled={isActive} />
                  <span className="nav-label">{l.label}</span>
                </>
              )}
            </NavLink>
          </li>
        )
      })}
    </ul>
  )
}

function useOutsideClose(open: boolean, close: () => void) {
  const ref = useRef<HTMLDivElement | null>(null)
  useEffect(() => {
    if (!open) return
    const onDown = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) close()
    }
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') close()
    }
    document.addEventListener('mousedown', onDown)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', onDown)
      document.removeEventListener('keydown', onKey)
    }
  }, [open, close])
  return ref
}

/** Header search — jumps to any page, like the Microsoft 365 search box. */
function CommandSearch({ items }: { items: NavItem[] }) {
  const nav = useNavigate()
  const [query, setQuery] = useState('')
  const [open, setOpen] = useState(false)
  const [active, setActive] = useState(0)
  const inputRef = useRef<HTMLInputElement | null>(null)
  const wrapRef = useOutsideClose(open, () => setOpen(false))

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault()
        inputRef.current?.focus()
        setOpen(true)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  const results = useMemo(() => {
    const q = query.trim().toLowerCase()
    if (!q) return items
    return items.filter((i) => `${i.label} ${i.keywords || ''}`.toLowerCase().includes(q))
  }, [items, query])

  const go = (item?: NavItem) => {
    if (!item) return
    nav(item.to)
    setQuery('')
    setOpen(false)
    inputRef.current?.blur()
  }

  return (
    <div className="topbar-search" ref={wrapRef}>
      <SearchRegular className="topbar-search-icon" />
      <input
        ref={inputRef}
        type="search"
        placeholder="Search pages"
        aria-label="Search pages"
        value={query}
        onFocus={() => setOpen(true)}
        onChange={(e) => {
          setQuery(e.target.value)
          setActive(0)
          setOpen(true)
        }}
        onKeyDown={(e) => {
          if (e.key === 'ArrowDown') {
            e.preventDefault()
            setActive((a) => Math.min(a + 1, results.length - 1))
          } else if (e.key === 'ArrowUp') {
            e.preventDefault()
            setActive((a) => Math.max(a - 1, 0))
          } else if (e.key === 'Enter') {
            go(results[active])
          }
        }}
      />
      <kbd className="topbar-kbd">Ctrl K</kbd>
      {open ? (
        <div className="menu search-menu" role="listbox">
          {results.length ? (
            results.map((r, i) => {
              const Icon = r.icon
              return (
                <button
                  key={r.to}
                  type="button"
                  role="option"
                  aria-selected={i === active}
                  className={`menu-item${i === active ? ' active' : ''}`}
                  onMouseEnter={() => setActive(i)}
                  onClick={() => go(r)}
                >
                  <Icon className="menu-ico" />
                  {r.label}
                </button>
              )
            })
          ) : (
            <div className="menu-empty">No pages match “{query}”</div>
          )}
        </div>
      ) : null}
    </div>
  )
}

function AccountMenu() {
  const { me, logout } = useAuth()
  const [open, setOpen] = useState(false)
  const ref = useOutsideClose(open, () => setOpen(false))
  const name = me?.name || me?.email || 'Starlight user'

  return (
    <div className="account" ref={ref}>
      <button
        type="button"
        className="topbar-btn account-btn"
        aria-haspopup="menu"
        aria-expanded={open}
        aria-label="Account manager"
        onClick={() => setOpen((v) => !v)}
      >
        <Avatar name={name} size={30} />
      </button>
      {open ? (
        <div className="menu account-menu" role="menu">
          <div className="account-card">
            <Avatar name={name} size={56} />
            <div className="account-meta">
              <strong>{me?.name || 'Starlight user'}</strong>
              <span>{me?.email}</span>
              {me?.role ? <span className="badge brand">{me.role}</span> : null}
            </div>
          </div>
          <div className="menu-divider" />
          <Link to="/settings" role="menuitem" className="menu-item" onClick={() => setOpen(false)}>
            <SettingsRegular className="menu-ico" /> Settings
          </Link>
          <Link to="/privacy" role="menuitem" className="menu-item" onClick={() => setOpen(false)}>
            <ShieldRegular className="menu-ico" /> Privacy policy
          </Link>
          <Link to="/terms" role="menuitem" className="menu-item" onClick={() => setOpen(false)}>
            <DocumentTextRegular className="menu-ico" /> Terms of service
          </Link>
          <div className="menu-divider" />
          <button type="button" role="menuitem" className="menu-item" onClick={logout}>
            <SignOutRegular className="menu-ico" /> Sign out
          </button>
        </div>
      ) : null}
    </div>
  )
}

function readCollapsed(): boolean {
  try {
    return localStorage.getItem('nav-collapsed') === '1'
  } catch {
    return false
  }
}

export function AppShell() {
  const { me } = useAuth()
  const { status } = useCampaign()
  const location = useLocation()
  const [collapsed, setCollapsed] = useState(readCollapsed)
  const [mobileOpen, setMobileOpen] = useState(false)
  const [adminOpen, setAdminOpen] = useState(true)
  const isAdmin = me?.role === 'owner' || me?.role === 'admin'
  const campaignLive = status === 'running' || status === 'reviewing' || status === 'paused'

  useEffect(() => setMobileOpen(false), [location.pathname])

  const toggleNav = () => {
    if (window.matchMedia('(max-width: 960px)').matches) {
      setMobileOpen((v) => !v)
      return
    }
    setCollapsed((v) => {
      try {
        localStorage.setItem('nav-collapsed', v ? '0' : '1')
      } catch {
        /* per-viewer convenience only */
      }
      return !v
    })
  }

  const searchable = isAdmin ? [...workspaceLinks, ...adminLinks] : workspaceLinks
  const closeMobile = () => setMobileOpen(false)

  return (
    <div className={`app-shell${collapsed ? ' nav-collapsed' : ''}${mobileOpen ? ' nav-mobile-open' : ''}`}>
      <a href="#main" className="skip-link">Skip to content</a>
      <header className="topbar">
        <div className="topbar-left">
          <button type="button" className="topbar-btn" aria-label="Toggle navigation" onClick={toggleNav}>
            <NavigationRegular />
          </button>
          <Link to="/" className="topbar-brand">
            <BrandMark size={24} />
            <span className="topbar-product">Starlight AI Mailer</span>
          </Link>
        </div>
        <CommandSearch items={searchable} />
        <div className="topbar-right">
          {campaignLive ? (
            <Link
              to={status === 'reviewing' ? '/campaigns/review' : '/campaigns/live'}
              className="topbar-live"
              title="A campaign is in progress"
            >
              <span className="live-dot" />
              <span className="topbar-live-label">{status === 'paused' ? 'Campaign paused' : 'Campaign in progress'}</span>
            </Link>
          ) : null}
          <a
            className="topbar-btn"
            href="mailto:info@starlightlinearled.com"
            aria-label="Help and support"
            title="Help and support"
          >
            <QuestionCircleRegular />
          </a>
          <AccountMenu />
        </div>
      </header>

      <div className="shell-body">
        <aside className="sidebar" aria-label="Main navigation">
          <nav className="sidebar-scroll">
            <NavGroup items={workspaceLinks} collapsed={collapsed} onNavigate={closeMobile} />
            {isAdmin ? (
              <div className="nav-group">
                <button
                  type="button"
                  className={`nav-section-toggle${adminOpen ? ' open' : ''}`}
                  onClick={() => setAdminOpen((v) => !v)}
                  aria-expanded={adminOpen}
                >
                  <span className="nav-label">Admin</span>
                  <ChevronDownRegular className="nav-chevron" />
                </button>
                {adminOpen || collapsed ? <NavGroup items={adminLinks} collapsed={collapsed} onNavigate={closeMobile} /> : null}
              </div>
            ) : null}
          </nav>
          <div className="sidebar-foot">
            <NavLink to="/privacy">Privacy</NavLink>
            <span aria-hidden>·</span>
            <NavLink to="/terms">Terms</NavLink>
          </div>
        </aside>
        <div className="nav-scrim" onClick={closeMobile} aria-hidden />
        <main className="main" id="main">
          <Outlet />
        </main>
      </div>
    </div>
  )
}

