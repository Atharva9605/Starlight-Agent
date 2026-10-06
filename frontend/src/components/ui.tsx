import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from 'react'
import { Link } from 'react-router-dom'
import {
  CheckmarkCircleFilled,
  ChevronRightRegular,
  DismissRegular,
  ErrorCircleFilled,
  InfoFilled,
  WarningFilled,
} from '@fluentui/react-icons'

const APP_NAME = 'Starlight AI Mailer'

/** Sets the browser tab title the way Microsoft 365 apps do: "Page - Product". */
export function useDocumentTitle(title?: string) {
  useEffect(() => {
    document.title = title ? `${title} - ${APP_NAME}` : APP_NAME
  }, [title])
}

/** Starlight brand mark: a lit linear-LED bar under a four-point star. */
export function BrandMark({ size = 32 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden className="brand-mark">
      <rect width="32" height="32" rx="7" fill="#0F6CBD" />
      <path d="M16 6.5l1.9 5.6 5.6 1.9-5.6 1.9L16 21.5l-1.9-5.6-5.6-1.9 5.6-1.9z" fill="#fff" />
      <rect x="8" y="23.5" width="16" height="2.5" rx="1.25" fill="#9CCBF5" />
    </svg>
  )
}

export type Crumb = { label: string; to?: string }

export function PageHeader({
  title,
  subtitle,
  breadcrumb,
  actions,
  kicker,
}: {
  title: ReactNode
  subtitle?: ReactNode
  breadcrumb?: Crumb[]
  actions?: ReactNode
  kicker?: ReactNode
}) {
  useDocumentTitle(typeof title === 'string' ? title : undefined)
  return (
    <header className="page-header">
      {breadcrumb?.length ? (
        <nav className="breadcrumb" aria-label="Breadcrumb">
          {breadcrumb.map((c, i) => (
            <span key={i} className="breadcrumb-item">
              {c.to ? <Link to={c.to}>{c.label}</Link> : <span aria-current="page">{c.label}</span>}
              {i < breadcrumb.length - 1 ? <ChevronRightRegular className="breadcrumb-sep" /> : null}
            </span>
          ))}
        </nav>
      ) : null}
      <div className="page-header-row">
        <div className="page-header-copy">
          {kicker ? <div className="page-kicker">{kicker}</div> : null}
          <h1 className="page-title">{title}</h1>
          {subtitle ? <p className="page-subtitle">{subtitle}</p> : null}
        </div>
        {actions ? <div className="page-actions">{actions}</div> : null}
      </div>
    </header>
  )
}

type Intent = 'info' | 'success' | 'warning' | 'error'

const INTENT_ICON: Record<Intent, ReactNode> = {
  info: <InfoFilled />,
  success: <CheckmarkCircleFilled />,
  warning: <WarningFilled />,
  error: <ErrorCircleFilled />,
}

/** Fluent MessageBar — inline, non-blocking status with optional actions. */
export function MessageBar({
  intent = 'info',
  title,
  children,
  actions,
  onDismiss,
  className = '',
}: {
  intent?: Intent
  title?: ReactNode
  children?: ReactNode
  actions?: ReactNode
  onDismiss?: () => void
  className?: string
}) {
  return (
    <div className={`message-bar ${intent} ${className}`} role={intent === 'error' ? 'alert' : 'status'}>
      <span className="message-bar-icon">{INTENT_ICON[intent]}</span>
      <div className="message-bar-body">
        {title ? <strong className="message-bar-title">{title}</strong> : null}
        {children ? <span className="message-bar-text">{children}</span> : null}
      </div>
      {actions ? <div className="message-bar-actions">{actions}</div> : null}
      {onDismiss ? (
        <button type="button" className="btn subtle icon-only sm" aria-label="Dismiss" onClick={onDismiss}>
          <DismissRegular />
        </button>
      ) : null}
    </div>
  )
}

export function EmptyState({
  icon,
  title,
  description,
  actions,
  compact = false,
}: {
  icon?: ReactNode
  title: ReactNode
  description?: ReactNode
  actions?: ReactNode
  compact?: boolean
}) {
  return (
    <div className={`empty-state${compact ? ' compact' : ''}`}>
      {icon ? <div className="empty-icon">{icon}</div> : null}
      <strong className="empty-title">{title}</strong>
      {description ? <p className="empty-desc">{description}</p> : null}
      {actions ? <div className="empty-actions">{actions}</div> : null}
    </div>
  )
}

export function Spinner({ label, size = 'md' }: { label?: ReactNode; size?: 'sm' | 'md' | 'lg' }) {
  return (
    <span className={`spinner-wrap ${size}`} role="progressbar" aria-busy="true">
      <span className="spinner" />
      {label ? <span className="spinner-label">{label}</span> : null}
    </span>
  )
}

export function CardHeader({
  icon,
  title,
  subtitle,
  actions,
}: {
  icon?: ReactNode
  title: ReactNode
  subtitle?: ReactNode
  actions?: ReactNode
}) {
  return (
    <div className="card-header">
      {icon ? <span className="card-header-icon">{icon}</span> : null}
      <div className="card-header-copy">
        <h2 className="card-title">{title}</h2>
        {subtitle ? <p className="card-subtitle">{subtitle}</p> : null}
      </div>
      {actions ? <div className="card-header-actions">{actions}</div> : null}
    </div>
  )
}

export function Switch({
  checked,
  onChange,
  label,
  description,
  disabled,
}: {
  checked: boolean
  onChange: (v: boolean) => void
  label: ReactNode
  description?: ReactNode
  disabled?: boolean
}) {
  return (
    <label className={`switch-row${disabled ? ' disabled' : ''}`}>
      <span className="switch-copy">
        <span className="switch-label">{label}</span>
        {description ? <span className="switch-desc">{description}</span> : null}
      </span>
      <input
        type="checkbox"
        role="switch"
        className="switch"
        checked={checked}
        disabled={disabled}
        onChange={(e) => onChange(e.target.checked)}
      />
    </label>
  )
}

/** Initials avatar with a stable colour per name, like Microsoft 365 personas. */
const PERSONA_COLORS = ['#0F6CBD', '#C239B3', '#038387', '#8764B8', '#CA5010', '#498205', '#4F6BED', '#986F0B', '#D13438', '#00727C']

export function Avatar({ name, size = 32 }: { name: string; size?: number }) {
  const clean = (name || '?').trim()
  const parts = clean.replace(/[@.].*$/, '').split(/[\s_-]+/).filter(Boolean)
  const initials = ((parts[0]?.[0] || '?') + (parts[1]?.[0] || '')).toUpperCase()
  let h = 0
  for (const ch of clean) h = (h * 31 + ch.charCodeAt(0)) | 0
  const bg = PERSONA_COLORS[Math.abs(h) % PERSONA_COLORS.length]
  return (
    <span
      className="avatar"
      style={{ width: size, height: size, fontSize: Math.round(size * 0.4), background: bg }}
      aria-hidden
    >
      {initials}
    </span>
  )
}

/* ---------- Toasts ---------- */

type Toast = { id: number; intent: Intent; title: string; body?: string }
type ToastApi = {
  success: (title: string, body?: string) => void
  error: (title: string, body?: string) => void
  info: (title: string, body?: string) => void
}

const ToastCtx = createContext<ToastApi | null>(null)

/* ---------- Confirm dialog ---------- */

type ConfirmOpts = {
  title: string
  body?: ReactNode
  confirmLabel?: string
  cancelLabel?: string
  danger?: boolean
}
type ConfirmFn = (opts: ConfirmOpts) => Promise<boolean>
const ConfirmCtx = createContext<ConfirmFn | null>(null)

export function UiProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([])
  const nextId = useRef(1)
  const [dialog, setDialog] = useState<(ConfirmOpts & { resolve: (v: boolean) => void }) | null>(null)

  const dismiss = useCallback((id: number) => setToasts((t) => t.filter((x) => x.id !== id)), [])
  const push = useCallback(
    (intent: Intent, title: string, body?: string) => {
      const id = nextId.current++
      setToasts((t) => [...t.slice(-3), { id, intent, title, body }])
      window.setTimeout(() => dismiss(id), intent === 'error' ? 7000 : 4000)
    },
    [dismiss],
  )
  const [toastApi] = useState<ToastApi>(() => ({
    success: (t, b) => push('success', t, b),
    error: (t, b) => push('error', t, b),
    info: (t, b) => push('info', t, b),
  }))

  const confirm = useCallback<ConfirmFn>(
    (opts) => new Promise<boolean>((resolve) => setDialog({ ...opts, resolve })),
    [],
  )
  const close = (v: boolean) => {
    dialog?.resolve(v)
    setDialog(null)
  }

  useEffect(() => {
    if (!dialog) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') close(false)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  })

  return (
    <ToastCtx.Provider value={toastApi}>
      <ConfirmCtx.Provider value={confirm}>
        {children}
        <div className="toaster" aria-live="polite">
          {toasts.map((t) => (
            <div key={t.id} className={`toast ${t.intent}`}>
              <span className="toast-icon">{INTENT_ICON[t.intent]}</span>
              <div className="toast-body">
                <strong>{t.title}</strong>
                {t.body ? <span>{t.body}</span> : null}
              </div>
              <button type="button" className="btn subtle icon-only sm" aria-label="Dismiss" onClick={() => dismiss(t.id)}>
                <DismissRegular />
              </button>
            </div>
          ))}
        </div>
        {dialog ? (
          <div className="dialog-layer">
            <div className="dialog-backdrop" onClick={() => close(false)} />
            <div className="dialog" role="alertdialog" aria-modal="true" aria-labelledby="dialog-title">
              <h2 id="dialog-title" className="dialog-title">{dialog.title}</h2>
              {dialog.body ? <div className="dialog-body">{dialog.body}</div> : null}
              <div className="dialog-actions">
                <button type="button" className="btn secondary" onClick={() => close(false)}>
                  {dialog.cancelLabel || 'Cancel'}
                </button>
                <button
                  type="button"
                  className={`btn ${dialog.danger ? 'danger' : ''}`}
                  autoFocus
                  onClick={() => close(true)}
                >
                  {dialog.confirmLabel || 'OK'}
                </button>
              </div>
            </div>
          </div>
        ) : null}
      </ConfirmCtx.Provider>
    </ToastCtx.Provider>
  )
}

export function useToast(): ToastApi {
  const ctx = useContext(ToastCtx)
  if (!ctx) throw new Error('useToast must be used inside UiProvider')
  return ctx
}

export function useConfirm(): ConfirmFn {
  const ctx = useContext(ConfirmCtx)
  if (!ctx) throw new Error('useConfirm must be used inside UiProvider')
  return ctx
}

/** "5 min ago" / "Yesterday" / "Mar 4" — the relative stamps Outlook uses in lists. */
export function relativeTime(value?: string | null): string {
  if (!value) return ''
  const d = new Date(value)
  if (Number.isNaN(d.getTime())) return ''
  const diff = Date.now() - d.getTime()
  const min = Math.round(diff / 60000)
  if (min < 1 && min > -1) return 'Just now'
  if (min > 0 && min < 60) return `${min} min ago`
  const sameDay = new Date().toDateString() === d.toDateString()
  if (sameDay) return d.toLocaleTimeString(undefined, { hour: 'numeric', minute: '2-digit' })
  const yesterday = new Date(Date.now() - 86400000).toDateString() === d.toDateString()
  if (yesterday) return 'Yesterday'
  const sameYear = new Date().getFullYear() === d.getFullYear()
  return d.toLocaleDateString(undefined, sameYear ? { month: 'short', day: 'numeric' } : { month: 'short', day: 'numeric', year: 'numeric' })
}
