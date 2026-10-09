import { useEffect, useState, type FormEvent } from 'react'
import { Link, useNavigate, useSearchParams } from 'react-router-dom'
import {
  CheckmarkCircleRegular,
  EyeOffRegular,
  EyeRegular,
  LockClosedRegular,
  MailRegular,
  MailInboxCheckmarkRegular,
  PersonRegular,
  SparkleRegular,
  ArrowRightRegular,
} from '@fluentui/react-icons'
import { api } from '../api/client'
import { useAuth } from '../auth/AuthContext'
import { BrandMark, MessageBar, Spinner, useDocumentTitle } from '../components/ui'

/** Exact product name — must match Google OAuth consent screen app name. */
const APP_NAME = 'Starlight AI Mailer'
const COMPANY = 'Starlight Linear LED'

/** Only same-app paths, so a crafted ?next= can't send people off-site. */
function safeNext(value: string | null): string {
  return value && value.startsWith('/') && !value.startsWith('//') ? value : '/'
}

function AuthArt() {
  return (
    <div className="auth-art">
      <div className="auth-art-inner">
        <div className="auth-art-brand">
          <BrandMark size={40} />
          <div>
            <div className="auth-product-name">{APP_NAME}</div>
            <div className="auth-company">by {COMPANY}</div>
          </div>
        </div>

        <h1>AI email outreach for <span className="auth-hl">LED sales teams</span></h1>
        <p className="auth-purpose">
          Draft, review, and send personalized sales emails from your connected Gmail inbox — grounded
          in your product catalogues, with every AI reply approved by a person.
        </p>

        <ul className="auth-features">
          <li><MailInboxCheckmarkRegular /> Connect any Google account and send from that inbox</li>
          <li><SparkleRegular /> Generate catalogue-aware campaign emails with live preview</li>
          <li><CheckmarkCircleRegular /> Approve AI reply drafts before they go out</li>
        </ul>
      </div>
    </div>
  )
}

function PasswordInput({
  value,
  onChange,
  autoComplete,
  minLength,
}: {
  value: string
  onChange: (v: string) => void
  autoComplete: string
  minLength?: number
}) {
  const [show, setShow] = useState(false)
  return (
    <div className="input-wrap">
      <LockClosedRegular />
      <input
        className="input"
        type={show ? 'text' : 'password'}
        autoComplete={autoComplete}
        value={value}
        minLength={minLength}
        onChange={(e) => onChange(e.target.value)}
        required
      />
      <button
        type="button"
        className="btn subtle icon-only sm input-suffix"
        style={{ height: 28, width: 28 }}
        aria-label={show ? 'Hide password' : 'Show password'}
        title={show ? 'Hide password' : 'Show password'}
        onClick={() => setShow((v) => !v)}
      >
        {show ? <EyeOffRegular /> : <EyeRegular />}
      </button>
    </div>
  )
}

function GoogleIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 48 48" aria-hidden>
      <path fill="#FFC107" d="M43.6 20.5H42V20H24v8h11.3C33.7 32.9 29.3 36 24 36c-6.6 0-12-5.4-12-12s5.4-12 12-12c3 0 5.8 1.1 7.9 3l5.7-5.7C34.2 6.1 29.4 4 24 4 12.9 4 4 12.9 4 24s8.9 20 20 20 20-8.9 20-20c0-1.3-.1-2.5-.4-3.5z" />
      <path fill="#FF3D00" d="M6.3 14.7l6.6 4.8C14.7 16.1 19 12 24 12c3 0 5.8 1.1 7.9 3l5.7-5.7C34.2 6.1 29.4 4 24 4 16.3 4 9.6 8.3 6.3 14.7z" />
      <path fill="#4CAF50" d="M24 44c5.2 0 10-2 13.5-5.2l-6.2-5.2C29.3 35.3 26.8 36 24 36c-5.3 0-9.7-3.1-11.3-7.5l-6.5 5C9.4 39.6 16.1 44 24 44z" />
      <path fill="#1976D2" d="M43.6 20.5H42V20H24v8h11.3c-1.1 3.2-3.5 5.7-6.5 7.1l.1.1 6.2 5.2C36.9 39.2 44 34 44 24c0-1.3-.1-2.5-.4-3.5z" />
    </svg>
  )
}

export function LoginPage() {
  const { setToken } = useAuth()
  const navigate = useNavigate()
  const [params] = useSearchParams()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [googleLoading, setGoogleLoading] = useState(false)
  useDocumentTitle('Sign in')

  useEffect(() => {
    const token = params.get('google_token')
    const googleError = params.get('google_error')
    if (googleError) {
      setError(decodeURIComponent(googleError))
      return
    }
    if (token) {
      setToken(token)
      navigate(safeNext(params.get('next')), { replace: true })
    }
  }, [params, setToken, navigate])

  const onSubmit = async (e: FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError('')
    try {
      const res = await api.login({ email, password })
      setToken(res.token)
      navigate(safeNext(params.get('next')), { replace: true })
    } catch (err: any) {
      setError(err.message || 'Login failed')
    } finally {
      setLoading(false)
    }
  }

  const signInWithGoogle = async () => {
    setGoogleLoading(true)
    setError('')
    try {
      const res = await api.googleAuthorize()
      if (!res?.url) {
        setError(res?.message || 'Google sign-in is not configured on the server.')
        return
      }
      window.location.href = res.url
    } catch (err: any) {
      setError(err.message || 'Could not start Google sign-in')
    } finally {
      setGoogleLoading(false)
    }
  }

  return (
    <div className="auth-screen">
      <AuthArt />
      <div className="auth-form-wrap">
        <form className="auth-card" onSubmit={onSubmit}>
          <div className="auth-card-brand">
            <BrandMark size={28} />
            <span>{APP_NAME}</span>
          </div>
          <h2>Sign in</h2>
          <p className="auth-sub">Use your work account to continue to {APP_NAME}.</p>

          <button
            className="btn google lg block"
            type="button"
            disabled={googleLoading || loading}
            onClick={() => void signInWithGoogle()}
          >
            {googleLoading ? <Spinner size="sm" /> : <GoogleIcon />}
            {googleLoading ? 'Redirecting to Google…' : 'Continue with Google'}
          </button>

          <div className="auth-divider">or use email</div>

          <label className="field">
            <span>Work email</span>
            <div className="input-wrap">
              <MailRegular />
              <input
                className="input"
                type="email"
                autoComplete="username"
                placeholder="name@company.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
              />
            </div>
          </label>
          <label className="field">
            <span>Password</span>
            <PasswordInput value={password} onChange={setPassword} autoComplete="current-password" />
          </label>
          {error ? (
            <MessageBar intent="error" title="Could not sign in">
              {error}
            </MessageBar>
          ) : null}
          <button className="btn lg block" type="submit" disabled={loading || googleLoading}>
            {loading ? <Spinner size="sm" /> : null}
            {loading ? 'Signing in…' : 'Sign in'}
            {!loading ? <ArrowRightRegular /> : null}
          </button>
          <div className="auth-switch">
            New to {APP_NAME}?{' '}
            <Link to="/signup" className="link">Create an account</Link>
          </div>
          <p className="auth-legal">
            <Link to="/privacy">Privacy Policy</Link>
            <span aria-hidden> · </span>
            <Link to="/terms">Terms of Service</Link>
          </p>
        </form>
      </div>
    </div>
  )
}

export function SignupPage() {
  const { setToken } = useAuth()
  const navigate = useNavigate()
  const [form, setForm] = useState({ email: '', password: '', name: '', org_name: COMPANY })
  useDocumentTitle('Create account')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [googleLoading, setGoogleLoading] = useState(false)

  const onSubmit = async (e: FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError('')
    try {
      const res = await api.signup({ ...form, org_name: form.org_name || COMPANY })
      setToken(res.token)
      navigate('/', { replace: true })
    } catch (err: any) {
      setError(err.message || 'Signup failed')
    } finally {
      setLoading(false)
    }
  }

  const signInWithGoogle = async () => {
    setGoogleLoading(true)
    setError('')
    try {
      const res = await api.googleAuthorize()
      if (!res?.url) {
        setError(res?.message || 'Google sign-in is not configured on the server.')
        return
      }
      window.location.href = res.url
    } catch (err: any) {
      setError(err.message || 'Could not start Google sign-in')
    } finally {
      setGoogleLoading(false)
    }
  }

  return (
    <div className="auth-screen">
      <AuthArt />
      <div className="auth-form-wrap">
        <form className="auth-card" onSubmit={onSubmit}>
          <div className="auth-card-brand">
            <BrandMark size={28} />
            <span>{APP_NAME}</span>
          </div>
          <h2>Create your account</h2>
          <p className="auth-sub">Set up a workspace for {COMPANY} in under a minute.</p>

          <button
            className="btn google lg block"
            type="button"
            disabled={googleLoading || loading}
            onClick={() => void signInWithGoogle()}
          >
            {googleLoading ? <Spinner size="sm" /> : <GoogleIcon />}
            {googleLoading ? 'Redirecting to Google…' : 'Continue with Google'}
          </button>

          <div className="auth-divider">or use email</div>

          <label className="field">
            <span>Full name</span>
            <div className="input-wrap">
              <PersonRegular />
              <input
                className="input"
                autoComplete="name"
                value={form.name}
                onChange={(e) => setForm({ ...form, name: e.target.value })}
                required
              />
            </div>
          </label>
          <label className="field">
            <span>Work email</span>
            <div className="input-wrap">
              <MailRegular />
              <input
                className="input"
                type="email"
                autoComplete="username"
                placeholder="name@company.com"
                value={form.email}
                onChange={(e) => setForm({ ...form, email: e.target.value })}
                required
              />
            </div>
          </label>
          <label className="field">
            <span>Password</span>
            <PasswordInput
              value={form.password}
              onChange={(v) => setForm({ ...form, password: v })}
              autoComplete="new-password"
              minLength={8}
            />
            <span className="field-hint">At least 8 characters.</span>
          </label>
          {error ? (
            <MessageBar intent="error" title="Could not create account">
              {error}
            </MessageBar>
          ) : null}
          <button className="btn lg block" type="submit" disabled={loading || googleLoading}>
            {loading ? <Spinner size="sm" /> : null}
            {loading ? 'Creating account…' : 'Create account'}
            {!loading ? <ArrowRightRegular /> : null}
          </button>
          <div className="auth-switch">
            Already have an account?{' '}
            <Link to="/login" className="link">Sign in</Link>
          </div>
          <p className="auth-legal">
            By continuing you agree to our{' '}
            <Link to="/terms">Terms</Link>
            {' '}and{' '}
            <Link to="/privacy">Privacy Policy</Link>.
          </p>
        </form>
      </div>
    </div>
  )
}
