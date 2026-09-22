import { useEffect, useState, type FormEvent } from 'react'
import { Link, useNavigate, useSearchParams } from 'react-router-dom'
import { api, getApiBase } from '../api/client'
import { useAuth } from '../auth/AuthContext'

function AuthArt() {
  return (
    <div className="auth-art">
      <div style={{ position: 'relative', zIndex: 1, marginTop: '14vh' }}>
        <div className="auth-art-brand">
          <div className="brand-mark">S</div>
          <div>
            <div className="brand" style={{ color: 'white', fontSize: '1.35rem' }}>
              <span style={{ color: 'white', WebkitTextFillColor: 'white' }}>Starlight</span>
            </div>
            <div style={{ color: 'rgba(255,255,255,0.75)', fontSize: 13, fontWeight: 600 }}>
              Linear LED · AI Mailer
            </div>
          </div>
        </div>
        <h1>Light up every conversation</h1>
        <p>
          Sign in with any Google account — campaigns send from the inbox you authenticate with.
        </p>
      </div>
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

  useEffect(() => {
    const token = params.get('google_token')
    const googleError = params.get('google_error')
    if (googleError) {
      setError(decodeURIComponent(googleError))
      return
    }
    if (token) {
      setToken(token)
      navigate('/inbox', { replace: true })
    }
  }, [params, setToken, navigate])

  const onSubmit = async (e: FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError('')
    try {
      const res = await api.login({ email, password })
      setToken(res.token)
      navigate('/inbox')
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
        <form className="auth-card stack" onSubmit={onSubmit}>
          <div>
            <h1 style={{ margin: 0, fontFamily: 'var(--display)', fontSize: '1.55rem', letterSpacing: '-0.02em' }}>
              Sign in
            </h1>
            <p className="muted" style={{ margin: '0.4rem 0 0' }}>
              Any Google account — mail sends from that inbox
            </p>
          </div>

          <button
            className="btn google"
            type="button"
            disabled={googleLoading || loading}
            onClick={() => void signInWithGoogle()}
          >
            <GoogleIcon />
            {googleLoading ? 'Redirecting…' : 'Continue with Google'}
          </button>

          <div className="auth-divider"><span>or email</span></div>

          <label className="field">
            <span>Work email</span>
            <input
              className="input"
              type="email"
              autoComplete="username"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
            />
          </label>
          <label className="field">
            <span>Password</span>
            <input
              className="input"
              type="password"
              autoComplete="current-password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />
          </label>
          {error ? (
            <div className="alert danger">
              <strong>Could not sign in</strong>
              <div style={{ marginTop: 4 }}>{error}</div>
            </div>
          ) : null}
          <button className="btn" type="submit" disabled={loading || googleLoading}>
            {loading ? 'Signing in…' : 'Continue'}
          </button>
          <div className="muted">
            New to Starlight?{' '}
            <Link to="/signup" style={{ color: 'var(--blue)', fontWeight: 700 }}>Create account</Link>
          </div>
          <p className="muted" style={{ fontSize: 12, margin: 0 }}>
            API: {getApiBase() || '(same origin)'}
          </p>
        </form>
      </div>
    </div>
  )
}

export function SignupPage() {
  const { setToken } = useAuth()
  const navigate = useNavigate()
  const [form, setForm] = useState({ email: '', password: '', name: '', org_name: 'Starlight Linear LED' })
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [googleLoading, setGoogleLoading] = useState(false)

  const onSubmit = async (e: FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError('')
    try {
      const res = await api.signup({ ...form, org_name: form.org_name || 'Starlight Linear LED' })
      setToken(res.token)
      navigate('/inbox')
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
        <form className="auth-card stack" onSubmit={onSubmit}>
          <div>
            <h1 style={{ margin: 0, fontFamily: 'var(--display)', fontSize: '1.55rem', letterSpacing: '-0.02em' }}>
              Create account
            </h1>
            <p className="muted" style={{ margin: '0.4rem 0 0' }}>Your AI CRM for LED outreach & replies</p>
          </div>

          <button
            className="btn google"
            type="button"
            disabled={googleLoading || loading}
            onClick={() => void signInWithGoogle()}
          >
            <GoogleIcon />
            {googleLoading ? 'Redirecting…' : 'Continue with Google'}
          </button>

          <div className="auth-divider"><span>or email</span></div>

          <label className="field">
            <span>Full name</span>
            <input
              className="input"
              value={form.name}
              onChange={(e) => setForm({ ...form, name: e.target.value })}
              required
            />
          </label>
          <label className="field">
            <span>Work email</span>
            <input
              className="input"
              type="email"
              autoComplete="username"
              value={form.email}
              onChange={(e) => setForm({ ...form, email: e.target.value })}
              required
            />
          </label>
          <label className="field">
            <span>Password</span>
            <input
              className="input"
              type="password"
              autoComplete="new-password"
              value={form.password}
              onChange={(e) => setForm({ ...form, password: e.target.value })}
              required
              minLength={8}
            />
          </label>
          {error ? (
            <div className="alert danger">
              <strong>Could not create account</strong>
              <div style={{ marginTop: 4 }}>{error}</div>
            </div>
          ) : null}
          <button className="btn amber" type="submit" disabled={loading || googleLoading}>
            {loading ? 'Creating…' : 'Get started'}
          </button>
          <div className="muted">
            Already have access?{' '}
            <Link to="/login" style={{ color: 'var(--blue)', fontWeight: 700 }}>Sign in</Link>
          </div>
        </form>
      </div>
    </div>
  )
}
