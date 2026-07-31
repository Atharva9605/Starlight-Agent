import { useState, type FormEvent } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { api } from '../api/client'
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
          Scrape prospects, ground emails in your catalogues, and reply with AI drafts —
          built exclusively for Starlight sales.
        </p>
      </div>
    </div>
  )
}

export function LoginPage() {
  const { setToken } = useAuth()
  const navigate = useNavigate()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

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

  return (
    <div className="auth-screen">
      <AuthArt />
      <div className="auth-form-wrap">
        <form className="auth-card stack" onSubmit={onSubmit}>
          <div>
            <h1 style={{ margin: 0, fontFamily: 'var(--display)', fontSize: '1.55rem', letterSpacing: '-0.02em' }}>
              Sign in
            </h1>
            <p className="muted" style={{ margin: '0.4rem 0 0' }}>Access Starlight AI Mailer</p>
          </div>
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
          <button className="btn" type="submit" disabled={loading}>
            {loading ? 'Signing in…' : 'Continue'}
          </button>
          <div className="muted">
            New to Starlight?{' '}
            <Link to="/signup" style={{ color: 'var(--blue)', fontWeight: 700 }}>Create account</Link>
          </div>
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
          <button className="btn amber" type="submit" disabled={loading}>
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
