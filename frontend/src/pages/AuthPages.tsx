import { useState, type FormEvent } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { api } from '../api/client'
import { useAuth } from '../auth/AuthContext'

function AuthArt() {
  return (
    <div className="auth-art">
      <div style={{ position: 'relative', zIndex: 1, marginTop: '18vh' }}>
        <div className="pill" style={{ background: 'rgba(255,255,255,0.2)', color: 'white', marginBottom: 16 }}>
          Starlight Linear LED
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
            <div className="brand" style={{ fontSize: '1.6rem' }}><span>Welcome back</span></div>
            <p className="muted" style={{ margin: '0.4rem 0 0' }}>Sign in to Starlight AI Mailer</p>
          </div>
          <input className="input" placeholder="Work email" value={email} onChange={(e) => setEmail(e.target.value)} />
          <input className="input" type="password" placeholder="Password" value={password} onChange={(e) => setPassword(e.target.value)} />
          {error ? <div style={{ color: 'var(--danger)', fontWeight: 600 }}>{error}</div> : null}
          <button className="btn" disabled={loading}>{loading ? 'Signing in…' : 'Continue'}</button>
          <div className="muted">New to Starlight? <Link to="/signup" style={{ color: 'var(--blue)', fontWeight: 700 }}>Create account</Link></div>
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
            <div className="brand" style={{ fontSize: '1.55rem' }}><span>Join Starlight</span></div>
            <p className="muted" style={{ margin: '0.4rem 0 0' }}>Your AI CRM for LED outreach & replies</p>
          </div>
          <input className="input" placeholder="Full name" value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} />
          <input className="input" placeholder="Work email" value={form.email} onChange={(e) => setForm({ ...form, email: e.target.value })} />
          <input className="input" type="password" placeholder="Password" value={form.password} onChange={(e) => setForm({ ...form, password: e.target.value })} />
          {error ? <div style={{ color: 'var(--danger)', fontWeight: 600 }}>{error}</div> : null}
          <button className="btn amber" disabled={loading}>{loading ? 'Creating…' : 'Get started'}</button>
          <div className="muted">Already have access? <Link to="/login" style={{ color: 'var(--blue)', fontWeight: 700 }}>Sign in</Link></div>
        </form>
      </div>
    </div>
  )
}
