import { useState, type FormEvent } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { api } from '../api/client'
import { useAuth } from '../auth/AuthContext'

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
    <div style={{ minHeight: '100vh', display: 'grid', placeItems: 'center', padding: 24 }}>
      <form className="panel stack" style={{ width: 'min(420px, 100%)' }} onSubmit={onSubmit}>
        <div>
          <div className="brand">Welcome back</div>
          <p className="muted">Sign in to your CRM Agent workspace.</p>
        </div>
        <input className="input" placeholder="Email" value={email} onChange={(e) => setEmail(e.target.value)} />
        <input className="input" type="password" placeholder="Password" value={password} onChange={(e) => setPassword(e.target.value)} />
        {error ? <div style={{ color: 'var(--danger)' }}>{error}</div> : null}
        <button className="btn" disabled={loading}>{loading ? 'Signing in…' : 'Sign in'}</button>
        <div className="muted">No account? <Link to="/signup">Create one</Link></div>
      </form>
    </div>
  )
}

export function SignupPage() {
  const { setToken } = useAuth()
  const navigate = useNavigate()
  const [form, setForm] = useState({ email: '', password: '', name: '', org_name: '' })
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  const onSubmit = async (e: FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError('')
    try {
      const res = await api.signup(form)
      setToken(res.token)
      navigate('/settings')
    } catch (err: any) {
      setError(err.message || 'Signup failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{ minHeight: '100vh', display: 'grid', placeItems: 'center', padding: 24 }}>
      <form className="panel stack" style={{ width: 'min(460px, 100%)' }} onSubmit={onSubmit}>
        <div>
          <div className="brand">Create workspace</div>
          <p className="muted">White-label CRM Agent for your organization.</p>
        </div>
        {(['name', 'email', 'password', 'org_name'] as const).map((k) => (
          <input
            key={k}
            className="input"
            type={k === 'password' ? 'password' : 'text'}
            placeholder={k === 'org_name' ? 'Organization name' : k[0].toUpperCase() + k.slice(1)}
            value={form[k]}
            onChange={(e) => setForm({ ...form, [k]: e.target.value })}
          />
        ))}
        {error ? <div style={{ color: 'var(--danger)' }}>{error}</div> : null}
        <button className="btn" disabled={loading}>{loading ? 'Creating…' : 'Create account'}</button>
        <div className="muted">Have an account? <Link to="/login">Sign in</Link></div>
      </form>
    </div>
  )
}
