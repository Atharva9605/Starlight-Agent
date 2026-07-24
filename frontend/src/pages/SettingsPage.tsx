import { useEffect, useState, type FormEvent } from 'react'
import { api } from '../api/client'
import { useAuth } from '../auth/AuthContext'

export function SettingsPage() {
  const { refresh } = useAuth()
  const [branding, setBranding] = useState({ display_name: '', logo_url: '', accent_color: '#0F766E' })
  const [sender, setSender] = useState<Record<string, string>>({})
  const [gmail, setGmail] = useState<{ connected: boolean; connected_email?: string }>({ connected: false })
  const [msg, setMsg] = useState('')

  useEffect(() => {
    Promise.all([api.branding(), api.sender(), api.gmailStatus().catch(() => ({ connected: false }))])
      .then(([b, s, g]) => {
        setBranding(b)
        setSender(s)
        setGmail(g as any)
      })
      .catch((e) => setMsg(e.message))
  }, [])

  const saveBranding = async (e: FormEvent) => {
    e.preventDefault()
    await api.updateBranding(branding)
    await refresh()
    setMsg('Branding saved')
  }

  const saveSender = async (e: FormEvent) => {
    e.preventDefault()
    await api.updateSender(sender)
    setMsg('Sender saved')
  }

  const connectGmail = async () => {
    const res = await api.gmailAuthorize()
    window.location.href = (res as any).authorize_url || (res as any).url
  }

  return (
    <div className="stack" style={{ gap: '1.25rem' }}>
      <div>
        <h1 style={{ margin: 0, fontFamily: 'var(--font-display)', fontWeight: 500 }}>Settings</h1>
        <p className="muted">White-label branding, sender identity, Gmail connection.</p>
      </div>
      {msg ? <div className="muted">{msg}</div> : null}

      <form className="panel stack" onSubmit={saveBranding}>
        <strong>Organization branding</strong>
        <input className="input" placeholder="Display name" value={branding.display_name} onChange={(e) => setBranding({ ...branding, display_name: e.target.value })} />
        <input className="input" placeholder="Logo URL" value={branding.logo_url} onChange={(e) => setBranding({ ...branding, logo_url: e.target.value })} />
        <input className="input" type="color" value={branding.accent_color} onChange={(e) => setBranding({ ...branding, accent_color: e.target.value })} />
        <button className="btn">Save branding</button>
      </form>

      <form className="panel stack" onSubmit={saveSender}>
        <strong>Sender</strong>
        {['sender_name', 'sender_company', 'sender_email', 'sender_phone', 'sender_website', 'company_logo_url'].map((k) => (
          <input key={k} className="input" placeholder={k} value={sender[k] || ''} onChange={(e) => setSender({ ...sender, [k]: e.target.value })} />
        ))}
        <button className="btn">Save sender</button>
      </form>

      <div className="panel stack">
        <strong>Gmail</strong>
        <div className="muted">
          {gmail.connected ? `Connected as ${gmail.connected_email}` : 'Not connected'}
        </div>
        <div className="row">
          <button className="btn" type="button" onClick={connectGmail}>Connect Gmail</button>
          {gmail.connected ? (
            <button className="btn secondary" type="button" onClick={async () => { await api.gmailDisconnect(); setGmail({ connected: false }) }}>
              Disconnect
            </button>
          ) : null}
        </div>
      </div>
    </div>
  )
}
