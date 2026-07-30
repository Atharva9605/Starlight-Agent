import { useEffect, useState, type FormEvent } from 'react'
import { api } from '../api/client'

export function SettingsPage() {
  const [sender, setSender] = useState<Record<string, string>>({
    sender_name: 'Vivek Dhondarkar',
    sender_company: 'Starlight Linear LED',
    sender_email: '',
    sender_phone: '',
    sender_website: 'www.starlightlinearled.com',
    company_logo_url: '',
  })
  const [gmail, setGmail] = useState<{ connected: boolean; connected_email?: string }>({ connected: false })
  const [msg, setMsg] = useState('')

  useEffect(() => {
    Promise.all([api.sender(), api.gmailStatus().catch(() => ({ connected: false }))])
      .then(([s, g]) => {
        setSender((prev) => ({ ...prev, ...s }))
        setGmail(g as any)
      })
      .catch((e) => setMsg(e.message))
  }, [])

  const saveSender = async (e: FormEvent) => {
    e.preventDefault()
    await api.updateSender(sender)
    setMsg('Starlight sender saved')
  }

  const connectGmail = async () => {
    const res = await api.gmailAuthorize()
    window.location.href = (res as any).authorize_url || (res as any).url
  }

  return (
    <div>
      <div className="page-hero">
        <div>
          <h1>Settings</h1>
          <p>Starlight identity, sender details, and Gmail — this workspace is for Starlight Linear LED only.</p>
        </div>
        {msg ? <span className="pill ok">{msg}</span> : null}
      </div>

      <div className="panel tint-blue stack" style={{ marginBottom: '1rem' }}>
        <strong style={{ fontFamily: 'var(--display)' }}>Product</strong>
        <div className="row">
          <div className="brand-mark">S</div>
          <div>
            <div style={{ fontWeight: 800, fontFamily: 'var(--display)', fontSize: '1.2rem' }}>Starlight Linear LED</div>
            <div className="muted">Award-winning Indian LED lighting · AI Mailer CRM</div>
          </div>
        </div>
      </div>

      <form className="panel stack" onSubmit={saveSender} style={{ marginBottom: '1rem' }}>
        <strong style={{ fontFamily: 'var(--display)' }}>Sender profile</strong>
        {['sender_name', 'sender_company', 'sender_email', 'sender_phone', 'sender_website', 'company_logo_url'].map((k) => (
          <input key={k} className="input" placeholder={k.replace(/_/g, ' ')} value={sender[k] || ''} onChange={(e) => setSender({ ...sender, [k]: e.target.value })} />
        ))}
        <button className="btn">Save sender</button>
      </form>

      <div className="panel tint-amber stack">
        <strong style={{ fontFamily: 'var(--display)' }}>Gmail</strong>
        <div className="muted">
          {gmail.connected ? `Connected as ${gmail.connected_email}` : 'Connect the Starlight Workspace inbox'}
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
