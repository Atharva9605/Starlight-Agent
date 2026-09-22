import { useEffect, useState, type FormEvent } from 'react'
import { api } from '../api/client'

const SENDER_FIELDS: { key: string; label: string; placeholder?: string }[] = [
  { key: 'sender_name', label: 'Name', placeholder: 'Vivek Dhondarkar' },
  { key: 'sender_company', label: 'Company', placeholder: 'Starlight Linear LED' },
  { key: 'sender_email', label: 'Email', placeholder: 'you@starlightlinearled.com' },
  { key: 'sender_phone', label: 'Phone' },
  { key: 'sender_website', label: 'Website', placeholder: 'www.starlightlinearled.com' },
  { key: 'company_logo_url', label: 'Logo URL' },
]

export function SettingsPage() {
  const [section, setSection] = useState<'identity' | 'gmail'>('identity')
  const [sender, setSender] = useState<Record<string, string>>({
    sender_name: 'Vivek Dhondarkar',
    sender_company: 'Starlight Linear LED',
    sender_email: '',
    sender_phone: '',
    sender_website: 'www.starlightlinearled.com',
    company_logo_url: '',
  })
  const [gmail, setGmail] = useState<{
    connected: boolean
    email?: string
    connected_email?: string
    mode?: string
    message?: string
  }>({ connected: false })
  const [msg, setMsg] = useState('')
  const [gmailNotice, setGmailNotice] = useState('')
  const [saving, setSaving] = useState(false)

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
    setSaving(true)
    try {
      await api.updateSender(sender)
      setMsg('Starlight sender saved')
    } catch (err: any) {
      setMsg(err.message || 'Save failed')
    } finally {
      setSaving(false)
    }
  }

  const connectGmail = async () => {
    setGmailNotice('')
    try {
      const res = await api.gmailAuthorize()
      if (!res?.url) {
        setGmailNotice(res?.message || 'Google OAuth is not configured on this server.')
        return
      }
      window.location.href = res.url
    } catch (e: any) {
      setGmailNotice(e.message || 'Could not start Gmail authorization')
    }
  }

  return (
    <div className="studio-screen">
      <div className="page-hero">
        <div>
          <h1>Settings</h1>
          <p>Sender identity and Gmail for this Starlight workspace.</p>
        </div>
        {msg ? <span className="pill ok">{msg}</span> : null}
      </div>

      <div className="studio-body">
        <aside className="studio-nav panel stack">
          <div className="studio-nav-label muted">Sections</div>
          <div className="studio-nav-list">
            <button
              type="button"
              className={`studio-nav-item is-link${section === 'identity' ? ' active' : ''}`}
              onClick={() => setSection('identity')}
            >
              <span className="studio-nav-title">Identity</span>
              <span className="studio-nav-key muted">Sender profile</span>
            </button>
            <button
              type="button"
              className={`studio-nav-item is-link${section === 'gmail' ? ' active' : ''}`}
              onClick={() => setSection('gmail')}
            >
              <span className="studio-nav-title">Gmail</span>
              <span className="studio-nav-key muted">
                {gmail.connected ? gmail.email || 'Connected' : 'Not connected'}
              </span>
            </button>
          </div>
        </aside>

        {section === 'identity' ? (
          <form className="studio-editor panel stack" onSubmit={saveSender}>
            <div className="studio-editor-head">
              <div>
                <h2>Sender profile</h2>
                <p>Appears on outbound Starlight emails.</p>
              </div>
              <button className="btn" type="submit" disabled={saving}>
                {saving ? 'Saving…' : 'Save sender'}
              </button>
            </div>
            {SENDER_FIELDS.map((f) => (
              <label key={f.key} className="field">
                <span>{f.label}</span>
                <input
                  className="input"
                  placeholder={f.placeholder}
                  value={sender[f.key] || ''}
                  onChange={(e) => setSender({ ...sender, [f.key]: e.target.value })}
                />
              </label>
            ))}
          </form>
        ) : (
          <div className="studio-editor panel tint-amber stack">
            <div className="studio-editor-head">
              <div>
                <h2>Gmail</h2>
                <p>Connect the inbox used for sync and sending.</p>
              </div>
              <div className="row">
                <button className="btn" type="button" onClick={connectGmail}>
                  {gmail.connected ? 'Reconnect' : 'Connect Gmail'}
                </button>
                {gmail.connected ? (
                  <button
                    className="btn secondary"
                    type="button"
                    onClick={async () => {
                      await api.gmailDisconnect()
                      setGmail({ connected: false })
                    }}
                  >
                    Disconnect
                  </button>
                ) : null}
              </div>
            </div>

            <div className="row">
              <span className={`pill ${gmail.connected ? 'ok' : 'warn'}`}>
                {gmail.connected ? 'Connected' : 'Not connected'}
              </span>
              {gmail.connected && gmail.email ? (
                <span className="muted" style={{ fontSize: 13 }}>{gmail.email}</span>
              ) : null}
            </div>

            {!gmail.connected ? (
              <p className="muted" style={{ margin: 0, fontSize: 13 }}>
                {gmail.message ||
                  'Sign in with Google on the login page (any Google account), or connect Gmail here. Outbound mail is sent from the connected inbox.'}
              </p>
            ) : (
              <p className="muted" style={{ margin: 0, fontSize: 13 }}>
                Campaigns and replies send as <strong>{gmail.email || gmail.connected_email}</strong>.
              </p>
            )}

            {!gmail.connected && gmail.mode === 'platform' && gmail.email ? (
              <p className="muted" style={{ margin: 0, fontSize: 13 }}>
                Platform fallback sender: <strong>{gmail.email}</strong>. Connect your Google account to send from your own inbox instead.
              </p>
            ) : null}

            {gmailNotice ? (
              <div className="alert danger">
                <strong>Gmail not connected</strong>
                <div style={{ marginTop: 4 }}>{gmailNotice}</div>
                <div className="muted" style={{ marginTop: 6, fontSize: 13 }}>
                  Set GOOGLE_OAUTH_CLIENT_ID, GOOGLE_OAUTH_CLIENT_SECRET and
                  GOOGLE_OAUTH_REDIRECT_URI on the server. OAuth client must allow any Google account
                  (External app type — not limited to a Workspace domain).
                </div>
              </div>
            ) : null}
          </div>
        )}
      </div>
    </div>
  )
}
