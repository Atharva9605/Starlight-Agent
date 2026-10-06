import { useEffect, useState, type FormEvent } from 'react'
import {
  CheckmarkCircleFilled,
  LinkRegular,
  MailRegular,
  PersonRegular,
  PlugConnectedRegular,
  PlugDisconnectedRegular,
  SaveRegular,
  WarningFilled,
} from '@fluentui/react-icons'
import { api } from '../api/client'
import { CardHeader, MessageBar, PageHeader, Spinner, useConfirm, useToast } from '../components/ui'

const SENDER_FIELDS: { key: string; label: string; placeholder?: string; type?: string; hint?: string; span?: boolean }[] = [
  { key: 'sender_name', label: 'Name', placeholder: 'Vivek Dhondarkar' },
  { key: 'sender_company', label: 'Company', placeholder: 'Starlight Linear LED' },
  { key: 'sender_email', label: 'Email', placeholder: 'you@starlightlinearled.com', type: 'email' },
  { key: 'sender_phone', label: 'Phone', type: 'tel' },
  { key: 'sender_website', label: 'Website', placeholder: 'www.starlightlinearled.com' },
  {
    key: 'company_logo_url',
    label: 'Logo URL',
    placeholder: 'https://…/logo.png',
    hint: 'A square PNG or SVG works best. Shown in the email signature.',
  },
]

type Section = 'identity' | 'gmail'

export function SettingsPage() {
  const toast = useToast()
  const confirm = useConfirm()
  const [section, setSection] = useState<Section>('identity')
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
  const [loadError, setLoadError] = useState('')
  const [gmailNotice, setGmailNotice] = useState('')
  const [saving, setSaving] = useState(false)
  const [dirty, setDirty] = useState(false)

  useEffect(() => {
    Promise.all([api.sender(), api.gmailStatus().catch(() => ({ connected: false }))])
      .then(([s, g]) => {
        setSender((prev) => ({ ...prev, ...s }))
        setGmail(g as any)
      })
      .catch((e) => setLoadError(e.message))
  }, [])

  const saveSender = async (e: FormEvent) => {
    e.preventDefault()
    setSaving(true)
    try {
      await api.updateSender(sender)
      setDirty(false)
      toast.success('Sender profile saved')
    } catch (err: any) {
      toast.error("Couldn't save the sender profile", err.message)
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

  const disconnectGmail = async () => {
    const ok = await confirm({
      title: 'Disconnect Gmail?',
      body: 'Campaigns and replies will stop sending from this inbox until you connect again.',
      confirmLabel: 'Disconnect',
      danger: true,
    })
    if (!ok) return
    try {
      await api.gmailDisconnect()
      setGmail({ connected: false })
      toast.info('Gmail disconnected')
    } catch (e: any) {
      toast.error("Couldn't disconnect Gmail", e.message)
    }
  }

  const connectedAs = gmail.email || gmail.connected_email

  return (
    <div>
      <PageHeader title="Settings" subtitle="Sender identity and Gmail for this Starlight workspace." />

      {loadError ? (
        <div className="page-alerts">
          <MessageBar intent="error" title="Couldn't load settings">{loadError}</MessageBar>
        </div>
      ) : null}

      <div className="studio">
        <nav className="card studio-nav" aria-label="Settings sections">
          <div className="studio-nav-label">Workspace</div>
          <div className="vnav">
            <button
              type="button"
              className={`vnav-item${section === 'identity' ? ' active' : ''}`}
              aria-current={section === 'identity' ? 'page' : undefined}
              onClick={() => setSection('identity')}
            >
              <PersonRegular />
              <span className="vnav-copy">
                <span className="vnav-title">Sender profile</span>
                <span className="vnav-sub">Name, company, signature</span>
              </span>
              {dirty ? <span className="dirty-dot" title="Unsaved changes" /> : null}
            </button>
            <button
              type="button"
              className={`vnav-item${section === 'gmail' ? ' active' : ''}`}
              aria-current={section === 'gmail' ? 'page' : undefined}
              onClick={() => setSection('gmail')}
            >
              <MailRegular />
              <span className="vnav-copy">
                <span className="vnav-title">Gmail</span>
                <span className="vnav-sub">{gmail.connected ? connectedAs || 'Connected' : 'Not connected'}</span>
              </span>
            </button>
          </div>
        </nav>

        {section === 'identity' ? (
          <form className="card flush" onSubmit={saveSender}>
            <div className="card-section">
              <CardHeader
                title="Sender profile"
                subtitle="Appears in the signature of every outbound Starlight email."
              />
            </div>
            <div className="card-section">
              <div className="settings-form">
                {SENDER_FIELDS.map((f) => (
                  <label key={f.key} className={`field${f.hint ? ' span-2' : ''}`}>
                    <span>{f.label}</span>
                    <input
                      className="input"
                      type={f.type || 'text'}
                      placeholder={f.placeholder}
                      value={sender[f.key] || ''}
                      onChange={(e) => {
                        setSender({ ...sender, [f.key]: e.target.value })
                        setDirty(true)
                      }}
                    />
                    {f.hint ? <span className="field-hint">{f.hint}</span> : null}
                  </label>
                ))}
                {sender.company_logo_url ? (
                  <div className="span-2 row">
                    <img className="logo-preview" src={sender.company_logo_url} alt="Logo preview" />
                    <span className="muted text-sm">Logo preview</span>
                  </div>
                ) : null}
              </div>
            </div>
            <div className="card-footer">
              {dirty ? <span className="muted text-sm grow">You have unsaved changes</span> : null}
              <button className="btn" type="submit" disabled={saving || !dirty}>
                {saving ? <Spinner size="sm" /> : <SaveRegular />}
                {saving ? 'Saving…' : 'Save'}
              </button>
            </div>
          </form>
        ) : (
          <section className="card">
            <CardHeader title="Gmail" subtitle="The inbox used to sync client replies and send campaigns." />

            <div className="stack" style={{ marginTop: 16 }}>
              <div className="connection-card">
                <span className="connection-icon">
                  <MailRegular style={{ fontSize: 24, color: '#C5221F' }} />
                </span>
                <div className="grow">
                  <div className="row" style={{ gap: 8 }}>
                    <strong>Google Gmail</strong>
                    {gmail.connected ? (
                      <span className="badge success"><CheckmarkCircleFilled /> Connected</span>
                    ) : (
                      <span className="badge warning"><WarningFilled /> Not connected</span>
                    )}
                  </div>
                  <div className="muted text-sm" style={{ marginTop: 2 }}>
                    {gmail.connected
                      ? <>Campaigns and replies send as <strong>{connectedAs}</strong>.</>
                      : gmail.message || 'Connect any Google account. Outbound mail is sent from the connected inbox.'}
                  </div>
                </div>
                <div className="row">
                  {gmail.connected ? (
                    <button className="btn danger-outline" type="button" onClick={disconnectGmail}>
                      <PlugDisconnectedRegular /> Disconnect
                    </button>
                  ) : null}
                  <button className={`btn${gmail.connected ? ' secondary' : ''}`} type="button" onClick={connectGmail}>
                    {gmail.connected ? <LinkRegular /> : <PlugConnectedRegular />}
                    {gmail.connected ? 'Reconnect' : 'Connect Gmail'}
                  </button>
                </div>
              </div>

              {!gmail.connected && gmail.mode === 'platform' && gmail.email ? (
                <MessageBar intent="info" title="Using the platform sender">
                  Mail currently sends from <strong>{gmail.email}</strong>. Connect your Google account to send from your own inbox instead.
                </MessageBar>
              ) : null}

              {gmailNotice ? (
                <MessageBar intent="error" title="Gmail not connected" className="stack-body" onDismiss={() => setGmailNotice('')}>
                  {gmailNotice}
                  <span className="muted" style={{ display: 'block', marginTop: 4 }}>
                    Set GOOGLE_OAUTH_CLIENT_ID, GOOGLE_OAUTH_CLIENT_SECRET and GOOGLE_OAUTH_REDIRECT_URI on the
                    server. The OAuth client must allow any Google account (External app type — not limited to a
                    Workspace domain).
                  </span>
                </MessageBar>
              ) : null}
            </div>
          </section>
        )}
      </div>
    </div>
  )
}
