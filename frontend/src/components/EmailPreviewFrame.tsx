import { useState } from 'react'
import { DesktopRegular, PhoneRegular } from '@fluentui/react-icons'
import { Avatar } from './ui'

type Props = {
  html?: string
  text?: string
  subject?: string
  fromLabel?: string
  compact?: boolean
  fullscreen?: boolean
  /** Show the desktop / mobile width toggle in the header. */
  deviceToggle?: boolean
  bare?: boolean
}

/** Sandboxed rendered email — what the customer sees. Never shows source. */
export function EmailPreviewFrame({
  html,
  text,
  subject,
  fromLabel = 'Starlight Linear LED',
  compact = false,
  fullscreen = false,
  deviceToggle = false,
  bare = false,
}: Props) {
  const [mobile, setMobile] = useState(false)
  const doc =
    html && html.trim()
      ? html
      : `<html><body style="font-family:'Segoe UI',Arial,sans-serif;font-size:14px;line-height:1.5;padding:16px;color:#242424;white-space:pre-wrap;margin:0">${escapeHtml(text || '')}</body></html>`

  const height = fullscreen ? '100%' : compact ? 180 : 300

  return (
    <div className={`email-chrome${fullscreen ? ' fullscreen' : ''}${bare ? ' bare' : ''}`}>
      {!bare ? (
        <div className="email-chrome-head">
          <Avatar name={fromLabel} size={36} />
          <div className="email-chrome-meta">
            <div className="email-chrome-subject" title={subject}>{subject || '(no subject)'}</div>
            <div className="email-chrome-from">
              <strong>{fromLabel}</strong>
            </div>
          </div>
          {deviceToggle ? (
            <div className="btn-group" role="group" aria-label="Preview width">
              <button
                type="button"
                className={`btn secondary icon-only${!mobile ? ' active' : ''}`}
                aria-pressed={!mobile}
                title="Desktop preview"
                onClick={() => setMobile(false)}
              >
                <DesktopRegular />
              </button>
              <button
                type="button"
                className={`btn secondary icon-only${mobile ? ' active' : ''}`}
                aria-pressed={mobile}
                title="Mobile preview"
                onClick={() => setMobile(true)}
              >
                <PhoneRegular />
              </button>
            </div>
          ) : null}
        </div>
      ) : null}
      <div className={`email-chrome-frame${mobile ? ' mobile' : ''}`}>
        <iframe
          title={subject ? `Email preview: ${subject}` : 'Email preview'}
          sandbox=""
          srcDoc={doc}
          style={{ height, minHeight: fullscreen ? 0 : height }}
        />
      </div>
    </div>
  )
}

function escapeHtml(s: string) {
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
}
