type Props = {
  html?: string
  text?: string
  subject?: string
  fromLabel?: string
  compact?: boolean
  fullscreen?: boolean
}

/** Sandboxed rendered email — what the customer sees. Never shows source. */
export function EmailPreviewFrame({
  html,
  text,
  subject,
  fromLabel = 'Starlight Linear LED',
  compact = false,
  fullscreen = false,
}: Props) {
  const doc =
    html && html.trim()
      ? html
      : `<html><body style="font-family:Segoe UI,Arial,sans-serif;padding:16px;color:#0f172a;white-space:pre-wrap">${escapeHtml(text || '')}</body></html>`

  const height = fullscreen ? '100%' : compact ? 160 : 280

  return (
    <div className={`email-chrome${fullscreen ? ' fullscreen' : ''}`}>
      <div className="email-chrome-bar">
        <div className="email-chrome-dots">
          <span /><span /><span />
        </div>
        <div className="email-chrome-meta">
          <div><strong>From</strong> {fromLabel}</div>
          {subject ? <div><strong>Subject</strong> {subject}</div> : null}
        </div>
      </div>
      <iframe
        title="email-preview"
        sandbox=""
        srcDoc={doc}
        style={{
          width: '100%',
          height,
          minHeight: fullscreen ? 0 : height,
          flex: fullscreen ? 1 : undefined,
          border: 0,
          background: 'white',
          display: 'block',
        }}
      />
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
