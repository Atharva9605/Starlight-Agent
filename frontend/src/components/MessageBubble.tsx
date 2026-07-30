import { EmailPreviewFrame } from './EmailPreviewFrame'

type Msg = {
  id: string
  direction: string
  status: string
  subject?: string
  body_html?: string
  body_text?: string
  ai_generated?: boolean
  created_at?: string
}

export function MessageBubble({ message }: { message: Msg }) {
  const inbound = message.direction === 'inbound'
  return (
    <div
      className={`msg-bubble ${inbound ? 'inbound' : 'outbound'}`}
    >
      <div className="row" style={{ justifyContent: 'space-between', marginBottom: 8 }}>
        <span className={`pill ${inbound ? 'warn' : 'ok'}`}>
          {inbound ? 'Client' : 'Starlight'}
          {message.status === 'draft' ? ' · Draft' : ''}
          {message.ai_generated ? ' · AI' : ''}
        </span>
        {message.created_at ? (
          <span className="muted" style={{ fontSize: 11 }}>
            {new Date(message.created_at).toLocaleString()}
          </span>
        ) : null}
      </div>
      {message.subject ? (
        <div style={{ fontWeight: 700, marginBottom: 8 }}>{message.subject}</div>
      ) : null}
      <EmailPreviewFrame
        html={message.body_html}
        text={message.body_text}
        subject={message.subject}
        fromLabel={inbound ? 'Client' : 'Starlight Linear LED'}
        compact
      />
    </div>
  )
}
