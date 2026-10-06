import { useState } from 'react'
import { ChevronDownRegular, ChevronUpRegular, SparkleRegular } from '@fluentui/react-icons'
import { EmailPreviewFrame } from './EmailPreviewFrame'
import { Avatar } from './ui'

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

/** One message in a thread, collapsible like Outlook's conversation view. */
export function MessageBubble({
  message,
  clientLabel = 'Client',
  defaultOpen = false,
}: {
  message: Msg
  clientLabel?: string
  defaultOpen?: boolean
}) {
  const [open, setOpen] = useState(defaultOpen)
  const inbound = message.direction === 'inbound'
  const who = inbound ? clientLabel : 'Starlight Linear LED'
  const preview = (message.body_text || message.body_html || '')
    .replace(/<style[\s\S]*?<\/style>/gi, ' ')
    .replace(/<[^>]+>/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .slice(0, 140)

  return (
    <div className={`msg-card ${inbound ? 'inbound' : 'outbound'}`}>
      <button type="button" className="msg-card-head" aria-expanded={open} onClick={() => setOpen((v) => !v)}>
        <Avatar name={who} size={32} />
        <div className="msg-card-who">
          <div className="row nowrap" style={{ gap: 6 }}>
            <strong className="truncate">{who}</strong>
            {message.ai_generated ? (
              <span className="badge brand"><SparkleRegular /> AI</span>
            ) : null}
          </div>
          <div className="text-sm muted truncate">{open ? message.subject || '(no subject)' : preview || message.subject}</div>
        </div>
        {message.created_at ? (
          <span className="msg-card-time">
            {new Date(message.created_at).toLocaleString(undefined, { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' })}
          </span>
        ) : null}
        {open ? <ChevronUpRegular /> : <ChevronDownRegular />}
      </button>
      {open ? (
        <div className="msg-card-body">
          <EmailPreviewFrame html={message.body_html} text={message.body_text} subject={message.subject} fromLabel={who} bare />
        </div>
      ) : null}
    </div>
  )
}
