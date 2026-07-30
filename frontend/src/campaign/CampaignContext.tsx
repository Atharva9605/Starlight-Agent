import { createContext, useCallback, useContext, useMemo, useRef, useState } from 'react'
import type { ReactNode } from 'react'
import { api, getApiBase } from '../api/client'

export type Lead = Record<string, any>
export type RunStatus = 'idle' | 'running' | 'done' | 'stopped' | 'failed'

export type PreviewItem = {
  rowIndex: number
  website: string
  subject: string
  html: string
}

export const TEMPLATES = [
  {
    value: 'email_template.html',
    label: 'Modern Soft',
    blurb: 'Rounded cards, gentle gradients. Safe default for cold outreach.',
    swatch: ['#2563eb', '#06b6d4'],
  },
  {
    value: 'email_template_minimalist.html',
    label: 'Minimalist',
    blurb: 'Single column, lots of whitespace. Reads like a personal note.',
    swatch: ['#0f172a', '#64748b'],
  },
  {
    value: 'email_template_bold.html',
    label: 'Bold & Vibrant',
    blurb: 'Strong headers and colour blocks. Best for product launches.',
    swatch: ['#f59e0b', '#e11d48'],
  },
]

/** A lead is "done" once the backend reports a terminal status for its row. */
function leadState(lead: Lead): 'sent' | 'failed' | 'processing' | 'pending' {
  const s = String(lead._status || '').toLowerCase()
  if (s.includes('sent') || s.includes('✅')) return 'sent'
  if (s.includes('fail') || s.includes('error') || s.includes('❌')) return 'failed'
  if (s.includes('process') || s.includes('⚙')) return 'processing'
  return 'pending'
}

type CampaignValue = {
  leads: Lead[]
  fileName: string
  template: string
  delay: number
  recipientOverride: string
  senderEmail: string
  status: RunStatus
  logs: string[]
  previews: PreviewItem[]
  currentRow: number | null
  counts: { total: number; sent: number; failed: number; pending: number; processed: number }
  setTemplate: (v: string) => void
  setDelay: (v: number) => void
  setRecipientOverride: (v: string) => void
  setSenderEmail: (v: string) => void
  uploadLeads: (file: File) => Promise<void>
  removeLead: (index: number) => void
  clearLeads: () => void
  start: () => Promise<void>
  stop: () => void
  reset: () => void
}

const Ctx = createContext<CampaignValue | null>(null)

export function CampaignProvider({ children }: { children: ReactNode }) {
  const [leads, setLeads] = useState<Lead[]>([])
  const [fileName, setFileName] = useState('')
  const [template, setTemplate] = useState(TEMPLATES[0].value)
  const [delay, setDelay] = useState(2)
  const [recipientOverride, setRecipientOverride] = useState('')
  const [senderEmail, setSenderEmail] = useState('')
  const [status, setStatus] = useState<RunStatus>('idle')
  const [logs, setLogs] = useState<string[]>([])
  const [previews, setPreviews] = useState<PreviewItem[]>([])
  const [currentRow, setCurrentRow] = useState<number | null>(null)

  const abortRef = useRef<AbortController | null>(null)
  // Previews arrive before the row they belong to is known, so track the live row.
  const currentRowRef = useRef<number | null>(null)
  const leadsRef = useRef<Lead[]>([])
  leadsRef.current = leads

  const uploadLeads = useCallback(async (file: File) => {
    const res = await api.uploadLeads(file)
    setLeads(res.leads || [])
    setFileName(file.name)
    setPreviews([])
    setLogs([])
    setStatus('idle')
  }, [])

  const removeLead = useCallback((index: number) => {
    setLeads((prev) => prev.filter((_, i) => i !== index))
  }, [])

  const clearLeads = useCallback(() => {
    setLeads([])
    setFileName('')
    setPreviews([])
    setLogs([])
    setStatus('idle')
  }, [])

  const reset = useCallback(() => {
    abortRef.current?.abort()
    setLeads((prev) => prev.map((l) => ({ ...l, _status: '' })))
    setPreviews([])
    setLogs([])
    setStatus('idle')
    setCurrentRow(null)
    currentRowRef.current = null
  }, [])

  const stop = useCallback(() => {
    abortRef.current?.abort()
    setStatus('stopped')
    setCurrentRow(null)
    currentRowRef.current = null
  }, [])

  const start = useCallback(async () => {
    setStatus('running')
    setLogs([])
    setPreviews([])
    setLeads((prev) => prev.map((l) => ({ ...l, _status: '' })))

    const ctrl = new AbortController()
    abortRef.current = ctrl
    const token = localStorage.getItem('token')

    try {
      const res = await fetch(`${getApiBase()}/api/process-stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        body: JSON.stringify({
          leads: leadsRef.current,
          template,
          delay,
          sender_email: senderEmail,
          recipient_override: recipientOverride,
        }),
        signal: ctrl.signal,
      })
      if (!res.ok || !res.body) throw new Error(await res.text())

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const chunks = buffer.split('\n\n')
        buffer = chunks.pop() || ''

        for (const chunk of chunks) {
          const line = chunk.split('\n').find((l) => l.startsWith('data:'))
          if (!line) continue
          let evt: any
          try {
            evt = JSON.parse(line.slice(5).trim())
          } catch {
            continue
          }

          if (evt.type === 'log') {
            setLogs((prev) => [...prev, evt.message || String(evt)])
          }

          if (evt.type === 'status_update') {
            setLeads((prev) =>
              prev.map((l, i) => (i === evt.row_index ? { ...l, _status: evt.status } : l)),
            )
            if (String(evt.status || '').toLowerCase().includes('process')) {
              currentRowRef.current = evt.row_index
              setCurrentRow(evt.row_index)
            }
          }

          if (evt.type === 'preview_html') {
            const rowIndex = evt.row_index ?? currentRowRef.current ?? 0
            const lead = leadsRef.current[rowIndex] || {}
            setPreviews((prev) =>
              [
                ...prev.filter((p) => p.rowIndex !== rowIndex),
                {
                  rowIndex,
                  website: lead.website || lead.company || `Lead ${rowIndex + 1}`,
                  subject: evt.subject || 'Starlight outreach',
                  html: evt.html || '',
                },
              ].sort((a, b) => a.rowIndex - b.rowIndex),
            )
          }

          if (evt.type === 'done') {
            setStatus('done')
            setCurrentRow(null)
            currentRowRef.current = null
          }
        }
      }
      setStatus((s) => (s === 'running' ? 'done' : s))
    } catch (e: any) {
      if (e.name === 'AbortError') return
      setLogs((prev) => [...prev, e.message])
      setStatus('failed')
    } finally {
      setCurrentRow(null)
      currentRowRef.current = null
    }
  }, [template, delay, senderEmail, recipientOverride])

  const counts = useMemo(() => {
    let sent = 0
    let failed = 0
    for (const l of leads) {
      const s = leadState(l)
      if (s === 'sent') sent += 1
      else if (s === 'failed') failed += 1
    }
    return {
      total: leads.length,
      sent,
      failed,
      pending: leads.length - sent - failed,
      processed: sent + failed,
    }
  }, [leads])

  const value: CampaignValue = {
    leads,
    fileName,
    template,
    delay,
    recipientOverride,
    senderEmail,
    status,
    logs,
    previews,
    currentRow,
    counts,
    setTemplate,
    setDelay,
    setRecipientOverride,
    setSenderEmail,
    uploadLeads,
    removeLead,
    clearLeads,
    start,
    stop,
    reset,
  }

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}

export function useCampaign() {
  const v = useContext(Ctx)
  if (!v) throw new Error('useCampaign must be used inside CampaignProvider')
  return v
}

export { leadState }
