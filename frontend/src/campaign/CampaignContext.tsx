import { createContext, useCallback, useContext, useMemo, useRef, useState } from 'react'
import type { ReactNode } from 'react'
import { api, getApiBase } from '../api/client'

export type Lead = Record<string, any>
export type RunStatus = 'idle' | 'running' | 'reviewing' | 'done' | 'stopped' | 'failed'

export type ChatMessage = { role: 'user' | 'assistant' | 'system'; content: string }

export type Draft = {
  draftId: string
  rowIndex: number
  website: string
  company: string
  subject: string
  html: string
  to: string
}

export const TEMPLATES = [
  {
    value: 'email_template.html',
    label: 'Modern Soft',
    blurb: 'Rounded cards, gentle gradients.',
    swatch: ['#2563eb', '#06b6d4'],
  },
  {
    value: 'email_template_minimalist.html',
    label: 'Minimalist',
    blurb: 'Single column, personal note.',
    swatch: ['#0f172a', '#64748b'],
  },
  {
    value: 'email_template_bold.html',
    label: 'Bold & Vibrant',
    blurb: 'Strong headers for launches.',
    swatch: ['#f59e0b', '#e11d48'],
  },
]

function leadState(lead: Lead): 'sent' | 'failed' | 'processing' | 'pending' | 'ready' | 'skipped' {
  const s = String(lead._status || '').toLowerCase()
  if (s.includes('sent') || s.includes('✅')) return 'sent'
  if (s.includes('skip')) return 'skipped'
  if (s.includes('ready') || s.includes('📝')) return 'ready'
  if (s.includes('fail') || s.includes('error') || s.includes('❌')) return 'failed'
  if (s.includes('process') || s.includes('⚙') || s.includes('generat')) return 'processing'
  return 'pending'
}

type CampaignValue = {
  leads: Lead[]
  fileName: string
  template: string
  delay: number
  recipientOverride: string
  senderEmail: string
  autosend: boolean
  status: RunStatus
  logs: string[]
  draft: Draft | null
  chat: ChatMessage[]
  generating: boolean
  revising: boolean
  sending: boolean
  currentIndex: number
  counts: { total: number; sent: number; failed: number; pending: number; processed: number; skipped: number }
  setTemplate: (v: string) => void
  setDelay: (v: number) => void
  setRecipientOverride: (v: string) => void
  setSenderEmail: (v: string) => void
  setAutosend: (v: boolean) => void
  uploadLeads: (file: File) => Promise<void>
  removeLead: (index: number) => void
  clearLeads: () => void
  start: () => Promise<'review' | 'live'>
  stop: () => void
  reset: () => void
  sendCurrent: () => Promise<void>
  skipCurrent: () => Promise<void>
  reviseCurrent: (message: string) => Promise<void>
}

const Ctx = createContext<CampaignValue | null>(null)

export function CampaignProvider({ children }: { children: ReactNode }) {
  const [leads, setLeads] = useState<Lead[]>([])
  const [fileName, setFileName] = useState('')
  const [template, setTemplate] = useState(TEMPLATES[0].value)
  const [delay, setDelay] = useState(2)
  const [recipientOverride, setRecipientOverride] = useState('')
  const [senderEmail, setSenderEmail] = useState('')
  const [autosend, setAutosend] = useState(false)
  const [status, setStatus] = useState<RunStatus>('idle')
  const [logs, setLogs] = useState<string[]>([])
  const [draft, setDraft] = useState<Draft | null>(null)
  const [chat, setChat] = useState<ChatMessage[]>([])
  const [generating, setGenerating] = useState(false)
  const [revising, setRevising] = useState(false)
  const [sending, setSending] = useState(false)
  const [currentIndex, setCurrentIndex] = useState(0)

  const abortRef = useRef<AbortController | null>(null)
  const stopReviewRef = useRef(false)
  const leadsRef = useRef<Lead[]>([])
  const optsRef = useRef({ template, delay, recipientOverride, senderEmail, autosend })
  leadsRef.current = leads
  optsRef.current = { template, delay, recipientOverride, senderEmail, autosend }

  const uploadLeads = useCallback(async (file: File) => {
    const res = await api.uploadLeads(file)
    setLeads(res.leads || [])
    setFileName(file.name)
    setLogs([])
    setDraft(null)
    setChat([])
    setStatus('idle')
    setCurrentIndex(0)
  }, [])

  const removeLead = useCallback((index: number) => {
    setLeads((prev) => prev.filter((_, i) => i !== index))
  }, [])

  const clearLeads = useCallback(() => {
    setLeads([])
    setFileName('')
    setDraft(null)
    setChat([])
    setLogs([])
    setStatus('idle')
    setCurrentIndex(0)
  }, [])

  const reset = useCallback(() => {
    abortRef.current?.abort()
    stopReviewRef.current = true
    setLeads((prev) => prev.map((l) => ({ ...l, _status: '' })))
    setDraft(null)
    setChat([])
    setLogs([])
    setStatus('idle')
    setCurrentIndex(0)
    setGenerating(false)
    setRevising(false)
    setSending(false)
  }, [])

  const stop = useCallback(() => {
    abortRef.current?.abort()
    stopReviewRef.current = true
    setStatus('stopped')
    setGenerating(false)
  }, [])

  const setLeadStatus = (index: number, st: string) => {
    setLeads((prev) => prev.map((l, i) => (i === index ? { ...l, _status: st } : l)))
  }

  const generateAt = useCallback(async (index: number): Promise<Draft | null> => {
    const lead = leadsRef.current[index]
    if (!lead?.website) return null
    const opts = optsRef.current
    setGenerating(true)
    setDraft(null)
    setChat([])
    setCurrentIndex(index)
    setLeadStatus(index, '⚙️ Generating…')
    try {
      const res = await api.campaignGenerate({
        lead,
        template: opts.template,
        recipient_override: opts.recipientOverride,
        sender_email: opts.senderEmail,
        row_index: index,
      })
      const d: Draft = {
        draftId: res.draft_id,
        rowIndex: index,
        website: res.website,
        company: res.company,
        subject: res.subject,
        html: res.html,
        to: res.to,
      }
      setDraft(d)
      setChat([
        {
          role: 'system',
          content: `Draft ready for ${res.company || res.website}. Tell me what to change, or hit Send.`,
        },
      ])
      setLeadStatus(index, '📝 Ready for review')
      return d
    } catch (e: any) {
      setLeadStatus(index, `❌ ${e.message || 'Failed'}`)
      setLogs((prev) => [...prev, e.message || String(e)])
      return null
    } finally {
      setGenerating(false)
    }
  }, [])

  const advanceReview = useCallback(async (fromIndex: number) => {
    const list = leadsRef.current
    for (let i = fromIndex + 1; i < list.length; i++) {
      if (stopReviewRef.current) {
        setStatus('stopped')
        return
      }
      if (!list[i]?.website) continue
      const d = await generateAt(i)
      if (d) {
        setStatus('reviewing')
        return
      }
    }
    setDraft(null)
    setStatus('done')
  }, [generateAt])

  const startReview = useCallback(async () => {
    stopReviewRef.current = false
    setStatus('reviewing')
    setLogs([])
    setLeads((prev) => prev.map((l) => ({ ...l, _status: '' })))
    for (let i = 0; i < leadsRef.current.length; i++) {
      if (stopReviewRef.current) {
        setStatus('stopped')
        return
      }
      if (!leadsRef.current[i]?.website) continue
      const d = await generateAt(i)
      if (d) return
    }
    setStatus('done')
  }, [generateAt])

  const startAutosend = useCallback(async () => {
    setStatus('running')
    setLogs([])
    setDraft(null)
    setLeads((prev) => prev.map((l) => ({ ...l, _status: '' })))

    const ctrl = new AbortController()
    abortRef.current = ctrl
    const token = localStorage.getItem('token')
    const opts = optsRef.current

    try {
      const res = await fetch(`${getApiBase()}/api/process-stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        body: JSON.stringify({
          leads: leadsRef.current,
          template: opts.template,
          delay: opts.delay,
          sender_email: opts.senderEmail,
          recipient_override: opts.recipientOverride,
          autosend: true,
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
          if (evt.type === 'log') setLogs((prev) => [...prev, evt.message || String(evt)])
          if (evt.type === 'status_update') setLeadStatus(evt.row_index, evt.status)
          if (evt.type === 'done') setStatus('done')
        }
      }
      setStatus((s) => (s === 'running' ? 'done' : s))
    } catch (e: any) {
      if (e.name === 'AbortError') return
      setLogs((prev) => [...prev, e.message])
      setStatus('failed')
    }
  }, [])

  const start = useCallback(async (): Promise<'review' | 'live'> => {
    if (optsRef.current.autosend) {
      void startAutosend()
      return 'live'
    }
    void startReview()
    return 'review'
  }, [startAutosend, startReview])

  const sendCurrent = useCallback(async () => {
    if (!draft) return
    setSending(true)
    try {
      await api.campaignSend(draft.draftId)
      setLeadStatus(draft.rowIndex, '✅ Sent')
      setChat((prev) => [...prev, { role: 'assistant', content: `Sent to ${draft.to || 'recipient'}.` }])
      setDraft(null)
      await advanceReview(draft.rowIndex)
    } catch (e: any) {
      setLeadStatus(draft.rowIndex, `❌ ${e.message || 'Send failed'}`)
      setChat((prev) => [...prev, { role: 'assistant', content: e.message || 'Send failed' }])
    } finally {
      setSending(false)
    }
  }, [draft, advanceReview])

  const skipCurrent = useCallback(async () => {
    if (!draft) return
    try {
      await api.campaignDiscard(draft.draftId)
    } catch {
      /* ignore */
    }
    setLeadStatus(draft.rowIndex, '⏭ Skipped')
    setDraft(null)
    await advanceReview(draft.rowIndex)
  }, [draft, advanceReview])

  const reviseCurrent = useCallback(async (message: string) => {
    if (!draft || !message.trim()) return
    setRevising(true)
    setChat((prev) => [...prev, { role: 'user', content: message.trim() }])
    try {
      const res = await api.campaignRevise(draft.draftId, message.trim())
      setDraft((d) =>
        d
          ? { ...d, subject: res.subject, html: res.html }
          : d,
      )
      setChat((prev) => [...prev, { role: 'assistant', content: 'Updated — check the preview.' }])
    } catch (e: any) {
      setChat((prev) => [...prev, { role: 'assistant', content: e.message || 'Could not revise' }])
    } finally {
      setRevising(false)
    }
  }, [draft])

  const counts = useMemo(() => {
    let sent = 0
    let failed = 0
    let skipped = 0
    for (const l of leads) {
      const s = leadState(l)
      if (s === 'sent') sent += 1
      else if (s === 'failed') failed += 1
      else if (s === 'skipped') skipped += 1
    }
    return {
      total: leads.length,
      sent,
      failed,
      skipped,
      pending: leads.length - sent - failed - skipped,
      processed: sent + failed + skipped,
    }
  }, [leads])

  const value: CampaignValue = {
    leads,
    fileName,
    template,
    delay,
    recipientOverride,
    senderEmail,
    autosend,
    status,
    logs,
    draft,
    chat,
    generating,
    revising,
    sending,
    currentIndex,
    counts,
    setTemplate,
    setDelay,
    setRecipientOverride,
    setSenderEmail,
    setAutosend,
    uploadLeads,
    removeLead,
    clearLeads,
    start,
    stop,
    reset,
    sendCurrent,
    skipCurrent,
    reviseCurrent,
  }

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}

export function useCampaign() {
  const v = useContext(Ctx)
  if (!v) throw new Error('useCampaign must be used inside CampaignProvider')
  return v
}

export { leadState }
