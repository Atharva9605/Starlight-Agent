import { createContext, useCallback, useContext, useMemo, useRef, useState } from 'react'
import type { ReactNode } from 'react'
import { api, getApiBase } from '../api/client'

export type Lead = Record<string, any>
export type RunStatus = 'idle' | 'running' | 'reviewing' | 'done' | 'stopped' | 'failed'

export type ChatMessage = { role: 'user' | 'assistant' | 'system'; content: string }

export type StageEvent = {
  stage: string
  label: string
  state: 'active' | 'done' | 'error' | string
  at?: number
}

export type LivePreview = {
  rowIndex: number
  html: string
  subject: string
  company?: string
  website?: string
  to?: string
  from?: string
  productCount?: number
  productSheet?: string
}


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
  attachProductSheet: boolean
  status: RunStatus
  logs: string[]
  draft: Draft | null
  chat: ChatMessage[]
  generating: boolean
  revising: boolean
  sending: boolean
  currentIndex: number
  livePreview: LivePreview | null
  stagesByLead: Record<number, StageEvent[]>
  runSender: string
  counts: {
    total: number
    sent: number
    failed: number
    pending: number
    processed: number
    skipped: number
    ready: number
    processing: number
    progressPct: number
  }
  setTemplate: (v: string) => void
  setDelay: (v: number) => void
  setRecipientOverride: (v: string) => void
  setSenderEmail: (v: string) => void
  setAutosend: (v: boolean) => void
  setAttachProductSheet: (v: boolean) => void
  uploadLeads: (file: File) => Promise<void>
  removeLead: (index: number) => void
  clearLeads: () => void
  start: () => Promise<'live'>
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
  const [attachProductSheet, setAttachProductSheet] = useState(true)
  const [status, setStatus] = useState<RunStatus>('idle')
  const [logs, setLogs] = useState<string[]>([])
  const [draft, setDraft] = useState<Draft | null>(null)
  const [chat, setChat] = useState<ChatMessage[]>([])
  const [generating, setGenerating] = useState(false)
  const [revising, setRevising] = useState(false)
  const [sending, setSending] = useState(false)
  const [currentIndex, setCurrentIndex] = useState(0)
  const [livePreview, setLivePreview] = useState<LivePreview | null>(null)
  const [stagesByLead, setStagesByLead] = useState<Record<number, StageEvent[]>>({})
  const [runSender, setRunSender] = useState('')

  const abortRef = useRef<AbortController | null>(null)
  const stopReviewRef = useRef(false)
  const leadsRef = useRef<Lead[]>([])
  const optsRef = useRef({ template, delay, recipientOverride, senderEmail, autosend, attachProductSheet })
  leadsRef.current = leads
  optsRef.current = { template, delay, recipientOverride, senderEmail, autosend, attachProductSheet }


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
    setLivePreview(null)
    setStagesByLead({})
    setRunSender('')
  }, [])

  const reset = useCallback(() => {
    abortRef.current?.abort()
    stopReviewRef.current = true
    setLeads((prev) => prev.map((l) => ({ ...l, _status: '', _preview_html: '', _subject: '' })))
    setDraft(null)
    setChat([])
    setLogs([])
    setStatus('idle')
    setCurrentIndex(0)
    setGenerating(false)
    setRevising(false)
    setSending(false)
    setLivePreview(null)
    setStagesByLead({})
    setRunSender('')
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

  const pushStage = (index: number, evt: StageEvent) => {
    setStagesByLead((prev) => {
      const list = [...(prev[index] || [])]
      const existing = list.findIndex((s) => s.stage === evt.stage)
      const next = { ...evt, at: Date.now() }
      if (existing >= 0) list[existing] = next
      else list.push(next)
      return { ...prev, [index]: list }
    })
  }

  const generateAt = useCallback(async (index: number): Promise<Draft | null> => {
    const lead = leadsRef.current[index]
    if (!lead?.website) return null
    const opts = optsRef.current
    setGenerating(true)
    setDraft(null)
    setLivePreview(null)
    setChat([])
    setCurrentIndex(index)
    setLeadStatus(index, '⚙️ Generating…')
    // Mirror autosend stage rail so Live progress logs shows work in review mode.
    for (const stage of ['queued', 'scrape', 'analyze', 'retrieve', 'draft'] as const) {
      pushStage(index, {
        stage,
        label: stage === 'retrieve' ? 'Catalogue' : stage[0].toUpperCase() + stage.slice(1),
        state: stage === 'draft' ? 'active' : 'done',
      })
    }
    pushStage(index, { stage: 'render', label: 'Preview', state: 'pending' })
    pushStage(index, { stage: 'send', label: 'Send', state: 'pending' })
    try {
      const res = await api.campaignGenerate({
        lead,
        template: opts.template,
        recipient_override: opts.recipientOverride,
        sender_email: opts.senderEmail,
        row_index: index,
        attach_product_sheet: opts.attachProductSheet,
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
      setLivePreview({
        rowIndex: index,
        html: res.html,
        subject: res.subject,
        company: res.company,
        website: res.website,
        to: res.to,
        from: opts.senderEmail || undefined,
        productCount: res.product_count,
        productSheet: res.product_sheet,
      })
      setLeads((prev) =>
        prev.map((l, i) =>
          i === index
            ? {
                ...l,
                _preview_html: res.html,
                _subject: res.subject,
                _product_sheet: res.product_sheet,
                _product_count: res.product_count,
              }
            : l,
        ),
      )
      pushStage(index, { stage: 'draft', label: 'Draft', state: 'done' })
      pushStage(index, { stage: 'render', label: 'Preview', state: 'done' })
      pushStage(index, { stage: 'send', label: 'Send', state: 'active' })
      setChat([
        {
          role: 'system',
          content: `Draft ready for ${res.company || res.website}. Tell me what to change, or hit Send.`,
        },
      ])
      setLeadStatus(index, '📝 Ready for review')
      return d
    } catch (e: any) {
      pushStage(index, { stage: 'error', label: e.message || 'Failed', state: 'error' })
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
    setStagesByLead({})
    setLivePreview(null)
    setLeads((prev) => prev.map((l) => ({ ...l, _status: '', _preview_html: '', _subject: '' })))
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
    setLivePreview(null)
    setStagesByLead({})
    setLeads((prev) => prev.map((l) => ({ ...l, _status: '', _preview_html: '', _subject: '' })))

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
          attach_product_sheet: opts.attachProductSheet,
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
          if (evt.type === 'run_meta') {
            if (evt.sender_email) setRunSender(evt.sender_email)
          }
          if (evt.type === 'status_update') {
            setLeadStatus(evt.row_index, evt.status)
            setCurrentIndex(evt.row_index)
          }
          if (evt.type === 'stage') {
            pushStage(evt.row_index, {
              stage: evt.stage,
              label: evt.label,
              state: evt.state || 'active',
            })
            setCurrentIndex(evt.row_index)
          }
          if (evt.type === 'preview_html') {
            const preview: LivePreview = {
              rowIndex: evt.row_index,
              html: evt.html || '',
              subject: evt.subject || 'Starlight outreach',
              company: evt.company,
              website: evt.website,
              to: evt.to,
              from: evt.from,
              productCount: evt.product_count,
              productSheet: evt.product_sheet,
            }
            setLivePreview(preview)
            setCurrentIndex(evt.row_index)
            setLeads((prev) =>
              prev.map((l, i) =>
                i === evt.row_index
                  ? {
                      ...l,
                      _preview_html: preview.html,
                      _subject: preview.subject,
                      _product_sheet: preview.productSheet,
                      _product_count: preview.productCount,
                    }
                  : l,
              ),
            )
          }
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

  const start = useCallback(async (): Promise<'live'> => {
    if (optsRef.current.autosend) {
      void startAutosend()
    } else {
      void startReview()
    }
    return 'live'
  }, [startAutosend, startReview])

  const sendCurrent = useCallback(async () => {
    if (!draft) return
    setSending(true)
    try {
      await api.campaignSend(draft.draftId)
      pushStage(draft.rowIndex, { stage: 'send', label: 'Send', state: 'done' })
      setLeadStatus(draft.rowIndex, '✅ Sent')
      setChat((prev) => [...prev, { role: 'assistant', content: `Sent to ${draft.to || 'recipient'}.` }])
      setDraft(null)
      setLivePreview(null)
      await advanceReview(draft.rowIndex)
    } catch (e: any) {
      pushStage(draft.rowIndex, { stage: 'send', label: 'Send', state: 'error' })
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
    setLivePreview(null)
    await advanceReview(draft.rowIndex)
  }, [draft, advanceReview])

  const reviseCurrent = useCallback(async (message: string) => {
    if (!draft || !message.trim()) return
    setRevising(true)
    setChat((prev) => [...prev, { role: 'user', content: message.trim() }])
    try {
      const res = await api.campaignRevise(draft.draftId, message.trim())
      setDraft((d) => (d ? { ...d, subject: res.subject, html: res.html } : d))
      setLivePreview((p) =>
        p && p.rowIndex === draft.rowIndex
          ? { ...p, html: res.html, subject: res.subject }
          : {
              rowIndex: draft.rowIndex,
              html: res.html,
              subject: res.subject,
              company: draft.company,
              website: draft.website,
              to: draft.to,
            },
      )
      setLeads((prev) =>
        prev.map((l, i) =>
          i === draft.rowIndex ? { ...l, _preview_html: res.html, _subject: res.subject } : l,
        ),
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
    let ready = 0
    let processing = 0
    for (const l of leads) {
      const s = leadState(l)
      if (s === 'sent') sent += 1
      else if (s === 'failed') failed += 1
      else if (s === 'skipped') skipped += 1
      else if (s === 'ready') ready += 1
      else if (s === 'processing') processing += 1
    }
    const finished = sent + failed + skipped
    const progressed = finished + ready + processing * 0.55
    const total = leads.length
    return {
      total,
      sent,
      failed,
      skipped,
      ready,
      processing,
      pending: Math.max(0, total - finished - ready - processing),
      processed: finished,
      progressPct: total ? Math.min(100, Math.round((progressed / total) * 100)) : 0,
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
    attachProductSheet,
    status,
    logs,
    draft,
    chat,
    generating,
    revising,
    sending,
    currentIndex,
    livePreview,
    stagesByLead,
    runSender,
    counts,
    setTemplate,
    setDelay,
    setRecipientOverride,
    setSenderEmail,
    setAutosend,
    setAttachProductSheet,
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
