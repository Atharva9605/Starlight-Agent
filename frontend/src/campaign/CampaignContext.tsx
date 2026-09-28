import { createContext, useCallback, useContext, useMemo, useRef, useState } from 'react'
import type { ReactNode } from 'react'
import { api, getApiBase } from '../api/client'

export type Lead = Record<string, any>
export type RunStatus = 'idle' | 'running' | 'reviewing' | 'paused' | 'done' | 'stopped' | 'failed'

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
  from?: string
}

const GEN_CONCURRENCY = 3

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

function leadHasIdentity(lead: Lead | undefined | null): boolean {
  if (!lead) return false
  const website = String(lead.website || '').trim()
  const company = String(lead.company || lead.name || '').trim()
  return Boolean(website || company)
}

function leadState(lead: Lead): 'sent' | 'failed' | 'processing' | 'pending' | 'ready' | 'skipped' | 'queued' {
  const s = String(lead._status || '').toLowerCase()
  if (s.includes('sent') || s.includes('✅')) return 'sent'
  if (s.includes('skip') || s.includes('discard')) return 'skipped'
  if (s.includes('fail') || s.includes('error') || s.includes('❌')) return 'failed'
  if (lead._queued || s.includes('bulk send')) return 'queued'
  if (s.includes('ready') || s.includes('📝')) return 'ready'
  if (s.includes('waiting') || s.includes('⏳')) return 'pending'
  if (
    s.includes('process') ||
    s.includes('⚙') ||
    s.includes('generat') ||
    s.includes('retry') ||
    s.includes('finding') ||
    s.includes('applying') ||
    s.includes('✨')
  ) {
    return 'processing'
  }
  return 'pending'
}

function isTransientGenerateError(message: string): boolean {
  return /429|502|503|504|timeout|timed out|failed to fetch|network|econnreset|overload|rate limit/i.test(
    message,
  )
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
  sendingIndex: number | null
  bulkSending: boolean
  currentIndex: number
  livePreview: LivePreview | null
  stagesByLead: Record<number, StageEvent[]>
  draftsByLead: Record<number, Draft>
  runSender: string
  counts: {
    total: number
    sent: number
    failed: number
    pending: number
    processed: number
    skipped: number
    ready: number
    queued: number
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
  start: () => Promise<'live' | 'review'>
  stop: () => void
  pause: () => void
  resume: () => Promise<void>
  reset: () => void
  selectLead: (index: number) => void
  sendCurrent: () => Promise<void>
  sendLead: (index: number) => Promise<void>
  skipCurrent: () => Promise<void>
  discardLead: (index: number) => Promise<void>
  queueLead: (index: number, queued?: boolean) => void
  sendBulk: () => Promise<void>
  retryLead: (index: number) => Promise<void>
  retryFailed: () => Promise<void>
  reviseCurrent: (message: string) => Promise<void>
  applyBulkEdit: (message: string) => Promise<void>
  removeBulkEdit: (index: number) => void
  clearBulkEdits: () => void
  bulkEdits: string[]
  bulkRevising: boolean
  bulkReviseProgress: { done: number; total: number }
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
  const [sendingIndex, setSendingIndex] = useState<number | null>(null)
  const [bulkSending, setBulkSending] = useState(false)
  const [currentIndex, setCurrentIndex] = useState(0)
  const [livePreview, setLivePreview] = useState<LivePreview | null>(null)
  const [stagesByLead, setStagesByLead] = useState<Record<number, StageEvent[]>>({})
  const [draftsByLead, setDraftsByLead] = useState<Record<number, Draft>>({})
  const [runSender, setRunSender] = useState('')
  const [bulkEdits, setBulkEdits] = useState<string[]>([])
  const [bulkRevising, setBulkRevising] = useState(false)
  const [bulkReviseProgress, setBulkReviseProgress] = useState({ done: 0, total: 0 })

  const abortRef = useRef<AbortController | null>(null)
  const stopReviewRef = useRef(false)
  const pauseRef = useRef(false)
  const bulkEditsRef = useRef<string[]>([])
  const leadsRef = useRef<Lead[]>([])
  const currentIndexRef = useRef(0)
  const inflightRef = useRef(0)
  const draftsRef = useRef<Record<number, Draft>>({})
  const chatByLeadRef = useRef<Record<number, ChatMessage[]>>({})
  const optsRef = useRef({ template, delay, recipientOverride, senderEmail, autosend, attachProductSheet })
  leadsRef.current = leads
  currentIndexRef.current = currentIndex
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
    setDraftsByLead({})
    draftsRef.current = {}
    chatByLeadRef.current = {}
    setSendingIndex(null)
    setBulkSending(false)
    setBulkEdits([])
    bulkEditsRef.current = []
    pauseRef.current = false
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
    setDraftsByLead({})
    draftsRef.current = {}
    chatByLeadRef.current = {}
    setRunSender('')
    setSendingIndex(null)
    setBulkSending(false)
    setBulkEdits([])
    bulkEditsRef.current = []
    pauseRef.current = false
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
    setSendingIndex(null)
    setBulkSending(false)
    setLivePreview(null)
    setStagesByLead({})
    setDraftsByLead({})
    draftsRef.current = {}
    chatByLeadRef.current = {}
    setRunSender('')
    inflightRef.current = 0
    pauseRef.current = false
    setBulkEdits([])
    bulkEditsRef.current = []
    setBulkRevising(false)
    setBulkReviseProgress({ done: 0, total: 0 })
  }, [])

  const stop = useCallback(() => {
    abortRef.current?.abort()
    pauseRef.current = false
    stopReviewRef.current = true
    setStatus('stopped')
    setGenerating(false)
  }, [])

  const pause = useCallback(() => {
    abortRef.current?.abort()
    pauseRef.current = true
    stopReviewRef.current = true
    setStatus('paused')
    setGenerating(false)
  }, [])

  const setLeadStatus = (index: number, st: string) => {
    setLeads((prev) => {
      const next = prev.map((l, i) => (i === index ? { ...l, _status: st } : l))
      leadsRef.current = next
      return next
    })
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

  const applyLeadChat = (index: number, messages: ChatMessage[]) => {
    chatByLeadRef.current = { ...chatByLeadRef.current, [index]: messages }
    if (currentIndexRef.current === index) setChat(messages)
  }

  const storeDraft = (index: number, d: Draft) => {
    draftsRef.current = { ...draftsRef.current, [index]: d }
    setDraftsByLead(draftsRef.current)
    if (currentIndexRef.current === index) {
      setDraft(d)
    }
  }

  const dropDraft = (index: number) => {
    const next = { ...draftsRef.current }
    delete next[index]
    draftsRef.current = next
    setDraftsByLead(next)
    if (currentIndexRef.current === index) {
      setDraft(null)
    }
  }

  const selectLead = useCallback((index: number) => {
    currentIndexRef.current = index
    setCurrentIndex(index)
    const d = draftsRef.current[index]
    setDraft(d || null)
    setChat(chatByLeadRef.current[index] || [])
    const lead = leadsRef.current[index]
    const html = d?.html || lead?._preview_html || ''
    if (html) {
      setLivePreview({
        rowIndex: index,
        html,
        subject: d?.subject || lead?._subject || 'Starlight outreach',
        company: d?.company || lead?.company,
        website: d?.website || lead?.website,
        to: d?.to || lead?._to,
        from: d?.from || optsRef.current.senderEmail || undefined,
        productCount: lead?._product_count,
        productSheet: lead?._product_sheet,
      })
    } else {
      setLivePreview(null)
    }
  }, [])

  const generateAt = useCallback(async (
    index: number,
    genOpts?: { forceDiscover?: boolean },
  ): Promise<Draft | null> => {
    const lead = { ...(leadsRef.current[index] || {}) }
    if (!leadHasIdentity(lead)) return null
    const forceDiscover = Boolean(genOpts?.forceDiscover)
    const opts = optsRef.current
    const companyLabel = String(lead?.company || lead?.name || '').trim()
    const failedSite = String(lead._failed_website || lead.website || '').trim()
    const exclude = Array.from(
      new Set(
        [...(Array.isArray(lead._exclude_websites) ? lead._exclude_websites : []), failedSite].filter(
          Boolean,
        ),
      ),
    )
    const needsDiscover = forceDiscover || !String(lead?.website || '').trim()

    inflightRef.current += 1
    setGenerating(true)
    setStagesByLead((prev) => ({ ...prev, [index]: [] }))
    setLeadStatus(
      index,
      forceDiscover || needsDiscover
        ? `🔎 OpenSERP · ${companyLabel || 'company'}`
        : '⚙️ Generating…',
    )

    pushStage(index, { stage: 'queued', label: 'Queued', state: 'done' })
    if (forceDiscover || needsDiscover) {
      pushStage(index, {
        stage: 'discover',
        label: `OpenSERP · ${companyLabel || 'company'}`,
        state: 'active',
      })
    } else {
      pushStage(index, { stage: 'scrape', label: 'Scraping website', state: 'active' })
    }

    try {
      const res = await (async () => {
        let lastErr = ''
        for (let attempt = 0; attempt < 2; attempt++) {
          try {
            return await api.campaignGenerate({
              lead,
              template: opts.template,
              recipient_override: opts.recipientOverride,
              sender_email: opts.senderEmail,
              row_index: index,
              attach_product_sheet: opts.attachProductSheet,
              force_discover: forceDiscover,
              exclude_websites: forceDiscover ? exclude : undefined,
            })
          } catch (e: any) {
            lastErr = e.message || 'Failed'
            if (attempt === 0 && isTransientGenerateError(lastErr) && !stopReviewRef.current) {
              setLeadStatus(index, '🔁 Retrying…')
              await new Promise((r) => setTimeout(r, 2000))
              continue
            }
            throw e
          }
        }
        throw new Error(lastErr || 'Failed')
      })()
      const usedDiscover = needsDiscover || Boolean(res.discovered)
      const d: Draft = {
        draftId: res.draft_id,
        rowIndex: index,
        website: res.website,
        company: res.company,
        subject: res.subject,
        html: res.html,
        to: res.to,
        from: res.from || opts.senderEmail || undefined,
      }
      storeDraft(index, d)
      if (currentIndexRef.current === index) {
        setLivePreview({
          rowIndex: index,
          html: res.html,
          subject: res.subject,
          company: res.company,
          website: res.website,
          to: res.to,
          from: d.from,
          productCount: res.product_count,
          productSheet: res.product_sheet,
        })
      }
      setLeads((prev) => {
        const next = prev.map((l, i) =>
          i === index
            ? {
                ...l,
                website: res.website || l.website,
                company: res.company || l.company,
                _preview_html: res.html,
                _subject: res.subject,
                _product_sheet: res.product_sheet,
                _product_count: res.product_count,
                _discovered: usedDiscover,
                _draft_id: res.draft_id,
                _to: res.to,
                _queued: false,
                _status: '📝 Ready for review',
                _error: '',
              }
            : l,
        )
        leadsRef.current = next
        return next
      })
      if (usedDiscover) {
        pushStage(index, {
          stage: 'discover',
          label: `Found ${res.website}`,
          state: 'done',
        })
      }
      const doneStages: { stage: string; label: string }[] = [
        { stage: 'scrape', label: 'Website profile ready' },
        { stage: 'analyze', label: 'Company analyzed' },
        { stage: 'retrieve', label: 'Catalogue matched' },
        { stage: 'draft', label: 'Draft written' },
        { stage: 'render', label: 'Preview ready' },
      ]
      for (const s of doneStages) {
        pushStage(index, { stage: s.stage, label: s.label, state: 'done' })
      }
      pushStage(index, { stage: 'send', label: 'Send', state: 'active' })
      applyLeadChat(index, [
        {
          role: 'system',
          content: usedDiscover
            ? `Found ${res.website} via OpenSERP for ${res.company || companyLabel}. Send now, add to bulk, or discard.`
            : `Draft ready for ${res.company || res.website}. Send now, add to bulk, or discard.`,
        },
      ])
      let finalDraft = d
      const instructions = bulkEditsRef.current.slice()
      if (instructions.length) {
        setLeadStatus(index, '✨ Applying bulk AI edit…')
        for (const msg of instructions) {
          try {
            const rev = await api.campaignRevise(finalDraft.draftId, msg)
            finalDraft = { ...finalDraft, subject: rev.subject, html: rev.html }
            storeDraft(index, finalDraft)
            if (currentIndexRef.current === index) {
              setLivePreview((p) =>
                p && p.rowIndex === index
                  ? { ...p, html: rev.html, subject: rev.subject }
                  : p,
              )
              setDraft(finalDraft)
            }
            setLeads((prev) => {
              const next = prev.map((l, i) =>
                i === index ? { ...l, _preview_html: rev.html, _subject: rev.subject } : l,
              )
              leadsRef.current = next
              return next
            })
          } catch {
            break
          }
        }
        applyLeadChat(index, [
          ...(chatByLeadRef.current[index] || []),
          {
            role: 'system',
            content: `Applied ${instructions.length} bulk AI edit${instructions.length === 1 ? '' : 's'} to this new email.`,
          },
        ])
      }
      setLeadStatus(index, '📝 Ready for review')
      setLogs((prev) => [
        ...prev,
        usedDiscover
          ? `OpenSERP → ${res.website} · draft ready`
          : `Draft ready for ${res.website}`,
      ])
      return finalDraft
    } catch (e: any) {
      const err = e.message || 'Failed'
      pushStage(index, { stage: 'error', label: err, state: 'error' })
      setLeads((prev) => {
        const next = prev.map((l, i) =>
          i === index
            ? {
                ...l,
                _status: `❌ ${err}`,
                _error: err,
                _failed_website: l._failed_website || l.website || '',
                _exclude_websites: Array.from(
                  new Set(
                    [
                      ...(Array.isArray(l._exclude_websites) ? l._exclude_websites : []),
                      l.website,
                    ].filter(Boolean),
                  ),
                ),
              }
            : l,
        )
        leadsRef.current = next
        return next
      })
      applyLeadChat(index, [
        {
          role: 'system',
          content: `Could not generate this email: ${err}`,
        },
      ])
      setLogs((prev) => [...prev, `Lead ${index + 1} failed: ${err}`])
      return null
    } finally {
      inflightRef.current = Math.max(0, inflightRef.current - 1)
      if (inflightRef.current === 0) setGenerating(false)
    }
  }, [])

  const reviewSettled = () => {
    if (inflightRef.current > 0) return
    if (pauseRef.current) {
      setStatus('paused')
      return
    }
    const open = leadsRef.current.some((l) => {
      const s = leadState(l)
      return s === 'ready' || s === 'queued' || s === 'processing' || s === 'pending'
    })
    if (!open) {
      setStatus('done')
      return
    }
    if (stopReviewRef.current) setStatus('stopped')
  }

  const remainingGenerateJobs = () =>
    leadsRef.current
      .map((_, i) => i)
      .filter((i) => {
        const lead = leadsRef.current[i]
        if (!leadHasIdentity(lead)) return false
        if (draftsRef.current[i]) return false
        return leadState(lead) === 'pending'
      })

  const runGenerateQueue = async (jobs: number[], genOpts?: { forceDiscover?: boolean }) => {
    let cursor = 0
    const worker = async () => {
      while (cursor < jobs.length) {
        if (stopReviewRef.current || pauseRef.current) return
        const i = jobs[cursor]
        cursor += 1
        await generateAt(i, genOpts)
      }
    }
    const n = Math.min(GEN_CONCURRENCY, jobs.length)
    if (n > 0) await Promise.all(Array.from({ length: n }, () => worker()))
  }

  const startReview = useCallback(async () => {
    stopReviewRef.current = false
    pauseRef.current = false
    inflightRef.current = 0
    setStatus('reviewing')
    setLogs([])
    setStagesByLead({})
    setLivePreview(null)
    setDraft(null)
    setChat([])
    setDraftsByLead({})
    draftsRef.current = {}
    chatByLeadRef.current = {}
    setSendingIndex(null)
    setBulkSending(false)
    setBulkEdits([])
    bulkEditsRef.current = []

    const list = leadsRef.current
    const jobs: number[] = []
    for (let i = 0; i < list.length; i++) {
      if (leadHasIdentity(list[i])) jobs.push(i)
    }
    setLeads((prev) => {
      const next = prev.map((l) => {
        const skip = !leadHasIdentity(l)
        return {
          ...l,
          _status: skip ? '⏭ Skipped (need company or website)' : '⏳ Waiting to generate',
          _preview_html: '',
          _subject: '',
          _discovered: false,
          _queued: false,
          _draft_id: '',
          _to: '',
          _error: '',
          _failed_website: '',
          _exclude_websites: [],
        }
      })
      leadsRef.current = next
      return next
    })

    const first = jobs[0] ?? 0
    currentIndexRef.current = first
    setCurrentIndex(first)

    await runGenerateQueue(jobs)
    reviewSettled()
  }, [generateAt])

  const startAutosend = useCallback(async (resume = false) => {
    setStatus('running')
    pauseRef.current = false
    stopReviewRef.current = false
    if (!resume) {
      setLogs([])
      setDraft(null)
      setLivePreview(null)
      setStagesByLead({})
      setLeads((prev) => prev.map((l) => ({ ...l, _status: '', _preview_html: '', _subject: '' })))
    }

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
          if (evt.type === 'lead_resolved') {
            setLeads((prev) =>
              prev.map((l, i) =>
                i === evt.row_index
                  ? {
                      ...l,
                      website: evt.website || l.website,
                      company: evt.company || l.company,
                    }
                  : l,
              ),
            )
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
      if (e.name === 'AbortError') {
        if (pauseRef.current) setStatus('paused')
        return
      }
      setLogs((prev) => [...prev, e.message])
      setStatus('failed')
    }
  }, [])

  const start = useCallback(async (): Promise<'live' | 'review'> => {
    if (optsRef.current.autosend) {
      void startAutosend()
      return 'live'
    }
    void startReview()
    return 'review'
  }, [startAutosend, startReview])

  const resume = useCallback(async () => {
    pauseRef.current = false
    stopReviewRef.current = false
    if (optsRef.current.autosend) {
      void startAutosend(true)
      return
    }
    setStatus('reviewing')
    const jobs = remainingGenerateJobs()
    await runGenerateQueue(jobs)
    reviewSettled()
  }, [startAutosend, generateAt])

  const sendLead = useCallback(async (index: number) => {
    const d = draftsRef.current[index]
    if (!d) return
    setSending(true)
    setSendingIndex(index)
    try {
      await api.campaignSend(d.draftId)
      pushStage(index, { stage: 'send', label: 'Send', state: 'done' })
      setLeads((prev) => {
        const next = prev.map((l, i) => (i === index ? { ...l, _status: '✅ Sent', _queued: false } : l))
        leadsRef.current = next
        return next
      })
      const nextChat = [
        ...(chatByLeadRef.current[index] || []),
        { role: 'assistant' as const, content: `Sent to ${d.to || 'recipient'}.` },
      ]
      applyLeadChat(index, nextChat)
      dropDraft(index)
      setLogs((prev) => [...prev, `Sent · ${d.company || d.website || `lead ${index + 1}`}`])
    } catch (e: any) {
      pushStage(index, { stage: 'send', label: 'Send', state: 'error' })
      setLeadStatus(index, `❌ ${e.message || 'Send failed'}`)
      const nextChat = [
        ...(chatByLeadRef.current[index] || []),
        { role: 'assistant' as const, content: e.message || 'Send failed' },
      ]
      applyLeadChat(index, nextChat)
    } finally {
      setSending(false)
      setSendingIndex((cur) => (cur === index ? null : cur))
      reviewSettled()
    }
  }, [])

  const sendCurrent = useCallback(async () => {
    await sendLead(currentIndexRef.current)
  }, [sendLead])

  const discardLead = useCallback(async (index: number) => {
    const d = draftsRef.current[index]
    if (d) {
      try {
        await api.campaignDiscard(d.draftId)
      } catch {
        /* ignore */
      }
    }
    setLeads((prev) => {
      const next = prev.map((l, i) => (i === index ? { ...l, _status: '⏭ Discarded', _queued: false } : l))
      leadsRef.current = next
      return next
    })
    dropDraft(index)
    setLogs((prev) => [...prev, `Discarded lead ${index + 1}`])
    reviewSettled()
  }, [])

  const skipCurrent = useCallback(async () => {
    await discardLead(currentIndexRef.current)
  }, [discardLead])

  const queueLead = useCallback((index: number, queued?: boolean) => {
    const d = draftsRef.current[index]
    if (!d) return
    const lead = leadsRef.current[index]
    const queuedNow = queued ?? !lead?._queued
    setLeads((prev) => {
      const updated = prev.map((l, i) =>
        i === index
          ? {
              ...l,
              _queued: queuedNow,
              _status: queuedNow ? '📦 In bulk send' : '📝 Ready for review',
            }
          : l,
      )
      leadsRef.current = updated
      return updated
    })
  }, [])

  const sendBulk = useCallback(async () => {
    const ids = leadsRef.current
      .map((l, i) => ({ l, i }))
      .filter(({ l, i }) => Boolean(l._queued) && draftsRef.current[i])
      .map(({ i }) => i)
    if (!ids.length) return
    stopReviewRef.current = false
    setBulkSending(true)
    setSending(true)
    try {
      for (let n = 0; n < ids.length; n++) {
        if (stopReviewRef.current) break
        await sendLead(ids[n])
        if (n < ids.length - 1 && !stopReviewRef.current) {
          const wait = Math.max(0, optsRef.current.delay) * 1000
          if (wait) await new Promise((r) => setTimeout(r, wait))
        }
      }
    } finally {
      setBulkSending(false)
      setSending(false)
      reviewSettled()
    }
  }, [sendLead])

  const retryLead = useCallback(async (index: number) => {
    stopReviewRef.current = false
    setStatus('reviewing')
    await generateAt(index, { forceDiscover: true })
    reviewSettled()
  }, [generateAt])

  const retryFailed = useCallback(async () => {
    stopReviewRef.current = false
    pauseRef.current = false
    setStatus('reviewing')
    const ids = leadsRef.current
      .map((l, i) => ({ l, i }))
      .filter(({ l }) => leadState(l) === 'failed')
      .map(({ i }) => i)
    if (!ids.length) return
    await runGenerateQueue(ids, { forceDiscover: true })
    reviewSettled()
  }, [generateAt])

  const applyRevision = async (index: number, message: string) => {
    const d = draftsRef.current[index]
    if (!d || !message.trim()) return
    const res = await api.campaignRevise(d.draftId, message.trim())
    const updated: Draft = { ...d, subject: res.subject, html: res.html }
    storeDraft(index, updated)
    if (currentIndexRef.current === index) {
      setDraft(updated)
      setLivePreview((p) =>
        p && p.rowIndex === index ? { ...p, html: res.html, subject: res.subject } : p,
      )
    }
    setLeads((prev) => {
      const next = prev.map((l, i) =>
        i === index ? { ...l, _preview_html: res.html, _subject: res.subject } : l,
      )
      leadsRef.current = next
      return next
    })
    applyLeadChat(index, [
      ...(chatByLeadRef.current[index] || []),
      { role: 'user', content: `[Bulk] ${message.trim()}` },
      { role: 'assistant', content: 'Updated from bulk AI edit.' },
    ])
  }

  const applyBulkEdit = useCallback(async (message: string) => {
    const msg = message.trim()
    if (!msg) return
    bulkEditsRef.current = [...bulkEditsRef.current, msg]
    setBulkEdits(bulkEditsRef.current.slice())
    const targets = Object.keys(draftsRef.current)
      .map(Number)
      .filter((i) => {
        const s = leadState(leadsRef.current[i] || {})
        return s === 'ready' || s === 'queued'
      })
    if (!targets.length) {
      setLogs((prev) => [...prev, `Bulk AI edit saved for upcoming emails: ${msg}`])
      return
    }
    setBulkRevising(true)
    setBulkReviseProgress({ done: 0, total: targets.length })
    let cursor = 0
    let done = 0
    const worker = async () => {
      while (cursor < targets.length) {
        const i = targets[cursor]
        cursor += 1
        try {
          await applyRevision(i, msg)
        } catch {
          /* keep going */
        }
        done += 1
        setBulkReviseProgress({ done, total: targets.length })
      }
    }
    await Promise.all(Array.from({ length: Math.min(GEN_CONCURRENCY, targets.length) }, () => worker()))
    setBulkRevising(false)
    setLogs((prev) => [...prev, `Bulk AI edit applied to ${targets.length} emails and all upcoming drafts.`])
  }, [])

  const removeBulkEdit = useCallback((index: number) => {
    bulkEditsRef.current = bulkEditsRef.current.filter((_, i) => i !== index)
    setBulkEdits(bulkEditsRef.current.slice())
  }, [])

  const clearBulkEdits = useCallback(() => {
    bulkEditsRef.current = []
    setBulkEdits([])
  }, [])

  const reviseCurrent = useCallback(async (message: string) => {
    const index = currentIndexRef.current
    const d = draftsRef.current[index]
    if (!d || !message.trim()) return
    setRevising(true)
    const withUser = [
      ...(chatByLeadRef.current[index] || []),
      { role: 'user' as const, content: message.trim() },
    ]
    applyLeadChat(index, withUser)
    try {
      const res = await api.campaignRevise(d.draftId, message.trim())
      const updated: Draft = { ...d, subject: res.subject, html: res.html }
      storeDraft(index, updated)
      setLivePreview((p) =>
        p && p.rowIndex === index
          ? { ...p, html: res.html, subject: res.subject }
          : {
              rowIndex: index,
              html: res.html,
              subject: res.subject,
              company: d.company,
              website: d.website,
              to: d.to,
              from: d.from,
            },
      )
      setLeads((prev) =>
        prev.map((l, i) =>
          i === index ? { ...l, _preview_html: res.html, _subject: res.subject } : l,
        ),
      )
      applyLeadChat(index, [
        ...withUser,
        { role: 'assistant', content: 'Updated — check the preview.' },
      ])
    } catch (e: any) {
      applyLeadChat(index, [
        ...withUser,
        { role: 'assistant', content: e.message || 'Could not revise' },
      ])
    } finally {
      setRevising(false)
    }
  }, [])

  const counts = useMemo(() => {
    let sent = 0
    let failed = 0
    let skipped = 0
    let ready = 0
    let queued = 0
    let processing = 0
    for (const l of leads) {
      const s = leadState(l)
      if (s === 'sent') sent += 1
      else if (s === 'failed') failed += 1
      else if (s === 'skipped') skipped += 1
      else if (s === 'queued') queued += 1
      else if (s === 'ready') ready += 1
      else if (s === 'processing') processing += 1
    }
    const finished = sent + failed + skipped
    const progressed = finished + ready + queued + processing * 0.55
    const total = leads.length
    return {
      total,
      sent,
      failed,
      skipped,
      ready,
      queued,
      processing,
      pending: Math.max(0, total - finished - ready - queued - processing),
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
    sendingIndex,
    bulkSending,
    currentIndex,
    livePreview,
    stagesByLead,
    draftsByLead,
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
    pause,
    resume,
    reset,
    selectLead,
    sendCurrent,
    sendLead,
    skipCurrent,
    discardLead,
    queueLead,
    sendBulk,
    retryLead,
    retryFailed,
    reviseCurrent,
    applyBulkEdit,
    removeBulkEdit,
    clearBulkEdits,
    bulkEdits,
    bulkRevising,
    bulkReviseProgress,
  }

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}

export function useCampaign() {
  const v = useContext(Ctx)
  if (!v) throw new Error('useCampaign must be used inside CampaignProvider')
  return v
}

export { leadState }
