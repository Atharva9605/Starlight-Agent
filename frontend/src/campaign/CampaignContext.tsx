import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react'
import type { ReactNode } from 'react'
import { api, getApiBase } from '../api/client'
import type { CampaignRun, CampaignRunStatus } from '../api/client'

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

/**
 * The run a tab should snap back to after a reload. The server is the source of
 * truth; this is only a hint so a finished run can still be offered up.
 */
const RUN_STORAGE_KEY = 'starlight_campaign_run_id'

function runStatusToLocal(status: CampaignRunStatus | string | undefined): RunStatus {
  switch (status) {
    case 'running':
      return 'running'
    case 'paused':
      return 'paused'
    case 'stopped':
      return 'stopped'
    case 'failed':
      return 'failed'
    case 'done':
      return 'done'
    default:
      return 'idle'
  }
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

function isFailedLead(lead: Lead | undefined | null): boolean {
  if (!lead) return false
  const st = leadState(lead)
  return st === 'failed' || (st === 'processing' && Boolean(lead._failed_website))
}

function firstPreviewIndex(leads: Lead[], skip?: number): number {
  const ok = (l: Lead, i: number) => i !== skip && !isFailedLead(l)
  const ready = leads.findIndex(
    (l, i) => ok(l, i) && (leadState(l) === 'ready' || leadState(l) === 'queued'),
  )
  if (ready >= 0) return ready
  const writing = leads.findIndex((l, i) => {
    if (!ok(l, i)) return false
    const st = leadState(l)
    return st === 'processing' || st === 'pending'
  })
  if (writing >= 0) return writing
  return leads.findIndex((l, i) => ok(l, i))
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
  /** Server-side run backing the current campaign, if any. */
  runId: string | null
  runRecord: CampaignRun | null
  /** True while a snapshot is being loaded, so pages wait instead of redirecting. */
  attaching: boolean
  /** A run that finished while this tab was away — offer it, do not hijack. */
  finishedRun: CampaignRun | null
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
  /** Queue an autosend run on the server for a later time, then clear setup. */
  schedule: (at: Date) => Promise<void>
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
  attachRun: (runId: string) => Promise<CampaignRun | undefined>
  dismissFinishedRun: () => void
  /** Pull a lead's email body from the run record (bodies are not kept in state). */
  loadLeadPreview: (index: number) => Promise<void>
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
  const [runId, setRunId] = useState<string | null>(null)
  const [runRecord, setRunRecord] = useState<CampaignRun | null>(null)
  const [attaching, setAttaching] = useState(false)
  const [finishedRun, setFinishedRun] = useState<CampaignRun | null>(null)
  const [bulkEdits, setBulkEdits] = useState<string[]>([])
  const [bulkRevising, setBulkRevising] = useState(false)
  const [bulkReviseProgress, setBulkReviseProgress] = useState({ done: 0, total: 0 })

  const stopReviewRef = useRef(false)
  const pauseRef = useRef(false)
  const bulkEditsRef = useRef<string[]>([])
  const leadsRef = useRef<Lead[]>([])
  const currentIndexRef = useRef(0)
  const inflightRef = useRef(0)
  const draftsRef = useRef<Record<number, Draft>>({})
  const chatByLeadRef = useRef<Record<number, ChatMessage[]>>({})
  const optsRef = useRef({ template, delay, recipientOverride, senderEmail, autosend, attachProductSheet })
  const streamAbortRef = useRef<AbortController | null>(null)
  const cursorRef = useRef(0)
  const runIdRef = useRef<string | null>(null)
  const previewFetchRef = useRef<Set<number>>(new Set())
  leadsRef.current = leads
  currentIndexRef.current = currentIndex
  runIdRef.current = runId
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

  const forgetRun = () => {
    streamAbortRef.current?.abort()
    streamAbortRef.current = null
    cursorRef.current = 0
    setRunId(null)
    setRunRecord(null)
    setFinishedRun(null)
    try {
      localStorage.removeItem(RUN_STORAGE_KEY)
    } catch {
      /* ignore */
    }
  }

  const dismissFinishedRun = useCallback(() => {
    setFinishedRun(null)
    try {
      localStorage.removeItem(RUN_STORAGE_KEY)
    } catch {
      /* ignore */
    }
  }, [])

  const clearLeads = useCallback(() => {
    forgetRun()
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
    forgetRun()
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
    pauseRef.current = false
    stopReviewRef.current = true
    setGenerating(false)
    const id = runIdRef.current
    if (id) {
      setStatus('stopped')
      void api.stopCampaignRun(id).catch((e) =>
        setLogs((prev) => [...prev, e?.message || 'Could not stop the run']),
      )
      return
    }
    setStatus('stopped')
  }, [])

  const pause = useCallback(() => {
    pauseRef.current = true
    stopReviewRef.current = true
    setGenerating(false)
    const id = runIdRef.current
    if (id) {
      // The worker finishes the lead in flight, then stops — never mid-send.
      setLogs((prev) => [...prev, 'Pausing after the lead currently in flight…'])
      void api.pauseCampaignRun(id).catch((e) =>
        setLogs((prev) => [...prev, e?.message || 'Could not pause the run']),
      )
      return
    }
    setStatus('paused')
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
      if (currentIndexRef.current === index) {
        const next = firstPreviewIndex(leadsRef.current, index)
        if (next >= 0) selectLead(next)
      }
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

  // -------------------------------------------------------------------------
  // Durable runs: the campaign lives on the server, the tab only watches it
  // -------------------------------------------------------------------------

  /**
   * Fetch one lead's email body on demand.
   *
   * Bodies are deliberately kept out of the event log and out of snapshots (a
   * few hundred leads is megabytes of markup), so the page asks for the one it
   * is about to show.
   */
  const loadLeadPreview = useCallback(async (index: number) => {
    const id = runIdRef.current
    if (!id || index < 0) return
    if (previewFetchRef.current.has(index)) return
    previewFetchRef.current.add(index)
    try {
      const res = await api.campaignRunLeadHtml(id, index)
      if (!res.html) return
      setLeads((prev) => {
        const next = prev.map((l, i) => (i === index ? { ...l, _preview_html: res.html } : l))
        leadsRef.current = next
        return next
      })
      setLivePreview((p) => (p && p.rowIndex === index && !p.html ? { ...p, html: res.html } : p))
    } catch {
      previewFetchRef.current.delete(index)
    }
  }, [])

  /** Fold one pipeline event into local state. Shared by live and replayed feeds. */
  const applyEvent = (evt: any) => {
    if (!evt || typeof evt !== 'object') return
    switch (evt.type) {
      case 'log':
        setLogs((prev) => [...prev, evt.message || String(evt)])
        break
      case 'run_meta':
        if (evt.sender_email) setRunSender(evt.sender_email)
        break
      case 'run_status':
        setStatus(runStatusToLocal(evt.status))
        break
      case 'status_update':
        setLeadStatus(evt.row_index, evt.status)
        setCurrentIndex(evt.row_index)
        break
      case 'lead_resolved':
        setLeads((prev) => {
          const next = prev.map((l, i) =>
            i === evt.row_index
              ? { ...l, website: evt.website || l.website, company: evt.company || l.company }
              : l,
          )
          leadsRef.current = next
          return next
        })
        break
      case 'stage':
        pushStage(evt.row_index, {
          stage: evt.stage,
          label: evt.label,
          state: evt.state || 'active',
        })
        setCurrentIndex(evt.row_index)
        break
      case 'preview_html': {
        // A replayed preview carries no HTML (the snapshot holds the body), so
        // fall back to whatever the lead already has rather than blanking it.
        const existing = leadsRef.current[evt.row_index] || {}
        const html = evt.html || existing._preview_html || ''
        if (!html && evt.html_in_snapshot) void loadLeadPreview(evt.row_index)
        const preview: LivePreview = {
          rowIndex: evt.row_index,
          html,
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
        setLeads((prev) => {
          const next = prev.map((l, i) =>
            i === evt.row_index
              ? {
                  ...l,
                  _preview_html: preview.html,
                  _subject: preview.subject,
                  _product_sheet: preview.productSheet,
                  _product_count: preview.productCount,
                }
              : l,
          )
          leadsRef.current = next
          return next
        })
        break
      }
      case 'done':
        setStatus('done')
        break
      default:
        break
    }
  }

  /**
   * Tail a run's event feed, reconnecting on its own.
   *
   * The cursor is what makes the run tab-independent: every reconnect asks for
   * the events after the last one seen, so nothing is replayed twice and
   * nothing is missed while the tab was closed, asleep or offline.
   */
  const followRun = useCallback(async (id: string, fromCursor: number) => {
    streamAbortRef.current?.abort()
    const ctrl = new AbortController()
    streamAbortRef.current = ctrl
    cursorRef.current = fromCursor
    const token = localStorage.getItem('token')

    for (let attempt = 0; !ctrl.signal.aborted; attempt += 1) {
      let sawEvents = false
      try {
        const res = await fetch(
          `${getApiBase()}/api/campaign/runs/${id}/stream?cursor=${cursorRef.current}`,
          {
            headers: { ...(token ? { Authorization: `Bearer ${token}` } : {}) },
            signal: ctrl.signal,
          },
        )
        if (!res.ok || !res.body) throw new Error(await res.text())

        const reader = res.body.getReader()
        const decoder = new TextDecoder()
        let buffer = ''
        let finished = false

        while (!finished) {
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
            sawEvents = true
            if (typeof evt._cursor === 'number') cursorRef.current = evt._cursor
            if (evt.type === 'run_closed') {
              setStatus(runStatusToLocal(evt.status))
              finished = true
              break
            }
            applyEvent(evt)
          }
        }
        if (finished) return
      } catch (e: any) {
        if (ctrl.signal.aborted || e?.name === 'AbortError') return
        setLogs((prev) => [
          ...prev,
          `Live feed dropped (${e?.message || e}) — the run keeps going, reconnecting…`,
        ])
      }
      if (ctrl.signal.aborted) return
      const backoff = sawEvents ? 500 : Math.min(10000, 1000 * (attempt + 1))
      await new Promise((r) => setTimeout(r, backoff))
    }
  }, [])

  /** Load a run's full state, then follow it if it is still going. */
  const attachRun = useCallback(
    async (id: string) => {
      setAttaching(true)
      try {
        const snap = await api.campaignRun(id)
        const runLeads = snap.leads.map((l) => ({ ...l }))
        leadsRef.current = runLeads
        setLeads(runLeads)
        setLogs(snap.logs || [])
        setFileName(snap.run.file_name || '')
        setRunSender(snap.run.sender_email || '')
        setRunId(snap.run.id)
        runIdRef.current = snap.run.id
        previewFetchRef.current = new Set()
        setRunRecord(snap.run)
        setStatus(runStatusToLocal(snap.run.status))
        setAutosend(true)
        setFinishedRun(null)

        const opts = snap.run.options || {}
        if (opts.template) setTemplate(opts.template)
        if (typeof opts.delay === 'number') setDelay(opts.delay)
        if (opts.recipient_override) setRecipientOverride(opts.recipient_override)
        if (typeof opts.attach_product_sheet === 'boolean') {
          setAttachProductSheet(opts.attach_product_sheet)
        }

        const stages: Record<number, StageEvent[]> = {}
        runLeads.forEach((l, i) => {
          if (Array.isArray(l._stages) && l._stages.length) stages[i] = l._stages
        })
        setStagesByLead(stages)

        let previewIdx = -1
        for (let i = runLeads.length - 1; i >= 0; i -= 1) {
          if (runLeads[i]._preview_html) {
            previewIdx = i
            break
          }
        }
        if (previewIdx >= 0) {
          const l = runLeads[previewIdx]
          setLivePreview({
            rowIndex: previewIdx,
            html: l._preview_html,
            subject: l._subject || 'Starlight outreach',
            company: l.company,
            website: l.website,
            to: l._to,
            from: snap.run.sender_email,
            productCount: l._product_count,
            productSheet: l._product_sheet,
          })
          setCurrentIndex(previewIdx)
        }

        cursorRef.current = snap.cursor
        try {
          localStorage.setItem(RUN_STORAGE_KEY, snap.run.id)
        } catch {
          /* private mode — the server still knows about the run */
        }
        if (snap.run.status === 'running') void followRun(snap.run.id, snap.cursor)
        return snap.run
      } finally {
        setAttaching(false)
      }
    },
    [followRun],
  )

  /** Hand the campaign to the server and start watching it. */
  const startRun = useCallback(async () => {
    const opts = optsRef.current
    setStatus('running')
    setLogs([])
    setDraft(null)
    setLivePreview(null)
    setStagesByLead({})
    setFinishedRun(null)
    pauseRef.current = false
    stopReviewRef.current = false
    setLeads((prev) => {
      const next = prev.map((l) => ({ ...l, _status: '', _preview_html: '', _subject: '' }))
      leadsRef.current = next
      return next
    })

    try {
      const res = await api.createCampaignRun({
        leads: leadsRef.current,
        template: opts.template,
        delay: opts.delay,
        sender_email: opts.senderEmail,
        recipient_override: opts.recipientOverride,
        attach_product_sheet: opts.attachProductSheet,
        file_name: fileName,
      })
      setRunId(res.run_id)
      runIdRef.current = res.run_id
      previewFetchRef.current = new Set()
      cursorRef.current = 0
      try {
        localStorage.setItem(RUN_STORAGE_KEY, res.run_id)
      } catch {
        /* ignore */
      }
      setLogs((prev) => [
        ...prev,
        'Campaign started on the server — you can close this tab and it keeps sending.',
      ])
      void followRun(res.run_id, 0)
    } catch (e: any) {
      setLogs((prev) => [...prev, e?.message || 'Could not start the campaign'])
      setStatus('failed')
    }
  }, [followRun, fileName])

  const start = useCallback(async (): Promise<'live' | 'review'> => {
    if (optsRef.current.autosend) {
      await startRun()
      return 'live'
    }
    void startReview()
    return 'review'
  }, [startRun, startReview])

  const schedule = useCallback(async (at: Date) => {
    const opts = optsRef.current
    await api.createCampaignRun({
      leads: leadsRef.current.map((l) => ({ ...l, _status: '', _preview_html: '', _subject: '' })),
      template: opts.template,
      delay: opts.delay,
      sender_email: opts.senderEmail,
      recipient_override: opts.recipientOverride,
      attach_product_sheet: opts.attachProductSheet,
      file_name: fileName,
      scheduled_at: at.toISOString(),
    })
    clearLeads()
  }, [fileName, clearLeads])

  const resume = useCallback(async () => {
    pauseRef.current = false
    stopReviewRef.current = false
    const id = runIdRef.current
    if (id) {
      // The server owns the queue, so resuming is just asking it to carry on
      // from the first lead that was never sent.
      setStatus('running')
      try {
        const res = await api.resumeCampaignRun(id)
        if (res.message) setLogs((prev) => [...prev, res.message as string])
        setStatus(runStatusToLocal(res.status))
        if (res.status === 'running') void followRun(id, cursorRef.current)
      } catch (e: any) {
        setLogs((prev) => [...prev, e?.message || 'Could not resume the campaign'])
        setStatus('paused')
      }
      return
    }
    setStatus('reviewing')
    const jobs = remainingGenerateJobs()
    await runGenerateQueue(jobs)
    reviewSettled()
  }, [followRun, generateAt])

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

  // A tab that opens (or reopens) rejoins whatever the server is still doing.
  useEffect(() => {
    let cancelled = false
    void (async () => {
      try {
        const { run } = await api.activeCampaignRun()
        if (cancelled) return
        if (run) {
          await attachRun(run.id)
          return
        }
        let stored: string | null = null
        try {
          stored = localStorage.getItem(RUN_STORAGE_KEY)
        } catch {
          stored = null
        }
        if (!stored) return
        const snap = await api.campaignRun(stored).catch(() => null)
        if (cancelled) return
        if (!snap) {
          try {
            localStorage.removeItem(RUN_STORAGE_KEY)
          } catch {
            /* ignore */
          }
          return
        }
        // It finished while this tab was gone — surface it, don't take over.
        setFinishedRun(snap.run)
      } catch {
        /* no active run, or the API is unreachable — setup still works */
      }
    })()
    return () => {
      cancelled = true
    }
  }, [attachRun])

  // Stop tailing when the provider unmounts; the run itself carries on.
  useEffect(() => () => streamAbortRef.current?.abort(), [])

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
    runId,
    runRecord,
    attaching,
    finishedRun,
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
    schedule,
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
    attachRun,
    dismissFinishedRun,
    loadLeadPreview,
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

export { leadState, isFailedLead, firstPreviewIndex }
