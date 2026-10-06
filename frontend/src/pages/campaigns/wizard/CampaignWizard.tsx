import { useEffect, useState, type ReactNode } from 'react'
import { Link, Navigate, Outlet, useLocation, useNavigate, useOutletContext } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import {
  ArrowLeftRegular,
  ArrowRightRegular,
  CheckmarkRegular,
  EyeRegular,
  PlugDisconnectedRegular,
} from '@fluentui/react-icons'
import { api } from '../../../api/client'
import { TEMPLATES, useCampaign } from '../../../campaign/CampaignContext'
import { MessageBar, PageHeader } from '../../../components/ui'

export type TemplateOption = { value: string; label: string; blurb: string; swatch: string[]; custom?: boolean }

const SWATCHES: Record<string, string[]> = {
  'email_template.html': ['#0F6CBD', '#2886DE'],
  'email_template_minimalist.html': ['#242424', '#707070'],
  'email_template_bold.html': ['#F7630C', '#C50F1F'],
}

function toOption(t: { name: string; label: string; is_custom?: boolean }): TemplateOption {
  return {
    value: t.name,
    label: t.label || t.name,
    blurb: t.is_custom ? 'Your saved design.' : TEMPLATES.find((x) => x.value === t.name)?.blurb || 'Starlight design.',
    swatch: SWATCHES[t.name] || (t.is_custom ? ['#038387', '#00B7C3'] : ['#0F6CBD', '#2886DE']),
    custom: t.is_custom,
  }
}

/** `<input type="datetime-local">` wants local wall-clock time without a zone. */
export function toLocalInput(d: Date): string {
  const pad = (n: number) => String(n).padStart(2, '0')
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`
}

export function formatScheduled(value: string): string {
  const d = new Date(value)
  if (Number.isNaN(d.getTime())) return ''
  return d.toLocaleString(undefined, { weekday: 'short', month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' })
}

export type SendMode = 'review' | 'auto' | 'schedule'

/** State shared by the wizard steps that the campaign context doesn't hold. */
export type WizardCtx = {
  templateOptions: TemplateOption[]
  scheduleAt: string
  setScheduleAt: (v: string) => void
  scheduleValid: boolean
  mode: SendMode
  setMode: (m: SendMode) => void
  noMailbox: boolean
}

export function useWizard() {
  return useOutletContext<WizardCtx>()
}

export const STEPS = [
  { path: '', label: 'Leads', hint: 'Who to email' },
  { path: 'design', label: 'Design', hint: 'How it looks' },
  { path: 'sending', label: 'Sending', hint: 'How and when' },
  { path: 'launch', label: 'Review & launch', hint: 'Check and start' },
] as const

const BASE = '/campaigns/new'

export function stepUrl(i: number) {
  const p = STEPS[i].path
  return p ? `${BASE}/${p}` : BASE
}

/** Bottom bar: Back / Next, or a custom primary action on the last step. */
export function WizardNav({ step, canNext = true, nextHint, primary }: { step: number; canNext?: boolean; nextHint?: ReactNode; primary?: ReactNode }) {
  const nav = useNavigate()
  return (
    <div className="wizard-nav">
      {step > 0 ? (
        <button type="button" className="btn secondary" onClick={() => nav(stepUrl(step - 1))}>
          <ArrowLeftRegular /> Back
        </button>
      ) : (
        <span />
      )}
      <div className="wizard-nav-right">
        {nextHint ? <span className="muted text-sm">{nextHint}</span> : null}
        {primary ?? (
          <button type="button" className="btn" disabled={!canNext} onClick={() => nav(stepUrl(step + 1))}>
            Next: {STEPS[step + 1].label} <ArrowRightRegular />
          </button>
        )}
      </div>
    </div>
  )
}

/** New campaign, split into Leads → Design → Sending → Review & launch. */
export function CampaignWizard() {
  const nav = useNavigate()
  const location = useLocation()
  const { leads, template, setTemplate, autosend, setAutosend, status, finishedRun, dismissFinishedRun } = useCampaign()
  const [templateOptions, setTemplateOptions] = useState<TemplateOption[]>(TEMPLATES)
  const [scheduleLater, setScheduleLater] = useState(false)
  const [scheduleAt, setScheduleAt] = useState('')
  const gmail = useQuery({ queryKey: ['gmail-status'], queryFn: api.gmailStatus })
  const noMailbox = Boolean(gmail.data && !gmail.data.connected && gmail.data.mode !== 'platform')

  useEffect(() => {
    let cancelled = false
    api
      .templates()
      .then((list) => {
        if (!cancelled && list.length) setTemplateOptions(list.map(toOption))
      })
      .catch(() => undefined)
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    if (templateOptions.length && !templateOptions.some((t) => t.value === template)) setTemplate(templateOptions[0].value)
  }, [templateOptions, template, setTemplate])

  const scheduleDate = scheduleAt ? new Date(scheduleAt) : null
  const scheduleValid = Boolean(scheduleDate && !Number.isNaN(scheduleDate.getTime()) && scheduleDate.getTime() > Date.now())
  const mode: SendMode = scheduleLater ? 'schedule' : autosend ? 'auto' : 'review'
  const setMode = (m: SendMode) => {
    setScheduleLater(m === 'schedule')
    setAutosend(m !== 'review')
    if (m === 'schedule' && !scheduleAt) {
      const inAnHour = new Date(Date.now() + 60 * 60 * 1000)
      inAnHour.setSeconds(0, 0)
      setScheduleAt(toLocalInput(inAnHour))
    }
  }

  const rel = location.pathname.replace(/\/+$/, '').slice(BASE.length).replace(/^\//, '')
  const step = Math.max(0, STEPS.findIndex((s) => s.path === rel))
  const inFlight = status === 'running' || status === 'reviewing' || status === 'paused'
  const goesLive = status === 'running' || (status === 'paused' && autosend)

  // Later steps need a lead list first.
  if (step > 0 && !leads.length) return <Navigate to={BASE} replace />

  const done = [leads.length > 0, Boolean(template), step > 2, false]
  const ctx: WizardCtx = { templateOptions, scheduleAt, setScheduleAt, scheduleValid, mode, setMode, noMailbox }

  return (
    <div>
      <PageHeader
        breadcrumb={[{ label: 'Campaigns', to: '/campaigns/runs' }, { label: 'New campaign' }]}
        title="New campaign"
        subtitle="Four short steps. Nothing is sent until you launch on the last one."
      />

      <ol className="wizard-steps" aria-label="Campaign steps">
        {STEPS.map((s, i) => {
          const reachable = i === 0 || leads.length > 0
          const state = i === step ? 'current' : done[i] && i < step ? 'done' : 'todo'
          return (
            <li key={s.label} className={`wizard-step ${state}`}>
              <button
                type="button"
                disabled={!reachable || inFlight}
                aria-current={i === step ? 'step' : undefined}
                onClick={() => nav(stepUrl(i))}
              >
                <span className="wizard-step-num">{state === 'done' ? <CheckmarkRegular /> : i + 1}</span>
                <span className="wizard-step-copy">
                  <strong>{s.label}</strong>
                  <span>{s.hint}</span>
                </span>
              </button>
            </li>
          )
        })}
      </ol>

      <div className="page-alerts">
        {inFlight ? (
          <MessageBar
            intent="info"
            title="A campaign is already in progress"
            actions={
              <button className="btn secondary sm" type="button" onClick={() => nav(goesLive ? '/campaigns/live' : '/campaigns/review')}>
                <EyeRegular /> Open it
              </button>
            }
          >
            Finish or stop it before starting another one.
          </MessageBar>
        ) : null}
        {noMailbox ? (
          <MessageBar
            intent="warning"
            title="Gmail isn't connected"
            actions={
              <Link to="/settings/gmail" className="btn secondary sm">
                <PlugDisconnectedRegular /> Connect Gmail
              </Link>
            }
          >
            You can prepare the campaign, but emails can't be sent until a mailbox is connected.
          </MessageBar>
        ) : null}
        {finishedRun && step === 0 ? (
          <MessageBar
            intent="success"
            title="Your last campaign finished while you were away"
            onDismiss={dismissFinishedRun}
            actions={
              <Link to={`/campaigns/runs/${finishedRun.id}`} className="btn secondary sm">
                See the run
              </Link>
            }
          >
            {finishedRun.counts.sent} sent · {finishedRun.counts.failed} failed
            {finishedRun.counts.pending ? ` · ${finishedRun.counts.pending} never sent` : ''}
            {finishedRun.file_name ? ` · ${finishedRun.file_name}` : ''}
          </MessageBar>
        ) : null}
      </div>

      <Outlet context={ctx} />
    </div>
  )
}
