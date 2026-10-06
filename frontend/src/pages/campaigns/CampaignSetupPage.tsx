import { useEffect, useState, type ReactNode } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import {
  CheckmarkCircleFilled,
  DeleteRegular,
  DismissRegular,
  DocumentTableRegular,
  GlobeSearchRegular,
  PlayRegular,
  CalendarClockRegular,
  SendRegular,
  WandRegular,
  EyeRegular,
} from '@fluentui/react-icons'
import { Dropzone } from '../../components/Dropzone'
import { TEMPLATES, useCampaign } from '../../campaign/CampaignContext'
import { api } from '../../api/client'
import { MessageBar, PageHeader, Spinner, Switch } from '../../components/ui'

type TemplateOption = { value: string; label: string; blurb: string; swatch: string[] }

const SWATCHES: Record<string, string[]> = {
  'email_template.html': ['#0F6CBD', '#2886DE'],
  'email_template_minimalist.html': ['#242424', '#707070'],
  'email_template_bold.html': ['#F7630C', '#C50F1F'],
}

/** `<input type="datetime-local">` wants local wall-clock time without a zone. */
function toLocalInput(d: Date): string {
  const pad = (n: number) => String(n).padStart(2, '0')
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`
}

function formatScheduled(value: string): string {
  const d = new Date(value)
  if (Number.isNaN(d.getTime())) return ''
  return d.toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })
}

function toOption(t: { name: string; label: string; is_custom?: boolean }): TemplateOption {
  return {
    value: t.name,
    label: t.label || t.name,
    blurb: t.is_custom ? 'Custom AI / saved design.' : TEMPLATES.find((x) => x.value === t.name)?.blurb || 'Org template.',
    swatch: SWATCHES[t.name] || (t.is_custom ? ['#038387', '#00B7C3'] : ['#0F6CBD', '#2886DE']),
  }
}

function Step({
  n,
  done,
  title,
  subtitle,
  actions,
  children,
}: {
  n: number
  done?: boolean
  title: string
  subtitle?: string
  actions?: ReactNode
  children: ReactNode
}) {
  return (
    <section className="card">
      <div className="card-header">
        <span className={`step-num${done ? ' done' : ''}`}>{done ? <CheckmarkCircleFilled /> : n}</span>
        <div className="card-header-copy">
          <h2 className="card-title">{title}</h2>
          {subtitle ? <p className="card-subtitle">{subtitle}</p> : null}
        </div>
        {actions ? <div className="card-header-actions">{actions}</div> : null}
      </div>
      <div className="stack" style={{ marginTop: 16 }}>{children}</div>
    </section>
  )
}

/** Page 1 — upload leads, pick template, launch (review or autosend). */
export function CampaignSetupPage() {
  const nav = useNavigate()
  const {
    leads,
    fileName,
    template,
    setTemplate,
    delay,
    setDelay,
    recipientOverride,
    setRecipientOverride,
    autosend,
    setAutosend,
    attachProductSheet,
    setAttachProductSheet,
    uploadLeads,
    removeLead,
    clearLeads,
    start,
    schedule,
    status,
    finishedRun,
    dismissFinishedRun,
  } = useCampaign()

  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [launching, setLaunching] = useState(false)
  const [templateOptions, setTemplateOptions] = useState<TemplateOption[]>(TEMPLATES)
  const [scheduleLater, setScheduleLater] = useState(false)
  const [scheduleAt, setScheduleAt] = useState('')
  const [scheduleError, setScheduleError] = useState('')

  const scheduleDate = scheduleAt ? new Date(scheduleAt) : null
  const scheduleValid = Boolean(scheduleDate && !Number.isNaN(scheduleDate.getTime()) && scheduleDate.getTime() > Date.now())

  useEffect(() => {
    let cancelled = false
    api
      .templates()
      .then((list) => {
        if (cancelled || !list.length) return
        const options = list.map(toOption)
        setTemplateOptions(options)
      })
      .catch(() => undefined)
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    if (!templateOptions.length) return
    if (!templateOptions.some((t) => t.value === template)) {
      setTemplate(templateOptions[0].value)
    }
  }, [templateOptions, template, setTemplate])

  const onFiles = async (files: FileList) => {
    setBusy(true)
    setError('')
    try {
      await uploadLeads(files[0])
    } catch (e: any) {
      setError(e.message || 'Could not read that file')
    } finally {
      setBusy(false)
    }
  }

  const toggleSchedule = (on: boolean) => {
    setScheduleLater(on)
    setScheduleError('')
    if (!on) return
    setAutosend(true)
    if (!scheduleAt) {
      const inAnHour = new Date(Date.now() + 60 * 60 * 1000)
      inAnHour.setSeconds(0, 0)
      setScheduleAt(toLocalInput(inAnHour))
    }
  }

  const launch = async () => {
    setLaunching(true)
    setScheduleError('')
    try {
      if (scheduleLater) {
        if (!scheduleDate || !scheduleValid) {
          setScheduleError('Pick a date and time in the future.')
          return
        }
        await schedule(scheduleDate)
        setScheduleLater(false)
        setScheduleAt('')
        nav('/campaigns/runs')
        return
      }
      const dest = await start()
      nav(dest === 'live' ? '/campaigns/live' : '/campaigns/review')
    } catch (e: any) {
      setScheduleError(e?.message || (scheduleLater ? 'Could not schedule the campaign' : 'Could not start the campaign'))
    } finally {
      setLaunching(false)
    }
  }

  const inFlight = status === 'running' || status === 'reviewing' || status === 'paused'
  const goesLive = status === 'running' || (status === 'paused' && autosend)
  const serpCount = leads.filter((l) => !l.website && (l.company || l.name)).length
  const canLaunch = leads.length > 0 && !launching && !inFlight && (!scheduleLater || scheduleValid)

  const launchLabel = launching
    ? scheduleLater
      ? 'Scheduling…'
      : 'Starting…'
    : scheduleLater
      ? `Schedule ${leads.length} email${leads.length === 1 ? '' : 's'}`
      : autosend
        ? `Autosend ${leads.length} email${leads.length === 1 ? '' : 's'}`
        : `Generate & review ${leads.length}`

  const launchIcon = launching ? <Spinner size="sm" /> : scheduleLater ? <CalendarClockRegular /> : autosend ? <SendRegular /> : <PlayRegular />

  return (
    <div>
      <PageHeader
        title="New campaign"
        subtitle="Upload leads, choose a design, then generate every email at once — review first or send automatically."
        actions={
          inFlight ? (
            <button className="btn" type="button" onClick={() => nav(goesLive ? '/campaigns/live' : '/campaigns/review')}>
              <EyeRegular /> {goesLive ? 'View live progress' : 'Review emails'}
            </button>
          ) : null
        }
      />

      <div className="page-alerts">
        {inFlight ? (
          <MessageBar
            intent="info"
            title="A campaign is already in progress"
            actions={
              <button className="btn secondary sm" type="button" onClick={() => nav(goesLive ? '/campaigns/live' : '/campaigns/review')}>
                Open it
              </button>
            }
          >
            Finish or stop it before starting another one.
          </MessageBar>
        ) : null}
        {finishedRun ? (
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

      <div className="setup-grid">
        <div className="setup-col">
          <Step
            n={1}
            done={leads.length > 0}
            title="Lead list"
            subtitle="Excel or CSV with company and website columns. Rows without a website are looked up automatically."
          >
            {leads.length ? (
              <div className="file-chip">
                <DocumentTableRegular className="file-chip-icon" />
                <div className="grow">
                  <strong className="truncate" style={{ display: 'block' }}>{fileName}</strong>
                  <span className="muted text-sm">
                    {leads.length} lead{leads.length === 1 ? '' : 's'}
                    {serpCount ? ` · ${serpCount} will be looked up by name` : ''}
                  </span>
                </div>
                <button className="btn subtle" type="button" onClick={clearLeads}>
                  <DeleteRegular /> Remove
                </button>
              </div>
            ) : (
              <Dropzone
                accept=".xlsx,.xls,.csv"
                busy={busy}
                title="Drop your lead sheet here"
                hint="XLSX, XLS or CSV"
                onFiles={onFiles}
              />
            )}
            {error ? <MessageBar intent="error" title="Couldn't read that file" onDismiss={() => setError('')}>{error}</MessageBar> : null}
          </Step>

          {leads.length ? (
            <section className="card flush">
              <div className="card-section row between">
                <strong>Leads</strong>
                <span className="badge">{leads.length}</span>
              </div>
              <div className="table-wrap" style={{ maxHeight: 360, borderTop: '1px solid var(--stroke-2)' }}>
                <table className="data-table">
                  <thead>
                    <tr>
                      <th className="num">#</th>
                      <th>Company</th>
                      <th className="hide-sm">Website</th>
                      <th>Source</th>
                      <th className="actions"><span className="sr-only">Remove</span></th>
                    </tr>
                  </thead>
                  <tbody>
                    {leads.map((l, i) => {
                      const viaSerp = !l.website && Boolean(l.company || l.name)
                      return (
                        <tr key={i}>
                          <td className="num muted">{i + 1}</td>
                          <td><strong>{l.company || l.name || <span className="muted">—</span>}</strong></td>
                          <td className="hide-sm truncate" style={{ maxWidth: 220 }}>
                            {l.website || <span className="muted">{viaSerp ? 'Will look up' : '—'}</span>}
                          </td>
                          <td>
                            {viaSerp ? (
                              <span className="badge teal"><GlobeSearchRegular /> Search</span>
                            ) : l.website ? (
                              <span className="badge">Website</span>
                            ) : (
                              <span className="muted">—</span>
                            )}
                          </td>
                          <td className="actions">
                            <button
                              className="btn subtle icon-only sm"
                              type="button"
                              aria-label={`Remove ${l.company || l.name || `lead ${i + 1}`}`}
                              title="Remove lead"
                              onClick={() => removeLead(i)}
                            >
                              <DismissRegular />
                            </button>
                          </td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            </section>
          ) : null}
        </div>

        <div className="setup-col">
          <Step
            n={2}
            done={Boolean(template)}
            title="Design"
            subtitle="The HTML layout every email is rendered with."
            actions={
              <Link to="/admin/templates/create" className="btn subtle sm">
                <WandRegular /> Create with AI
              </Link>
            }
          >
            <div className="stack tight" role="radiogroup" aria-label="Email design">
              {templateOptions.map((t) => (
                <button
                  key={t.value}
                  type="button"
                  role="radio"
                  aria-checked={template === t.value}
                  className={`choice-card${template === t.value ? ' selected' : ''}`}
                  onClick={() => setTemplate(t.value)}
                >
                  <span className="swatch" style={{ background: `linear-gradient(135deg, ${t.swatch[0]}, ${t.swatch[1]})` }} />
                  <span className="choice-card-copy">
                    <strong>{t.label}</strong>
                    <span>{t.blurb}</span>
                  </span>
                  {template === t.value ? <CheckmarkCircleFilled className="choice-card-check" /> : <span className="choice-card-ring" />}
                </button>
              ))}
            </div>
          </Step>

          <Step n={3} title="Send options" subtitle="Choose how and when emails go out.">
            <div>
              <Switch
                checked={autosend}
                onChange={(v) => {
                  setAutosend(v)
                  if (!v) toggleSchedule(false)
                }}
                label="Autosend"
                description="Skip review — research, write and send every lead automatically while you watch live progress."
              />
              <Switch
                checked={scheduleLater}
                onChange={toggleSchedule}
                label="Schedule for later"
                description="The server starts the campaign at the chosen time and autosends every lead. You can close the browser."
              />
              {scheduleLater ? (
                <label className="field" style={{ paddingBottom: 12 }}>
                  <span>Send at (your local time)</span>
                  <input
                    className="input"
                    type="datetime-local"
                    min={toLocalInput(new Date())}
                    value={scheduleAt}
                    onChange={(e) => {
                      setScheduleAt(e.target.value)
                      setScheduleError('')
                    }}
                  />
                  {scheduleValid ? <span className="field-hint">Starts {formatScheduled(scheduleAt)}</span> : null}
                </label>
              ) : null}
              <Switch
                checked={attachProductSheet}
                onChange={setAttachProductSheet}
                label="Attach product sheet PDF"
                description="A branded Starlight PDF of the catalogue products suggested in each email."
              />
            </div>

            {scheduleError ? <MessageBar intent="error" onDismiss={() => setScheduleError('')}>{scheduleError}</MessageBar> : null}

            <MessageBar intent="info">
              {!autosend
                ? 'Emails are generated together, then you send, bulk-send or discard each one from the review board.'
                : scheduleLater
                  ? 'Scheduled campaigns appear under Campaign runs — start them early or cancel from there.'
                  : 'Autosend opens the live view with a preview and stage timeline for each lead.'}
            </MessageBar>

            <hr className="divider" />

            <label className="field">
              <span>Test inbox</span>
              <input
                className="input"
                type="email"
                placeholder="you@starlightlinearled.com"
                value={recipientOverride}
                onChange={(e) => setRecipientOverride(e.target.value)}
              />
              <span className="field-hint">Optional. When set, every email goes here instead of to the lead.</span>
            </label>

            <div className="field">
              <span className="field-label">{autosend ? 'Pause between sends' : 'Pause between bulk sends'}</span>
              <div className="slider-row">
                <input
                  type="range"
                  min={1}
                  max={15}
                  value={delay}
                  aria-label="Seconds between sends"
                  onChange={(e) => setDelay(Number(e.target.value))}
                />
                <span className="slider-value">{delay}s</span>
              </div>
            </div>
          </Step>
        </div>
      </div>

      {leads.length && !inFlight ? (
        <div className="launch-bar" style={{ marginTop: 20 }}>
          <div className="launch-bar-copy">
            <strong>
              {leads.length} lead{leads.length === 1 ? '' : 's'} ready
            </strong>
            <span>
              {scheduleLater
                ? scheduleValid
                  ? `Will start ${formatScheduled(scheduleAt)}`
                  : 'Pick a time in the future'
                : autosend
                  ? 'Emails send automatically as they are written'
                  : 'You review every email before it is sent'}
            </span>
          </div>
          <button className="btn lg" type="button" disabled={!canLaunch} onClick={launch}>
            {launchIcon} {launchLabel}
          </button>
        </div>
      ) : null}
    </div>
  )
}
