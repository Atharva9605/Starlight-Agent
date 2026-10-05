import { useEffect, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { Dropzone } from '../../components/Dropzone'
import { TEMPLATES, useCampaign } from '../../campaign/CampaignContext'
import { api } from '../../api/client'

type TemplateOption = { value: string; label: string; blurb: string; swatch: string[] }

const SWATCHES: Record<string, string[]> = {
  'email_template.html': ['#2563eb', '#06b6d4'],
  'email_template_minimalist.html': ['#0f172a', '#64748b'],
  'email_template_bold.html': ['#f59e0b', '#e11d48'],
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
    swatch: SWATCHES[t.name] || (t.is_custom ? ['#0d9488', '#14b8a6'] : ['#2563eb', '#06b6d4']),
  }
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

  return (
    <div>
      <div className="page-hero">
        <div>
          <h1>Campaigns</h1>
          <p>Upload leads, choose a look, then generate every email at once — review, send now, or bulk send.</p>
        </div>
        <div className="row">
          {inFlight ? (
            <button
              className="btn"
              type="button"
              onClick={() =>
                nav(status === 'running' || (status === 'paused' && autosend) ? '/campaigns/live' : '/campaigns/review')
              }
            >
              {status === 'running' || (status === 'paused' && autosend)
                ? 'View Live progress Logs'
                : 'Review emails'}
            </button>
          ) : (
            <button
              className="btn"
              type="button"
              disabled={!leads.length || launching || (scheduleLater && !scheduleValid)}
              onClick={launch}
            >
              {launching
                ? scheduleLater
                  ? 'Scheduling…'
                  : 'Starting…'
                : scheduleLater
                  ? 'Schedule campaign'
                  : 'Start campaign'}
            </button>
          )}
        </div>
      </div>

      {finishedRun ? (
        <div className="panel stack">
          <strong style={{ fontFamily: 'var(--display)' }}>
            Your last campaign finished while you were away
          </strong>
          <p className="muted" style={{ margin: 0 }}>
            {finishedRun.counts.sent} sent · {finishedRun.counts.failed} failed
            {finishedRun.counts.pending ? ` · ${finishedRun.counts.pending} never sent` : ''}
            {finishedRun.file_name ? ` · ${finishedRun.file_name}` : ''}
          </p>
          <div className="row" style={{ gap: 8 }}>
            <Link to={`/campaigns/runs/${finishedRun.id}`} className="btn">
              See the whole run
            </Link>
            <button className="btn secondary" type="button" onClick={dismissFinishedRun}>
              Dismiss
            </button>
          </div>
        </div>
      ) : null}

      <div className="setup-grid">
        <div className="stack">
          <div className="panel stack">
            <strong style={{ fontFamily: 'var(--display)' }}>1 · Lead list</strong>
            <Dropzone
              accept=".xlsx,.xls,.csv"
              busy={busy}
              title="Drop Excel / CSV here"
              hint="Two columns: company + website. Website used when present; otherwise OpenSERP finds it from company."
              onFiles={onFiles}
            />
            {error ? <div className="alert danger">{error}</div> : null}
            {leads.length ? (
              <div className="row" style={{ justifyContent: 'space-between' }}>
                <span className="muted">
                  <strong>{leads.length}</strong> leads · {fileName}
                  {leads.some((l) => !l.website && (l.company || l.name)) ? (
                    <> · <span className="pill discover">OpenSERP for name-only rows</span></>
                  ) : null}
                </span>
                <button className="btn secondary" type="button" onClick={clearLeads}>
                  Clear
                </button>
              </div>
            ) : null}
          </div>

          {leads.length ? (
            <div className="panel stack">
              <strong style={{ fontFamily: 'var(--display)' }}>Leads</strong>
              <div className="table-wrap" style={{ maxHeight: 280 }}>
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>#</th>
                      <th>Company</th>
                      <th>Website</th>
                      <th>Path</th>
                      <th />
                    </tr>
                  </thead>
                  <tbody>
                    {leads.map((l, i) => {
                      const viaSerp = !l.website && Boolean(l.company || l.name)
                      return (
                      <tr key={i}>
                        <td className="muted">{i + 1}</td>
                        <td style={{ fontWeight: 600 }}>{l.company || l.name || <span className="muted">—</span>}</td>
                        <td>
                          {l.website || (
                            <span className="muted">{viaSerp ? 'will look up' : '—'}</span>
                          )}
                        </td>
                        <td>
                          {viaSerp ? (
                            <span className="pill discover">OpenSERP</span>
                          ) : l.website ? (
                            <span className="pill">Website</span>
                          ) : (
                            <span className="muted">—</span>
                          )}
                        </td>
                        <td>
                          <button className="icon-btn" type="button" onClick={() => removeLead(i)}>
                            ✕
                          </button>
                        </td>
                      </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          ) : null}
        </div>

        <div className="stack">
          <div className="panel stack">
            <strong style={{ fontFamily: 'var(--display)' }}>2 · Template</strong>
            {templateOptions.map((t) => (
              <button
                key={t.value}
                type="button"
                className={`template-card${template === t.value ? ' selected' : ''}`}
                onClick={() => setTemplate(t.value)}
              >
                <span
                  className="template-swatch"
                  style={{ background: `linear-gradient(135deg, ${t.swatch[0]}, ${t.swatch[1]})` }}
                />
                <span className="template-copy">
                  <strong>{t.label}</strong>
                  <span className="muted">{t.blurb}</span>
                </span>
                <span className="template-check">{template === t.value ? '✓' : ''}</span>
              </button>
            ))}
            <p className="muted" style={{ margin: 0, fontSize: 13 }}>
              Need a new look? Create one under Admin → Create with AI.
            </p>
          </div>

          <div className="panel stack">
            <strong style={{ fontFamily: 'var(--display)' }}>3 · Send options</strong>

            <label className="toggle-row">
              <input
                type="checkbox"
                checked={autosend}
                onChange={(e) => {
                  setAutosend(e.target.checked)
                  if (!e.target.checked) toggleSchedule(false)
                }}
              />
              <span>
                <strong>Autosend</strong>
                <span className="muted" style={{ display: 'block', fontSize: 13 }}>
                  Skip review — scrape, write, and send every lead automatically on Live progress Logs.
                </span>
              </span>
            </label>

            <label className="toggle-row">
              <input
                type="checkbox"
                checked={scheduleLater}
                onChange={(e) => toggleSchedule(e.target.checked)}
              />
              <span>
                <strong>Schedule for later</strong>
                <span className="muted" style={{ display: 'block', fontSize: 13 }}>
                  Pick a date and time — the server starts the campaign then and autosends every lead.
                  You can close the browser.
                </span>
              </span>
            </label>

            {scheduleLater ? (
              <label className="field">
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
              </label>
            ) : null}

            {scheduleError ? <div className="alert danger" style={{ margin: 0 }}>{scheduleError}</div> : null}

            <label className="toggle-row">
              <input
                type="checkbox"
                checked={attachProductSheet}
                onChange={(e) => setAttachProductSheet(e.target.checked)}
              />
              <span>
                <strong>Attach product sheet PDF</strong>
                <span className="muted" style={{ display: 'block', fontSize: 13 }}>
                  Optional branded Starlight PDF of catalogue products suggested in the email.
                </span>
              </span>
            </label>

            {!autosend ? (
              <div className="alert warn" style={{ margin: 0 }}>
                All emails are generated together. Open the review board to send now, add to bulk send, or discard.
              </div>
            ) : scheduleLater ? (
              <div className="alert" style={{ margin: 0, borderColor: '#bfdbfe', background: '#eff6ff', color: '#1e3a8a' }}>
                Scheduled campaigns show under <strong>Campaign Runs</strong> — start them early or cancel from there.
              </div>
            ) : (
              <div className="alert" style={{ margin: 0, borderColor: '#bfdbfe', background: '#eff6ff', color: '#1e3a8a' }}>
                Autosend runs on <strong>/campaigns/live</strong> with live preview and per-lead stage timeline.
              </div>
            )}

            <label className="field">
              <span>Test inbox (optional)</span>
              <input
                className="input"
                placeholder="you@starlightlinearled.com"
                value={recipientOverride}
                onChange={(e) => setRecipientOverride(e.target.value)}
              />
            </label>

            <label className="field">
              <span>{autosend ? 'Pause between sends' : 'Pause between bulk sends'}</span>
              <div className="row" style={{ gap: '0.75rem' }}>
                <input
                  type="range"
                  min={1}
                  max={15}
                  value={delay}
                  onChange={(e) => setDelay(Number(e.target.value))}
                  style={{ flex: 1 }}
                />
                <span className="pill">{delay}s</span>
              </div>
            </label>
          </div>

          {leads.length ? (
            <div className="setup-actions">
              <span className="muted" style={{ marginRight: 'auto', fontSize: 13 }}>
                {leads.length} lead{leads.length === 1 ? '' : 's'} ready
              </span>
              <button
                className="btn"
                type="button"
                disabled={!leads.length || launching || inFlight || (scheduleLater && !scheduleValid)}
                onClick={launch}
              >
                {launching
                  ? scheduleLater
                    ? 'Scheduling…'
                    : 'Starting…'
                  : scheduleLater
                    ? `Schedule ${leads.length} for ${formatScheduled(scheduleAt)}`
                    : autosend
                      ? `Autosend ${leads.length}`
                      : `Generate & review ${leads.length}`}
              </button>
            </div>
          ) : null}
        </div>
      </div>
    </div>
  )
}
