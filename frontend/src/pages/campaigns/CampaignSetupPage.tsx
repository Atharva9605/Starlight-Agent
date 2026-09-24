import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Dropzone } from '../../components/Dropzone'
import { TEMPLATES, useCampaign } from '../../campaign/CampaignContext'
import { api } from '../../api/client'

type TemplateOption = { value: string; label: string; blurb: string; swatch: string[] }

const SWATCHES: Record<string, string[]> = {
  'email_template.html': ['#2563eb', '#06b6d4'],
  'email_template_minimalist.html': ['#0f172a', '#64748b'],
  'email_template_bold.html': ['#f59e0b', '#e11d48'],
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
    status,
  } = useCampaign()

  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [launching, setLaunching] = useState(false)
  const [templateOptions, setTemplateOptions] = useState<TemplateOption[]>(TEMPLATES)

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

  const launch = async () => {
    setLaunching(true)
    try {
      await start()
      nav('/campaigns/live')
    } finally {
      setLaunching(false)
    }
  }

  const inFlight = status === 'running' || status === 'reviewing'

  return (
    <div>
      <div className="page-hero">
        <div>
          <h1>Campaigns</h1>
          <p>Upload leads, choose a look, then start — review and send on the live progress page.</p>
        </div>
        <div className="row">
          {inFlight ? (
            <button className="btn" type="button" onClick={() => nav('/campaigns/live')}>
              View Live progress Logs
            </button>
          ) : (
            <button
              className="btn"
              type="button"
              disabled={!leads.length || launching}
              onClick={launch}
            >
              {launching ? 'Starting…' : 'Start campaign'}
            </button>
          )}
        </div>
      </div>

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
                onChange={(e) => setAutosend(e.target.checked)}
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
                You'll review and send each email on <strong>Live progress Logs</strong> after you hit Start campaign.
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

            {autosend ? (
              <label className="field">
                <span>Pause between sends</span>
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
            ) : null}
          </div>

          {leads.length ? (
            <div className="setup-actions">
              <span className="muted" style={{ marginRight: 'auto', fontSize: 13 }}>
                {leads.length} lead{leads.length === 1 ? '' : 's'} ready
              </span>
              <button
                className="btn"
                type="button"
                disabled={!leads.length || launching || inFlight}
                onClick={launch}
              >
                {launching
                  ? 'Starting…'
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
