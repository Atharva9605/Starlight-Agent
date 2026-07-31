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
      const mode = await start()
      nav(mode === 'live' ? '/campaigns/live' : '/campaigns/review')
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
          <p>Upload leads, choose a look, then review each email — or turn on autosend.</p>
        </div>
        <div className="row">
          {inFlight ? (
            <button
              className="btn secondary"
              type="button"
              onClick={() => nav(autosend ? '/campaigns/live' : '/campaigns/review')}
            >
              Resume →
            </button>
          ) : null}
          <button
            className="btn"
            type="button"
            disabled={!leads.length || launching || inFlight}
            onClick={launch}
          >
            {launching
              ? 'Starting…'
              : autosend
                ? `Autosend ${leads.length || ''} leads`
                : `Generate & review ${leads.length || ''} leads`}
          </button>
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
              hint="Needs a website column · optional email column is used as the recipient"
              onFiles={onFiles}
            />
            {error ? <div className="alert danger">{error}</div> : null}
            {leads.length ? (
              <div className="row" style={{ justifyContent: 'space-between' }}>
                <span className="muted">
                  <strong>{leads.length}</strong> leads · {fileName}
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
                      <th>Website</th>
                      <th>Email</th>
                      <th>Company</th>
                      <th />
                    </tr>
                  </thead>
                  <tbody>
                    {leads.map((l, i) => (
                      <tr key={i}>
                        <td className="muted">{i + 1}</td>
                        <td style={{ fontWeight: 600 }}>{l.website || '—'}</td>
                        <td>
                          {l.email ? (
                            l.email
                          ) : (
                            <span className="muted">scrape</span>
                          )}
                        </td>
                        <td>{l.company || <span className="muted">auto</span>}</td>
                        <td>
                          <button className="icon-btn" type="button" onClick={() => removeLead(i)}>
                            ✕
                          </button>
                        </td>
                      </tr>
                    ))}
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
                  Skip review — scrape, write, and send every lead automatically.
                </span>
              </span>
            </label>

            {!autosend ? (
              <div className="alert warn" style={{ margin: 0 }}>
                You'll review each email full-screen and only send when you click Send.
              </div>
            ) : null}

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
