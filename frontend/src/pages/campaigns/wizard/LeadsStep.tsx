import { useMemo, useState } from 'react'
import {
  DeleteRegular,
  DismissRegular,
  DocumentTableRegular,
  GlobeSearchRegular,
  MailRegular,
  SearchRegular,
  ArrowDownloadRegular,
  PersonRegular,
} from '@fluentui/react-icons'
import { Dropzone } from '../../../components/Dropzone'
import { useCampaign } from '../../../campaign/CampaignContext'
import { MessageBar } from '../../../components/ui'
import { WizardNav } from './CampaignWizard'

const SAMPLE_COLUMNS = [
  { name: 'company', need: 'Required*' },
  { name: 'website', need: 'Required*' },
  { name: 'email', need: 'Recommended' },
  { name: 'contact_name', need: 'Optional' },
]

const SAMPLE_ROWS = [
  ['Studio Lotus', 'studiolotus.in', 'info@studiolotus.in', 'Ankur Choksi'],
  ['Morphogenesis', '', 'contact@morphogenesis.org', ''],
  ['Abin Design Studio', 'abindesignstudio.com', '', 'Abin Chaudhuri'],
]

/** A ready-to-fill CSV with the right column names. */
function downloadSample() {
  const csv = [SAMPLE_COLUMNS.map((c) => c.name), ...SAMPLE_ROWS].map((r) => r.join(',')).join('\r\n')
  const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv;charset=utf-8' }))
  const a = document.createElement('a')
  a.href = url
  a.download = 'starlight-leads-sample.csv'
  a.click()
  URL.revokeObjectURL(url)
}

function leadEmail(l: Record<string, any>): string {
  for (const [k, v] of Object.entries(l)) {
    if (['email', 'emails', 'e_mail', 'mail', 'email_address', 'contact_email'].includes(k.trim().toLowerCase())) {
      const t = String(v ?? '').split(',')[0].trim()
      if (t.includes('@')) return t
    }
  }
  return ''
}

/** Step 1 — upload the lead sheet and prune it. */
export function LeadsStep() {
  const { leads, fileName, uploadLeads, removeLead, clearLeads } = useCampaign()
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [search, setSearch] = useState('')

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

  const lookups = leads.filter((l) => !l.website && (l.company || l.name)).length
  const withEmail = leads.filter((l) => leadEmail(l)).length
  const rows = useMemo(() => {
    const term = search.trim().toLowerCase()
    return leads
      .map((l, i) => ({ l, i }))
      .filter(({ l }) => !term || `${l.company || ''} ${l.name || ''} ${l.website || ''} ${leadEmail(l)}`.toLowerCase().includes(term))
  }, [leads, search])

  return (
    <>
      {!leads.length ? (
        <section className="card upload-card">
          <Dropzone
            accept=".xlsx,.xls,.csv"
            busy={busy}
            title={busy ? 'Reading your sheet…' : 'Drop your lead sheet here'}
            hint="Excel or CSV, one company per row"
            onFiles={onFiles}
          />
          {error ? (
            <MessageBar intent="error" title="Couldn't read that file" onDismiss={() => setError('')}>
              {error}
            </MessageBar>
          ) : null}

          <div className="sheet-guide">
            <div className="sheet-guide-head">
              <div>
                <strong>Your sheet should look like this</strong>
                <span className="muted text-sm">* Either company or website is enough. Column names matter; order doesn't, and extra columns are ignored.</span>
              </div>
              <button type="button" className="btn secondary sm" onClick={downloadSample}>
                <ArrowDownloadRegular /> Download sample sheet
              </button>
            </div>
            <div className="sheet-sample" role="table" aria-label="Example lead sheet">
              <div className="sheet-row head" role="row">
                {SAMPLE_COLUMNS.map((c) => (
                  <span key={c.name} role="columnheader">
                    <code>{c.name}</code>
                    <em className={c.need === 'Optional' ? '' : 'need'}>{c.need}</em>
                  </span>
                ))}
              </div>
              {SAMPLE_ROWS.map((r, i) => (
                <div key={i} className="sheet-row" role="row">
                  {r.map((v, j) => (
                    <span key={j} role="cell" className={v ? '' : 'empty'}>{v || 'empty'}</span>
                  ))}
                </div>
              ))}
            </div>
            <ul className="sheet-notes">
              <li><GlobeSearchRegular /> No website? We look it up from the company name.</li>
              <li><MailRegular /> No email? We use the contact address on their website.</li>
              <li><PersonRegular /> A contact name makes the email open with “Dear Priya,”.</li>
            </ul>
          </div>
        </section>
      ) : (
        <section className="card flush">
          <div className="card-section">
            <div className="file-chip">
              <DocumentTableRegular className="file-chip-icon" />
              <div className="grow" style={{ minWidth: 0 }}>
                <strong className="truncate" style={{ display: 'block' }}>{fileName}</strong>
                <span className="muted text-sm">
                  {leads.length} lead{leads.length === 1 ? '' : 's'} · {withEmail} with an email address
                  {lookups ? ` · ${lookups} website lookup${lookups === 1 ? '' : 's'}` : ''}
                </span>
              </div>
              <button className="btn subtle" type="button" onClick={clearLeads}>
                <DeleteRegular /> Use another file
              </button>
            </div>
          </div>
          <div className="card-section row between" style={{ borderTop: '1px solid var(--stroke-2)' }}>
            <span className="muted text-sm">Remove anyone you don't want to email.</span>
            <div className="input-wrap" style={{ width: 'min(280px, 100%)' }}>
              <SearchRegular />
              <input className="input" type="search" placeholder="Search leads" aria-label="Search leads" value={search} onChange={(e) => setSearch(e.target.value)} />
            </div>
          </div>
          <div className="table-wrap" style={{ maxHeight: 480, borderTop: '1px solid var(--stroke-2)' }}>
            <table className="data-table">
              <thead>
                <tr>
                  <th className="num">#</th>
                  <th>Company</th>
                  <th className="hide-sm">Website</th>
                  <th className="hide-sm">Send to</th>
                  <th className="actions"><span className="sr-only">Remove</span></th>
                </tr>
              </thead>
              <tbody>
                {rows.map(({ l, i }) => {
                  const lookup = !l.website && Boolean(l.company || l.name)
                  const email = leadEmail(l)
                  return (
                    <tr key={i}>
                      <td className="num muted">{i + 1}</td>
                      <td><strong>{l.company || l.name || <span className="muted">—</span>}</strong></td>
                      <td className="hide-sm truncate" style={{ maxWidth: 240 }}>
                        {l.website ? (
                          l.website
                        ) : lookup ? (
                          <span className="badge teal"><GlobeSearchRegular /> Will look up</span>
                        ) : (
                          <span className="muted">—</span>
                        )}
                      </td>
                      <td className="hide-sm truncate" style={{ maxWidth: 240 }}>
                        {email ? (
                          email
                        ) : (
                          <span className="muted"><MailRegular /> From their website</span>
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
                {rows.length === 0 ? (
                  <tr>
                    <td colSpan={5} className="muted" style={{ textAlign: 'center' }}>No leads match “{search}”</td>
                  </tr>
                ) : null}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {leads.length ? <WizardNav step={0} nextHint={`${leads.length} lead${leads.length === 1 ? '' : 's'} ready`} /> : null}
    </>
  )
}
