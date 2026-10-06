import { useMemo, useState } from 'react'
import {
  DeleteRegular,
  DismissRegular,
  DocumentTableRegular,
  GlobeSearchRegular,
  MailRegular,
  SearchRegular,
} from '@fluentui/react-icons'
import { Dropzone } from '../../../components/Dropzone'
import { useCampaign } from '../../../campaign/CampaignContext'
import { CardHeader, MessageBar } from '../../../components/ui'
import { WizardNav } from './CampaignWizard'

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
        <div className="wizard-body">
          <section className="card">
            <CardHeader title="Upload your lead list" subtitle="An Excel or CSV file, one company per row." />
            <div className="stack" style={{ marginTop: 16 }}>
              <Dropzone accept=".xlsx,.xls,.csv" busy={busy} title="Drop your lead sheet here" hint="XLSX, XLS or CSV" onFiles={onFiles} />
              {error ? (
                <MessageBar intent="error" title="Couldn't read that file" onDismiss={() => setError('')}>
                  {error}
                </MessageBar>
              ) : null}
            </div>
          </section>
          <aside className="card wizard-aside">
            <CardHeader title="What the sheet needs" />
            <ul className="column-guide">
              <li>
                <code className="inline">website</code> or <code className="inline">company</code>
                <span>At least one. Rows with only a company name get their website looked up.</span>
              </li>
              <li>
                <code className="inline">email</code>
                <span>Who receives the email. Without it, an address from their website is used.</span>
              </li>
              <li>
                <code className="inline">contact_name</code>
                <span>Optional. Lets the email greet a person: “Dear Priya,”.</span>
              </li>
            </ul>
          </aside>
        </div>
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

      <WizardNav step={0} canNext={leads.length > 0} nextHint={leads.length ? `${leads.length} lead${leads.length === 1 ? '' : 's'}` : 'Upload a sheet to continue'} />
    </>
  )
}
