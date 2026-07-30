import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Dropzone } from '../../components/Dropzone'
import { useCampaign } from '../../campaign/CampaignContext'

export function StepLeads() {
  const nav = useNavigate()
  const { leads, fileName, uploadLeads, removeLead, clearLeads } = useCampaign()
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

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

  const missingEmail = leads.filter((l) => !l.email && !l.emails).length

  return (
    <div className="step-body">
      <div className="panel stack">
        <div>
          <strong style={{ fontFamily: 'var(--display)' }}>Upload your lead list</strong>
          <p className="muted" style={{ margin: '0.25rem 0 0' }}>
            Excel or CSV with a <code>website</code> column. Optional: <code>company</code>, <code>email</code>.
          </p>
        </div>

        <Dropzone
          accept=".xlsx,.xls,.csv"
          busy={busy}
          title="Drop your file here"
          hint="or click to browse — .xlsx, .xls, .csv"
          onFiles={onFiles}
        />

        {error ? <div className="alert danger">{error}</div> : null}

        {leads.length ? (
          <div className="row" style={{ justifyContent: 'space-between' }}>
            <span className="muted">
              <strong>{leads.length}</strong> leads from {fileName || 'your file'}
              {missingEmail ? ` · ${missingEmail} without an email (we'll scrape one)` : ''}
            </span>
            <button className="btn secondary" onClick={clearLeads}>Clear list</button>
          </div>
        ) : null}
      </div>

      {leads.length ? (
        <div className="panel stack">
          <strong style={{ fontFamily: 'var(--display)' }}>Review leads</strong>
          <div className="table-wrap">
            <table className="data-table">
              <thead>
                <tr>
                  <th style={{ width: 44 }}>#</th>
                  <th>Website</th>
                  <th>Company</th>
                  <th>Email</th>
                  <th style={{ width: 60 }} />
                </tr>
              </thead>
              <tbody>
                {leads.map((l, i) => (
                  <tr key={i}>
                    <td className="muted">{i + 1}</td>
                    <td style={{ fontWeight: 600 }}>{l.website || '—'}</td>
                    <td>{l.company || <span className="muted">auto</span>}</td>
                    <td>{l.email || l.emails || <span className="muted">from site</span>}</td>
                    <td>
                      <button
                        className="icon-btn"
                        title="Remove lead"
                        onClick={() => removeLead(i)}
                      >
                        ✕
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      ) : (
        <div className="panel empty-state">
          <div className="empty-icon">📄</div>
          <strong>No leads yet</strong>
          <p className="muted">
            Your file stays in this browser session until you launch the campaign.
          </p>
        </div>
      )}

      <div className="step-actions">
        <span className="muted">{leads.length ? 'Looks good?' : 'Add a file to continue'}</span>
        <button
          className="btn"
          disabled={!leads.length}
          onClick={() => nav('/campaigns/new/design')}
        >
          Continue to design →
        </button>
      </div>
    </div>
  )
}
