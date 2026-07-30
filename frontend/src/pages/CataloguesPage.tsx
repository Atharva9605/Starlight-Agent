import { useQuery } from '@tanstack/react-query'
import { useEffect, useRef, useState } from 'react'
import { api, type CatalogueFileResult } from '../api/client'
import { Link } from 'react-router-dom'
import { Dropzone } from '../components/Dropzone'

const POLL_MS = 2000

/** Sales-facing catalogues page — no JSON / RAG dump. */
export function CataloguesPage() {
  const kb = useQuery({ queryKey: ['kb'], queryFn: api.kbStatus })
  const [msg, setMsg] = useState('')
  const [error, setError] = useState('')
  const [fileResults, setFileResults] = useState<CatalogueFileResult[]>([])
  const [busy, setBusy] = useState(false)
  const [progress, setProgress] = useState(0)
  const cancelled = useRef(false)

  useEffect(() => () => { cancelled.current = true }, [])

  const upload = async (files?: FileList | null) => {
    if (!files?.length) return
    setBusy(true)
    setError('')
    setFileResults([])
    setProgress(0)
    setMsg('Uploading…')
    try {
      const { job_id } = await api.uploadCatalogues(files)
      setMsg('Reading catalogue… scanned PDFs take a few minutes.')

      // Ingestion runs server-side; poll so a slow catalogue can't time the
      // request out, and so progress is visible while it works.
      for (;;) {
        await new Promise((r) => setTimeout(r, POLL_MS))
        if (cancelled.current) return
        const job = await api.catalogueJob(job_id)
        setProgress(job.progress || 0)
        setMsg(job.message || 'Working…')
        setFileResults(job.results || [])
        if (job.status !== 'running') {
          if (job.status === 'error') setError(job.message || 'Ingestion failed')
          break
        }
      }
      await kb.refetch()
    } catch (e: any) {
      setError(e.message || 'Upload failed')
      setFileResults(e.results || [])
      setMsg('')
      await kb.refetch()
    } finally {
      setBusy(false)
    }
  }

  const catalogues = kb.data?.catalogues || []

  return (
    <div>
      <div className="page-hero">
        <div>
          <h1>Catalogues</h1>
          <p>Upload Starlight product PDFs that ground every outbound email.</p>
        </div>
        <button
          className="btn danger"
          type="button"
          disabled={busy}
          onClick={async () => {
            await api.clearKb()
            setFileResults([])
            setError('')
            setMsg('Cleared')
            kb.refetch()
          }}
        >
          Clear all
        </button>
      </div>

      <div className="stat-row">
        <div className="stat blue"><div className="label">Chunks</div><div className="value">{kb.data?.chunk_count ?? '—'}</div></div>
        <div className="stat cyan"><div className="label">Catalogues</div><div className="value">{catalogues.length}</div></div>
        <div className="stat amber"><div className="label">Status</div><div className="value" style={{ fontSize: '1rem', marginTop: 8 }}>{busy ? 'Working…' : msg || 'Ready'}</div></div>
      </div>

      <div className="panel tint-cyan stack" style={{ marginBottom: '1rem' }}>
        <strong style={{ fontFamily: 'var(--display)' }}>Upload catalogues</strong>
        <Dropzone
          accept=".pdf,.txt,.docx"
          multiple
          busy={busy}
          title="Drop PDF catalogues here"
          hint="or click to choose files · PDF, TXT, DOCX"
          onFiles={upload}
        />

        {busy ? (
          <div className="stack" style={{ gap: 6 }}>
            <div className="progress">
              <div
                className="progress-fill animated"
                style={{ width: `${Math.max(3, Math.round(progress * 100))}%` }}
              />
            </div>
            <div className="muted" style={{ fontSize: 12 }}>
              {msg} · {Math.round(progress * 100)}%
            </div>
          </div>
        ) : null}

        {error ? (
          <div className="alert danger">
            <strong>Could not index that catalogue.</strong>
            <div style={{ marginTop: 4 }}>{error}</div>
          </div>
        ) : null}

        {fileResults.length ? (
          <div className="stack" style={{ gap: 6 }}>
            {fileResults.map((r) => (
              <div key={r.filename} className="list-row">
                <div style={{ minWidth: 0, flex: 1 }}>
                  <div style={{ fontWeight: 700 }}>{r.filename}</div>
                  <div className="muted" style={{ fontSize: 12 }}>{r.message}</div>
                </div>
                <span className={`pill ${r.success ? 'ok' : 'pink'}`}>
                  {r.success ? 'Indexed' : 'Failed'}
                </span>
              </div>
            ))}
          </div>
        ) : null}

        {!catalogues.length && !fileResults.length ? (
          <div className="muted">No catalogues yet — upload a PDF to get started.</div>
        ) : (
          <div className="stack">
            {catalogues.map((name: string) => (
              <div key={name} className="list-row">
                <span style={{ fontWeight: 700 }}>{name}</span>
                <span className="pill ok">Indexed</span>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="panel">
        <p className="muted" style={{ margin: 0 }}>
          Need to test retrieval quality? Open{' '}
          <Link to="/admin/rag" style={{ color: 'var(--blue)', fontWeight: 700 }}>Admin → RAG Lab</Link>.
        </p>
      </div>
    </div>
  )
}
