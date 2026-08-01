import { useQuery } from '@tanstack/react-query'
import { useEffect, useRef, useState } from 'react'
import { api, type CatalogueFileResult } from '../api/client'
import { Link } from 'react-router-dom'
import { Dropzone } from '../components/Dropzone'

const POLL_MS = 2000

/** Sales-facing catalogues page — no JSON / RAG dump. */
export function CataloguesPage() {
  const kb = useQuery({ queryKey: ['kb'], queryFn: api.kbStatus })
  const library = useQuery({ queryKey: ['library-catalogues'], queryFn: api.listCatalogues })
  const [msg, setMsg] = useState('')
  const [error, setError] = useState('')
  const [fileResults, setFileResults] = useState<CatalogueFileResult[]>([])
  const [busy, setBusy] = useState(false)
  const [progress, setProgress] = useState(0)
  const [copiedId, setCopiedId] = useState('')
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
      await Promise.all([kb.refetch(), library.refetch()])
    } catch (e: any) {
      setError(e.message || 'Upload failed')
      setFileResults(e.results || [])
      setMsg('')
      await Promise.all([kb.refetch(), library.refetch()])
    } finally {
      setBusy(false)
    }
  }

  const catalogues = library.data?.catalogues || []
  const legacyNames = kb.data?.catalogues || []

  const copyShare = async (url: string, id: string) => {
    try {
      await navigator.clipboard.writeText(url)
      setCopiedId(id)
      setTimeout(() => setCopiedId(''), 2000)
    } catch {
      setError('Could not copy link')
    }
  }

  return (
    <div className="studio-screen">
      <div className="page-hero">
        <div>
          <h1>Catalogues</h1>
          <p>Upload Starlight product PDFs that ground every outbound email — and power your digital catalogue.</p>
        </div>
        <div className="row">
          <Link to="/admin/rag" className="btn secondary">RAG Lab</Link>
          <button
            className="btn danger"
            type="button"
            disabled={busy || (!catalogues.length && !legacyNames.length)}
            onClick={async () => {
              await api.clearKb()
              setFileResults([])
              setError('')
              setMsg('Cleared')
              kb.refetch()
              library.refetch()
            }}
          >
            Clear all
          </button>
        </div>
      </div>

      <div className="stat-row">
        <div className="stat blue"><div className="label">Chunks</div><div className="value">{kb.data?.chunk_count ?? '—'}</div></div>
        <div className="stat cyan"><div className="label">Catalogues</div><div className="value">{catalogues.length || legacyNames.length}</div></div>
        <div className="stat amber"><div className="label">Status</div><div className="value" style={{ fontSize: '1rem', marginTop: 8 }}>{busy ? 'Working…' : msg || 'Ready'}</div></div>
      </div>

      <div className="setup-grid">
        <div className="panel tint-cyan stack">
          <strong style={{ fontFamily: 'var(--display)' }}>Upload</strong>
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
        </div>

        <div className="panel stack">
          <strong style={{ fontFamily: 'var(--display)' }}>Digital library</strong>
          {!catalogues.length && !legacyNames.length && !fileResults.length ? (
            <div className="empty-state" style={{ padding: '1.5rem 0.5rem' }}>
              <strong>No catalogues yet</strong>
              <p className="muted" style={{ margin: '0.4rem 0 0' }}>Upload a PDF to get started.</p>
            </div>
          ) : catalogues.length ? (
            <div className="stack" style={{ gap: 8 }}>
              {catalogues.map((c) => (
                <div key={c.id} className="list-row" style={{ alignItems: 'flex-start' }}>
                  <div style={{ minWidth: 0, flex: 1 }}>
                    <div style={{ fontWeight: 700 }}>{c.name}</div>
                    <div className="muted" style={{ fontSize: 12 }}>
                      {c.product_count ?? 0} products
                      {c.page_count ? ` · ${c.page_count} pages` : ''}
                    </div>
                    {c.share_url ? (
                      <div className="muted" style={{ fontSize: 11, marginTop: 4, wordBreak: 'break-all' }}>
                        {c.share_url}
                      </div>
                    ) : null}
                  </div>
                  <div className="row" style={{ gap: 6, flexShrink: 0 }}>
                    {c.share_url ? (
                      <>
                        <a className="btn secondary" href={c.share_url} target="_blank" rel="noreferrer">
                          Open
                        </a>
                        <button
                          type="button"
                          className="btn secondary"
                          onClick={() => copyShare(c.share_url!, c.id)}
                        >
                          {copiedId === c.id ? 'Copied' : 'Copy link'}
                        </button>
                      </>
                    ) : (
                      <span className="pill ok">Indexed</span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="stack" style={{ gap: 6 }}>
              {legacyNames.map((name: string) => (
                <div key={name} className="list-row">
                  <span style={{ fontWeight: 700 }}>{name}</span>
                  <span className="pill ok">Indexed</span>
                </div>
              ))}
            </div>
          )}
          <p className="muted" style={{ margin: 0, fontSize: 13 }}>
            Share links open the public digital catalogue. Re-upload a PDF after deploy to seed the library.
          </p>
        </div>
      </div>
    </div>
  )
}
