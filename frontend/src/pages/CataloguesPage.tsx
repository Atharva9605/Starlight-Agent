import { useQuery } from '@tanstack/react-query'
import { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import {
  BookOpenRegular,
  CheckmarkCircleFilled,
  CopyRegular,
  DatabaseRegular,
  DeleteRegular,
  DocumentPdfRegular,
  DocumentSearchRegular,
  DismissCircleFilled,
  LibraryRegular,
  OpenRegular,
  ArrowUploadRegular,
} from '@fluentui/react-icons'
import { api, type CatalogueFileResult } from '../api/client'
import { Dropzone } from '../components/Dropzone'
import { CardHeader, EmptyState, MessageBar, PageHeader, useConfirm, useToast } from '../components/ui'

const POLL_MS = 2000

/** Sales-facing catalogues page — no JSON / RAG dump. */
export function CataloguesPage() {
  const toast = useToast()
  const confirm = useConfirm()
  const kb = useQuery({ queryKey: ['kb'], queryFn: api.kbStatus })
  const library = useQuery({ queryKey: ['library-catalogues'], queryFn: api.listCatalogues })
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

      for (;;) {
        await new Promise((r) => setTimeout(r, POLL_MS))
        if (cancelled.current) return
        const job = await api.catalogueJob(job_id)
        setProgress(job.progress || 0)
        setMsg(job.message || 'Working…')
        setFileResults(job.results || [])
        if (job.status !== 'running') {
          if (job.status === 'error') setError(job.message || 'Ingestion failed')
          else toast.success('Catalogue indexed', job.message || undefined)
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
  const hasAny = catalogues.length > 0 || legacyNames.length > 0

  const copyShare = async (url: string) => {
    try {
      await navigator.clipboard.writeText(url)
      toast.success('Link copied', 'Paste it into an email or chat to share the catalogue.')
    } catch {
      toast.error("Couldn't copy the link")
    }
  }

  const clearAll = async () => {
    const ok = await confirm({
      title: 'Clear all catalogues?',
      body: 'Every indexed catalogue and its share link will be removed. Outbound emails will no longer be grounded until you upload again.',
      confirmLabel: 'Clear all',
      danger: true,
    })
    if (!ok) return
    try {
      await api.clearKb()
      setFileResults([])
      setError('')
      setMsg('')
      toast.info('Catalogues cleared')
      kb.refetch()
      library.refetch()
    } catch (e: any) {
      toast.error("Couldn't clear catalogues", e.message)
    }
  }

  return (
    <div>
      <PageHeader
        title="Catalogues"
        subtitle="Upload Starlight product PDFs that ground every outbound email — and power your digital catalogue."
        actions={
          <>
            <Link to="/admin/rag" className="btn secondary">
              <DocumentSearchRegular /> RAG lab
            </Link>
            <button className="btn danger-outline" type="button" disabled={busy || !hasAny} onClick={clearAll}>
              <DeleteRegular /> Clear all
            </button>
          </>
        }
      />

      <div className="kpi-grid">
        <div className="kpi">
          <span className="kpi-icon"><LibraryRegular /></span>
          <div className="kpi-copy">
            <div className="kpi-label">Catalogues</div>
            <div className="kpi-value">{catalogues.length || legacyNames.length}</div>
          </div>
        </div>
        <div className="kpi">
          <span className="kpi-icon teal"><DatabaseRegular /></span>
          <div className="kpi-copy">
            <div className="kpi-label">Indexed chunks</div>
            <div className="kpi-value">{kb.data?.chunk_count ?? '—'}</div>
          </div>
        </div>
        <div className="kpi">
          <span className={`kpi-icon ${busy ? '' : error ? 'danger' : 'success'}`}>
            {error ? <DismissCircleFilled /> : <CheckmarkCircleFilled />}
          </span>
          <div className="kpi-copy">
            <div className="kpi-label">Status</div>
            <div className="kpi-value sm truncate">{busy ? 'Indexing…' : error ? 'Needs attention' : 'Ready'}</div>
          </div>
        </div>
      </div>

      <div className="setup-grid">
        <section className="card">
          <CardHeader
            icon={<ArrowUploadRegular />}
            title="Upload catalogues"
            subtitle="PDF, TXT or DOCX. Scanned PDFs are read with OCR and can take a few minutes."
          />
          <div className="stack" style={{ marginTop: 16 }}>
            <Dropzone
              accept=".pdf,.txt,.docx"
              multiple
              busy={busy}
              title="Drop catalogue files here"
              hint="PDF, TXT, DOCX"
              onFiles={upload}
            />

            {busy ? (
              <div className="stack tight">
                <div className="row between text-sm">
                  <span className="truncate grow">{msg}</span>
                  <strong>{Math.round(progress * 100)}%</strong>
                </div>
                <div className="progress">
                  <div className="progress-fill" style={{ width: `${Math.max(3, Math.round(progress * 100))}%` }} />
                </div>
              </div>
            ) : null}

            {error ? (
              <MessageBar intent="error" title="Couldn't index that catalogue" onDismiss={() => setError('')}>
                {error}
              </MessageBar>
            ) : null}

            {fileResults.length ? (
              <div className="list">
                {fileResults.map((r) => (
                  <div key={r.filename} className="list-item">
                    <span className="cell-icon pdf"><DocumentPdfRegular /></span>
                    <div className="list-item-copy">
                      <span className="list-item-title">{r.filename}</span>
                      <span className="list-item-sub">{r.message}</span>
                    </div>
                    <span className={`badge ${r.success ? 'success' : 'danger'}`}>
                      {r.success ? 'Indexed' : 'Failed'}
                    </span>
                  </div>
                ))}
              </div>
            ) : null}
          </div>
        </section>

        <section className="card">
          <CardHeader
            icon={<BookOpenRegular />}
            title="Digital library"
            subtitle="Share links open a public, searchable version of each catalogue."
          />
          <div className="list" style={{ marginTop: 12 }}>
            {!hasAny && !fileResults.length ? (
              <EmptyState
                compact
                icon={<LibraryRegular />}
                title="No catalogues yet"
                description="Upload a product PDF to start grounding emails and build your digital library."
              />
            ) : catalogues.length ? (
              catalogues.map((c) => (
                <div key={c.id} className="list-item">
                  <span className="cell-icon pdf"><DocumentPdfRegular /></span>
                  <div className="list-item-copy">
                    <span className="list-item-title">{c.name}</span>
                    <span className="list-item-sub">
                      {c.product_count ?? 0} products
                      {c.page_count ? ` · ${c.page_count} pages` : ''}
                    </span>
                  </div>
                  {c.share_url ? (
                    <div className="row nowrap" style={{ gap: 4 }}>
                      <button
                        type="button"
                        className="btn subtle icon-only"
                        title="Copy share link"
                        aria-label={`Copy share link for ${c.name}`}
                        onClick={() => copyShare(c.share_url!)}
                      >
                        <CopyRegular />
                      </button>
                      <a
                        className="btn subtle icon-only"
                        href={c.share_url}
                        target="_blank"
                        rel="noreferrer"
                        title="Open digital catalogue"
                        aria-label={`Open ${c.name}`}
                      >
                        <OpenRegular />
                      </a>
                    </div>
                  ) : (
                    <span className="badge success">Indexed</span>
                  )}
                </div>
              ))
            ) : (
              legacyNames.map((name: string) => (
                <div key={name} className="list-item">
                  <span className="cell-icon pdf"><DocumentPdfRegular /></span>
                  <div className="list-item-copy">
                    <span className="list-item-title">{name}</span>
                    <span className="list-item-sub">Re-upload to add it to the digital library.</span>
                  </div>
                  <span className="badge success">Indexed</span>
                </div>
              ))
            )}
          </div>
        </section>
      </div>
    </div>
  )
}
