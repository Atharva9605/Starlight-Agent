import { useQuery } from '@tanstack/react-query'
import { useEffect, useMemo, useState, type ReactNode } from 'react'
import {
  ArrowUndoRegular,
  BookDatabaseRegular,
  ChatRegular,
  DocumentSearchRegular,
  GlobeSearchRegular,
  MailRegular,
  SaveRegular,
  SparkleRegular,
} from '@fluentui/react-icons'
import { api } from '../api/client'
import { EmptyState, MessageBar, PageHeader, Spinner, useConfirm, useToast } from '../components/ui'

const CATEGORY_ORDER = ['email', 'scraping', 'rag', 'knowledge_base', 'conversation']

const CATEGORY_LABELS: Record<string, string> = {
  email: 'Email',
  scraping: 'Scraping',
  rag: 'RAG',
  knowledge_base: 'Knowledge',
  conversation: 'Conversation',
}

const CATEGORY_ICONS: Record<string, ReactNode> = {
  email: <MailRegular />,
  scraping: <GlobeSearchRegular />,
  rag: <DocumentSearchRegular />,
  knowledge_base: <BookDatabaseRegular />,
  conversation: <ChatRegular />,
}

function categoryLabel(c: string) {
  return CATEGORY_LABELS[c] || c.replace(/_/g, ' ')
}

type PromptItem = {
  key: string
  label: string
  description?: string
  category: string
  content: string
}

export function PromptsPage() {
  const toast = useToast()
  const confirm = useConfirm()
  const q = useQuery({ queryKey: ['prompts'], queryFn: api.prompts })
  const [category, setCategory] = useState('email')
  const [selectedKey, setSelectedKey] = useState('')
  const [drafts, setDrafts] = useState<Record<string, string>>({})
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')

  const allPrompts: PromptItem[] = (q.data?.prompts || []) as PromptItem[]

  const categories = useMemo(() => {
    const found = new Set(allPrompts.map((p) => p.category))
    const ordered = CATEGORY_ORDER.filter((c) => found.has(c))
    for (const c of found) {
      if (!ordered.includes(c)) ordered.push(c)
    }
    return ordered.length ? ordered : CATEGORY_ORDER
  }, [allPrompts])

  const prompts = useMemo(
    () => allPrompts.filter((p) => p.category === category),
    [allPrompts, category],
  )

  useEffect(() => {
    if (!prompts.length) {
      setSelectedKey('')
      return
    }
    if (!prompts.some((p) => p.key === selectedKey)) {
      setSelectedKey(prompts[0].key)
    }
  }, [prompts, selectedKey])

  const selected = prompts.find((p) => p.key === selectedKey) || null
  const draftValue = selected ? drafts[selected.key] ?? selected.content : ''
  const dirty = selected ? draftValue !== selected.content : false
  const dirtyCount = allPrompts.filter((p) => (drafts[p.key] ?? p.content) !== p.content).length

  const save = async () => {
    if (!selected || !dirty) return
    setSaving(true)
    setError('')
    try {
      await api.updatePrompt(selected.key, draftValue)
      toast.success('Prompt saved', selected.label)
      await q.refetch()
      setDrafts((prev) => {
        const next = { ...prev }
        delete next[selected.key]
        return next
      })
    } catch (e: any) {
      setError(e.message || 'Save failed')
    } finally {
      setSaving(false)
    }
  }

  const revert = async () => {
    if (!selected) return
    const ok = await confirm({
      title: 'Discard changes?',
      body: `Your edits to “${selected.label}” will be lost.`,
      confirmLabel: 'Discard',
      danger: true,
    })
    if (!ok) return
    setDrafts((prev) => {
      const next = { ...prev }
      delete next[selected.key]
      return next
    })
  }

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 's') {
        e.preventDefault()
        void save()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  })

  const lines = draftValue ? draftValue.split('\n').length : 0

  return (
    <div>
      <PageHeader
        breadcrumb={[{ label: 'Admin' }, { label: 'Prompt studio' }]}
        title="Prompt studio"
        subtitle="Tune Starlight’s voice across scraping, RAG, email writing and replies."
        actions={dirtyCount ? <span className="badge warning lg">{dirtyCount} unsaved</span> : null}
      />

      <div className="tablist" role="tablist" aria-label="Prompt categories">
        {categories.map((c) => (
          <button
            key={c}
            type="button"
            role="tab"
            aria-selected={category === c}
            className={`tab${category === c ? ' active' : ''}`}
            onClick={() => setCategory(c)}
          >
            {CATEGORY_ICONS[c] || <SparkleRegular />}
            {categoryLabel(c)}
            <span className="tab-count">{allPrompts.filter((p) => p.category === c).length}</span>
          </button>
        ))}
      </div>

      {q.isLoading ? (
        <div className="card center-fill">
          <Spinner label="Loading prompts…" />
        </div>
      ) : q.isError ? (
        <MessageBar intent="error" title="Couldn't load prompts">{(q.error as Error).message}</MessageBar>
      ) : !prompts.length ? (
        <div className="card">
          <EmptyState
            icon={<SparkleRegular />}
            title={`No prompts in ${categoryLabel(category)}`}
            description="Pick another category above."
          />
        </div>
      ) : (
        <div className="studio">
          <nav className="card studio-nav" aria-label="Prompts">
            <div className="studio-nav-label">{categoryLabel(category)} prompts</div>
            <div className="vnav">
              {prompts.map((p) => {
                const isDirty = (drafts[p.key] ?? p.content) !== p.content
                return (
                  <button
                    key={p.key}
                    type="button"
                    className={`vnav-item${selectedKey === p.key ? ' active' : ''}`}
                    aria-current={selectedKey === p.key ? 'page' : undefined}
                    onClick={() => setSelectedKey(p.key)}
                  >
                    <span className="vnav-copy">
                      <span className="vnav-title">{p.label}</span>
                      <span className="vnav-sub mono">{p.key}</span>
                    </span>
                    {isDirty ? <span className="dirty-dot" title="Unsaved changes" /> : null}
                  </button>
                )
              })}
            </div>
          </nav>

          {selected ? (
            <section className="card studio-editor">
              <div className="card-header">
                <div className="card-header-copy">
                  <h2 className="card-title">{selected.label}</h2>
                  {selected.description ? <p className="card-subtitle">{selected.description}</p> : null}
                  <code className="inline" style={{ display: 'inline-block', marginTop: 6 }}>{selected.key}</code>
                </div>
                <div className="card-header-actions">
                  <button className="btn secondary" type="button" disabled={saving || !dirty} onClick={revert}>
                    <ArrowUndoRegular /> Discard
                  </button>
                  <button className="btn" type="button" disabled={saving || !dirty} onClick={save}>
                    {saving ? <Spinner size="sm" /> : <SaveRegular />}
                    {saving ? 'Saving…' : 'Save'}
                  </button>
                </div>
              </div>

              {error ? (
                <MessageBar intent="error" title="Save failed" onDismiss={() => setError('')}>
                  {error}
                </MessageBar>
              ) : null}

              {selected.key === 'draft_system' ? (
                <MessageBar intent="warning" title="Keep the JSON keys">
                  <span>
                    The output must keep <code className="inline">subject, preamble, opening_line, intro, feature_highlights, use_cases, cta</code>.{' '}
                    <code className="inline">opening_line</code> should greet the client from CLIENT DATA (Hey {'{client}'}, …).
                    Renaming keys empties campaign emails.
                  </span>
                </MessageBar>
              ) : null}

              <textarea
                className="code-editor"
                value={draftValue}
                spellCheck={false}
                aria-label={`${selected.label} prompt`}
                onChange={(e) => setDrafts({ ...drafts, [selected.key]: e.target.value })}
              />
              <div className="editor-status">
                <span>
                  {dirty ? 'Unsaved changes' : 'All changes saved'} · {lines} lines · {draftValue.length.toLocaleString()} characters
                </span>
                <span>
                  <kbd className="kbd">Ctrl</kbd> + <kbd className="kbd">S</kbd> to save
                </span>
              </div>
            </section>
          ) : null}
        </div>
      )}
    </div>
  )
}
