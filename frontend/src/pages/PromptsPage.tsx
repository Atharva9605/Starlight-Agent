import { useQuery } from '@tanstack/react-query'
import { useEffect, useMemo, useState } from 'react'
import { api } from '../api/client'

const CATEGORY_ORDER = ['email', 'scraping', 'rag', 'knowledge_base', 'conversation']

const CATEGORY_LABELS: Record<string, string> = {
  email: 'Email',
  scraping: 'Scraping',
  rag: 'RAG',
  knowledge_base: 'Knowledge',
  conversation: 'Conversation',
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
  const q = useQuery({ queryKey: ['prompts'], queryFn: api.prompts })
  const [category, setCategory] = useState('email')
  const [selectedKey, setSelectedKey] = useState('')
  const [drafts, setDrafts] = useState<Record<string, string>>({})
  const [msg, setMsg] = useState('')
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

  const save = async () => {
    if (!selected || !dirty) return
    setSaving(true)
    setError('')
    try {
      await api.updatePrompt(selected.key, draftValue)
      setMsg(`Saved “${selected.label}”`)
      await q.refetch()
    } catch (e: any) {
      setError(e.message || 'Save failed')
      setMsg('')
    } finally {
      setSaving(false)
    }
  }

  const revert = () => {
    if (!selected) return
    setDrafts((prev) => {
      const next = { ...prev }
      delete next[selected.key]
      return next
    })
  }

  return (
    <div className="studio-screen">
      <div className="page-hero">
        <div>
          <h1>Prompt Studio</h1>
          <p>Tune Starlight’s voice across scraping, RAG, email writing, and replies.</p>
        </div>
        {msg ? <span className="pill ok">{msg}</span> : null}
      </div>

      <div className="studio-tabs" role="tablist" aria-label="Prompt categories">
        {categories.map((c) => (
          <button
            key={c}
            type="button"
            role="tab"
            aria-selected={category === c}
            className={`studio-tab${category === c ? ' active' : ''}`}
            onClick={() => setCategory(c)}
          >
            {categoryLabel(c)}
            <span className="studio-tab-count">
              {allPrompts.filter((p) => p.category === c).length}
            </span>
          </button>
        ))}
      </div>

      {q.isLoading ? (
        <div className="panel empty-state">Loading prompts…</div>
      ) : !prompts.length ? (
        <div className="panel empty-state">
          <strong>No prompts in {categoryLabel(category)}</strong>
          <p className="muted">Pick another category above.</p>
        </div>
      ) : (
        <div className="studio-body">
          <aside className="studio-nav panel stack">
            <div className="studio-nav-label muted">Prompts</div>
            <div className="studio-nav-list">
              {prompts.map((p) => {
                const isDirty = (drafts[p.key] ?? p.content) !== p.content
                return (
                  <button
                    key={p.key}
                    type="button"
                    className={`studio-nav-item${selectedKey === p.key ? ' active' : ''}`}
                    onClick={() => setSelectedKey(p.key)}
                  >
                    <span className="studio-nav-title">{p.label}</span>
                    <span className="studio-nav-key muted">{p.key}</span>
                    {isDirty ? <span className="studio-dirty" title="Unsaved changes" /> : null}
                  </button>
                )
              })}
            </div>
          </aside>

          {selected ? (
            <div className="studio-editor panel stack">
              <div className="studio-editor-head">
                <div style={{ minWidth: 0 }}>
                  <h2>{selected.label}</h2>
                  {selected.description ? (
                    <p className="muted">{selected.description}</p>
                  ) : null}
                  <code className="studio-key">{selected.key}</code>
                </div>
                <div className="row" style={{ flexShrink: 0 }}>
                  {dirty ? (
                    <button className="btn secondary" type="button" disabled={saving} onClick={revert}>
                      Discard
                    </button>
                  ) : null}
                  <button className="btn" type="button" disabled={saving || !dirty} onClick={save}>
                    {saving ? 'Saving…' : dirty ? 'Save prompt' : 'Saved'}
                  </button>
                </div>
              </div>

              {error ? (
                <div className="alert danger">
                  <strong>Save failed</strong>
                  <div style={{ marginTop: 4 }}>{error}</div>
                </div>
              ) : null}

              {selected.key === 'draft_system' ? (
                <div className="alert warn" style={{ margin: 0 }}>
                  Keep JSON keys <code>subject, preamble, opening_line, intro, feature_highlights, use_cases, cta</code>.
                  Changing to other key names empties campaign emails.
                </div>
              ) : null}

              <textarea
                className="textarea studio-textarea"
                value={draftValue}
                spellCheck={false}
                onChange={(e) => setDrafts({ ...drafts, [selected.key]: e.target.value })}
              />
            </div>
          ) : null}
        </div>
      )}
    </div>
  )
}
