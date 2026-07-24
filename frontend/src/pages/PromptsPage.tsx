import { useQuery } from '@tanstack/react-query'
import { useMemo, useState } from 'react'
import { api } from '../api/client'

export function PromptsPage() {
  const q = useQuery({ queryKey: ['prompts'], queryFn: api.prompts })
  const [category, setCategory] = useState('email')
  const [drafts, setDrafts] = useState<Record<string, string>>({})
  const [msg, setMsg] = useState('')

  const categories = useMemo(() => {
    const set = new Set((q.data?.prompts || []).map((p: any) => p.category))
    return Array.from(set)
  }, [q.data])

  const prompts = (q.data?.prompts || []).filter((p: any) => p.category === category)

  const save = async (key: string) => {
    const content = drafts[key]
    if (content == null) return
    await api.updatePrompt(key, content)
    setMsg(`Saved ${key}`)
    q.refetch()
  }

  return (
    <div className="stack" style={{ gap: '1.25rem' }}>
      <div>
        <h1 style={{ margin: 0, fontFamily: 'var(--font-display)', fontWeight: 500 }}>Prompt Studio</h1>
        <p className="muted">Edit AI behavior without deploying code.</p>
      </div>
      <div className="row">
        {(categories.length ? categories : ['email', 'scraping', 'rag', 'knowledge_base', 'conversation']).map((c) => (
          <button key={c} className={`btn ${category === c ? '' : 'secondary'}`} onClick={() => setCategory(c)}>
            {c}
          </button>
        ))}
      </div>
      {msg ? <div className="muted">{msg}</div> : null}
      {prompts.map((p: any) => (
        <div key={p.key} className="panel stack">
          <div>
            <strong>{p.label}</strong>
            <div className="muted" style={{ fontSize: 13 }}>{p.description}</div>
          </div>
          <textarea
            className="textarea"
            rows={8}
            value={drafts[p.key] ?? p.content}
            onChange={(e) => setDrafts({ ...drafts, [p.key]: e.target.value })}
          />
          <button className="btn secondary" onClick={() => save(p.key)}>Save</button>
        </div>
      ))}
    </div>
  )
}
