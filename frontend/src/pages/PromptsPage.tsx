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
    <div>
      <div className="page-hero">
        <div>
          <h1>Prompt Studio</h1>
          <p>Admin: tune Starlight’s voice, HyDE, and conversation agent.</p>
        </div>
        {msg ? <span className="pill ok">{msg}</span> : null}
      </div>

      <div className="row" style={{ marginBottom: '1rem' }}>
        {(categories.length ? categories : ['email', 'scraping', 'rag', 'knowledge_base', 'conversation']).map((c) => (
          <button key={c} className={`btn ${category === c ? '' : 'secondary'}`} onClick={() => setCategory(c)}>
            {c}
          </button>
        ))}
      </div>

      {prompts.map((p: any) => (
        <div key={p.key} className="panel stack" style={{ marginBottom: '0.9rem' }}>
          <div>
            <strong style={{ fontFamily: 'var(--display)' }}>{p.label}</strong>
            <div className="muted" style={{ fontSize: 13 }}>{p.description}</div>
          </div>
          <textarea
            className="textarea"
            rows={8}
            value={drafts[p.key] ?? p.content}
            onChange={(e) => setDrafts({ ...drafts, [p.key]: e.target.value })}
          />
          <button className="btn secondary" onClick={() => save(p.key)}>Save prompt</button>
        </div>
      ))}
    </div>
  )
}
