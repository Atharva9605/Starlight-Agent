const API_URL = (import.meta.env.VITE_API_URL as string | undefined)?.replace(/\/$/, '') || ''

export type OrgMembership = {
  organization_id: string
  role: string
  name?: string
}

function authHeaders(extra: HeadersInit = {}): HeadersInit {
  const token = localStorage.getItem('token')
  return {
    ...(extra || {}),
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, {
    ...init,
    headers: authHeaders({
      'Content-Type': 'application/json',
      ...(init.headers || {}),
    }),
  })
  if (res.status === 401) {
    localStorage.removeItem('token')
    if (!window.location.pathname.startsWith('/login')) {
      window.location.href = '/login'
    }
  }
  if (!res.ok) {
    let detail = res.statusText
    try {
      const body = await res.json()
      detail = body.detail || JSON.stringify(body)
    } catch {
      /* ignore */
    }
    throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail))
  }
  if (res.status === 204) return undefined as T
  return res.json()
}

export const api = {
  signup: (body: { email: string; password: string; name: string; org_name: string }) =>
    request<{ token: string }>('/api/auth/signup', { method: 'POST', body: JSON.stringify(body) }),
  login: (body: { email: string; password: string }) =>
    request<{ token: string; organizations: OrgMembership[] }>('/api/auth/login', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  me: () =>
    request<{
      user?: { email: string; name: string; id?: string }
      email?: string
      name?: string
      organization?: OrgMembership
      organization_id?: string
      organizations: OrgMembership[]
    }>('/api/auth/me'),
  switchOrg: (organization_id: string) =>
    request<{ token: string }>('/api/auth/switch-org', {
      method: 'POST',
      body: JSON.stringify({ organization_id }),
    }),
  branding: () =>
    request<{ display_name: string; logo_url: string; accent_color: string }>('/api/config/branding'),
  updateBranding: (body: Record<string, string>) =>
    request('/api/config/branding', { method: 'PUT', body: JSON.stringify(body) }),
  sender: () => request<Record<string, string>>('/api/config/sender'),
  updateSender: (body: Record<string, string>) =>
    request('/api/config/sender', { method: 'PUT', body: JSON.stringify(body) }),
  gmailStatus: () =>
    request<{ connected: boolean; connected_email?: string }>('/api/integrations/gmail/status'),
  gmailAuthorize: () =>
    request<{ authorize_url: string }>('/api/integrations/gmail/authorize'),
  gmailDisconnect: () => request('/api/integrations/gmail', { method: 'DELETE' }),
  conversations: (mailboxOnly = true) =>
    request<{ conversations: any[] }>(
      `/api/conversations?mailbox_only=${mailboxOnly ? 'true' : 'false'}`,
    ),
  conversation: (id: string) => request<any>(`/api/conversations/${id}`),
  generateDraft: (id: string, instructions = '') =>
    request(`/api/conversations/${id}/generate-draft`, {
      method: 'POST',
      body: JSON.stringify({ instructions }),
    }),
  updateDraft: (cid: string, mid: string, body: Record<string, string>) =>
    request(`/api/conversations/${cid}/drafts/${mid}`, {
      method: 'PUT',
      body: JSON.stringify(body),
    }),
  approveDraft: (cid: string, mid: string) =>
    request(`/api/conversations/${cid}/drafts/${mid}/approve`, { method: 'POST' }),
  rejectDraft: (cid: string, mid: string) =>
    request(`/api/conversations/${cid}/drafts/${mid}/reject`, { method: 'POST' }),
  gmailSync: () => request('/api/gmail/sync', { method: 'POST' }),
  kbStatus: () =>
    request<{
      chunk_count?: number
      chunks?: number
      catalogues: string[]
      backend?: string
      store?: any
    }>('/api/kb-status').then((r) => ({
      ...r,
      chunk_count: r.chunk_count ?? r.chunks ?? 0,
      catalogues: r.catalogues || [],
    })),
  clearKb: () => request('/api/clear-kb', { method: 'POST' }),
  prompts: () => request<{ prompts: any[] }>('/api/config/prompts'),
  updatePrompt: (key: string, content: string) =>
    request(`/api/config/prompts/${key}`, {
      method: 'PUT',
      body: JSON.stringify({ content }),
    }),
  templates: () => request<{ templates: any[] } | any[]>('/api/templates'),
  getTemplate: (name: string) => request<{ name: string; content: string }>(`/api/templates/${name}`),
  saveTemplate: (name: string, content: string, label?: string) =>
    request(`/api/templates/${name}`, {
      method: 'PUT',
      body: JSON.stringify({ content, label }),
    }),
  previewTemplate: (template_name: string, sample_data?: Record<string, any>) =>
    request<{ html: string }>('/api/templates/preview/json', {
      method: 'POST',
      body: JSON.stringify({ template_name, sample_data }),
    }),
  ragQuery: (client_desc: string, k = 5) =>
    request('/api/rag/query', { method: 'POST', body: JSON.stringify({ client_desc, k }) }),
  uploadLeads: async (file: File) => {
    const fd = new FormData()
    fd.append('file', file)
    const res = await fetch(`${API_URL}/api/upload-leads`, {
      method: 'POST',
      headers: authHeaders(),
      body: fd,
    })
    if (!res.ok) throw new Error(await res.text())
    return res.json()
  },
  uploadCatalogues: async (files: FileList) => {
    const fd = new FormData()
    Array.from(files).forEach((f) => fd.append('files', f))
    const res = await fetch(`${API_URL}/api/upload-catalogues`, {
      method: 'POST',
      headers: authHeaders(),
      body: fd,
    })
    if (!res.ok) throw new Error(await res.text())
    return res.json()
  },
  aiEvents: () => request<{ events: any[] }>('/api/ai/events'),
}

export function getApiBase() {
  return API_URL
}
