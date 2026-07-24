# Lovable Prompt — Starlight AI Mailer Frontend (Vercel)

Copy everything below the line into Lovable as your project prompt.

---

## Project: Starlight AI-CRM Mailer Dashboard

Build a premium B2B SaaS admin dashboard for **Starlight AI Mailer** — an AI-powered CRM that scrapes prospect websites, generates personalized sales emails using RAG over product catalogues, and sends them via GSuite.

**Backend API (already deployed):** `https://atharva9605-starlight-ai-mailer.hf.space`  
(Set `VITE_API_URL` env var on Vercel to this URL — no trailing slash)

**Tech stack for frontend:**
- React + TypeScript + Vite
- Tailwind CSS + shadcn/ui
- React Router
- TanStack Query for API calls
- Deploy on **Vercel**

**Design:** Dark cyberpunk/glassmorphism aesthetic — deep navy (#0F172A) background, purple (#8B5CF6) and cyan (#06B6D4) accents, Plus Jakarta Sans font, glass cards with blur, subtle gradients. Professional B2B feel, not playful.

---

## Pages & Features

### 1. Login (simple)
- Email/password form (cosmetic — backend returns dummy token)
- `POST /api/auth/login` → store token in localStorage
- Redirect to Dashboard

### 2. Dashboard (main mailing pipeline)
**Layout:** Sidebar navigation + main content area.

**Sidebar sections:**
- Logo / "STARLIGHT" branding
- Nav: Dashboard | Prompt Studio | Template Editor | Knowledge Base | Settings
- KB status widget: chunk count + catalogue names from `GET /api/kb-status`

**Dashboard content:**
- **Metrics row:** Catalogues count | Total Leads | Processed | Status (Idle/Running)
- **Upload leads:** Excel file uploader → `POST /api/upload-leads` → returns `{ leads: [...] }` with `website` column required
- **Mailer settings panel:**
  - GSuite sender email (text input)
  - Recipient override (optional, for testing)
  - Template dropdown from `GET /api/templates` → use `name` field as value
  - Delay between sends (seconds, default 3)
- **Leads table:** Show uploaded leads with status column (⏳ Pending → ⚙️ Processing → ✅ Sent / ❌ Failed)
- **START PIPELINE button:** Calls `POST /api/process-stream` with body:
```json
{
  "leads": [...],
  "sender_email": "user@company.com",
  "recipient_override": "",
  "template": "email_template.html",
  "delay": 3
}
```
- **SSE streaming:** Parse `text/event-stream` responses. Event types:
  - `log` → append to terminal log panel
  - `status_update` → update row status by `row_index`
  - `preview_html` → show in email preview panel
  - `rag_trace` → show RAG debug info in expandable panel
  - `done` → pipeline complete
- **Live terminal log:** Monospace green-on-black terminal UI, auto-scroll
- **Email preview panel:** Render `preview_html` events in an iframe or dangerouslySetInnerHTML sandbox
- **Actions:** Stop (`POST /api/stop-processing`), Reset queue (`POST /api/reset-queue`), Download EMLs (`GET /api/download-emls`)

### 3. Prompt Studio (CRITICAL — business-editable AI prompts)
Fetch all prompts: `GET /api/config/prompts`

Display prompts grouped by **category** tabs:
| Category | Prompts |
|----------|---------|
| **Email** | Email Writing (System), Email Writing (User) |
| **Scraping** | Website Analysis |
| **RAG** | RAG Query Expansion (HyDE), RAG Query User Message |
| **Knowledge Base** | Catalogue Text Extraction, Catalogue Vision Extraction, Vision User Message |

**Each prompt card shows:**
- Label + description
- Available template variables as chips (e.g. `{rag_context}`, `{client_json}`)
- Large monospace textarea for editing prompt content
- Save button → `PUT /api/config/prompts/{key}` with `{ "content": "..." }`
- Character count

**Top actions:**
- Save All → `PUT /api/config/prompts` with `{ "prompts": { "key": "content", ... } }`
- Reset All to Defaults → `POST /api/config/reset` (confirm dialog)

### 4. Template Editor (CRITICAL — HTML email templates)
**Left panel:** Template list from `GET /api/templates`
- Show label, name, builtin/custom badge
- Select template → load content via `GET /api/templates/{name}`

**Center:** Code editor (Monaco or CodeMirror) for HTML/Jinja2 template
- Syntax highlighting for HTML
- Variable reference sidebar from `GET /api/templates/variables` — show all `{{ variable }}` names with descriptions

**Right panel:** Live preview
- On edit debounce (500ms), call `POST /api/templates/preview/json`:
```json
{
  "template_content": "<html>...",
  "sample_data": null
}
```
- Render returned `{ html }` in iframe preview (desktop email width ~660px)
- Toggle: Desktop / Mobile preview widths

**Actions:**
- Save → `PUT /api/templates/{name}` with `{ "content": "...", "label": "My Template" }`
- Reset to Default (builtin only) → `POST /api/templates/{name}/reset`
- Create New Template → prompt for name → save as custom
- Delete custom template → `DELETE /api/templates/{name}`

### 5. Knowledge Base
- Upload PDF catalogues (multi-file) → `POST /api/upload-catalogues` (multipart form, field name `files`)
- Show upload results (success/error per file)
- Clear KB button with confirm → `POST /api/clear-kb`
- Display current status from `GET /api/kb-status`

### 6. Settings (Sender / Business Info)
Form fields from `GET /api/config/sender`:
- Sender Name, Company, Phone, Website, Email
- Company Logo URL
- Subject Fallback template (supports `{company_name}`)

Save → `PUT /api/config/sender`

---

## API Client Setup

Create `src/lib/api.ts`:
```typescript
const API_URL = import.meta.env.VITE_API_URL || 'https://atharva9605-starlight-ai-mailer.hf.space';

export async function api<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || err.error || 'API error');
  }
  return res.json();
}
```

For file uploads, use `FormData` without Content-Type header.
For SSE, use `fetch` with `ReadableStream` reader or EventSource won't work for POST — use fetch streaming.

---

## SSE Helper Example
```typescript
export async function processLeadsStream(
  body: ProcessRequest,
  onEvent: (event: Record<string, unknown>) => void
) {
  const res = await fetch(`${API_URL}/api/process-stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const reader = res.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';
    for (const line of lines) {
      if (line.startsWith('data: ')) {
        onEvent(JSON.parse(line.slice(6)));
      }
    }
  }
}
```

---

## Vercel Deployment

1. Set environment variable: `VITE_API_URL=https://atharva9605-starlight-ai-mailer.hf.space`
2. Build command: `npm run build`
3. Output: `dist`
4. No backend on Vercel — frontend only, all API calls go to HuggingFace Space

---

## UX Requirements

- Toast notifications on save success/error (sonner)
- Loading skeletons while fetching
- Confirm dialogs for destructive actions (reset prompts, clear KB, delete template)
- Responsive but optimized for desktop (admin tool)
- Unsaved changes warning when navigating away from Prompt Studio or Template Editor
- All API errors shown as user-friendly toasts

---

## Do NOT build

- Do not implement backend logic — it already exists on HuggingFace
- Do not use Supabase or any database on frontend
- Do not hardcode prompts — always load from API
- Do not embed Streamlit

---

## Priority order

1. Dashboard with SSE pipeline (core value)
2. Prompt Studio (business requirement)
3. Template Editor with live preview (business requirement)
4. Knowledge Base upload
5. Settings page

Build all pages with consistent sidebar navigation. Make it production-ready and beautiful.
