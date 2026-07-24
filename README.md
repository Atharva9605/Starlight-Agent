---
title: Starlight AI Mailer
emoji: ⚡
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
license: mit
---

# Starlight AI-CRM Mailer

White-label **AI CRM Agent** platform: Azure OpenAI + RAG outbound mailer, Gmail conversation agent (HITL), FastAPI SaaS API, and a React product UI.

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the system map.

## Product UI (frontend)

```bash
cd frontend
cp .env.example .env   # set VITE_API_URL to your API (HF Space or localhost:7860)
npm install
npm run dev            # http://localhost:5173
```

Pages: **Inbox** · **Campaigns** · **Knowledge** · **Prompts** · **Templates** · **Settings** (branding + Gmail).

Streamlit `app.py` is legacy/dev-only. Production UX is the SPA.

## Deployment on Hugging Face Spaces

This app is running in a Docker container.

### Environment Variables Required:
Ensure you add these in the Space settings (Variables and Secrets):

- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_DEPLOYMENT_NAME`
- `AZURE_OPENAI_EMBEDDING_DEPLOYMENT`
- `GSUITE_DELEGATED_USER`
- `GSUITE_SERVICE_ACCOUNT_JSON_CONTENT`
- `DATABASE_URL` — **Neon PostgreSQL** connection string with pgvector (required for persistent knowledge base **and** conversation CRM)

### SaaS multi-tenant mode

The API runs in SaaS mode by default (`SAAS_MODE=true`). Each organization gets isolated prompts, templates, knowledge base, and CRM data.

**Required secrets:**

- `JWT_SECRET` — long random string for signing login tokens
- `DATABASE_URL` — required for SaaS (users, orgs, per-tenant config)
- `CORS_ORIGINS` — comma-separated frontend URLs (e.g. your Vercel app)
- `FRONTEND_URL` — used for Gmail OAuth redirect back to Settings

**Bootstrap existing deployments:**

- On first startup with an empty `users` table, set `SAAS_BOOTSTRAP_EMAIL` and `SAAS_BOOTSTRAP_PASSWORD` to create an admin linked to the default organization (existing data is migrated automatically).

**Optional per-org Gmail OAuth:**

- `GOOGLE_OAUTH_CLIENT_ID`, `GOOGLE_OAUTH_CLIENT_SECRET`, `GOOGLE_OAUTH_REDIRECT_URI`
- Orgs without OAuth use the platform service account (`GSUITE_*` secrets).

**Auth endpoints:** `POST /api/auth/signup`, `POST /api/auth/login`, `GET /api/auth/me`

Set `SAAS_MODE=false` only for local dev without Postgres (single-tenant fallback).

### Gmail Inbound (Conversations)

The conversation agent polls the delegated inbox for client replies and drafts contextual responses for human approval.

**Google Workspace setup (required for inbound sync):**

1. Open [Google Admin Console](https://admin.google.com) → Security → API controls → Domain-wide delegation
2. Edit your service account client ID and ensure these OAuth scopes are authorized:
   - `https://www.googleapis.com/auth/gmail.send`
   - `https://www.googleapis.com/auth/gmail.readonly`
3. Set `GSUITE_DELEGATED_USER` to the inbox that receives client replies (same account used for outbound send)

**Troubleshooting `invalid_grant` / `Invalid signature for token`:**

This means Google rejected the service account JWT — almost always a **corrupted private key** in `GSUITE_SERVICE_ACCOUNT_JSON_CONTENT` (common when pasting JSON into HuggingFace Secrets).

1. Google Cloud Console → IAM → Service Accounts → your account → Keys → **Add key → JSON** (download fresh)
2. In HF Space Secrets, set `GSUITE_SERVICE_ACCOUNT_JSON_CONTENT` to the **entire JSON file contents** (one line is fine)
3. Or base64-encode the file and paste the base64 string instead: `base64 -w0 service_account.json`
4. Verify Admin Console domain-wide delegation uses the service account **numeric Client ID** with scopes:
   - `https://www.googleapis.com/auth/gmail.send`
   - `https://www.googleapis.com/auth/gmail.readonly`
5. Hit `GET /api/gmail/status` after redeploy — should return `"ok": true`

**Optional:** `GMAIL_POLL_INTERVAL_SEC` — inbox poll interval in seconds (default: `120`)

Conversations require `DATABASE_URL`. Without Postgres, outbound pipeline still works but CRM features return HTTP 503.

### Persistent Knowledge Base (Neon + pgvector)

Catalogue chunks are stored in hosted Postgres, not on the container disk. Without this, uploads are lost when the Space rebuilds.

1. Create a free project at [neon.tech](https://neon.tech)
2. In the Neon SQL editor run: `CREATE EXTENSION IF NOT EXISTS vector;`
3. Add `DATABASE_URL` as a **Secret** in HF Space settings (format: `postgresql://user:pass@ep-xxx.neon.tech/neondb?sslmode=require`)

4. **Embedding dimension** must match your Azure embedding model:

| Deployment | Dimensions |
|------------|------------|
| `text-embedding-ada-002` | 1536 |
| `text-embedding-3-small` | 1536 |
| `text-embedding-3-large` | 3072 |

The app auto-detects from `AZURE_OPENAI_EMBEDDING_DEPLOYMENT`. Override with `EMBEDDING_DIMENSION` only if needed.

5. Check `/api/kb-status` — response includes `store.embedding_dimension` and `store.backend` for verification.

Without `DATABASE_URL`, the app falls back to local ChromaDB (development only).
