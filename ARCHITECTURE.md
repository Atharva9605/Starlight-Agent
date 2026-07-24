# Architecture — CRM Agent Platform (as built)

## What this is

White-label **AI CRM Agent** platform:

- FastAPI backend (`api.py`) on Hugging Face / Docker
- React + Vite SPA (`frontend/`) for the product UI
- HITL: AI drafts, humans approve sends

## Current vs target

| Area | Status |
|------|--------|
| Multi-org Gmail sync | Done — `gmail_sync_loop` iterates `list_orgs_with_gmail()` |
| RAG kind / conversation filters | Done — `vector_store.query(where=...)` |
| SaaS Chroma guard | Done — `require_scoped_backend()` |
| Pydantic structured outputs | Done — `schemas.py` |
| Empty-RAG policy | Done — outbound draft path |
| Neutral / white-label prompts | Done — `defaults/business_config.json` |
| Tool-calling reply agent | Done — `conversation_agent.py` (max 3 steps) |
| Memory summary | Done — `conversation_summary` |
| Draft gate | Done — `should_auto_draft` |
| AI event log | Done — `ai_events.py` + `/api/ai/events` |
| Offline evals | Done — `evals/run_evals.py` |
| White-label UI | Done — `frontend/` (Inbox, Campaigns, Knowledge, Prompts, Templates, Settings) |

## Pipelines

**Outbound:** upload leads → scrape → HyDE catalogue RAG → structured draft → Jinja template → GSuite send → CRM record

**Inbound:** Gmail poll (per org) → noise filter → draft gate → tool-calling agent → draft → human approve → send

## Frontend

```
frontend/
  src/api/client.ts
  src/auth/AuthContext.tsx
  src/components/AppShell.tsx
  src/pages/{Auth,Inbox,Thread,Campaigns,Knowledge,Prompts,Templates,Settings}
```

Env: `VITE_API_URL` (API origin). CORS + `FRONTEND_URL` on backend.

## Architect vocabulary

- **Pipeline** — fixed host-orchestrated steps (outbound mailer)
- **Agent** — model chooses tools in a bounded loop (reply drafter)
- **HITL** — human approves before send
- **Tenancy** — org-scoped JWT + vector + conversations
