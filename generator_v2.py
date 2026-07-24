"""
Email Generator v2 – Azure OpenAI edition with metadata-aware RAG references.

Flow per client record:
  1. get_rag_context()          – HyDE query expansion → ChromaDB retrieval
                                  Returns text chunks + product reference metadata
                                  (blob_url, product_name, page_number, catalogue_name)
  2. generate_creative_draft()  – GPT-4o drafts structured email JSON,
                                  grounded strictly in retrieved catalogue context
  3. [DEPRECATED] Pass 3 removed to avoid safety triggers.
  4. generate_eml_from_record() – renders Jinja2 template, writes .html + .eml
                                  Template receives `referenced_products` list so
                                  each email shows the actual catalogue page images

Anti-hallucination guarantees:
  • Catalogue context injected verbatim — model forbidden from inventing specs
  • `json_mode=True` on generation → no JSON parse failures
  • Python-based normalization ensures list fields are plain strings
"""
import os
import sys
import json
import time
import re
from datetime import datetime
from urllib.parse import urlparse

from dotenv import load_dotenv
from jinja2 import Environment, BaseLoader
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage

load_dotenv()

from azure_client import azure_manager
from config_manager import get_prompt, get_sender
from vector_store import query as vector_query

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
def _sender_cfg() -> dict:
    return get_sender()

logo_path = os.path.join(os.path.dirname(__file__), "starlight.jpg")

COLLECTION_NAME = "starlight_vision"

infile      = sys.argv[1] if len(sys.argv) > 1 else "scraped_results.jsonl"
outdir_default = sys.argv[2] if len(sys.argv) > 2 else "out_emails_streamlit"
os.makedirs(outdir_default, exist_ok=True)


# ---------------------------------------------------------------------------
# RAG context retrieval with HyDE + reference extraction
# ---------------------------------------------------------------------------

def _build_rag_results(
    docs: list[str],
    metas: list[dict],
    distances: list[float] | None = None,
) -> tuple[str, list[str], list[dict], list[dict]]:
    """Build context string, raw docs, product refs, and ranked chunk trace."""
    context_parts: list[str] = []
    product_references: list[dict] = []
    chunks: list[dict] = []
    seen_blobs: set[str] = set()

    for i, (doc, meta) in enumerate(zip(docs, metas)):
        page_num = meta.get("page_number", 0)
        cat_name = meta.get("catalogue_name", "Starlight Catalogue")
        prod_name = meta.get("product_name", "")
        blob_url = meta.get("blob_url", "")

        ref_label = f"[{cat_name} — Page {page_num}]" if page_num else f"[{cat_name}]"
        context_parts.append(f"{ref_label}\n{doc}")

        dist = distances[i] if distances and i < len(distances) else None
        chunks.append({
            "rank": i + 1,
            "document": doc,
            "metadata": meta,
            "distance": dist,
        })

        if blob_url and blob_url not in seen_blobs and not blob_url.startswith("["):
            seen_blobs.add(blob_url)
            product_references.append({
                "product_name": prod_name or "Starlight Product",
                "catalogue_name": cat_name,
                "page_number": page_num,
                "blob_url": blob_url,
                "category": meta.get("category", ""),
                "specs_preview": meta.get("specs_preview", ""),
            })

    context_str = "\n\n---\n\n".join(context_parts) if context_parts else "No specific catalogue context found."
    return context_str, docs, product_references, chunks


def query_rag_with_trace(
    client_desc: str,
    k: int = 5,
    *,
    where: dict | None = None,
    max_distance: float | None = None,
) -> dict:
    """
    Run HyDE + vector search and return full trace for RAG Admin / debugging.

    Default where=catalogue_only so email attachment chunks never pollute sales RAG.
    """
    from ai_events import timed_ai_event

    if where is None:
        where = {"catalogue_only": True}
    if max_distance is None:
        max_distance = float(os.getenv("RAG_MAX_DISTANCE", "0.55"))

    with timed_ai_event("rag_hyde_query", prompt_key="hyde_system") as bag:
        hyde_user = get_prompt("hyde_user").format(client_desc=client_desc)
        hyde_messages = [
            {"role": "system", "content": get_prompt("hyde_system")},
            {"role": "user", "content": hyde_user},
        ]
        hyde_doc = azure_manager.chat_completion(hyde_messages, temperature=0.1, max_tokens=512)

        combined_query = (
            f"Client Context:\n{client_desc}\n\n"
            f"Ideal Product Characteristics:\n{hyde_doc}"
        )

        try:
            query_vector = azure_manager.embed_text(combined_query)
        except Exception as exc:
            return {
                "hyde_doc": hyde_doc,
                "combined_query": combined_query,
                "context_str": "No specific catalogue context found.",
                "chunks": [],
                "product_refs": [],
                "raw_docs": [],
                "empty_rag": True,
                "error": f"Embedding failed: {exc}",
            }

        try:
            results = vector_query(query_vector, k=k, where=where)
        except Exception as exc:
            return {
                "hyde_doc": hyde_doc,
                "combined_query": combined_query,
                "context_str": "No specific catalogue context found.",
                "chunks": [],
                "product_refs": [],
                "raw_docs": [],
                "empty_rag": True,
                "error": f"Vector search failed: {exc}",
            }

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0] if results.get("distances") else []

        # Drop weak matches
        if docs and dists:
            kept = [
                (d, m, dist)
                for d, m, dist in zip(docs, metas, dists)
                if dist is None or float(dist) <= max_distance
            ]
            if kept:
                docs, metas, dists = map(list, zip(*kept))
            else:
                docs, metas, dists = [], [], []

        if not docs:
            bag["meta"] = {"empty_rag": True}
            return {
                "hyde_doc": hyde_doc,
                "combined_query": combined_query,
                "context_str": (
                    "No specific catalogue context found. "
                    "Do NOT invent product names, specs, or model numbers. "
                    "Write a relationship-building email without product claims."
                ),
                "chunks": [],
                "product_refs": [],
                "raw_docs": [],
                "empty_rag": True,
            }

        context_str, raw_docs, product_refs, chunks = _build_rag_results(docs, metas, dists)
        bag["chunk_ids"] = [
            str((c.get("metadata") or {}).get("source", c.get("rank"))) for c in chunks
        ]
        return {
            "hyde_doc": hyde_doc,
            "combined_query": combined_query,
            "context_str": context_str,
            "chunks": chunks,
            "product_refs": product_refs,
            "raw_docs": raw_docs,
            "empty_rag": False,
        }


def get_rag_context(
    client_desc: str,
    k: int = 5,
) -> tuple[str, list[str], list[dict]]:
    """
    Retrieve the most relevant catalogue chunks for a client profile.

    Uses Hypothetical Document Embedding (HyDE): GPT-4o first writes an
    idealised product description matching the client's needs, then the
    combined (real + hypothetical) query is embedded and used to search
    the vector store.

    Returns:
        context_str:       Formatted string for injection into prompts.
        raw_docs:          List of raw chunk strings.
        product_references: List of dicts with product reference info.
    """
    trace = query_rag_with_trace(client_desc, k=k)
    if trace.get("error"):
        print(f"Warning: {trace['error']}")
    return trace["context_str"], trace["raw_docs"], trace["product_refs"]


# ---------------------------------------------------------------------------
# Jinja2 template helper
# ---------------------------------------------------------------------------

def get_template(template_name: str = "email_template.html"):
    from config_manager import get_template_content
    try:
        env = Environment(loader=BaseLoader())
        return env.from_string(get_template_content(template_name))
    except Exception as exc:
        print(f"Warning: Could not load template '{template_name}': {exc}")
        env = Environment()
        return env.from_string(
            "<html><body>"
            "<p>{{ intro_paragraph | safe }}</p>"
            "<ul>{% for item in bullets %}<li>{{ item }}</li>{% endfor %}</ul>"
            "<p>{{ closing_paragraph }}</p>"
            "<p>-- {{ sender_name }}<br>{{ sender_company }}</p>"
            "</body></html>"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_json(txt: str) -> dict | None:
    if not txt:
        return None
    try:
        start = txt.find("{")
        end = txt.rfind("}")
        if start != -1 and end > start:
            return json.loads(txt[start: end + 1])
    except (json.JSONDecodeError, ValueError):
        pass
    return None


def sanitize_filename(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-@.]", "_", s or "no_email")


# ---------------------------------------------------------------------------
# Draft generation
# ---------------------------------------------------------------------------

_DRAFT_SYSTEM = """\
You are an expert B2B sales copywriter for Starlight Linear LED — an award-winning
Indian LED lighting manufacturer (Top 10 Brands in Lightings 2025, Homes India Magazine).

Company Focus: End-to-end LED lighting solutions, custom manufacturing, supply, installation.
Address: 3 Vedant 3, P&T Colony, Gandhi Nagar, Dombivali East, Thane 421203.
Phone: 9619436066 | Email: vivek@starlightlinearled.com

Core Principles:
1. Ground your suggestions in the product names and applications found in the CATALOGUE CONTEXT.
2. Focus on writing a pleasing, relationship-building email. Avoid listing dense technical specifications (like dimensions or IP ratings) in the main body.
3. Incorporate project names or portfolio items from the provided client data to show personalized relevance.
4. If specific projects aren't available, focus on their general architectural or design style.

Tone and Style:
- Professional and personalized: Mention their work and suggest relevant Starlight solutions.
- Clear and easy to read.
- Use plain text for links (no raw HTML).
- Subject: Engaging and relevant to their industry.
- Salutation: Personalized to the team or firm.

Expected Output Format (JSON):
{
  "subject": "string",
  "preamble": "string (one elegant tagline, ≤12 words)",
  "opening_line": "string",
  "intro": "string (1–2 sentences on their projects and our synergy)",
  "feature_highlights": ["string", "string", "string"],
  "use_cases": ["string", "string"],
  "cta": "string"
}
"""

_DRAFT_USER_TMPL = """\
Below is the CATALOGUE CONTEXT (products and specifications) and the CLIENT DATA for personalization.

CATALOGUE CONTEXT:
---
{rag_context}
---

CLIENT DATA:
{client_json}
"""


def generate_creative_draft(rec: dict) -> tuple[str, list[str], list[dict], dict]:
    from schemas import OutboundDraft, parse_with_retry
    from ai_events import timed_ai_event

    client_json = json.dumps(rec, ensure_ascii=False)
    print(f"  [RAG] Querying catalogue knowledge base for: {rec.get('company', '?')}…")
    rag_trace = query_rag_with_trace(client_json)
    rag_context = rag_trace["context_str"]
    raw_docs = rag_trace["raw_docs"]
    product_refs = rag_trace["product_refs"]

    if rag_trace.get("empty_rag"):
        product_refs = []

    messages = [
        {"role": "system", "content": get_prompt("draft_system")},
        {
            "role": "user",
            "content": get_prompt("draft_user").format(
                rag_context=rag_context,
                client_json=client_json,
            ),
        },
    ]

    with timed_ai_event("outbound_draft", prompt_key="draft_system") as bag:
        bag["chunk_ids"] = [
            str((c.get("metadata") or {}).get("source", ""))
            for c in rag_trace.get("chunks") or []
        ]

        def _call():
            return azure_manager.chat_completion(
                messages,
                temperature=0.25,
                max_tokens=2048,
                json_mode=True,
            )

        raw = _call()
        if not raw:
            raise RuntimeError("Azure OpenAI returned empty draft response.")

        def _retry():
            retry_messages = messages + [
                {
                    "role": "user",
                    "content": (
                        "Your previous response failed schema validation. "
                        "Return ONLY a valid JSON object with keys: "
                        "subject, preamble, opening_line, intro, "
                        "feature_highlights, use_cases, cta."
                    ),
                }
            ]
            return azure_manager.chat_completion(
                retry_messages, temperature=0.1, max_tokens=2048, json_mode=True
            )

        draft = parse_with_retry(OutboundDraft, raw, retry_fn=_retry)
        return draft.model_dump_json(), raw_docs, product_refs, rag_trace


# ---------------------------------------------------------------------------
# List cleaners (Pass 2 removed - Pass 1 JSON mode is sufficient)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# List cleaners
# ---------------------------------------------------------------------------

def clean_item(x) -> str:
    if isinstance(x, dict):
        title = x.get("title", "")
        desc = x.get("description", x.get("desc", ""))
        if title and desc:
            return f"<b>{title}</b><br>{desc}"
        return " – ".join(str(v) for v in x.values() if v)
    if isinstance(x, str):
        x = x.strip()
        if x.startswith("{") and x.endswith("}"):
            import ast
            try:
                d = ast.literal_eval(x)
                if isinstance(d, dict):
                    t = d.get("title", "")
                    d2 = d.get("description", d.get("desc", ""))
                    if t and d2:
                        return f"<b>{t}</b><br>{d2}"
            except Exception:
                pass
        x = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", x)
    return x


def ensure_list(val) -> list:
    if isinstance(val, list):
        return [clean_item(x) for x in val]
    if isinstance(val, str):
        val = val.strip()
        if not val:
            return []
        items = [v.strip() for v in val.splitlines() if v.strip()]
        return [clean_item(x) for x in items] if items else [clean_item(val)]
    return []


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def generate_eml_from_record(
    rec: dict,
    idx: int,
    outdir: str,
    template_name: str = "email_template.html",
    *,
    template_content: str | None = None,
) -> tuple[str, dict] | None:
    """
    Generate a single .html + .eml email from a scraped client record.

    The generated email includes:
    - Personalised intro and feature bullets grounded in catalogue context
    - `referenced_products` list passed to the template so it can render
      product cards with the catalogue page image (blob_url), product name,
      and specs preview.

    Returns (eml_path, trace_info) on success, or None on failure.
    """
    if not rec:
        print(f"Error: Empty record at index {idx}.")
        return None

    print(f"\n── Record {idx} | template='{template_name}'")

    creative_draft, raw_docs, product_refs, rag_trace = generate_creative_draft(rec)
    if not creative_draft:
        raise RuntimeError("Generation failed: no creative draft.")

    parsed = extract_json(creative_draft)
    if not parsed:
        raise RuntimeError("Generation failed: could not parse draft JSON from pass 1.")

    sender = _sender_cfg()

    # Subject validation
    subject = parsed.get("subject", "")
    if not subject or "{{" in subject or "}}" in subject:
        company_name = rec.get("company", rec.get("website", "your company"))
        if "http" in str(company_name):
            company_name = urlparse(company_name).netloc.replace("www.", "")
        fallback = sender.get(
            "subject_fallback",
            "Starlight LED – Precision Lighting Solutions for {company_name}",
        )
        subject = fallback.format(company_name=company_name)

    preamble         = parsed.get("preamble",  "Precision-engineered LED solutions, delivered on time.")
    opening_line     = parsed.get("opening_line", "Hope this email finds you well.")
    intro            = parsed.get("intro", "").replace("\n", "<br>")
    feature_highlights = ensure_list(parsed.get("feature_highlights", []))
    use_cases        = ensure_list(parsed.get("use_cases", []))
    technical_specs  = []
    bullets          = []
    cta              = parsed.get(
        "cta",
        "Would you be available for a brief call next week to explore how we can "
        "illuminate your next project?",
    )

    # Render template
    if template_content:
        env = Environment(loader=BaseLoader())
        template = env.from_string(template_content)
    else:
        template = get_template(template_name)
    html_out = template.render(
        subject=subject,
        preamble=preamble,
        opening_line=opening_line,
        intro_paragraph=intro,
        bullets=bullets,
        feature_highlights=feature_highlights,
        use_cases=use_cases,
        technical_specs=technical_specs,
        closing_paragraph=cta,
        sender_name=sender["sender_name"],
        sender_company=sender["sender_company"],
        sender_phone=sender["sender_phone"],
        sender_website=sender["sender_website"],
        sender_email=sender["sender_email"],
        company_logo_url=sender.get("company_logo_url", "cid:company_logo"),
        catalog_chunks=raw_docs,
        referenced_products=product_refs,   # ← NEW: product reference cards
    )

    safe_name = sanitize_filename(
        rec.get("emails") or rec.get("company") or f"record_{idx}"
    )
    html_path = os.path.join(outdir, f"{idx}_{safe_name}.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_out)

    # Build .eml
    msg = MIMEMultipart("alternative")
    msg["From"]    = f'"{sender["sender_name"]}" <{sender["sender_email"]}>'
    msg["To"]      = str(rec.get("emails", ""))
    msg["Subject"] = subject
    msg.attach(MIMEText(html_out, "html"))

    if sender.get("company_logo_url", "cid:company_logo") == "cid:company_logo" and os.path.exists(logo_path):
        with open(logo_path, "rb") as f:
            img = MIMEImage(f.read())
            img.add_header("Content-ID", "<company_logo>")
            img.add_header("Content-Disposition", "inline", filename="logo.jpg")
            msg.attach(img)

    eml_path = os.path.join(outdir, f"{idx}_{safe_name}.eml")
    with open(eml_path, "w", encoding="utf-8") as f:
        f.write(msg.as_string())

    print(f"  ✓ {eml_path}  ({len(product_refs)} product reference(s))")

    trace_info = {
        "hyde_doc": rag_trace.get("hyde_doc", ""),
        "combined_query": rag_trace.get("combined_query", ""),
        "chunks": rag_trace.get("chunks", []),
        "raw_docs": raw_docs,
        "product_refs": product_refs,
        "creative_draft": creative_draft,
        "subject": subject,
        "html": html_out,
        "from": msg["From"],
        "to": msg["To"],
        "company": rec.get("company", ""),
        "website": rec.get("website", ""),
    }
    return eml_path, trace_info


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    try:
        with open(infile, encoding="utf-8") as f:
            for idx, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    print(f"Warning: Skipping malformed JSON on line {idx}.")
                    continue
                generate_eml_from_record(rec, idx, outdir_default)
                time.sleep(0.5)
    except FileNotFoundError:
        print(f"Error: '{infile}' not found.")


if __name__ == "__main__":
    main()
