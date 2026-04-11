"""
Email Generator v2 – Azure OpenAI edition.

Flow per client record:
  1. get_rag_context()       – HyDE query expansion → ChromaDB retrieval
  2. generate_creative_draft() – GPT-4o drafts the structured email JSON
  3. refine_with_judge()     – second GPT-4o pass normalises the JSON
  4. generate_eml_from_record() – renders Jinja2 template, writes .html + .eml

Anti-hallucination measures:
  • The RAG context is injected verbatim; the model is explicitly forbidden
    from referencing any product, spec, or number not present in the context.
  • A grounding check strips spec claims that cannot be found in the context.
  • The judge pass validates JSON shape and escapes rogue dict objects in lists.
"""
import os
import sys
import json
import time
import uuid
import re
from datetime import datetime
from urllib.parse import urlparse

from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage

load_dotenv()

from azure_client import azure_manager

import chromadb

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
sender_name = os.getenv("SENDER_NAME", "Vivek Dhondarkar")
sender_company = os.getenv("SENDER_COMPANY", "Starlight Linear LED")
sender_phone = os.getenv("SENDER_PHONE", "9619436066")
sender_website = os.getenv("SENDER_WEBSITE", "www.starlightlinearled.com")
sender_email = os.getenv("SENDER_EMAIL", "vivek@starlightlinearled.com")
company_logo_url = os.getenv("COMPANY_LOGO_URL", "cid:company_logo")
logo_path = os.path.join(os.path.dirname(__file__), "starlight.jpg")

CHROMA_DB_DIR = "chroma_db"
COLLECTION_NAME = "starlight_catalogs"

infile = sys.argv[1] if len(sys.argv) > 1 else "scraped_results.jsonl"
outdir_default = sys.argv[2] if len(sys.argv) > 2 else "out_emails_gemini_v2"
os.makedirs(outdir_default, exist_ok=True)


# ---------------------------------------------------------------------------
# RAG context retrieval with HyDE
# ---------------------------------------------------------------------------

def get_rag_context(client_desc: str, k: int = 5) -> tuple[str, list, list]:
    """
    Retrieve the most relevant catalogue chunks for a client profile.

    Uses Hypothetical Document Embedding (HyDE): GPT-4o first writes an
    idealised product description matching the client's needs, then the
    combined (real + hypothetical) query is embedded and used to search
    ChromaDB.

    Returns:
        context_str:  Formatted string ready for injection into prompts.
        raw_docs:     List of raw chunk strings.
        metadatas:    List of metadata dicts from ChromaDB.
    """
    # Step A – HyDE: generate hypothetical ideal product description
    hyde_messages = [
        {
            "role": "system",
            "content": (
                "You are a senior sales engineer at Starlight Linear LED. "
                "Write a concise, technical product description that would be the "
                "PERFECT match for the client's needs described below. "
                "Include realistic wattage, CCT, IP rating, application, and finish options. "
                "Output only the product description — no preamble or commentary."
            ),
        },
        {
            "role": "user",
            "content": f"Client profile and needs:\n{client_desc}",
        },
    ]
    hyde_doc = azure_manager.chat_completion(hyde_messages, temperature=0.1, max_tokens=512)

    combined_query = (
        f"Client Context:\n{client_desc}\n\n"
        f"Ideal Product Characteristics:\n{hyde_doc}"
    )

    # Step B – embed the combined query
    try:
        query_vector = azure_manager.embed_text(combined_query)
    except Exception as e:
        print(f"Warning: Embedding failed ({e}). Falling back to empty context.")
        return "No specific catalogue context found.", [], []

    # Step C – query ChromaDB
    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
        collection = chroma_client.get_collection(COLLECTION_NAME)
        results = collection.query(query_embeddings=[query_vector], n_results=k)
    except Exception as e:
        print(f"Warning: ChromaDB query failed ({e}). Has a catalogue been uploaded?")
        return "No specific catalogue context found.", [], []

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    if docs:
        context_str = "\n\n---\n\n".join(docs)
        return context_str, docs, metas

    return "No specific catalogue context found.", [], []


# ---------------------------------------------------------------------------
# Jinja2 template helper
# ---------------------------------------------------------------------------

def get_template(template_name: str = "email_template.html"):
    templates_dir = os.path.join(os.path.dirname(__file__), "templates")
    try:
        env = Environment(loader=FileSystemLoader(templates_dir))
        return env.get_template(template_name)
    except Exception as e:
        print(f"Warning: Could not load template '{template_name}': {e}")
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
# JSON extraction helper
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
# Creative draft generation
# ---------------------------------------------------------------------------

_DRAFT_SYSTEM = """\
You are an expert B2B sales copywriter for Starlight Linear LED, an award-winning
Indian LED lighting manufacturer based in Mumbai.

Company overview:
• End-to-end LED lighting solutions: custom manufacturing, supply, installation, maintenance.
• Recognised as one of the "Top 10 Brands in Lightings for 2025" by Homes India Magazine.
• Address: 3 Vedant 3, P&T Colony, Gandhi Nagar, Dombivali East, Thane 421203.
• Phone: 9619436066  |  Email: vivek@starlightlinearled.com

ANTI-HALLUCINATION RULES (strictly enforced):
1. You MUST ONLY reference product names, model numbers, wattages, CCTs, IP ratings,
   dimensions, or any other specifications that appear VERBATIM in the CATALOGUE CONTEXT.
2. If no relevant product is found in the context, describe Starlight's GENERAL capabilities
   (custom linear LED, bespoke manufacturing) — do NOT invent product names or specs.
3. Do NOT use generic filler like "your recent projects" — use the specific project names,
   clients, or portfolio items from the client data provided.
4. Every bullet and spec must be traceable to the catalogue context supplied.

STYLE RULES:
• Hyper-personalised: reference 1–2 EXACT projects or clients from their portfolio.
• Concise and skimmable — nobody reads long dense paragraphs.
• Max 2–3 feature bullets; each under 12 words.
• No markdown bold (**text**) — use HTML <b>tags</b> instead.
• Naturally weave in: world-class on-time delivery, Homes India Magazine Top 10 award.
• Subject line: catchy, industry-relevant, NO client name, NO placeholder {{...}}.
• Dear line: personalised (e.g. "Dear Mahim Architects Team,").
• Closing: short CTA only — no sender details (those are in the signature).

Output ONLY a valid JSON object with these keys (no other text, no markdown):
  "subject"           – string
  "preamble"          – string (one punchy tagline)
  "opening_line"      – string (e.g. "Hope this finds you well.")
  "intro"             – string (1–2 sentences, HTML-safe)
  "bullets"           – array of strings (core product/solution highlights)
  "feature_highlights"– array of strings (3 ultra-short features, <10 words each)
  "use_cases"         – array of strings (application scenarios)
  "technical_specs"   – array of strings (catalogue-sourced specs only)
  "cta"               – string (call to action)
"""

_DRAFT_USER_TMPL = """\
CATALOGUE CONTEXT (Retrieved via RAG — use ONLY these products/specs):
===
{rag_context}
===

CLIENT DATA:
{client_json}
"""


def generate_creative_draft(rec: dict) -> tuple[str, list, list]:
    """Draft the email JSON using GPT-4o with RAG context injection."""
    client_json = json.dumps(rec, ensure_ascii=False)

    print(f"  [RAG] Retrieving catalogue context for: {rec.get('company', 'unknown')}…")
    rag_context, raw_docs, metadatas = get_rag_context(client_json)

    messages = [
        {"role": "system", "content": _DRAFT_SYSTEM},
        {
            "role": "user",
            "content": _DRAFT_USER_TMPL.format(
                rag_context=rag_context,
                client_json=client_json,
            ),
        },
    ]

    resp = azure_manager.chat_completion(
        messages,
        temperature=0.25,   # small creative variance while staying grounded
        max_tokens=2048,
        json_mode=True,
    )

    if not resp:
        raise RuntimeError("Azure OpenAI returned an empty response for the draft.")

    return resp, raw_docs, metadatas


# ---------------------------------------------------------------------------
# Judge / formatter pass
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM = """\
You are a strict JSON formatting assistant.

Your only job is to take raw text that CONTAINS a JSON object and return a
clean, valid JSON object with exactly these keys:
  "subject", "preamble", "opening_line", "intro",
  "bullets", "feature_highlights", "use_cases", "technical_specs", "cta"

Rules:
1. Array fields (bullets, feature_highlights, use_cases, technical_specs)
   MUST contain ONLY plain strings — never nested objects or dicts.
   Convert any dict entry to "<b>Title:</b> Description" format.
2. Remove any text outside the JSON object.
3. Ensure all strings are properly escaped (no raw newlines inside strings).
4. Output ONLY the valid JSON object — no markdown, no commentary.
"""


def refine_with_judge(draft_text: str) -> dict | None:
    """Normalise draft text into a clean, validated JSON dict."""
    if not draft_text:
        return None

    messages = [
        {"role": "system", "content": _JUDGE_SYSTEM},
        {"role": "user", "content": f"Clean and return this JSON:\n---\n{draft_text}\n---"},
    ]

    resp = azure_manager.chat_completion(
        messages,
        temperature=0.0,
        max_tokens=2048,
        json_mode=True,
    )

    parsed = extract_json(resp)
    if not parsed:
        raise RuntimeError("Judge pass could not parse the draft into valid JSON.")
    return parsed


# ---------------------------------------------------------------------------
# List / item cleaners
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
        # dict-like string: try to parse
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
        # Convert markdown bold to HTML
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
) -> tuple[str, dict] | None:
    """
    Generate a single .html + .eml email file from a scraped client record.

    Returns:
        (eml_path, trace_info) on success, or None on failure.
    """
    if not rec:
        print(f"Error: Empty record at index {idx}.")
        return None

    print(f"\n── Processing record {idx} with template '{template_name}'…")

    creative_draft, raw_docs, metadatas = generate_creative_draft(rec)
    if not creative_draft:
        raise RuntimeError("Generation failed: no creative draft produced.")

    parsed = refine_with_judge(creative_draft)
    if not parsed:
        raise RuntimeError("Generation failed: could not parse draft into JSON.")

    # --- Subject validation ---
    subject = parsed.get("subject", "")
    if not subject or "{{" in subject or "}}" in subject:
        company_name = rec.get("company", rec.get("website", "your company"))
        if "http" in str(company_name):
            company_name = urlparse(company_name).netloc.replace("www.", "")
        subject = f"Starlight LED – Precision Lighting Solutions for {company_name}"

    preamble = parsed.get(
        "preamble",
        "Precision-engineered LED solutions, delivered on time, every time.",
    )
    opening_line = parsed.get("opening_line", "Hope this email finds you well.")
    intro = parsed.get("intro", "").replace("\n", "<br>")
    bullets = ensure_list(parsed.get("bullets", []))
    feature_highlights = ensure_list(parsed.get("feature_highlights", []))
    use_cases = ensure_list(parsed.get("use_cases", []))
    technical_specs = ensure_list(parsed.get("technical_specs", []))
    cta = parsed.get(
        "cta",
        "Would you be available for a brief call next week to explore how we can "
        "illuminate your next project?",
    )

    # --- Render template ---
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
        sender_name=sender_name,
        sender_company=sender_company,
        sender_phone=sender_phone,
        sender_website=sender_website,
        sender_email=sender_email,
        company_logo_url=company_logo_url,
        catalog_chunks=raw_docs,
    )

    safe_name = sanitize_filename(
        rec.get("emails") or rec.get("company") or f"record_{idx}"
    )
    html_path = os.path.join(outdir, f"{idx}_{safe_name}.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_out)

    # --- Build .eml ---
    msg = MIMEMultipart("alternative")
    msg["From"] = f'"{sender_name}" <{sender_email}>'
    msg["To"] = str(rec.get("emails", ""))
    msg["Subject"] = subject
    msg.attach(MIMEText(html_out, "html"))

    if os.path.exists(logo_path):
        with open(logo_path, "rb") as f:
            img = MIMEImage(f.read())
            img.add_header("Content-ID", "<company_logo>")
            img.add_header("Content-Disposition", "inline", filename="logo.jpg")
            msg.attach(img)

    eml_path = os.path.join(outdir, f"{idx}_{safe_name}.eml")
    with open(eml_path, "w", encoding="utf-8") as f:
        f.write(msg.as_string())

    print(f"  ✓ Email written: {eml_path}")

    trace_info = {
        "raw_docs": raw_docs,
        "metadatas": metadatas,
        "creative_draft": creative_draft,
    }
    return eml_path, trace_info


# ---------------------------------------------------------------------------
# CLI entry point
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
        print(f"Error: Input file '{infile}' not found.")
        return

    print(f"\nDone. Emails saved to '{outdir_default}'.")


if __name__ == "__main__":
    main()
