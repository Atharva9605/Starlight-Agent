import os
import sys
import json
import time
import uuid
import re
from datetime import datetime, timedelta
from dateutil import tz
from urllib.parse import urlparse  # Import urlparse
from dotenv import load_dotenv
from google import genai
from jinja2 import Environment, FileSystemLoader
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage

# --- Configuration ---
load_dotenv()
from key_manager import with_key_rotation, key_manager

# --- ChromaDB Setup ---
import chromadb
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Helper to retrieve context
@with_key_rotation
def get_rag_context(client_desc: str, k: int = 5) -> tuple[str, list, list]:
    chroma_client = chromadb.PersistentClient(path="chroma_db")
    collection = chroma_client.get_collection("starlight_catalogs")
    current_api_key = os.environ.get("GEMINI_API_KEY")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=current_api_key
    )
    # --- HyDE: Generate hypothetical product match ---
    client = key_manager.get_client()
    hyde_prompt = f"""
    You are an expert sales engineer for Starlight Linear LED.
    A client has the following profile and needs: {client_desc}
    
    Write a hypothetical excerpt from a Starlight product catalog that perfectly describes the ideal lighting product to solve this client's specific needs. Include hypothetical product features, forms, and technical specs that would be relevant. 
    Keep it concise. Do not include any conversational filler.
    """
    resp = client.models.generate_content(
        model=key_manager.get_current_model(),
        contents=hyde_prompt,
        config={"max_output_tokens": 1024}
    )
    hyde_doc = resp.text
    combined_query = f"Client Context:\\n{client_desc}\\n\\nIdeal Product Characteristics:\\n{hyde_doc}"
    
    print(f"Executing HyDE Search for Client...")
    # Embed the enhanced query
    query_embedding = embeddings.embed_query(combined_query)
    
    # Query Chroma
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )
    
    if results and results['documents'] and results['documents'][0]:
        return "\n\n---\n\n".join(results['documents'][0]), results['documents'][0], results['metadatas'][0] if 'metadatas' in results else []
    return "No specific catalog context found.", [], []


# --- Jinja2 Template Helper ---
def get_template(template_name="email_template.html"):
    try:
        env = Environment(loader=FileSystemLoader("templates"))
        return env.get_template(template_name)
    except Exception as e:
        print(f"Warning: Could not load Jinja2 template '{template_name}'. {e}")
        # Create a basic fallback template
        env = Environment(loader=FileSystemLoader("."))
        return env.from_string("""
        <html><body>
        <p>{{ intro_paragraph | safe }}</p>
        <ul>
        {% for item in bullets %}
            <li>{{ item }}</li>
        {% endfor %}
        </ul>
        <p>{{ closing_paragraph }}</p>
        <p>
        --<br>
        {{ sender_name }}<br>
        {{ sender_company }}
        </p>
        </body></html>
        """)


# --- File and Directory Setup ---
infile = sys.argv[1] if len(sys.argv) > 1 else "scraped_results.jsonl"
outdir_default = sys.argv[2] if len(sys.argv) > 2 else "out_emails_gemini_v2"
os.makedirs(outdir_default, exist_ok=True)

# --- Sender Information ---
sender_name = os.getenv("SENDER_NAME", "Vivek Dhondarkar")
sender_company = os.getenv("SENDER_COMPANY", "Starlight Linear LED")
sender_phone = os.getenv("SENDER_PHONE", "9619436066")
sender_website = os.getenv("SENDER_WEBSITE", "www.starlightlinearled.com")
sender_email = os.getenv("SENDER_EMAIL", "vivek@starlightlinearled.com")
company_logo_url = os.getenv("COMPANY_LOGO_URL", "cid:company_logo")
logo_path = r"AI-CRM-Mailer\starlight.jpg"

# --- Helper and AI Generation Functions ---
def extract_json(txt):
    if not txt:
        return None
    try:
        start = txt.find('{')
        end = txt.rfind('}')
        if start != -1 and end != -1 and end > start:
            json_str = txt[start:end+1]
            return json.loads(json_str)
    except (json.JSONDecodeError, IndexError):
        print("Warning: Failed to extract JSON from response.")
        return None
    return None

def sanitize_filename(s):
    return re.sub(r'[^a-zA-Z0-9_\-@.]', '_', s or "no_email_found")

@with_key_rotation
def generate_creative_draft(rec):
    client = key_manager.get_client()
    client_info_str = json.dumps(rec)
    
    # Retrieve relevant context from ChromaDB based on client record
    print(f"Retrieving RAG context for client: {rec.get('company', 'Unknown')}")
    rag_context, raw_docs, metadatas = get_rag_context(client_info_str)
    
    prompt = f"""
Company Context (Starlight Linear LED):
Starlight Linear LED is a manufacturer and supplier of linear LED modules and products based in Mumbai, India. We provide end-to-end lighting solutions, from concept and customized manufacturing to installation and maintenance.
Vision: To be a leading manufacturer of innovative linear lighting solutions.
Mission: To create customized LED lighting solutions to assist clients in achieving their desired lighting impacts.
"We're proud to announce that Starlight LED Lighting has been selected by the prestigious Homes India MAGAZINE as one of the 'Top 10 Brands in Lightings for 2025.'"

--- RELEVANT CATALOG CONTEXT (Retrieved via RAG) ---
{rag_context}
-----------------------------------------------------

Using the above context about Starlight Linear LED and the retrieved catalog information, tailor the email to the client's needs.
It is important to include only those products that are relevant to the client's industry and projects.

Contact Information
Thane (Head Office)
Address: 3 Vedant 3, P & T COLONY, Gandhi Nagar, Dombivali – East, Thane 421203
Phone: 9619436066
Email: vivek@starlightlinearled.com

STRICT INSTRUCTIONS & ANTI-HALLUCINATION RULES:
- HYPER-PERSONALIZATION REQUIRED: You must thoroughly read the client's information below. You MUST explicitly name-drop 1-2 of their exact, specific projects, previous clients, or unique portfolio items to prove we have deeply researched their firm. DO NOT use generic phrases like "your recent projects".
- Subtly and intelligently connect these specific projects to Starlight's capabilities.
- ZERO HALLUCINATIONS: You MUST ONLY recommend product names, features, or technical specs that are explicitly and directly mentioned in the "RELEVANT CATALOG CONTEXT" above. Do NOT invent, assume, or fabricate any Starlight LED products. If the context does not explicitly list a product suitable for the client, rely on general Starlight capabilities (e.g. "custom linear LED manufacturing") instead of making up a product/component name.
- The subject line must be catchy, relevant to the client's company and projects, and NOT use placeholders like {{{{subject}}}}. Do NOT include the client's name in the subject, just make it related to their industry/needs.
- The intro must directly address the client's unique approach, values, or recent projects, and explain how Starlight's solutions are a perfect fit.
- Every section (bullets, features, use cases, specs, CTA) must be tailored to the client's context, not generic.
- Explicitly connect Starlight's TRUE catalog offerings to the client's needs, project types, and design philosophy based ONLY on the retrieved context.
- Do NOT use bold, headings, or extra formatting.
- The closing paragraph must be a short call to action only (no sender details).
- Do NOT repeat sender details in the closing; they will be in the signature.
- Keep the email VERY CONCISE, spread out, and punchy. Limit the intro to 1-2 short sentences. Do NOT write long paragraphs. Nobody reads long cramped emails.
- ALWAYS FINISH YOUR SENTENCES AND PARAGRAPHS. DO NOT TRUNCATE OR STOP RANDOMLY ABORTING THE OUTPUT.
- Output extremely concise feature highlights (under 10 words each) that fit well within a card-like or icon-driven layout.
- Each item in 'bullets', 'feature_highlights', 'use_cases', and 'technical_specs' MUST be a plain string primitive. Use HTML <b> tags for titles if needed, like: "<b>Short Title:</b> One sentence desc." DO NOT output dictionaries or JSON objects inside these lists.
- Max 2-3 features in an email to keep it short.
- The reading flow must be smooth, logical, and skimmable. Ensure complete sentences after bullets.
- The parts of the email rendered in HTML should not be overlapping.
- The Dear.. section should be personalized based on the client's data (e.g., Dear Mahim Architects Team).
- After the Dear.. section, add a newline, greet the client, and give a short one-line intro about Starlight Linear LED (e.g., "Hope this email finds you well...").
- Weave the "Why choose Starlight Linear LED" (specifically our world-class on-time delivery) naturally into the content.
- Every email MUST concisely contain the "Homes India MAGAZINE Top 10 Brands in Lightings for 2025" award info seamlessly in the text.

Client data (JSON): {client_info_str}

Output JSON only, with these exact keys:
- "subject"
- "preamble"
- "opening_line"
- "intro"
- "bullets"
- "feature_highlights"
- "use_cases"
- "technical_specs"
- "cta"

Do not include any extra text, markdown, or HTML outside the JSON. Ensure JSON is valid and properly closed.
"""
    # Removed broad try-except to allow @with_key_rotation to catch quota errors
    resp = client.models.generate_content(
        model=key_manager.get_current_model(),
        contents=prompt,
        config={"max_output_tokens": 4096}
    )
    if not resp or not resp.text:
        raise Exception("AI failed to generate a creative draft.")
        
    return resp.text, raw_docs, metadatas

@with_key_rotation
def refine_with_judge(draft_text):
    if not draft_text: return None
    client = key_manager.get_client()
    prompt = f"""
You are an expert data formatting assistant. Your task is to take the provided raw text containing a draft email and perfectly format it into a valid JSON object.

RULES:
1.  Parse the incoming text and identify all the required fields: "subject", "preamble", "opening_line", "intro", "bullets", "feature_highlights", "use_cases", "technical_specs", "cta".
2.  Clean up any conversational text or explanations.
3.  Ensure that fields intended to be lists ("bullets", "feature_highlights", "use_cases", "technical_specs") are formatted as JSON arrays of string primitives ONLY. If the input contains objects/dicts in these lists, convert them into formatted strings (e.g., "<b>Title:</b> Description").
4.  The final output MUST be ONLY a single, valid JSON object.
5.  Make sure there are no formatting inconsistencies in the HTML-rendered sections.

Raw draft text to process:
---
{draft_text}
---
"""
    # Removed broad try-except to allow decorator to handle rotation
    resp = client.models.generate_content(
        model=key_manager.get_current_model(),
        contents=prompt,
        config={"max_output_tokens": 4096}
    )
    
    parsed = extract_json(resp.text)
    if not parsed:
        raise Exception("AI failed to refine/format the draft into JSON.")
    return parsed

def clean_item(x):
    if isinstance(x, dict):
        title = x.get('title', '')
        desc = x.get('description', x.get('desc', ''))
        if title and desc:
            return f"<b>{title}</b><br>{desc}"
        return str(x)
    if isinstance(x, str):
        x = x.strip()
        if x.startswith("{") and x.endswith("}"):
            import ast
            try:
                d = ast.literal_eval(x)
                if isinstance(d, dict):
                    title = d.get('title', '')
                    desc = d.get('description', d.get('desc', ''))
                    if title and desc:
                        return f"<b>{title}</b><br>{desc}"
            except:
                pass
        x = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", x)
    return x

def ensure_list(val):
    if isinstance(val, list):
        return [clean_item(x) for x in val]
    if isinstance(val, str):
        val = val.strip()
        if not val: return []
        items = [v.strip() for v in val.splitlines() if v.strip()] if "\n" in val else [val]
        return [clean_item(x) for x in items]
    return []

# --- NEW Function for Pipeline ---
def generate_eml_from_record(rec, idx, outdir, template_name="email_template.html"):
    """Generates a single .eml file from a scraped record."""
    if not rec:
        print(f"Error: Received empty record for index {idx}")
        return None
        
    print(f"\nProcessing record {idx} using template {template_name}...")
    
    creative_draft, raw_docs, metadatas = generate_creative_draft(rec)
    if not creative_draft:
        raise Exception("AI Generation failed: Could not create initial draft.")
        
    parsed = refine_with_judge(creative_draft)
    if not parsed:
        raise Exception("AI Generation failed: Could not parse draft into structured JSON.")

    # --- VALIDATION & FALLBACK LOGIC FOR SUBJECT ---
    subject = parsed.get("subject")
    
    # Check for invalid or placeholder subjects returned by the AI
    if not subject or "{{" in subject or "}}" in subject:
        print(f"Warning: Invalid or placeholder subject generated for {rec.get('company')}. Using fallback.")
        company_name = rec.get('company', rec.get('website', 'your company'))
        if 'http' in str(company_name):
             company_name = urlparse(company_name).netloc.replace("www.", "")
        subject = f"Starlight LED Lighting – Tailored Solutions for {company_name}"
    # --- END OF VALIDATION BLOCK ---

    preamble = parsed.get("preamble", "Bring unparalleled precision and brilliance to your projects with Starlight LED Lighting.")
    opening_line = parsed.get("opening_line", "Hope this email finds you well.")
    intro = parsed.get("intro", "").replace("\n", "<br>")
    bullets = ensure_list(parsed.get("bullets", []))
    feature_highlights = ensure_list(parsed.get("feature_highlights", []))
    use_cases = ensure_list(parsed.get("use_cases", []))
    technical_specs = ensure_list(parsed.get("technical_specs", []))
    cta = parsed.get("cta", f"I would welcome the opportunity to discuss how we can illuminate your next project. Are you available for a brief call next week?")

    template = get_template(template_name)
    html_out = template.render(
        subject=subject, preamble=preamble, opening_line=opening_line,
        intro_paragraph=intro, bullets=bullets, feature_highlights=feature_highlights,
        use_cases=use_cases, technical_specs=technical_specs, closing_paragraph=cta,
        sender_name=sender_name, sender_company=sender_company, sender_phone=sender_phone,
        sender_website=sender_website, sender_email=sender_email, company_logo_url=company_logo_url,
        catalog_chunks=raw_docs  # Pass the raw chunks to the template
    )

    safe_email_name = sanitize_filename(rec.get("emails") or rec.get("company") or f"record_{idx}")
    html_path = os.path.join(outdir, f"{idx}_{safe_email_name}.html")
    with open(html_path, "w", encoding="utf-8") as g:
        g.write(html_out)

    msg = MIMEMultipart("alternative")
    msg["From"] = f'"{sender_name}" <{sender_email}>'
    msg["To"] = str(rec.get("emails", ""))
    msg["Subject"] = str(subject)
    
    # Create the HTML part
    html_part = MIMEText(html_out, "html")
    msg.attach(html_part)

    # Embed the logo image if it exists
    if os.path.exists(logo_path):
        with open(logo_path, 'rb') as f:
            img_data = f.read()
            msg_img = MIMEImage(img_data)
            msg_img.add_header('Content-ID', '<company_logo>')
            msg_img.add_header('Content-Disposition', 'inline', filename="logo.jpg")
            msg.attach(msg_img)

    eml_path = os.path.join(outdir, f"{idx}_{safe_email_name}.eml")
    with open(eml_path, "w", encoding="utf-8") as h:
        h.write(msg.as_string())
    
    print(f"-> Generated email for {rec.get('company')}: {eml_path}")
    
    # Return path and trace context
    trace_info = {
        "raw_docs": raw_docs,
        "metadatas": metadatas,
        "creative_draft": creative_draft
    }
    return eml_path, trace_info

# --- Main Execution Loop (Modified) ---
def main():
    """Main function to process clients and GENERATE email files."""
    try:
        with open(infile, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f, 1):
                if not line.strip(): continue
                
                rec = None
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    print(f"Warning: Skipping malformed JSON on line {idx}")
                    continue
                
                # Call the refactored function
                generate_eml_from_record(rec, idx, outdir_default)
                
                time.sleep(1) # Be respectful of API rate limits

    except FileNotFoundError:
        print(f"Error: Input file '{infile}' not found. Please create it.")
        return

    print(f"\nEmail generation complete. Files are in the '{outdir_default}' directory.")

if __name__ == "__main__":
    main()