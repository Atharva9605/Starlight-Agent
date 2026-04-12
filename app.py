"""
Starlight AI-CRM Mailer — Premium Streamlit UI.
Full UX journey: Knowledge Base → Upload Leads → Configure → Process → Review → Send.

All AI powered by Azure OpenAI GPT-4o.
Catalogues persist in ChromaDB — no re-upload needed between sessions.
"""
import streamlit as st
import pandas as pd
import os
import time
import shutil
import base64
from pathlib import Path

import chromadb

# Import project modules
from scraper import scrape_and_process
from generator_v2 import generate_eml_from_record
from send_eml_gsuite import send_email_gsuite
from rag_uploader import process_pdf_to_chroma

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CHROMA_DB_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "starlight_vision"

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Starlight AI-CRM Mailer",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Premium CSS  — Dark sidebar, glassmorphism cards, modern typography
# ---------------------------------------------------------------------------
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

  /* ── Global ── */
  html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
  }
  .main .block-container {
    padding: 1.5rem 2rem 2rem;
    max-width: 1200px;
  }

  /* ── Hide Streamlit branding ── */
  #MainMenu, footer, header { visibility: hidden; }

  /* ── Sidebar — dark premium ── */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    border-right: 1px solid rgba(63, 161, 72, 0.2);
  }
  
  [data-testid="stSidebar"] h1,
  [data-testid="stSidebar"] h2,
  [data-testid="stSidebar"] h3 {
    color: #4CAF50 !important;
    font-weight: 700 !important;
  }
  [data-testid="stSidebar"] .stSelectbox label,
  [data-testid="stSidebar"] .stTextInput label,
  [data-testid="stSidebar"] .stFileUploader label,
  [data-testid="stSidebar"] .stNumberInput label {
    color: #94a3b8 !important;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    font-size: 0.7rem;
    font-weight: 600;
  }
  
  /* Fix input field visibility */
  .stTextInput input, .stNumberInput input {
    color: #0f172a !important;
  }

  /* ── Glassmorphism metric cards ── */
  div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.5rem;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
  }
  div[data-testid="stMetric"]:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 40px rgba(63, 161, 72, 0.12);
    border-color: rgba(63, 161, 72, 0.4);
  }
  div[data-testid="stMetric"] > div:first-child {
    color: #94a3b8 !important;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-size: 0.7rem;
  }
  div[data-testid="stMetric"] > div:nth-child(2) {
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #3FA148, #81C784);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }

  /* ── Buttons — green accent ── */
  .stButton > button {
    border-radius: 10px;
    font-weight: 700;
    letter-spacing: 0.5px;
    padding: 0.6rem 1.2rem;
    transition: all 0.25s ease;
    border: 1px solid transparent;
  }
  .stButton > button[kind="primary"],
  .stButton > button:first-child {
    background: linear-gradient(135deg, #4CAF50, #2E7D32) !important;
    color: #ffffff !important;
    border: none;
    box-shadow: 0 4px 14px rgba(76, 175, 80, 0.3);
  }
  .stButton > button[kind="primary"]:hover,
  .stButton > button:first-child:hover {
    box-shadow: 0 6px 20px rgba(76, 175, 80, 0.5);
    transform: translateY(-1px);
  }

  /* ── Tabs — underline style ── */
  .stTabs [data-baseweb="tab-list"] {
    gap: 0;
    border-bottom: 2px solid #1e293b;
  }
  .stTabs [data-baseweb="tab"] {
    font-weight: 600;
    letter-spacing: 0.5px;
    padding: 0.8rem 1.5rem;
    border-bottom: 3px solid transparent;
    transition: all 0.2s;
  }
  .stTabs [aria-selected="true"] {
    border-bottom-color: #4CAF50 !important;
    color: #4CAF50 !important;
  }

  /* ── Data editor ── */
  [data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
  }

  /* ── Knowledge Base status cards ── */
  .kb-card {
    background: linear-gradient(135deg, rgba(76, 175, 80, 0.08), rgba(76, 175, 80, 0.02));
    border: 1px solid rgba(76, 175, 80, 0.25);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.5rem;
  }
  .kb-card h4 {
    margin: 0 0 4px;
    color: #4CAF50;
    font-size: 0.85rem;
    font-weight: 700;
  }
  .kb-card p {
    margin: 0;
    color: #94a3b8;
    font-size: 0.75rem;
  }

  /* ── Hero banner ── */
  .hero-banner {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 1.5rem;
    border: 1px solid rgba(76, 175, 80, 0.2);
    position: relative;
    overflow: hidden;
  }
  .hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(76, 175, 80, 0.1) 0%, transparent 70%);
    border-radius: 50%;
  }
  .hero-banner h1 {
    color: #f8fafc;
    font-size: 1.8rem;
    font-weight: 800;
    margin: 0 0 0.3rem;
    letter-spacing: -0.5px;
  }
  .hero-banner p {
    color: #94a3b8;
    font-size: 0.95rem;
    margin: 0;
  }
  .hero-banner .accent {
    color: #4CAF50;
    font-weight: 700;
  }

  /* ── Status badges ── */
  .status-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.5px;
  }
  .badge-ready { background: rgba(16,185,129,0.15); color: #10b981; }
  .badge-empty { background: rgba(239,68,68,0.15); color: #ef4444; }
  .badge-processing { background: rgba(76, 175, 80, 0.15); color: #4CAF50; }

  /* ── Step indicator ── */
  .step-indicator {
    display: flex;
    gap: 0;
    margin: 1.5rem 0;
  }
  .step-item {
    flex: 1;
    text-align: center;
    padding: 0.8rem 0.5rem;
    position: relative;
    font-size: 0.75rem;
    font-weight: 600;
    color: #475569;
    letter-spacing: 0.5px;
  }
  .step-item.active {
    color: #4CAF50;
  }
  .step-item.done {
    color: #10b981;
  }
  .step-item::after {
    content: '';
    position: absolute;
    bottom: 0;
    left: 10%;
    width: 80%;
    height: 3px;
    background: #1e293b;
    border-radius: 2px;
  }
  .step-item.active::after {
    background: linear-gradient(90deg, #3FA148, #81C784);
    box-shadow: 0 0 8px rgba(76, 175, 80, 0.5);
  }
  .step-item.done::after {
    background: #10b981;
  }

  /* ── Expander ── */
  .streamlit-expanderHeader {
    font-weight: 600 !important;
    font-size: 0.9rem;
  }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def get_kb_status():
    """Get Knowledge Base status from ChromaDB."""
    try:
        chroma = chromadb.PersistentClient(path=CHROMA_DB_DIR)
        collection = chroma.get_collection(COLLECTION_NAME)
        count = collection.count()
        # Get unique catalogue names
        if count > 0:
            results = collection.get(limit=min(count, 500), include=["metadatas"])
            catalogues = set()
            for m in (results.get("metadatas") or []):
                name = m.get("catalogue_name", "")
                if name:
                    catalogues.add(name)
            return count, list(catalogues)
        return count, []
    except Exception:
        return 0, []


def initialize_session_state():
    """Initialize session state variables."""
    defaults = {
        "df": None,
        "processing": False,
        "run_log": [],
        "output_dir": "out_emails_streamlit",
        "latest_eml_html": "",
        "sender_email": os.getenv("GSUITE_DELEGATED_USER", "your-email@gsuite.com"),
        "recipient_override": "",
        "selected_template": "email_template.html",
        "admin_traces": [],
        "current_step": 0,   # 0=setup, 1=processing, 2=review, 3=done
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def add_log(message, level="info"):
    """Add a message to the run log."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    icon_map = {"info": "ℹ️", "success": "✅", "warning": "⚠️", "error": "❌"}
    icon = icon_map.get(level, "ℹ️")
    st.session_state.run_log.insert(0, f"{timestamp} - {icon} {message}")


def update_status_in_df(index, status):
    """Update the status for a specific row in the DataFrame."""
    if st.session_state.df is not None:
        st.session_state.df.loc[index, 'status'] = status


def get_dashboard_metrics():
    """Calculate dashboard metrics from the DataFrame."""
    if st.session_state.df is None:
        return 0, 0, 0, 0
    total = len(st.session_state.df)
    sent = st.session_state.df['status'].str.contains("Sent").sum()
    failed = st.session_state.df['status'].str.contains("Failed").sum()
    pending = total - sent - failed
    return total, sent, pending, failed


def create_download_zip():
    """Create a zip file of the EMLs and return its path."""
    if not os.path.exists(st.session_state.output_dir):
        return None
    zip_path = "generated_emls"
    shutil.make_archive(zip_path, 'zip', st.session_state.output_dir)
    return f"{zip_path}.zip"


# ---------------------------------------------------------------------------
# Initialize
# ---------------------------------------------------------------------------
initialize_session_state()


# ---------------------------------------------------------------------------
# SIDEBAR — Dark premium panel
# ---------------------------------------------------------------------------
with st.sidebar:
    # Logo
    logo_path = "starlight.jpg"
    if os.path.exists(logo_path):
        st.image(logo_path, use_container_width=True)
    else:
        st.markdown(
            '<div style="text-align:center; padding:1rem 0;">'
            '<span style="font-size:1.5rem; font-weight:800; color:#4CAF50;">✦ STARLIGHT LED</span>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Knowledge Base Status (PERSISTENT) ──
    st.markdown("### 📚 Knowledge Base")

    kb_chunks, kb_catalogues = get_kb_status()

    if kb_chunks > 0:
        st.markdown(
            f'<div class="kb-card">'
            f'<h4>✅ Database Active</h4>'
            f'<p><b>{kb_chunks}</b> product chunks indexed</p>'
            f'</div>',
            unsafe_allow_html=True,
        )
        for cat in kb_catalogues:
            st.markdown(
                f'<div class="kb-card">'
                f'<h4>📄 {cat}</h4>'
                f'<p>Indexed in ChromaDB</p>'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<div class="kb-card">'
            '<h4>⚠️ Empty Database</h4>'
            '<p>Upload PDF catalogues below to populate the knowledge base.</p>'
            '</div>',
            unsafe_allow_html=True,
        )

    # Upload new catalogues
    with st.expander("➕ Add New Catalogues", expanded=(kb_chunks == 0)):
        uploaded_pdfs = st.file_uploader(
            "Upload PDF Catalogues",
            type="pdf",
            accept_multiple_files=True,
            help="Upload Starlight product catalogues. They are permanently stored — no need to re-upload.",
            key="pdf_uploader",
        )

        if uploaded_pdfs and st.button("🔄 Process & Embed", type="primary", use_container_width=True):
            for pdf_file in uploaded_pdfs:
                with st.status(f"Processing: {pdf_file.name}", expanded=True) as status:
                    progress_bar = st.progress(0.0)
                    status_text = st.empty()

                    def update_progress(fraction, message):
                        progress_bar.progress(min(fraction, 1.0))
                        status_text.text(message)

                    try:
                        success, msg = process_pdf_to_chroma(pdf_file, progress_callback=update_progress)
                        if success:
                            status.update(label=f"✅ {pdf_file.name}", state="complete")
                            st.success(msg)
                        else:
                            status.update(label=f"❌ {pdf_file.name}", state="error")
                            st.error(msg)
                    except Exception as e:
                        status.update(label=f"❌ {pdf_file.name}", state="error")
                        st.error(f"Error: {e}")

            st.rerun()

    # ── Clear Database ──
    if kb_chunks > 0:
        with st.expander("🗑️ Database Admin"):
            st.caption("⚠️ This will delete ALL indexed catalogue data.")
            if st.button("Clear Entire Database", type="secondary"):
                try:
                    chroma = chromadb.PersistentClient(path=CHROMA_DB_DIR)
                    chroma.delete_collection(COLLECTION_NAME)
                    st.success("Database cleared. Refresh to see changes.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

    st.markdown("---")

    # ── Mailer Settings ──
    st.markdown("### ⚙️ Mailer Settings")

    uploaded_file = st.file_uploader(
        "Upload Client Excel File",
        type=["xlsx", "xls"],
        key="excel_uploader",
    )

    st.session_state.sender_email = st.text_input(
        "GSuite Sender Email",
        value=st.session_state.sender_email,
    )

    st.session_state.recipient_override = st.text_input(
        "Override Recipient (Testing)",
        help="All emails go to this address instead of scraped client email.",
        value=st.session_state.recipient_override,
    )

    template_options = {
        "✨ Modern Soft": "email_template.html",
        "🎯 Minimalist": "email_template_minimalist.html",
        "🔥 Bold & Vibrant": "email_template_bold.html",
    }
    selected_template_label = st.selectbox(
        "Email Template",
        options=list(template_options.keys()),
    )
    st.session_state.selected_template = template_options[selected_template_label]

    sleep_delay = st.number_input(
        "Delay Between Records (sec)",
        min_value=1, max_value=60, value=3,
    )

    # Load Excel
    if uploaded_file is not None and st.session_state.df is None:
        try:
            df = pd.read_excel(uploaded_file)
            if 'website' not in df.columns:
                st.error("Excel file must have a 'website' column.")
            else:
                df['status'] = "⏳ Pending"
                df['notes'] = ""
                st.session_state.df = df
                st.session_state.run_log = []
                st.session_state.latest_eml_html = ""
                st.session_state.current_step = 0
                if os.path.exists(st.session_state.output_dir):
                    shutil.rmtree(st.session_state.output_dir)
                os.makedirs(st.session_state.output_dir)
                add_log("File loaded successfully.", "success")
        except Exception as e:
            st.error(f"Failed to read file: {e}")
            add_log(f"Failed to read file: {e}", "error")

    st.markdown("---")
    st.caption("Powered by Azure OpenAI GPT-4o")


# ---------------------------------------------------------------------------
# MAIN PAGE
# ---------------------------------------------------------------------------

# ── Hero Banner ──
st.markdown(
    '<div class="hero-banner">'
    '<h1>✦ Starlight <span class="accent">AI-CRM</span> Mailer</h1>'
    '<p>Intelligent cold outreach powered by <span class="accent">Azure OpenAI GPT-4o</span> '
    '&amp; RAG-grounded catalogue knowledge</p>'
    '</div>',
    unsafe_allow_html=True,
)

if st.session_state.df is None:
    # ── Empty state — onboarding ──
    col_info1, col_info2, col_info3 = st.columns(3)

    with col_info1:
        st.markdown("### 📚 Step 1: Knowledge Base")
        if kb_chunks > 0:
            st.success(f"✅ {kb_chunks} product chunks ready")
        else:
            st.warning("Upload catalogues in the sidebar →")

    with col_info2:
        st.markdown("### 📋 Step 2: Upload Leads")
        st.info("Upload an Excel file with a `website` column in the sidebar →")

    with col_info3:
        st.markdown("### 🚀 Step 3: Process & Send")
        st.info("Click 'Start Processing' to scrape, generate, and send emails")

else:
    # ── Step Indicator ──
    total_records, sent_count, pending_count, failed_count = get_dashboard_metrics()
    processing_active = st.session_state.processing

    # Determine step
    if sent_count == total_records and total_records > 0:
        current_step = 3
    elif processing_active:
        current_step = 1
    elif sent_count > 0 or failed_count > 0:
        current_step = 2
    else:
        current_step = 0

    step_labels = ["Setup", "Processing", "Review", "Complete"]
    step_html = '<div class="step-indicator">'
    for i, label in enumerate(step_labels):
        cls = "done" if i < current_step else ("active" if i == current_step else "")
        icon = "✓" if i < current_step else str(i + 1)
        step_html += f'<div class="step-item {cls}">{icon}. {label}</div>'
    step_html += '</div>'
    st.markdown(step_html, unsafe_allow_html=True)

    # ── Dashboard Metrics ──
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Leads", total_records)
    with col2:
        st.metric("Sent", sent_count)
    with col3:
        st.metric("Pending", pending_count)
    with col4:
        st.metric("Failed", failed_count)
    with col5:
        st.metric("Knowledge Base", f"{kb_chunks} chunks")

    st.divider()

    # ── Control Buttons ──
    col_btn1, col_btn2, col_btn3, col_btn4 = st.columns([1.2, 1, 1, 1.5])

    with col_btn1:
        if not st.session_state.processing:
            if st.button("🚀 Start Processing", type="primary", use_container_width=True):
                if pending_count > 0:
                    if kb_chunks == 0:
                        st.toast("⚠️ No catalogues uploaded. Emails will have generic content.", icon="⚠️")
                    st.session_state.processing = True
                    st.rerun()
                else:
                    st.toast("No pending records to process.")
        else:
            if st.button("🛑 Stop Processing", use_container_width=True):
                st.session_state.processing = False
                add_log("Processing stopped by user.", "warning")
                st.rerun()

    with col_btn2:
        zip_path = create_download_zip()
        if zip_path and os.path.exists(zip_path):
            with open(zip_path, "rb") as f:
                st.download_button(
                    "💾 Download EMLs",
                    data=f,
                    file_name="generated_emls.zip",
                    mime="application/zip",
                    use_container_width=True,
                )

    with col_btn3:
        if st.button("🔄 Reset Queue", use_container_width=True):
            st.session_state.df = None
            st.session_state.run_log = []
            st.session_state.latest_eml_html = ""
            st.session_state.admin_traces = []
            st.rerun()

    st.divider()

    # ── Progress Bar & Status ──
    progress_bar = st.progress(0.0)
    status_text = st.empty()

    # ── Tabs ──
    tab_records, tab_log, tab_preview, tab_admin = st.tabs([
        "📋 Client Records",
        "📜 Processing Log",
        "👁️ Email Preview",
        "🔬 RAG Admin",
    ])

    with tab_records:
        data_editor_placeholder = st.empty()
        data_editor_placeholder.data_editor(
            st.session_state.df,
            use_container_width=True,
            height=350,
            column_config={
                "website": st.column_config.LinkColumn("Website", width="medium"),
                "status": st.column_config.TextColumn("Status", width="small"),
                "notes": st.column_config.TextColumn("Notes", width="large"),
            },
            disabled=["website", "status"],
        )

    with tab_log:
        if st.session_state.run_log:
            log_text = "\n".join(st.session_state.run_log)
            st.code(log_text, language="log")
        else:
            st.info("Processing log will appear here once you start.")

    with tab_preview:
        if not st.session_state.latest_eml_html:
            st.info("No email generated yet. Start processing to see a live preview.")
        else:
            st.components.v1.html(
                st.session_state.latest_eml_html,
                height=550,
                scrolling=True,
            )

    with tab_admin:
        st.markdown("### 🔎 RAG Trace Inspector")

        if not st.session_state.admin_traces:
            st.info("No query traces available yet. Process some records to see RAG traces.")
        else:
            for i, trace in enumerate(reversed(st.session_state.admin_traces)):
                with st.expander(
                    f"Trace {len(st.session_state.admin_traces) - i} — {trace['company']}",
                    expanded=(i == 0),
                ):
                    st.markdown("**1. Retrieved Catalogue Chunks:**")
                    if trace['trace_info']['raw_docs']:
                        for j, doc in enumerate(trace['trace_info']['raw_docs']):
                            st.info(doc)
                    else:
                        st.warning("No documents retrieved from knowledge base.")

                    st.markdown("**2. Creative Draft (Azure OpenAI GPT-4o output):**")
                    st.code(trace['trace_info']['creative_draft'], language="json")

        st.divider()

        st.markdown("### 📊 Database Overview")
        if st.button("Query Entire Database"):
            try:
                chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
                collection = chroma_client.get_collection(COLLECTION_NAME)
                results = collection.get()
                st.write(f"Total Chunks in DB: **{len(results['ids'])}**")
                if results and results['documents']:
                    db_df = pd.DataFrame({
                        "ID": results['ids'],
                        "Document": results['documents'],
                        "Metadata": [str(m) for m in (results['metadatas'] or [])],
                    })
                    st.dataframe(db_df, use_container_width=True, height=400)
            except Exception as e:
                st.error(f"Error querying ChromaDB: {e}")

    # ── Core Processing Loop ──
    if st.session_state.processing:
        pending_df = st.session_state.df[st.session_state.df['status'] == "⏳ Pending"]
        processed_count = total_records - len(pending_df)

        for index, row in pending_df.iterrows():
            if not st.session_state.processing:
                break

            website = row['website']
            update_status_in_df(index, "⚙️ Processing...")

            # Update UI
            progress = (processed_count + 1) / total_records
            progress_bar.progress(progress)
            status_text.info(f"⏳ Processing ({processed_count + 1}/{total_records}): {website}")

            try:
                # Step 1: Scrape
                add_log(f"Scraping: {website}")
                scraped_data = scrape_and_process(website)
                if not scraped_data:
                    raise Exception("Scraper failed to return data.")
                add_log(f"Scraped {website} successfully.", "success")

                # Apply Recipient Override
                if st.session_state.recipient_override.strip():
                    scraped_data["emails"] = st.session_state.recipient_override.strip()
                    add_log(f"Overriding target email with: {scraped_data['emails']}", "info")

                # Step 2: Generate EML
                add_log(f"Generating EML for: {website} using {st.session_state.selected_template}")
                eml_path, trace_info = generate_eml_from_record(
                    scraped_data, index + 1, st.session_state.output_dir, st.session_state.selected_template
                )
                if not eml_path or not os.path.exists(eml_path):
                    raise Exception("EML generation failed.")

                # Store trace
                st.session_state.admin_traces.append({
                    "company": scraped_data.get('company', website),
                    "trace_info": trace_info,
                })

                # Load HTML preview
                html_path = Path(eml_path).with_suffix(".html")
                if html_path.exists():
                    with open(html_path, 'r', encoding='utf-8') as f:
                        html_content = f.read()
                    
                    try:
                        import base64
                        if os.path.exists("starlight.jpg"):
                            with open("starlight.jpg", "rb") as img_f:
                                b64 = base64.b64encode(img_f.read()).decode()
                            html_content = html_content.replace('cid:company_logo', f'data:image/jpeg;base64,{b64}')
                    except Exception:
                        pass
                        
                    st.session_state.latest_eml_html = html_content

                add_log(f"Generated EML: {eml_path}", "success")

                # Step 3: Send Email
                add_log(f"Sending email for: {website}")
                send_success = send_email_gsuite(
                    eml_path,
                    sender_email=st.session_state.sender_email,
                )
                if not send_success:
                    raise Exception("GSuite sending function returned False.")

                # Success
                add_log(f"Email sent successfully for: {website}", "success")
                update_status_in_df(index, "✅ Sent")

            except Exception as e:
                error_msg = f"Failed: {e}"
                add_log(f"Error processing {website}: {e}", "error")
                update_status_in_df(index, f"❌ {str(e)[:50]}...")

            processed_count += 1
            time.sleep(sleep_delay)

        # Loop Finished
        st.session_state.processing = False
        status_text.success("✅ Processing complete!")
        time.sleep(2)
        st.rerun()