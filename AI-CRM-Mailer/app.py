import streamlit as st
import pandas as pd
import os
import time
import shutil
import base64
import base64
from pathlib import Path
import chromadb # Added for admin panel

# Import your existing functions
from scrape_gemini import scrape_and_process
from generator_v2 import generate_eml_from_record
from send_eml_gsuite import send_email_gsuite
from rag_uploader import process_pdf_to_chroma

# --- Page Configuration ---
st.set_page_config(
    page_title="AI-CRM Mailer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Starlight LED Styling ---
st.markdown("""
<style>
    /* Main container matching Starlight brand */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        font-family: 'Segoe UI', Inter, sans-serif;
    }

    /* Metric Cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #ffffff, #f8fafc);
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1);
    }
    div[data-testid="stMetric"] > div {
        color: #64748b; /* Metric label */
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-size: 0.85rem;
    }
    div[data-testid="stMetric"] > div:nth-child(2) {
        color: #0f172a; /* Metric value */
        font-size: 2.5rem;
        font-weight: 700;
        margin-top: 0.5rem;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        letter-spacing: 0.5px;
        padding: 0.5rem 1rem;
        transition: all 0.2s;
    }
    
    /* Primary buttons (Green matching logo) */
    .stButton > button[kind="primary"] {
        background-color: #4CAF50;
        border: none;
        box-shadow: 0 4px 6px rgba(76, 175, 80, 0.2);
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #388E3C;
        box-shadow: 0 6px 10px rgba(76, 175, 80, 0.3);
    }
    
    /* Status Icons in Data Editor */
    .status-pending { color: #f59e0b; font-weight: 500;}
    .status-processing { color: #4CAF50; font-weight: 500;}
    .status-success { color: #10b981; font-weight: 500;}
    .status-failed { color: #ef4444; font-weight: 500;}

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #f8fafc;
        border-right: 1px solid #e2e8f0;
        padding: 1.5rem 1rem;
    }
    
    h1, h2, h3 {
        color: #0f172a;
    }
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---

def initialize_session_state():
    """Initialize session state variables."""
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    if 'run_log' not in st.session_state:
        st.session_state.run_log = []
    if 'output_dir' not in st.session_state:
        st.session_state.output_dir = "out_emails_streamlit"
    if 'latest_eml_html' not in st.session_state:
        st.session_state.latest_eml_html = ""
    if 'sender_email' not in st.session_state:
        st.session_state.sender_email = "your-email@gsuite.com"
    if 'recipient_override' not in st.session_state:
        st.session_state.recipient_override = ""
    if 'selected_template' not in st.session_state:
        st.session_state.selected_template = "email_template.html"
    if 'admin_traces' not in st.session_state:
        st.session_state.admin_traces = []

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

# --- Initialize ---
initialize_session_state()

# --- Sidebar ---
with st.sidebar:
    # Use standard logo rendering if no image is defined for Starlight LED yet. 
    # Can update this path once local image is found.
    logo_path = "starlight.jpg"
    if os.path.exists(logo_path):
        st.image(logo_path, use_container_width=True)
    else:
        st.image("https://placehold.co/400x120/003366/FFFFFF?text=STARLIGHT+LED", use_container_width=True)
    st.title("Mailer Settings")
    
    uploaded_file = st.file_uploader("Upload Client Excel File", type=["xlsx", "xls"])
    
    st.session_state.sender_email = st.text_input(
        "Your GSuite Sender Email", 
        value=st.session_state.sender_email
    )
    
    st.session_state.recipient_override = st.text_input(
        "Override Recipient Email (Optional)",
        help="If set, all generated emails will be sent to this address instead of the scraped client email. Useful for testing.",
        value=st.session_state.recipient_override
    )
    
    template_options = {
        "Modern Soft (email_template.html)": "email_template.html",
        "Minimalist Elegance (email_template_minimalist.html)": "email_template_minimalist.html",
        "Bold & Vibrant (email_template_bold.html)": "email_template_bold.html"
    }
    selected_template_label = st.selectbox(
        "Select Email Template",
        options=list(template_options.keys())
    )
    st.session_state.selected_template = template_options[selected_template_label]
    
    st.markdown("---")
    st.markdown("### 📚 Knowledge Base (RAG)")
    uploaded_files = st.file_uploader(
        "Upload PDF Catalogs", 
        type="pdf", 
        accept_multiple_files=True,
        help="Upload new product catalogs. They will be immediately analyzed and embedded to improve AI recommendations."
    )
    
    if uploaded_files and st.button("Process & Embed Catalogs", type="primary"):
        for pdf_file in uploaded_files:
            st.markdown(f"**Processing:** {pdf_file.name}")
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            
            def update_progress(fraction, message):
                progress_bar.progress(fraction)
                status_text.text(message)
                
            try:
                success, msg = process_pdf_to_chroma(pdf_file, progress_callback=update_progress)
                if success:
                    st.success(msg)
                else:
                    st.error(msg)
            except Exception as e:
                st.error(f"Error processing {pdf_file.name}: {e}")
                
            progress_bar.empty()
            status_text.empty()
            
    st.markdown("---")

    sleep_delay = st.number_input(
        "Delay Between Records (sec)", 
        min_value=1, max_value=60, value=3
    )
    
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
                # Clean or create output directory
                if os.path.exists(st.session_state.output_dir):
                    shutil.rmtree(st.session_state.output_dir)
                os.makedirs(st.session_state.output_dir)
                add_log("File loaded successfully.", "success")
        except Exception as e:
            st.error(f"Failed to read file: {e}")
            add_log(f"Failed to read file: {e}", "error")

# --- Main Page ---
if st.session_state.df is None:
    st.info("Please upload an Excel file in the sidebar to begin.")
else:
    # --- Dashboard Metrics ---
    total_records, sent_count, pending_count, failed_count = get_dashboard_metrics()
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", total_records)
    with col2:
        st.metric("Sent", f"{sent_count} ✅")
    with col3:
        st.metric("Pending", f"{pending_count} ⏳")
    with col4:
        st.metric("Failed", f"{failed_count} ❌")
    
    st.divider()

    # --- Control Buttons ---
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
    with col_btn1:
        if not st.session_state.processing:
            if st.button("🚀 Start Processing", type="primary", use_container_width=True):
                if pending_count > 0:
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
        if zip_path:
            with open(zip_path, "rb") as f:
                st.download_button(
                    "💾 Download EMLs",
                    data=f,
                    file_name="generated_emls.zip",
                    mime="application/zip",
                    use_container_width=True
                )
    
    st.divider()

    # --- Progress Bar & Status ---
    progress_bar = st.progress(0.0)
    status_text = st.empty()
    
    # --- Data Table ---
    st.subheader("Client Records")
    data_editor_placeholder = st.empty()
    data_editor_placeholder.data_editor(
        st.session_state.df,
        use_container_width=True,
        height=300,
        column_config={
            "website": st.column_config.LinkColumn("Website", width="medium"),
            "status": st.column_config.TextColumn("Status", width="small"),
            "notes": st.column_config.TextColumn("Notes", width="large")
        },
        disabled=["website", "status"]
    )
    
    # --- Logs & Email Preview ---
    tab1, tab2, tab3 = st.tabs(["Processing Log", "Latest Email Preview", "Admin Panel (RAG Traces)"])
    with tab1:
        st.code("\n".join(st.session_state.run_log), language="log")
    
    with tab2:
        if not st.session_state.latest_eml_html:
            st.info("No email has been generated yet.")
        else:
            st.components.v1.html(
                st.session_state.latest_eml_html, 
                height=500, 
                scrolling=True
            )
            
    with tab3:
        st.markdown("### ChromaDB Traces")
        if not st.session_state.admin_traces:
            st.info("No query traces available yet. Process some records to see RAG traces.")
        else:
            for i, trace in enumerate(reversed(st.session_state.admin_traces)):
                with st.expander(f"Trace {len(st.session_state.admin_traces) - i} - {trace['company']}", expanded=(i==0)):
                    st.markdown("**1. Raw Retrieved Documents (Chunks):**")
                    if trace['trace_info']['raw_docs']:
                        for j, doc in enumerate(trace['trace_info']['raw_docs']):
                            st.info(doc)
                    else:
                        st.warning("No documents retrieved.")
                        
                    st.markdown("**2. Creative Draft (Gemini raw output):**")
                    st.code(trace['trace_info']['creative_draft'], language="json")
        
        st.divider()
        st.markdown("### Existing Database Chunks")
        if st.button("Query Entire Database"):
            try:
                chroma_client = chromadb.PersistentClient(path="chroma_db")
                collection = chroma_client.get_collection("starlight_catalogs")
                results = collection.get()
                st.write(f"Total Chunks in DB: **{len(results['ids'])}**")
                if results and results['documents']:
                    db_df = pd.DataFrame({
                        "ID": results['ids'],
                        "Document": results['documents'],
                        "Metadata": [str(m) for m in (results['metadatas'] or [])]
                    })
                    st.dataframe(db_df, use_container_width=True)
            except Exception as e:
                st.error(f"Error querying ChromaDB: {e}")

    # --- Core Processing Loop ---
    if st.session_state.processing:
        pending_df = st.session_state.df[st.session_state.df['status'] == "⏳ Pending"]
        processed_count = total_records - len(pending_df)

        for index, row in pending_df.iterrows():
            if not st.session_state.processing:
                break
            
            website = row['website']
            update_status_in_df(index, "Processing...")
            
            # Update UI
            progress = (processed_count + 1) / total_records
            progress_bar.progress(progress)
            status_text.info(f"⏳ Processing ({processed_count + 1}/{total_records}): {website}")
            data_editor_placeholder.data_editor(
                st.session_state.df,
                use_container_width=True,
                height=300,
                column_config={
                    "website": st.column_config.LinkColumn("Website", width="medium"),
                    "status": st.column_config.TextColumn("Status", width="small"),
                    "notes": st.column_config.TextColumn("Notes", width="large")
                },
                disabled=["website", "status"]
            )
            
            try:
                # --- Step 1: Scrape ---
                add_log(f"Scraping: {website}")
                scraped_data = scrape_and_process(website)
                if not scraped_data:
                    raise Exception("Scraper failed to return data.")
                add_log(f"Scraped {website} successfully.", "success")
                
                # Apply Recipient Override if provided by user in the UI
                if st.session_state.recipient_override.strip():
                    scraped_data["emails"] = st.session_state.recipient_override.strip()
                    add_log(f"Overriding target email with: {scraped_data['emails']}", "info")

                # --- Step 2: Generate EML ---
                add_log(f"Generating EML for: {website} using {st.session_state.selected_template}")
                eml_path, trace_info = generate_eml_from_record(
                    scraped_data, index + 1, st.session_state.output_dir, st.session_state.selected_template
                )
                if not eml_path or not os.path.exists(eml_path):
                    raise Exception("EML generation failed.")
                
                # Store trace
                st.session_state.admin_traces.append({
                    "company": scraped_data.get('company', website),
                    "trace_info": trace_info
                })
                
                # Try to load HTML preview
                html_path = Path(eml_path).with_suffix(".html")
                if html_path.exists():
                    with open(html_path, 'r', encoding='utf-8') as f:
                        st.session_state.latest_eml_html = f.read()
                
                add_log(f"Generated EML: {eml_path}", "success")

                # --- Step 3: Send Email ---
                add_log(f"Sending email for: {website}")
                send_success = send_email_gsuite(
                    eml_path, 
                    sender_email=st.session_state.sender_email
                )
                if not send_success:
                    raise Exception("GSuite sending function returned False.")
                
                # --- Success ---
                add_log(f"Email sent successfully for: {website}", "success")
                update_status_in_df(index, "✅ Sent")

            except Exception as e:
                error_msg = f"Failed: {e}"
                add_log(f"Error processing {website}: {e}", "error")
                update_status_in_df(index, f"❌ {error_msg[:50]}...")
            
            processed_count += 1
            time.sleep(sleep_delay)

        # --- Loop Finished ---
        st.session_state.processing = False
        status_text.success("✅ Processing complete!")
        time.sleep(2)
        st.rerun()