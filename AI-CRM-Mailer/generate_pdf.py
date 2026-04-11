from fpdf import FPDF
import os

class RAGArchitecturePDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 16)
        self.cell(0, 10, 'Starlight AI CRM Mailer: RAG Architecture', 0, 1, 'C')
        self.ln(10)

    def chapter_title(self, title):
        self.set_font('Helvetica', 'B', 14)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 10, f' {title}', 0, 1, 'L', 1)
        self.ln(5)

    def chapter_body(self, body):
        self.set_font('Helvetica', '', 11)
        self.multi_cell(0, 7, body)
        self.ln()

pdf = RAGArchitecturePDF()
pdf.add_page()

# --- 1. OVERVIEW ---
pdf.chapter_title("1. Architecture Overview")
overview_text = (
    "The Starlight AI CRM Mailer uses a state-of-the-art Retrieval-Augmented Generation (RAG) "
    "architecture to bridge the gap between static catalog data and personalized client outreach. "
    "The goal is to provide 'Zero-Hallucination' recommendations by grounding the AI in the "
    "actual technical specifications of Starlight Linear LED products."
)
pdf.chapter_body(overview_text)

# --- 2. INGESTION ---
pdf.chapter_title("2. Ingestion Pipeline (rag_uploader.py)")
ingestion_text = (
    "- Multimodal Extraction: We utilize Gemini's Vision capabilities to parse complex PDF catalogs. "
    "Unlike traditional text extractors, our system reasons about tables, bullet points, and images "
    "to maintain structural integrity.\n"
    "- Semantic Chunking: The AI extracts discrete 'intelligent product chunks'. Each chunk contains "
    "the product name, series, technical specs (Voltage, CRI, CCT), and best-use applications.\n"
    "- Vectorization: Each chunk is converted into a high-dimensional vector using the 'gemini-embedding-001' model.\n"
    "- Vector Storage: These vectors are stored in ChromaDB, an open-source vector database, indexed for fast retrieval."
)
pdf.chapter_body(ingestion_text)

# --- 3. RETRIEVAL & HYDE ---
pdf.chapter_title("3. Query & Retrieval (generator_v2.py)")
retrieval_text = (
    "- Client context: Processing begins with raw scraped data from a client's website (projects, values, industry).\n"
    "- HyDE (Hypothetical Document Embeddings): To improve retrieval accuracy, we use a two-step process. "
    "First, we ask Gemini to write a 'hypothetical ideal product excerpt' for the client's needs. "
    "Second, we use this hypothetical document as the search query.\n"
    "- Vector Search: HyDE helps align the 'Sales Language' of the client with the 'Technical Language' "
    "of the catalog by searching for semantic similarity in ChromaDB.\n"
    "- Top-K Fetching: The system retrieves the Top relevant product chunks (Context) from the database."
)
pdf.chapter_body(retrieval_text)

# --- 4. GENERATION & GUARDRAILS ---
pdf.chapter_title("4. Content Generation")
generation_text = (
    "- Context Injection: The retrieved catalog chunks are injected into the final prompt as the 'Single Source of Truth'.\n"
    "- Zero-Hallucination Rules: Strict constraints prevent the AI from recommending any product or spec "
    "not found in the retrieved context.\n"
    "- Hyper-Personalization: The AI uses the client's own project names and design philosophy to frame "
    "the Starlight LED recommendations."
)
pdf.chapter_body(generation_text)

# --- 5. TECH STACK ---
pdf.chapter_title("5. Technology Stack")
tech_stack = (
    "- LLM Engine: Google Gemini 2.0 Flash\n"
    "- Embeddings: Google Gemini-Embedding-001\n"
    "- Vector Database: ChromaDB (Local Persistent)\n"
    "- Orchestration: Python / Streamlit\n"
)
pdf.chapter_body(tech_stack)

output_path = "RAG_Architecture_Starlight.pdf"
pdf.output(output_path)
print(f"PDF generated at: {os.path.abspath(output_path)}")
