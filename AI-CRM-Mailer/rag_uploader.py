import os
import time
import json
import tempfile
import chromadb
from google import genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from key_manager import with_key_rotation, key_manager

CHROMA_DB_DIR = "chroma_db"
COLLECTION_NAME = "starlight_catalogs"

@with_key_rotation
def process_pdf_to_chroma(pdf_file, progress_callback=None):
    """
    Intelligently parses a complex PDF (with tables and images) using Gemini's native File API,
    extracts semantic product chunks, and saves embeddings into ChromaDB.
    """
    if progress_callback:
        progress_callback(0.05, "Saving PDF temporarily...")
        
    temp_pdf_path = ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_file.read())
        temp_pdf_path = tmp.name
        
    client = key_manager.get_client()
    uploaded = None
    try:
        if progress_callback:
            progress_callback(0.1, "Uploading Catalog to AI Vision Engine...")
            
        # New SDK upload syntax
        with open(temp_pdf_path, "rb") as f:
            uploaded = client.files.upload(file=f, config={"mime_type": "application/pdf"})
        
        # New SDK file status check
        while uploaded.state == "PROCESSING":
            time.sleep(2)
            uploaded = client.files.get(name=uploaded.name)
            
        if uploaded.state == "FAILED":
            raise ValueError("Gemini failed to process the PDF.")
            
        if progress_callback:
            progress_callback(0.3, "AI Semantic Chunking (Analyzing tables and product features)...")
            
        extraction_prompt = """
        You are an expert AI extraction system for lighting catalogs.
        Analyze this catalog thoroughly. The catalog contains technical specifications, tables, and images of products.
        Extract every distinct product, product line, or accessory from this document.
        For each item, provide a comprehensive summary including:
        1. Product Name / Series
        2. Description & Application
        3. Features
        4. Technical Details (Voltage, Size, CCT, Power, CRI, etc. from tables)
        
        Output ONLY a valid JSON array of strings. Each string should comprehensively describe ONE distinct product/series in plain text (so it can be embedded cleanly into a vector database).
        Example: [
          "Product: Baton Linear LED\\nDescription: Shining from upwards linear lighting.\\nFeatures: High brightness LED, saves electricity.\\nSpecs: Size 2000x7.60x9mm, Voltage 12/24 VDC, LED Type SMD2835, CCT 2700-6500K RGB CCT Tunable, Power 10x2 W/m, CRI >80, <90. Switch: No sensor or Door trigger switch.",
          "Product: Glanza Face Ring\\nVariants: Rose-gold, Chrome, Pearl Shining Black, Fog-silver, White, Matt-black, Ellipses Ring, Round Circle Ring..."
        ]
        
        Do NOT include markdown blocks like ```json outside the array, just the raw JSON array.
        """
        response = client.models.generate_content(
            model=key_manager.get_current_model(),
            contents=[uploaded, extraction_prompt]
        )
        
        # Parse output securely
        output_text = response.text.strip()
        if output_text.startswith("```json"):
            output_text = output_text[7:]
        if output_text.startswith("```"):
            output_text = output_text[3:]
        if output_text.endswith("```"):
            output_text = output_text[:-3]
            
        output_text = output_text.strip()
        
        try:
            chunks = json.loads(output_text)
            if not isinstance(chunks, list):
                 raise ValueError("AI did not return a list of chunks.")
        except json.JSONDecodeError:
            # Fallback parsing if AI hallucinates formatting
            chunks = [chunk.strip() for chunk in output_text.split('\n\n') if len(chunk.strip()) > 50]
            
        if not chunks:
            raise ValueError("No chunks could be extracted from this PDF.")

        if progress_callback:
            progress_callback(0.7, f"AI Extracted {len(chunks)} Intelligent Product Chunks. Embedding DB...")

        # Initialize Chroma locally
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
        collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

        # Generate IDs for each chunk
        ids = [f"{pdf_file.name}_product_{i}" for i in range(len(chunks))]
        metadatas = [{"source": pdf_file.name, "chunk": i} for i in range(len(chunks))]
        
        # Generate embeddings and insert into Chroma
        vectors = embeddings.embed_documents(chunks)
        collection.add(
            ids=ids,
            embeddings=vectors,
            metadatas=metadatas,
            documents=chunks
        )

        if progress_callback:
            progress_callback(1.0, f"Successfully Processed {len(chunks)} Intelligent Chunks!")

        return True, f"Success: {len(chunks)} complex semantic product chunks from {pdf_file.name} added."
        
    finally:
        # Clean up temp file
        if os.path.exists(temp_pdf_path):
            try:
                os.remove(temp_pdf_path)
            except:
                pass
        # Clean up gemini file to avoid quota buildup
        if uploaded:
            try:
                client.files.delete(name=uploaded.name)
            except:
                pass
