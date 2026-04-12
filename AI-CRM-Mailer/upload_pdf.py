"""
Upload PDF catalogues to ChromaDB using Azure OpenAI embeddings.

Upload PDF catalogues to ChromaDB using Azure OpenAI embeddings.
It uses the project's azure_client module for embeddings and
the local persistent ChromaDB for storage.

Usage:
    python upload_pdf.py
"""
import os
import glob
import logging

from dotenv import load_dotenv
import fitz  # PyMuPDF
import chromadb

from azure_client import azure_manager

load_dotenv()
log = logging.getLogger("upload_pdf")

# Verify Azure OpenAI is configured
if not os.getenv("AZURE_OPENAI_API_KEY"):
    print("Error: AZURE_OPENAI_API_KEY not found in environment.")
    exit(1)

# Configuration
CATALOGS_DIR = "catalogs"
CHROMA_DB_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "starlight_vision"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c.strip() for c in chunks if len(c.strip()) > 40]


def main():
    print(f"Starting PDF ingestion from '{CATALOGS_DIR}' directory...")

    # 1. Connect to ChromaDB (local persistent)
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    print("Connected to ChromaDB (local persistent).")

    # 2. Get or Create Collection
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # 3. Load PDFs
    pdf_files = glob.glob(os.path.join(CATALOGS_DIR, "*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in '{CATALOGS_DIR}'. Please add some PDFs and run again.")
        return

    print(f"Found {len(pdf_files)} PDF(s).")

    for pdf_file in pdf_files:
        print(f"Processing: {pdf_file}")
        try:
            doc = fitz.open(pdf_file)
            pages = []
            for i, page in enumerate(doc, start=1):
                text = page.get_text("text").strip()
                if len(text) > 40:
                    pages.append(f"[Page {i}]\n{text}")
            doc.close()
            print(f"  - Loaded {len(pages)} text-bearing pages.")

            if not pages:
                print(f"  - No readable text found in {pdf_file}. Skipping.")
                continue

            # Chunk all pages
            full_text = "\n\n".join(pages)
            chunks = _chunk_text(full_text)
            print(f"  - Split into {len(chunks)} chunks.")

            if not chunks:
                continue

            # 4. Generate Embeddings with Azure OpenAI & Add to ChromaDB
            print("  - Generating Azure OpenAI embeddings and adding to ChromaDB...")

            base_id = os.path.basename(pdf_file)
            ids = [f"{base_id}_chunk_{i}" for i in range(len(chunks))]
            metadatas = [
                {
                    "source": base_id,
                    "catalogue_name": base_id,
                    "catalogue_type": "text_pdf",
                    "page_number": 0,
                    "blob_url": "",
                    "product_name": "",
                    "category": "",
                    "specs_preview": "",
                    "char_count": len(c),
                }
                for c in chunks
            ]

            embeddings_list = azure_manager.embed_documents(chunks)

            # Replace stale entries
            try:
                existing = collection.get(where={"source": base_id})
                if existing.get("ids"):
                    collection.delete(ids=existing["ids"])
            except Exception:
                pass

            collection.add(
                embeddings=embeddings_list,
                documents=chunks,
                metadatas=metadatas,
                ids=ids,
            )
            print("  - Successfully added to vector database.")

        except Exception as e:
            print(f"  - Error processing {pdf_file}: {e}")

    print("\nIngestion complete. ChromaDB is updated.")


if __name__ == "__main__":
    main()
