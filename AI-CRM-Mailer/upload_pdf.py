import os
import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import chromadb

# Load environment variables (for GEMINI_API_KEY)
load_dotenv()

# Verify API key
if not os.getenv("GEMINI_API_KEY"):
    print("Error: GEMINI_API_KEY not found in environment.")
    exit(1)

# Configuration
CATALOGS_DIR = "catalogs"
CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
COLLECTION_NAME = "starlight_catalogs"

def main():
    print(f"Starting PDF ingestion from '{CATALOGS_DIR}' directory...")
    
    # 1. Connect to ChromaDB
    try:
        chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        print("Connected to ChromaDB.")
    except Exception as e:
        print(f"Failed to connect to ChromaDB at {CHROMA_HOST}:{CHROMA_PORT} - Ensure Docker is running.")
        print(f"Error: {e}")
        return

    # 2. Setup Embeddings
    print("Initializing Gemini Embeddings (models/gemini-embedding-001)...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    # 3. Get or Create Collection
    # Note: Using native Chroma client for collection management
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
    
    # 4. Load PDFs
    pdf_files = glob.glob(os.path.join(CATALOGS_DIR, "*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in '{CATALOGS_DIR}'. Please add some PDFs and run again.")
        return
        
    print(f"Found {len(pdf_files)} PDF(s).")
    
    # 5. Split Text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    
    for pdf_file in pdf_files:
        print(f"Processing: {pdf_file}")
        try:
            loader = PyPDFLoader(pdf_file)
            docs = loader.load()
            print(f"  - Loaded {len(docs)} pages.")
            
            chunks = text_splitter.split_documents(docs)
            print(f"  - Split into {len(chunks)} chunks.")
            
            if not chunks:
                continue
            
            # 6. Generate Embeddings & Add to Chroma
            print("  - Generating embeddings and adding to ChromaDB...")
            
            # Extract texts and metadata
            texts = [chunk.page_content for chunk in chunks]
            metadatas = [chunk.metadata for chunk in chunks]
            
            # Add an 'id' for each chunk
            base_id = os.path.basename(pdf_file)
            ids = [f"{base_id}_chunk_{i}" for i in range(len(chunks))]
            
            # Generate embeddings using Langchain's Google Embedder
            embeddings_list = embeddings.embed_documents(texts)
            
            # Add to Chroma
            collection.add(
                embeddings=embeddings_list,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            print("  - Successfully added to vector database.")
            
        except Exception as e:
            print(f"  - Error processing {pdf_file}: {e}")

    print("\nIngestion complete. ChromaDB is updated.")

if __name__ == "__main__":
    main()
