import os
import tempfile
import logging
from typing import Dict, Any, List
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from pydantic import BaseModel
import uvicorn

# Internal modules
try:
    from src.ingestion.pdf_parser import CatalogParser
    from src.ingestion.spec_extractor import extract_specs_from_text
    from src.ingestion.embedder import Embedder
    from src.ingestion.db_upsert import PipelineUpsert
    from src.retrieval.hybrid_search import HybridRetriever
    from src.retrieval.reranker import Reranker
    from src.generation.crm_generator import generate_crm_pitch
except ImportError as e:
    logging.error(f"Failed to import internal modules. Ensure you run this from the project root. {e}")

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("api")

app = FastAPI(
    title="Starlight Enterprise Multimodal Catalog Intelligence",
    description="Full-stack Structured-First Hybrid RAG System for Technical Catalogs",
    version="1.0.0"
)

# Shared DB Upserter (creates tables on init if not exist)
try:
    upserter = PipelineUpsert()
except Exception as e:
    log.warning(f"Could not initialize Postgres/Qdrant. Are Docker containers running? {e}")
    upserter = None


class QueryRequest(BaseModel):
    query: str
    client_context: str = ""
    top_k: int = 5

class QueryResponse(BaseModel):
    crm_pitch: Dict[str, Any]
    retrieved_products: List[Dict[str, Any]]


@app.post("/ingest", summary="Upload a PDF Catalog to the ingestion pipeline")
async def ingest_catalog(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Ingests a catalog. Handles Layout Extraction -> Spec Structuring -> Multi-Vector Embedding -> DB Upserts.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        
    if not upserter:
        raise HTTPException(status_code=500, detail="Database connection not initialized.")
        
    # Save uploaded file temporarily
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    content = await file.read()
    tmp.write(content)
    tmp.close()
    
    # Process asynchronously to free up API
    background_tasks.add_task(process_pdf_pipeline, tmp.name)
    
    return {"message": "Catalog received. Processing started in the background.", "filename": file.filename}

def process_pdf_pipeline(pdf_path: str):
    log.info(f"Starting ingestion pipeline for {pdf_path}")
    try:
        # 1. Parse PDF Blocks & Images
        parser = CatalogParser(pdf_path)
        pages = parser.parse_catalog()
        
        for page in pages:
            # Reconstruct text block
            full_text = "\n".join([b["text"] for b in page.get("text_blocks", [])])
            if len(full_text) < 50:
                continue # Skip blank/cover pages
                
            # 2. Extract Specs via LLM
            log.info(f"Extracting specs from page {page['page_num']}...")
            structured_specs = extract_specs_from_text(full_text)
            
            # Simple heuristic chunking: If the page has multiple products, an advanced 
            # parser would split them. For this pipeline, assuming 1 product per page/block.
            
            # 3. Generate Embeddings
            log.info("Generating multi-vector embeddings...")
            text_emb = Embedder.embed_text(full_text)
            spec_emb = Embedder.embed_specs(structured_specs)
            
            images_data = []
            for img in page.get("images", []):
                try:
                    img_emb = Embedder.embed_image(img["path"])
                    images_data.append({"url": img["path"], "embedding": img_emb})
                except Exception as e:
                    log.warning(f"Could not embed image {img['path']}: {e}")
                    
            # 4. Upsert to DBs
            log.info("Upserting to structured and vector databases...")
            upserter.upsert_product(
                product_data=structured_specs,
                text_emb=text_emb,
                spec_emb=spec_emb,
                images_data=images_data
            )
            
        log.info(f"Successfully processed and ingested {pdf_path}")
    except Exception as e:
        log.error(f"Pipeline failed for {pdf_path}: {e}")
    finally:
        if os.path.exists(pdf_path):
            os.remove(pdf_path)

@app.post("/query", response_model=QueryResponse, summary="Semantic Search and CRM Generation")
async def process_query(req: QueryRequest):
    """
    Executes the Hybrid Search & Reranking pipeline, then generates a CRM pitch.
    """
    if not upserter:
        raise HTTPException(status_code=500, detail="Database connection not initialized.")
        
    log.info(f"Received query: {req.query}")
    
    # Needs a DB Session for Hybrid Search
    session = upserter.SessionLocal()
    try:
        # Step 1 & 2: LLM Filter Extraction + SQL/Vector Hybrid Search
        retriever = HybridRetriever(session)
        candidates = retriever.retrieve(req.query)
        
        if not candidates:
            raise HTTPException(status_code=404, detail="No matching products found within parameters.")
            
        # Step 3: Cross-Encoder Reranking
        log.info(f"Reranking top {len(candidates)} candidates...")
        top_products = Reranker.rerank_candidates(req.query, candidates, top_k=req.top_k)
        
        # Step 4: LLM Generation (with Anti-Hallucination)
        log.info("Generating strict CRM pitch...")
        pitch = generate_crm_pitch(req.query, req.client_context, top_products)
        
        return {
            "crm_pitch": pitch,
            "retrieved_products": top_products
        }
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Query pipeline failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
