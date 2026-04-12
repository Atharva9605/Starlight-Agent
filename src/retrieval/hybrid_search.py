import logging
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from src.database.models import Product, ProductSpec
from src.database.qdrant_db import get_qdrant_client, COLLECTION_TEXT
from src.ingestion.embedder import Embedder
from src.retrieval.query_parser import extract_query_filters

log = logging.getLogger("hybrid_search")

class HybridRetriever:
    def __init__(self, db_session: Session):
        self.session = db_session
        self.qdrant = get_qdrant_client()
        
    def _stage1_sql_filter(self, filters: Dict[str, Any]) -> List[int]:
        """
        Stage 1: Pre-filtering using PostgreSQL JSONB constraints.
        Returns a list of Product IDs that match the hard constraints.
        """
        query = self.session.query(Product.id).join(ProductSpec)
        
        # Example filters logic 
        # (Postgres JSONB filtering can get complex, simplifying here for demonstration)
        category = filters.get("category")
        if category:
            query = query.filter(Product.category.ilike(f"%{category}%"))
            
        environment = filters.get("environment")
        if environment:
            query = query.filter(ProductSpec.application.ilike(f"%{environment}%"))
            
        # ... Add numeric/JSON filtering for wattage, beam angle here ...
        
        results = query.all()
        return [r.id for r in results]

    def _stage2_vector_search(self, query_text: str, allowed_ids: List[int] = None, top_k: int = 20) -> List[Dict[str, Any]]:
        """
        Stage 2: Semantic Vector Search in Qdrant.
        Filters by `allowed_ids` if provided (outputs of Stage 1).
        """
        query_vector = Embedder.embed_text(query_text)
        
        # Build filter payload for Qdrant
        qdrant_filter = None
        if allowed_ids:
            from qdrant_client.models import Filter, FieldCondition, MatchAny
            qdrant_filter = Filter(
                must=[
                    FieldCondition(
                        key="product_id",
                        match=MatchAny(any=allowed_ids)
                    )
                ]
            )
            
        search_result = self.qdrant.search(
            collection_name=COLLECTION_TEXT,
            query_vector=query_vector,
            query_filter=qdrant_filter,
            limit=top_k
        )
        
        return [{"id": hit.payload["product_id"], "score": hit.score, "payload": hit.payload} for hit in search_result]

    def retrieve(self, query_text: str) -> List[Dict[str, Any]]:
        """
        Executes the Hybrid Search Pipeline: 
        1. LLM Query Parsing -> 2. SQL Pre-filtering -> 3. Vector Semantic Search
        """
        log.info(f"Retrieving candidates for: '{query_text}'")
        
        # 1. Parse Query
        filters = extract_query_filters(query_text)
        log.info(f"Extracted Filters: {filters}")
        
        # 2. SQL Pre-filtering
        allowed_ids = None
        if filters:
             allowed_ids = self._stage1_sql_filter(filters)
             log.info(f"SQL Pre-filter passed {len(allowed_ids)} candidates.")
             if not allowed_ids:
                 log.warning("SQL Pre-filter removed all candidates.")
                 return []
                 
        # 3. Vector Search
        candidates = self._stage2_vector_search(query_text, allowed_ids=allowed_ids)
        log.info(f"Vector search retrieved {len(candidates)} candidates.")
        
        return candidates

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Testing would require an active DB session
