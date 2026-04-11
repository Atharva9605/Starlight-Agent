import logging
from typing import List, Dict, Any
from sentence_transformers import CrossEncoder

log = logging.getLogger("reranker")

_reranker_model = None

def get_reranker():
    """Loads the bge-reranker-large model lazily."""
    global _reranker_model
    if _reranker_model is None:
        log.info("Loading BGE Reranker Large model...")
        _reranker_model = CrossEncoder('BAAI/bge-reranker-large')
    return _reranker_model

class Reranker:
    @staticmethod
    def rerank_candidates(query: str, candidates: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Stage 3: Cross-Encoder Reranking.
        Takes the query and the top candidates from Stage 2 (Vector Search) and scores 
        their exact semantic overlap for hyper-precise ordering.
        """
        if not candidates:
            return []
            
        model = get_reranker()
        
        # Prepare pairs: [ (query, spec1), (query, spec2) ]
        # We use a summarized string of the product for reranking
        pairs = []
        for c in candidates:
            # Reconstruct basic spec string or fetch from DB.
            # Assuming payload contains product_name and category for now
            payload = c.get("payload", {})
            desc_str = f"{payload.get('product_name')} ({payload.get('category')})"
            pairs.append([query, desc_str])
            
        # Get raw logits from CrossEncoder
        scores = model.predict(pairs)
        
        # Attach new scores
        for idx, score in enumerate(scores):
            candidates[idx]["rerank_score"] = float(score)
            
        # Sort by rerank score descending
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        # Return top-k
        return candidates[:top_k]

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Testing logic here
