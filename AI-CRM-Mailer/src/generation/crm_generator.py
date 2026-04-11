import json
import logging
from typing import List, Dict, Any
from pydantic import BaseModel, Field

try:
    from key_manager import key_manager, with_key_rotation
except ImportError:
    import sys
    sys.path.append("..")
    import os
    if os.path.exists("key_manager.py"):
        from key_manager import key_manager, with_key_rotation
    else:
        # Dummy if running completely isolated for tests
        def with_key_rotation(f): return f
        class KM:
            def get_client(self): pass
            def get_current_model(self): return "gemini-2.0-flash"
        key_manager = KM()

log = logging.getLogger("crm_generator")

class CRMPitch(BaseModel):
    subject: str = Field(description="Catchy email subject line tailored to the client's industry.")
    recommended_product: str = Field(description="The primary name of the recommended product from the context.")
    explanation: str = Field(description="Strict explanation of WHY this product fits their query using ONLY provided specs.")
    reasons: List[str] = Field(description="Bullet points of reasons and extracted specs.")
    cta: str = Field(description="Call to action for the client.")

def construct_product_context(top_products: List[Dict[str, Any]]) -> str:
    """
    Compiles the top 5 reranked products into a strict context block for the LLM.
    """
    context_blocks = []
    for idx, prod in enumerate(top_products, start=1):
        payload = prod.get("payload", {})
        block = f"Product {idx}:\n"
        for k, v in payload.items():
            if k not in ['id', 'embedding'] and v is not None:
                # Format list values nicely
                if isinstance(v, list):
                    if len(v) == 0: continue
                    val_str = ", ".join(str(x) for x in v)
                else:
                    val_str = str(v)
                block += f"{k.replace('_', ' ').title()}: {val_str}\n"
        context_blocks.append(block)
        
    return "\n\n".join(context_blocks)

@with_key_rotation
def generate_crm_pitch(user_query: str, client_context: str, top_products: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Stage 5: LLM Generation Layer.
    Uses the strictest anti-hallucination guardrails to assemble the CRM pitch 
    solely based on the structured context provided.
    """
    client = key_manager.get_client()
    if not client:
        return {}
        
    structured_context = construct_product_context(top_products)
    
    prompt = f"""
    You are an expert technical sales engineer constructing a CRM pitch for a client.
    
    CLIENT OR QUERY:
    {user_query}
    {client_context}
    
    AVAILABLE PRODUCT CONTEXT (Extracted from Catalog Database):
    ---
    {structured_context}
    ---
    
    STRICT ANTI-HALLUCINATION CONSTRAINTS:
    1. You MUST recommend the best fitting product from the AVAILABLE PRODUCT CONTEXT ONLY.
    2. Do NOT hallucinate, invent, or assume any specifications, features, or product names not explicitly written in the context.
    3. If a specification (like driver or IP rating) is missing from the context, state "not specified" or omit it. Do not guess.
    4. Focus on connecting the client's exact query to the specific numerical or categorical specs retrieved.
    
    Format the output perfectly matching the requested JSON structure.
    """
    
    try:
        response = client.models.generate_content(
            model=key_manager.get_current_model(),
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": CRMPitch,
                "temperature": 0.0, 
            },
        )
        return json.loads(response.text)
    except Exception as e:
        log.error(f"Failed to generate CRM Pitch: {e}")
        return {}

def agentic_corrective_loop(user_query: str, client_context: str, top_products: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Optional Stage: If the Cross-Encoder scores are exceedingly low, we can 
    dynamically decide to expand the retrieval window or alter the query. 
    (Placeholder for advanced iterative logic).
    """
    # E.g., if max(rerank_score) < threshold: return False
    # For now, immediately generate the pitch:
    return generate_crm_pitch(user_query, client_context, top_products)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Testing logic here
