import json
import logging
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

# Ensure key manager handles rotation
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

log = logging.getLogger("query_parser")

class NumericFilter(BaseModel):
    min_val: Optional[float] = Field(None, description="Minimum value")
    max_val: Optional[float] = Field(None, description="Maximum value")

class QueryFilters(BaseModel):
    category: Optional[str] = Field(None, description="Target category, e.g., 'spotlight', 'linear'")
    environment: Optional[str] = Field(None, description="Environment, e.g., 'outdoor', 'indoor'")
    wattage: Optional[NumericFilter] = Field(None, description="Wattage range")
    beam_angle: Optional[NumericFilter] = Field(None, description="Beam angle range in degrees")
    ip_rating_min: Optional[int] = Field(None, description="Minimum numeric IP rating (e.g., 65 for IP65)")

@with_key_rotation
def extract_query_filters(user_query: str) -> Dict[str, Any]:
    """
    Parses a natural language query into structured SQL/DB filters.
    """
    client = key_manager.get_client()
    if not client:
        return {}
        
    prompt = f"""
    You are an expert product search assistant.
    Analyze the following user query for lighting products and extract any structured filters.
    Return ONLY a JSON object matching the requested schema. If a filter is not mentioned, omit it or set it to null.
    
    For ranges (e.g., "around 12W", "10-15W"), use min_val and max_val. "Around 12W" could mean min: 10, max: 14.
    For IP ratings (e.g., "IP65"), extract just the number (65) as ip_rating_min.
    
    USER QUERY:
    ---
    {user_query}
    ---
    """
    
    try:
        response = client.models.generate_content(
            model=key_manager.get_current_model(),
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": QueryFilters,
                "temperature": 0.0,
            },
        )
        return json.loads(response.text)
    except Exception as e:
        log.error(f"Failed to extract filters: {e}")
        return {}

def build_sql_filters(filters: Dict[str, Any]) -> str:
    """
    Converts the structured JSON into a SQL WHERE clause for SQLite/Postgres.
    This creates the pre-filtering string (Stage 1).
    """
    clauses = []
    
    if filters.get("category"):
        clauses.append(f"category ILIKE '%{filters['category']}%'")
        
    if filters.get("environment"):
        # Could match against environment/application column
        clauses.append(f"application ILIKE '%{filters['environment']}%'")
        
    if filters.get("ip_rating_min"):
        # Basic exact match or regex depending on DB support, using simple ILIKE for now
        clauses.append(f"specs.ip_rating ILIKE '%IP{filters['ip_rating_min']}%'")
        
    # JSON array bounds checking is DB specific. 
    # For Postgres: clauses.append("EXISTS (SELECT 1 FROM json_array_elements_text(specs.wattage) as w WHERE w::float >= min AND ...)")
    # For now, we will return the parameter dictionary and use SQLAlchemy's JSON pathing in the actual Hybrid Search module.
    
    return clauses

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # test_query = "Need outdoor spotlight around 12W with narrow beam"
    # filters = extract_query_filters(test_query)
    # print(json.dumps(filters, indent=2))
