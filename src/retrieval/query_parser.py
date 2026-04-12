"""
Query Parser – Azure OpenAI edition.

Parses natural-language queries into structured filters
for the hybrid search pipeline using Azure OpenAI GPT-4o.
"""
import json
import logging
import sys
import os
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

try:
    from azure_client import azure_manager
except ImportError:
    azure_manager = None  # type: ignore

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


_QUERY_SYSTEM = """\
You are an expert product search assistant.
Analyze the following user query for lighting products and extract any structured filters.
Return ONLY a JSON object matching the requested schema.
If a filter is not mentioned, omit it or set it to null.

For ranges (e.g., "around 12W", "10-15W"), use min_val and max_val.
"Around 12W" could mean min: 10, max: 14.
For IP ratings (e.g., "IP65"), extract just the number (65) as ip_rating_min.

JSON keys: category, environment, wattage (object with min_val/max_val),
beam_angle (object with min_val/max_val), ip_rating_min.
"""


def extract_query_filters(user_query: str) -> Dict[str, Any]:
    """
    Parses a natural language query into structured SQL/DB filters
    using Azure OpenAI.
    """
    if azure_manager is None:
        log.warning("azure_manager not available — returning empty filters.")
        return {}

    messages = [
        {"role": "system", "content": _QUERY_SYSTEM},
        {"role": "user", "content": f"USER QUERY:\n---\n{user_query}\n---"},
    ]

    try:
        raw = azure_manager.chat_completion(
            messages, temperature=0.0, max_tokens=512, json_mode=True
        )
        return json.loads(raw)
    except Exception as e:
        log.error(f"Failed to extract filters: {e}")
        return {}


def build_sql_filters(filters: Dict[str, Any]) -> list:
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
        clauses.append(f"specs.ip_rating ILIKE '%IP{filters['ip_rating_min']}%'")

    return clauses


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # test_query = "Need outdoor spotlight around 12W with narrow beam"
    # filters = extract_query_filters(test_query)
    # print(json.dumps(filters, indent=2))
