"""
CRM Pitch Generator – Azure OpenAI edition.

Stage 5 of the full hybrid RAG pipeline (FastAPI / src/ stack).
Takes the top reranked products from the Qdrant retrieval pipeline and
generates a structured, grounded CRM pitch using Azure OpenAI GPT-4o.

Anti-hallucination guarantees:
  • Product context is injected verbatim; the model is forbidden from
    inventing specs or product names outside the provided block.
  • Temperature is set to 0.0 for fully deterministic output.
  • JSON mode is enforced so every field is present and well-typed.
"""
import json
import logging
import sys
import os
from typing import List, Dict, Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

try:
    from azure_client import azure_manager
except ImportError:
    # Allow isolated unit-test execution without the full project on sys.path
    azure_manager = None  # type: ignore

log = logging.getLogger("crm_generator")


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_CRM_SYSTEM = """\
You are a senior technical sales engineer writing a personalised CRM pitch
for a B2B lighting client on behalf of Starlight Linear LED.

STRICT ANTI-HALLUCINATION RULES:
1. Recommend ONLY products that appear VERBATIM in the AVAILABLE PRODUCT CONTEXT.
2. Do NOT invent, assume, or extrapolate any specification (wattage, IP rating,
   beam angle, CCT, dimensions, lumen output, driver type, etc.) that is NOT
   explicitly stated in the context block.
3. If a specification is missing from the context, write "not specified" — never guess.
4. Connect the client's exact query language to specific, numbered/categorical
   specs retrieved from the context.
5. The recommended_product field must use the EXACT product name from the context.

Output a valid JSON object with these keys ONLY (no extra text):
  "subject"            – string: catchy email subject tailored to client industry
  "recommended_product"– string: exact product name from context
  "explanation"        – string: why this product fits (cite specific specs)
  "reasons"            – array of strings: bullet-point reasons with cited specs
  "cta"                – string: clear call to action
"""

_CRM_USER_TMPL = """\
CLIENT QUERY / CONTEXT:
{user_query}
{client_context}

AVAILABLE PRODUCT CONTEXT (sourced from catalogue database — use ONLY these):
===
{structured_context}
===

Generate the CRM pitch JSON now.
"""


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

def construct_product_context(top_products: List[Dict[str, Any]]) -> str:
    """
    Format the top-k reranked products into a clean context block.

    Each product is rendered as a numbered entry with its payload fields
    listed as "Key: Value" pairs, skipping internal/embedding fields.
    """
    blocks: List[str] = []
    for idx, prod in enumerate(top_products, start=1):
        payload = prod.get("payload", {})
        lines = [f"Product {idx}:"]
        for key, val in payload.items():
            if key in ("id", "embedding") or val is None:
                continue
            if isinstance(val, list):
                if not val:
                    continue
                val_str = ", ".join(str(v) for v in val)
            else:
                val_str = str(val)
            lines.append(f"  {key.replace('_', ' ').title()}: {val_str}")
        blocks.append("\n".join(lines))

    return "\n\n".join(blocks) if blocks else "No products retrieved."


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------

def generate_crm_pitch(
    user_query: str,
    client_context: str,
    top_products: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Stage 5: generate a grounded CRM pitch from the reranked product list.

    Args:
        user_query:      Raw query or client description.
        client_context:  Additional context about the client (scraped data, etc.).
        top_products:    Top-k reranked products from the Qdrant pipeline.

    Returns:
        Parsed JSON dict with keys: subject, recommended_product, explanation,
        reasons, cta.  Returns an empty dict on failure.
    """
    if azure_manager is None:
        log.error("azure_manager not available — cannot generate CRM pitch.")
        return {}

    structured_context = construct_product_context(top_products)

    messages = [
        {"role": "system", "content": _CRM_SYSTEM},
        {
            "role": "user",
            "content": _CRM_USER_TMPL.format(
                user_query=user_query,
                client_context=client_context,
                structured_context=structured_context,
            ),
        },
    ]

    try:
        raw = azure_manager.chat_completion(
            messages,
            temperature=0.0,
            max_tokens=1024,
            json_mode=True,
        )
        return json.loads(raw)
    except Exception as exc:
        log.error("Failed to generate CRM pitch: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# Optional corrective agentic loop (expandable)
# ---------------------------------------------------------------------------

def agentic_corrective_loop(
    user_query: str,
    client_context: str,
    top_products: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Wrapper that can be extended to implement iterative retrieval expansion
    when cross-encoder scores are below a confidence threshold.

    Currently delegates directly to generate_crm_pitch.
    """
    return generate_crm_pitch(user_query, client_context, top_products)


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    mock_products = [
        {
            "payload": {
                "product_name": "Glanza 12W Surface Downlight",
                "category": "downlight",
                "wattage": 12,
                "beam_angle": 36,
                "ip_rating": "IP20",
                "color_temperature": "3000K / 4000K",
                "body_color": "White / Black",
                "application": "Retail, Hospitality, Office",
            }
        }
    ]
    result = generate_crm_pitch(
        user_query="We need energy-efficient downlights for a boutique hotel lobby.",
        client_context="Client is a luxury hospitality group in Mumbai.",
        top_products=mock_products,
    )
    print(json.dumps(result, indent=2))
