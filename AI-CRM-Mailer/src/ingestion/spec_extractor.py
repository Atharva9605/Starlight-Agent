"""
Spec Extractor – Azure OpenAI edition.

Uses Azure OpenAI GPT-4o to extract structured product specifications
from raw catalogue text using Azure OpenAI GPT-4o.
"""
import json
import logging
import sys
import os
from typing import Dict, Any, List
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

try:
    from azure_client import azure_manager
except ImportError:
    azure_manager = None  # type: ignore

log = logging.getLogger("spec_extractor")


class ProductSpecSchema(BaseModel):
    product_name: str = Field(description="The primary name of the product")
    series: str = Field(default="", description="The product series or family name")
    category: str = Field(description="Main category, e.g., 'spotlight', 'linear', 'downlight'")
    subcategory: str = Field(default="", description="Specific subcategory")
    wattage: List[float] = Field(default=[], description="List of wattage options (numeric values only)")
    diameter: List[float] = Field(default=[], description="List of diameters in mm")
    height: List[float] = Field(default=[], description="List of heights in mm")
    cutout: List[float] = Field(default=[], description="List of cutout sizes in mm")
    beam_angle: List[float] = Field(default=[], description="List of beam angles in degrees")
    ip_rating: str = Field(default="", description="IP rating, e.g., 'IP65', 'IP20'")
    driver: List[str] = Field(default=[], description="Compatible driver brands")
    led_type: str = Field(default="", description="Type of LED used, e.g. 'SMD2835'")
    color_temperature: List[str] = Field(default=[], description="Available CCTs, e.g. ['3000K', '4000K']")
    lumen_efficiency: str = Field(default="", description="Lumen efficiency, e.g. '100lm/W'")
    body_color: List[str] = Field(default=[], description="Available body or housing colors")
    material: str = Field(default="", description="Construction material")
    application: str = Field(default="", description="Recommended applications, e.g. 'outdoor', 'retail'")
    mounting_type: str = Field(default="", description="How the product is mounted, e.g. 'recessed', 'surface'")


_SPEC_SYSTEM = """\
You are an expert lighting catalog data extractor.
Analyze the following raw text extracted from a product catalog page.
Extract the product specifications and return them in STRICT structured JSON format.
If a numerical value has multiple options (like 12W, 20W, 30W), return them as a list of numbers: [12, 20, 30].
If a specification is missing, use the default empty value for its type. Do NOT guess or hallucinate specs.

Return a JSON object with keys: product_name, series, category, subcategory,
wattage, diameter, height, cutout, beam_angle, ip_rating, driver, led_type,
color_temperature, lumen_efficiency, body_color, material, application, mounting_type.
"""


def extract_specs_from_text(raw_text: str) -> Dict[str, Any]:
    """Uses Azure OpenAI to extract specifications into a structured JSON format."""
    if azure_manager is None:
        log.error("azure_manager not available — cannot extract specs.")
        return {}

    messages = [
        {"role": "system", "content": _SPEC_SYSTEM},
        {"role": "user", "content": f"RAW TEXT:\n---\n{raw_text}\n---"},
    ]

    try:
        raw = azure_manager.chat_completion(
            messages, temperature=0.0, max_tokens=2048, json_mode=True
        )
        return json.loads(raw)
    except Exception as e:
        log.error(f"Failed to extract specs: {e}")
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # test_text = "AQUA Spotlight Series. IP65 rated outdoor luminaire. 12W, 15W options. Beam angle 24, 36. Philips driver."
    # print(json.dumps(extract_specs_from_text(test_text), indent=2))
