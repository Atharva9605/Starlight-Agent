import json
import logging
from typing import Dict, Any, List
from google import genai
from pydantic import BaseModel, Field

# Ensure key manager handles rotation
try:
    from key_manager import key_manager, with_key_rotation
except ImportError:
    import sys
    sys.path.append("..")
    from key_manager import key_manager, with_key_rotation

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

@with_key_rotation
def extract_specs_from_text(raw_text: str) -> Dict[str, Any]:
    """Uses Gemini Structured Outputs to extract specifications into a strict JSON schema."""
    client = key_manager.get_client()
    
    prompt = f"""
    You are an expert lighting catalog data extractor. 
    Analyze the following raw text extracted from a product catalog page.
    Extract the product specifications and return them in STRICT structured JSON format matching the schema rules.
    If a numerical value has multiple options (like 12W, 20W, 30W), return them as a list of numbers: [12, 20, 30].
    If a specification is missing, use the default empty value for its type. Do NOT guess or hallucinate specs.
    
    RAW TEXT:
    ---
    {raw_text}
    ---
    """
    
    try:
        response = client.models.generate_content(
            model=key_manager.get_current_model(),
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": ProductSpecSchema,
                "temperature": 0.0,
            },
        )
        return json.loads(response.text)
    except Exception as e:
        log.error(f"Failed to extract specs: {e}")
        raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # test_text = "AQUA Spotlight Series. IP65 rated outdoor luminaire. 12W, 15W options. Beam angle 24, 36. Philips driver."
    # print(json.dumps(extract_specs_from_text(test_text), indent=2))
