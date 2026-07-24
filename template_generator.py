"""AI-assisted HTML email template generation using Azure OpenAI."""
from __future__ import annotations

import re

from azure_client import azure_manager
from config_manager import get_template_variables, get_template_content

_STYLE_HINTS = {
    "modern": "Clean modern B2B layout with soft gradients and rounded sections.",
    "minimal": "Minimal whitespace, single column, subtle typography.",
    "bold": "Vibrant header band, strong accent colors, energetic B2B sales feel.",
}


def _strip_markdown_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:html)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def generate_template_html(
    instructions: str,
    style: str | None = None,
    reference_template: str | None = None,
) -> str:
    variables = get_template_variables()
    var_lines = "\n".join(
        f"  - {{{{ {v['name']} }}}} — {v['description']}" for v in variables
    )

    reference_block = ""
    if reference_template:
        try:
            ref_content = get_template_content(reference_template)
            reference_block = (
                f"\n\nReference template ({reference_template}) for layout inspiration:\n"
                f"---\n{ref_content[:8000]}\n---\n"
            )
        except FileNotFoundError:
            pass

    style_hint = _STYLE_HINTS.get((style or "modern").lower(), style or "modern professional")

    system = f"""You are an expert HTML email developer for B2B sales outreach.
Generate a complete, self-contained HTML email template using Jinja2 syntax.

Rules:
- Output ONLY raw HTML starting with <!DOCTYPE html> or <html — no markdown fences.
- Use inline CSS suitable for email clients (tables acceptable).
- Use ONLY these Jinja2 variables (do not invent new ones):
{var_lines}
- Include loops for list variables: feature_highlights, use_cases, bullets, technical_specs, referenced_products, catalog_chunks.
- Use | safe filter for HTML paragraph fields when appropriate.
- Brand: Starlight Linear LED — professional Indian LED lighting manufacturer.
- Style direction: {style_hint}
"""

    user = f"""Create an HTML email template with these requirements:

{instructions}
{reference_block}
Return the full HTML template only."""

    raw = azure_manager.chat_completion(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.4,
        max_tokens=8192,
    )
    content = _strip_markdown_fences(raw)
    if "<html" not in content.lower():
        raise ValueError("AI did not return valid HTML. Try more specific instructions.")
    return content
