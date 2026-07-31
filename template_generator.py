"""AI-assisted HTML email template generation using Azure OpenAI."""
from __future__ import annotations

import re

from azure_client import azure_manager
from config_manager import get_template_variables, get_template_content


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
    """
    Generate a fresh Jinja HTML email template from freeform instructions.

    `style` is optional free text (not limited to modern/minimal/bold).
    `reference_template` is optional — when omitted, do not steer toward stock shells.
    """
    variables = get_template_variables()
    var_lines = "\n".join(
        f"  - {{{{ {v['name']} }}}} — {v['description']}" for v in variables
    )

    reference_block = ""
    if reference_template:
        try:
            ref_content = get_template_content(reference_template)
            reference_block = (
                f"\n\nOptional reference ({reference_template}) — use only for structural "
                f"inspiration, not a clone:\n"
                f"---\n{ref_content[:6000]}\n---\n"
            )
        except FileNotFoundError:
            pass

    style_line = ""
    if style and style.strip() and style.strip().lower() not in ("modern", "minimal", "bold"):
        style_line = f"\n- Extra style notes from the user: {style.strip()}"
    elif style and style.strip().lower() in ("modern", "minimal", "bold"):
        # Legacy callers may still pass presets — treat as soft hint only.
        hints = {
            "modern": "soft modern B2B",
            "minimal": "minimal single-column",
            "bold": "bold vibrant accents",
        }
        style_line = f"\n- Soft style hint (do not lock to a stock shell): {hints.get(style.lower(), style)}"

    system = f"""You are an expert HTML email developer for B2B sales outreach.
Generate a complete, self-contained HTML email template using Jinja2 syntax.

Rules:
- Output ONLY raw HTML starting with <!DOCTYPE html> or <html — no markdown fences.
- Invent a fresh layout from the user's instructions. Do NOT default to or copy the
  stock "Modern Soft", "Minimalist", or "Bold & Vibrant" shells unless the user
  explicitly asks for one of those looks.
- Use inline CSS suitable for email clients (tables acceptable).
- Use ONLY these Jinja2 variables (do not invent new ones):
{var_lines}
- Include loops for list variables: feature_highlights, use_cases, bullets,
  technical_specs, referenced_products, catalog_chunks.
- Use | safe filter for HTML paragraph fields when appropriate.
- Brand: Starlight Linear LED — professional Indian LED lighting manufacturer.
{style_line}
"""

    user = f"""Create an HTML email template with these requirements:

{instructions}
{reference_block}
Return the full HTML template only."""

    raw = azure_manager.chat_completion(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.45,
        max_tokens=8192,
    )
    content = _strip_markdown_fences(raw)
    if "<html" not in content.lower():
        raise ValueError("AI did not return valid HTML. Try more specific instructions.")
    return content
