"""
Business configuration manager — prompts, sender info, and template metadata.

Per-organization config stored in PostgreSQL (organization_config).
Built-in templates are global; custom templates live under data/custom_templates/{org_id}/.
"""
from __future__ import annotations

import copy
import json
import os
import re
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from tenant import get_organization_id

_BASE_DIR = Path(__file__).parent
_DEFAULTS_PATH = _BASE_DIR / "defaults" / "business_config.json"
_TEMPLATES_DIR = _BASE_DIR / "templates"
_CUSTOM_TEMPLATES_ROOT = _BASE_DIR / "data" / "custom_templates"

_BUILTIN_TEMPLATES = {
    "email_template.html": {"label": "Modern Soft", "builtin": True},
    "email_template_minimalist.html": {"label": "Minimalist", "builtin": True},
    "email_template_bold.html": {"label": "Bold & Vibrant", "builtin": True},
}

_TEMPLATE_VARIABLES = [
    {"name": "subject", "description": "Email subject line"},
    {"name": "preamble", "description": "Short tagline below header"},
    {"name": "opening_line", "description": "Opening greeting line"},
    {"name": "intro_paragraph", "description": "Main intro paragraph (HTML safe)"},
    {"name": "feature_highlights", "description": "List of feature bullet strings"},
    {"name": "use_cases", "description": "List of use-case bullet strings"},
    {"name": "bullets", "description": "Generic bullet list"},
    {"name": "technical_specs", "description": "Technical spec bullets"},
    {"name": "closing_paragraph", "description": "Call-to-action closing"},
    {"name": "sender_name", "description": "Sender display name"},
    {"name": "sender_company", "description": "Company name"},
    {"name": "sender_phone", "description": "Contact phone"},
    {"name": "sender_website", "description": "Company website"},
    {"name": "sender_email", "description": "Reply-to email"},
    {"name": "company_logo_url", "description": "Logo URL or cid:company_logo"},
    {"name": "referenced_products", "description": "List of product ref dicts (blob_url, product_name, …)"},
    {"name": "catalog_chunks", "description": "Raw RAG text chunks"},
]

_SAMPLE_PREVIEW_DATA = {
    "subject": "Illuminating Your Next Hospitality Project",
    "preamble": "Precision-engineered LED solutions",
    "opening_line": "Hope this email finds you well.",
    "intro_paragraph": (
        "We admire your portfolio of boutique hotel interiors and believe "
        "Starlight's linear LED systems would complement your design language."
    ),
    "feature_highlights": [
        "<b>Architectural Linear Profiles</b><br>Seamless cove and shelf lighting",
        "<b>High CRI Options</b><br>90+ CRI for premium hospitality spaces",
        "<b>Custom Lengths</b><br>Manufactured to your exact millimetre specs",
    ],
    "use_cases": [
        "Lobby accent lighting and reception backdrops",
        "Guest room headboard and wardrobe illumination",
    ],
    "bullets": [],
    "technical_specs": [],
    "closing_paragraph": (
        "Would you be available for a brief call next week to explore "
        "how we can illuminate your next project?"
    ),
    "sender_name": "Vivek Dhondarkar",
    "sender_company": "Starlight Linear LED",
    "sender_phone": "9619436066",
    "sender_website": "www.starlightlinearled.com",
    "sender_email": "vivek@starlightlinearled.com",
    "company_logo_url": "https://i.ibb.co/b5NVtcS3/starlight.jpg",
    "referenced_products": [
        {
            "product_name": "SL-LINEAR-PRO 12W",
            "catalogue_name": "Starlight Linear Catalogue",
            "page_number": 14,
            "blob_url": "https://via.placeholder.com/400x280/0f172a/4CAF50?text=Product+Preview",
            "category": "linear",
            "specs_preview": "12W · 3000K · IP44 · 1200 lm/m",
        }
    ],
    "catalog_chunks": [],
}

_lock = threading.Lock()
_cache: dict[str, dict] = {}
_preview_local = threading.local()


@contextmanager
def preview_overrides(
    prompts: dict[str, str] | None = None,
    sender: dict[str, str] | None = None,
) -> Iterator[None]:
    """Temporarily override prompts/sender for draft preview (does not persist)."""
    prev_prompts = getattr(_preview_local, "prompt_overrides", None)
    prev_sender = getattr(_preview_local, "sender_overrides", None)
    _preview_local.prompt_overrides = prompts
    _preview_local.sender_overrides = sender
    try:
        yield
    finally:
        _preview_local.prompt_overrides = prev_prompts
        _preview_local.sender_overrides = prev_sender


def _org_id(organization_id: str | None = None) -> str:
    return organization_id or get_organization_id()


def _custom_templates_dir(organization_id: str | None = None) -> Path:
    d = _CUSTOM_TEMPLATES_ROOT / _org_id(organization_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_defaults() -> dict:
    with open(_DEFAULTS_PATH, encoding="utf-8") as f:
        return json.load(f)


def _load_org_config(organization_id: str | None = None) -> dict:
    org = _org_id(organization_id)
    with _lock:
        if org in _cache:
            return copy.deepcopy(_cache[org])
    from org_store import get_org_config
    from vector_store import use_postgres

    if use_postgres():
        config = get_org_config(org)
    else:
        legacy = _BASE_DIR / "data" / "business_config.json"
        if legacy.exists():
            with open(legacy, encoding="utf-8") as f:
                config = json.load(f)
        else:
            config = copy.deepcopy(_load_defaults())
    with _lock:
        _cache[org] = copy.deepcopy(config)
    return copy.deepcopy(config)


def _save_org_config(config: dict, organization_id: str | None = None) -> dict:
    org = _org_id(organization_id)
    from org_store import save_org_config
    from vector_store import use_postgres

    if use_postgres():
        save_org_config(org, config)
    else:
        legacy = _BASE_DIR / "data" / "business_config.json"
        legacy.parent.mkdir(parents=True, exist_ok=True)
        with open(legacy, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    with _lock:
        _cache[org] = copy.deepcopy(config)
    return copy.deepcopy(config)


def get_config(force_reload: bool = False, organization_id: str | None = None) -> dict:
    org = _org_id(organization_id)
    if force_reload:
        with _lock:
            _cache.pop(org, None)
    return _load_org_config(org)


def save_config(config: dict, organization_id: str | None = None) -> dict:
    return _save_org_config(config, organization_id)


def reset_config(organization_id: str | None = None) -> dict:
    defaults = _load_defaults()
    return save_config(defaults, organization_id)


def get_prompt(key: str, organization_id: str | None = None) -> str:
    overrides = getattr(_preview_local, "prompt_overrides", None)
    if overrides and key in overrides:
        return overrides[key]

    config = get_config(organization_id=organization_id)
    entry = config.get("prompts", {}).get(key, {})
    if isinstance(entry, dict):
        content = entry.get("content", "")
        if content:
            return content
    defaults = _load_defaults()
    return defaults.get("prompts", {}).get(key, {}).get("content", "")


def format_prompt(template: str, **values: Any) -> str:
    """
    Substitute {placeholders} without str.format().

    Prompts are user-editable and routinely contain literal JSON braces such as
    {"products": [...]}, which str.format() reads as a field name and rejects
    with KeyError. Only the named placeholders are replaced; every other brace
    is left untouched. Single pass, so a substituted value that happens to
    contain a placeholder is not re-expanded.
    """
    if not values:
        return template
    pattern = re.compile(
        "|".join(re.escape("{" + k + "}") for k in sorted(values, key=len, reverse=True))
    )
    return pattern.sub(lambda m: str(values[m.group(0)[1:-1]]), template)


def get_sender(organization_id: str | None = None) -> dict:
    config = get_config(organization_id=organization_id)
    sender = config.get("sender", {})
    defaults = _load_defaults().get("sender", {})
    merged = {**defaults, **sender}
    sender_overrides = getattr(_preview_local, "sender_overrides", None)
    if sender_overrides:
        merged = {**merged, **{k: v for k, v in sender_overrides.items() if v}}
    env_map = {
        "sender_name": "SENDER_NAME",
        "sender_company": "SENDER_COMPANY",
        "sender_phone": "SENDER_PHONE",
        "sender_website": "SENDER_WEBSITE",
        "sender_email": "SENDER_EMAIL",
        "company_logo_url": "COMPANY_LOGO_URL",
    }
    for k, env_key in env_map.items():
        val = os.getenv(env_key, "").strip()
        if val:
            merged[k] = val
    return merged


REQUIRED_DRAFT_JSON_KEYS = (
    "subject",
    "preamble",
    "opening_line",
    "intro",
    "feature_highlights",
    "use_cases",
    "cta",
)


def validate_draft_system_prompt(content: str) -> None:
    """
    Keep Prompt Studio from reintroducing the empty-email bug:
    draft_system must document the template JSON keys (not alternate schemas).
    """
    lower = (content or "").lower()
    missing = [k for k in REQUIRED_DRAFT_JSON_KEYS if f'"{k}"' not in lower and f"'{k}'" not in lower and k not in lower]
    # Require the critical body keys at minimum
    critical = ["intro", "feature_highlights", "cta"]
    missing_critical = [k for k in critical if k not in lower]
    if missing_critical:
        raise ValueError(
            "Email Writing (System) must keep the template JSON keys "
            f"({', '.join(REQUIRED_DRAFT_JSON_KEYS)}). "
            f"Missing: {', '.join(missing_critical)}. "
            "Do not switch to alternate keys like opening_observation / positioning_line / ask."
        )


def update_prompt(key: str, content: str, organization_id: str | None = None) -> dict:
    if key == "draft_system":
        validate_draft_system_prompt(content)
    config = get_config(organization_id=organization_id)
    if key not in config.get("prompts", {}) and key not in _load_defaults().get("prompts", {}):
        raise KeyError(f"Unknown prompt key: {key}")
    if key not in config.setdefault("prompts", {}):
        config["prompts"][key] = copy.deepcopy(_load_defaults()["prompts"][key])
    config["prompts"][key]["content"] = content
    return save_config(config, organization_id)


def update_sender(updates: dict, organization_id: str | None = None) -> dict:
    config = get_config(organization_id=organization_id)
    sender = config.setdefault("sender", {})
    allowed = set(_load_defaults().get("sender", {}).keys())
    for k, v in updates.items():
        if k in allowed:
            sender[k] = v
    return save_config(config, organization_id)


def get_branding(organization_id: str | None = None) -> dict:
    config = get_config(organization_id=organization_id)
    defaults = _load_defaults().get("branding", {})
    branding = {**defaults, **(config.get("branding") or {})}
    # Prefer sender company when display_name empty
    if not branding.get("display_name"):
        branding["display_name"] = get_sender(organization_id).get("sender_company", "CRM Agent")
    if not branding.get("logo_url"):
        branding["logo_url"] = get_sender(organization_id).get("company_logo_url", "")
    return branding


def update_branding(updates: dict, organization_id: str | None = None) -> dict:
    config = get_config(organization_id=organization_id)
    branding = config.setdefault("branding", copy.deepcopy(_load_defaults().get("branding", {})))
    allowed = {"display_name", "logo_url", "accent_color"}
    for k, v in updates.items():
        if k in allowed and v is not None:
            branding[k] = v
    return save_config(config, organization_id)


def list_prompts(organization_id: str | None = None) -> list[dict]:
    config = get_config(organization_id=organization_id)
    defaults = _load_defaults()
    result = []
    for key, meta in defaults.get("prompts", {}).items():
        current = config.get("prompts", {}).get(key, meta)
        result.append({
            "key": key,
            "label": meta.get("label", key),
            "description": meta.get("description", ""),
            "category": meta.get("category", "general"),
            "variables": meta.get("variables", []),
            "content": current.get("content", meta.get("content", "")),
        })
    return result


def _template_path(name: str, organization_id: str | None = None) -> Path:
    if not name.endswith(".html"):
        name = f"{name}.html"
    custom = _custom_templates_dir(organization_id) / name
    if custom.exists():
        return custom
    builtin = _TEMPLATES_DIR / name
    if builtin.exists():
        return builtin
    raise FileNotFoundError(f"Template not found: {name}")


def list_templates(organization_id: str | None = None) -> list[dict]:
    org_dir = _custom_templates_dir(organization_id)
    seen: set[str] = set()
    templates = []
    for directory in (_TEMPLATES_DIR, org_dir):
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.html")):
            if path.name in seen:
                continue
            seen.add(path.name)
            meta = _BUILTIN_TEMPLATES.get(path.name, {"label": path.stem, "builtin": False})
            templates.append({
                "name": path.name,
                "label": meta.get("label", path.stem),
                "builtin": meta.get("builtin", False),
                "is_custom": path.parent == org_dir,
            })
    return templates


def get_template_content(name: str, organization_id: str | None = None) -> str:
    path = _template_path(name, organization_id)
    return path.read_text(encoding="utf-8")


def save_template(
    name: str,
    content: str,
    label: str | None = None,
    organization_id: str | None = None,
) -> dict:
    if not name.endswith(".html"):
        name = f"{name}.html"
    if ".." in name or "/" in name or "\\" in name:
        raise ValueError("Invalid template name")
    dest = _custom_templates_dir(organization_id) / name
    dest.write_text(content, encoding="utf-8")
    return {"name": name, "label": label or name, "builtin": False, "is_custom": True}


def delete_template(name: str, organization_id: str | None = None) -> None:
    if not name.endswith(".html"):
        name = f"{name}.html"
    custom = _custom_templates_dir(organization_id) / name
    if custom.exists():
        custom.unlink()
        return
    if name in _BUILTIN_TEMPLATES:
        return
    raise FileNotFoundError(f"Template not found: {name}")


def reset_template(name: str, organization_id: str | None = None) -> str:
    if not name.endswith(".html"):
        name = f"{name}.html"
    builtin = _TEMPLATES_DIR / name
    if not builtin.exists():
        raise FileNotFoundError(f"No built-in template: {name}")
    custom = _custom_templates_dir(organization_id) / name
    if custom.exists():
        custom.unlink()
    return builtin.read_text(encoding="utf-8")


def get_template_variables() -> list[dict]:
    return list(_TEMPLATE_VARIABLES)


def get_sample_preview_data(organization_id: str | None = None) -> dict:
    sender = get_sender(organization_id)
    data = copy.deepcopy(_SAMPLE_PREVIEW_DATA)
    data.update({
        "sender_name": sender.get("sender_name", data["sender_name"]),
        "sender_company": sender.get("sender_company", data["sender_company"]),
        "sender_phone": sender.get("sender_phone", data["sender_phone"]),
        "sender_website": sender.get("sender_website", data["sender_website"]),
        "sender_email": sender.get("sender_email", data["sender_email"]),
        "company_logo_url": sender.get("company_logo_url", data["company_logo_url"]),
    })
    return data


def preview_template(
    template_content: str | None = None,
    template_name: str | None = None,
    sample_data: dict | None = None,
    organization_id: str | None = None,
) -> str:
    from jinja2 import Environment, BaseLoader

    if template_content:
        source = template_content
    elif template_name:
        source = get_template_content(template_name, organization_id)
    else:
        raise ValueError("Provide template_content or template_name")

    data = sample_data or get_sample_preview_data(organization_id)
    env = Environment(loader=BaseLoader())
    template = env.from_string(source)
    return template.render(**data)
