"""Helpers to keep only real sellable lighting products in the public catalogue."""
from __future__ import annotations

import json
import re
from typing import Any

# Pages / extractions that are not products customers browse
_NON_PRODUCT_NAME = re.compile(
    r"("
    r"\bBIS\b|certificate|certification|license|licence|"
    r"\bIS\s*10322\b|bureau of indian|"
    r"cover\b|contents\b|table of contents|index\b|application index|"
    r"about us|company profile|contact\b|warranty policy|"
    r"fixed general purpose|"
    r"smart collection\b|"
    r"profiles continued|"
    r"^page\s*\d+$"
    r")",
    re.I,
)

_NON_PRODUCT_CATEGORY = re.compile(
    r"("
    r"certificate|certification|compliance|document|cover|index|"
    r"fixed general purpose"
    r")",
    re.I,
)

_PRODUCT_CODE = re.compile(r"\bSLL[A-Z0-9\-]+\b", re.I)
_DIMENSION = re.compile(r"\d+(\.\d+)?\s*[x×]\s*\d+(\.\d+)?", re.I)


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value if str(v).strip())
    if isinstance(value, dict):
        try:
            return json.dumps(value, ensure_ascii=False)
        except Exception:
            return str(value)
    return str(value).strip()


def normalize_specs(specs: Any) -> dict[str, str]:
    """Flatten messy vision/text specs into clean string values."""
    if not isinstance(specs, dict):
        return {}
    out: dict[str, str] = {}
    for k, v in specs.items():
        key = re.sub(r"[^a-z0-9]+", "_", str(k).strip().lower()).strip("_")
        if not key or key in {"anomaly", "product", "raw"}:
            continue
        if isinstance(v, (list, tuple)):
            text = ", ".join(str(x).strip() for x in v if str(x).strip())
        else:
            text = _as_text(v)
        # Avoid dumping raw Python/JSON arrays into the UI
        if text.startswith("[") and "SLL" in text.upper():
            try:
                parsed = json.loads(text.replace("'", '"'))
                if isinstance(parsed, list):
                    text = ", ".join(str(x) for x in parsed)
            except Exception:
                text = re.sub(r"[\[\]'\"]", "", text)
        if not text or text.lower() in {"not stated", "n/a", "none", "-"}:
            continue
        # Skip absurd label keys pasted as values
        if len(key) > 48:
            continue
        # CCT lists get huge — keep a readable summary for catalogue UI
        if key in {"cct", "colour_temp_range", "color_temp"} and len(text) > 72:
            parts = re.split(r"\s*,\s*", text)
            text = ", ".join(parts[:6]) + ("…" if len(parts) > 6 else "")
        out[key] = text[:220]
    return out


def is_catalogue_product(product: dict[str, Any]) -> bool:
    """
    Return True only for real lighting products suitable for a public catalogue.

    Drops BIS certificates, covers, contents pages, and weak extractions rows.
    """
    name = _as_text(product.get("product_name") or product.get("name") or "")
    category = _as_text(product.get("category") or "")
    description = _as_text(product.get("description") or "")
    specs = normalize_specs(product.get("specs") or {})
    blob = _as_text(product.get("image_url") or product.get("blob_url") or "")

    if not name or len(name) < 2:
        return False
    if _NON_PRODUCT_NAME.search(name):
        return False
    if _NON_PRODUCT_CATEGORY.search(category):
        return False
    if _NON_PRODUCT_NAME.search(description[:180]):
        return False

    # Certificate-ish rows often only have brand / IS no / models
    spec_keys = set(specs.keys())
    if spec_keys and spec_keys <= {"brand", "is_no", "models", "is", "standard"}:
        return False
    if any(k.startswith("fixed_general") for k in spec_keys):
        return False

    has_code = bool(specs.get("code") or _PRODUCT_CODE.search(name) or _PRODUCT_CODE.search(json.dumps(specs)))
    has_dims = bool(specs.get("dimensions") or any(_DIMENSION.search(v) for v in specs.values()))
    has_electrical = bool(specs.get("wattage") or specs.get("voltage") or specs.get("ip_rating") or specs.get("ip"))

    # Keep rows that look like real fixtures
    if has_code and (has_dims or has_electrical):
        return True
    if has_dims and has_electrical and blob:
        return True
    # Named products with an image and at least one hard spec
    if blob and name and (has_code or has_dims or has_electrical) and len(specs) >= 2:
        return True

    return False


def sanitize_product_for_public(product: dict[str, Any]) -> dict[str, Any]:
    """Clean a product dict for public API response."""
    specs = normalize_specs(product.get("specs") or {})
    features = product.get("features") or []
    if isinstance(features, str):
        features = [features]
    features = [str(f).strip() for f in features if str(f).strip()][:6]

    preview_keys = ("code", "dimensions", "wattage", "ip_rating", "voltage")
    preview_parts = [specs[k] for k in preview_keys if k in specs]
    # Keep CCT out of the one-line preview (too long)
    preview = " · ".join(preview_parts[:4])

    return {
        "id": product.get("id"),
        "product_name": _as_text(product.get("product_name"))[:120],
        "category": _as_text(product.get("category"))[:40],
        "description": _as_text(product.get("description"))[:280],
        "features": features,
        "specs": specs,
        "variants": product.get("variants") if isinstance(product.get("variants"), list) else [],
        "page_number": int(product.get("page_number") or 0),
        "image_url": _as_text(product.get("image_url") or product.get("blob_url") or ""),
        "specs_preview": preview or _as_text(product.get("specs_preview"))[:120],
    }
