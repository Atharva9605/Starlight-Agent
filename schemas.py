"""
Pydantic schemas for LLM structured outputs.

Treat model JSON as a typed API: validate, retry once on failure.
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional, Type, TypeVar

from pydantic import BaseModel, Field, ValidationError

T = TypeVar("T", bound=BaseModel)


class CompanyProfile(BaseModel):
    summary: str = ""
    products_services: str = ""
    target_audience: str = ""
    usp: str = ""
    key_phrases: str = ""
    tone_style: str = ""


class OutboundDraft(BaseModel):
    subject: str
    preamble: str = ""
    opening_line: str = ""
    intro: str = ""
    feature_highlights: list[str] = Field(default_factory=list)
    use_cases: list[str] = Field(default_factory=list)
    cta: str = ""


class ConversationDraft(BaseModel):
    subject: str
    body_text: str = ""
    body_html: str = ""
    banner_html: str = ""
    internal_note: str = ""


class VisionProduct(BaseModel):
    product_name: str = ""
    category: str = "other"
    description: str = ""
    features: list[str] = Field(default_factory=list)
    specs: dict[str, Any] = Field(default_factory=dict)
    variants: list[Any] = Field(default_factory=list)
    page_context: str = ""


class VisionProducts(BaseModel):
    products: list[VisionProduct] = Field(default_factory=list)


class DraftGateResult(BaseModel):
    should_draft: bool = True
    reason: str = ""


def extract_json_object(txt: str) -> dict | list | None:
    """Best-effort JSON extract from model text (fences / surrounding prose)."""
    if not txt:
        return None
    cleaned = txt.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    start_obj, end_obj = cleaned.find("{"), cleaned.rfind("}")
    if start_obj != -1 and end_obj > start_obj:
        try:
            return json.loads(cleaned[start_obj : end_obj + 1])
        except json.JSONDecodeError:
            pass
    start_arr, end_arr = cleaned.find("["), cleaned.rfind("]")
    if start_arr != -1 and end_arr > start_arr:
        try:
            return json.loads(cleaned[start_arr : end_arr + 1])
        except json.JSONDecodeError:
            pass
    return None


def parse_model(model: Type[T], raw: str | dict | list) -> T:
    if isinstance(raw, model):
        return raw
    if isinstance(raw, str):
        data = extract_json_object(raw)
        if data is None:
            raise ValueError("Model response is not valid JSON")
    else:
        data = raw

    # Vision sometimes returns a bare array
    if model is VisionProducts and isinstance(data, list):
        data = {"products": data}

    return model.model_validate(data)


def parse_with_retry(
    model: Type[T],
    raw: str,
    *,
    retry_fn=None,
) -> T:
    """
    Validate once; if fail and retry_fn provided, call it for a second attempt.
    retry_fn() -> str (new raw completion)
    """
    try:
        return parse_model(model, raw)
    except (ValidationError, ValueError) as first_err:
        if retry_fn is None:
            raise
        second = retry_fn()
        try:
            return parse_model(model, second)
        except (ValidationError, ValueError) as second_err:
            raise ValueError(
                f"Schema validation failed after retry: {second_err}"
            ) from first_err
