"""
Offline golden evals for CRM Agent AI quality.

Usage (from AI-CRM-Mailer/):
  python evals/run_evals.py

These checks are schema / policy unit tests that do not require Azure.
Add live LLM evals later behind EVAL_LIVE=1.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from schemas import (
    CompanyProfile,
    ConversationDraft,
    OutboundDraft,
    VisionProducts,
    parse_model,
)
from draft_gate import should_auto_draft


def _ok(name: str) -> None:
    print(f"  PASS  {name}")


def _fail(name: str, err: Exception) -> None:
    print(f"  FAIL  {name}: {err}")
    raise SystemExit(1)


def test_schemas() -> None:
    parse_model(
        CompanyProfile,
        {
            "summary": "Architecture firm",
            "products_services": "design",
            "target_audience": "hotels",
            "usp": "boutique",
            "key_phrases": "hospitality",
            "tone_style": "elegant",
        },
    )
    _ok("CompanyProfile")

    parse_model(
        OutboundDraft,
        {
            "subject": "Hello",
            "preamble": "Light done right",
            "opening_line": "Hi team,",
            "intro": "Loved your lobby work.",
            "feature_highlights": ["A", "B"],
            "use_cases": ["Lobby"],
            "cta": "Free for a call?",
        },
    )
    _ok("OutboundDraft")

    parse_model(
        ConversationDraft,
        {
            "subject": "Re: Quote",
            "body_text": "Happy to share options.",
            "body_html": "<p>Happy to share options.</p>",
            "banner_html": "",
            "internal_note": "Asked for pricing",
        },
    )
    _ok("ConversationDraft")

    parse_model(
        VisionProducts,
        {"products": [{"product_name": "SL-1", "category": "linear"}]},
    )
    parse_model(VisionProducts, [{"product_name": "SL-2"}])
    _ok("VisionProducts array wrap")


def test_draft_gate() -> None:
    g = should_auto_draft("thanks")
    assert not g.should_draft
    _ok("gate skips thanks")

    g = should_auto_draft("Out of Office: I am away until Monday")
    assert not g.should_draft
    _ok("gate skips OOO")

    g = should_auto_draft(
        "Can you share IP65 linear options for the pool deck and a rough budget?"
    )
    assert g.should_draft
    _ok("gate allows real inquiry")


def test_fixtures_roundtrip() -> None:
    fixture = ROOT / "evals" / "fixtures" / "sample_outbound_draft.json"
    fixture.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "subject": "Lighting for your hospitality portfolio",
        "preamble": "Engineered for ambience",
        "opening_line": "Hope you are well.",
        "intro": "Your recent hotel lobby project stood out.",
        "feature_highlights": ["High CRI", "Custom lengths"],
        "use_cases": ["Lobby coves"],
        "cta": "Open to a 15-minute call?",
    }
    fixture.write_text(json.dumps(data, indent=2), encoding="utf-8")
    parse_model(OutboundDraft, fixture.read_text(encoding="utf-8"))
    _ok("fixture OutboundDraft")


if __name__ == "__main__":
    print("Running CRM Agent evals…")
    try:
        test_schemas()
        test_draft_gate()
        test_fixtures_roundtrip()
    except SystemExit:
        raise
    except Exception as exc:
        _fail("unexpected", exc)
    print("All evals passed.")
