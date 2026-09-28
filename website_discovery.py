"""
Resolve company websites via self-hosted OpenSERP.

No API keys and no per-search billing — points at a local/docker OpenSERP
instance (see OPENSERP_BASE_URL). Used when a lead row has a name/company
but no website, before the existing scrape → draft pipeline.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any
from urllib.parse import urlparse

import requests

log = logging.getLogger(__name__)

OPENSERP_BASE_URL = os.getenv("OPENSERP_BASE_URL", "http://127.0.0.1:7000").rstrip("/")
OPENSERP_TIMEOUT = float(os.getenv("OPENSERP_TIMEOUT", "60"))
# duckduckgo / ecosia / bing tend to work better without CAPTCHA than google
OPENSERP_ENGINES = os.getenv("OPENSERP_ENGINES", "duckduckgo,bing,ecosia")

_EMPTY = frozenset({"", "nan", "none", "null", "-", "n/a", "na"})

# Directories / social / aggregators — never treat these as the company site
_SKIP_HOST_SUFFIXES = (
    "wikipedia.org",
    "linkedin.com",
    "facebook.com",
    "fb.com",
    "twitter.com",
    "x.com",
    "instagram.com",
    "youtube.com",
    "youtu.be",
    "tiktok.com",
    "crunchbase.com",
    "bloomberg.com",
    "glassdoor.com",
    "indeed.com",
    "yelp.com",
    "yellowpages.com",
    "mapquest.com",
    "bing.com",
    "google.com",
    "google.co",
    "duckduckgo.com",
    "baidu.com",
    "yandex.ru",
    "yandex.com",
    "amazon.com",
    "ebay.com",
    "reddit.com",
    "pinterest.com",
    "zoominfo.com",
    "dnb.com",
    "apollo.io",
    "rocketreach.co",
    "owler.com",
    "craft.co",
    "pitchbook.com",
    "forbes.com",
    "medium.com",
    "wordpress.com",
    "blogspot.com",
    "wixsite.com",
    "squarespace.com",
)


def clean_lead_value(val: Any) -> str:
    if val is None:
        return ""
    text = str(val).strip()
    if not text or text.lower() in _EMPTY:
        return ""
    return text


def lead_search_name(lead: dict | None) -> str:
    """Best company/business name from a lead row for SERP lookup."""
    if not lead:
        return ""
    # Prefer post-normalize keys first
    for key in ("company", "name"):
        found = clean_lead_value(lead.get(key))
        if found:
            return found
    for key, val in lead.items():
        norm = str(key).strip().lower().replace("-", "_").replace(" ", "_")
        if norm in (
            "company",
            "name",
            "company_name",
            "business",
            "business_name",
            "organisation",
            "organization",
            "org",
            "firm",
            "lead_name",
            "account_name",
            "prospect",
        ):
            found = clean_lead_value(val)
            if found:
                return found
    return ""


def _host(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower().removeprefix("www.")
    except Exception:
        return ""


def _hosts_from(values: list[str] | None) -> set[str]:
    out: set[str] = set()
    for val in values or []:
        text = clean_lead_value(val)
        if not text:
            continue
        url = text if re.match(r"^https?://", text, re.I) else "https://" + text.lstrip("/")
        host = _host(url)
        if host:
            out.add(host)
    return out


def _host_excluded(host: str, exclude_hosts: set[str]) -> bool:
    if not host or host in exclude_hosts:
        return True
    return any(host == b or host.endswith("." + b) for b in exclude_hosts)


def _is_skipped_host(host: str) -> bool:
    if not host:
        return True
    return any(host == s or host.endswith("." + s) for s in _SKIP_HOST_SUFFIXES)


def _normalize_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return ""
    if not re.match(r"^https?://", url, re.I):
        url = "https://" + url.lstrip("/")
    # Prefer homepage: drop deep paths for discovery (keep host only)
    parsed = urlparse(url)
    if not parsed.netloc:
        return ""
    scheme = parsed.scheme or "https"
    return f"{scheme}://{parsed.netloc}/"


def _name_tokens(name: str) -> list[str]:
    raw = re.sub(r"[^\w\s]", " ", name.lower())
    stop = {
        "the", "and", "of", "for", "inc", "llc", "ltd", "co", "corp", "corporation",
        "company", "companies", "group", "pvt", "private", "limited", "plc",
    }
    return [t for t in raw.split() if len(t) > 1 and t not in stop]


def _score_result(url: str, title: str, snippet: str, query: str) -> float:
    host = _host(url)
    if _is_skipped_host(host):
        return -1.0
    tokens = _name_tokens(query)
    if not tokens:
        return 0.0
    hay = f"{host} {title} {snippet}".lower()
    host_compact = host.replace(".", "").replace("-", "")
    hits = 0.0
    for t in tokens:
        if t in host or t in host_compact:
            hits += 2.0
        elif t in hay:
            hits += 1.0
    # Prefer shorter apex-ish hosts (company.com over blog.company.com/foo)
    depth_penalty = host.count(".") * 0.15
    return hits - depth_penalty


def _pick_best_url(
    results: list[dict], query: str, exclude_hosts: set[str] | None = None
) -> str | None:
    banned = exclude_hosts or set()
    ranked: list[tuple[float, str]] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        if str(item.get("type") or "organic").lower() not in ("organic", "web", ""):
            continue
        raw = clean_lead_value(item.get("url") or item.get("link"))
        if not raw:
            continue
        url = _normalize_url(raw)
        if not url:
            continue
        host = _host(url)
        if _host_excluded(host, banned):
            continue
        score = _score_result(
            url,
            clean_lead_value(item.get("title")),
            clean_lead_value(item.get("snippet") or item.get("description")),
            query,
        )
        if score < 0:
            continue
        ranked.append((score, url))
    if not ranked:
        for item in results:
            if not isinstance(item, dict):
                continue
            raw = clean_lead_value(item.get("url") or item.get("link"))
            url = _normalize_url(raw)
            if url and not _is_skipped_host(_host(url)) and not _host_excluded(_host(url), banned):
                return url
        return None
    ranked.sort(key=lambda x: x[0], reverse=True)
    best_score, best_url = ranked[0]
    if _name_tokens(query) and best_score < 1.0:
        return best_url
    return best_url


def discover_website(company_name: str, exclude_hosts: set[str] | None = None) -> str | None:
    """
    Query OpenSERP for the official website of ``company_name``.
    Returns a normalized homepage URL, or None if nothing usable is found.
    """
    query = clean_lead_value(company_name)
    if not query:
        return None

    text = f"{query} official website"
    params = {
        "text": text,
        "limit": 8,
        "mode": "any",
        "engines": OPENSERP_ENGINES,
        "format": "json",
    }
    url = f"{OPENSERP_BASE_URL}/mega/search"
    try:
        resp = requests.get(url, params=params, timeout=OPENSERP_TIMEOUT)
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:
        log.warning("OpenSERP lookup failed for %r: %s", query, exc)
        return None

    results = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(results, list):
        log.warning("OpenSERP returned no results list for %r", query)
        return None

    chosen = _pick_best_url(results, query, exclude_hosts=exclude_hosts)
    if chosen:
        log.info("OpenSERP resolved %r → %s", query, chosen)
    else:
        log.warning("OpenSERP found no suitable website for %r", query)
    return chosen


def ensure_lead_website(
    lead: dict,
    *,
    force_discover: bool = False,
    exclude_websites: list[str] | None = None,
) -> str:
    """
    Ensure ``lead['website']`` is set. Uses OpenSERP when the row only has a name,
    or when ``force_discover`` is set (failed-site retry).
    """
    website = clean_lead_value(lead.get("website"))
    exclude = _hosts_from(exclude_websites)

    if force_discover:
        if website:
            exclude |= _hosts_from([website])
        name = lead_search_name(lead)
        if not name and website:
            host = _host(website if "://" in website else "https://" + website)
            name = host.split(".")[0].replace("-", " ") if host else ""
        if not name:
            raise ValueError("Need a company name to retry via OpenSERP")
        found = discover_website(name, exclude_hosts=exclude)
        if not found:
            raise ValueError(
                f"OpenSERP found no other website for '{name}'"
                + (f" besides {website}" if website else "")
            )
        lead["website"] = found
        if not clean_lead_value(lead.get("company")):
            lead["company"] = name
        return found

    if website:
        if not re.match(r"^https?://", website, re.I):
            website = "https://" + website.lstrip("/")
        lead["website"] = website
        return website

    name = lead_search_name(lead)
    if not name:
        raise ValueError("Lead must include a website or a company/name")

    found = discover_website(name, exclude_hosts=exclude)
    if not found:
        raise ValueError(f"Could not find a website for '{name}' via OpenSERP")

    lead["website"] = found
    if not clean_lead_value(lead.get("company")):
        lead["company"] = name
    return found
