"""
Prepare rendered HTML for delivery to real mail clients.

Templates are authored with a single <style> block and CSS classes, which the
in-app preview iframe renders faithfully. Gmail, Outlook and most mobile
clients strip or partially apply that block, so the delivered mail looks
nothing like the preview. Inlining the CSS onto each element fixes that.
"""
from __future__ import annotations

import logging
import os
import re
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

log = logging.getLogger("email_html")

LOGO_CID = "company_logo"


def inline_css(html: str, base_url: str = "") -> str:
    """
    Inline <style> rules as style="" attributes.

    Returns the original HTML unchanged if premailer is unavailable or the
    document cannot be transformed — sending a slightly degraded email beats
    failing the send.
    """
    if not html or "<style" not in html.lower():
        return html

    try:
        from premailer import transform
    except ImportError:
        log.warning("premailer not installed — sending email with non-inlined CSS")
        return html

    try:
        return transform(
            html,
            base_url=base_url or None,
            keep_style_tags=True,          # retain @media rules for clients that honour them
            strip_important=False,
            cssutils_logging_level=logging.CRITICAL,
            disable_validation=True,
        )
    except Exception as exc:
        log.warning("CSS inlining failed (%s) — sending original HTML", exc)
        return html


def html_to_text(html: str) -> str:
    """Plain-text alternative so the mail isn't HTML-only (helps deliverability)."""
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html or "", flags=re.I | re.S)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"</(p|div|tr|h[1-6]|li)>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def build_eml_message(
    *,
    subject: str,
    html: str,
    from_addr: str,
    to_addr: str,
    logo_path: str = "",
    embed_logo: bool = True,
) -> MIMEMultipart:
    """
    Build a mail-client-friendly message.

    Structure is multipart/related wrapping multipart/alternative so the inline
    logo resolves against cid: references in the HTML. Building the .eml the
    same way at generation and at send time keeps the delivered mail identical
    to what was previewed.
    """
    alternative = MIMEMultipart("alternative")
    alternative.attach(MIMEText(html_to_text(html), "plain", "utf-8"))
    alternative.attach(MIMEText(html, "html", "utf-8"))

    root = MIMEMultipart("related")
    root["Subject"] = subject
    if from_addr:
        root["From"] = from_addr
    if to_addr:
        root["To"] = to_addr
    root.attach(alternative)

    references_cid = f"cid:{LOGO_CID}" in (html or "")
    if embed_logo and references_cid and logo_path and os.path.exists(logo_path):
        img = _logo_part(logo_path)
        if img is not None:
            root.attach(img)

    return root


def _logo_part(logo_path: str) -> MIMEImage | None:
    """
    Build the inline logo part, or None if the file isn't a usable image.

    MIMEImage raises TypeError when it cannot detect the format, which would
    otherwise abort the whole send. A missing logo is not worth failing a
    campaign over, so the mail goes out without it and the cause is logged.
    """
    try:
        with open(logo_path, "rb") as f:
            data = f.read()
    except OSError as exc:
        log.warning("Could not read logo %s: %s", logo_path, exc)
        return None

    if data[:40].startswith(b"version https://git-lfs"):
        log.warning(
            "Logo %s is an unresolved Git LFS pointer, not an image — "
            "emails will be sent without the inline logo",
            logo_path,
        )
        return None

    try:
        img = MIMEImage(data)
    except (TypeError, ValueError) as exc:
        log.warning("Logo %s is not a usable image (%s)", logo_path, exc)
        return None

    img.add_header("Content-ID", f"<{LOGO_CID}>")
    img.add_header("Content-Disposition", "inline", filename=os.path.basename(logo_path))
    return img
