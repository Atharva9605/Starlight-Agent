"""Branded Starlight product recommendation PDF for campaign email attachments."""
from __future__ import annotations

import logging
import re
from typing import Any
from urllib.request import Request, urlopen

log = logging.getLogger("product_sheet")

# Brand palette (Starlight Linear LED)
NAVY = (15, 23, 42)
BLUE = (37, 99, 235)
CYAN = (6, 182, 212)
SLATE = (71, 85, 105)
LIGHT = (248, 250, 252)
WHITE = (255, 255, 255)
BORDER = (226, 232, 240)


def _slug(s: str) -> str:
    s = re.sub(r"[^\w\-]+", "-", (s or "products").strip()).strip("-").lower()
    return (s[:50] or "products")


def _fetch_image(url: str, timeout: float = 8.0) -> bytes | None:
    if not url or not str(url).startswith("http"):
        return None
    try:
        req = Request(url, headers={"User-Agent": "StarlightMailer/1.0"})
        with urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        if not data or data[:40].startswith(b"version https://git-lfs"):
            return None
        if len(data) > 8 * 1024 * 1024:
            return None
        return data
    except Exception as exc:
        log.debug("Could not fetch product image %s: %s", url, exc)
        return None


def build_product_sheet_pdf(
    products: list[dict[str, Any]],
    *,
    sender: dict[str, Any] | None = None,
    client_company: str = "",
    catalogue_url: str = "",
) -> tuple[bytes, str]:
    """
    Build a branded multi-card PDF of suggested products.
    Returns (pdf_bytes, filename).
    """
    import fitz

    sender = sender or {}
    company = sender.get("sender_company") or "Starlight Linear LED"
    sender_name = sender.get("sender_name") or ""
    phone = sender.get("sender_phone") or ""
    website = sender.get("sender_website") or "www.starlightlinearled.com"
    email = sender.get("sender_email") or ""
    logo_url = sender.get("company_logo_url") or ""

    items = [p for p in (products or []) if p][:8]
    if not items:
        raise ValueError("No products to include in the sheet")

    doc = fitz.open()
    page = doc.new_page(width=595, height=842)  # A4
    W, H = page.rect.width, page.rect.height
    margin = 36

    def new_page():
        nonlocal page
        page = doc.new_page(width=595, height=842)
        return page

    # ── Header bar ────────────────────────────────────────────
    header_h = 78
    page.draw_rect(fitz.Rect(0, 0, W, header_h), color=None, fill=NAVY)
    page.draw_rect(fitz.Rect(0, header_h - 4, W, header_h), color=None, fill=BLUE)

    logo_bytes = _fetch_image(logo_url) if logo_url.startswith("http") else None
    text_x = margin
    if logo_bytes:
        try:
            logo_rect = fitz.Rect(margin, 16, margin + 44, 60)
            page.insert_image(logo_rect, stream=logo_bytes)
            text_x = margin + 56
        except Exception:
            pass

    page.insert_text((text_x, 32), company, fontsize=16, fontname="helv", color=WHITE)
    page.insert_text(
        (text_x, 50),
        "Recommended product sheet",
        fontsize=10,
        fontname="helv",
        color=(191, 219, 254),
    )
    if client_company:
        page.insert_text(
            (W - margin - 180, 36),
            f"Prepared for {client_company[:40]}",
            fontsize=9,
            fontname="helv",
            color=(186, 230, 253),
        )

    y = header_h + 22
    intro = (
        "The following Starlight Linear LED products were selected for this outreach "
        "from our digital catalogue. Full specifications are available online."
    )
    y = _wrap_text(page, intro, margin, y, W - 2 * margin, fontsize=10, color=SLATE)
    y += 14

    # ── Product cards ─────────────────────────────────────────
    for i, prod in enumerate(items):
        name = str(prod.get("product_name") or f"Product {i + 1}").strip()
        category = str(prod.get("category") or "").strip()
        specs = str(prod.get("specs_preview") or "").strip()
        cat_name = str(prod.get("catalogue_name") or "").strip()
        page_no = prod.get("page_number")
        img_url = str(prod.get("blob_url") or "").strip()
        prod_cat_url = str(prod.get("catalogue_url") or catalogue_url or "").strip()

        card_h = 128
        if y + card_h > H - 70:
            _draw_footer(page, W, H, margin, company, phone, email, website)
            page = new_page()
            y = margin

        card = fitz.Rect(margin, y, W - margin, y + card_h)
        page.draw_rect(card, color=BORDER, fill=LIGHT, width=0.6)
        # Accent bar
        page.draw_rect(
            fitz.Rect(margin, y, margin + 4, y + card_h),
            color=None,
            fill=BLUE if i % 2 == 0 else CYAN,
        )

        img_w = 100
        content_x = margin + 16
        img_data = _fetch_image(img_url)
        if img_data:
            try:
                irect = fitz.Rect(margin + 14, y + 12, margin + 14 + img_w, y + card_h - 12)
                page.draw_rect(irect, color=BORDER, fill=WHITE, width=0.4)
                page.insert_image(irect + (2, 2, -2, -2), stream=img_data, keep_proportion=True)
                content_x = margin + 14 + img_w + 14
            except Exception:
                content_x = margin + 16

        ty = y + 22
        page.insert_text((content_x, ty), name[:70], fontsize=12, fontname="helv", color=NAVY)
        ty += 16
        meta_bits = [b for b in [category, cat_name, f"p.{page_no}" if page_no else ""] if b]
        if meta_bits:
            page.insert_text(
                (content_x, ty),
                " · ".join(meta_bits)[:90],
                fontsize=8,
                fontname="helv",
                color=SLATE,
            )
            ty += 14
        if specs:
            ty = _wrap_text(page, specs[:220], content_x, ty, W - margin - content_x - 8, fontsize=9, color=SLATE)
        if prod_cat_url:
            ty += 8
            page.insert_text(
                (content_x, min(ty, y + card_h - 14)),
                f"View in digital catalogue → {prod_cat_url[:70]}",
                fontsize=8,
                fontname="helv",
                color=BLUE,
            )

        y += card_h + 12

    if catalogue_url:
        if y + 40 > H - 70:
            _draw_footer(page, W, H, margin, company, phone, email, website)
            page = new_page()
            y = margin
        page.insert_text((margin, y + 12), "Browse the full digital catalogue:", fontsize=10, fontname="helv", color=NAVY)
        page.insert_text((margin, y + 28), catalogue_url[:100], fontsize=9, fontname="helv", color=BLUE)

    _draw_footer(page, W, H, margin, company, phone, email, website)

    pdf_bytes = doc.tobytes()
    doc.close()

    label = _slug(client_company or "starlight")
    filename = f"Starlight_Product_Sheet_{label}.pdf"
    return pdf_bytes, filename


def _draw_footer(page, W, H, margin, company, phone, email, website) -> None:
    import fitz

    y = H - 42
    page.draw_rect(fitz.Rect(0, y - 8, W, H), color=None, fill=NAVY)
    line = " · ".join(p for p in [company, phone, email, website] if p)
    page.insert_text((margin, y + 10), line[:110], fontsize=8, fontname="helv", color=(203, 213, 225))
    page.insert_text(
        (margin, y + 24),
        "Starlight Linear LED — architectural lighting, manufactured in India",
        fontsize=7,
        fontname="helv",
        color=(148, 163, 184),
    )


def _wrap_text(page, text: str, x: float, y: float, max_w: float, *, fontsize: int, color) -> float:
    import fitz

    words = (text or "").split()
    line = ""
    for w in words:
        trial = f"{line} {w}".strip()
        if fitz.get_text_length(trial, fontsize=fontsize, fontname="helv") > max_w:
            if line:
                page.insert_text((x, y), line, fontsize=fontsize, fontname="helv", color=color)
                y += fontsize + 4
            line = w
        else:
            line = trial
    if line:
        page.insert_text((x, y), line, fontsize=fontsize, fontname="helv", color=color)
        y += fontsize + 4
    return y


def write_product_sheet(
    out_path: str,
    products: list[dict[str, Any]],
    **kwargs,
) -> str:
    """Write PDF to disk; return path."""
    data, _name = build_product_sheet_pdf(products, **kwargs)
    with open(out_path, "wb") as f:
        f.write(data)
    return out_path
