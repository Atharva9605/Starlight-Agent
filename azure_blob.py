"""
Blob Storage manager for Starlight catalogue page images.

Catalogue images are stored on the VPS only under:
  AI-CRM-Mailer/static/page_images/<slug>/page_NNN.png

They are served by the FastAPI app at:
  {PUBLIC_API_URL}/media/page_images/<slug>/page_NNN.png

Env:
  STATIC_SERVER_URL  — preferred public base ending in /media
                       e.g. https://api.mailer.starlightlinearled.com/media
  PUBLIC_API_URL     — fallback API origin ( /media is appended )
  BLOB_BACKEND       — ignored for catalogues (always local/VPS)

Legacy Cloudinary/Azure backends are disabled; catalogues stay on the VPS.
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger("azure_blob")

BLOB_BACKEND = os.getenv("BLOB_BACKEND", "local").lower().strip()
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static", "page_images")
_DEFAULT_API = "http://127.0.0.1:7860"


def media_base_url() -> str:
    """
    Public base URL for FastAPI StaticFiles mount at /media.
    Accepts either a full .../media URL or an API origin.
    """
    raw = (
        os.getenv("STATIC_SERVER_URL", "").strip()
        or os.getenv("PUBLIC_API_URL", "").strip()
        or _DEFAULT_API
    ).rstrip("/")
    if raw.endswith("/media"):
        return raw
    return f"{raw}/media"


def page_image_public_url(catalogue_slug: str, page_number: int | str) -> str:
    """Canonical public URL for a catalogue page image on this VPS."""
    slug = str(catalogue_slug or "").strip().strip("/")
    try:
        page = int(page_number)
    except (TypeError, ValueError):
        page = 0
    filename = f"page_{page:03d}.png"
    return f"{media_base_url()}/page_images/{slug}/{filename}"


def page_image_local_path(catalogue_slug: str, page_number: int | str) -> str:
    try:
        page = int(page_number)
    except (TypeError, ValueError):
        page = 0
    return os.path.join(
        STATIC_DIR,
        str(catalogue_slug).strip().strip("/"),
        f"page_{page:03d}.png",
    )


_PAGE_RE = re.compile(
    r"(?:/media)?/page_images/(?P<slug>[^/]+)/page_(?P<page>\d+)\.png",
    re.IGNORECASE,
)


def resolve_blob_url(
    blob_url: str | None,
    *,
    catalogue_slug: str = "",
    page_number: int | str = 0,
) -> str:
    """
    Map any stored blob_url (localhost, Cloudinary, Azure, stale host) onto the
    current VPS /media URL when we know catalogue_slug + page_number.
    """
    slug = (catalogue_slug or "").strip()
    try:
        page = int(page_number or 0)
    except (TypeError, ValueError):
        page = 0

    if slug and page > 0:
        return page_image_public_url(slug, page)

    raw = (blob_url or "").strip()
    if not raw or raw.startswith("["):
        return ""

    match = _PAGE_RE.search(raw.replace("\\", "/"))
    if match:
        return page_image_public_url(match.group("slug"), match.group("page"))

    if "/page_images/" in raw:
        tail = raw.split("/page_images/", 1)[1]
        return f"{media_base_url()}/page_images/{tail.lstrip('/')}"

    return raw


class _LocalBlobManager:
    """Saves page images under static/page_images/ and returns public /media URLs."""

    def upload_page_image(
        self,
        image_bytes: bytes,
        catalogue_slug: str,
        page_number: int,
        **_kwargs,
    ) -> str:
        folder = os.path.join(STATIC_DIR, catalogue_slug)
        os.makedirs(folder, exist_ok=True)

        filename = f"page_{page_number:03d}.png"
        filepath = os.path.join(folder, filename)
        with open(filepath, "wb") as f:
            f.write(image_bytes)

        url = page_image_public_url(catalogue_slug, page_number)
        log.info("Saved catalogue page image: %s → %s", filepath, url)
        return url


def _make_manager():
    if BLOB_BACKEND in ("cloudinary", "azure"):
        log.warning(
            "BLOB_BACKEND=%s ignored — catalogue images are stored on the VPS only",
            BLOB_BACKEND,
        )
    log.info("Blob backend: local VPS  (%s)  media=%s", STATIC_DIR, media_base_url())
    return _LocalBlobManager()


class AzureBlobManager:
    """Public façade — catalogues always use the local VPS backend."""

    def __init__(self) -> None:
        self._backend = None

    def _get_backend(self):
        if self._backend is None:
            self._backend = _make_manager()
        return self._backend

    @property
    def available(self) -> bool:
        return True

    def upload_page_image(
        self,
        image_bytes: bytes,
        catalogue_slug: str,
        page_number: int,
        content_type: str = "image/png",
    ) -> str:
        try:
            return self._get_backend().upload_page_image(
                image_bytes=image_bytes,
                catalogue_slug=catalogue_slug,
                page_number=page_number,
            )
        except Exception as exc:
            log.error("Blob upload failed (page %d): %s", page_number, exc)
            try:
                return _LocalBlobManager().upload_page_image(
                    image_bytes, catalogue_slug, page_number
                )
            except Exception:
                return f"[upload_failed]/{catalogue_slug}/page_{page_number:03d}.png"


blob_manager = AzureBlobManager()
Path(STATIC_DIR).mkdir(parents=True, exist_ok=True)
