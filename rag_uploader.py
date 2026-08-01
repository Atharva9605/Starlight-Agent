"""
RAG Uploader – unified entry point for both text and image-based PDF catalogues.

Detects whether the uploaded PDF is text-based or image-based and routes to
the correct ingestion pipeline:

  • Image-based (scanned) PDF  → catalogue_ingestor.py
       Uses GPT-4o Vision page-by-page, uploads page PNGs to Azure Blob Storage,
       stores rich product chunks with blob_url metadata in ChromaDB "starlight_vision".

  • Text-based PDF             → legacy text extraction path
       Uses PyMuPDF text + GPT-4o for extraction, stores in ChromaDB "starlight_vision"
       (same collection, so all catalogues are queried together).

Usage (Streamlit):
    from rag_uploader import process_pdf_to_chroma
    ok, msg = process_pdf_to_chroma(uploaded_file, progress_callback)
"""
import os
import io
import json
import tempfile
import logging

import fitz  # PyMuPDF

from azure_client import azure_manager
from config_manager import format_prompt, get_prompt
from vector_store import add_chunks, delete_by_source

log = logging.getLogger("rag_uploader")

COLLECTION_NAME = "starlight_vision"   # shared with catalogue_ingestor


# ---------------------------------------------------------------------------
# Detection helper
# ---------------------------------------------------------------------------

def _is_image_only(pdf_path: str, sample_pages: int = 5) -> bool:
    """
    Return True if the PDF contains no extractable text (i.e. is scanned).
    Samples up to `sample_pages` pages.
    """
    doc = fitz.open(pdf_path)
    pages_to_check = min(sample_pages, len(doc))
    text_found = 0
    for i in range(pages_to_check):
        text = doc[i].get_text("text").strip()
        if len(text) > 40:
            text_found += 1
    doc.close()
    return text_found == 0


# ---------------------------------------------------------------------------
# Text-based fallback pipeline (for PDFs that do contain text)
# ---------------------------------------------------------------------------

_TEXT_SYSTEM = """\
You are an expert extraction engine for LED lighting product catalogues.
Extract EVERY distinct product, series, or accessory from the supplied catalogue text.

For each item output a rich, self-contained description covering:
1. Product Name / Series
2. Application (residential, commercial, hospitality, retail, outdoor, etc.)
3. Key Features
4. All Technical Specifications: wattage (W), voltage (V), CCT (K), CRI, IP rating,
   beam angle (°), dimensions (mm), lumen output (lm), LED type, driver, mounting
5. Available variants (sizes, finishes, CCT options)

Output ONLY a valid JSON array of plain-text strings.
Each string = one product, fully self-contained. No markdown fences.
"""

_TEXT_USER_TMPL = "Extract all products from this catalogue section:\n\n{text}"

MAX_BATCH_CHARS = 4000


def _batch_pages(pages: list[str], max_chars: int = MAX_BATCH_CHARS) -> list[str]:
    batches: list[str] = []
    current: list[str] = []
    current_len = 0
    for page in pages:
        if current and current_len + len(page) > max_chars:
            batches.append("\n\n".join(current))
            current, current_len = [page], len(page)
        else:
            current.append(page)
            current_len += len(page)
    if current:
        batches.append("\n\n".join(current))
    return batches


def _extract_text_chunks(batch: str) -> list[str]:
    raw = azure_manager.chat_completion(
        [
            {"role": "system", "content": get_prompt("rag_text_system")},
            {"role": "user", "content": format_prompt(get_prompt("rag_text_user"), text=batch)},
        ],
        temperature=0.0,
        max_tokens=4096,
    )
    cleaned = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
    try:
        chunks = json.loads(cleaned)
        if isinstance(chunks, list):
            return [str(c).strip() for c in chunks if str(c).strip()]
    except json.JSONDecodeError:
        return [p.strip() for p in cleaned.split("\n\n") if len(p.strip()) > 80]
    return []


def _ingest_text_pdf(
    pdf_path: str,
    source_name: str,
    progress_callback=None,
) -> tuple[bool, str]:
    """
    Text-extraction pipeline for readable PDFs.

    Still renders every page to a PNG so the digital catalogue has real
    page images (diagrams / product art), not empty placeholders.
    """
    from azure_blob import blob_manager
    from catalogue_ingestor import (
        BLOB_IMAGE_MAX_W,
        _catalogue_display_name,
        _catalogue_slug,
        _render_page,
        _resize_png,
    )

    slug = _catalogue_slug(source_name)
    display_name = _catalogue_display_name(source_name)

    doc = fitz.open(pdf_path)
    total_pages = len(doc)

    all_chunks: list[str] = []
    all_metas: list[dict] = []
    all_ids: list[str] = []
    blob_url_map: dict[int, str] = {}
    pages_with_text = 0

    for page_idx in range(total_pages):
        page_num = page_idx + 1
        if progress_callback:
            pct = 0.08 + 0.78 * (page_idx / max(total_pages, 1))
            progress_callback(pct, f"Page {page_num}/{total_pages} — image + text…")

        page = doc[page_idx]

        # Always save the page image (diagrams / profile drawings live here)
        try:
            native_png = _render_page(page)
            blob_png = _resize_png(native_png, BLOB_IMAGE_MAX_W)
            blob_url = blob_manager.upload_page_image(blob_png, slug, page_num)
            blob_url_map[page_num] = blob_url
        except Exception:
            log.exception("Failed to render page %s of %s", page_num, source_name)
            blob_url = ""

        text = page.get_text("text").strip()
        if len(text) < 40:
            continue

        pages_with_text += 1
        page_chunks = _extract_text_chunks(f"[Page {page_num}]\n{text}")
        for i, chunk in enumerate(page_chunks):
            chunk = str(chunk).strip()
            if not chunk:
                continue
            # Prefer a clean product name from the first pipe segment
            name_guess = chunk.split("|", 1)[0].split("\n", 1)[0].strip().strip('[]"')
            if len(name_guess) > 80:
                name_guess = name_guess[:80].rsplit(" ", 1)[0]

            chunk_id = f"{source_name}::page::{page_num:03d}::text::{i:02d}"
            all_ids.append(chunk_id)
            all_chunks.append(chunk)
            all_metas.append({
                "source": source_name,
                "catalogue_name": display_name,
                "catalogue_slug": slug,
                "catalogue_type": "text_pdf",
                "page_number": page_num,
                "blob_url": blob_url,
                "product_name": name_guess,
                "category": "",
                "specs_preview": "",
                "char_count": len(chunk),
            })

    doc.close()

    if not all_chunks:
        return False, "No product information could be extracted."

    # Deduplicate by chunk text while keeping first page image association
    seen: set[str] = set()
    unique_docs: list[str] = []
    unique_metas: list[dict] = []
    unique_ids: list[str] = []
    for doc_text, meta, cid in zip(all_chunks, all_metas, all_ids):
        key = doc_text.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        unique_docs.append(doc_text)
        unique_metas.append(meta)
        unique_ids.append(cid)

    if progress_callback:
        progress_callback(0.90, f"Embedding {len(unique_docs)} chunks…")

    vectors = azure_manager.embed_documents(unique_docs)

    if progress_callback:
        progress_callback(0.95, "Writing to vector database…")

    delete_by_source(source_name)
    add_chunks(
        ids=unique_ids,
        embeddings=vectors,
        documents=unique_docs,
        metadatas=unique_metas,
    )

    try:
        from catalogue_library import seed_library_from_chunks, upsert_catalogue

        cover = blob_url_map.get(1) or next(iter(blob_url_map.values()), "")
        upsert_catalogue(
            slug=slug,
            name=display_name,
            source_filename=source_name,
            page_count=total_pages,
            cover_image_url=cover or "",
        )
        seeded = seed_library_from_chunks(source=source_name)
        log.info(
            "Text PDF library: %s pages=%d chunks=%d images=%d seeded=%s",
            source_name,
            pages_with_text,
            len(unique_docs),
            len(blob_url_map),
            seeded,
        )
    except Exception:
        log.exception("Structured catalogue library write failed for text PDF %s", source_name)

    if progress_callback:
        progress_callback(
            1.0,
            f"Done! {len(unique_docs)} products · {len(blob_url_map)} page images.",
        )

    return (
        True,
        f"Success: {len(unique_docs)} products + {len(blob_url_map)} page images from '{source_name}'.",
    )


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def process_pdf_to_chroma(
    pdf_file,
    progress_callback=None,
) -> tuple[bool, str]:
    """
    Ingest a catalogue PDF into the shared ChromaDB 'starlight_vision' collection.

    Automatically detects whether the PDF is image-only (uses GPT-4o Vision per
    page) or text-based (uses text extraction + GPT-4o), then ingests accordingly.

    Args:
        pdf_file:          File-like object with a ``.name`` attribute
                           (Streamlit UploadedFile or open() handle).
        progress_callback: Optional ``callable(fraction: float, message: str)``.

    Returns:
        ``(True, success_message)`` or ``(False, error_message)``.
    """
    temp_path = ""
    source_name: str = getattr(pdf_file, "name", "unknown_catalogue.pdf")

    try:
        if progress_callback:
            progress_callback(0.03, "Saving catalogue file…")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_file.read())
            temp_path = tmp.name

        # ── Detect PDF type ─────────────────────────────────────────────
        if progress_callback:
            progress_callback(0.06, "Detecting catalogue format (text vs image)…")

        is_image = _is_image_only(temp_path)
        log.info("'%s' is_image_only=%s", source_name, is_image)

        if is_image:
            # ── Vision pipeline ─────────────────────────────────────────
            if progress_callback:
                progress_callback(0.08, "Image-based PDF detected → GPT-4o Vision pipeline…")

            from catalogue_ingestor import ingest_catalogue

            # Wrap progress: ingestor uses 0–1 scale, we offset slightly
            def _wrapped_cb(pct, msg):
                if progress_callback:
                    # Map ingestor 0–1 into our 0.08–1.0 range
                    progress_callback(0.08 + 0.92 * pct, msg)

            # The ingestor takes a path and derives the catalogue name, slug and
            # dedup key from its basename. Copy into a temp *directory* rather
            # than prefixing the temp name, otherwise the random tmp segment
            # ends up in the display name and changes on every upload — which
            # makes force_reingest miss the previous rows and pile up duplicates.
            import shutil

            staging_dir = tempfile.mkdtemp()
            named_temp = os.path.join(staging_dir, os.path.basename(source_name))
            shutil.copy(temp_path, named_temp)

            try:
                summary = ingest_catalogue(
                    pdf_path=named_temp,
                    progress_callback=_wrapped_cb,
                    force_reingest=True,
                )
            finally:
                shutil.rmtree(staging_dir, ignore_errors=True)

            if summary["total_chunks"] == 0:
                return False, (
                    "No products were found in this catalogue. "
                    "Pages may be purely decorative or the quality may be too low."
                )

            return True, (
                f"Success: {summary['total_chunks']} product chunks indexed from "
                f"{summary['pages_processed']} pages of '{summary['catalogue_name']}'."
            )

        else:
            # ── Text pipeline ────────────────────────────────────────────
            if progress_callback:
                progress_callback(0.08, "Text-based PDF detected → extraction pipeline…")

            return _ingest_text_pdf(temp_path, source_name, progress_callback)

    except Exception as exc:
        log.error("Ingestion failed for '%s': %s", source_name, exc, exc_info=True)
        return False, f"Error processing catalogue: {exc}"

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
