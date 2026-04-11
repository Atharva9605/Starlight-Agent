"""
RAG Uploader – Lighting Catalogue PDF → ChromaDB vector store.

Pipeline:
  1. Extract raw text per page using PyMuPDF (fitz)
  2. Batch pages into manageable text windows
  3. Use Azure OpenAI GPT-4o to parse each window into rich product chunk strings
  4. Embed chunks with Azure OpenAI text-embedding-3-large
  5. Upsert into ChromaDB (replaces existing entries for the same source)

Usage (from Streamlit or standalone):
    from rag_uploader import process_pdf_to_chroma
    success, msg = process_pdf_to_chroma(pdf_file_object, progress_callback)
"""
import os
import json
import tempfile
import logging

import chromadb
import fitz  # PyMuPDF

from azure_client import azure_manager

log = logging.getLogger("rag_uploader")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CHROMA_DB_DIR = "chroma_db"
COLLECTION_NAME = "starlight_catalogs"
MAX_BATCH_CHARS = 4000  # Characters per text batch sent to GPT-4o

# ---------------------------------------------------------------------------
# Extraction prompt
# ---------------------------------------------------------------------------
_EXTRACTION_SYSTEM = """\
You are a precision extraction engine for professional LED lighting catalogues.

Extract EVERY distinct product, product series, module, or accessory from the supplied catalogue text.

For each item output a rich, self-contained description that covers:
1. Product Name / Series
2. Application (residential, commercial, hospitality, retail, outdoor, architectural cove, etc.)
3. Key Features (optics, dimming, colour-tuning, IP rating, design highlights)
4. Technical Specifications — include ALL numbers found:
   Wattage (W), Input Voltage (V/VAC/VDC), CCT range (K), CRI, Beam Angle (°),
   IP Rating, Dimensions (mm), Lumen Output (lm), LED Type, Driver type,
   Mounting type, Material / Finish options
5. Available Variants (sizes, finishes, CCT options, wattages)

RULES:
- Output ONLY a valid JSON array of plain-text strings.
- Each string = one product/series, fully self-contained (it is embedded independently).
- Include every numerical specification you find — do NOT omit specs.
- Do NOT invent numbers or features not present in the text.
- Do NOT wrap the array in markdown code fences.
"""

_EXTRACTION_USER_TMPL = """\
Extract all distinct lighting products from this catalogue section and output a JSON array of rich description strings.

CATALOGUE TEXT:
---
{text}
---
"""


# ---------------------------------------------------------------------------
# PDF helpers
# ---------------------------------------------------------------------------

def _extract_pages(pdf_path: str) -> list[str]:
    """Return a list of non-empty page text strings from the PDF."""
    doc = fitz.open(pdf_path)
    pages: list[str] = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        if len(text) > 40:  # skip nearly-blank pages
            pages.append(f"[Page {page_num}]\n{text}")
    doc.close()
    return pages


def _batch_pages(pages: list[str], max_chars: int = MAX_BATCH_CHARS) -> list[str]:
    """Combine consecutive pages into batches under max_chars each."""
    batches: list[str] = []
    current: list[str] = []
    current_len = 0

    for page in pages:
        if current and current_len + len(page) > max_chars:
            batches.append("\n\n".join(current))
            current = [page]
            current_len = len(page)
        else:
            current.append(page)
            current_len += len(page)

    if current:
        batches.append("\n\n".join(current))

    return batches


# ---------------------------------------------------------------------------
# GPT-4o extraction
# ---------------------------------------------------------------------------

def _extract_product_chunks(text_batch: str) -> list[str]:
    """
    Call Azure OpenAI GPT-4o to extract structured product descriptions from
    a page-text batch.  Returns a list of chunk strings.
    """
    messages = [
        {"role": "system", "content": _EXTRACTION_SYSTEM},
        {"role": "user", "content": _EXTRACTION_USER_TMPL.format(text=text_batch)},
    ]

    raw = azure_manager.chat_completion(
        messages,
        temperature=0.0,
        max_tokens=4096,
        json_mode=False,  # We request a JSON array; json_mode forces an object
    )

    # Strip accidental markdown fences
    cleaned = raw.strip()
    for fence in ("```json", "```"):
        if cleaned.startswith(fence):
            cleaned = cleaned[len(fence):]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    try:
        chunks = json.loads(cleaned)
        if isinstance(chunks, list):
            return [str(c).strip() for c in chunks if str(c).strip()]
    except json.JSONDecodeError:
        log.warning("JSON parse failed – falling back to paragraph split.")
        return [p.strip() for p in cleaned.split("\n\n") if len(p.strip()) > 80]

    return []


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_pdf_to_chroma(
    pdf_file,
    progress_callback=None,
) -> tuple[bool, str]:
    """
    Ingest a lighting catalogue PDF into ChromaDB.

    Args:
        pdf_file:          File-like object with a ``.name`` attribute
                           (e.g. Streamlit UploadedFile or standard open() handle).
        progress_callback: Optional ``callable(fraction: float, message: str)``
                           for real-time UI progress updates.

    Returns:
        ``(True, success_message)`` or ``(False, error_message)``.
    """
    temp_path = ""
    source_name: str = getattr(pdf_file, "name", "unknown_catalogue.pdf")

    try:
        # ------------------------------------------------------------------
        # Step 1 – save to temp file
        # ------------------------------------------------------------------
        if progress_callback:
            progress_callback(0.05, "Saving catalogue to disk…")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_file.read())
            temp_path = tmp.name

        # ------------------------------------------------------------------
        # Step 2 – extract text pages with PyMuPDF
        # ------------------------------------------------------------------
        if progress_callback:
            progress_callback(0.12, "Extracting text from PDF pages…")

        pages = _extract_pages(temp_path)
        if not pages:
            return False, (
                "No readable text found in this PDF. "
                "It may be a scanned image-only catalogue — please use a text-layer PDF."
            )

        log.info("Extracted %d text pages from '%s'.", len(pages), source_name)

        # ------------------------------------------------------------------
        # Step 3 – GPT-4o semantic extraction per batch
        # ------------------------------------------------------------------
        batches = _batch_pages(pages)
        all_chunks: list[str] = []

        for i, batch in enumerate(batches):
            if progress_callback:
                pct = 0.15 + 0.45 * (i / max(len(batches), 1))
                progress_callback(
                    pct,
                    f"Azure OpenAI GPT-4o: extracting products from batch {i + 1}/{len(batches)}…",
                )
            chunks = _extract_product_chunks(batch)
            all_chunks.extend(chunks)
            log.info("Batch %d/%d → %d chunks extracted.", i + 1, len(batches), len(chunks))

        # Deduplicate while preserving order
        seen: set[str] = set()
        unique_chunks: list[str] = []
        for c in all_chunks:
            if c not in seen:
                seen.add(c)
                unique_chunks.append(c)
        all_chunks = unique_chunks

        if not all_chunks:
            return False, (
                "No product information could be extracted. "
                "Check that the PDF contains product descriptions and specifications."
            )

        log.info("Total unique product chunks: %d", len(all_chunks))

        # ------------------------------------------------------------------
        # Step 4 – embed with Azure OpenAI text-embedding-3-large
        # ------------------------------------------------------------------
        if progress_callback:
            progress_callback(
                0.62,
                f"Generating embeddings for {len(all_chunks)} product chunks…",
            )

        vectors = azure_manager.embed_documents(all_chunks)

        # ------------------------------------------------------------------
        # Step 5 – upsert into ChromaDB
        # ------------------------------------------------------------------
        if progress_callback:
            progress_callback(0.88, "Writing to vector database…")

        ids = [f"{source_name}::chunk::{i}" for i in range(len(all_chunks))]
        metadatas = [
            {
                "source": source_name,
                "chunk_index": i,
                "char_count": len(chunk),
            }
            for i, chunk in enumerate(all_chunks)
        ]

        chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
        collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

        # Remove stale entries from a previous upload of the same file
        try:
            existing = collection.get(where={"source": source_name})
            if existing.get("ids"):
                collection.delete(ids=existing["ids"])
                log.info(
                    "Replaced %d existing chunks for '%s'.",
                    len(existing["ids"]),
                    source_name,
                )
        except Exception:
            pass  # Collection may be empty — ignore

        collection.add(
            ids=ids,
            embeddings=vectors,
            metadatas=metadatas,
            documents=all_chunks,
        )

        if progress_callback:
            progress_callback(
                1.0,
                f"Done! {len(all_chunks)} product chunks indexed successfully.",
            )

        return True, (
            f"Success: {len(all_chunks)} semantic product chunks from "
            f"'{source_name}' added to the knowledge base."
        )

    except Exception as exc:
        log.error("PDF ingestion failed: %s", exc, exc_info=True)
        return False, f"Error processing catalogue: {exc}"

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
