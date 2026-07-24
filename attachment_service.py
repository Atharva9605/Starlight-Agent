"""Attachment ingest, RAG indexing, and AI document generation."""
from __future__ import annotations

import logging
import mimetypes
import tempfile
from pathlib import Path
from typing import Optional

from azure_client import azure_manager
from vector_store import add_chunks, delete_by_source
import attachment_store as att_store

log = logging.getLogger("attachment_service")

MAX_ATTACHMENT_BYTES = 15 * 1024 * 1024
TEXT_MIME_PREFIXES = ("text/",)
INGESTIBLE_MIME = {
    "application/pdf",
    "text/plain",
    "text/csv",
    "text/markdown",
    "text/html",
    "application/json",
}
IMAGE_MIME_PREFIXES = ("image/",)


def guess_mime(filename: str, fallback: str = "application/octet-stream") -> str:
    mime, _ = mimetypes.guess_type(filename)
    return mime or fallback


def extract_text_from_bytes(filename: str, mime_type: str, data: bytes) -> str:
    """Best-effort text extraction for RAG."""
    mime = (mime_type or guess_mime(filename)).lower()
    name = filename.lower()

    if mime.startswith("text/") or name.endswith((".txt", ".csv", ".md", ".json", ".html", ".htm")):
        try:
            return data.decode("utf-8", errors="replace")[:80000]
        except Exception:
            return data.decode("latin-1", errors="replace")[:80000]

    if mime == "application/pdf" or name.endswith(".pdf"):
        try:
            import fitz

            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(data)
                tmp_path = tmp.name
            try:
                doc = fitz.open(tmp_path)
                parts = []
                for i, page in enumerate(doc):
                    if i >= 40:
                        break
                    parts.append(page.get_text("text"))
                doc.close()
                return "\n".join(parts).strip()[:80000]
            finally:
                Path(tmp_path).unlink(missing_ok=True)
        except Exception as exc:
            log.warning("PDF text extract failed for %s: %s", filename, exc)
            return ""

    if any(mime.startswith(p) for p in IMAGE_MIME_PREFIXES):
        try:
            caption = azure_manager.vision_completion(
                data,
                (
                    "Describe this email attachment image for a B2B sales CRM. "
                    "Extract any readable text, product names, quantities, PO numbers, "
                    "and key facts. Be concise (≤200 words)."
                ),
            )
            return (caption or f"[Image attachment: {filename}]").strip()[:8000]
        except Exception as exc:
            log.warning("Image caption failed for %s: %s", filename, exc)
            return f"[Image attachment: {filename}]"

    return f"[Binary attachment: {filename} ({mime}, {len(data)} bytes)]"


def ingest_attachment_to_rag(attachment: dict) -> bool:
    """Chunk extracted text into the org vector store, tagged by conversation."""
    att_id = attachment["id"]
    source = f"email_attachment:{att_id}"
    text = (attachment.get("extracted_text") or "").strip()
    if not text or text.startswith("[Binary attachment"):
        # Re-extract if we have bytes
        try:
            data = att_store.read_bytes(attachment["storage_path"])
            text = extract_text_from_bytes(
                attachment["filename"], attachment["mime_type"], data
            )
        except Exception as exc:
            log.warning("Cannot read attachment %s for RAG: %s", att_id, exc)
            return False

    if not text or len(text.strip()) < 20:
        att_store.mark_rag_ingested(att_id, text or "")
        return False

    delete_by_source(source)

    # Split into ~3k char chunks
    chunks: list[str] = []
    buf = text
    while buf:
        chunks.append(buf[:3000].strip())
        buf = buf[2800:]  # slight overlap
    chunks = [c for c in chunks if c][:30]

    embeddings = []
    for c in chunks:
        emb = azure_manager.embed_text(c)
        embeddings.append(emb)

    ids = [f"{source}::chunk::{i:03d}" for i in range(len(chunks))]
    metas = [
        {
            "source": source,
            "catalogue_name": f"Email: {attachment['filename']}",
            "kind": "email_attachment",
            "attachment_id": att_id,
            "conversation_id": attachment["conversation_id"],
            "filename": attachment["filename"],
        }
        for _ in chunks
    ]
    add_chunks(ids, embeddings, chunks, metas)
    att_store.mark_rag_ingested(att_id, text)
    log.info("RAG ingested attachment %s (%d chunks)", att_id, len(chunks))
    return True


def save_and_ingest(
    *,
    conversation_id: str,
    filename: str,
    data: bytes,
    mime_type: str = "",
    source: str = "inbound",
    message_id: str | None = None,
    gmail_attachment_id: str | None = None,
    include_on_send: bool = False,
) -> dict:
    if len(data) > MAX_ATTACHMENT_BYTES:
        raise ValueError(f"Attachment exceeds {MAX_ATTACHMENT_BYTES // (1024*1024)}MB limit")
    if not data:
        raise ValueError("Empty attachment")

    mime = mime_type or guess_mime(filename)
    extracted = extract_text_from_bytes(filename, mime, data)
    att = att_store.create_attachment(
        conversation_id=conversation_id,
        filename=filename,
        mime_type=mime,
        data=data,
        source=source,
        message_id=message_id,
        gmail_attachment_id=gmail_attachment_id,
        extracted_text=extracted,
        include_on_send=include_on_send,
    )
    try:
        ingest_attachment_to_rag(att)
        refreshed = att_store.get_attachment(att["id"])
        return refreshed or att
    except Exception as exc:
        log.exception("RAG ingest failed for %s: %s", att["id"], exc)
        return att


def attachment_context_for_conversation(conversation_id: str) -> str:
    """Text block for the draft agent prompt."""
    atts = att_store.list_attachments(conversation_id)
    if not atts:
        return "(No email attachments)"

    blocks = []
    for a in atts:
        preview = (a.get("extracted_text") or "").strip()
        if len(preview) > 2500:
            preview = preview[:2500] + "…"
        blocks.append(
            f"- {a['filename']} ({a['source']}, {a['mime_type']}, "
            f"{a['size_bytes']} bytes, rag={'yes' if a.get('rag_ingested') else 'no'}):\n"
            f"{preview or '(no extractable text)'}"
        )
    return "\n\n".join(blocks)


def generate_document(
    conversation_id: str,
    *,
    instructions: str,
    title: str = "",
    fmt: str = "pdf",
) -> dict:
    """
    AI-create a document (PDF or TXT) and attach it for outbound send.
    """
    import conversation_store as store

    conv = store.get_conversation(conversation_id)
    if not conv:
        raise ValueError("Conversation not found")

    client = conv.get("client", {})
    history = []
    for m in conv.get("messages", [])[-8:]:
        if m.get("status") == "draft":
            continue
        role = "Client" if m["direction"] == "inbound" else "Us"
        body = (m.get("body_text") or "")[:800]
        history.append(f"{role}: {body}")

    att_ctx = attachment_context_for_conversation(conversation_id)
    doc_title = title.strip() or "Starlight follow-up document"

    raw = azure_manager.chat_completion(
        [
            {
                "role": "system",
                "content": (
                    "You write professional B2B documents for Starlight Linear LED. "
                    "Return plain document body text only — no markdown fences, no JSON. "
                    "Use clear headings and short paragraphs suitable for a PDF attachment."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Create a document titled: {doc_title}\n\n"
                    f"Instructions:\n{instructions.strip()}\n\n"
                    f"Client: {client.get('company') or ''} <{client.get('email')}>\n"
                    f"Thread subject: {conv.get('subject', '')}\n\n"
                    f"Recent messages:\n" + "\n".join(history) + "\n\n"
                    f"Existing attachments context:\n{att_ctx}\n"
                ),
            },
        ],
        temperature=0.35,
        max_tokens=3500,
    )
    body_text = (raw or "").strip()
    if not body_text:
        raise RuntimeError("Document generation returned empty content")

    fmt = (fmt or "pdf").lower()
    if fmt == "txt":
        filename = f"{_slug(doc_title)}.txt"
        data = body_text.encode("utf-8")
        mime = "text/plain"
    else:
        filename = f"{_slug(doc_title)}.pdf"
        data = _text_to_pdf_bytes(doc_title, body_text)
        mime = "application/pdf"

    return save_and_ingest(
        conversation_id=conversation_id,
        filename=filename,
        data=data,
        mime_type=mime,
        source="generated",
        include_on_send=True,
    )


def _slug(s: str) -> str:
    import re

    s = re.sub(r"[^\w\-]+", "-", (s or "document").strip()).strip("-").lower()
    return (s[:60] or "document")


def _text_to_pdf_bytes(title: str, body: str) -> bytes:
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    margin = 50
    y = margin
    page.insert_text((margin, y), title[:120], fontsize=14, fontname="helv")
    y += 28
    # Simple word wrap
    max_width = page.rect.width - 2 * margin
    for para in body.split("\n"):
        para = para.strip()
        if not para:
            y += 10
            continue
        words = para.split()
        line = ""
        for w in words:
            trial = f"{line} {w}".strip()
            if fitz.get_text_length(trial, fontsize=10, fontname="helv") > max_width:
                if y > page.rect.height - margin:
                    page = doc.new_page()
                    y = margin
                page.insert_text((margin, y), line, fontsize=10, fontname="helv")
                y += 14
                line = w
            else:
                line = trial
        if line:
            if y > page.rect.height - margin:
                page = doc.new_page()
                y = margin
            page.insert_text((margin, y), line, fontsize=10, fontname="helv")
            y += 16
    data = doc.tobytes()
    doc.close()
    return data
