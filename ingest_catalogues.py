"""
One-shot ingestion script for the two Starlight catalogues.

Run this ONCE after setting up your .env to populate ChromaDB
with all product chunks + page image blobs.

Usage:
    cd AI-CRM-Mailer
    python ingest_catalogues.py

The script looks for the PDFs in the repo root (../). You can also pass
explicit paths as arguments:
    python ingest_catalogues.py path/to/cat1.pdf path/to/cat2.pdf
"""
import sys
import os
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ingest")

# Ensure we run from the AI-CRM-Mailer directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from catalogue_ingestor import ingest_catalogue

# Default catalogue paths (relative to repo root)
DEFAULT_PDFS = [
    os.path.join("..", "STARLIGHT KITCHEN & FURNITURE 2024-2025 (3).pdf"),
    os.path.join("..", "Starlight_Linear_Catalogue.PDF"),
]


def progress_bar(pct: float, msg: str) -> None:
    filled = int(pct * 40)
    bar = "█" * filled + "░" * (40 - filled)
    print(f"\r  [{bar}] {pct:5.1%}  {msg:<55}", end="", flush=True)


def run(pdf_paths: list[str]) -> None:
    total_chunks = 0
    start = time.time()

    for i, path in enumerate(pdf_paths, 1):
        if not os.path.exists(path):
            log.warning("PDF not found — skipping: %s", path)
            continue

        filename = os.path.basename(path)
        print(f"\n[{i}/{len(pdf_paths)}] Ingesting: {filename}")
        print(f"  Size: {os.path.getsize(path) / 1_048_576:.1f} MB")

        t0 = time.time()
        summary = ingest_catalogue(path, progress_callback=progress_bar, force_reingest=True)
        elapsed = time.time() - t0
        total_chunks += summary["total_chunks"]

        print()  # newline after progress bar
        print(f"  Catalogue : {summary['catalogue_name']}")
        print(f"  Type      : {summary['catalogue_type']}")
        print(f"  Pages     : {summary['pages_processed']} processed, {summary['pages_skipped']} skipped")
        print(f"  Chunks    : {summary['total_chunks']} product chunks indexed")
        print(f"  Time      : {elapsed:.1f}s")

    elapsed_total = time.time() - start
    print(f"\n{'─'*55}")
    print(f"  Total chunks indexed : {total_chunks}")
    print(f"  Total time           : {elapsed_total:.1f}s")
    print(f"\n  ChromaDB collection  : starlight_vision")
    print(f"  Page images          : AI-CRM-Mailer/static/page_images/")
    print(f"\n  Run  start.sh / start.bat  to launch the app.\n")


if __name__ == "__main__":
    paths = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_PDFS
    run(paths)
