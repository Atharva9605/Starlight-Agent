"""
Embedding dimension resolution for vector store schema.

Priority:
  1. EMBEDDING_DIMENSION env var (explicit override)
  2. Known Azure deployment name → dimension map
  3. Live probe via Azure embeddings API (cached)
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache

log = logging.getLogger("embedding_config")

# Default dimensions when deployment name contains these substrings (case-insensitive).
# Azure returns full vectors unless dimensions= is passed in the API call (we don't pass it).
KNOWN_DEPLOYMENT_DIMENSIONS: dict[str, int] = {
    "text-embedding-ada-002": 1536,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "embed-v-4-0": 1536,  # common Azure alias patterns — override via env if different
}


@lru_cache(maxsize=1)
def get_embedding_dimension() -> int:
    """Return the embedding vector size used by the active Azure deployment."""
    explicit = os.getenv("EMBEDDING_DIMENSION", "").strip()
    if explicit:
        dim = int(explicit)
        log.info("Using EMBEDDING_DIMENSION=%s from environment", dim)
        return dim

    deployment = (
        os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002").strip().lower()
    )
    for key, dim in KNOWN_DEPLOYMENT_DIMENSIONS.items():
        if key in deployment:
            log.info(
                "Inferred embedding dimension %s from deployment name '%s'",
                dim,
                deployment,
            )
            return dim

    # Probe Azure (requires credentials)
    try:
        from azure_client import azure_manager

        vec = azure_manager.embed_text("dimension probe")
        dim = len(vec)
        log.info(
            "Probed embedding dimension %s from deployment '%s'",
            dim,
            deployment,
        )
        return dim
    except Exception as exc:
        log.warning(
            "Could not probe embedding dimension (%s); defaulting to 1536 (ada-002). "
            "Set EMBEDDING_DIMENSION explicitly if using a different model.",
            exc,
        )
        return 1536


def validate_embedding(vector: list[float], *, context: str = "") -> None:
    """Raise ValueError if vector length does not match configured dimension."""
    expected = get_embedding_dimension()
    actual = len(vector)
    if actual != expected:
        raise ValueError(
            f"Embedding dimension mismatch{' (' + context + ')' if context else ''}: "
            f"got {actual}, expected {expected}. "
            f"Set EMBEDDING_DIMENSION={actual} or use a deployment that outputs {expected}-dim vectors."
        )
