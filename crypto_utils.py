"""Encrypt/decrypt integration credentials at rest."""
from __future__ import annotations

import base64
import hashlib
import os

from cryptography.fernet import Fernet


def _fernet() -> Fernet:
    key = os.getenv("INTEGRATION_ENCRYPTION_KEY", "").strip()
    if not key:
        # Derive from JWT_SECRET for dev only
        secret = os.getenv("JWT_SECRET", "dev-insecure-key-change-me")
        derived = base64.urlsafe_b64encode(hashlib.sha256(secret.encode()).digest())
        return Fernet(derived)
    if len(key) == 44 and key.endswith("="):
        return Fernet(key.encode())
    derived = base64.urlsafe_b64encode(hashlib.sha256(key.encode()).digest())
    return Fernet(derived)


def encrypt_text(plain: str) -> str:
    return _fernet().encrypt(plain.encode()).decode()


def decrypt_text(cipher: str) -> str:
    return _fernet().decrypt(cipher.encode()).decode()
