"""
CognitiveOC v3 — API Authentication
=====================================
32-byte random key stored in var/auth_key.txt.
Generated once on first startup; persists across restarts.
Every API endpoint except /api/auth-key requires X-CognitiveOC-Key header.
"""
from __future__ import annotations
import os
import secrets
from pathlib import Path

try:
    from config import STORE_DIR
except ImportError:
    STORE_DIR = Path("var")

_KEY_PATH = Path(str(STORE_DIR)) / "auth_key.txt"
_secret: str | None = None


def get_secret() -> str:
    """Return the API key, generating it on first call."""
    global _secret
    if _secret:
        return _secret
    _KEY_PATH.parent.mkdir(parents=True, exist_ok=True)
    if _KEY_PATH.exists():
        _secret = _KEY_PATH.read_text().strip()
    else:
        _secret = secrets.token_hex(32)
        _KEY_PATH.write_text(_secret)
    return _secret


def check(handler) -> bool:
    """Validate X-CognitiveOC-Key header on an HTTP request handler.
    Returns True if authenticated, False otherwise.
    """
    key = handler.headers.get("X-CognitiveOC-Key", "")
    return secrets.compare_digest(key, get_secret())
