from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .io import read_textual_content


def approx_tokens(text: str) -> int:
    return max(1, len(text.split())) if text.strip() else 0


def stable_doc_id(relative_path: str, text: str) -> str:
    return hashlib.sha256(f"{relative_path}\n{text}".encode("utf-8", errors="ignore")).hexdigest()


def canonical_doc(*, relative_path: str, source_path: str, text: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    extra = extra or {}
    doc_id = stable_doc_id(relative_path, text)
    return {
        "schema_version": 1,
        "id": doc_id,
        "relative_path": relative_path,
        "source_path": source_path,
        "text": text,
        "chars": len(text),
        "tokens": approx_tokens(text),
        **extra,
    }


def doc_from_path(path: Path, root: Path, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    relative = str(path.relative_to(root)).replace("\\", "/")
    text = read_textual_content(path)
    return canonical_doc(relative_path=relative, source_path=str(path), text=text, extra=extra)
