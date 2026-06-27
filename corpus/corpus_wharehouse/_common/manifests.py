from __future__ import annotations

import json
from pathlib import Path

from .io import write_json


def manifest_path(root: Path, stage: str) -> Path:
    return root / "pipeline_state" / f"{stage}.manifest.json"


def load_manifest(root: Path, stage: str) -> dict:
    path = manifest_path(root, stage)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_manifest(root: Path, stage: str, payload: dict) -> Path:
    path = manifest_path(root, stage)
    write_json(path, payload)
    return path
