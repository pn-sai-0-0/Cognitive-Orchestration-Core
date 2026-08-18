from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

DEFAULT_CONFIG: dict[str, Any] = {
    "runtime": {
        "workers": max(1, (os.cpu_count() or 4) - 1),
        "chunk_size": 500,
        "memory_limit_mb": 2048,
        "resume": True,
        "incremental": True,
        "seed": 42,
    },
    "validation": {
        "allowed_extensions": [".txt", ".md", ".json", ".jsonl", ".csv", ".xml", ".parquet"],
        "materialize_mode": "link",
    },
    "deduplication": {
        "simhash_hamming_threshold": 3,
        "minhash_jaccard_threshold": 0.9,
        "signature_bands": 8,
    },
    "quality": {
        "min_chars": 32,
        "min_quality_score": 0.35,
        "max_nonprintable_ratio": 0.05,
        "max_ocr_noise_ratio": 0.25,
    },
    "language": {
        "default": "en",
        "confidence_threshold": 0.65,
        "mixed_threshold": 0.20,
    },
    "balancing": {
        "enabled": True,
        "max_category_ratio": 1.5,
        "min_docs_per_bucket": 1,
    },
    "splits": {
        "train": 0.96,
        "validation": 0.02,
        "test": 0.02,
    },
    "tokenizer": {
        "target_chunk_tokens": 200000,
        "max_docs_per_chunk": 2000,
    },
    "logging": {
        "jsonl": True,
        "console": True,
    },
}


def _merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    cfg = deepcopy(DEFAULT_CONFIG)
    cfg_path = root / "pipeline_config.yaml"
    if cfg_path.exists() and yaml is not None:
        user_cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        cfg = _merge(cfg, user_cfg)
    return cfg
