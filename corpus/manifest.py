"""
CognitiveOC v3 — Corpus Manifest System
=========================================

Generates, validates, and diffs manifests at two levels:

  Source manifest:  Per-source record produced after pipeline completion.
                    Stored in warehouse/manifests/<source_id>.json

  Release manifest: Release-level record produced by release_builder.
                    Stored in releases/v<N>/manifest.json

Release manifest schema:
  See §5.3 of the master architecture document.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

try:
    from config import CORPUS_WAREHOUSE_DIR
    _WAREHOUSE = Path(CORPUS_WAREHOUSE_DIR)
except (ImportError, AttributeError):
    _WAREHOUSE = Path("var/corpus_warehouse")

_MANIFESTS_DIR = _WAREHOUSE / "manifests"
_RELEASES_DIR  = _WAREHOUSE / "releases"


# ── Utilities ─────────────────────────────────────────────────────────

def _sha256_file(path: str) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def _estimate_tokens(path: str, chars_per_token: float = 3.8) -> int:
    """Estimate token count from file character count."""
    size = Path(path).stat().st_size
    return int(size / chars_per_token)


# ── Source-level manifest ─────────────────────────────────────────────

def generate_source_manifest(
    source_id:      str,
    category:       str,
    licence:        str,
    licence_risk:   float,
    raw_path:       str,
    cleaned_path:   str,
    deduped_path:   str,
    approved_path:  str,
    n_raw:          int,
    n_after_clean:  int,
    n_after_dedup:  int,
    n_approved:     int,
    score_summary:  dict,
    acquired_by:    str,
    approved_by:    str,
) -> dict:
    """
    Generate and write a source-level manifest.

    Returns the manifest dict.
    """
    _MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "manifest_version":   "1.0",
        "coc_version":        "v3",
        "source_id":          source_id,
        "category":           category,
        "licence":            licence,
        "licence_risk":       licence_risk,
        "created":            time.strftime("%Y-%m-%dT%H:%M:%S"),
        "acquired_by":        acquired_by,
        "approved_by":        approved_by,
        "paths": {
            "raw":     raw_path,
            "cleaned": cleaned_path,
            "deduped": deduped_path,
            "approved":approved_path,
        },
        "pipeline_stats": {
            "n_raw":          n_raw,
            "n_after_clean":  n_after_clean,
            "n_after_dedup":  n_after_dedup,
            "n_approved":     n_approved,
            "retention_pct":  round(n_approved / n_raw * 100, 1) if n_raw else 0,
        },
        "score_summary":      score_summary,
        "tokens_estimate":    _estimate_tokens(approved_path) if Path(approved_path).exists() else 0,
        "sha256_approved":    _sha256_file(approved_path) if Path(approved_path).exists() else "",
    }

    out = _MANIFESTS_DIR / f"{source_id}.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)

    return manifest


def load_source_manifest(source_id: str) -> dict | None:
    """Load and return a source manifest, or None if not found."""
    path = _MANIFESTS_DIR / f"{source_id}.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# ── Release manifest ──────────────────────────────────────────────────

def generate_release_manifest(
    version:         str,
    sources:         list[dict],    # list of source manifests or registry records
    train_path:      str,
    val_path:        str,
    test_path:       str,
    split_ratios:    tuple[float, float, float],
    shuffle_seed:    int,
    released_by:     str,
    notes:           str = "",
    quality_gate:    dict | None = None,
) -> dict:
    """
    Generate a release-level manifest and write it to releases/v<N>/.

    Args:
        version:      Release version string e.g. "v1".
        sources:      List of source info dicts from registry or source manifests.
        train_path:   Path to assembled train.txt.
        val_path:     Path to assembled val.txt.
        test_path:    Path to assembled test.txt.
        split_ratios: (train, val, test) as floats.
        shuffle_seed: Seed used for shuffle.
        released_by:  Operator username.
        notes:        Free-text notes.
        quality_gate: Dict of quality gate checks that passed.

    Returns:
        The full manifest dict.
    """
    release_dir = _RELEASES_DIR / version
    release_dir.mkdir(parents=True, exist_ok=True)

    train_tokens = _estimate_tokens(train_path) if Path(train_path).exists() else 0
    val_tokens   = _estimate_tokens(val_path)   if Path(val_path).exists()   else 0
    test_tokens  = _estimate_tokens(test_path)  if Path(test_path).exists()  else 0

    sha_train = _sha256_file(train_path) if Path(train_path).exists() else ""
    sha_val   = _sha256_file(val_path)   if Path(val_path).exists()   else ""
    sha_test  = _sha256_file(test_path)  if Path(test_path).exists()  else ""

    # Source entries for manifest
    source_entries = []
    for s in sources:
        sid  = s.get("source_id", "unknown")
        source_entries.append({
            "source_id":          sid,
            "category":           s.get("category", "?"),
            "token_contribution": s.get("tokens_estimate", s.get("approved_tokens_est", 0)),
            "licence":            s.get("licence", "unknown"),
            "licence_risk":       s.get("licence_risk", 0.8),
            "approved_by":        s.get("approved_by", "unknown"),
            "approval_date":      s.get("approval_date", "unknown"),
        })

    manifest: dict[str, Any] = {
        "manifest_version":  "1.0",
        "coc_version":       "v3",
        "release_id":        version,
        "release_date":      time.strftime("%Y-%m-%d"),
        "released_by":       released_by,
        "tokenizer":         "48K-SentencePiece-Unigram",
        "total_tokens_estimate": train_tokens + val_tokens + test_tokens,
        "train_tokens":      train_tokens,
        "val_tokens":        val_tokens,
        "test_tokens":       test_tokens,
        "split_ratios":      list(split_ratios),
        "shuffle_seed":      shuffle_seed,
        "categories_included": sorted({s.get("category","?") for s in sources}),
        "sources":           source_entries,
        "quality_gate":      quality_gate or {},
        "checksums": {
            "train": f"sha256:{sha_train}",
            "val":   f"sha256:{sha_val}",
            "test":  f"sha256:{sha_test}",
        },
        "paths": {
            "train": str(train_path),
            "val":   str(val_path),
            "test":  str(test_path),
        },
        "notes": notes,
        "status": "draft",   # changed to "signed" by release_builder after approval
    }

    out_manifest = release_dir / "manifest.json"
    with open(out_manifest, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)

    # Write checksums file
    out_checksums = release_dir / "checksums.sha256"
    with open(out_checksums, "w", encoding="utf-8") as fh:
        fh.write(f"{sha_train}  train.txt\n")
        fh.write(f"{sha_val}    val.txt\n")
        fh.write(f"{sha_test}   test.txt\n")

    return manifest


def validate_manifest(manifest_path: str) -> dict:
    """
    Validate a release manifest.

    Checks:
      - Required fields present
      - Checksums match current files on disk
      - Split ratios sum to ~1.0

    Returns:
      {"valid": bool, "errors": list[str], "warnings": list[str]}
    """
    path = Path(manifest_path)
    if not path.exists():
        return {"valid": False, "errors": [f"Manifest not found: {manifest_path}"], "warnings": []}

    with open(path, encoding="utf-8") as fh:
        m = json.load(fh)

    errors:   list[str] = []
    warnings: list[str] = []

    # Required fields
    required = ["release_id", "release_date", "released_by", "checksums",
                "sources", "split_ratios", "shuffle_seed"]
    for field in required:
        if field not in m:
            errors.append(f"Missing required field: {field}")

    # Split ratios
    ratios = m.get("split_ratios", [])
    if ratios and abs(sum(ratios) - 1.0) > 0.01:
        errors.append(f"split_ratios sum to {sum(ratios):.3f}, not 1.0")

    # Checksum verification
    for split_name, file_key in [("train", "train.txt"),
                                  ("val",   "val.txt"),
                                  ("test",  "test.txt")]:
        expected_raw = m.get("checksums", {}).get(split_name, "")
        expected     = expected_raw.replace("sha256:", "")
        file_path    = path.parent / file_key
        if not file_path.exists():
            warnings.append(f"Split file not found for checksum verification: {file_path}")
            continue
        actual = _sha256_file(str(file_path))
        if expected and actual != expected:
            errors.append(f"Checksum mismatch for {file_key}: "
                          f"expected {expected[:12]}... got {actual[:12]}...")

    return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}


def diff_manifests(path1: str, path2: str) -> dict:
    """
    Diff two release manifests and return a summary of changes.

    Returns:
      {
        "added_sources":   [source_ids in v2 not in v1],
        "removed_sources": [source_ids in v1 not in v2],
        "token_delta":     v2_total_tokens - v1_total_tokens,
        "category_changes": {...}
      }
    """
    def _load(p: str) -> dict:
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)

    m1 = _load(path1)
    m2 = _load(path2)

    ids1 = {s["source_id"] for s in m1.get("sources", [])}
    ids2 = {s["source_id"] for s in m2.get("sources", [])}

    cats1 = m1.get("categories_included", [])
    cats2 = m2.get("categories_included", [])

    return {
        "v1":              m1.get("release_id"),
        "v2":              m2.get("release_id"),
        "added_sources":   sorted(ids2 - ids1),
        "removed_sources": sorted(ids1 - ids2),
        "token_delta":     (m2.get("total_tokens_estimate", 0)
                           - m1.get("total_tokens_estimate", 0)),
        "category_changes": {
            "added":   sorted(set(cats2) - set(cats1)),
            "removed": sorted(set(cats1) - set(cats2)),
        },
    }


def load_release_manifest(version: str) -> dict | None:
    """Load a release manifest by version string (e.g. 'v1')."""
    path = _RELEASES_DIR / version / "manifest.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
