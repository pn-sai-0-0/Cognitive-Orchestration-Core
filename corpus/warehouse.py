"""
CognitiveOC v3 — Corpus Warehouse
====================================

The Warehouse is the long-term storage layer. It manages:
  - Approved raw + cleaned source archives
  - Cross-source organisation by category
  - Token and paragraph counts
  - Version snapshots of warehouse state

The warehouse lives on the 1TB additional SSD (path from CORPUS_WAREHOUSE_DIR
in config.py). The project repository only stores the registry and manifests.

Warehouse directory layout:
  <CORPUS_WAREHOUSE_DIR>/
    raw/               ← immutable source archives, never modified
      books/
      educational/
      reasoning/
      conversations/
      technical_docs/
      articles/
      research_papers/
      synthetic/
      cognition/
      retrieval/
      kg/
    cleaned/           ← post-pipeline UTF-8 text
    deduplicated/      ← cross-source dedup output
    scored/            ← scored paragraph metadata
    review_queue/      ← pending / approved / rejected JSONL
    approved/          ← final approved paragraphs, split by category
    rejected/          ← failed content, kept for audit
    synthetic/         ← versioned synthetic data from COC generator
      v1/
      v2/
    manifests/         ← source-level manifests
    releases/          ← assembled release artifacts
      v1/
        train.txt
        val.txt
        test.txt
        manifest.json
        checksums.sha256
    governance_logs/   ← daily audit log shards
    archive/           ← retired versions and recalled releases
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Any

try:
    from config import CORPUS_WAREHOUSE_DIR, CORPUS
    _WAREHOUSE     = Path(CORPUS_WAREHOUSE_DIR)
    _TARGET_V1     = CORPUS.get("target_v1_tokens", 30_000_000_000)
    _TARGET_WHOUSE = CORPUS.get("target_warehouse_tokens", 65_000_000_000)
except (ImportError, KeyError):
    _WAREHOUSE     = Path("var/corpus_warehouse")
    _TARGET_V1     = 30_000_000_000
    _TARGET_WHOUSE = 65_000_000_000

_CATEGORY_DIRS = {
    "A": "books",
    "B": "educational",
    "C": "reasoning",
    "D": "conversations",
    "E": "technical_docs",
    "F": "articles",
    "G": "research_papers",
    "H": "synthetic",
    "I": "cognition",
    "J": "retrieval",
    "K": "kg",
}


# ── Directory management ──────────────────────────────────────────────

def ensure_warehouse_dirs() -> None:
    """Create all required warehouse subdirectories."""
    subdirs = [
        "raw", "cleaned", "deduplicated", "scored",
        "review_queue", "approved", "rejected",
        "synthetic", "manifests", "releases",
        "governance_logs", "archive",
    ]
    for sub in subdirs:
        (_WAREHOUSE / sub).mkdir(parents=True, exist_ok=True)

    # Per-category raw dirs
    for cat, dirname in _CATEGORY_DIRS.items():
        (_WAREHOUSE / "raw" / dirname).mkdir(parents=True, exist_ok=True)
        (_WAREHOUSE / "cleaned" / dirname).mkdir(parents=True, exist_ok=True)
        (_WAREHOUSE / "deduplicated" / dirname).mkdir(parents=True, exist_ok=True)
        (_WAREHOUSE / "approved" / dirname).mkdir(parents=True, exist_ok=True)


def raw_dir(category: str) -> Path:
    """Return the raw source directory for a category."""
    return _WAREHOUSE / "raw" / _CATEGORY_DIRS.get(category, category)


def cleaned_dir(category: str) -> Path:
    return _WAREHOUSE / "cleaned" / _CATEGORY_DIRS.get(category, category)


def deduped_dir(category: str) -> Path:
    return _WAREHOUSE / "deduplicated" / _CATEGORY_DIRS.get(category, category)


def approved_dir(category: str) -> Path:
    return _WAREHOUSE / "approved" / _CATEGORY_DIRS.get(category, category)


def synthetic_dir(version: str = "v1") -> Path:
    d = _WAREHOUSE / "synthetic" / version
    d.mkdir(parents=True, exist_ok=True)
    return d


def release_dir(version: str) -> Path:
    d = _WAREHOUSE / "releases" / version
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── Storage stats ─────────────────────────────────────────────────────

def _dir_size_gb(path: Path) -> float:
    """Return directory size in GB."""
    if not path.exists():
        return 0.0
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return round(total / (1024 ** 3), 3)


def _count_files(path: Path, suffix: str = ".txt") -> int:
    if not path.exists():
        return 0
    return sum(1 for f in path.rglob(f"*{suffix}"))


def _estimate_tokens_from_dir(path: Path, chars_per_token: float = 3.8) -> int:
    """Estimate token count for all text files under a directory."""
    if not path.exists():
        return 0
    total_chars = sum(
        f.stat().st_size
        for f in path.rglob("*.txt")
        if f.is_file()
    )
    return int(total_chars / chars_per_token)


def warehouse_stats() -> dict:
    """
    Return aggregate statistics about the warehouse.

    Includes storage sizes, file counts, token estimates per category,
    and progress toward v1 release and long-term warehouse targets.
    """
    ensure_warehouse_dirs()

    # Per-category approved stats
    cat_stats: dict[str, dict] = {}
    total_approved_tokens = 0

    for cat, dirname in _CATEGORY_DIRS.items():
        adir   = _WAREHOUSE / "approved" / dirname
        tokens = _estimate_tokens_from_dir(adir)
        size   = _dir_size_gb(adir)
        cat_stats[cat] = {
            "category":          cat,
            "dirname":           dirname,
            "approved_tokens":   tokens,
            "approved_size_gb":  size,
            "files":             _count_files(adir),
        }
        total_approved_tokens += tokens

    # Overall storage
    raw_gb     = _dir_size_gb(_WAREHOUSE / "raw")
    cleaned_gb = _dir_size_gb(_WAREHOUSE / "cleaned")
    deduped_gb = _dir_size_gb(_WAREHOUSE / "deduplicated")
    approved_gb= _dir_size_gb(_WAREHOUSE / "approved")

    # Check for completed releases
    releases_dir = _WAREHOUSE / "releases"
    releases     = sorted([d.name for d in releases_dir.iterdir() if d.is_dir()]) \
                   if releases_dir.exists() else []

    return {
        "warehouse_path":        str(_WAREHOUSE),
        "total_approved_tokens": total_approved_tokens,
        "target_v1_tokens":      _TARGET_V1,
        "target_warehouse_tokens":_TARGET_WHOUSE,
        "v1_progress_pct":       round(total_approved_tokens / _TARGET_V1 * 100, 1)
                                  if _TARGET_V1 else 0,
        "warehouse_progress_pct":round(total_approved_tokens / _TARGET_WHOUSE * 100, 1)
                                  if _TARGET_WHOUSE else 0,
        "storage_gb": {
            "raw":          raw_gb,
            "cleaned":      cleaned_gb,
            "deduplicated": deduped_gb,
            "approved":     approved_gb,
            "total_warehouse": _dir_size_gb(_WAREHOUSE),
        },
        "categories":            cat_stats,
        "completed_releases":    releases,
    }


def print_warehouse_stats() -> None:
    """Pretty-print warehouse statistics to stdout."""
    stats = warehouse_stats()
    print(f"\n{'='*68}")
    print(f"CognitiveOC v3 — Corpus Warehouse Status")
    print(f"Path: {stats['warehouse_path']}")
    print(f"{'='*68}")

    total_t = stats["total_approved_tokens"]
    v1_pct  = stats["v1_progress_pct"]
    wh_pct  = stats["warehouse_progress_pct"]

    print(f"\nApproved tokens: {total_t:>20,}")
    print(f"v1 target:       {stats['target_v1_tokens']:>20,}  ({v1_pct:.1f}% complete)")
    print(f"Warehouse target:{stats['target_warehouse_tokens']:>20,}  ({wh_pct:.1f}% complete)")

    print(f"\nStorage usage:")
    sg = stats["storage_gb"]
    for k, v in sg.items():
        print(f"  {k:<18} {v:>8.1f} GB")

    print(f"\nApproved tokens by category:")
    print(f"  {'Cat':<4} {'Name':<20} {'Tokens':>15}  {'GB':>8}")
    print(f"  {'-'*52}")
    for cat in sorted(stats["categories"]):
        cs = stats["categories"][cat]
        print(f"  {cs['category']:<4} {cs['dirname']:<20} "
              f"{cs['approved_tokens']:>15,}  {cs['approved_size_gb']:>6.1f}")

    if stats["completed_releases"]:
        print(f"\nCompleted releases: {', '.join(stats['completed_releases'])}")
    else:
        print(f"\nNo releases built yet.")

    print(f"{'='*68}\n")


# ── Archive management ────────────────────────────────────────────────

def archive_recalled_release(version: str, reason: str,
                              operator: str = "system") -> None:
    """
    Move a recalled release to the archive directory with an incident log.
    """
    src = _WAREHOUSE / "releases" / version
    if not src.exists():
        raise FileNotFoundError(f"Release directory not found: {src}")

    archive_path = _WAREHOUSE / "archive" / "recalled" / version
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(archive_path))

    # Write incident log
    incident = {
        "ts":          time.strftime("%Y-%m-%dT%H:%M:%S"),
        "version":     version,
        "reason":      reason,
        "operator":    operator,
        "archived_to": str(archive_path),
    }
    incident_path = _WAREHOUSE / "archive" / "recalled" / f"{version}_incident.json"
    import json
    with open(incident_path, "w", encoding="utf-8") as fh:
        json.dump(incident, fh, indent=2)

    from audit.logger import log_event
    log_event("archive", version, "release_recalled", "ok", operator=operator,
              details={"reason": reason, "archived_to": str(archive_path)})

    print(f"[warehouse] Release {version} archived to {archive_path}")
    print(f"[warehouse] Incident log: {incident_path}")
