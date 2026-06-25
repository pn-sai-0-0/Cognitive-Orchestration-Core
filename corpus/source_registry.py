"""
CognitiveOC v3 — Source Registry
==================================

Loads, validates, saves, and queries the governance/source_registry.json file.
This is the authoritative record of every source family in the corpus warehouse.

Source record schema:
  {
    "source_id":          str   — unique key e.g. "A-gutenberg-20260701"
    "name":               str   — human-readable name
    "url":                str   — canonical source URL
    "category":           str   — one of A-K
    "licence":            str   — licence display name
    "licence_id":         str   — key from license_rules.json
    "licence_risk":       float — from license_rules.json
    "acquisition_date":   str   — YYYY-MM-DD
    "acquired_by":        str   — operator username
    "raw_path":           str   — path under warehouse/raw/
    "raw_sha256":         str   — SHA-256 of raw archive
    "source_validated":   bool
    "validation_date":    str | null
    "validated_by":       str | null
    "status":             str   — pending|validated|cleaning|scored|approved|rejected|archived
    "approved_paragraphs":int   — set after pipeline completes
    "approved_tokens_est":int   — set after pipeline completes
    "approval_date":      str | null
    "approved_by":        str | null
    "in_release":         list[str]  — release version IDs this source appears in
    "notes":              str
  }
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

try:
    from config import CORPUS_REGISTRY_PATH
    _REGISTRY_PATH = Path(CORPUS_REGISTRY_PATH)
except ImportError:
    _REGISTRY_PATH = Path("governance/source_registry.json")

VALID_CATEGORIES = {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"}
VALID_STATUSES   = {
    "pending", "validated", "normalizing", "cleaning",
    "deduplicating", "scoring", "review", "approved", "rejected", "archived",
}


# ── Registry load/save ───────────────────────────────────────────────

def load_registry() -> dict:
    """Load and return the full source registry dict."""
    if not _REGISTRY_PATH.exists():
        return {
            "registry_version": "1.0",
            "coc_version": "v3",
            "last_updated": None,
            "registry_hash": None,
            "sources": [],
        }
    with open(_REGISTRY_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def save_registry(registry: dict) -> None:
    """Write registry dict back to disk, updating metadata."""
    registry["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    # Compute registry hash (hash of sources list only)
    payload = json.dumps(registry["sources"], sort_keys=True, ensure_ascii=False)
    registry["registry_hash"] = hashlib.sha256(payload.encode()).hexdigest()
    _REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_REGISTRY_PATH, "w", encoding="utf-8") as fh:
        json.dump(registry, fh, indent=2, ensure_ascii=False)


def _source_index(registry: dict) -> dict[str, int]:
    """Return {source_id: list_index} mapping."""
    return {s["source_id"]: i for i, s in enumerate(registry["sources"])}


# ── CRUD operations ──────────────────────────────────────────────────

def register_source(
    source_id:        str,
    name:             str,
    url:              str,
    category:         str,
    licence:          str,
    licence_id:       str,
    licence_risk:     float,
    raw_path:         str,
    raw_sha256:       str,
    acquired_by:      str,
    notes:            str = "",
    acquisition_date: str | None = None,
) -> dict:
    """
    Register a new source in the registry.

    Raises ValueError if source_id already exists or category is invalid.
    Returns the new source record.
    """
    if category not in VALID_CATEGORIES:
        raise ValueError(f"Invalid category '{category}'. Must be one of {VALID_CATEGORIES}")

    registry = load_registry()
    idx      = _source_index(registry)

    if source_id in idx:
        raise ValueError(f"Source '{source_id}' already exists. Use update_source() to modify.")

    record: dict[str, Any] = {
        "source_id":           source_id,
        "name":                name,
        "url":                 url,
        "category":            category,
        "licence":             licence,
        "licence_id":          licence_id,
        "licence_risk":        licence_risk,
        "acquisition_date":    acquisition_date or time.strftime("%Y-%m-%d"),
        "acquired_by":         acquired_by,
        "raw_path":            raw_path,
        "raw_sha256":          raw_sha256,
        "source_validated":    False,
        "validation_date":     None,
        "validated_by":        None,
        "status":              "pending",
        "approved_paragraphs": 0,
        "approved_tokens_est": 0,
        "approval_date":       None,
        "approved_by":         None,
        "in_release":          [],
        "notes":               notes,
    }

    registry["sources"].append(record)
    save_registry(registry)
    return record


def get_source(source_id: str) -> dict | None:
    """Return the source record or None if not found."""
    registry = load_registry()
    idx      = _source_index(registry)
    if source_id not in idx:
        return None
    return registry["sources"][idx[source_id]]


def update_source(source_id: str, updates: dict[str, Any]) -> dict:
    """
    Apply field updates to an existing source record.

    Raises KeyError if source_id not found.
    Returns updated record.
    """
    registry = load_registry()
    idx      = _source_index(registry)
    if source_id not in idx:
        raise KeyError(f"Source '{source_id}' not found in registry.")
    record = registry["sources"][idx[source_id]]
    for key, value in updates.items():
        record[key] = value
    save_registry(registry)
    return record


def validate_source(source_id: str, passed: bool,
                    validated_by: str, reason: str = "") -> dict:
    """
    Mark a source as validated (or rejected) after licence + content review.
    Updates status and logs to audit system.
    """
    updates: dict[str, Any] = {
        "source_validated": passed,
        "validation_date":  time.strftime("%Y-%m-%d"),
        "validated_by":     validated_by,
        "status":           "validated" if passed else "rejected",
        "notes":            reason,
    }
    record = update_source(source_id, updates)
    # Audit log
    from audit.logger import log_validate
    log_validate(source_id, passed, reason, operator=validated_by)
    return record


def approve_source(source_id: str, approved_by: str,
                   n_paragraphs: int, n_tokens_est: int) -> dict:
    """
    Mark a source as approved after pipeline completion and human review.
    """
    updates: dict[str, Any] = {
        "status":              "approved",
        "approval_date":       time.strftime("%Y-%m-%d"),
        "approved_by":         approved_by,
        "approved_paragraphs": n_paragraphs,
        "approved_tokens_est": n_tokens_est,
    }
    record = update_source(source_id, updates)
    from audit.logger import log_event
    log_event("approve", source_id, "source_approved", "ok",
              operator=approved_by,
              details={"n_paragraphs": n_paragraphs, "n_tokens_est": n_tokens_est})
    return record


def reject_source(source_id: str, rejected_by: str, reason: str) -> dict:
    """Mark a source as rejected."""
    record = update_source(source_id, {"status": "rejected"})
    from audit.logger import log_event
    log_event("reject", source_id, "source_rejected", "fail",
              operator=rejected_by, details={"reason": reason})
    return record


# ── Query helpers ────────────────────────────────────────────────────

def list_sources(category: str | None = None,
                 status:   str | None = None) -> list[dict]:
    """Return filtered list of source records."""
    registry = load_registry()
    sources  = registry["sources"]
    if category:
        sources = [s for s in sources if s["category"] == category]
    if status:
        sources = [s for s in sources if s["status"] == status]
    return sources


def get_sources_by_status(status: str) -> list[dict]:
    """Return all sources with the given status."""
    return list_sources(status=status)


def validate_source_entry(entry: dict) -> list[str]:
    """
    Validate a source record dict against required schema.
    Returns list of error strings (empty = valid).
    """
    required = ["source_id", "name", "category", "licence", "raw_path",
                "raw_sha256", "acquired_by"]
    errors = []
    for field in required:
        if field not in entry or not entry[field]:
            errors.append(f"Missing required field: {field}")
    if entry.get("category") not in VALID_CATEGORIES:
        errors.append(f"Invalid category: {entry.get('category')}")
    if not isinstance(entry.get("licence_risk"), (int, float)):
        errors.append("licence_risk must be a float")
    return errors


def registry_stats() -> dict:
    """Return aggregate statistics about the registry."""
    registry = load_registry()
    sources  = registry["sources"]
    by_cat: dict[str, int] = {}
    by_status: dict[str, int] = {}
    total_tokens = 0

    for s in sources:
        by_cat[s["category"]]   = by_cat.get(s["category"], 0) + 1
        by_status[s["status"]]  = by_status.get(s["status"], 0) + 1
        total_tokens            += s.get("approved_tokens_est", 0)

    return {
        "total_sources":        len(sources),
        "by_category":          by_cat,
        "by_status":            by_status,
        "total_approved_tokens":total_tokens,
        "registry_hash":        registry.get("registry_hash"),
        "last_updated":         registry.get("last_updated"),
    }
