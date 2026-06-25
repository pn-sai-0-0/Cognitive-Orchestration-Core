"""
CognitiveOC v3 — Audit Logger
==============================

Append-only structured JSONL audit log consumed by every corpus pipeline
stage and the governance system.

Log file location: governance/approval_log.jsonl (inside repo for commits)
                   warehouse/governance_logs/audit_<YYYYMMDD>.jsonl (daily shards)

Rules:
  - Append-only. No deletes. No overwrites.
  - Every pipeline action that changes a source's status MUST call log_event().
  - log_event() is synchronous and flushes immediately.
  - Thread-safe via file-level locking (portalocker if available, else OS lock).

Event schema (JSON, one object per line):
  {
    "ts":        ISO-8601 timestamp,
    "stage":     pipeline stage name (acquire|validate|normalize|clean|dedup|
                 score|review|approve|split|release|archive|reject),
    "source_id": source identifier string,
    "action":    specific action taken,
    "result":    "ok" | "fail" | "queued" | "skipped",
    "operator":  human or "system",
    "hash":      optional SHA-256 of affected file,
    "details":   optional dict of extra context
  }
"""

from __future__ import annotations

import fcntl
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ── Config fallback (import config if available) ─────────────────────
try:
    from config import CORPUS_AUDIT_LOG_DIR, CORPUS_APPROVAL_LOG
    _APPROVAL_LOG  = Path(CORPUS_APPROVAL_LOG)
    _AUDIT_LOG_DIR = Path(CORPUS_AUDIT_LOG_DIR)
except ImportError:
    _APPROVAL_LOG  = Path("governance/approval_log.jsonl")
    _AUDIT_LOG_DIR = Path("var/logs/corpus_audit")


def _approval_log_path() -> Path:
    _APPROVAL_LOG.parent.mkdir(parents=True, exist_ok=True)
    return _APPROVAL_LOG


def _daily_log_path() -> Path:
    """Return path for today's daily audit shard."""
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    _AUDIT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    return _AUDIT_LOG_DIR / f"audit_{today}.jsonl"


def _write_event(path: Path, event: dict) -> None:
    """Write event dict as a single JSONL line, file-locked."""
    line = json.dumps(event, ensure_ascii=False) + "\n"
    with open(path, "a", encoding="utf-8") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX)
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def log_event(
    stage:     str,
    source_id: str,
    action:    str,
    result:    str,
    operator:  str = "system",
    hash_val:  str | None = None,
    details:   dict[str, Any] | None = None,
) -> dict:
    """
    Write a single audit event to both the approval log and the daily shard.

    Args:
        stage:     Pipeline stage (see module docstring for valid values).
        source_id: Source identifier (e.g. "A-gutenberg-20260701").
        action:    Specific action string (e.g. "source_registered").
        result:    "ok" | "fail" | "queued" | "skipped".
        operator:  Human username or "system".
        hash_val:  Optional SHA-256 hex digest of the affected artifact.
        details:   Optional dict of additional context.

    Returns:
        The event dict that was written.
    """
    event: dict[str, Any] = {
        "ts":        datetime.now(timezone.utc).isoformat(),
        "ts_unix":   time.time(),
        "stage":     stage,
        "source_id": source_id,
        "action":    action,
        "result":    result,
        "operator":  operator,
    }
    if hash_val is not None:
        event["hash"] = hash_val
    if details:
        event["details"] = details

    # Write to both logs
    _write_event(_approval_log_path(), event)
    _write_event(_daily_log_path(), event)

    return event


def log_acquire(source_id: str, path: str, sha256: str,
                operator: str = "system") -> dict:
    """Shortcut: log a source acquisition event."""
    return log_event(
        stage="acquire", source_id=source_id,
        action="source_acquired", result="ok", operator=operator,
        hash_val=sha256, details={"path": path},
    )


def log_validate(source_id: str, passed: bool,
                 reason: str = "", operator: str = "system") -> dict:
    """Shortcut: log source validation outcome."""
    return log_event(
        stage="validate", source_id=source_id,
        action="source_validated" if passed else "source_rejected",
        result="ok" if passed else "fail",
        operator=operator, details={"reason": reason},
    )


def log_stage(stage: str, source_id: str, action: str,
              n_in: int, n_out: int, operator: str = "system") -> dict:
    """Shortcut: log a processing stage with input/output paragraph counts."""
    return log_event(
        stage=stage, source_id=source_id, action=action,
        result="ok", operator=operator,
        details={"n_in": n_in, "n_out": n_out,
                 "removed": n_in - n_out,
                 "retention_pct": round(n_out / n_in * 100, 1) if n_in else 0},
    )


def log_review(source_id: str, item_id: str, decision: str,
               operator: str, reason: str = "") -> dict:
    """Shortcut: log a human review decision."""
    return log_event(
        stage="review", source_id=source_id,
        action=f"review_{decision}",   # review_approve | review_reject
        result="ok", operator=operator,
        details={"item_id": item_id, "reason": reason},
    )


def log_release(release_id: str, operator: str,
                token_count: int, sha256_train: str) -> dict:
    """Shortcut: log a release approval/signing event."""
    return log_event(
        stage="release", source_id="release",
        action="release_signed", result="ok", operator=operator,
        hash_val=sha256_train,
        details={"release_id": release_id, "token_count": token_count},
    )
