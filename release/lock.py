"""
CognitiveOC v3 — Release Lock
================================

Once a release is used for training it becomes permanently locked.
A locked release guarantees:
  - No source mutation
  - No train/val/test file mutation
  - No manifest mutation
  - No checksum mutation
  - No silent substitution

Lock file: <warehouse>/releases/<version>/LOCK
  Contents: JSON with lock metadata

Lock is verified by ResumeGuard before every training session.
Attempting to build a new release with the same version as a locked
release is blocked unless the lock is explicitly broken (admin only,
with audit trail).

Usage:
    from release.lock import lock_release, is_locked, verify_lock
    lock_release("v1", operator="mpssp")
    is_locked("v1")              # → True
    verify_lock("v1")            # → raises if tampered
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

from audit.logger import log_event

try:
    from config import CORPUS_WAREHOUSE_DIR
    _WHOUSE = Path(CORPUS_WAREHOUSE_DIR)
except (ImportError, AttributeError):
    _WHOUSE = Path("var/corpus_warehouse")


class ReleaseLockError(RuntimeError):
    """Raised when a release lock check fails."""
    pass


def _release_dir(release_id: str) -> Path:
    return _WHOUSE / "releases" / release_id


def _lock_path(release_id: str) -> Path:
    return _release_dir(release_id) / "LOCK"


def _sha256_file(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _manifest_hash(release_id: str) -> str:
    return _sha256_file(_release_dir(release_id) / "manifest.json")


def _split_hashes(release_id: str) -> dict[str, str]:
    rdir = _release_dir(release_id)
    return {
        "train": _sha256_file(rdir / "train.txt"),
        "val":   _sha256_file(rdir / "val.txt"),
        "test":  _sha256_file(rdir / "test.txt"),
    }


# ── Lock operations ───────────────────────────────────────────────────

def lock_release(release_id: str, operator: str,
                 notes: str = "") -> dict:
    """
    Lock a signed release for training.

    Records a cryptographic snapshot of every release artifact at lock time.
    Any subsequent mutation of these files will be detected by verify_lock().

    Args:
        release_id: Release version string (e.g. "v1").
        operator:   Username of the person locking the release.
        notes:      Optional notes (e.g. "Locking for Phase 1 pre-training").

    Returns:
        The lock record dict.

    Raises:
        ReleaseLockError if the release is not signed or already locked.
    """
    rdir = _release_dir(release_id)
    if not rdir.exists():
        raise ReleaseLockError(
            f"Release directory not found: {rdir}. "
            f"Build and sign the release first."
        )

    manifest_path = rdir / "manifest.json"
    if not manifest_path.exists():
        raise ReleaseLockError(f"Release manifest not found: {manifest_path}")

    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)

    if manifest.get("status") not in ("signed",):
        raise ReleaseLockError(
            f"Release '{release_id}' must be signed before locking. "
            f"Current status: {manifest.get('status', 'unknown')}. "
            f"Run: python main.py corpus sign-release {release_id} --operator {operator}"
        )

    if is_locked(release_id):
        raise ReleaseLockError(
            f"Release '{release_id}' is already locked. "
            f"Inspect: {_lock_path(release_id)}"
        )

    lock_record = {
        "release_id":     release_id,
        "locked_at":      time.strftime("%Y-%m-%dT%H:%M:%S"),
        "locked_by":      operator,
        "notes":          notes,
        "manifest_hash":  _manifest_hash(release_id),
        "split_hashes":   _split_hashes(release_id),
        "checksums_hash": _sha256_file(rdir / "checksums.sha256"),
        "coc_version":    "v3",
    }

    lock_path = _lock_path(release_id)
    tmp       = lock_path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(lock_record, fh, indent=2, ensure_ascii=False)
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(lock_path)

    log_event(
        stage="release", source_id=f"release-{release_id}",
        action="release_locked", result="ok", operator=operator,
        hash_val=lock_record["manifest_hash"],
        details={"notes": notes, "split_hashes": lock_record["split_hashes"]},
    )

    print(f"\n✓ Release '{release_id}' locked by {operator}")
    print(f"  Manifest hash : {lock_record['manifest_hash'][:24]}...")
    print(f"  Lock file     : {lock_path}\n")
    return lock_record


def is_locked(release_id: str) -> bool:
    """Return True if the release has a LOCK file."""
    return _lock_path(release_id).exists()


def verify_lock(release_id: str) -> dict:
    """
    Verify the lock is intact — no files have been mutated since locking.

    Returns:
        {"valid": True, "errors": [], "lock": lock_record}

    Raises:
        ReleaseLockError if the lock is missing or any hash mismatches.
    """
    lp = _lock_path(release_id)
    if not lp.exists():
        raise ReleaseLockError(
            f"Release '{release_id}' has no LOCK file. "
            f"Lock it before training: python main.py corpus lock-release {release_id}"
        )

    with open(lp, encoding="utf-8") as fh:
        lock = json.load(fh)

    errors: list[str] = []

    # Manifest hash
    current_manifest_hash = _manifest_hash(release_id)
    if lock["manifest_hash"] != current_manifest_hash:
        errors.append(
            f"Manifest hash mismatch. "
            f"Locked: {lock['manifest_hash'][:16]}... "
            f"Current: {current_manifest_hash[:16]}... "
            f"The release manifest was modified after locking."
        )

    # Split file hashes
    current_splits = _split_hashes(release_id)
    for split_name, expected_hash in lock.get("split_hashes", {}).items():
        current = current_splits.get(split_name, "")
        if expected_hash and current != expected_hash:
            errors.append(
                f"{split_name}.txt hash mismatch. "
                f"Locked: {expected_hash[:16]}... "
                f"Current: {current[:16]}... "
                f"The {split_name} split was modified after locking."
            )

    # Checksums file hash
    current_csum = _sha256_file(_release_dir(release_id) / "checksums.sha256")
    if lock.get("checksums_hash") and lock["checksums_hash"] != current_csum:
        errors.append(
            f"checksums.sha256 hash mismatch. "
            f"File was modified after locking."
        )

    if errors:
        log_event(
            stage="release", source_id=f"release-{release_id}",
            action="lock_verify_failed", result="fail", operator="system",
            details={"errors": errors},
        )
        raise ReleaseLockError(
            f"Lock verification failed for release '{release_id}':\n"
            + "\n".join(f"  - {e}" for e in errors)
        )

    log_event(
        stage="release", source_id=f"release-{release_id}",
        action="lock_verified", result="ok", operator="system",
    )

    return {"valid": True, "errors": [], "lock": lock}


def break_lock(release_id: str, operator: str,
               reason: str) -> None:
    """
    Admin-only: remove the lock file from a release.

    This is a destructive admin action and requires an explicit reason.
    It is permanently logged in the audit trail.

    ONLY use this to:
      - Recall a bad release
      - Fix a lock file corruption
    """
    lp = _lock_path(release_id)
    if not lp.exists():
        print(f"Release '{release_id}' is not locked.")
        return

    with open(lp, encoding="utf-8") as fh:
        lock = json.load(fh)

    # Move to archive rather than delete
    archive = _release_dir(release_id) / f"LOCK.broken.{int(time.time())}"
    lp.rename(archive)

    log_event(
        stage="release", source_id=f"release-{release_id}",
        action="lock_broken", result="ok", operator=operator,
        details={
            "reason":       reason,
            "original_lock":lock,
            "archived_to":  str(archive),
        },
    )

    print(f"\n⚠  Lock broken for release '{release_id}' by {operator}")
    print(f"   Reason   : {reason}")
    print(f"   Archived : {archive}")
    print(f"   This action is permanently recorded in the audit log.\n")


def lock_status(release_id: str) -> dict:
    """Return lock status information without raising on failure."""
    if not is_locked(release_id):
        return {"locked": False, "release_id": release_id}
    lp = _lock_path(release_id)
    with open(lp, encoding="utf-8") as fh:
        lock = json.load(fh)
    try:
        vr = verify_lock(release_id)
        intact = vr["valid"]
        errors = []
    except ReleaseLockError as e:
        intact = False
        errors = [str(e)]
    return {
        "locked":      True,
        "release_id":  release_id,
        "locked_at":   lock.get("locked_at"),
        "locked_by":   lock.get("locked_by"),
        "intact":      intact,
        "errors":      errors,
    }
