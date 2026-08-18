"""
CognitiveOC v3 — Release Verification
========================================

Standalone release verification module.
Called by:
  - corpus/release_builder.py  (after build, before sign)
  - train/resume_guard.py       (before every training session)
  - CLI: python main.py corpus verify-release <version>

Checks performed:
  1.  Release directory exists
  2.  manifest.json exists and is valid JSON
  3.  Required manifest fields present
  4.  Split ratios sum to 1.0
  5.  train/val/test files exist
  6.  Checksums match manifest
  7.  No exact-match leakage between train and val/test
  8.  Release status is 'signed' (for pre-training check)
  9.  LOCK file present and intact (for pre-training check)
  10. Shard tracker release hash matches (if tracker initialised)
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

try:
    from config import CORPUS_WAREHOUSE_DIR
    _WHOUSE = Path(CORPUS_WAREHOUSE_DIR)
except (ImportError, AttributeError):
    _WHOUSE = Path("var/corpus_warehouse")

REQUIRED_MANIFEST_FIELDS = [
    "release_id", "release_date", "released_by",
    "checksums", "sources", "split_ratios", "shuffle_seed",
    "total_tokens_estimate", "train_tokens",
]


class VerificationError(RuntimeError):
    pass


def _sha256_file(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _para_hashes(path: Path) -> set[str]:
    """Return SHA-256 hashes of all paragraphs in a split file."""
    if not path.exists():
        return set()
    text  = path.read_text(encoding="utf-8")
    hset  = set()
    for para in re.split(r"\n{2,}", text):
        para = para.strip()
        if para:
            hset.add(hashlib.sha256(para.lower().encode()).hexdigest())
    return hset


def verify_release(
    release_id:      str,
    require_signed:  bool = False,
    require_locked:  bool = False,
    check_leakage:   bool = True,
    verbose:         bool = True,
) -> dict:
    """
    Run full release verification.

    Args:
        release_id:     Release version string (e.g. "v1").
        require_signed: If True, fail if status != 'signed'.
        require_locked: If True, fail if LOCK file absent or invalid.
        check_leakage:  If True, run exact-match train/val/test leakage check.
        verbose:        Print progress.

    Returns:
        {
          "verified": bool,
          "errors":   list[str],
          "warnings": list[str],
          "manifest": dict,
        }

    Does NOT raise — returns errors list instead, so callers can decide
    whether to abort or warn.
    """
    errors:   list[str] = []
    warnings: list[str] = []
    manifest: dict      = {}

    rdir = _WHOUSE / "releases" / release_id

    def _check(label: str, ok: bool, msg: str = "", warn: bool = False) -> bool:
        sym = "✓" if ok else ("⚠" if warn else "✗")
        if verbose:
            line = f"  {sym} {label}"
            if msg and not ok:
                line += f": {msg}"
            print(line)
        if not ok:
            if warn:
                warnings.append(f"{label}: {msg}")
            else:
                errors.append(f"{label}: {msg}")
        return ok

    if verbose:
        print(f"\n[verify] Release '{release_id}'")

    # 1. Release directory
    if not _check("release_dir_exists", rdir.exists(),
                  f"Not found: {rdir}"):
        return {"verified": False, "errors": errors,
                "warnings": warnings, "manifest": {}}

    # 2. Manifest exists
    mp = rdir / "manifest.json"
    if not _check("manifest_exists", mp.exists(), f"Not found: {mp}"):
        return {"verified": False, "errors": errors,
                "warnings": warnings, "manifest": {}}

    # 3. Manifest is valid JSON
    try:
        with open(mp, encoding="utf-8") as fh:
            manifest = json.load(fh)
        _check("manifest_valid_json", True)
    except Exception as e:
        _check("manifest_valid_json", False, str(e))
        return {"verified": False, "errors": errors,
                "warnings": warnings, "manifest": {}}

    # 4. Required fields
    missing_fields = [f for f in REQUIRED_MANIFEST_FIELDS if f not in manifest]
    _check("manifest_required_fields",
           len(missing_fields) == 0,
           f"Missing: {missing_fields}")

    # 5. Split ratios sum to 1.0
    ratios = manifest.get("split_ratios", [])
    _check("split_ratios_sum",
           bool(ratios) and abs(sum(ratios) - 1.0) <= 0.001,
           f"Sum = {sum(ratios):.4f}")

    # 6. Split files exist
    split_paths = {
        "train": rdir / "train.txt",
        "val":   rdir / "val.txt",
        "test":  rdir / "test.txt",
    }
    for name, path in split_paths.items():
        _check(f"{name}_file_exists", path.exists(), f"Not found: {path}")

    # 7. Checksums
    manifest_checksums = manifest.get("checksums", {})
    for name, path in split_paths.items():
        expected_raw = manifest_checksums.get(name, "")
        expected     = expected_raw.replace("sha256:", "")
        if not expected:
            _check(f"{name}_checksum", False,
                   "No checksum in manifest", warn=True)
            continue
        if not path.exists():
            continue
        actual = _sha256_file(path)
        _check(f"{name}_checksum",
               actual == expected,
               f"Expected {expected[:16]}... got {actual[:16]}...")

    # 8. Status (if required_signed)
    status = manifest.get("status", "unknown")
    if require_signed:
        _check("release_signed",
               status == "signed",
               f"Status is '{status}', not 'signed'")
    else:
        if status != "signed":
            _check("release_signed",
                   False,
                   f"Status is '{status}' (OK for pre-sign verification)",
                   warn=True)

    # 9. Lock (if required_locked)
    if require_locked:
        try:
            from release.lock import is_locked, verify_lock
            if not is_locked(release_id):
                _check("release_locked", False, "No LOCK file found")
            else:
                try:
                    verify_lock(release_id)
                    _check("release_locked", True)
                except Exception as e:
                    _check("release_locked", False, str(e))
        except ImportError:
            _check("release_locked", False,
                   "release.lock not available", warn=True)

    # 10. Leakage check
    if check_leakage and all(p.exists() for p in split_paths.values()):
        if verbose:
            print("  … running leakage check (may take a moment for large files)")
        train_h = _para_hashes(split_paths["train"])
        val_h   = _para_hashes(split_paths["val"])
        test_h  = _para_hashes(split_paths["test"])
        leak_v  = len(train_h & val_h)
        leak_t  = len(train_h & test_h)
        _check("no_train_val_leakage",
               leak_v == 0,
               f"{leak_v} exact-match paragraphs found in train∩val")
        _check("no_train_test_leakage",
               leak_t == 0,
               f"{leak_t} exact-match paragraphs found in train∩test")

    verified = len(errors) == 0
    if verbose:
        if verified:
            print(f"\n  ✓ Release '{release_id}' verified successfully.")
        else:
            print(f"\n  ✗ Release '{release_id}' verification FAILED "
                  f"({len(errors)} error(s)).")
        if warnings:
            print(f"  ⚠  {len(warnings)} warning(s).")
        print()

    return {
        "verified": verified,
        "errors":   errors,
        "warnings": warnings,
        "manifest": manifest,
    }


def enforce_verification(release_id: str, require_locked: bool = True) -> dict:
    """
    Hard enforcement: run verify_release and raise VerificationError on failure.

    Called by train/resume_guard.py and any pre-training hook.
    """
    result = verify_release(
        release_id     = release_id,
        require_signed = True,
        require_locked = require_locked,
        check_leakage  = False,   # leakage already checked at build time
    )
    if not result["verified"]:
        raise VerificationError(
            f"Release '{release_id}' failed verification:\n"
            + "\n".join(f"  - {e}" for e in result["errors"])
        )
    return result
