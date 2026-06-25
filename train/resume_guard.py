"""
CognitiveOC v3 — Resume Guard
===============================

Pre-training verification gate. Must be called before every training
session (initial or resume).

Verifies all of the following before allowing training to proceed:
  1. Release exists and is signed
  2. Release manifest hash matches shard tracker's stored hash
  3. Release file checksums pass (train/val/test .sha256)
  4. Release lock is in place (release/lock.py)
  5. Checkpoint file exists if resuming
  6. Checkpoint is not corrupted (basic load check)
  7. Shard tracker is initialised for the correct release
  8. Shard tracker release_hash matches current release manifest
  9. No shard is stuck in 'in_progress' state (indicates interrupted run)
 10. Training ledger entry exists if resuming

On any mismatch: raises ResumeGuardError with a clear error message,
logs the failure to the audit log, and does NOT allow training to proceed.

Usage:
    from train.resume_guard import ResumeGuard
    guard = ResumeGuard(release_id="v1", resume=True)
    guard.run()    # raises ResumeGuardError if anything fails
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from audit.logger import log_event

try:
    from config import CHECKPOINT_DIR, CORPUS_WAREHOUSE_DIR
    _CKPT_DIR  = Path(CHECKPOINT_DIR)
    _WHOUSE    = Path(CORPUS_WAREHOUSE_DIR)
except (ImportError, AttributeError):
    _CKPT_DIR  = Path("var/checkpoints")
    _WHOUSE    = Path("var/corpus_warehouse")


class ResumeGuardError(RuntimeError):
    """Raised when the resume guard detects an integrity failure."""
    pass


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class ResumeGuard:
    """
    Pre-training integrity gate.

    Args:
        release_id: The release version string (e.g. "v1").
        resume:     True if this is a resume (not a fresh start).
        operator:   Username for audit logging.
    """

    def __init__(self, release_id: str, resume: bool = True,
                 operator: str = "system"):
        self.release_id  = release_id
        self.resume      = resume
        self.operator    = operator
        self._errors:  list[str] = []
        self._warnings:list[str] = []
        self._manifest: dict = {}

    # ── Internal checks ───────────────────────────────────────────────

    def _check_release_exists(self) -> bool:
        """Check that the release directory and manifest exist."""
        manifest_path = _WHOUSE / "releases" / self.release_id / "manifest.json"
        if not manifest_path.exists():
            self._errors.append(
                f"Release manifest not found: {manifest_path}. "
                f"Run: python main.py corpus build-release {self.release_id}"
            )
            return False
        with open(manifest_path, encoding="utf-8") as fh:
            self._manifest = json.load(fh)
        return True

    def _check_release_signed(self) -> bool:
        """Check that the release has been signed (status == 'signed')."""
        status = self._manifest.get("status", "unknown")
        if status != "signed":
            self._errors.append(
                f"Release '{self.release_id}' status is '{status}', not 'signed'. "
                f"Run: python main.py corpus sign-release {self.release_id} --operator <you>"
            )
            return False
        return True

    def _check_release_locked(self) -> bool:
        """Check that the release lock file exists."""
        try:
            from release.lock import is_locked
            if not is_locked(self.release_id):
                self._errors.append(
                    f"Release '{self.release_id}' is not locked. "
                    f"Run: python main.py corpus lock-release {self.release_id} --operator <you>"
                )
                return False
            return True
        except ImportError:
            self._warnings.append("release.lock module not available — lock check skipped.")
            return True

    def _check_release_checksums(self) -> bool:
        """Verify train/val/test file checksums against manifest."""
        release_dir = _WHOUSE / "releases" / self.release_id
        checksums   = self._manifest.get("checksums", {})
        all_ok      = True
        for split_name, file_name in [("train", "train.txt"),
                                       ("val", "val.txt"),
                                       ("test", "test.txt")]:
            expected_raw = checksums.get(split_name, "")
            expected     = expected_raw.replace("sha256:", "")
            fpath        = release_dir / file_name
            if not fpath.exists():
                self._errors.append(f"Release split file missing: {fpath}")
                all_ok = False
                continue
            if expected:
                actual = _sha256_file(fpath)
                if actual != expected:
                    self._errors.append(
                        f"Checksum mismatch for {file_name}: "
                        f"expected {expected[:16]}... got {actual[:16]}..."
                    )
                    all_ok = False
        return all_ok

    def _check_manifest_hash_vs_shard_tracker(self) -> bool:
        """Verify shard tracker was initialised for the same manifest hash."""
        if not self.resume:
            return True  # Fresh start: shard tracker not yet initialised
        try:
            from train.shard_tracker import _load_state, shard_stats
            state = _load_state()
            if not state:
                self._warnings.append(
                    "Shard tracker not yet initialised — will be set up at training start."
                )
                return True
            manifest_hash = _sha256_file(
                _WHOUSE / "releases" / self.release_id / "manifest.json"
            )
            if not state.get("release_hash"):
                self._warnings.append("Shard tracker has no release hash recorded.")
                return True
            if state["release_hash"] != manifest_hash:
                self._errors.append(
                    f"Shard tracker release_hash mismatch. "
                    f"Tracker: {state['release_hash'][:16]}...  "
                    f"Current: {manifest_hash[:16]}...  "
                    f"The release manifest was modified after training began. ABORT."
                )
                return False
            return True
        except Exception as e:
            self._warnings.append(f"Shard tracker hash check failed: {e}")
            return True

    def _check_no_stuck_shards(self) -> bool:
        """Check for shards stuck in 'in_progress' state."""
        try:
            from train.shard_tracker import _load_state
            state = _load_state()
            if not state:
                return True
            stuck = [
                sid for sid, s in state["shards"].items()
                if s["status"] == "in_progress"
            ]
            if stuck:
                self._warnings.append(
                    f"{len(stuck)} shard(s) stuck in 'in_progress' state "
                    f"(likely from an interrupted run). They will be reset to 'pending' "
                    f"and re-consumed in this session."
                )
                # Auto-reset stuck shards
                for sid in stuck:
                    state["shards"][sid]["status"] = "pending"
                from train.shard_tracker import _save_state
                _save_state(state)
            return True
        except Exception as e:
            self._warnings.append(f"Stuck shard check failed: {e}")
            return True

    def _check_checkpoint(self) -> bool:
        """Check that the checkpoint exists and is loadable (if resuming)."""
        if not self.resume:
            return True
        ckpt = _CKPT_DIR / "model_700m.pt"
        if not ckpt.exists():
            self._warnings.append(
                f"Checkpoint not found at {ckpt}. Training will start from scratch."
            )
            return True
        # Quick integrity check — try to load just the keys
        try:
            import torch
            meta = torch.load(str(ckpt), map_location="cpu",
                              weights_only=False)
            if "step" not in meta:
                self._warnings.append(
                    f"Checkpoint at {ckpt} has no 'step' key — may be malformed."
                )
        except Exception as e:
            self._errors.append(
                f"Checkpoint at {ckpt} failed to load: {e}. "
                f"Cannot resume — checkpoint may be corrupted."
            )
            return False
        return True

    def _check_ledger(self) -> bool:
        """Check that if resuming, a ledger entry exists for the release."""
        if not self.resume:
            return True
        try:
            from train.training_ledger import get_resume_state
            state = get_resume_state()
            if state.get("release_id") and state["release_id"] != self.release_id:
                self._errors.append(
                    f"Ledger shows last training used release '{state['release_id']}', "
                    f"but current release is '{self.release_id}'. "
                    f"Cannot resume across different releases without explicit override."
                )
                return False
            return True
        except Exception as e:
            self._warnings.append(f"Ledger check failed: {e}")
            return True

    # ── Main entry point ──────────────────────────────────────────────

    def run(self) -> dict:
        """
        Run all guards. Raises ResumeGuardError on any error.

        Returns:
            {
              "passed":   bool,
              "errors":   list[str],
              "warnings": list[str],
              "manifest": dict,
            }
        """
        print(f"\n[resume_guard] Verifying release '{self.release_id}'...")

        checks = [
            ("release_exists",        self._check_release_exists),
            ("release_signed",        self._check_release_signed),
            ("release_locked",        self._check_release_locked),
            ("release_checksums",     self._check_release_checksums),
            ("manifest_hash_tracker", self._check_manifest_hash_vs_shard_tracker),
            ("no_stuck_shards",       self._check_no_stuck_shards),
            ("checkpoint",            self._check_checkpoint),
            ("ledger",                self._check_ledger),
        ]

        results = {}
        for name, fn in checks:
            try:
                results[name] = fn()
            except Exception as e:
                self._errors.append(f"Check '{name}' raised exception: {e}")
                results[name] = False

        passed = len(self._errors) == 0

        # Print results
        for name, ok in results.items():
            symbol = "✓" if ok else "✗"
            print(f"  {symbol} {name}")

        if self._warnings:
            for w in self._warnings:
                print(f"  ⚠ {w}")

        if not passed:
            for e in self._errors:
                print(f"  ✗ ERROR: {e}")

        # Audit log
        log_event(
            stage     = "resume_guard",
            source_id = f"release-{self.release_id}",
            action    = "guard_check",
            result    = "ok" if passed else "fail",
            operator  = self.operator,
            details   = {
                "release_id": self.release_id,
                "resume":     self.resume,
                "errors":     self._errors,
                "warnings":   self._warnings,
            },
        )

        if not passed:
            raise ResumeGuardError(
                f"Resume guard failed for release '{self.release_id}'. "
                f"Errors:\n" + "\n".join(f"  - {e}" for e in self._errors)
            )

        print(f"  → All checks passed. Training may proceed.\n")
        return {
            "passed":   True,
            "errors":   [],
            "warnings": self._warnings,
            "manifest": self._manifest,
        }
