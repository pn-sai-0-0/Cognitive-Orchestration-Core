"""
CognitiveOC v3 — Shard Tracker
================================

Tracks exact shard (data chunk) consumption during training to prevent:
  - duplicate shard consumption across sessions
  - loss of position after restart
  - silent data drift between runs

A "shard" is a fixed-size token window drawn from the release train split.
The shard tracker divides the training corpus into deterministic shards
numbered 0..N-1, and records which have been fully consumed.

Shard state file: var/checkpoints/shard_tracker.json
  (written atomically — temp file then rename)

Shard record schema per shard:
{
  "shard_id":    int   — sequential shard number
  "token_start": int   — byte offset into train.txt in tokens
  "token_end":   int   — exclusive end
  "token_count": int
  "sha256":      str   — SHA-256 of the shard text (first 512 chars as proxy)
  "status":      str   — "pending" | "in_progress" | "completed"
  "consumed_at": str | None  — ISO timestamp when completed
  "run_id":      str | None  — which training run consumed it
  "step_range":  [start, end] | None  — training steps during which it was consumed
}

Tracker state:
{
  "release_id":    str
  "release_hash":  str   — SHA-256 of manifest; mismatch = abort
  "shard_size":    int   — tokens per shard
  "total_shards":  int
  "shards":        {str(shard_id): shard_record}
  "last_updated":  str
}
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Iterator

try:
    from config import CHECKPOINT_DIR
    _STATE_PATH = Path(CHECKPOINT_DIR) / "shard_tracker.json"
except (ImportError, AttributeError):
    _STATE_PATH = Path("var/checkpoints/shard_tracker.json")

# Default shard size: 10M tokens (~1 hour of training at 2000 tok/s × 3600s)
_DEFAULT_SHARD_SIZE = 10_000_000


def _state_path() -> Path:
    _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    return _STATE_PATH


def _atomic_write(path: Path, data: dict) -> None:
    """Write JSON atomically via temp-file rename."""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(path)


def _sha256_sample(text: str, n: int = 512) -> str:
    """Hash the first n characters of text as a shard identity proxy."""
    return hashlib.sha256(text[:n].encode(errors="replace")).hexdigest()


# ── State management ──────────────────────────────────────────────────

def _load_state() -> dict | None:
    p = _state_path()
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def _save_state(state: dict) -> None:
    state["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    _atomic_write(_state_path(), state)


# ── Initialisation ────────────────────────────────────────────────────

def initialise(
    release_id:   str,
    release_hash: str,
    train_tokens: list[int],
    shard_size:   int = _DEFAULT_SHARD_SIZE,
    force:        bool = False,
) -> dict:
    """
    Initialise the shard tracker for a new training run.

    Args:
        release_id:   Release version string (e.g. "v1").
        release_hash: SHA-256 of the release manifest.
        train_tokens: Full list of tokenised training tokens.
        shard_size:   Tokens per shard (default 10M).
        force:        If True, reinitialise even if state exists.

    Returns:
        Shard tracker state dict.

    Raises:
        RuntimeError if state exists with a different release and force=False.
    """
    existing = _load_state()
    if existing and not force:
        if existing["release_id"] != release_id:
            raise RuntimeError(
                f"Shard tracker already initialised for release "
                f"'{existing['release_id']}', not '{release_id}'. "
                f"Use force=True to reinitialise (WARNING: resets progress)."
            )
        if existing["release_hash"] != release_hash:
            raise RuntimeError(
                f"Shard tracker release_hash mismatch. "
                f"Release manifest may have been modified after training started. "
                f"This is a data integrity violation — abort."
            )
        # Already initialised for this release — return existing
        return existing

    n       = len(train_tokens)
    shards  = {}
    shard_id = 0
    offset   = 0

    while offset < n:
        end    = min(offset + shard_size, n)
        sample = " ".join(str(t) for t in train_tokens[offset:offset + 64])
        shards[str(shard_id)] = {
            "shard_id":    shard_id,
            "token_start": offset,
            "token_end":   end,
            "token_count": end - offset,
            "sha256":      _sha256_sample(sample),
            "status":      "pending",
            "consumed_at": None,
            "run_id":      None,
            "step_range":  None,
        }
        shard_id += 1
        offset    = end

    state = {
        "release_id":   release_id,
        "release_hash": release_hash,
        "shard_size":   shard_size,
        "total_shards": shard_id,
        "shards":       shards,
        "last_updated": None,
    }
    _save_state(state)
    return state


# ── Consumption tracking ──────────────────────────────────────────────

def get_next_shard() -> dict | None:
    """
    Return the next pending shard to consume, or None if all are done.
    Also marks it as 'in_progress'.
    """
    state = _load_state()
    if not state:
        return None

    for sid in sorted(state["shards"], key=int):
        sh = state["shards"][sid]
        if sh["status"] == "pending":
            sh["status"]     = "in_progress"
            _save_state(state)
            return sh

    return None  # All shards consumed


def mark_shard_complete(
    shard_id:   int,
    run_id:     str,
    step_start: int,
    step_end:   int,
) -> None:
    """
    Mark a shard as fully consumed.

    Args:
        shard_id:   The shard index.
        run_id:     The training run that consumed it.
        step_start: First training step during this shard.
        step_end:   Last training step during this shard.
    """
    state = _load_state()
    if not state:
        raise RuntimeError("Shard tracker not initialised.")

    sid = str(shard_id)
    if sid not in state["shards"]:
        raise KeyError(f"Shard {shard_id} not found in tracker.")

    state["shards"][sid].update({
        "status":      "completed",
        "consumed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "run_id":      run_id,
        "step_range":  [step_start, step_end],
    })
    _save_state(state)


def mark_shard_interrupted(shard_id: int) -> None:
    """
    Reset an in-progress shard back to pending (called on training interruption).
    This allows safe resume — the shard will be re-consumed.
    """
    state = _load_state()
    if not state:
        return
    sid = str(shard_id)
    if sid in state["shards"] and state["shards"][sid]["status"] == "in_progress":
        state["shards"][sid]["status"] = "pending"
        _save_state(state)


def verify_release_hash(release_hash: str) -> bool:
    """
    Verify that the current tracker was initialised for the given release hash.
    Returns False if mismatch (caller must abort).
    """
    state = _load_state()
    if not state:
        return True  # Not yet initialised — no mismatch
    return state.get("release_hash") == release_hash


def shard_stats() -> dict:
    """Return aggregate shard consumption statistics."""
    state = _load_state()
    if not state:
        return {"initialised": False}

    shards    = state["shards"]
    pending   = sum(1 for s in shards.values() if s["status"] == "pending")
    in_prog   = sum(1 for s in shards.values() if s["status"] == "in_progress")
    completed = sum(1 for s in shards.values() if s["status"] == "completed")
    total     = state["total_shards"]
    tok_done  = sum(s["token_count"] for s in shards.values()
                    if s["status"] == "completed")
    tok_total = sum(s["token_count"] for s in shards.values())

    return {
        "initialised":    True,
        "release_id":     state["release_id"],
        "release_hash":   state["release_hash"][:12] + "...",
        "total_shards":   total,
        "pending":        pending,
        "in_progress":    in_prog,
        "completed":      completed,
        "pct_complete":   round(completed / total * 100, 1) if total else 0,
        "tokens_done":    tok_done,
        "tokens_total":   tok_total,
        "tokens_pct":     round(tok_done / tok_total * 100, 1) if tok_total else 0,
        "shard_size":     state["shard_size"],
    }


def print_shard_status() -> None:
    """Print a human-readable shard status to stdout."""
    stats = shard_stats()
    print(f"\n{'='*68}")
    print(f"COC v3 — Shard Tracker Status")
    print(f"{'='*68}")
    if not stats.get("initialised"):
        print("  Not yet initialised. Run a training session first.")
    else:
        print(f"  Release:       {stats['release_id']}  [{stats['release_hash']}]")
        print(f"  Total shards:  {stats['total_shards']}")
        print(f"  Completed:     {stats['completed']}  ({stats['pct_complete']}%)")
        print(f"  In progress:   {stats['in_progress']}")
        print(f"  Pending:       {stats['pending']}")
        print(f"  Tokens done:   {stats['tokens_done']:,}  / {stats['tokens_total']:,}  ({stats['tokens_pct']}%)")
    print(f"{'='*68}\n")
