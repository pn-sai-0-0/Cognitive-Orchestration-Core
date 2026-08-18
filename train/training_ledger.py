"""
CognitiveOC v3 — Training Ledger
==================================

Persistent, append-only record of every training run and resume event.

The ledger answers:
  - What runs have occurred?
  - What release was used in each run?
  - What step range was covered?
  - How many tokens were consumed?
  - What checkpoints were produced?
  - When did each session start and end?

Ledger file: var/training_ledger.jsonl  (append-only, never overwritten)

Run record schema:
{
  "run_id":          str   — unique e.g. "run_20260701_143022"
  "release_id":      str   — e.g. "v1"
  "release_hash":    str   — SHA-256 of release manifest at run start
  "run_type":        str   — "pretrain" | "sft" | "resume"
  "start_step":      int   — step at which this session started
  "end_step":        int   — step at which this session ended (or None if interrupted)
  "global_step":     int   — total steps ever taken on this model
  "tokens_session":  int   — tokens processed in this session
  "tokens_total":    int   — cumulative tokens across all sessions
  "start_ts":        str   — ISO timestamp
  "end_ts":          str | None
  "checkpoint_path": str   — checkpoint written at end of session
  "optimizer_hash":  str   — SHA-256 of optimizer state (truncated)
  "scheduler_state": dict  — scheduler state dict
  "resume_count":    int   — number of times this run was resumed
  "status":          str   — "running" | "completed" | "interrupted"
  "notes":           str
}
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

try:
    from config import CHECKPOINT_DIR
    _LEDGER_PATH = Path(CHECKPOINT_DIR) / "training_ledger.jsonl"
except (ImportError, AttributeError):
    _LEDGER_PATH = Path("var/checkpoints/training_ledger.jsonl")


def _ledger_path() -> Path:
    _LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    return _LEDGER_PATH


def _write(record: dict) -> None:
    """Append a record to the ledger (file-locked via O_APPEND)."""
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with open(_ledger_path(), "a", encoding="utf-8") as fh:
        fh.write(line)
        fh.flush()
        os.fsync(fh.fileno())


def _read_all() -> list[dict]:
    """Return all ledger records in order."""
    p = _ledger_path()
    if not p.exists():
        return []
    records = []
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def _hash_optimizer(opt_state: dict) -> str:
    """Produce a short deterministic hash of optimizer state for integrity tracking."""
    try:
        payload = json.dumps(
            {k: str(v)[:64] for k, v in opt_state.items() if k != "state"},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]
    except Exception:
        return "unknown"


def make_run_id() -> str:
    """Generate a unique run ID from current timestamp."""
    return "run_" + time.strftime("%Y%m%d_%H%M%S")


# ── Public API ────────────────────────────────────────────────────────

def open_run(
    run_id:         str,
    release_id:     str,
    release_hash:   str,
    run_type:       str,
    start_step:     int,
    global_step:    int,
    tokens_total:   int,
    resume_count:   int = 0,
    notes:          str = "",
) -> dict:
    """
    Open (record the start of) a training session.

    Call at the beginning of every training session, including resumes.
    Returns the run record dict.
    """
    record: dict[str, Any] = {
        "run_id":          run_id,
        "release_id":      release_id,
        "release_hash":    release_hash,
        "run_type":        run_type,
        "start_step":      start_step,
        "end_step":        None,
        "global_step":     global_step,
        "tokens_session":  0,
        "tokens_total":    tokens_total,
        "start_ts":        time.strftime("%Y-%m-%dT%H:%M:%S"),
        "end_ts":          None,
        "checkpoint_path": "",
        "optimizer_hash":  "",
        "scheduler_state": {},
        "resume_count":    resume_count,
        "status":          "running",
        "notes":           notes,
    }
    _write(record)
    return record


def close_run(
    run_id:           str,
    end_step:         int,
    global_step:      int,
    tokens_session:   int,
    tokens_total:     int,
    checkpoint_path:  str,
    optimizer_state:  dict,
    scheduler_state:  dict,
    status:           str = "completed",
) -> dict:
    """
    Close (record the end of) a training session.

    Call at the end of every training session, including interrupted ones.
    """
    record: dict[str, Any] = {
        "run_id":          run_id,
        "end_step":        end_step,
        "global_step":     global_step,
        "tokens_session":  tokens_session,
        "tokens_total":    tokens_total,
        "end_ts":          time.strftime("%Y-%m-%dT%H:%M:%S"),
        "checkpoint_path": checkpoint_path,
        "optimizer_hash":  _hash_optimizer(optimizer_state),
        "scheduler_state": scheduler_state,
        "status":          status,
        "_event":          "close",
    }
    _write(record)
    return record


def get_latest_run() -> dict | None:
    """Return the most recent completed or interrupted run record."""
    records = _read_all()
    # Find the last 'close' event
    closes = [r for r in records if r.get("_event") == "close"]
    return closes[-1] if closes else None


def get_run_history() -> list[dict]:
    """
    Return a consolidated run history — one entry per run_id,
    combining open and close events.
    """
    records = _read_all()
    runs: dict[str, dict] = {}

    for r in records:
        rid = r.get("run_id", "unknown")
        if rid not in runs:
            runs[rid] = dict(r)
        else:
            # Merge close event into run record
            runs[rid].update({k: v for k, v in r.items() if v is not None})

    return sorted(runs.values(), key=lambda x: x.get("start_ts", ""))


def get_total_tokens() -> int:
    """Return the total tokens consumed across all completed sessions."""
    history = get_run_history()
    completed = [r for r in history if r.get("status") in ("completed", "interrupted")]
    return max((r.get("tokens_total", 0) for r in completed), default=0)


def get_resume_state() -> dict:
    """
    Return the state needed to resume training:
      {
        "run_id":        last run ID
        "global_step":   last global step
        "tokens_total":  total tokens consumed
        "resume_count":  how many times training was resumed
        "checkpoint":    path to last checkpoint
        "release_id":    release used in last run
        "release_hash":  hash of release manifest at last run
      }
    """
    latest = get_latest_run()
    if not latest:
        return {
            "run_id": None,
            "global_step": 0,
            "tokens_total": 0,
            "resume_count": 0,
            "checkpoint": None,
            "release_id": None,
            "release_hash": None,
        }
    return {
        "run_id":       latest.get("run_id"),
        "global_step":  latest.get("global_step", 0),
        "tokens_total": latest.get("tokens_total", 0),
        "resume_count": latest.get("resume_count", 0),
        "checkpoint":   latest.get("checkpoint_path"),
        "release_id":   latest.get("release_id"),
        "release_hash": latest.get("release_hash"),
    }


def print_ledger_summary() -> None:
    """Print a human-readable ledger summary to stdout."""
    history = get_run_history()
    print(f"\n{'='*68}")
    print(f"COC v3 — Training Ledger")
    print(f"{'='*68}")
    if not history:
        print("  No training sessions recorded yet.")
    for r in history:
        print(f"\n  Run:       {r.get('run_id')}")
        print(f"  Type:      {r.get('run_type')}")
        print(f"  Release:   {r.get('release_id')}  [{r.get('release_hash','?')[:12]}...]")
        print(f"  Steps:     {r.get('start_step')} → {r.get('end_step','?')}")
        print(f"  Tokens:    session={r.get('tokens_session',0):,}  total={r.get('tokens_total',0):,}")
        print(f"  Status:    {r.get('status')}")
        print(f"  Start:     {r.get('start_ts')}")
        print(f"  End:       {r.get('end_ts','—')}")
        print(f"  Ckpt:      {r.get('checkpoint_path','—')}")
    total = get_total_tokens()
    print(f"\n  Cumulative tokens trained: {total:,}")
    print(f"{'='*68}\n")
