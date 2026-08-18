"""
CognitiveOC v3 — Human Review Queue
=====================================

Manages the human review queue for corpus paragraphs that:
  - Score between 0.45 and 0.70 (borderline quality)
  - Have risk_score between 0.20 and 0.50
  - Are synthetic data (all synthetic is mandatory review)

Queue file: warehouse/review_queue/pending.jsonl
Approved:   warehouse/review_queue/approved.jsonl
Rejected:   warehouse/review_queue/rejected.jsonl

CLI walkthrough:
  python main.py corpus review --queue
  python main.py corpus review --queue --source A-gutenberg-20260701
  python main.py corpus review --stats
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Iterator

from audit.logger import log_review

try:
    from config import CORPUS_WAREHOUSE_DIR
    _BASE = Path(CORPUS_WAREHOUSE_DIR) / "review_queue"
except (ImportError, AttributeError):
    _BASE = Path("var/corpus/review_queue")

_PENDING_PATH  = _BASE / "pending.jsonl"
_APPROVED_PATH = _BASE / "approved.jsonl"
_REJECTED_PATH = _BASE / "rejected.jsonl"


# ── Queue item schema ─────────────────────────────────────────────────
# {
#   "item_id":       str  — unique ID for this queue item
#   "source_id":     str  — which source it came from
#   "category":      str  — A-K
#   "is_synthetic":  bool — True if synthetic data
#   "paragraph":     str  — the text to review
#   "quality_score": float
#   "category_score":float
#   "risk_score":    float
#   "reason":        str  — why it's in queue (borderline_quality|risk|synthetic)
#   "ts_queued":     float — unix timestamp
#   "decision":      str | null — approve|reject
#   "decided_by":    str | null — operator who reviewed
#   "reject_reason": str | null — reason if rejected
#   "ts_decided":    float | null
# }


def _ensure_dirs() -> None:
    _BASE.mkdir(parents=True, exist_ok=True)


def _read_jsonl(path: Path) -> list[dict]:
    """Read all records from a JSONL file."""
    if not path.exists():
        return []
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def _append_jsonl(path: Path, record: dict) -> None:
    """Append a record to a JSONL file."""
    _ensure_dirs()
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _remove_from_pending(item_id: str) -> None:
    """Remove an item from the pending queue by rewriting without it."""
    records  = _read_jsonl(_PENDING_PATH)
    filtered = [r for r in records if r.get("item_id") != item_id]
    _PENDING_PATH.unlink(missing_ok=True)
    for r in filtered:
        _append_jsonl(_PENDING_PATH, r)


# ── Public API ────────────────────────────────────────────────────────

def add_to_queue(
    paragraph:      str,
    source_id:      str,
    category:       str,
    quality_score:  float,
    category_score: float,
    risk_score:     float,
    reason:         str,
    is_synthetic:   bool = False,
) -> str:
    """
    Add a paragraph to the human review queue.

    Args:
        paragraph:      The cleaned paragraph text.
        source_id:      Source identifier.
        category:       Category letter A-K.
        quality_score:  From scorer.py.
        category_score: From scorer.py.
        risk_score:     From scorer.py.
        reason:         Why queued: "borderline_quality" | "risk" | "synthetic"
        is_synthetic:   True if this came from COC's own generator.

    Returns:
        item_id: str
    """
    item_id = f"{source_id}_{int(time.time() * 1000)}"
    record  = {
        "item_id":        item_id,
        "source_id":      source_id,
        "category":       category,
        "is_synthetic":   is_synthetic,
        "paragraph":      paragraph,
        "quality_score":  quality_score,
        "category_score": category_score,
        "risk_score":     risk_score,
        "reason":         reason,
        "ts_queued":      time.time(),
        "decision":       None,
        "decided_by":     None,
        "reject_reason":  None,
        "ts_decided":     None,
    }
    _append_jsonl(_PENDING_PATH, record)
    return item_id


def pending_items(source_id: str | None = None) -> list[dict]:
    """Return all pending queue items, optionally filtered by source."""
    records = _read_jsonl(_PENDING_PATH)
    if source_id:
        records = [r for r in records if r.get("source_id") == source_id]
    return [r for r in records if r.get("decision") is None]


def next_for_review(source_id: str | None = None) -> dict | None:
    """Return the next unreviewed item from the queue."""
    items = pending_items(source_id)
    return items[0] if items else None


def approve(item_id: str, operator: str) -> bool:
    """
    Mark an item as approved.
    Moves it from pending → approved JSONL.

    Returns True if item was found and approved.
    """
    records = _read_jsonl(_PENDING_PATH)
    found   = None
    for r in records:
        if r.get("item_id") == item_id:
            found = r
            break
    if not found:
        return False

    found["decision"]    = "approve"
    found["decided_by"]  = operator
    found["ts_decided"]  = time.time()

    _remove_from_pending(item_id)
    _append_jsonl(_APPROVED_PATH, found)
    log_review(found["source_id"], item_id, "approve", operator)
    return True


def reject(item_id: str, operator: str, reason: str = "") -> bool:
    """
    Mark an item as rejected.
    Moves it from pending → rejected JSONL.

    Returns True if item was found and rejected.
    """
    records = _read_jsonl(_PENDING_PATH)
    found   = None
    for r in records:
        if r.get("item_id") == item_id:
            found = r
            break
    if not found:
        return False

    found["decision"]     = "reject"
    found["decided_by"]   = operator
    found["reject_reason"]= reason
    found["ts_decided"]   = time.time()

    _remove_from_pending(item_id)
    _append_jsonl(_REJECTED_PATH, found)
    log_review(found["source_id"], item_id, "reject", operator, reason=reason)
    return True


def queue_stats(source_id: str | None = None) -> dict:
    """Return counts of pending, approved, rejected items."""
    pending  = _read_jsonl(_PENDING_PATH)
    approved = _read_jsonl(_APPROVED_PATH)
    rejected = _read_jsonl(_REJECTED_PATH)

    if source_id:
        pending  = [r for r in pending  if r.get("source_id") == source_id]
        approved = [r for r in approved if r.get("source_id") == source_id]
        rejected = [r for r in rejected if r.get("source_id") == source_id]

    return {
        "pending":  len([r for r in pending if r.get("decision") is None]),
        "approved": len(approved),
        "rejected": len(rejected),
    }


def get_approved_for_source(source_id: str) -> list[str]:
    """
    Return the list of approved paragraph texts for a given source.
    Used by the release builder to include reviewed content.
    """
    records = _read_jsonl(_APPROVED_PATH)
    return [r["paragraph"] for r in records
            if r.get("source_id") == source_id
            and r.get("decision") == "approve"]


# ── Interactive CLI walkthrough ───────────────────────────────────────

def interactive_review(source_id: str | None = None,
                       operator:  str = "reviewer") -> None:
    """
    Run an interactive terminal-based review session.

    Controls:
      a  — approve this item
      r  — reject (will prompt for reason)
      s  — skip (leave in queue)
      q  — quit session
    """
    stats = queue_stats(source_id)
    print(f"\n{'='*60}")
    print(f"CognitiveOC v3 — Corpus Review Session")
    print(f"Operator : {operator}")
    if source_id:
        print(f"Source   : {source_id}")
    print(f"Pending  : {stats['pending']}")
    print(f"{'='*60}\n")

    reviewed = 0
    while True:
        item = next_for_review(source_id)
        if not item:
            print("✓ Queue empty. No more items to review.")
            break

        print(f"\n[{reviewed + 1}] Source  : {item['source_id']}")
        print(f"     Category: {item['category']}")
        print(f"     Synthetic: {item['is_synthetic']}")
        print(f"     Quality : {item['quality_score']:.3f}")
        print(f"     Risk    : {item['risk_score']:.3f}")
        print(f"     Reason  : {item['reason']}")
        print(f"\n{'-'*60}")
        print(item["paragraph"][:800])
        if len(item["paragraph"]) > 800:
            print("  ... [truncated]")
        print(f"{'-'*60}")
        print("  [a]pprove  [r]eject  [s]kip  [q]uit")

        try:
            choice = input("  > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nSession interrupted.")
            break

        if choice == "a":
            approve(item["item_id"], operator)
            print("  → Approved.")
            reviewed += 1
        elif choice == "r":
            try:
                reason = input("  Reject reason: ").strip()
            except (EOFError, KeyboardInterrupt):
                reason = "no reason given"
            reject(item["item_id"], operator, reason)
            print("  → Rejected.")
            reviewed += 1
        elif choice == "s":
            print("  → Skipped.")
        elif choice == "q":
            print(f"\nSession ended. Reviewed {reviewed} items.")
            break
        else:
            print("  Unknown input. Use a/r/s/q.")

    final = queue_stats(source_id)
    print(f"\nSession summary: reviewed {reviewed} | "
          f"pending {final['pending']} | approved {final['approved']} | "
          f"rejected {final['rejected']}")
