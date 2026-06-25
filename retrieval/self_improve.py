"""
CognitiveOC v3 — Self-Improving Retrieval
==========================================

Feedback-aware retrieval improvement subsystem.
Logs misses, captures relevance feedback, mines hard examples,
and surfaces improvement signals — WITHOUT auto-retraining.

All captured data goes to:
  var/cache/retrieval_misses.jsonl     — queries with no results
  var/cache/retrieval_feedback.jsonl   — hit/miss feedback per chunk
  var/cache/retrieval_analytics.json   — aggregate analytics

Hard examples (low-confidence retrievals) can be exported via
the DatasetGenerator for human review → future reranker training.

File: retrieval/self_improve.py
Used by: retrieval/rag.py (HybridRetriever), ui/app.py, dataset/generator.py
"""

from __future__ import annotations

import json
import threading
import time
from collections import defaultdict
from pathlib import Path

try:
    from config import CACHE_DIR, ensure_dirs
except ImportError:
    CACHE_DIR = Path("var/cache")
    def ensure_dirs(): pass

_MISS_PATH      = Path(str(CACHE_DIR)) / "retrieval_misses.jsonl"
_FEEDBACK_PATH  = Path(str(CACHE_DIR)) / "retrieval_feedback.jsonl"
_ANALYTICS_PATH = Path(str(CACHE_DIR)) / "retrieval_analytics.json"
_LOCK           = threading.Lock()


def _append(path: Path, record: dict):
    """Append one JSONL record (thread-safe)."""
    try:
        ensure_dirs()
        path.parent.mkdir(parents=True, exist_ok=True)
        with _LOCK:
            with open(str(path), "a") as f:
                f.write(json.dumps(record) + "\n")
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════
# Miss logging
# ═══════════════════════════════════════════════════════════════════

def log_miss(query: str, reason: str = "no_results"):
    """Log a retrieval miss (query returned no useful results)."""
    _append(_MISS_PATH, {
        "ts":     time.time(),
        "query":  query[:300],
        "reason": reason,
    })
    _update_analytics("misses", 1)


def get_misses(limit: int = 100) -> list[dict]:
    """Return recent retrieval misses for inspection."""
    if not _MISS_PATH.exists():
        return []
    lines = _MISS_PATH.read_text().strip().splitlines()
    results = []
    for line in reversed(lines[-limit:]):
        try:
            results.append(json.loads(line))
        except Exception:
            pass
    return results


# ═══════════════════════════════════════════════════════════════════
# Hit / feedback recording
# ═══════════════════════════════════════════════════════════════════

def record_hit(query: str, chunk_text: str,
               chunk_source: str, useful: bool):
    """Record whether a retrieved chunk was useful for a query."""
    _append(_FEEDBACK_PATH, {
        "ts":     time.time(),
        "query":  query[:300],
        "chunk":  chunk_text[:300],
        "source": chunk_source[:100],
        "useful": int(useful),
    })
    _update_analytics("useful" if useful else "not_useful", 1)


def get_feedback(limit: int = 200) -> list[dict]:
    """Return recent retrieval feedback records."""
    if not _FEEDBACK_PATH.exists():
        return []
    lines = _FEEDBACK_PATH.read_text().strip().splitlines()
    results = []
    for line in reversed(lines[-limit:]):
        try:
            results.append(json.loads(line))
        except Exception:
            pass
    return results


# ═══════════════════════════════════════════════════════════════════
# Analytics
# ═══════════════════════════════════════════════════════════════════

def _update_analytics(key: str, delta: int):
    """Increment an analytics counter (best-effort)."""
    try:
        ensure_dirs()
        _ANALYTICS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _LOCK:
            if _ANALYTICS_PATH.exists():
                data = json.loads(_ANALYTICS_PATH.read_text())
            else:
                data = {"misses": 0, "useful": 0, "not_useful": 0,
                        "hard_examples": 0, "updated": 0}
            data[key]      = data.get(key, 0) + delta
            data["updated"]= time.time()
            _ANALYTICS_PATH.write_text(json.dumps(data, indent=2))
    except Exception:
        pass


def stats() -> dict:
    """Return current retrieval improvement analytics."""
    base = {"misses": 0, "useful": 0, "not_useful": 0,
            "hard_examples": 0, "updated": 0}
    if _ANALYTICS_PATH.exists():
        try:
            base.update(json.loads(_ANALYTICS_PATH.read_text()))
        except Exception:
            pass
    total_fb = base["useful"] + base["not_useful"]
    base["hit_rate"]  = round(base["useful"] / max(total_fb,1), 3)
    base["miss_count"]= base["misses"]
    return base


# ═══════════════════════════════════════════════════════════════════
# Hard example mining
# ═══════════════════════════════════════════════════════════════════

def mine_hard_examples(min_not_useful: int = 3,
                       limit: int = 50) -> list[dict]:
    """Find queries where retrieved chunks were consistently not useful.

    A query is a hard example if ≥ min_not_useful chunks were marked
    not useful. These are candidates for query rewriting / index updates.

    Returns list of hard example dicts for human review.
    NO automatic retraining occurs.
    """
    feedback = get_feedback(limit=2000)
    counts: dict[str, dict] = defaultdict(lambda: {"useful":0,"not_useful":0,"chunks":[]})

    for fb in feedback:
        q = fb.get("query","")
        if not q:
            continue
        label = "useful" if fb.get("useful",0) else "not_useful"
        counts[q][label]    += 1
        counts[q]["chunks"].append(fb.get("chunk","")[:80])

    hard = [
        {
            "query":       q,
            "useful":      v["useful"],
            "not_useful":  v["not_useful"],
            "sample_chunks": v["chunks"][:3],
            "type":        "hard_retrieval_example",
        }
        for q, v in counts.items()
        if v["not_useful"] >= min_not_useful
    ]
    hard.sort(key=lambda x: -x["not_useful"])
    _update_analytics("hard_examples", len(hard[:limit]))
    return hard[:limit]


# ═══════════════════════════════════════════════════════════════════
# Query rewrite candidates
# ═══════════════════════════════════════════════════════════════════

def export_query_rewrites() -> dict:
    """Export miss queries as rewrite training candidates.

    Returns manifest for human review. NOT auto-fed into training.
    """
    misses = get_misses(limit=500)
    if not misses:
        return {"count": 0, "note": "No misses recorded yet"}

    path = Path(str(CACHE_DIR)) / "query_rewrite_candidates.jsonl"
    written = 0
    with open(str(path), "w") as f:
        for m in misses:
            f.write(json.dumps({
                "query":  m.get("query",""),
                "reason": m.get("reason",""),
                "ts":     m.get("ts",0),
                "type":   "query_rewrite_candidate",
            }) + "\n")
            written += 1

    return {
        "path":   str(path),
        "count":  written,
        "note":   "Human review required before use in query rewriter training.",
        "reviewed": False,
    }


def export_reranker_examples() -> dict:
    """Export retrieval feedback as reranker training examples.

    Returns manifest for human review. NOT auto-fed into training.
    """
    feedback = get_feedback(limit=2000)
    if not feedback:
        return {"count": 0, "note": "No feedback recorded yet"}

    path = Path(str(CACHE_DIR)) / "reranker_examples.jsonl"
    written = 0
    with open(str(path), "w") as f:
        for fb in feedback:
            if not fb.get("query") or not fb.get("chunk"):
                continue
            f.write(json.dumps({
                "query":  fb["query"],
                "passage":fb["chunk"],
                "source": fb.get("source",""),
                "label":  fb.get("useful",0),
                "type":   "reranker_example",
            }) + "\n")
            written += 1

    return {
        "path":     str(path),
        "count":    written,
        "note":     "Human review required before use in reranker training.",
        "reviewed": False,
    }
