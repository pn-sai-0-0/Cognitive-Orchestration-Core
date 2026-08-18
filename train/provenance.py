"""
CognitiveOC v3 — Training Provenance
======================================

Produces and maintains a permanent provenance record that maps every
model checkpoint to the exact corpus release and training sessions
that produced it.

Provenance answers after the fact:
  - Which release trained this model?
  - Which source families contributed, in what proportions?
  - What training steps and token counts were used?
  - What checkpoints exist and how were they produced?
  - What evaluations were run and what scores were achieved?
  - How was training resumed, and how many sessions occurred?

Provenance file: var/checkpoints/provenance.json  (append-per-run, not overwritten)

Run:
  python main.py corpus provenance-report
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

try:
    from config import CHECKPOINT_DIR, CORPUS_WAREHOUSE_DIR
    _CKPT_DIR = Path(CHECKPOINT_DIR)
    _WHOUSE   = Path(CORPUS_WAREHOUSE_DIR)
except (ImportError, AttributeError):
    _CKPT_DIR = Path("var/checkpoints")
    _WHOUSE   = Path("var/corpus_warehouse")

_PROV_PATH = _CKPT_DIR / "provenance.json"


def _sha256_file(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_provenance() -> dict:
    if not _PROV_PATH.exists():
        return {"coc_version": "v3", "runs": []}
    with open(_PROV_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def _save_provenance(data: dict) -> None:
    _CKPT_DIR.mkdir(parents=True, exist_ok=True)
    data["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    tmp = _PROV_PATH.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(_PROV_PATH)


# ── Provenance record builder ─────────────────────────────────────────

def record_run(
    run_id:           str,
    release_id:       str,
    run_type:         str,        # "pretrain" | "sft" | "resume"
    start_step:       int,
    end_step:         int,
    global_step:      int,
    tokens_session:   int,
    tokens_total:     int,
    train_loss:       float,
    val_loss:         float,
    perplexity:       float,
    checkpoint_path:  str,
    duration_s:       float,
    tokens_per_sec:   float,
    resume_count:     int,
    eval_results:     dict | None = None,
    notes:            str = "",
) -> dict:
    """
    Record a completed training session in the provenance file.

    Also loads and embeds release manifest summary for full traceability.
    Returns the provenance entry dict.
    """
    # Load release manifest for source breakdown
    manifest_path = _WHOUSE / "releases" / release_id / "manifest.json"
    manifest      = {}
    manifest_hash = ""
    source_breakdown: list[dict] = []

    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as fh:
            manifest = json.load(fh)
        manifest_hash = _sha256_file(manifest_path)
        source_breakdown = [
            {
                "source_id":   s.get("source_id"),
                "category":    s.get("category"),
                "tokens":      s.get("token_contribution", 0),
                "licence":     s.get("licence"),
            }
            for s in manifest.get("sources", [])
        ]

    entry: dict[str, Any] = {
        "run_id":          run_id,
        "release_id":      release_id,
        "release_hash":    manifest_hash,
        "run_type":        run_type,
        "timestamp":       time.strftime("%Y-%m-%dT%H:%M:%S"),

        # Step / token accounting
        "start_step":      start_step,
        "end_step":        end_step,
        "global_step":     global_step,
        "tokens_session":  tokens_session,
        "tokens_total":    tokens_total,
        "resume_count":    resume_count,

        # Loss and quality
        "train_loss":      round(train_loss, 6),
        "val_loss":        round(val_loss, 6),
        "perplexity":      round(perplexity, 4),

        # Performance
        "duration_s":      round(duration_s, 1),
        "tokens_per_sec":  round(tokens_per_sec, 1),
        "duration_hours":  round(duration_s / 3600, 2),

        # Checkpoint
        "checkpoint_path": checkpoint_path,
        "checkpoint_hash": _sha256_file(Path(checkpoint_path))
                           if checkpoint_path and Path(checkpoint_path).exists() else "",

        # Release source breakdown
        "release_summary": {
            "release_id":      release_id,
            "total_tokens":    manifest.get("total_tokens_estimate", 0),
            "train_tokens":    manifest.get("train_tokens", 0),
            "categories":      manifest.get("categories_included", []),
            "signed_by":       manifest.get("signed_by", "unknown"),
            "release_date":    manifest.get("release_date", "unknown"),
        },
        "source_breakdown": source_breakdown,

        # Evaluation
        "eval_results":    eval_results or {},
        "notes":           notes,
    }

    prov = _load_provenance()
    prov["runs"].append(entry)
    _save_provenance(prov)
    return entry


# ── Report generation ─────────────────────────────────────────────────

def generate_report(verbose: bool = True) -> str:
    """
    Generate a human-readable provenance report.

    Returns the report as a string and optionally prints it.
    """
    prov  = _load_provenance()
    runs  = prov.get("runs", [])
    lines = []

    lines.append("=" * 72)
    lines.append("CognitiveOC v3 — Training Provenance Report")
    lines.append(f"Generated: {time.strftime('%Y-%m-%dT%H:%M:%S')}")
    lines.append("=" * 72)

    if not runs:
        lines.append("\nNo training sessions recorded yet.")
    else:
        total_tokens  = max((r.get("tokens_total", 0) for r in runs), default=0)
        total_hours   = sum(r.get("duration_hours", 0) for r in runs)
        lines.append(f"\nTotal sessions  : {len(runs)}")
        lines.append(f"Total tokens    : {total_tokens:,}")
        lines.append(f"Total hours     : {total_hours:.1f} h")

        for i, r in enumerate(runs, 1):
            lines.append(f"\n{'─'*72}")
            lines.append(f"Session {i}: {r['run_id']}")
            lines.append(f"  Type         : {r['run_type']}")
            lines.append(f"  Release      : {r['release_id']}  [{r.get('release_hash','')[:16]}...]")
            lines.append(f"  Steps        : {r['start_step']:,} → {r['end_step']:,}  "
                         f"(global: {r['global_step']:,})")
            lines.append(f"  Tokens sess  : {r['tokens_session']:,}")
            lines.append(f"  Tokens total : {r['tokens_total']:,}")
            lines.append(f"  Train loss   : {r['train_loss']}")
            lines.append(f"  Val loss     : {r['val_loss']}")
            lines.append(f"  Perplexity   : {r['perplexity']}")
            lines.append(f"  Duration     : {r['duration_hours']:.2f} h  "
                         f"({r['tokens_per_sec']:.0f} tok/s)")
            lines.append(f"  Checkpoint   : {r['checkpoint_path']}")
            if r.get("checkpoint_hash"):
                lines.append(f"  Ckpt SHA-256 : {r['checkpoint_hash'][:24]}...")
            rs = r.get("release_summary", {})
            lines.append(f"  Rel. tokens  : {rs.get('train_tokens', 0):,}  "
                         f"(categories: {', '.join(rs.get('categories', []))})")
            lines.append(f"  Signed by    : {rs.get('signed_by', '?')}  "
                         f"on {rs.get('release_date', '?')}")
            if r.get("eval_results"):
                lines.append(f"  Eval results :")
                for k, v in r["eval_results"].items():
                    lines.append(f"    {k:<24} {v}")
            if r.get("notes"):
                lines.append(f"  Notes        : {r['notes']}")

    lines.append(f"\n{'='*72}\n")
    report = "\n".join(lines)
    if verbose:
        print(report)
    return report


def get_checkpoint_lineage(checkpoint_path: str) -> list[dict]:
    """
    Return all provenance entries that produced or contributed to
    the given checkpoint.
    """
    prov = _load_provenance()
    return [
        r for r in prov.get("runs", [])
        if r.get("checkpoint_path") == checkpoint_path
    ]
