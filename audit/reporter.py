"""
CognitiveOC v3 — Audit Reporter
=================================

Reads the append-only JSONL audit logs and produces:
  - Per-source compliance summaries
  - Date-range activity reports
  - Stage coverage checks (has every required stage been completed?)
  - Release readiness reports

Used by:
  python main.py corpus audit-report [--source <id>] [--from <date>] [--to <date>]
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

try:
    from config import CORPUS_AUDIT_LOG_DIR, CORPUS_APPROVAL_LOG
    _APPROVAL_LOG  = Path(CORPUS_APPROVAL_LOG)
    _AUDIT_LOG_DIR = Path(CORPUS_AUDIT_LOG_DIR)
except ImportError:
    _APPROVAL_LOG  = Path("governance/approval_log.jsonl")
    _AUDIT_LOG_DIR = Path("var/logs/corpus_audit")

# Stages that MUST be completed in order for a source to be release-eligible
REQUIRED_STAGES = ["acquire", "validate", "normalize", "clean", "dedup",
                   "score", "approve"]


def _iter_events(path: Path) -> Iterator[dict]:
    """Yield parsed JSON events from a JSONL log file."""
    if not path.exists():
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    pass


def load_all_events(
    date_from: str | None = None,
    date_to:   str | None = None,
) -> list[dict]:
    """
    Load all audit events, optionally filtered by date range.

    Args:
        date_from: ISO date string "YYYY-MM-DD" (inclusive lower bound).
        date_to:   ISO date string "YYYY-MM-DD" (inclusive upper bound).

    Returns:
        Sorted list of event dicts (ascending by ts_unix).
    """
    events: list[dict] = []

    # Load from approval log (primary — always present)
    for ev in _iter_events(_approval_LOG()):
        events.append(ev)

    # Load additional events from daily shards not yet in approval log
    # (daily shards may include events from in-progress sessions)
    if _AUDIT_LOG_DIR.exists():
        for shard in sorted(_AUDIT_LOG_DIR.glob("audit_*.jsonl")):
            for ev in _iter_events(shard):
                # Avoid duplicates already in approval log
                if ev not in events:
                    events.append(ev)

    # Date filter
    if date_from or date_to:
        df = date_from or "0000-01-01"
        dt = date_to   or "9999-12-31"
        events = [
            e for e in events
            if df <= e.get("ts", "")[:10] <= dt
        ]

    events.sort(key=lambda e: e.get("ts_unix", 0))
    return events


def _approval_LOG() -> Path:
    _APPROVAL_LOG.parent.mkdir(parents=True, exist_ok=True)
    return _APPROVAL_LOG


def events_by_source(events: list[dict]) -> dict[str, list[dict]]:
    """Group events by source_id."""
    by_source: dict[str, list[dict]] = defaultdict(list)
    for ev in events:
        by_source[ev.get("source_id", "unknown")].append(ev)
    return dict(by_source)


def check_compliance(source_id: str, events: list[dict]) -> dict:
    """
    Check whether a source has completed all required pipeline stages.

    Returns:
        {
          "source_id": str,
          "compliant": bool,
          "completed_stages": list[str],
          "missing_stages": list[str],
          "failed_stages": list[str],
          "events": int,
        }
    """
    src_events = [e for e in events if e.get("source_id") == source_id]
    completed = {e["stage"] for e in src_events if e.get("result") == "ok"}
    failed    = {e["stage"] for e in src_events if e.get("result") == "fail"}
    missing   = [s for s in REQUIRED_STAGES if s not in completed]

    return {
        "source_id":        source_id,
        "compliant":        len(missing) == 0 and "validate" in completed,
        "completed_stages": sorted(completed),
        "missing_stages":   missing,
        "failed_stages":    sorted(failed),
        "events":           len(src_events),
    }


def source_summary(events: list[dict]) -> list[dict]:
    """
    Produce a per-source compliance summary for all sources in the log.

    Returns:
        List of compliance dicts, sorted by source_id.
    """
    all_sources = {e.get("source_id") for e in events
                   if e.get("source_id") not in (None, "release")}
    return sorted(
        [check_compliance(sid, events) for sid in all_sources],
        key=lambda x: x["source_id"],
    )


def activity_report(
    events:    list[dict],
    date_from: str | None = None,
    date_to:   str | None = None,
) -> dict:
    """
    Generate a date-ranged activity report.

    Returns dict with:
      total_events, events_by_stage, events_by_result,
      sources_touched, operators, date_range
    """
    by_stage:  dict[str, int] = defaultdict(int)
    by_result: dict[str, int] = defaultdict(int)
    sources:   set[str] = set()
    operators: set[str] = set()

    for ev in events:
        by_stage[ev.get("stage", "unknown")] += 1
        by_result[ev.get("result", "unknown")] += 1
        sources.add(ev.get("source_id", "unknown"))
        operators.add(ev.get("operator", "unknown"))

    return {
        "date_range":        {"from": date_from, "to": date_to},
        "total_events":      len(events),
        "events_by_stage":   dict(by_stage),
        "events_by_result":  dict(by_result),
        "sources_touched":   sorted(sources),
        "operators":         sorted(operators),
    }


def release_readiness(release_id: str, source_ids: list[str],
                      events: list[dict]) -> dict:
    """
    Check whether all sources in a planned release are pipeline-complete.

    Args:
        release_id:  e.g. "v1"
        source_ids:  List of source_ids that should be in this release.
        events:      All audit events.

    Returns:
        {
          "release_id": str,
          "ready": bool,
          "total_sources": int,
          "compliant_sources": int,
          "non_compliant": list[str],
          "details": list[compliance_dict]
        }
    """
    details = [check_compliance(sid, events) for sid in source_ids]
    non_compliant = [d["source_id"] for d in details if not d["compliant"]]

    return {
        "release_id":        release_id,
        "ready":             len(non_compliant) == 0,
        "total_sources":     len(source_ids),
        "compliant_sources": len(source_ids) - len(non_compliant),
        "non_compliant":     non_compliant,
        "details":           details,
    }


def format_report(report: dict) -> str:
    """Format a report dict as a readable plaintext string."""
    lines = []
    lines.append("=" * 72)
    lines.append("CognitiveOC v3 — Corpus Audit Report")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append("=" * 72)
    lines.append("")

    if "total_events" in report:
        dr = report.get("date_range", {})
        if dr.get("from") or dr.get("to"):
            lines.append(f"Date range: {dr.get('from','all')} → {dr.get('to','now')}")
        lines.append(f"Total events:    {report['total_events']}")
        lines.append(f"Sources touched: {len(report.get('sources_touched', []))}")
        lines.append(f"Operators:       {', '.join(report.get('operators', []))}")
        lines.append("")
        lines.append("Events by stage:")
        for stage, count in sorted(report.get("events_by_stage", {}).items()):
            lines.append(f"  {stage:<18} {count:>6}")
        lines.append("")
        lines.append("Events by result:")
        for result, count in sorted(report.get("events_by_result", {}).items()):
            lines.append(f"  {result:<18} {count:>6}")

    elif "compliant_sources" in report:
        # Release readiness report
        lines.append(f"Release:      {report['release_id']}")
        lines.append(f"Ready:        {'YES' if report['ready'] else 'NO'}")
        lines.append(f"Sources:      {report['compliant_sources']} / {report['total_sources']} compliant")
        if report["non_compliant"]:
            lines.append("")
            lines.append("Non-compliant sources:")
            for sid in report["non_compliant"]:
                lines.append(f"  - {sid}")

    elif isinstance(report, list):
        # Source summary list
        lines.append(f"{'Source ID':<40} {'Compliant':<10} {'Missing stages'}")
        lines.append("-" * 72)
        for rec in report:
            c = "YES" if rec["compliant"] else "NO"
            m = ", ".join(rec["missing_stages"]) or "-"
            lines.append(f"  {rec['source_id']:<38} {c:<10} {m}")

    lines.append("")
    lines.append("=" * 72)
    return "\n".join(lines)
