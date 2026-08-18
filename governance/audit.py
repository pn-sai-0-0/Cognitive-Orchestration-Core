"""
CognitiveOC v3 — Governance Audit Interface
=============================================

High-level audit interface used by the corpus CLI and desktop UI.
Wraps audit.reporter with governance-aware checks.

Used by:
  python main.py corpus audit-report [--source <id>] [--from <date>] [--to <date>]
  python main.py corpus check-compliance <source_id>
  python main.py corpus release-readiness <release_id>
"""

from __future__ import annotations

from pathlib import Path

from audit.reporter import (
    activity_report, check_compliance, events_by_source,
    format_report, load_all_events, release_readiness, source_summary,
)
from audit.logger import log_event


def full_activity_report(date_from: str | None = None,
                         date_to:   str | None = None,
                         verbose: bool = True) -> str:
    """Return a formatted activity report for the given date range."""
    events = load_all_events(date_from=date_from, date_to=date_to)
    report = activity_report(events, date_from=date_from, date_to=date_to)
    text   = format_report(report)
    if verbose:
        print(text)
    return text


def source_compliance_report(source_id: str, verbose: bool = True) -> dict:
    """Return compliance check for a single source."""
    events = load_all_events()
    result = check_compliance(source_id, events)
    if verbose:
        print(f"\nCompliance check: {source_id}")
        print(f"  Compliant      : {'YES' if result['compliant'] else 'NO'}")
        print(f"  Completed      : {', '.join(result['completed_stages']) or '-'}")
        print(f"  Missing        : {', '.join(result['missing_stages']) or '-'}")
        print(f"  Failed stages  : {', '.join(result['failed_stages']) or '-'}")
    return result


def all_sources_report(verbose: bool = True) -> list[dict]:
    """Return compliance summary for all registered sources."""
    events = load_all_events()
    summary = source_summary(events)
    if verbose:
        print(format_report(summary))
    return summary


def release_readiness_report(release_id: str, source_ids: list[str],
                              verbose: bool = True) -> dict:
    """Check whether all sources planned for a release are pipeline-complete."""
    events = load_all_events()
    result = release_readiness(release_id, source_ids, events)
    if verbose:
        print(format_report(result))
    return result
