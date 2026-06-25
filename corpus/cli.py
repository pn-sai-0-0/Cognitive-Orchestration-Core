"""
CognitiveOC v3 — Corpus CLI
==============================

Routes all 'python main.py corpus <subcmd>' commands.

Available subcommands:
  register-source   Register a new source in the governance registry
  validate-source   Mark a source as validated (or rejected)
  run-pipeline      Run pipeline stages for a source
  review            Interactive human review session
  build-release     Assemble a training release
  verify-release    Verify a built release (checksums + leakage)
  warehouse-stats   Print warehouse statistics
  audit-report      Generate an audit report

Usage examples:
  python main.py corpus register-source A-gutenberg-20260701 \
    --category A --name "Project Gutenberg English" \
    --url https://gutenberg.org \
    --licence "Public Domain" --licence-id pd --licence-risk 0.0 \
    --raw-path /mnt/corpus/raw/books/gutenberg/ \
    --raw-sha256 abc123 --operator mpssp

  python main.py corpus validate-source A-gutenberg-20260701 --operator mpssp

  python main.py corpus run-pipeline A-gutenberg-20260701 \
    --stages clean,score --verbose

  python main.py corpus review --source A-gutenberg-20260701 --operator mpssp

  python main.py corpus build-release v1 --categories A,B,C,D,E,F,G,H,I,J,K

  python main.py corpus verify-release v1

  python main.py corpus warehouse-stats

  python main.py corpus audit-report
  python main.py corpus audit-report --source A-gutenberg-20260701
  python main.py corpus audit-report --from 2026-07-01 --to 2026-07-31
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# ── Subcommand handlers ───────────────────────────────────────────────

def cmd_register_source(args: argparse.Namespace) -> None:
    """Register a new source in the governance registry."""
    from corpus.source_registry import register_source
    from audit.logger import log_acquire

    try:
        record = register_source(
            source_id     = args.source_id,
            name          = args.name or args.source_id,
            url           = args.url or "",
            category      = args.category,
            licence       = args.licence,
            licence_id    = args.licence_id or "unknown",
            licence_risk  = float(args.licence_risk),
            raw_path      = args.raw_path or "",
            raw_sha256    = args.raw_sha256 or "",
            acquired_by   = args.operator or "system",
            notes         = args.notes or "",
        )
        log_acquire(
            source_id = args.source_id,
            path      = args.raw_path or "",
            sha256    = args.raw_sha256 or "",
            operator  = args.operator or "system",
        )
        print(f"\n✓ Source registered: {record['source_id']}")
        print(f"  Category   : {record['category']}")
        print(f"  Licence    : {record['licence']} (risk {record['licence_risk']})")
        print(f"  Status     : {record['status']}")
        print(f"\n  Next step: python main.py corpus validate-source {args.source_id}"
              f" --operator {args.operator or 'your_name'}\n")
    except ValueError as e:
        print(f"\n✗ Registration failed: {e}\n", file=sys.stderr)
        sys.exit(1)


def cmd_validate_source(args: argparse.Namespace) -> None:
    """Validate (or reject) a registered source."""
    from corpus.source_registry import validate_source, get_source

    record = get_source(args.source_id)
    if not record:
        print(f"\n✗ Source not found: {args.source_id}\n", file=sys.stderr)
        sys.exit(1)

    passed   = not args.reject
    reason   = args.reason or ("Validated by operator" if passed else "Rejected by operator")
    operator = args.operator or "system"

    updated = validate_source(args.source_id, passed, operator, reason)
    status  = "✓ VALIDATED" if passed else "✗ REJECTED"
    print(f"\n{status}: {args.source_id}")
    print(f"  Status: {updated['status']}")
    print(f"  By    : {operator}")
    print(f"  Reason: {reason}\n")


def cmd_run_pipeline(args: argparse.Namespace) -> None:
    """
    Run pipeline stages for a registered, validated source.

    This is the orchestration entry point for:
      clean → dedup → score → (review routing)

    Actual heavy processing is delegated to corpus/cleaner.py,
    corpus/dedup.py, and corpus/scorer.py.
    """
    from corpus.source_registry import get_source
    from audit.logger import log_stage

    stages_requested = [s.strip().lower() for s in (args.stages or "clean,score").split(",")]
    verbose          = getattr(args, "verbose", False)
    operator         = args.operator or "system"

    record = get_source(args.source_id)
    if not record:
        print(f"\n✗ Source not found: {args.source_id}\n", file=sys.stderr)
        sys.exit(1)

    if record["status"] == "pending":
        print(f"\n✗ Source '{args.source_id}' has not been validated yet. "
              f"Run validate-source first.\n", file=sys.stderr)
        sys.exit(1)

    if record["status"] == "rejected":
        print(f"\n✗ Source '{args.source_id}' was rejected. Cannot run pipeline.\n",
              file=sys.stderr)
        sys.exit(1)

    category  = record["category"]
    raw_path  = Path(record["raw_path"])

    print(f"\n[pipeline] Source  : {args.source_id}")
    print(f"[pipeline] Category: {category}")
    print(f"[pipeline] Stages  : {', '.join(stages_requested)}")

    # ── CLEAN ─────────────────────────────────────────────────────────
    if "clean" in stages_requested:
        from corpus.cleaner import clean_for_category
        from corpus.warehouse import cleaned_dir
        from data.pipeline import split_paragraphs

        print(f"\n[clean] Reading from: {raw_path}")
        if not raw_path.exists():
            print(f"  ✗ Raw path does not exist: {raw_path}", file=sys.stderr)
        else:
            # Gather all .txt files recursively under raw_path
            txt_files = list(raw_path.rglob("*.txt")) if raw_path.is_dir() else [raw_path]
            all_paras: list[str] = []
            for f in txt_files:
                text = f.read_text(encoding="utf-8", errors="replace")
                paras = clean_for_category(text, category)
                all_paras.extend(paras)

            out_dir  = cleaned_dir(category)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / f"{args.source_id}.txt"
            out_file.write_text("\n\n".join(all_paras), encoding="utf-8")

            log_stage("clean", args.source_id, "clean_complete",
                      len(txt_files), len(all_paras), operator)
            print(f"[clean] {len(all_paras):,} paragraphs written to {out_file}")

    # ── DEDUP (within-source) ─────────────────────────────────────────
    if "dedup" in stages_requested:
        from corpus.warehouse import cleaned_dir, deduped_dir
        from data.pipeline import dedup

        cfile = cleaned_dir(category) / f"{args.source_id}.txt"
        if not cfile.exists():
            print(f"[dedup] ✗ Cleaned file not found. Run 'clean' stage first.")
        else:
            text = cfile.read_text(encoding="utf-8")
            paras = [p.strip() for p in text.split("\n\n") if p.strip()]
            n_in  = len(paras)

            deduped = dedup(paras)  # uses pipeline.py within-source dedup
            n_out   = len(deduped)

            out_dir  = deduped_dir(category)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / f"{args.source_id}.txt"
            out_file.write_text("\n\n".join(deduped), encoding="utf-8")

            log_stage("dedup", args.source_id, "dedup_complete", n_in, n_out, operator)
            print(f"[dedup] {n_in:,} → {n_out:,} paragraphs "
                  f"({n_in - n_out:,} removed, {n_out/n_in*100:.1f}% retained)")

    # ── SCORE + ROUTE ─────────────────────────────────────────────────
    if "score" in stages_requested:
        from corpus.scorer import batch_score, score_summary
        from corpus.reviewer import add_to_queue
        from corpus.warehouse import deduped_dir, approved_dir
        from corpus.source_registry import approve_source, update_source

        is_synthetic = (category == "H")

        # Prefer deduped file; fall back to cleaned
        src_file = (deduped_dir(category) / f"{args.source_id}.txt")
        if not src_file.exists():
            src_file = cleaned_dir(category) / f"{args.source_id}.txt"
        if not src_file.exists():
            print(f"[score] ✗ No cleaned/deduped file found for {args.source_id}. "
                  f"Run 'clean' stage first.")
        else:
            text  = src_file.read_text(encoding="utf-8")
            paras = [p.strip() for p in text.split("\n\n") if p.strip()]

            print(f"[score] Scoring {len(paras):,} paragraphs...")
            scores = batch_score(paras, args.source_id, category,
                                 is_synthetic=is_synthetic, verbose=verbose)

            approved_paras: list[str] = []
            n_auto_approve = n_human_queue = n_auto_reject = 0

            for para, score in zip(paras, scores):
                if score.decision == "auto_approve":
                    approved_paras.append(para)
                    n_auto_approve += 1
                elif score.decision == "human_review":
                    reason = ("synthetic" if is_synthetic
                              else ("borderline_quality" if score.quality_score < 0.70
                                    else "risk"))
                    add_to_queue(
                        paragraph      = para,
                        source_id      = args.source_id,
                        category       = category,
                        quality_score  = score.quality_score,
                        category_score = score.category_score,
                        risk_score     = score.risk_score,
                        reason         = reason,
                        is_synthetic   = is_synthetic,
                    )
                    n_human_queue += 1
                else:
                    n_auto_reject += 1

            # Write auto-approved paragraphs to approved dir
            out_dir = approved_dir(category)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / f"{args.source_id}.txt"
            out_file.write_text("\n\n".join(approved_paras), encoding="utf-8")

            summary = score_summary(scores)
            log_stage("score", args.source_id, "score_complete",
                      len(paras), n_auto_approve, operator)

            print(f"[score] Results for {args.source_id}:")
            print(f"  Total scored   : {len(paras):,}")
            print(f"  Auto-approved  : {n_auto_approve:,}")
            print(f"  Human queue    : {n_human_queue:,}  ← run: corpus review")
            print(f"  Auto-rejected  : {n_auto_reject:,}")
            print(f"  Avg quality    : {summary.get('avg_quality', 0):.3f}")
            print(f"  Avg risk       : {summary.get('avg_risk', 0):.3f}")

            # Update registry with preliminary approved counts
            update_source(args.source_id, {
                "status":              "scoring",
                "approved_paragraphs": n_auto_approve,
            })

            if n_human_queue > 0:
                print(f"\n  → {n_human_queue:,} items pending human review.")
                print(f"    Run: python main.py corpus review "
                      f"--source {args.source_id} --operator {operator}")
            else:
                # No review needed — auto-approve the source
                est_tokens = int(len("\n\n".join(approved_paras)) / 3.8)
                approve_source(args.source_id, operator, n_auto_approve, est_tokens)
                print(f"\n  ✓ Source '{args.source_id}' auto-approved "
                      f"({n_auto_approve:,} paragraphs, ~{est_tokens/1e6:.0f}M tokens)")


def cmd_review(args: argparse.Namespace) -> None:
    """Run the interactive human review session."""
    from corpus.reviewer import interactive_review, queue_stats

    source_id = getattr(args, "source", None)
    operator  = getattr(args, "operator", None) or "reviewer"

    if getattr(args, "stats", False):
        stats = queue_stats(source_id)
        print(f"\nReview queue stats:")
        if source_id:
            print(f"  Source  : {source_id}")
        print(f"  Pending  : {stats['pending']}")
        print(f"  Approved : {stats['approved']}")
        print(f"  Rejected : {stats['rejected']}\n")
        return

    interactive_review(source_id=source_id, operator=operator)


def cmd_build_release(args: argparse.Namespace) -> None:
    """Assemble a training release."""
    from corpus.release_builder import ReleaseBuilder

    version    = args.version
    categories = [c.strip() for c in (args.categories or "A,B,C,D,E,F,G,H,I,J,K").split(",")]
    budget     = int(args.token_budget or 30_000_000_000)
    dry_run    = getattr(args, "dry_run", False)
    verbose    = getattr(args, "verbose", True)

    rb = ReleaseBuilder(version)
    rb.build(
        categories   = categories,
        token_budget = budget,
        dry_run      = dry_run,
        verbose      = verbose,
    )


def cmd_verify_release(args: argparse.Namespace) -> None:
    """Verify a built release."""
    from corpus.release_builder import ReleaseBuilder

    rb = ReleaseBuilder(args.version)
    result = rb.verify()
    if not result["verified"]:
        sys.exit(1)


def cmd_sign_release(args: argparse.Namespace) -> None:
    """Sign (lock) a verified release."""
    from corpus.release_builder import ReleaseBuilder

    operator = args.operator or "system"
    rb       = ReleaseBuilder(args.version)
    ok       = rb.sign(operator=operator)
    if not ok:
        sys.exit(1)


def cmd_warehouse_stats(args: argparse.Namespace) -> None:
    """Print warehouse statistics."""
    from corpus.warehouse import print_warehouse_stats, ensure_warehouse_dirs
    ensure_warehouse_dirs()
    print_warehouse_stats()


def cmd_audit_report(args: argparse.Namespace) -> None:
    """Generate an audit report."""
    from governance.audit import (
        full_activity_report, source_compliance_report, all_sources_report,
    )

    source_id = getattr(args, "source", None)
    date_from = getattr(args, "from_date", None)
    date_to   = getattr(args, "to_date",   None)

    if source_id:
        source_compliance_report(source_id)
    else:
        full_activity_report(date_from=date_from, date_to=date_to)
        all_sources_report()


def cmd_list_releases(args: argparse.Namespace) -> None:
    """List all training releases."""
    from corpus.release_builder import ReleaseBuilder
    ReleaseBuilder.list_releases()


# ── Parser setup ──────────────────────────────────────────────────────

def _cmd_lock_release(args):
    from release.lock import lock_release, ReleaseLockError
    import sys
    try:
        lock_release(args.version, operator=args.operator, notes=getattr(args, "notes", ""))
    except ReleaseLockError as e:
        print("ERROR: " + str(e), file=sys.stderr)
        sys.exit(1)

def _cmd_unlock_release(args):
    from release.lock import break_lock
    break_lock(args.version, operator=args.operator, reason=args.reason)

def _cmd_training_ledger(args):
    from train.training_ledger import print_ledger_summary
    print_ledger_summary()

def _cmd_shard_status(args):
    from train.shard_tracker import print_shard_status
    print_shard_status()

def _cmd_resume_verify(args):
    from train.resume_guard import ResumeGuard, ResumeGuardError
    import sys
    try:
        g = ResumeGuard(release_id=args.version, resume=not getattr(args,"fresh",False),
                        operator=getattr(args,"operator","system"))
        g.run()
    except ResumeGuardError as e:
        print("ERROR: " + str(e), file=sys.stderr); sys.exit(1)

def _cmd_provenance_report(args):
    from train.provenance import generate_report
    generate_report()

_SUBCOMMAND_MAP = {
    "register-source":  cmd_register_source,
    "validate-source":  cmd_validate_source,
    "run-pipeline":     cmd_run_pipeline,
    "review":           cmd_review,
    "build-release":    cmd_build_release,
    "verify-release":   cmd_verify_release,
    "sign-release":     cmd_sign_release,
    "warehouse-stats":  cmd_warehouse_stats,
    "audit-report":     cmd_audit_report,
    "list-releases":    cmd_list_releases,
    "lock-release":     _cmd_lock_release,
    "unlock-release":   _cmd_unlock_release,
    "training-ledger":  _cmd_training_ledger,
    "shard-status":     _cmd_shard_status,
    "resume-verify":    _cmd_resume_verify,
    "provenance-report":_cmd_provenance_report,
}


def build_corpus_parser(sub: argparse._SubParsersAction) -> None:
    """
    Add the corpus subparser tree to the main argument parser.

    Call from main.py during parser setup:
      from corpus.cli import build_corpus_parser
      build_corpus_parser(subparsers)
    """
    cp = sub.add_parser("corpus", help="Corpus engineering commands",
                        formatter_class=argparse.RawDescriptionHelpFormatter,
                        description=__doc__)
    cs = cp.add_subparsers(dest="corpus_subcmd", metavar="SUBCMD")
    cs.required = True

    # ── register-source ───────────────────────────────────────────────
    p = cs.add_parser("register-source", help="Register a new corpus source")
    p.add_argument("source_id",         help="Unique source ID, e.g. A-gutenberg-20260701")
    p.add_argument("--category",        required=True, help="Category letter A-K")
    p.add_argument("--name",            default="",    help="Human-readable source name")
    p.add_argument("--url",             default="",    help="Canonical source URL")
    p.add_argument("--licence",         required=True, help="Licence display name")
    p.add_argument("--licence-id",      default="unknown", help="Licence ID key")
    p.add_argument("--licence-risk",    default="0.8", help="Licence risk 0.0-1.0")
    p.add_argument("--raw-path",        default="",    help="Path to raw source files")
    p.add_argument("--raw-sha256",      default="",    help="SHA-256 of raw archive")
    p.add_argument("--operator",        default="system", help="Your username")
    p.add_argument("--notes",           default="",    help="Optional notes")
    p.set_defaults(func=cmd_register_source)

    # ── validate-source ───────────────────────────────────────────────
    p = cs.add_parser("validate-source",
                      help="Mark a source as validated after licence + content review")
    p.add_argument("source_id")
    p.add_argument("--operator", default="system")
    p.add_argument("--reject",   action="store_true", help="Reject instead of validate")
    p.add_argument("--reason",   default="", help="Reason (required when rejecting)")
    p.set_defaults(func=cmd_validate_source)

    # ── run-pipeline ──────────────────────────────────────────────────
    p = cs.add_parser("run-pipeline", help="Run pipeline stages for a source")
    p.add_argument("source_id")
    p.add_argument("--stages",   default="clean,score",
                   help="Comma-separated stages: clean,dedup,score")
    p.add_argument("--operator", default="system")
    p.add_argument("--verbose",  action="store_true")
    p.set_defaults(func=cmd_run_pipeline)

    # ── review ────────────────────────────────────────────────────────
    p = cs.add_parser("review", help="Interactive human review session for queued items")
    p.add_argument("--source",   default=None, help="Limit review to one source")
    p.add_argument("--operator", default="reviewer", help="Your username")
    p.add_argument("--stats",    action="store_true", help="Show queue stats instead")
    p.set_defaults(func=cmd_review)

    # ── build-release ─────────────────────────────────────────────────
    p = cs.add_parser("build-release", help="Assemble a training release")
    p.add_argument("version", help="Release version, e.g. v1")
    p.add_argument("--categories",   default="A,B,C,D,E,F,G,H,I,J,K")
    p.add_argument("--token-budget", default="30000000000", help="Max tokens")
    p.add_argument("--dry-run",      action="store_true")
    p.add_argument("--verbose",      action="store_true", default=True)
    p.set_defaults(func=cmd_build_release)

    # ── verify-release ────────────────────────────────────────────────
    p = cs.add_parser("verify-release",
                      help="Verify checksums and leakage for a built release")
    p.add_argument("version")
    p.set_defaults(func=cmd_verify_release)

    # ── sign-release ──────────────────────────────────────────────────
    p = cs.add_parser("sign-release",
                      help="Sign (lock) a verified release — mandatory before training")
    p.add_argument("version")
    p.add_argument("--operator", required=True, help="Your username")
    p.set_defaults(func=cmd_sign_release)

    # ── warehouse-stats ───────────────────────────────────────────────
    p = cs.add_parser("warehouse-stats", help="Print warehouse statistics")
    p.set_defaults(func=cmd_warehouse_stats)

    # ── audit-report ──────────────────────────────────────────────────
    p = cs.add_parser("audit-report", help="Generate an audit report")
    p.add_argument("--source",    dest="source",    default=None)
    p.add_argument("--from",      dest="from_date", default=None, help="YYYY-MM-DD")
    p.add_argument("--to",        dest="to_date",   default=None, help="YYYY-MM-DD")
    p.set_defaults(func=cmd_audit_report)

    # ── list-releases ─────────────────────────────────────────────────
    p = cs.add_parser("list-releases", help="List all training releases")
    p.set_defaults(func=cmd_list_releases)

    # ── lock-release ───────────────────────────────────────────────
    p = cs.add_parser("lock-release", help="Lock a signed release for training")
    p.add_argument("version")
    p.add_argument("--operator", required=True)
    p.add_argument("--notes", default="")
    p.set_defaults(func=_cmd_lock_release)

    # ── unlock-release ─────────────────────────────────────────────
    p = cs.add_parser("unlock-release", help="Admin: break a release lock")
    p.add_argument("version")
    p.add_argument("--operator", required=True)
    p.add_argument("--reason",   required=True)
    p.set_defaults(func=_cmd_unlock_release)

    # ── training-ledger ────────────────────────────────────────────
    p = cs.add_parser("training-ledger", help="Print training session ledger")
    p.set_defaults(func=_cmd_training_ledger)

    # ── shard-status ───────────────────────────────────────────────
    p = cs.add_parser("shard-status", help="Print shard tracker status")
    p.set_defaults(func=_cmd_shard_status)

    # ── resume-verify ──────────────────────────────────────────────
    p = cs.add_parser("resume-verify", help="Run resume guard pre-flight check")
    p.add_argument("version")
    p.add_argument("--fresh",    action="store_true")
    p.add_argument("--operator", default="system")
    p.set_defaults(func=_cmd_resume_verify)

    # ── provenance-report ──────────────────────────────────────────
    p = cs.add_parser("provenance-report", help="Print training provenance report")
    p.set_defaults(func=_cmd_provenance_report)

    cp.set_defaults(func=_dispatch_corpus)


def _dispatch_corpus(args: argparse.Namespace) -> None:
    """Dispatch to the correct corpus subcommand handler."""
    fn = getattr(args, "func", None)
    if fn and fn is not _dispatch_corpus:
        fn(args)
    else:
        subcmd = getattr(args, "corpus_subcmd", None)
        fn     = _SUBCOMMAND_MAP.get(subcmd)
        if fn:
            fn(args)
        else:
            print(f"Unknown corpus subcommand: {subcmd}", file=sys.stderr)
            sys.exit(1)


# ── Legacy flat dispatch (for main.py compatibility) ──────────────────

def run_corpus_command(args) -> None:
    """
    Entry point called by main.py::cmd_corpus().

    args may be an argparse.Namespace or a list of strings.
    If it's a list, parse it here.
    """
    if isinstance(args, list):
        parser = argparse.ArgumentParser(prog="main.py corpus")
        sub    = parser.add_subparsers(dest="corpus_subcmd")
        build_corpus_parser(sub)
        parsed = parser.parse_args(args)
        fn     = getattr(parsed, "func", None)
        if fn:
            fn(parsed)
    elif hasattr(args, "corpus_subcmd"):
        fn = getattr(args, "func", None)
        if fn:
            fn(args)
        else:
            _dispatch_corpus(args)
    else:
        print("Usage: python main.py corpus <subcommand> [options]")
        print("Run: python main.py corpus --help")
