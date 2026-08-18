#!/usr/bin/env python3
"""
CognitiveOC - Extended Full Corpus Pipeline Runner
===================================================
One command: build skeleton -> run original 02 -> run extended 02b -> health.

Usage:
    # Recommended first run: priority 1, skip large sources
    python 04b_run_extended_pipeline.py --priority 1 --skip-large

    # All priority-1 sources including large ones (Wikipedia, FineWeb-Edu)
    python 04b_run_extended_pipeline.py --priority 1

    # Fill only the empty categories E, F, I, K, M (gap-fill)
    python 04b_run_extended_pipeline.py --only-extended --categories E F I K M

    # Re-download / force refresh
    python 04b_run_extended_pipeline.py --priority 1 --force

    # Skip the build / health report steps
    python 04b_run_extended_pipeline.py --skip-build --skip-health

    # Dry run (plan only)
    python 04b_run_extended_pipeline.py --dry-run
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_WAREHOUSE = Path(os.environ.get(
    "COC_WAREHOUSE_DIR",
    r"D:\projects\CognitiveOC\Final_Versions\cognitiveoc_v3\corpus\corpus_wharehouse",
))


def run(cmd, dry=False):
    print(f"\n$ {' '.join(str(c) for c in cmd)}")
    if dry:
        return 0
    return subprocess.call([str(c) for c in cmd])


def main():
    p = argparse.ArgumentParser(
        description="CognitiveOC extended full corpus pipeline runner")
    p.add_argument("--root", default=str(DEFAULT_WAREHOUSE))
    p.add_argument("--priority", type=int, choices=[1, 2, 3], default=1)
    p.add_argument("--skip-large", action="store_true", default=False)
    p.add_argument("--no-skip-large", dest="skip_large", action="store_false")
    p.add_argument("--include-warehouse-only", action="store_true")
    p.add_argument("--categories", nargs="*", default=None,
                   help="Limit extended phase to specific category letters")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true")
    # phase toggles
    p.add_argument("--skip-build", action="store_true")
    p.add_argument("--skip-original", action="store_true",
                   help="Skip 02_* original acquisition phase")
    p.add_argument("--skip-extended", action="store_true",
                   help="Skip 02b_* extended acquisition phase")
    p.add_argument("--only-extended", action="store_true",
                   help="Run only the extended (02b) phase, skip the original 02")
    p.add_argument("--skip-health", action="store_true")
    args = p.parse_args()

    py = sys.executable
    r = ["--root", args.root]

    # ---- Phase 1: build warehouse skeleton ----
    if not args.skip_build:
        rc = run([py, str(HERE / "01_create_warehouse_architecture.py"), *r],
                 dry=args.dry_run)
        if rc:
            print("Build failed.", file=sys.stderr)
            return rc

    # ---- Phase 2: original acquisition (02_*) ----
    if not args.skip_original and not args.only_extended:
        cmd = [py, str(HERE / "02_corpus_acquisition_manager.py"), *r,
               "--all", "--priority", str(args.priority)]
        if args.skip_large:
            cmd.append("--skip-large")
        if args.include_warehouse_only:
            cmd.append("--include-warehouse-only")
        if args.force:
            cmd.append("--force")
        rc = run(cmd, dry=args.dry_run)
        if rc:
            print("Original acquisition had issues (continuing).",
                  file=sys.stderr)

    # ---- Phase 3: extended acquisition (02b_*) ----
    if not args.skip_extended:
        cmd = [py, str(HERE / "02b_corpus_acquisition_manager_extended.py"), *r,
               "--all", "--priority", str(args.priority)]
        if args.skip_large:
            cmd.append("--skip-large")
        if args.include_warehouse_only:
            cmd.append("--include-warehouse-only")
        if args.force:
            cmd.append("--force")
        if args.categories:
            cmd.extend(["--categories", *args.categories])
        rc = run(cmd, dry=args.dry_run)
        if rc:
            print("Extended acquisition had issues (continuing).",
                  file=sys.stderr)

    # ---- Phase 4: health report ----
    if not args.skip_health:
        run([py, str(HERE / "03_corpus_health_report.py"), *r],
            dry=args.dry_run)

    print("\n[pipeline-extended] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
