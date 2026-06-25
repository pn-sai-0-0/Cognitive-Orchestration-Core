#!/usr/bin/env python3
"""
CognitiveOC - Full Corpus Pipeline Runner
==========================================
One command: build skeleton -> download sources -> health report.

Usage:
    python 04_run_full_pipeline.py                      # Priority 1, release-safe, skip large
    python 04_run_full_pipeline.py --priority 2         # Priority 2 sources
    python 04_run_full_pipeline.py --no-skip-large      # Include large downloads
    python 04_run_full_pipeline.py --dry-run            # Plan only
"""
from __future__ import annotations
import argparse, os, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_WAREHOUSE = Path(os.environ.get(
    "COC_WAREHOUSE_DIR",
    r"D:\projects\CognitiveOC\Final_Versions\cognitiveoc_v3\corpus\corpus_wharehouse",
))

def run(cmd, dry=False):
    print(f"\n$ {' '.join(str(c) for c in cmd)}")
    if dry: return 0
    return subprocess.call([str(c) for c in cmd])

def main():
    p = argparse.ArgumentParser(description="CognitiveOC full corpus pipeline runner")
    p.add_argument("--root", default=str(DEFAULT_WAREHOUSE))
    p.add_argument("--priority", type=int, choices=[1,2,3], default=1)
    p.add_argument("--skip-large", action="store_true", default=True)
    p.add_argument("--no-skip-large", dest="skip_large", action="store_false")
    p.add_argument("--include-warehouse-only", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-build", action="store_true")
    p.add_argument("--skip-acquire", action="store_true")
    p.add_argument("--skip-health", action="store_true")
    args = p.parse_args()

    py = sys.executable
    r = ["--root", args.root]

    if not args.skip_build:
        rc = run([py, str(HERE / "01_create_warehouse_architecture.py"), *r])
        if rc: print("Build failed.", file=sys.stderr); return rc

    if not args.skip_acquire:
        cmd = [py, str(HERE / "02_corpus_acquisition_manager.py"), *r,
               "--all", "--priority", str(args.priority)]
        if args.skip_large: cmd.append("--skip-large")
        if args.include_warehouse_only: cmd.append("--include-warehouse-only")
        rc = run(cmd, dry=args.dry_run)
        if rc: print("Acquisition had issues.", file=sys.stderr)

    if not args.skip_health:
        run([py, str(HERE / "03_corpus_health_report.py"), *r], dry=args.dry_run)

    print("\n[pipeline] done.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
