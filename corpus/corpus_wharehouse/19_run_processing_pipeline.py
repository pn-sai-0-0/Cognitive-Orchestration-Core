#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from _common.paths import DEFAULT_WAREHOUSE

HERE = Path(__file__).resolve().parent
STEPS = [
    "06_validate_corpus.py",
    "07_clean_corpus.py",
    "08_normalize_corpus.py",
    "09_deduplicate_corpus.py",
    "10_quality_filter.py",
    "11_language_detection.py",
    "12_generate_metadata.py",
    "13_classify_corpus.py",
    "14_balance_corpus.py",
    "15_corpus_statistics.py",
    "16_build_training_splits.py",
    "17_prepare_tokenizer_input.py",
    "18_pipeline_report.py",
]


def main() -> int:
    p = argparse.ArgumentParser(description="Run CognitiveOC corpus processing pipeline 06-18")
    p.add_argument("--root", default=str(DEFAULT_WAREHOUSE))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--verify", action="store_true")
    args = p.parse_args()
    py = sys.executable
    for step in STEPS:
        cmd = [py, str(HERE / step), "--root", args.root]
        if args.resume:
            cmd.append("--resume")
        if args.workers:
            cmd += ["--workers", str(args.workers)]
        if args.dry_run:
            print("DRY-RUN:", " ".join(cmd))
            continue
        print("$", " ".join(cmd))
        rc = subprocess.call(cmd)
        if rc:
            return rc
    if args.verify and not args.dry_run:
        required = [Path(args.root) / "PIPELINE_STATUS.md", Path(args.root) / "splits" / "train.jsonl", Path(args.root) / "tokenizer_input" / "tokenizer_manifest.json"]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            print("verification failed:", missing, file=sys.stderr)
            return 2
    print("[pipeline] processing complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
