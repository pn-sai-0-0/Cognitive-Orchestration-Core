#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from _common.io import iter_files, write_json, write_jsonl, read_textual_content
from _common.logging_ import StageLogger
from _common.manifests import write_manifest
from _common.stage_runner import build_parser, prepare
from _common.transforms import clean_text

STAGE = "07_clean_corpus"


def main() -> int:
    parser = build_parser("Clean validated corpus files")
    args = parser.parse_args()
    paths, _cfg = prepare(args.root)
    logger = StageLogger(STAGE, paths.logs / f"{STAGE}.jsonl")
    files = list(iter_files(paths.validated))
    logger.log("start", inputs=len(files))
    rows = []
    cleaned = 0
    for src in files:
        rel = src.relative_to(paths.validated)
        dst = (paths.cleaned / rel).with_suffix(".txt")
        if args.resume and dst.exists():
            continue
        text = read_textual_content(src)
        out = clean_text(text)
        if not args.dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(out, encoding="utf-8")
        rows.append({"source": str(src), "cleaned": str(dst), "chars_in": len(text), "chars_out": len(out)})
        cleaned += 1
    if not args.dry_run:
        write_jsonl(paths.cleaned / "cleaning_manifest.jsonl", rows)
    report = {"stage": STAGE, "status": "ok", "input_files": len(files), "cleaned_files": cleaned}
    write_json(paths.reports / "cleaning_report.json", report)
    write_manifest(paths.root, STAGE, report)
    logger.log("done", **report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
