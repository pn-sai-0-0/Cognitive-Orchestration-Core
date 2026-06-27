#!/usr/bin/env python3
from __future__ import annotations

from _common.doc import canonical_doc
from _common.io import iter_files, write_json
from _common.logging_ import StageLogger
from _common.manifests import write_manifest
from _common.stage_runner import build_parser, prepare

STAGE = "08_normalize_corpus"


def main() -> int:
    parser = build_parser("Normalize cleaned files to canonical schema")
    args = parser.parse_args()
    paths, _cfg = prepare(args.root)
    logger = StageLogger(STAGE, paths.logs / f"{STAGE}.jsonl")
    files = list(iter_files(paths.cleaned, extensions={".txt", ".md", ".json", ".jsonl", ".csv", ".xml", ".parquet"}))
    logger.log("start", inputs=len(files))
    count = 0
    for src in files:
        rel = src.relative_to(paths.cleaned)
        dst = (paths.normalized / rel).with_suffix(".json")
        if args.resume and dst.exists():
            continue
        text = src.read_text(encoding="utf-8", errors="ignore")
        doc = canonical_doc(relative_path=str(rel).replace("\\", "/"), source_path=str(src), text=text, extra={"processing_history": [STAGE]})
        if not args.dry_run:
            write_json(dst, doc)
        count += 1
    report = {"stage": STAGE, "status": "ok", "normalized_docs": count}
    write_json(paths.reports / "normalization_report.json", report)
    write_manifest(paths.root, STAGE, report)
    logger.log("done", **report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
