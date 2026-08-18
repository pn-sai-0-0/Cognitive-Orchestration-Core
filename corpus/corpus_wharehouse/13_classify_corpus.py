#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter

from _common.io import iter_files, read_json, write_json
from _common.logging_ import StageLogger
from _common.manifests import write_manifest
from _common.stage_runner import build_parser, prepare
from _common.transforms import classify_text

STAGE = "13_classify_corpus"


def main() -> int:
    parser = build_parser("Classify metadata-enriched corpus")
    args = parser.parse_args()
    paths, _cfg = prepare(args.root)
    logger = StageLogger(STAGE, paths.logs / f"{STAGE}.jsonl")
    files = list(iter_files(paths.metadata, extensions={".json"}))
    counts = Counter()
    done = 0
    for src in files:
        rel = str(src.relative_to(paths.metadata)).replace("\\", "/")
        dst = paths.classified / rel
        if args.resume and dst.exists():
            continue
        doc = read_json(src)
        cls = classify_text(doc.get("text", ""))
        doc["classification"] = cls
        doc["metadata"]["category"] = cls["primary"]
        doc.setdefault("processing_history", []).append(STAGE)
        if not args.dry_run:
            write_json(dst, doc)
        counts.update(cls["labels"])
        done += 1
    report = {"stage": STAGE, "status": "ok", "classified_docs": done, "category_counts": dict(counts)}
    write_json(paths.reports / "classification_report.json", report)
    write_manifest(paths.root, STAGE, report)
    logger.log("done", docs=done, categories=len(counts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
