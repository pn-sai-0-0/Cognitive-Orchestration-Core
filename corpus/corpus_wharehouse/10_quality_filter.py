#!/usr/bin/env python3
from __future__ import annotations

from _common.io import iter_files, read_json, write_json
from _common.logging_ import StageLogger
from _common.manifests import write_manifest
from _common.stage_runner import build_parser, prepare
from _common.transforms import quality_metrics

STAGE = "10_quality_filter"


def main() -> int:
    parser = build_parser("Score and filter deduplicated corpus")
    args = parser.parse_args()
    paths, cfg = prepare(args.root)
    logger = StageLogger(STAGE, paths.logs / f"{STAGE}.jsonl")
    files = list(iter_files(paths.deduplicated, extensions={".json"}))
    logger.log("start", inputs=len(files))
    kept = rejected = 0
    details = []
    min_chars = cfg["quality"]["min_chars"]
    min_score = cfg["quality"]["min_quality_score"]
    for src in files:
        rel = str(src.relative_to(paths.deduplicated)).replace("\\", "/")
        dst = paths.filtered / rel
        if args.resume and dst.exists():
            continue
        doc = read_json(src)
        metrics = quality_metrics(doc.get("text", ""))
        passed = metrics["chars"] >= min_chars and metrics["quality_score"] >= min_score and metrics["ocr_noise_ratio"] <= cfg["quality"]["max_ocr_noise_ratio"]
        metrics["passed"] = passed
        details.append({"id": doc.get("id"), "relative_path": rel, **metrics})
        if passed:
            doc["quality"] = metrics
            doc.setdefault("processing_history", []).append(STAGE)
            if not args.dry_run:
                write_json(dst, doc)
            kept += 1
        else:
            rejected += 1
    report = {"stage": STAGE, "status": "ok", "input_docs": len(files), "kept_docs": kept, "rejected_docs": rejected, "details": details[:200]}
    write_json(paths.quality / "quality_report.json", report)
    write_manifest(paths.root, STAGE, report)
    logger.log("done", kept=kept, rejected=rejected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
