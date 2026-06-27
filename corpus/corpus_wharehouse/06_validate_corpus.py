#!/usr/bin/env python3
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from _common.hashing import sha256_file
from _common.io import SUPPORTED_EXTENSIONS, iter_files, link_or_copy, write_json, write_jsonl
from _common.logging_ import StageLogger
from _common.manifests import write_manifest
from _common.stage_runner import build_parser, prepare


STAGE = "06_validate_corpus"


def _validate_one(path_str: str, root_str: str) -> dict:
    path = Path(path_str)
    root = Path(root_str)
    rel = str(path.relative_to(root)).replace("\\", "/")
    size = path.stat().st_size
    suffix = path.suffix.lower()
    status = "valid"
    error = None
    sha = None
    try:
        sha = sha256_file(path)
        if suffix not in SUPPORTED_EXTENSIONS:
            status = "unsupported"
        elif size <= 0:
            status = "empty"
    except Exception as e:
        status = "error"
        error = repr(e)
    return {
        "relative_path": rel,
        "source_path": str(path),
        "suffix": suffix,
        "bytes": size,
        "sha256": sha,
        "status": status,
        "error": error,
    }


def main() -> int:
    parser = build_parser("Validate raw corpus files and freeze validated inventory")
    args = parser.parse_args()
    paths, cfg = prepare(args.root)
    logger = StageLogger(STAGE, paths.logs / f"{STAGE}.jsonl")
    input_files = list(iter_files(paths.raw))
    workers = args.workers or cfg["runtime"]["workers"]
    logger.log("start", inputs=len(input_files), workers=workers)
    if not input_files:
        report = {"stage": STAGE, "status": "no_input", "valid_files": 0, "invalid_files": 0}
        write_json(paths.reports / "validation_report.json", report)
        write_manifest(paths.root, STAGE, report)
        return 0

    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(_validate_one, [str(p) for p in input_files], [str(paths.raw)] * len(input_files)))

    materialized = 0
    for row in rows:
        if row["status"] != "valid":
            continue
        src = Path(row["source_path"])
        dst = paths.validated / row["relative_path"]
        row["validated_path"] = str(dst)
        if not args.dry_run:
            mode = link_or_copy(src, dst)
            row["materialized_as"] = mode
        materialized += 1

    if not args.dry_run:
        write_jsonl(paths.validated / "validated_manifest.jsonl", rows)
    report = {
        "stage": STAGE,
        "status": "ok",
        "input_files": len(rows),
        "valid_files": sum(r["status"] == "valid" for r in rows),
        "invalid_files": sum(r["status"] != "valid" for r in rows),
        "materialized": materialized,
    }
    write_json(paths.reports / "validation_report.json", report)
    write_manifest(paths.root, STAGE, report)
    logger.log("done", **report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
