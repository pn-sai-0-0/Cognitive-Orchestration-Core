#!/usr/bin/env python3
from __future__ import annotations

from _common.io import iter_files, read_json, write_jsonl
from _common.logging_ import StageLogger
from _common.manifests import write_manifest
from _common.stage_runner import build_parser, prepare
from _common.transforms import detect_language

STAGE = "11_language_detection"


def main() -> int:
    parser = build_parser("Detect language for filtered corpus")
    args = parser.parse_args()
    paths, _cfg = prepare(args.root)
    logger = StageLogger(STAGE, paths.logs / f"{STAGE}.jsonl")
    files = list(iter_files(paths.filtered, extensions={".json"}))
    logger.log("start", inputs=len(files))
    rows = []
    for src in files:
        doc = read_json(src)
        lang = detect_language(doc.get("text", ""))
        rows.append({"id": doc.get("id"), "relative_path": str(src.relative_to(paths.filtered)).replace("\\", "/"), **lang})
    if not args.dry_run:
        write_jsonl(paths.quality / "language_annotations.jsonl", rows)
    report = {"stage": STAGE, "status": "ok", "input_docs": len(files), "languages": sorted({r['language'] for r in rows})}
    write_manifest(paths.root, STAGE, report)
    logger.log("done", docs=len(rows), languages=report["languages"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
