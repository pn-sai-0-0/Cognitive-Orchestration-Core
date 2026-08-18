#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime, timezone

from _common.hashing import sha256_text
from _common.io import iter_files, read_json, read_jsonl, write_json
from _common.logging_ import StageLogger
from _common.manifests import write_manifest
from _common.stage_runner import build_parser, prepare

STAGE = "12_generate_metadata"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def main() -> int:
    parser = build_parser("Attach immutable metadata to filtered corpus")
    args = parser.parse_args()
    paths, _cfg = prepare(args.root)
    logger = StageLogger(STAGE, paths.logs / f"{STAGE}.jsonl")
    files = list(iter_files(paths.filtered, extensions={".json"}))
    annotations = {row["id"]: row for row in read_jsonl(paths.quality / "language_annotations.jsonl")} if (paths.quality / "language_annotations.jsonl").exists() else {}
    logger.log("start", inputs=len(files), annotations=len(annotations))
    count = 0
    for src in files:
        rel = str(src.relative_to(paths.filtered)).replace("\\", "/")
        dst = paths.metadata / rel
        if args.resume and dst.exists():
            continue
        doc = read_json(src)
        text = doc.get("text", "")
        meta = {
            "id": doc["id"],
            "checksum": sha256_text(text),
            "license": "inherit-from-source-manifest",
            "language": annotations.get(doc["id"], {}).get("language", "unknown"),
            "language_confidence": annotations.get(doc["id"], {}).get("confidence", 0.0),
            "category": None,
            "tokens": doc.get("tokens", 0),
            "quality": doc.get("quality", {}),
            "source": {"relative_path": rel, "source_path": doc.get("source_path")},
            "processing_history": doc.get("processing_history", []) + [STAGE],
            "timestamps": {"metadata_created_utc": utc_now()},
            "manifest_reference": "pipeline_state/12_generate_metadata.manifest.json",
        }
        doc["metadata"] = meta
        if not args.dry_run:
            write_json(dst, doc)
        count += 1
    report = {"stage": STAGE, "status": "ok", "metadata_docs": count}
    write_manifest(paths.root, STAGE, report)
    logger.log("done", docs=count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
