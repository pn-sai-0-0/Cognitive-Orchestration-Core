#!/usr/bin/env python3
from __future__ import annotations

import hashlib
from collections import defaultdict

from _common.io import iter_files, read_json, write_json
from _common.logging_ import StageLogger
from _common.manifests import write_manifest
from _common.stage_runner import build_parser, prepare

STAGE = "14_balance_corpus"


def stable_rank(doc_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{doc_id}".encode()).hexdigest()


def main() -> int:
    parser = build_parser("Balance classified corpus across category/language/difficulty")
    args = parser.parse_args()
    paths, cfg = prepare(args.root)
    logger = StageLogger(STAGE, paths.logs / f"{STAGE}.jsonl")
    seed = cfg["runtime"]["seed"]
    files = list(iter_files(paths.classified, extensions={".json"}))
    buckets = defaultdict(list)
    for src in files:
        doc = read_json(src)
        key = (
            doc.get("classification", {}).get("primary", "Unknown"),
            doc.get("metadata", {}).get("language", "unknown"),
            doc.get("classification", {}).get("difficulty", "basic"),
        )
        buckets[key].append((stable_rank(doc["id"], seed), src, doc))
    target = min((len(v) for v in buckets.values()), default=0)
    max_ratio = cfg["balancing"]["max_category_ratio"]
    selected = 0
    distribution = {}
    for key, items in buckets.items():
        items.sort(key=lambda x: x[0])
        take = max(cfg["balancing"]["min_docs_per_bucket"], min(len(items), int(max(1, target * max_ratio)))) if target else len(items)
        distribution["|".join(key)] = take
        for _, src, doc in items[:take]:
            rel = str(src.relative_to(paths.classified)).replace("\\", "/")
            dst = paths.balanced / rel
            doc.setdefault("processing_history", []).append(STAGE)
            if not args.dry_run:
                write_json(dst, doc)
            selected += 1
    report = {"stage": STAGE, "status": "ok", "input_docs": len(files), "balanced_docs": selected, "bucket_distribution": distribution}
    write_json(paths.reports / "balancing_report.json", report)
    write_manifest(paths.root, STAGE, report)
    logger.log("done", docs=selected, buckets=len(distribution))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
