#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json

from _common.io import iter_files, read_json
from _common.logging_ import StageLogger
from _common.manifests import write_manifest
from _common.stage_runner import build_parser, prepare

STAGE = "16_build_training_splits"


def stable_sort_key(doc_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{doc_id}".encode()).hexdigest()


def dump_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = build_parser("Create deterministic training splits")
    args = parser.parse_args()
    paths, cfg = prepare(args.root)
    logger = StageLogger(STAGE, paths.logs / f"{STAGE}.jsonl")
    seed = cfg["runtime"]["seed"]
    docs = [read_json(p) for p in iter_files(paths.balanced, extensions={".json"})]
    benchmark = [d for d in docs if "evaluation" in d.get("relative_path", "").lower() or d.get("classification", {}).get("primary") == "Evaluation"]
    main_docs = [d for d in docs if d not in benchmark]
    main_docs.sort(key=lambda d: stable_sort_key(d["id"], seed))
    n = len(main_docs)
    n_val = int(n * cfg["splits"]["validation"])
    n_test = int(n * cfg["splits"]["test"])
    val = main_docs[:n_val]
    test = main_docs[n_val:n_val + n_test]
    train = main_docs[n_val + n_test:]
    if not args.dry_run:
        dump_jsonl(paths.splits / "train.jsonl", train)
        dump_jsonl(paths.splits / "validation.jsonl", val)
        dump_jsonl(paths.splits / "test.jsonl", test)
        dump_jsonl(paths.splits / "benchmark.jsonl", benchmark)
    report = {
        "stage": STAGE,
        "status": "ok",
        "train_docs": len(train),
        "validation_docs": len(val),
        "test_docs": len(test),
        "benchmark_docs": len(benchmark),
    }
    write_manifest(paths.root, STAGE, report)
    logger.log("done", **report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
