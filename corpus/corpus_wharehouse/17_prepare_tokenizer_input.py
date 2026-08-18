#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json

from _common.io import iter_files, read_json, write_json
from _common.logging_ import StageLogger
from _common.manifests import write_manifest
from _common.stage_runner import build_parser, prepare

STAGE = "17_prepare_tokenizer_input"


def stable_sort_key(doc_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{doc_id}".encode()).hexdigest()


def main() -> int:
    parser = build_parser("Prepare tokenizer input chunks from balanced corpus")
    args = parser.parse_args()
    paths, cfg = prepare(args.root)
    logger = StageLogger(STAGE, paths.logs / f"{STAGE}.jsonl")
    seed = cfg["runtime"]["seed"]
    target_tokens = cfg["tokenizer"]["target_chunk_tokens"]
    max_docs = cfg["tokenizer"]["max_docs_per_chunk"]
    docs = [read_json(p) for p in iter_files(paths.balanced, extensions={".json"})]
    docs.sort(key=lambda d: stable_sort_key(d["id"], seed))
    out_dir = paths.tokenizer_input / "chunks"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    buf = []
    tok_count = 0
    chunk_idx = 0
    total_docs = 0
    for doc in docs:
        if buf and (tok_count + doc.get("tokens", 0) > target_tokens or len(buf) >= max_docs):
            path = out_dir / f"chunk-{chunk_idx:05d}.jsonl"
            with path.open("w", encoding="utf-8") as f:
                for row in buf:
                    f.write(json.dumps({"id": row["id"], "text": row["text"]}, ensure_ascii=False) + "\n")
            manifest_rows.append({"chunk": path.name, "docs": len(buf), "tokens": tok_count})
            total_docs += len(buf)
            chunk_idx += 1
            buf = []
            tok_count = 0
        buf.append(doc)
        tok_count += doc.get("tokens", 0)
    if buf:
        path = out_dir / f"chunk-{chunk_idx:05d}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for row in buf:
                f.write(json.dumps({"id": row["id"], "text": row["text"]}, ensure_ascii=False) + "\n")
        manifest_rows.append({"chunk": path.name, "docs": len(buf), "tokens": tok_count})
        total_docs += len(buf)
    manifest = {"stage": STAGE, "status": "ok", "chunks": manifest_rows, "total_docs": total_docs}
    write_json(paths.tokenizer_input / "tokenizer_manifest.json", manifest)
    write_manifest(paths.root, STAGE, manifest)
    logger.log("done", docs=total_docs, chunks=len(manifest_rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
