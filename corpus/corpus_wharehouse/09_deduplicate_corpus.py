#!/usr/bin/env python3
from __future__ import annotations

import sqlite3
from pathlib import Path

from _common.hashing import hamming_distance, jaccard, sha256_text, simhash, word_shingles
from _common.io import iter_files, read_json, write_json
from _common.logging_ import StageLogger
from _common.manifests import write_manifest
from _common.stage_runner import build_parser, prepare

STAGE = "09_deduplicate_corpus"


def main() -> int:
    parser = build_parser("Remove exact and near duplicate normalized documents")
    args = parser.parse_args()
    paths, cfg = prepare(args.root)
    logger = StageLogger(STAGE, paths.logs / f"{STAGE}.jsonl")
    files = list(iter_files(paths.normalized, extensions={".json"}))
    logger.log("start", inputs=len(files))
    db = sqlite3.connect(paths.pipeline_state / "dedup_index.sqlite3")
    db.execute("create table if not exists seen_exact (hash text primary key, doc_id text, rel text)")
    db.execute("create table if not exists seen_near (doc_id text, rel text, simhash integer, shingles text)")
    kept = exact_dups = near_dups = semantic_dups = 0
    examples = []
    threshold = cfg["deduplication"]["simhash_hamming_threshold"]
    jac_thr = cfg["deduplication"]["minhash_jaccard_threshold"]
    for src in files:
        rel = str(src.relative_to(paths.normalized)).replace("\\", "/")
        dst = paths.deduplicated / rel
        if args.resume and dst.exists():
            continue
        doc = read_json(src)
        text = doc.get("text", "")
        text_hash = sha256_text(text)
        row = db.execute("select doc_id, rel from seen_exact where hash=?", (text_hash,)).fetchone()
        if row:
            exact_dups += 1
            examples.append({"type": "exact", "doc": doc.get("id"), "duplicate_of": row[0], "relative_path": rel})
            continue
        current_sim = simhash(text)
        current_sh = word_shingles(text)
        dup_type = None
        dup_of = None
        for doc_id, prior_rel, prior_sim, prior_sh in db.execute("select doc_id, rel, simhash, shingles from seen_near"):
            dist = hamming_distance(current_sim, int(prior_sim))
            if dist <= threshold:
                dup_type = "near"
                dup_of = doc_id
                near_dups += 1
                break
            prior_set = set(prior_sh.split("\u241f")) if prior_sh else set()
            if jaccard(current_sh, prior_set) >= jac_thr:
                dup_type = "semantic"
                dup_of = doc_id
                semantic_dups += 1
                break
        if dup_type:
            examples.append({"type": dup_type, "doc": doc.get("id"), "duplicate_of": dup_of, "relative_path": rel})
            continue
        doc.setdefault("processing_history", []).append(STAGE)
        if not args.dry_run:
            write_json(dst, doc)
        db.execute("insert or ignore into seen_exact(hash, doc_id, rel) values (?,?,?)", (text_hash, doc.get("id"), rel))
        db.execute(
            "insert into seen_near(doc_id, rel, simhash, shingles) values (?,?,?,?)",
            (doc.get("id"), rel, str(current_sim), "\u241f".join(sorted(current_sh))),
        )
        kept += 1
    db.commit()
    db.close()
    report = {
        "stage": STAGE,
        "status": "ok",
        "input_docs": len(files),
        "kept_docs": kept,
        "exact_duplicates": exact_dups,
        "near_duplicates": near_dups,
        "semantic_duplicates": semantic_dups,
        "duplicate_examples": examples[:50],
    }
    write_json(paths.reports / "duplicate_report.json", report)
    write_manifest(paths.root, STAGE, report)
    logger.log("done", kept=kept, exact=exact_dups, near=near_dups, semantic=semantic_dups)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
