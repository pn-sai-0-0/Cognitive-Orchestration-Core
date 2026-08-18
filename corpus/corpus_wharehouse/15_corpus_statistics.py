#!/usr/bin/env python3
from __future__ import annotations

import math
from collections import Counter

from _common.hashing import tokenize
from _common.io import iter_files, read_json, read_json as read_json_file, write_json
from _common.logging_ import StageLogger
from _common.manifests import write_manifest
from _common.stage_runner import build_parser, prepare

STAGE = "15_corpus_statistics"


def percentile(sorted_vals, p):
    if not sorted_vals:
        return 0
    idx = int((len(sorted_vals) - 1) * p)
    return sorted_vals[idx]


def main() -> int:
    parser = build_parser("Compute balanced corpus analytics and statistics")
    args = parser.parse_args()
    paths, _cfg = prepare(args.root)
    logger = StageLogger(STAGE, paths.logs / f"{STAGE}.jsonl")
    files = list(iter_files(paths.balanced, extensions={".json"}))
    toks = []
    chars = []
    vocab = Counter()
    categories = Counter()
    languages = Counter()
    for src in files:
        doc = read_json(src)
        text = doc.get("text", "")
        toks.append(doc.get("tokens", 0))
        chars.append(doc.get("chars", len(text)))
        vocab.update(tokenize(text))
        categories.update([doc.get("classification", {}).get("primary", "Unknown")])
        languages.update([doc.get("metadata", {}).get("language", "unknown")])
    toks_sorted = sorted(toks)
    dup_report = paths.reports / "duplicate_report.json"
    dup_pct = 0.0
    if dup_report.exists():
        rep = read_json_file(dup_report)
        total = rep.get("input_docs", 0)
        dups = rep.get("exact_duplicates", 0) + rep.get("near_duplicates", 0) + rep.get("semantic_duplicates", 0)
        dup_pct = round(100.0 * dups / total, 4) if total else 0.0
    stats = {
        "stage": STAGE,
        "status": "ok",
        "document_count": len(files),
        "token_count": sum(toks),
        "vocab_size": len(vocab),
        "char_count": sum(chars),
        "token_stats": {
            "min": min(toks_sorted) if toks_sorted else 0,
            "p50": percentile(toks_sorted, 0.50),
            "p90": percentile(toks_sorted, 0.90),
            "max": max(toks_sorted) if toks_sorted else 0,
            "mean": round(sum(toks) / len(toks), 4) if toks else 0,
        },
        "category_distribution": dict(categories),
        "language_distribution": dict(languages),
        "duplicate_percent": dup_pct,
    }
    md = [
        "# Corpus Statistics",
        f"- Documents: {stats['document_count']}",
        f"- Tokens: {stats['token_count']}",
        f"- Vocabulary size: {stats['vocab_size']}",
        f"- Duplicate percent removed: {stats['duplicate_percent']}%",
        "",
        "## Category distribution",
    ]
    md += [f"- {k}: {v}" for k, v in sorted(categories.items())]
    write_json(paths.statistics / "corpus_statistics.json", stats)
    (paths.statistics / "corpus_statistics.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    write_manifest(paths.root, STAGE, stats)
    logger.log("done", docs=len(files), tokens=sum(toks), vocab=len(vocab))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
