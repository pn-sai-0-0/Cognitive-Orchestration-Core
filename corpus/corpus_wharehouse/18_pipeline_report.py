#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from _common.manifests import load_manifest
from _common.stage_runner import build_parser, prepare

STAGE = "18_pipeline_report"
STAGES = [
    "06_validate_corpus", "07_clean_corpus", "08_normalize_corpus", "09_deduplicate_corpus",
    "10_quality_filter", "11_language_detection", "12_generate_metadata", "13_classify_corpus",
    "14_balance_corpus", "15_corpus_statistics", "16_build_training_splits", "17_prepare_tokenizer_input",
]


def main() -> int:
    parser = build_parser("Create consolidated pipeline status report")
    args = parser.parse_args()
    paths, _cfg = prepare(args.root)
    manifests = {stage: load_manifest(paths.root, stage) for stage in STAGES}
    completed = [s for s, m in manifests.items() if m]
    stats = manifests.get("15_corpus_statistics", {})
    lines = [
        "# PIPELINE STATUS",
        "",
        f"- Completed stages: {len(completed)}/{len(STAGES)}",
        f"- Documents: {stats.get('document_count', 0)}",
        f"- Tokens: {stats.get('token_count', 0)}",
        f"- Vocabulary size: {stats.get('vocab_size', 0)}",
        "",
        "## Stage status",
    ]
    for s in STAGES:
        state = "done" if manifests.get(s) else "pending"
        lines.append(f"- {s}: {state}")
    lines += [
        "",
        "## Readiness assessment",
        "- Architecture readiness: good for deterministic workstation-scale corpus processing.",
        "- Scalability readiness: moderate; current implementation is file-oriented and can be adapted to cloud sharding.",
        "- Governance readiness: moderate; source-license inheritance is stubbed and should be connected to source manifests.",
        "",
        "## Remaining steps before tokenizer training",
        "1. Attach source-level license metadata from acquisition manifests.",
        "2. Run the full pipeline on real downloaded corpora and validate performance at scale.",
        "3. Review quality thresholds against corpus samples.",
        "4. Verify balancing ratios against target training mixture.",
        "5. Freeze a release manifest and checksum all tokenizer input chunks.",
    ]
    out = paths.reports / "PIPELINE_STATUS.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (paths.root / "PIPELINE_STATUS.md").write_text(out.read_text(encoding="utf-8"), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
