#!/usr/bin/env python3
"""
CognitiveOC - Warehouse Architecture Builder
=============================================
Creates the complete corpus_wharehouse directory structure.

Run:
    python 01_create_warehouse_architecture.py
    python 01_create_warehouse_architecture.py --root D:/custom/path
    python 01_create_warehouse_architecture.py --dry-run
"""

from __future__ import annotations
import argparse, json, os
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_WAREHOUSE = Path(os.environ.get(
    "COC_WAREHOUSE_DIR",
    r"D:\projects\CognitiveOC\Final_Versions\cognitiveoc_v3\corpus\corpus_wharehouse",
))

CATEGORIES = {
    "A_books":              "Books & long-form knowledge (Gutenberg, PG-19, Standard Ebooks, NCBI Bookshelf)",
    "B_educational":        "Educational content (OpenStax, CK-12, Saylor, Wikibooks, Wikiversity)",
    "C_reasoning":          "STEM reasoning & math (GSM8K, MATH, NuminaMath, ARC, OpenWebMath, Proof-Pile-2)",
    "D_conversations":      "Instruction & dialogue (Dolly, OASST2, Tulu-3, FLAN, Aya)",
    "E_technical_docs":     "Technical documentation (Python, NumPy, PyTorch, Linux, RFCs, GNU)",
    "F_articles":           "Encyclopedic articles (Wikipedia, Wikisource)",
    "G_research_papers":    "Research papers (arXiv, PMC OA, OpenAlex, S2ORC, eLife)",
    "H_synthetic":          "Synthetic data (Cosmopedia, self-generated CoT traces)",
    "I_cognition":          "Cognition & psychology (OpenStax Psych, NCBI Bookshelf cognition)",
    "J_retrieval":          "Retrieval & QA (MS MARCO, NQ, HotpotQA, TriviaQA, FEVER, BEIR, MIRACL)",
    "K_knowledge_graph":    "Structured knowledge (Wikidata, DBpedia, ConceptNet, YAGO)",
    "L_language_resources": "Linguistic resources (WordNet, FrameNet, VerbNet, Universal Dependencies)",
    "M_legal_government":   "Legal & government (CourtListener, SEC EDGAR, NIST, IETF RFCs, data.gov)",
    "N_evaluation":         "Eval/benchmark sets (MMLU, HellaSwag, TruthfulQA, BIG-Bench, AGIEval, LongBench)",
}

STAGES = ["raw", "cleaned", "deduplicated", "scored", "approved", "rejected", "synthetic"]

GOVERNANCE_DIRS = {
    "manifests": "Per-source JSON manifests (license, sha256, record counts, provenance)",
    "provenance": "Per-record provenance ledgers",
    "releases": "Versioned training release bundles (v1, v2, v3...)",
    "audit": "Risk audits, license audits, leakage audits",
    "governance_logs": "Append-only governance event log",
    "logs": "Acquisition logs, processing logs, error logs",
    "review_queue": "Items requiring human review before approval",
    "quarantine": "Suspected leakage / license violations / poisoned data",
    "archive": "Recalled or retired sources (audit trail kept)",
    "hf_cache": "Hugging Face datasets/hub cache (gitignored)",
    "tokenized": "Tokenized uint16 shards ready for training",
    "shards": "Sharded training-ready files",
    "checkpoints": "Tokenizer + corpus snapshot checkpoints",
    "embeddings": "Document embeddings for dedup / retrieval",
    "holdout": "Held-out splits never seen during training",
    "benchmark": "Eval/benchmark corpora (kept strictly separate from training)",
    "training": "Training plan ledger + per-release training metadata",
    "registry": "Source registry YAML/JSON + dataset catalog",
    "reports": "Health reports, coverage reports, statistics dashboards",
}

RELEASE_VERSIONS = ["v1", "v2", "v3"]
RELEASE_PLACEHOLDERS = {
    "manifest.json": {"release": "placeholder", "status": "not_built"},
    "provenance.json": {"release": "placeholder", "sources": []},
    "training_plan.json": {"release": "placeholder", "epochs": 0, "tokens": 0},
    "dataset_stats.json": {"release": "placeholder", "categories": {}},
}


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ensure_dir(p, dry=False):
    if dry:
        print(f"  [DRY] mkdir {p}")
        return
    p.mkdir(parents=True, exist_ok=True)


def write_readme(p, content, dry=False):
    if dry:
        return
    r = p / "README.md"
    if r.exists():
        return
    r.write_text(content, encoding="utf-8")


def write_json(p, obj, dry=False):
    if dry:
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def build_warehouse(root, dry=False):
    print(f"\n{'='*70}\n  CognitiveOC Warehouse Builder\n{'='*70}")
    print(f"  Root: {root}")
    print(f"  Dry-run: {dry}\n{'='*70}\n")

    created = 0
    print("[1/5] Creating warehouse root...")
    ensure_dir(root, dry)
    write_readme(root, f"# CognitiveOC Corpus Warehouse\n\nCreated: {utc_now()}\n", dry)
    created += 1

    print("[2/5] Creating stage x category matrix...")
    for stage in STAGES:
        sp = root / stage
        ensure_dir(sp, dry)
        write_readme(sp, f"# {stage}/\n\nPipeline stage: **{stage}**\n", dry)
        for cat, desc in CATEGORIES.items():
            cp = sp / cat
            ensure_dir(cp, dry)
            write_readme(cp, f"# {cat} -- {stage}\n\n{desc}\n", dry)
            created += 1

    print("[3/5] Creating governance directories...")
    for g, gd in GOVERNANCE_DIRS.items():
        gp = root / g
        ensure_dir(gp, dry)
        write_readme(gp, f"# {g}/\n\n{gd}\n", dry)
        created += 1

    print("[4/5] Creating release version folders...")
    for v in RELEASE_VERSIONS:
        rp = root / "releases" / v
        ensure_dir(rp, dry)
        ensure_dir(rp / "shards", dry)
        write_readme(rp, f"# Release {v}\n\nStatus: **not built**\n", dry)
        for name, payload in RELEASE_PLACEHOLDERS.items():
            write_json(rp / name, {**payload, "version": v}, dry)
        created += 2

    print("[5/5] Creating archive sub-folders...")
    for sub in ["recalled", "retired_sources", "license_violations"]:
        ensure_dir(root / "archive" / sub, dry)
        created += 1

    inv = {
        "created_at": utc_now(), "warehouse_root": str(root),
        "categories": CATEGORIES, "stages": STAGES,
        "governance_dirs": list(GOVERNANCE_DIRS.keys()),
        "release_versions": RELEASE_VERSIONS,
        "total_directories": created,
    }
    write_json(root / "warehouse_inventory.json", inv, dry)

    if not dry:
        gi = root / ".gitignore"
        if not gi.exists():
            gi.write_text(
                "# CognitiveOC warehouse - large files\n"
                "hf_cache/\nraw/*/*/\ncleaned/*/*/\ndeduplicated/*/*/\n"
                "scored/*/*/\ntokenized/\nshards/\nembeddings/\n"
                "*.bin\n*.tar.gz\n*.tar.bz2\n*.parquet\n*.arrow\n*.xml.bz2\n*.csv.gz\n"
                "!**/README.md\n!**/manifest.json\n!**/provenance.json\n",
                encoding="utf-8",
            )

    print(f"\n{'='*70}")
    print(f"  Warehouse skeleton ready")
    print(f"  Directories: {created}")
    print(f"  Categories:  {len(CATEGORIES)}")
    print(f"  Stages:      {len(STAGES)}")
    print(f"  Governance:  {len(GOVERNANCE_DIRS)}")
    print(f"  Releases:    {len(RELEASE_VERSIONS)}")
    print(f"{'='*70}\n")
    return inv


def main():
    parser = argparse.ArgumentParser(description="Build CognitiveOC warehouse skeleton.")
    parser.add_argument("--root", default=str(DEFAULT_WAREHOUSE))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    build_warehouse(Path(args.root), dry=args.dry_run)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
