from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_pipeline_end_to_end(tmp_path: Path):
    root = tmp_path / "warehouse"
    raw = root / "raw"
    write(raw / "A_books" / "book1.txt", "Chapter one. This is a short book about systems and architecture.")
    write(raw / "A_books" / "book1_dup.txt", "Chapter one. This is a short book about systems and architecture.")
    write(raw / "E_technical_docs" / "spec.md", "# Specification\nThe system architecture defines a protocol and deployment workflow.")
    script_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["COC_WAREHOUSE_DIR"] = str(root)
    for step in [
        "06_validate_corpus.py", "07_clean_corpus.py", "08_normalize_corpus.py", "09_deduplicate_corpus.py",
        "10_quality_filter.py", "11_language_detection.py", "12_generate_metadata.py", "13_classify_corpus.py",
        "14_balance_corpus.py", "15_corpus_statistics.py", "16_build_training_splits.py", "17_prepare_tokenizer_input.py",
        "18_pipeline_report.py",
    ]:
        subprocess.check_call([sys.executable, str(script_root / step), "--root", str(root)], env=env)
    assert (root / "reports" / "PIPELINE_STATUS.md").exists() or (root / "PIPELINE_STATUS.md").exists()
    assert (root / "splits" / "train.jsonl").exists()
    stats = json.loads((root / "statistics" / "corpus_statistics.json").read_text(encoding="utf-8"))
    assert stats["document_count"] >= 1
    manifest = json.loads((root / "tokenizer_input" / "tokenizer_manifest.json").read_text(encoding="utf-8"))
    assert manifest["total_docs"] >= 1
