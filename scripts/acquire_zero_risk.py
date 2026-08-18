#!/usr/bin/env python3
"""
CognitiveOC v3 — Zero-Risk Corpus Acquisition Helper
======================================================

Downloads all zero-risk source families and places them in the warehouse
raw directories. Run AFTER setting CORPUS_WAREHOUSE_DIR in config.py.

Usage:
    cd cognitiveoc_v3/
    python scripts/acquire_zero_risk.py --category A
    python scripts/acquire_zero_risk.py --category C
    python scripts/acquire_zero_risk.py --all
    python scripts/acquire_zero_risk.py --list

Each acquisition step:
1. Downloads to the correct warehouse/raw/<category>/ directory
2. Does NOT register or pipeline the source — that is a separate step
3. Prints the exact register command to run next

Dependencies (install as needed):
    pip install datasets gutenberg requests tqdm
    pip install pdfminer.six ebooklib html2text mwparserfromhell
"""

import argparse
import json
import os
import pathlib
import sys
import textwrap

# ── Config ────────────────────────────────────────────────────────────
try:
    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
    from config import CORPUS_WAREHOUSE_DIR
    WHOUSE = pathlib.Path(CORPUS_WAREHOUSE_DIR)
except Exception as e:
    print(f"ERROR: Could not load config.py: {e}")
    print("Make sure you run from the cognitiveoc_v3/ directory.")
    sys.exit(1)


def _ensure(path: pathlib.Path) -> pathlib.Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _print_register_cmd(source_id: str, category: str, name: str,
                         url: str, licence: str, licence_id: str,
                         risk: float, raw_path: str) -> None:
    print(f"\n  Next step — register this source:")
    print(f"  python main.py corpus register-source {source_id} \\")
    print(f"    --category {category} --name \"{name}\" \\")
    print(f"    --url \"{url}\" \\")
    print(f"    --licence \"{licence}\" --licence-id {licence_id} \\")
    print(f"    --licence-risk {risk} \\")
    print(f"    --raw-path \"{raw_path}\" \\")
    print(f"    --operator mpssp")
    print(f"  python main.py corpus validate-source {source_id} --operator mpssp")
    print(f"  python main.py corpus run-pipeline {source_id} "
          f"--stages clean,dedup,score --verbose\n")


# ═══════════════════════════════════════════════════════════════════
# CATEGORY A — Books (Public Domain)
# ═══════════════════════════════════════════════════════════════════

def acquire_gutenberg_top():
    """
    Download the top Project Gutenberg English texts using rsync mirror.

    Project Gutenberg provides a full rsync mirror. We target English PD texts
    in plain-text format (.txt). This avoids the web scraping that their robots.txt
    discourages.

    Mirror: ftp.ibiblio.org::gutenberg/
    Docs: https://www.gutenberg.org/help/mirroring.html
    """
    out_dir = _ensure(WHOUSE / "raw" / "books" / "gutenberg")
    print("\n" + "=" * 60)
    print("CATEGORY A — Project Gutenberg (Public Domain)")
    print("=" * 60)
    print(f"Output directory: {out_dir}")
    print()
    print("Recommended acquisition command (run in a terminal):")
    print()
    print("  # Full mirror of all English PD plain-text files (~40 GB):")
    print("  rsync -av --del --include='*.txt' --exclude='*' \\")
    print(f"    ftp@ftp.ibiblio.org::gutenberg/cache/epub/ \\")
    print(f"    \"{out_dir}/\"")
    print()
    print("  # Or use the gutenberg Python package for selective download:")
    print("  pip install gutenberg")
    print("  python -c \"")
    print("    from gutenberg.acquire import load_etext")
    print("    from gutenberg.cleanup import strip_headers")
    print("    import pathlib")
    print(f"    out = pathlib.Path(r'{out_dir}')")
    print("    # Top 100 by download count (example IDs — expand to top 10000):")
    print("    for eid in [1342, 84, 11, 98, 1661, 2701, 174, 1952, 76, 46]:")
    print("        try:")
    print("            text = strip_headers(load_etext(eid)).strip()")
    print("            (out / f'{eid}.txt').write_text(text, encoding='utf-8')")
    print("            print(f'Downloaded {eid}')")
    print("        except Exception as e:")
    print("            print(f'Skip {eid}: {e}')")
    print("  \"")
    print()
    print("  Licence: PUBLIC DOMAIN — zero risk")
    _print_register_cmd(
        "A-gutenberg-v1", "A",
        "Project Gutenberg English PD Texts",
        "https://www.gutenberg.org",
        "Public Domain", "pd", 0.0, str(out_dir)
    )


def acquire_standard_ebooks():
    """Standard Ebooks — high-quality CC0 corrected versions."""
    out_dir = _ensure(WHOUSE / "raw" / "books" / "standard_ebooks")
    print("\n" + "=" * 60)
    print("CATEGORY A — Standard Ebooks (CC0)")
    print("=" * 60)
    print(f"Output directory: {out_dir}")
    print()
    print("  Standard Ebooks provides CC0 EPUB files for all ~600+ titles.")
    print("  Download from: https://standardebooks.org/ebooks")
    print()
    print("  Bulk download (Git clone — recommended):")
    print("  git clone https://github.com/standardebooks/tools.git se-tools")
    print("  python se-tools/se get \\")
    print(f"    --directory \"{out_dir}\" \\")
    print("    $(curl -s https://standardebooks.org/opds | grep '<id>' | \\")
    print("      grep -o 'url:https://standardebooks.org/ebooks/[^<]*' | \\")
    print("      sed 's/url://')")
    print()
    print("  After download, extract text with:")
    print("  pip install ebooklib html2text")
    print("  python -c \"")
    print("    import ebooklib, html2text, pathlib")
    print("    from ebooklib import epub")
    print(f"    epubs = list(pathlib.Path(r'{out_dir}').rglob('*.epub'))")
    print("    h = html2text.HTML2Text(); h.ignore_links = True")
    print("    for ep in epubs:")
    print("        try:")
    print("            book = epub.read_epub(str(ep))")
    print("            text = ' '.join(h.handle(")
    print("                item.get_content().decode('utf-8','replace'))")
    print("                for item in book.get_items_of_type(9))")
    print("            ep.with_suffix('.txt').write_text(text, encoding='utf-8')")
    print("        except: pass")
    print("  \"")
    print()
    print("  Licence: CC0 — zero risk")
    _print_register_cmd(
        "A-standard-ebooks-v1", "A",
        "Standard Ebooks Complete Catalogue",
        "https://standardebooks.org",
        "CC0 1.0", "cc0", 0.0, str(out_dir)
    )


# ═══════════════════════════════════════════════════════════════════
# CATEGORY B — Educational Content
# ═══════════════════════════════════════════════════════════════════

def acquire_openstax():
    """OpenStax textbooks — CC-BY 4.0."""
    out_dir = _ensure(WHOUSE / "raw" / "educational" / "openstax")
    print("\n" + "=" * 60)
    print("CATEGORY B — OpenStax Textbooks (CC-BY 4.0)")
    print("=" * 60)
    print(f"Output directory: {out_dir}")
    print()
    print("  OpenStax provides free PDF and HTML for all textbooks.")
    print("  Available titles: https://openstax.org/subjects")
    print()
    print("  Recommended: download HTML versions (cleaner for text extraction)")
    print("  Tool: pip install requests beautifulsoup4 html2text")
    print()

    openstax_titles = [
        ("College Algebra", "https://openstax.org/books/college-algebra/pages/1-introduction-to-prerequisites"),
        ("Calculus Volume 1", "https://openstax.org/books/calculus-volume-1/pages/1-introduction"),
        ("University Physics 1", "https://openstax.org/books/university-physics-volume-1/pages/1-introduction"),
        ("Biology 2e", "https://openstax.org/books/biology-2e/pages/1-introduction"),
        ("Chemistry Atoms First 2e", "https://openstax.org/books/chemistry-atoms-first-2e/pages/1-introduction"),
        ("Introduction to Sociology 3e", "https://openstax.org/books/introduction-sociology-3e/pages/1-introduction"),
        ("Psychology 2e", "https://openstax.org/books/psychology-2e/pages/1-introduction"),
        ("Principles of Economics 3e", "https://openstax.org/books/principles-economics-3e/pages/1-introduction"),
        ("Statistics", "https://openstax.org/books/introductory-statistics/pages/1-introduction"),
        ("Anatomy and Physiology", "https://openstax.org/books/anatomy-and-physiology-2e/pages/1-introduction"),
        ("US History", "https://openstax.org/books/us-history/pages/1-introduction"),
        ("Philosophy", "https://openstax.org/books/introduction-philosophy/pages/1-introduction"),
        ("Business Ethics", "https://openstax.org/books/business-ethics/pages/1-introduction"),
        ("Principles of Management", "https://openstax.org/books/principles-management/pages/1-introduction"),
        ("College Success", "https://openstax.org/books/college-success/pages/1-introduction"),
    ]

    print("  Available titles to download:")
    for title, url in openstax_titles:
        print(f"    • {title}")
    print()
    print("  All licenced CC-BY 4.0 — safe for release v1")
    print()
    print("  Download script:")
    print("  python -c \"")
    print("    import requests, html2text, pathlib")
    print(f"    out = pathlib.Path(r'{out_dir}')")
    for title, _ in openstax_titles[:3]:
        safe = title.lower().replace(' ', '_')
        print(f"    # {title} — download HTML from OpenStax and save as {safe}.txt")
    print("    # (Use the OpenStax GitHub repos for bulk text access:")
    print("    # github.com/openstax — each textbook has a repo with .cnxml content)")
    print("  \"")
    print()
    print("  Licence: CC-BY 4.0 — risk 0.10 — safe for release v1")
    _print_register_cmd(
        "B-openstax-v1", "B",
        "OpenStax Textbooks (CC-BY 4.0)",
        "https://openstax.org",
        "CC-BY 4.0", "cc-by-4", 0.10, str(out_dir)
    )


# ═══════════════════════════════════════════════════════════════════
# CATEGORY C — Reasoning / STEM (HuggingFace datasets)
# ═══════════════════════════════════════════════════════════════════

def acquire_reasoning_datasets():
    """
    Download all zero-risk reasoning / STEM datasets from HuggingFace.
    All are MIT / Apache 2.0 / CC-BY — zero risk for release v1.
    """
    out_dir = _ensure(WHOUSE / "raw" / "reasoning")
    print("\n" + "=" * 60)
    print("CATEGORY C — Reasoning / STEM Datasets")
    print("=" * 60)
    print(f"Output directory: {out_dir}")
    print()
    print("  All datasets below are MIT / Apache 2.0 / CC-BY — zero risk.")
    print()

    datasets_to_download = [
        # (hf_name, config, split, output_filename, licence, description)
        ("gsm8k", "main", "train", "gsm8k_train.jsonl", "MIT",
         "Grade school math with solutions (7473 problems)"),
        ("gsm8k", "main", "test", "gsm8k_test.jsonl", "MIT",
         "GSM8K test split (for evaluation only — move to holdout)"),
        ("AI-MO/NuminaMath-CoT", None, "train", "numinamath_train.jsonl", "Apache 2.0",
         "Competition math with chain-of-thought (859K examples)"),
        ("allenai/ai2_arc", "ARC-Challenge", "train", "arc_challenge_train.jsonl", "CC-BY 4.0",
         "Science reasoning challenge set (1119 examples)"),
        ("allenai/ai2_arc", "ARC-Easy", "train", "arc_easy_train.jsonl", "CC-BY 4.0",
         "Science reasoning easy set (2251 examples)"),
        ("EleutherAI/logiqa", None, "train", "logiqa_train.jsonl", "MIT",
         "Logical reasoning (7376 examples)"),
        ("lighteval/MATH", None, "train", "math_train.jsonl", "MIT",
         "Competition math with solutions (7500 examples)"),
        ("allenai/sciq", None, "train", "sciq_train.jsonl", "CC-BY 4.0",
         "Science QA (11679 examples)"),
        ("Rowan/hellaswag", None, "train", "hellaswag_train.jsonl", "MIT",
         "Commonsense NLI (39905 training, rest holdout)"),
        ("winogrande", "winogrande_xl", "train", "winogrande_train.jsonl", "CC-BY",
         "Commonsense reasoning (40938 examples)"),
        ("allenai/strategyqa", None, "train", "strategyqa_train.jsonl", "MIT",
         "Multi-step strategy reasoning (2290 examples)"),
    ]

    print("  Datasets to acquire:")
    for hf, cfg, split, outfile, lic, desc in datasets_to_download:
        print(f"    [{lic}] {hf} ({split}) → {outfile}")
        print(f"           {desc}")
    print()

    print("  Download script:")
    print("  pip install datasets")
    print("  python -c \"")
    print("    from datasets import load_dataset")
    print("    import json, pathlib")
    print(f"    out_dir = pathlib.Path(r'{out_dir}')")
    print("    datasets = [")
    for hf, cfg, split, outfile, lic, desc in datasets_to_download:
        cfg_str = f'"{cfg}"' if cfg else 'None'
        print(f"        ('{hf}', {cfg_str}, '{split}', '{outfile}'),")
    print("    ]")
    print("    for hf_name, cfg, split, outfile in datasets:")
    print("        try:")
    print("            ds = load_dataset(hf_name, cfg, split=split,")
    print("                             trust_remote_code=False)")
    print("            out_path = out_dir / outfile")
    print("            with open(out_path, 'w', encoding='utf-8') as f:")
    print("                for row in ds:")
    print("                    f.write(json.dumps(row, ensure_ascii=False) + chr(10))")
    print("            print(f'OK {outfile}: {len(ds)} rows')")
    print("        except Exception as e:")
    print("            print(f'FAIL {hf_name}: {e}')")
    print("  \"")
    _print_register_cmd(
        "C-reasoning-hf-v1", "C",
        "HuggingFace Reasoning / STEM Datasets (MIT/Apache/CC-BY)",
        "https://huggingface.co/datasets",
        "MIT/Apache 2.0/CC-BY 4.0", "mit", 0.05, str(out_dir)
    )


# ═══════════════════════════════════════════════════════════════════
# CATEGORY D — Conversations / Instruction
# ═══════════════════════════════════════════════════════════════════

def acquire_instruction_datasets():
    """Download zero-risk instruction datasets."""
    out_dir = _ensure(WHOUSE / "raw" / "conversations")
    print("\n" + "=" * 60)
    print("CATEGORY D — Instruction / Conversation Datasets")
    print("=" * 60)
    print(f"Output directory: {out_dir}")
    print()

    instruction_datasets = [
        ("databricks/databricks-dolly-15k", None, "train", "dolly_15k.jsonl",
         "CC-BY-SA 3.0", "15K human-written instruction examples"),
        ("Open-Orca/FLAN", None, "train", "flan_train.jsonl",
         "Apache 2.0", "FLAN formatted instruction data"),
        ("OpenAssistant/oasst2", None, "train", "oasst2_train.jsonl",
         "Apache 2.0", "Human conversation trees (English subset only)"),
    ]

    print("  IMPORTANT: OASST2 — filter to English only:")
    print("    ds = ds.filter(lambda x: x['lang'] == 'en')")
    print()
    print("  REJECTED (do not download for release v1):")
    print("    ✗ ShareGPT — unclear licence")
    print("    ✗ UltraChat — GPT-4 outputs, ToS violation")
    print("    ✗ Alpaca — CC-BY-NC, warehouse-only")
    print()
    print("  Download script:")
    print("  python -c \"")
    print("    from datasets import load_dataset")
    print("    import json, pathlib")
    print(f"    out_dir = pathlib.Path(r'{out_dir}')")
    for hf, cfg, split, outfile, lic, desc in instruction_datasets:
        cfg_str = f'"{cfg}"' if cfg else 'None'
        print(f"    # [{lic}] {desc}")
        print(f"    ds = load_dataset('{hf}', {cfg_str}, split='{split}')")
        if 'oasst' in hf:
            print("    ds = ds.filter(lambda x: x.get('lang','') == 'en')")
        print(f"    out = out_dir / '{outfile}'")
        print("    with open(out,'w') as f:")
        print("        for row in ds: f.write(json.dumps(row)+chr(10))")
        print(f"    print(f'OK: {{len(ds)}} rows')")
        print()
    print("  \"")
    _print_register_cmd(
        "D-instruction-hf-v1", "D",
        "Instruction Datasets (Dolly + FLAN + OASST2)",
        "https://huggingface.co/datasets",
        "CC-BY-SA 3.0 / Apache 2.0", "cc-by-sa-3", 0.20, str(out_dir)
    )


# ═══════════════════════════════════════════════════════════════════
# CATEGORY E — Technical Documentation (Wikipedia + Official docs)
# ═══════════════════════════════════════════════════════════════════

def acquire_wikipedia():
    """Wikipedia English dump — CC-BY-SA 3.0."""
    out_dir = _ensure(WHOUSE / "raw" / "technical_docs" / "wikipedia")
    print("\n" + "=" * 60)
    print("CATEGORY E — Wikipedia English (CC-BY-SA 3.0)")
    print("=" * 60)
    print(f"Output directory: {out_dir}")
    print()
    print("  WARNING: The full English Wikipedia dump is ~22 GB compressed.")
    print("  We filter to CS / math / science / linguistics articles only.")
    print("  This reduces volume by ~80% while keeping the highest-value content.")
    print()
    print("  Step 1: Download the dump")
    print("  wget -P \"{out_dir}\" \\")
    print("    https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2")
    print()
    print("  Step 2: Extract with wikiextractor (filters prose from wikitext)")
    print("  pip install wikiextractor")
    print("  python -m wikiextractor.WikiExtractor \\")
    print(f"    \"{out_dir}/enwiki-latest-pages-articles.xml.bz2\" \\")
    print(f"    --output \"{out_dir}/extracted/\" \\")
    print("    --bytes 10M --json --no-templates --filter_disambig_pages")
    print()
    print("  Step 3: Filter to target categories (CS/math/science/linguistics)")
    print("  python -c \"")
    print("    import json, pathlib, re")
    print("    # Categories to keep (partial match, case-insensitive)")
    print("    KEEP = ['computer', 'algorithm', 'mathematics', 'physics',")
    print("            'chemistry', 'biology', 'neuroscience', 'cognitive',")
    print("            'linguistics', 'statistics', 'logic', 'artificial']")
    print(f"    src = pathlib.Path(r'{out_dir}/extracted')")
    print(f"    out = pathlib.Path(r'{out_dir}/filtered')")
    print("    out.mkdir(exist_ok=True)")
    print("    kept = 0")
    print("    for f in src.rglob('*.jsonl'):")
    print("        with open(f) as fi, open(out/f.name,'w') as fo:")
    print("            for line in fi:")
    print("                d = json.loads(line)")
    print("                title = d.get('title','').lower()")
    print("                text = d.get('text','')")
    print("                if (len(text.split()) > 200 and")
    print("                    any(k in title for k in KEEP)):")
    print("                    fo.write(line)")
    print("                    kept += 1")
    print("    print(f'Kept {kept} articles')")
    print("  \"")
    print()
    print("  Licence: CC-BY-SA 3.0 — risk 0.20 — safe for release v1")
    _print_register_cmd(
        "E-wikipedia-cs-v1", "E",
        "Wikipedia English CS/Math/Science Articles (CC-BY-SA 3.0)",
        "https://dumps.wikimedia.org/enwiki/",
        "CC-BY-SA 3.0", "cc-by-sa-3", 0.20,
        str(out_dir / "filtered")
    )


def acquire_official_docs():
    """Python, PyTorch, HuggingFace official documentation."""
    out_dir = _ensure(WHOUSE / "raw" / "technical_docs" / "official_docs")
    print("\n" + "=" * 60)
    print("CATEGORY E — Official Technical Documentation")
    print("=" * 60)
    print(f"Output directory: {out_dir}")
    print()
    print("  Python stdlib docs (PSF — permissive):")
    print("  git clone --depth 1 https://github.com/python/cpython.git /tmp/cpython")
    print(f"  cp -r /tmp/cpython/Doc/ \"{out_dir}/python_docs/\"")
    print("  pip install sphinx")
    print(f"  cd /tmp/cpython/Doc && make text BUILDDIR=\"{out_dir}/python_docs_text\"")
    print()
    print("  PyTorch docs (BSD-3):")
    print("  git clone --depth 1 https://github.com/pytorch/pytorch.git /tmp/pytorch")
    print(f"  cp -r /tmp/pytorch/docs/ \"{out_dir}/pytorch_docs/\"")
    print()
    print("  HuggingFace docs (Apache 2.0):")
    print("  git clone --depth 1 https://github.com/huggingface/transformers.git /tmp/hf")
    print(f"  cp -r /tmp/hf/docs/ \"{out_dir}/hf_docs/\"")
    print()
    print("  Licences: PSF / BSD-3 / Apache 2.0 — zero risk")
    _print_register_cmd(
        "E-official-docs-v1", "E",
        "Python + PyTorch + HuggingFace Official Docs",
        "https://docs.python.org",
        "PSF/BSD-3/Apache 2.0", "psf", 0.05, str(out_dir)
    )


# ═══════════════════════════════════════════════════════════════════
# CATEGORY G — Research Papers (ArXiv CC-BY)
# ═══════════════════════════════════════════════════════════════════

def acquire_arxiv():
    """ArXiv papers filtered to CC-BY licence only."""
    out_dir = _ensure(WHOUSE / "raw" / "research_papers" / "arxiv")
    print("\n" + "=" * 60)
    print("CATEGORY G — ArXiv Research Papers (CC-BY 4.0 only)")
    print("=" * 60)
    print(f"Output directory: {out_dir}")
    print()
    print("  ArXiv provides S3 bulk access to paper metadata.")
    print("  We filter to CC-BY licenced papers ONLY.")
    print()
    print("  Step 1: Download metadata (no AWS account needed for metadata)")
    print("  pip install requests")
    print("  python -c \"")
    print("    # Use ArXiv API to fetch CC-BY paper metadata")
    print("    # https://info.arxiv.org/help/api/index.html")
    print("    import requests, json, pathlib, time")
    print(f"    out = pathlib.Path(r'{out_dir}')")
    print("    # Categories: cs.*, math.*, stat.*, q-bio.NC, q-bio.QM")
    print("    categories = ['cs.AI','cs.LG','cs.CL','cs.NE','cs.IR',")
    print("                   'math.ST','math.LO','stat.ML','q-bio.NC']")
    print("    base = 'http://export.arxiv.org/api/query'")
    print("    for cat in categories:")
    print("        params = {'search_query': f'cat:{cat}',")
    print("                  'max_results': 10000, 'sortBy': 'lastUpdatedDate'}")
    print("        r = requests.get(base, params=params, timeout=60)")
    print("        (out / f'meta_{cat.replace(\".\"  ,\"_\")}.xml').write_bytes(r.content)")
    print("        print(f'OK {cat}'); time.sleep(3)")
    print("  \"")
    print()
    print("  Step 2: Filter to CC-BY papers and download PDFs")
    print("  # Parse XML metadata, filter licence='http://creativecommons.org/licenses/by/4.0/'")
    print("  # Then download PDFs from https://arxiv.org/pdf/<id>.pdf")
    print("  # Extract text with: pip install pdfminer.six")
    print("  # from pdfminer.high_level import extract_text")
    print()
    print("  Licence: CC-BY 4.0 — risk 0.10 — safe for release v1")
    print("  IMPORTANT: Verify per-paper licence from metadata before including.")
    _print_register_cmd(
        "G-arxiv-ccby-v1", "G",
        "ArXiv CS/Math/Stat Papers (CC-BY 4.0 only)",
        "https://arxiv.org",
        "CC-BY 4.0", "cc-by-4", 0.10, str(out_dir)
    )


def acquire_acl_anthology():
    """ACL Anthology — CC-BY 4.0 NLP papers."""
    out_dir = _ensure(WHOUSE / "raw" / "research_papers" / "acl_anthology")
    print("\n" + "=" * 60)
    print("CATEGORY G — ACL Anthology (CC-BY 4.0)")
    print("=" * 60)
    print(f"Output directory: {out_dir}")
    print()
    print("  ACL Anthology provides bulk access to NLP/CL research papers.")
    print("  https://aclanthology.org/anthology+abstracts.bib")
    print()
    print("  Step 1: Download metadata")
    print("  wget -P \"{out_dir}\" \\")
    print("    https://aclanthology.org/anthology+abstracts.bib")
    print()
    print("  Step 2: Download paper PDFs in bulk")
    print("  wget -P \"{out_dir}/pdfs/\" \\")
    print("    https://aclanthology.org/anthology.bib.gz")
    print("  # Or use the ACL Anthology Corpus downloader:")
    print("  # pip install acl-anthology")
    print()
    print("  Licence: CC-BY 4.0 — risk 0.10 — safe for release v1")
    _print_register_cmd(
        "G-acl-anthology-v1", "G",
        "ACL Anthology NLP Papers (CC-BY 4.0)",
        "https://aclanthology.org",
        "CC-BY 4.0", "cc-by-4", 0.10, str(out_dir)
    )


# ═══════════════════════════════════════════════════════════════════
# CATEGORIES J + K — Retrieval + KG
# ═══════════════════════════════════════════════════════════════════

def acquire_retrieval_kg():
    """MS MARCO, NQ, TriviaQA, DBpedia, ConceptNet, Wikidata."""
    out_j = _ensure(WHOUSE / "raw" / "retrieval")
    out_k = _ensure(WHOUSE / "raw" / "kg")
    print("\n" + "=" * 60)
    print("CATEGORIES J+K — Retrieval + KG Datasets")
    print("=" * 60)

    retrieval_datasets = [
        ("ms_marco", "v2.1", "train", "msmarco_train.jsonl", out_j, "MIT"),
        ("natural_questions", None, "train", "nq_train.jsonl", out_j, "CC-BY-SA 3.0"),
        ("trivia_qa", "rc", "train", "triviaqa_train.jsonl", out_j, "Apache 2.0"),
        ("hotpot_qa", "distractor", "train", "hotpotqa_train.jsonl", out_j, "CC-BY-SA 4.0"),
    ]
    kg_datasets = [
        ("conceptnet5/conceptnet5", None, "train", "conceptnet_train.jsonl", out_k, "CC-BY-SA 4.0"),
    ]

    print(f"\n  Category J (Retrieval) → {out_j}")
    print(f"  Category K (KG) → {out_k}")
    print()
    print("  Download script:")
    print("  pip install datasets")
    print("  python -c \"")
    print("    from datasets import load_dataset")
    print("    import json")

    for hf, cfg, split, outfile, out_dir, lic in retrieval_datasets + kg_datasets:
        cfg_str = f'"{cfg}"' if cfg else 'None'
        print(f"    # [{lic}] {hf}")
        print(f"    try:")
        print(f"      ds = load_dataset('{hf}', {cfg_str}, split='{split}', trust_remote_code=False)")
        print(f"      with open(r'{out_dir}/{outfile}','w') as f:")
        print(f"        for row in ds: f.write(json.dumps(row)+chr(10))")
        print(f"      print('OK {outfile}: {{len(ds)}} rows')")
        print(f"    except Exception as e: print('FAIL {hf}:', e)")

    print()
    print("  Wikidata text (CC0):")
    print("  wget -P \"{out_k}/wikidata/\" \\")
    print("    https://dumps.wikimedia.org/wikidatawiki/latest/")
    print("    ← Download wikidatawiki-latest-pages-articles.xml.bz2")
    print("  Then extract entity descriptions using wikiextractor")
    print()
    print("  DBpedia abstracts (CC-BY-SA 3.0):")
    print("  wget -P \"{out_k}/dbpedia/\" \\")
    print("    https://databus.dbpedia.org/dbpedia/text/short-abstracts/2022.09.01/short-abstracts_lang=en.ttl.bz2")
    print()
    _print_register_cmd("J-retrieval-hf-v1", "J", "Retrieval Datasets (MIT/Apache/CC-BY-SA)",
                        "https://huggingface.co/datasets", "MIT/Apache 2.0/CC-BY-SA", "mit", 0.10, str(out_j))
    _print_register_cmd("K-kg-hf-v1", "K", "KG Datasets (ConceptNet + Wikidata + DBpedia)",
                        "https://huggingface.co/datasets", "CC-BY-SA 4.0 / CC0", "cc-by-sa-4", 0.20, str(out_k))


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════




def acquire_articles():
    """ArXiv abstracts+intros (CC-BY) + Wikinews (CC-BY 2.5)."""
    out_dir = _ensure(WHOUSE / "raw" / "articles")
    print("\n" + "=" * 60)
    print("CATEGORY F — Long-Form Articles")
    print("=" * 60)
    print(f"Output directory: {out_dir}")
    print()
    print("  Source 1: ArXiv abstracts + introductions (CC-BY papers only)")
    print("  Rationale: Same ArXiv papers as Category G, but here we use")
    print("  ONLY the abstract+intro for non-CC-BY papers (abstract is always open).")
    print()
    print("  python -c \"")
    print("    # Use ArXiv OAI-PMH API to get abstracts for all papers")
    print("    # https://info.arxiv.org/help/oa/index.html")
    print("    import urllib.request, xml.etree.ElementTree as ET")
    print("    import pathlib, time, json")
    print(f"    out = pathlib.Path(r'{out_dir}') / 'arxiv_abstracts.jsonl'")
    print("    base = 'https://export.arxiv.org/oai2?verb=ListRecords'")
    print("    params = '&metadataPrefix=arXiv&set=cs'")
    print("    url = base + params")
    print("    # Continue fetching with resumptionToken until exhausted")
    print("    # This yields all CS ArXiv abstracts (no licence restriction on abstracts)")
    print("    with open(out, 'w') as f:")
    print("        # Parse XML, extract: id, title, abstract, licence")
    print("        pass  # implement with your preferred XML parser")
    print("  \"")
    print()
    print("  Source 2: Wikinews English (CC-BY 2.5)")
    print("  wget -P \"{out_dir}/wikinews/\" \\")
    print("    https://dumps.wikimedia.org/enwikinews/latest/")
    print("    ← Download enwikinews-latest-pages-articles.xml.bz2")
    print("  python -m wikiextractor.WikiExtractor \\")
    print(f"    \"{out_dir}/wikinews/enwikinews-latest-pages-articles.xml.bz2\" \\")
    print(f"    --output \"{out_dir}/wikinews/extracted/\" --json")
    print()
    print("  Licence: CC-BY 4.0 (ArXiv abstracts) + CC-BY 2.5 (Wikinews)")
    _print_register_cmd(
        "F-articles-v1", "F",
        "ArXiv Abstracts + Wikinews (CC-BY)",
        "https://arxiv.org + https://en.wikinews.org",
        "CC-BY 4.0 / CC-BY 2.5", "cc-by-4", 0.10, str(out_dir)
    )


# ═══════════════════════════════════════════════════════════════════
# CATEGORY G — PubMed Open Access (CC-BY subset)
# ═══════════════════════════════════════════════════════════════════

def acquire_pubmed():
    """PubMed Open Access CC-BY papers."""
    out_dir = _ensure(WHOUSE / "raw" / "research_papers" / "pubmed_oa")
    print("\n" + "=" * 60)
    print("CATEGORY G — PubMed Open Access CC-BY Papers")
    print("=" * 60)
    print(f"Output directory: {out_dir}")
    print()
    print("  PubMed Central provides FTP bulk access to Open Access papers.")
    print("  We use ONLY the CC-BY licensed subset (oa_comm).")
    print()
    print("  Step 1: Download the CC-BY file list")
    print("  wget https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_comm/txt/")
    print("  # Lists all CC-BY licensed articles in plain text format")
    print()
    print("  Step 2: Download all CC-BY article tarballs")
    print("  wget -r -l1 -nd -A '*.tar.gz' \\")
    print("    https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_comm/txt/ \\")
    print(f"    -P \"{out_dir}/\"")
    print()
    print("  Step 3: Extract and process")
    print("  python -c \"")
    print("    import tarfile, pathlib, json")
    print(f"    src = pathlib.Path(r'{out_dir}')")
    print("    for tf in src.glob('*.tar.gz'):")
    print("        with tarfile.open(tf) as t:")
    print("            t.extractall(src / 'extracted')")
    print("  \"")
    print()
    print("  Licence: CC-BY 4.0 — risk 0.10 — safe for release v1")
    print("  Focus areas: neuroscience, cognitive science, clinical psychology")
    _print_register_cmd(
        "G-pubmed-oa-v1", "G",
        "PubMed Open Access CC-BY Papers",
        "https://www.ncbi.nlm.nih.gov/pmc/tools/ftp/",
        "CC-BY 4.0", "cc-by-4", 0.10, str(out_dir)
    )


# ═══════════════════════════════════════════════════════════════════
# CATEGORY I — Human Cognition Material
# ═══════════════════════════════════════════════════════════════════

def acquire_cognition():
    """PG psychology classics + OpenStax Psychology."""
    out_dir = _ensure(WHOUSE / "raw" / "cognition")
    print("\n" + "=" * 60)
    print("CATEGORY I — Human Cognition Material")
    print("=" * 60)
    print(f"Output directory: {out_dir}")
    print()
    print("  Source 1: Project Gutenberg Psychology Classics (Public Domain)")
    print("  Key texts and their PG IDs:")
    pg_psych = [
        (621,   "William James — The Principles of Psychology Vol.1"),
        (622,   "William James — The Principles of Psychology Vol.2"),
        (11501, "John Dewey — How We Think"),
        (16287, "Sigmund Freud — Dream Psychology"),
        (15040, "Carl Jung — Psychology of the Unconscious"),
        (9846,  "G. Stanley Hall — Adolescence Vol.1"),
        (20673, "John B. Watson — Behaviorism"),
        (15776, "William McDougall — An Introduction to Social Psychology"),
        (4129,  "William James — Talks to Teachers on Psychology"),
        (948,   "Wilhelm Wundt — Outlines of Psychology"),
    ]
    for eid, title in pg_psych:
        print(f"    PG #{eid:>5} — {title}")
    print()
    print("  Download:")
    print("  pip install gutenberg")
    print("  python -c \"")
    print("    from gutenberg.acquire import load_etext")
    print("    from gutenberg.cleanup import strip_headers")
    print("    import pathlib")
    print(f"    out = pathlib.Path(r'{out_dir}/gutenberg_psychology')")
    print("    out.mkdir(parents=True, exist_ok=True)")
    ids = [str(x[0]) for x in pg_psych]
    print(f"    for eid in [{', '.join(ids)}]:")
    print("        try:")
    print("            text = strip_headers(load_etext(eid)).strip()")
    print("            (out / f'{eid}.txt').write_text(text, encoding='utf-8')")
    print("            print(f'Downloaded PG #{eid}')")
    print("        except Exception as e:")
    print("            print(f'Skip {eid}: {e}')")
    print("  \"")
    print()
    print("  Source 2: OpenStax Psychology 2e (CC-BY 4.0)")
    print("  git clone https://github.com/openstax/psychology-2e.git /tmp/psych2e")
    print(f"  cp -r /tmp/psych2e/modules/ \"{out_dir}/openstax_psychology/\"")
    print("  # Convert .cnxml files to plain text:")
    print("  python -c \"")
    print("    import xml.etree.ElementTree as ET, pathlib")
    print(f"    src = pathlib.Path(r'{out_dir}/openstax_psychology')")
    print(f"    out_txt = pathlib.Path(r'{out_dir}/openstax_psychology_text')")
    print("    out_txt.mkdir(exist_ok=True)")
    print("    for f in src.rglob('*.cnxml'):")
    print("        try:")
    print("            tree = ET.parse(f)")
    print("            # Extract text from para elements")
    print("            paras = [el.text or '' for el in tree.iter() if 'para' in el.tag.lower()]")
    print("            text = chr(10).join(p.strip() for p in paras if len(p.strip()) > 20)")
    print("            (out_txt / f.stem).with_suffix('.txt').write_text(text, encoding='utf-8')")
    print("        except: pass")
    print("  \"")
    print()
    print("  Source 3: CogSci Proceedings papers (CC-BY)")
    print("  Available at: https://escholarship.org/uc/cognitivesciencesociety")
    print("  Download via: wget or institutional bulk download")
    print()
    print("  Licence: PD (PG texts) + CC-BY 4.0 (OpenStax) — zero risk")
    _print_register_cmd(
        "I-cognition-v1", "I",
        "Human Cognition Material (PG psychology + OpenStax Psych)",
        "https://gutenberg.org + https://github.com/openstax/psychology-2e",
        "Public Domain / CC-BY 4.0", "pd", 0.05, str(out_dir)
    )


ACQUIRERS = {
    "A-gutenberg":      (acquire_gutenberg_top, "A"),
    "A-standard-ebooks":(acquire_standard_ebooks, "A"),
    "B-openstax":       (acquire_openstax, "B"),
    "C-reasoning":      (acquire_reasoning_datasets, "C"),
    "D-instruction":    (acquire_instruction_datasets, "D"),
    "E-wikipedia":      (acquire_wikipedia, "E"),
    "E-docs":           (acquire_official_docs, "E"),
    "F-articles":       (acquire_articles, "F"),
    "G-arxiv":          (acquire_arxiv, "G"),
    "G-acl":            (acquire_acl_anthology, "G"),
    "G-pubmed":         (acquire_pubmed, "G"),
    "I-cognition":      (acquire_cognition, "I"),
    "JK-retrieval-kg":  (acquire_retrieval_kg, "JK"),
}

def main():
    parser = argparse.ArgumentParser(
        description="CognitiveOC v3 — Zero-Risk Corpus Acquisition Helper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Examples:
              python scripts/acquire_zero_risk.py --list
              python scripts/acquire_zero_risk.py --source A-gutenberg
              python scripts/acquire_zero_risk.py --category C
              python scripts/acquire_zero_risk.py --all
        """)
    )
    parser.add_argument("--list", action="store_true",
                        help="List all available acquisition targets")
    parser.add_argument("--all", action="store_true",
                        help="Print instructions for all sources")
    parser.add_argument("--source", metavar="NAME",
                        help="Specific source to acquire (see --list)")
    parser.add_argument("--category", metavar="A-L",
                        help="Acquire all sources for a category")
    args = parser.parse_args()

    print(f"\nCognitiveOC v3 — Zero-Risk Corpus Acquisition Helper")
    print(f"Warehouse: {WHOUSE}")
    print(f"Zero-risk policy: licence_risk ≤ 0.20 for release v1")

    if args.list:
        print("\nAvailable acquisition targets:")
        for name, (fn, cat) in ACQUIRERS.items():
            print(f"  {name:<30} [Category {cat}]")
        print()
        return

    if args.all:
        for name, (fn, _) in ACQUIRERS.items():
            fn()
        return

    if args.source:
        if args.source in ACQUIRERS:
            ACQUIRERS[args.source][0]()
        else:
            print(f"Unknown source: {args.source}. Use --list to see options.")
            sys.exit(1)
        return

    if args.category:
        cat = args.category.upper()
        found = False
        for name, (fn, c) in ACQUIRERS.items():
            if cat in c:
                fn()
                found = True
        if not found:
            print(f"No acquirers for category {cat}.")
            sys.exit(1)
        return

    parser.print_help()



ACQUIRERS = {
    "A-gutenberg":      (acquire_gutenberg_top, "A"),
    "A-standard-ebooks":(acquire_standard_ebooks, "A"),
    "B-openstax":       (acquire_openstax, "B"),
    "C-reasoning":      (acquire_reasoning_datasets, "C"),
    "D-instruction":    (acquire_instruction_datasets, "D"),
    "E-wikipedia":      (acquire_wikipedia, "E"),
    "E-docs":           (acquire_official_docs, "E"),
    "F-articles":       (acquire_articles, "F"),
    "G-arxiv":          (acquire_arxiv, "G"),
    "G-acl":            (acquire_acl_anthology, "G"),
    "G-pubmed":         (acquire_pubmed, "G"),
    "I-cognition":      (acquire_cognition, "I"),
    "JK-retrieval-kg":  (acquire_retrieval_kg, "JK"),
}


if __name__ == "__main__":
    main()
