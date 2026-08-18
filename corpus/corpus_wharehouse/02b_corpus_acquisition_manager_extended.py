#!/usr/bin/env python3
"""
CognitiveOC - Extended Corpus Acquisition Manager (Gap-Filling Edition)
========================================================================
Drop-in companion to 02_corpus_acquisition_manager.py.

This script ONLY contains the *new* sources that were missing from the
original registry. It reuses the same patterns (SourceSpec, acquire_hf,
acquire_http, manifests, logs) so the two scripts behave identically
and write into the same warehouse directories.

Run AFTER 02_corpus_acquisition_manager.py has done its pass.

Usage:
    # List all new sources
    python 02b_corpus_acquisition_manager_extended.py --list

    # Acquire all priority-1 missing sources (safest, smallest first)
    python 02b_corpus_acquisition_manager_extended.py --all --priority 1 --skip-large

    # Only specific categories (e.g. fill empty E, F, I, K, M categories)
    python 02b_corpus_acquisition_manager_extended.py --all --categories E F I K M

    # Only specific keys
    python 02b_corpus_acquisition_manager_extended.py --only wikipedia_simple python_docs

    # Include large downloads (Wikipedia, Wikidata)
    python 02b_corpus_acquisition_manager_extended.py --all --priority 1

Install:
    pip install -U datasets huggingface_hub requests pyarrow tqdm
"""

from __future__ import annotations
import argparse, hashlib, json, os, re, sys, time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests
try:
    from datasets import DatasetDict, load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

# =====================================================================
# Config (identical to 02_*.py for warehouse compatibility)
# =====================================================================
DEFAULT_WAREHOUSE = Path(os.environ.get(
    "COC_WAREHOUSE_DIR",
    r"D:\projects\CognitiveOC\Final_Versions\cognitiveoc_v3\corpus\corpus_wharehouse",
))
USER_AGENT = "CognitiveOC-CorpusAcquisition-Ext/2.0"

CATEGORY_DIRS = {
    "A": "A_books", "B": "B_educational", "C": "C_reasoning",
    "D": "D_conversations", "E": "E_technical_docs", "F": "F_articles",
    "G": "G_research_papers", "H": "H_synthetic", "I": "I_cognition",
    "J": "J_retrieval", "K": "K_knowledge_graph", "L": "L_language_resources",
    "M": "M_legal_government", "N": "N_evaluation",
}

# Sources that are large (>2 GB) or extremely large (>20 GB)
LARGE_SOURCES = {
    "wikipedia_en_full", "wikidata_truthy_full", "openalex_works",
    "s2orc_odc", "pmc_oa_bulk", "the_stack_v2_smol", "fineweb_edu_sample",
    "redpajama_arxiv", "stackexchange_archive", "cc_news",
    "tulu3_sft", "aya_collection", "infinity_instruct",
}

# =====================================================================
# Extended Source Registry
# =====================================================================
@dataclass(frozen=True)
class SourceSpec:
    key: str
    kind: str                 # "hf" | "http" | "hf_warehouse_only" | "http_warehouse_only"
    category: str             # A..N
    subdir: str
    license_name: str
    license_risk: float
    notes: str
    priority: int = 2
    hf_id: Optional[str] = None
    hf_config: Optional[str] = None
    hf_split: Optional[str] = None
    url: Optional[str] = None
    extra_urls: tuple = ()
    warehouse_only: bool = False
    estimated_tokens: str = "unknown"


# ---------------------------------------------------------------------
# THE GAP-FILLING REGISTRY
# Every entry below is a *new* source not present in the original
# 02_corpus_acquisition_manager.py. License risk <= 0.2 unless noted.
# ---------------------------------------------------------------------
EXTENDED_SOURCES: list[SourceSpec] = [

    # =================================================================
    # A. BOOKS  -- expand from ~10 GB toward 25 B target
    # =================================================================
    SourceSpec(
        "standard_ebooks", "http", "A", "standard_ebooks",
        "Public Domain / CC0", 0.0,
        "Standard Ebooks OPDS feed; curated public-domain books.", 1,
        url="https://standardebooks.org/feeds/opds/all",
        estimated_tokens="~500M",
    ),
    SourceSpec(
        "gutenberg_au_index", "http", "A", "gutenberg_au",
        "Public Domain (AU)", 0.1,
        "Project Gutenberg Australia titles index.", 2,
        url="https://gutenberg.net.au/titles-a-m.html",
        extra_urls=("https://gutenberg.net.au/titles-n-z.html",),
        estimated_tokens="~500M",
    ),
    SourceSpec(
        "gutenberg_ca_index", "http", "A", "gutenberg_ca",
        "Public Domain (CA)", 0.1,
        "Project Gutenberg Canada title index.", 2,
        url="http://gutenberg.ca/index.html",
        estimated_tokens="~300M",
    ),
    SourceSpec(
        "wikisource_en_dump", "http", "A", "wikisource",
        "CC-BY-SA-4.0 + GFDL", 0.1,
        "English Wikisource articles dump (~600 MB).", 1,
        url="https://dumps.wikimedia.org/enwikisource/latest/enwikisource-latest-pages-articles.xml.bz2",
        estimated_tokens="~1B",
    ),
    SourceSpec(
        "libritts_text", "http", "A", "libritts_text",
        "Public Domain (LibriVox)", 0.0,
        "LibriTTS transcripts derived from LibriVox PD audiobooks.", 2,
        url="https://www.openslr.org/resources/60/train-clean-100.tar.gz",
        estimated_tokens="~50M",
    ),

    # =================================================================
    # B. EDUCATIONAL  -- expand toward 10 B target
    # =================================================================
    SourceSpec(
        "wikiversity_en_dump", "http", "B", "wikiversity",
        "CC-BY-SA-4.0", 0.0,
        "Wikiversity educational content dump.", 1,
        url="https://dumps.wikimedia.org/enwikiversity/latest/enwikiversity-latest-pages-articles.xml.bz2",
        estimated_tokens="~80M",
    ),
    SourceSpec(
        "wikibooks_en_dump", "http", "B", "wikibooks_dump",
        "CC-BY-SA-4.0", 0.0,
        "Full English Wikibooks XML dump.", 1,
        url="https://dumps.wikimedia.org/enwikibooks/latest/enwikibooks-latest-pages-articles.xml.bz2",
        estimated_tokens="~300M",
    ),
    SourceSpec(
        "openstax_search_extra", "http", "B", "openstax_extra",
        "CC-BY-4.0", 0.0,
        "Additional OpenStax book seed URLs (concrete books).", 1,
        url="https://openstax.org/books/college-physics-2e",
        extra_urls=(
            "https://openstax.org/books/chemistry-2e",
            "https://openstax.org/books/biology-2e",
            "https://openstax.org/books/microbiology",
            "https://openstax.org/books/anatomy-and-physiology-2e",
            "https://openstax.org/books/psychology-2e",
            "https://openstax.org/books/introduction-sociology-3e",
            "https://openstax.org/books/principles-economics-3e",
            "https://openstax.org/books/calculus-volume-1",
            "https://openstax.org/books/calculus-volume-2",
            "https://openstax.org/books/calculus-volume-3",
            "https://openstax.org/books/precalculus-2e",
            "https://openstax.org/books/elementary-algebra-2e",
            "https://openstax.org/books/intermediate-algebra-2e",
            "https://openstax.org/books/college-algebra-2e",
            "https://openstax.org/books/introductory-statistics-2e",
            "https://openstax.org/books/introductory-business-statistics-2e",
            "https://openstax.org/books/world-history-volume-1",
            "https://openstax.org/books/world-history-volume-2",
            "https://openstax.org/books/us-history",
            "https://openstax.org/books/astronomy-2e",
        ),
        estimated_tokens="~30M",
    ),
    SourceSpec(
        "ck12_search", "http", "B", "ck12",
        "CC-BY-NC-3.0", 0.3,
        "CK-12 FlexBooks landing page (NC license -- warehouse only).", 3,
        url="https://www.ck12.org/cbrowse/",
        warehouse_only=True,
        estimated_tokens="~200M",
    ),
    SourceSpec(
        "saylor_courses", "http", "B", "saylor",
        "CC-BY-3.0 (mixed)", 0.1,
        "Saylor Academy course catalog seed.", 2,
        url="https://learn.saylor.org/course/index.php",
        estimated_tokens="~100M",
    ),
    SourceSpec(
        "oer_commons_search", "http", "B", "oer_commons",
        "Mixed CC (filter to BY/BY-SA)", 0.2,
        "OERCommons search index (license-filter downstream).", 2,
        url="https://www.oercommons.org/oer",
        estimated_tokens="~1B",
    ),

    # =================================================================
    # C. REASONING / STEM  -- add high-quality math/reasoning datasets
    # =================================================================
    SourceSpec(
        "math_qa_aqua", "hf", "C", "aqua_rat",
        "Apache-2.0", 0.0,
        "AQuA-RAT algebraic word problems with rationales.", 2,
        hf_id="deepmind/aqua_rat", hf_config="raw",
        estimated_tokens="~30M",
    ),
    SourceSpec(
        "math_qa_general", "hf", "C", "math_qa",
        "Apache-2.0", 0.0,
        "MathQA reasoning dataset.", 2,
        hf_id="allenai/math_qa",
        estimated_tokens="~20M",
    ),
    SourceSpec(
        "tabmwp", "hf", "C", "tabmwp",
        "CC-BY-4.0", 0.0,
        "Table-based math word problems (TabMWP).", 2,
        hf_id="UCLA-AGI/TabMWP",
        estimated_tokens="~10M",
    ),
    SourceSpec(
        "logiqa", "hf", "C", "logiqa",
        "CC-BY-SA-4.0", 0.1,
        "LogiQA logical reasoning dataset.", 2,
        hf_id="lucasmccabe/logiqa",
        estimated_tokens="~3M",
    ),
    SourceSpec(
        "open_orca_subset", "hf", "C", "open_orca",
        "MIT", 0.1,
        "OpenOrca: reasoning traces (1M sample subset).", 2,
        hf_id="Open-Orca/OpenOrca",
        estimated_tokens="~1B",
    ),

    # =================================================================
    # D. CONVERSATIONS / INSTRUCTION  -- expand toward 15 B target
    # =================================================================
    SourceSpec(
        "tulu3_sft", "hf", "D", "tulu3_sft",
        "ODC-By-1.0", 0.1,
        "Allen AI Tulu-3 SFT mixture: high-quality instruction data.", 1,
        hf_id="allenai/tulu-3-sft-mixture",
        estimated_tokens="~1.5B",
    ),
    SourceSpec(
        "aya_collection", "hf", "D", "aya_collection",
        "Apache-2.0", 0.0,
        "Cohere Aya: multilingual instruction collection (English subset).", 2,
        hf_id="CohereForAI/aya_collection_language_split",
        hf_config="english",
        estimated_tokens="~3B",
    ),
    SourceSpec(
        "no_robots", "hf", "D", "no_robots",
        "CC-BY-NC-4.0", 0.3,
        "HuggingFace H4 No-Robots (NC -- warehouse-only).", 3,
        hf_id="HuggingFaceH4/no_robots",
        warehouse_only=True,
        estimated_tokens="~30M",
    ),
    SourceSpec(
        "flan_v2_subset", "hf", "D", "flan_v2",
        "Apache-2.0", 0.0,
        "FLAN v2 small instruction collection.", 1,
        hf_id="Open-Orca/FLAN",
        estimated_tokens="~5B",
    ),
    SourceSpec(
        "wildchat_1m", "hf", "D", "wildchat",
        "ODC-By-1.0", 0.2,
        "WildChat 1M real user conversations (filter PII downstream).", 2,
        hf_id="allenai/WildChat-1M",
        estimated_tokens="~1B",
    ),
    SourceSpec(
        "infinity_instruct", "hf", "D", "infinity_instruct",
        "CC-BY-SA-4.0", 0.1,
        "BAAI Infinity-Instruct: large instruction dataset.", 2,
        hf_id="BAAI/Infinity-Instruct",
        hf_config="7M",
        estimated_tokens="~2B",
    ),

    # =================================================================
    # E. TECHNICAL DOCUMENTATION  -- CRITICAL EMPTY CATEGORY
    # Strategy: download docs as static HTML/MD bundles via official
    # release archives (zip/tar) where available, plus README/source.
    # =================================================================
    SourceSpec(
        "python_docs", "http", "E", "python_docs",
        "PSF License (BSD-compatible)", 0.0,
        "Python 3 official documentation (HTML archive).", 1,
        url="https://docs.python.org/3/archives/python-3.12.7-docs-html.zip",
        estimated_tokens="~30M",
    ),
    SourceSpec(
        "numpy_docs", "http", "E", "numpy_docs",
        "BSD-3-Clause", 0.0,
        "NumPy documentation seed (latest stable).", 1,
        url="https://numpy.org/doc/stable/numpy-html.zip",
        extra_urls=(
            "https://numpy.org/doc/stable/",
            "https://numpy.org/doc/stable/user/index.html",
            "https://numpy.org/doc/stable/reference/index.html",
        ),
        estimated_tokens="~20M",
    ),
    SourceSpec(
        "pandas_docs", "http", "E", "pandas_docs",
        "BSD-3-Clause", 0.0,
        "Pandas documentation seed.", 1,
        url="https://pandas.pydata.org/docs/",
        extra_urls=(
            "https://pandas.pydata.org/docs/user_guide/index.html",
            "https://pandas.pydata.org/docs/reference/index.html",
        ),
        estimated_tokens="~25M",
    ),
    SourceSpec(
        "pytorch_docs", "http", "E", "pytorch_docs",
        "BSD-3-Clause", 0.0,
        "PyTorch official documentation seed.", 1,
        url="https://pytorch.org/docs/stable/index.html",
        extra_urls=(
            "https://pytorch.org/docs/stable/torch.html",
            "https://pytorch.org/docs/stable/nn.html",
            "https://pytorch.org/tutorials/",
        ),
        estimated_tokens="~30M",
    ),
    SourceSpec(
        "tensorflow_docs", "http", "E", "tensorflow_docs",
        "Apache-2.0", 0.0,
        "TensorFlow docs seed (api_docs index).", 1,
        url="https://www.tensorflow.org/api_docs",
        extra_urls=(
            "https://www.tensorflow.org/tutorials",
            "https://www.tensorflow.org/guide",
        ),
        estimated_tokens="~30M",
    ),
    SourceSpec(
        "fastapi_docs", "http", "E", "fastapi_docs",
        "MIT", 0.0,
        "FastAPI documentation seed.", 1,
        url="https://fastapi.tiangolo.com/",
        extra_urls=(
            "https://fastapi.tiangolo.com/tutorial/",
            "https://fastapi.tiangolo.com/advanced/",
            "https://fastapi.tiangolo.com/reference/",
        ),
        estimated_tokens="~10M",
    ),
    SourceSpec(
        "sqlite_docs", "http", "E", "sqlite_docs",
        "Public Domain", 0.0,
        "SQLite official documentation index.", 2,
        url="https://www.sqlite.org/docs.html",
        extra_urls=(
            "https://www.sqlite.org/lang.html",
            "https://www.sqlite.org/c3ref/intro.html",
        ),
        estimated_tokens="~10M",
    ),
    SourceSpec(
        "linux_kernel_docs", "http", "E", "linux_docs",
        "GPL-2.0 (docs portion)", 0.1,
        "Linux kernel documentation index.", 1,
        url="https://www.kernel.org/doc/html/latest/",
        extra_urls=(
            "https://www.kernel.org/doc/html/latest/admin-guide/index.html",
            "https://www.kernel.org/doc/html/latest/process/index.html",
        ),
        estimated_tokens="~50M",
    ),
    SourceSpec(
        "llvm_docs", "http", "E", "llvm_docs",
        "Apache-2.0 WITH LLVM-exception", 0.0,
        "LLVM project documentation.", 2,
        url="https://llvm.org/docs/",
        extra_urls=(
            "https://llvm.org/docs/LangRef.html",
            "https://llvm.org/docs/CommandGuide/index.html",
        ),
        estimated_tokens="~15M",
    ),
    SourceSpec(
        "git_docs", "http", "E", "git_docs",
        "GPL-2.0 (docs portion)", 0.1,
        "Git official documentation.", 2,
        url="https://git-scm.com/docs",
        extra_urls=(
            "https://git-scm.com/book/en/v2",
        ),
        estimated_tokens="~10M",
    ),
    SourceSpec(
        "rust_book", "http", "E", "rust_book",
        "MIT OR Apache-2.0", 0.0,
        "The Rust Programming Language book.", 1,
        url="https://doc.rust-lang.org/book/",
        extra_urls=(
            "https://doc.rust-lang.org/std/",
            "https://doc.rust-lang.org/reference/",
        ),
        estimated_tokens="~5M",
    ),
    SourceSpec(
        "go_docs", "http", "E", "go_docs",
        "BSD-3-Clause", 0.0,
        "Go programming language documentation.", 2,
        url="https://go.dev/doc/",
        extra_urls=(
            "https://go.dev/ref/spec",
            "https://pkg.go.dev/std",
        ),
        estimated_tokens="~10M",
    ),
    SourceSpec(
        "gnu_manuals_index", "http", "E", "gnu_manuals",
        "GFDL", 0.1,
        "GNU project manuals index (GCC, Bash, Make, etc.).", 1,
        url="https://www.gnu.org/manual/manual.html",
        extra_urls=(
            "https://gcc.gnu.org/onlinedocs/",
            "https://www.gnu.org/software/bash/manual/",
            "https://www.gnu.org/software/make/manual/",
        ),
        estimated_tokens="~30M",
    ),
    SourceSpec(
        "the_stack_v2_smol", "hf", "E", "the_stack_v2_smol",
        "Permissive (per-file)", 0.1,
        "BigCode The-Stack v2 SMOL subset (permissive license filter).", 2,
        hf_id="bigcode/the-stack-v2-train-smol-ids",
        estimated_tokens="~3B (docstrings+comments)",
    ),
    SourceSpec(
        "ietf_rfcs_bulk", "http", "E", "ietf_rfcs_bulk",
        "BSD-like (IETF Trust)", 0.0,
        "IETF RFC bulk archive (full text).", 2,
        url="https://www.rfc-editor.org/in-notes/tar/rfc-all.tar.gz",
        estimated_tokens="~200M",
    ),

    # =================================================================
    # F. ARTICLES  -- CRITICAL EMPTY CATEGORY (target 30 B)
    # =================================================================
    SourceSpec(
        "wikipedia_simple", "hf", "F", "wikipedia_simple",
        "CC-BY-SA-4.0", 0.0,
        "Simple-English Wikipedia (~250 MB) -- great quick win.", 1,
        hf_id="wikimedia/wikipedia", hf_config="20231101.simple",
        estimated_tokens="~50M",
    ),
    SourceSpec(
        "wikipedia_en_full", "hf", "F", "wikipedia_en",
        "CC-BY-SA-4.0", 0.0,
        "Full English Wikipedia via wikimedia parquet dump (~20 GB).", 1,
        hf_id="wikimedia/wikipedia", hf_config="20231101.en",
        estimated_tokens="~4B",
    ),
    SourceSpec(
        "wikinews_en_dump", "http", "F", "wikinews",
        "CC-BY-2.5", 0.0,
        "English Wikinews dump.", 2,
        url="https://dumps.wikimedia.org/enwikinews/latest/enwikinews-latest-pages-articles.xml.bz2",
        estimated_tokens="~30M",
    ),
    SourceSpec(
        "wikiquote_en_dump", "http", "F", "wikiquote",
        "CC-BY-SA-4.0", 0.0,
        "English Wikiquote dump.", 3,
        url="https://dumps.wikimedia.org/enwikiquote/latest/enwikiquote-latest-pages-articles.xml.bz2",
        estimated_tokens="~30M",
    ),
    SourceSpec(
        "fineweb_edu_sample", "hf", "F", "fineweb_edu",
        "ODC-By-1.0", 0.2,
        "FineWeb-Edu 10BT sample (highest-quality educational web).", 1,
        hf_id="HuggingFaceFW/fineweb-edu", hf_config="sample-10BT",
        estimated_tokens="~10B",
    ),
    SourceSpec(
        "stackexchange_archive", "http", "F", "stackexchange",
        "CC-BY-SA-4.0", 0.0,
        "Stack Exchange data dump (Math, Physics, Stats, CS).", 1,
        url="https://archive.org/download/stackexchange/math.stackexchange.com.7z",
        extra_urls=(
            "https://archive.org/download/stackexchange/physics.stackexchange.com.7z",
            "https://archive.org/download/stackexchange/stats.stackexchange.com.7z",
            "https://archive.org/download/stackexchange/cs.stackexchange.com.7z",
            "https://archive.org/download/stackexchange/cstheory.stackexchange.com.7z",
            "https://archive.org/download/stackexchange/datascience.stackexchange.com.7z",
            "https://archive.org/download/stackexchange/ai.stackexchange.com.7z",
            "https://archive.org/download/stackexchange/philosophy.stackexchange.com.7z",
            "https://archive.org/download/stackexchange/cognitivesciences.stackexchange.com.7z",
            "https://archive.org/download/stackexchange/linguistics.stackexchange.com.7z",
        ),
        estimated_tokens="~3B",
    ),
    SourceSpec(
        "cc_news", "hf", "F", "cc_news",
        "Apache-2.0", 0.0,
        "CC-News (Common Crawl news) -- HF mirror.", 2,
        hf_id="vblagoje/cc_news",
        estimated_tokens="~3B",
    ),

    # =================================================================
    # G. RESEARCH PAPERS  -- expand toward 10 B target
    # =================================================================
    SourceSpec(
        "openalex_works_sample", "http", "G", "openalex",
        "CC0", 0.0,
        "OpenAlex works snapshot manifest (240M+ scholarly works).", 1,
        url="https://docs.openalex.org/download-all-data/openalex-snapshots",
        extra_urls=(
            "https://api.openalex.org/works?per-page=200&select=id,title,abstract_inverted_index,authorships,concepts,doi,publication_year,language,open_access",
        ),
        estimated_tokens="~30B (titles+abstracts)",
    ),
    SourceSpec(
        "redpajama_arxiv", "hf", "G", "redpajama_arxiv",
        "ODC-By-1.0", 0.2,
        "RedPajama-Data arXiv subset (full paper text, CC subset).", 2,
        hf_id="togethercomputer/RedPajama-Data-1T",
        hf_config="arxiv",
        estimated_tokens="~28B",
    ),
    SourceSpec(
        "pubmed_qa_artificial", "hf", "G", "pubmed_qa_artificial",
        "MIT", 0.0,
        "PubMedQA artificial subset (biomedical QA).", 2,
        hf_id="qiaojin/PubMedQA", hf_config="pqa_artificial",
        estimated_tokens="~100M",
    ),
    SourceSpec(
        "pubmed_qa_labeled", "hf", "G", "pubmed_qa_labeled",
        "MIT", 0.0,
        "PubMedQA labeled subset (curated biomedical QA).", 1,
        hf_id="qiaojin/PubMedQA", hf_config="pqa_labeled",
        estimated_tokens="~5M",
    ),
    SourceSpec(
        "s2orc_odc", "hf", "G", "s2orc",
        "ODC-By-1.0", 0.2,
        "Allen AI S2ORC ODC-By release (academic papers).", 2,
        hf_id="allenai/peS2o",
        estimated_tokens="~67B",
    ),
    SourceSpec(
        "pmc_oa_bulk", "http", "G", "pmc_oa_bulk",
        "Mixed CC (filter to BY/0 downstream)", 0.2,
        "PMC Open Access bulk packages (use oa_file_list.csv with filter).", 2,
        url="https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_comm/txt/",
        estimated_tokens="~25B",
    ),
    SourceSpec(
        "elife_archive_info", "http", "G", "elife",
        "CC-BY-4.0", 0.0,
        "eLife open-access archive landing.", 2,
        url="https://elifesciences.org/archive/",
        estimated_tokens="~1B",
    ),
    SourceSpec(
        "crossref_works_sample", "http", "G", "crossref",
        "CC0 (metadata)", 0.0,
        "Crossref public REST API sample (metadata).", 2,
        url="https://api.crossref.org/works?rows=100&select=title,abstract,DOI,published-print,subject",
        estimated_tokens="~5B (metadata)",
    ),

    # =================================================================
    # I. COGNITION  -- CRITICAL EMPTY CATEGORY (target 5 B)
    # =================================================================
    SourceSpec(
        "cognitive_atlas", "http", "I", "cognitive_atlas",
        "CC-BY-3.0", 0.0,
        "Cognitive Atlas: concept ontology for cognitive science.", 1,
        url="https://www.cognitiveatlas.org/api/v-alpha/concept",
        extra_urls=(
            "https://www.cognitiveatlas.org/api/v-alpha/task",
            "https://www.cognitiveatlas.org/api/v-alpha/disorder",
        ),
        estimated_tokens="~5M",
    ),
    SourceSpec(
        "openneuro_metadata", "http", "I", "openneuro",
        "CC0", 0.0,
        "OpenNeuro public datasets metadata (neuroimaging).", 2,
        url="https://openneuro.org/api/datasets",
        estimated_tokens="~10M",
    ),
    SourceSpec(
        "psyarxiv_oai", "http", "I", "psyarxiv",
        "Mixed CC (per-paper)", 0.2,
        "PsyArXiv OAI-PMH listing (psychology preprints).", 2,
        url="https://psyarxiv.com/oai?verb=ListRecords&metadataPrefix=oai_dc",
        estimated_tokens="~2B",
    ),
    SourceSpec(
        "ncbi_bookshelf_cognition", "http", "I", "ncbi_bookshelf",
        "Mixed CC (filter)", 0.2,
        "NCBI Bookshelf open-access neuroscience & cognition titles.", 1,
        url="https://www.ncbi.nlm.nih.gov/books/NBK10844/",
        extra_urls=(
            "https://www.ncbi.nlm.nih.gov/books/NBK11108/",
            "https://www.ncbi.nlm.nih.gov/books/NBK20367/",
            "https://www.ncbi.nlm.nih.gov/books/NBK21054/",
        ),
        estimated_tokens="~1B",
    ),
    SourceSpec(
        "openstax_psych_extra", "http", "I", "openstax_psych",
        "CC-BY-4.0", 0.0,
        "OpenStax Psychology + Anatomy/Physiology textbook pages.", 1,
        url="https://openstax.org/books/psychology-2e/pages/1-introduction",
        extra_urls=(
            "https://openstax.org/books/anatomy-and-physiology-2e/pages/11-introduction",
            "https://openstax.org/books/anatomy-and-physiology-2e/pages/12-introduction",
            "https://openstax.org/books/anatomy-and-physiology-2e/pages/13-introduction",
            "https://openstax.org/books/anatomy-and-physiology-2e/pages/14-introduction",
        ),
        estimated_tokens="~50M",
    ),
    SourceSpec(
        "neuroscience_open_book", "http", "I", "neuroscience_open",
        "CC-BY-NC-SA (NCBI subset)", 0.3,
        "Neuroscience textbook (Purves) via NCBI Bookshelf (warehouse-only NC).", 3,
        url="https://www.ncbi.nlm.nih.gov/books/NBK10799/",
        warehouse_only=True,
        estimated_tokens="~50M",
    ),

    # =================================================================
    # K. KNOWLEDGE GRAPH  -- CRITICAL EMPTY CATEGORY (target 5-10 B)
    # =================================================================
    SourceSpec(
        "wikidata_lexemes", "http", "K", "wikidata_lexemes",
        "CC0", 0.0,
        "Wikidata lexemes dump (linguistic + concept-level).", 1,
        url="https://dumps.wikimedia.org/wikidatawiki/entities/latest-lexemes.json.bz2",
        estimated_tokens="~500M verbalized",
    ),
    SourceSpec(
        "wikidata_truthy_full", "http", "K", "wikidata_truthy",
        "CC0", 0.0,
        "Wikidata truthy RDF (FULL ~80 GB compressed -- huge).", 2,
        url="https://dumps.wikimedia.org/wikidatawiki/entities/latest-truthy.nt.gz",
        estimated_tokens="~10B verbalized",
    ),
    SourceSpec(
        "dbpedia_short_abstracts", "http", "K", "dbpedia",
        "CC-BY-SA-3.0", 0.1,
        "DBpedia short abstracts (English).", 1,
        url="https://databus.dbpedia.org/dbpedia/text/short-abstracts/2022.12.01/short-abstracts_lang=en.ttl.bzip2",
        extra_urls=(
            "https://databus.dbpedia.org/dbpedia/text/long-abstracts/2022.12.01/long-abstracts_lang=en.ttl.bzip2",
        ),
        estimated_tokens="~3B verbalized",
    ),
    SourceSpec(
        "yago_4_5", "http", "K", "yago",
        "CC-BY-SA-3.0", 0.1,
        "YAGO 4.5 knowledge graph (Wikidata-derived, taxonomy-cleaned).", 2,
        url="https://yago-knowledge.org/data/yago4.5/yago-4.5.0.1.tar.gz",
        estimated_tokens="~3B verbalized",
    ),
    SourceSpec(
        "conceptnet_assertions", "http", "K", "conceptnet_assertions",
        "CC-BY-SA-4.0", 0.1,
        "ConceptNet 5.7 assertions (relational commonsense).", 1,
        url="https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz",
        estimated_tokens="~1B verbalized",
    ),

    # =================================================================
    # L. LANGUAGE RESOURCES  -- expand toward 3 B target
    # =================================================================
    SourceSpec(
        "ud_treebanks", "http", "L", "universal_dependencies",
        "CC-BY-SA-4.0 (per-treebank)", 0.1,
        "Universal Dependencies treebanks v2.13 (all languages).", 2,
        url="https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-5287/ud-treebanks-v2.13.tgz",
        estimated_tokens="~500M",
    ),
    SourceSpec(
        "framenet_archive", "http", "L", "framenet_data",
        "CC-BY-3.0", 0.0,
        "FrameNet 1.7 release archive.", 1,
        url="https://framenet.icsi.berkeley.edu/fndrupal/framenet_request_data",
        extra_urls=(
            "https://framenet.icsi.berkeley.edu/fndrupal/framenet_data",
        ),
        estimated_tokens="~30M",
    ),
    SourceSpec(
        "verbnet_data", "http", "L", "verbnet",
        "Apache-2.0", 0.0,
        "VerbNet 3.4 release.", 2,
        url="https://verbs.colorado.edu/verbnet_downloads/downloads/verbnet3.4.tar.gz",
        estimated_tokens="~10M",
    ),
    SourceSpec(
        "wiktionary_en_dump", "http", "L", "wiktionary",
        "CC-BY-SA-4.0", 0.0,
        "English Wiktionary dump (dictionary + etymology).", 1,
        url="https://dumps.wikimedia.org/enwiktionary/latest/enwiktionary-latest-pages-articles.xml.bz2",
        estimated_tokens="~500M",
    ),
    SourceSpec(
        "opus_books", "hf", "L", "opus_books",
        "Public Domain / CC", 0.1,
        "OPUS Books parallel corpora (multilingual book pairs).", 3,
        hf_id="Helsinki-NLP/opus_books", hf_config="en-fr",
        estimated_tokens="~50M",
    ),

    # =================================================================
    # M. LEGAL / GOVERNMENT  -- CRITICAL EMPTY CATEGORY (target 5-10 B)
    # =================================================================
    SourceSpec(
        "us_code_xml", "http", "M", "us_code",
        "Public Domain (US Gov)", 0.0,
        "US Code in XML (full federal statutes).", 1,
        url="https://uscode.house.gov/download/releasepoints/us/pl/118/47/xml_uscAll@118-47.zip",
        estimated_tokens="~500M",
    ),
    SourceSpec(
        "courtlistener_search_seed", "http", "M", "courtlistener",
        "Public Domain (US federal)", 0.0,
        "CourtListener search seed (Free Law Project).", 1,
        url="https://www.courtlistener.com/api/rest/v3/search/?q=*&type=o",
        extra_urls=(
            "https://www.courtlistener.com/api/bulk-info/",
        ),
        estimated_tokens="~20B (bulk)",
    ),
    SourceSpec(
        "sec_edgar_full_index", "http", "M", "sec_edgar",
        "Public Domain (US Gov)", 0.0,
        "SEC EDGAR full-index (10-K, 10-Q quarterly).", 1,
        url="https://www.sec.gov/Archives/edgar/full-index/2024/QTR1/form.idx",
        extra_urls=(
            "https://www.sec.gov/Archives/edgar/full-index/2024/QTR2/form.idx",
            "https://www.sec.gov/Archives/edgar/full-index/2024/QTR3/form.idx",
            "https://www.sec.gov/Archives/edgar/full-index/2024/QTR4/form.idx",
            "https://www.sec.gov/Archives/edgar/full-index/2025/QTR1/form.idx",
        ),
        estimated_tokens="~15B (full)",
    ),
    SourceSpec(
        "nist_publications", "http", "M", "nist",
        "Public Domain (US Gov)", 0.0,
        "NIST publications search index.", 2,
        url="https://www.nist.gov/publications",
        extra_urls=(
            "https://csrc.nist.gov/publications/sp",
        ),
        estimated_tokens="~500M",
    ),
    SourceSpec(
        "cdc_data", "http", "M", "cdc",
        "Public Domain (US Gov)", 0.0,
        "CDC health publications seed.", 2,
        url="https://www.cdc.gov/library/researchguides/index.html",
        estimated_tokens="~500M",
    ),
    SourceSpec(
        "nih_open", "http", "M", "nih",
        "Public Domain (US Gov)", 0.0,
        "NIH publications landing.", 2,
        url="https://www.nih.gov/health-information",
        estimated_tokens="~500M",
    ),
    SourceSpec(
        "who_publications", "http", "M", "who",
        "WHO Open Access (CC-BY-NC-SA)", 0.3,
        "WHO publications (NC -- warehouse-only).", 3,
        url="https://www.who.int/publications",
        warehouse_only=True,
        estimated_tokens="~1B",
    ),
    SourceSpec(
        "eur_lex_search", "http", "M", "eur_lex",
        "EUR-Lex re-use (CC-BY-4.0-like)", 0.1,
        "EUR-Lex EU legislation search.", 2,
        url="https://eur-lex.europa.eu/homepage.html",
        estimated_tokens="~3B",
    ),
    SourceSpec(
        "data_gov_index", "http", "M", "data_gov",
        "Public Domain / Mixed Open", 0.1,
        "data.gov catalog API.", 2,
        url="https://catalog.data.gov/api/3/action/package_search?rows=100",
        estimated_tokens="~5B",
    ),
    SourceSpec(
        "nasa_open", "http", "M", "nasa",
        "Public Domain (NASA)", 0.0,
        "NASA Open Data Portal API.", 2,
        url="https://data.nasa.gov/data.json",
        estimated_tokens="~200M",
    ),
    SourceSpec(
        "un_digital_library", "http", "M", "un_library",
        "UN (mixed open)", 0.2,
        "UN Digital Library landing (filter open per-record).", 3,
        url="https://digitallibrary.un.org/",
        estimated_tokens="~1B",
    ),
]


# =====================================================================
# Utilities (identical signatures to 02_*.py for compatibility)
# =====================================================================
def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def ensure_dir(p):
    p.mkdir(parents=True, exist_ok=True)
    return p

def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def write_json(path, obj):
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def append_jsonl(path, obj):
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def fmt_bytes(n):
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}PB"

def http_stream(url, timeout=180):
    r = requests.get(
        url, stream=True, timeout=timeout,
        headers={"User-Agent": USER_AGENT, "Accept": "*/*"},
        allow_redirects=True,
    )
    r.raise_for_status()
    return r

def download_file(url, dest, force=False, max_retries=3):
    """Download with exponential-backoff retry."""
    ensure_dir(dest.parent)
    if dest.exists() and dest.stat().st_size > 0 and not force:
        return {"status": "skipped_exists", "url": url, "path": str(dest),
                "bytes": dest.stat().st_size, "sha256": None}
    tmp = dest.with_suffix(dest.suffix + ".part")
    if tmp.exists():
        tmp.unlink()
    print(f"    downloading: {url}")
    last_err = None
    for attempt in range(1, max_retries + 1):
        t0 = time.time()
        total = 0
        try:
            r = http_stream(url)
            with tmp.open("wb") as f:
                for chunk in r.iter_content(1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        total += len(chunk)
            tmp.replace(dest)
            dur = time.time() - t0
            sha = sha256_file(dest)
            print(f"    ok ({fmt_bytes(total)} in {dur:.1f}s) sha={sha[:16]}...")
            return {"status": "downloaded", "url": url, "path": str(dest),
                    "bytes": total, "sha256": sha, "duration_sec": round(dur, 2),
                    "attempt": attempt}
        except Exception as e:
            last_err = e
            if tmp.exists():
                tmp.unlink()
            backoff = min(60, 5 * (2 ** (attempt - 1)))
            print(f"    attempt {attempt}/{max_retries} failed: {e!r} -- backoff {backoff}s")
            if attempt < max_retries:
                time.sleep(backoff)
    return {"status": "error", "url": url, "path": str(dest),
            "bytes": 0, "error": repr(last_err)}

def source_dir(root, spec):
    return root / "raw" / CATEGORY_DIRS[spec.category] / spec.subdir

def manifest_path(root, spec):
    return root / "manifests" / f"{spec.subdir}.json"

def already_done(root, spec):
    d = source_dir(root, spec)
    m = manifest_path(root, spec)
    if not (d.exists() and m.exists()):
        return False
    if not any(d.iterdir()):
        return False
    try:
        return json.loads(m.read_text(encoding="utf-8")).get("status") == "ok"
    except Exception:
        return False

def log_event(root, event):
    append_jsonl(root / "logs" / "acquisition_log.jsonl",
                 {"ts": utc_now(), "manager": "extended", **event})

def common_manifest(spec, root, status, out_dir):
    return {
        "key": spec.key, "kind": spec.kind, "category": spec.category,
        "category_dir": CATEGORY_DIRS[spec.category], "subdir": spec.subdir,
        "license_name": spec.license_name, "license_risk": spec.license_risk,
        "warehouse_only": spec.warehouse_only, "priority": spec.priority,
        "estimated_tokens": spec.estimated_tokens, "notes": spec.notes,
        "status": status, "downloaded_at": utc_now(),
        "warehouse_root": str(root), "output_dir": str(out_dir),
        "acquisition_manager": "02b_extended",
    }


# =====================================================================
# Acquisition handlers (identical to 02_*.py interface)
# =====================================================================
def acquire_hf(spec, root, force=False):
    if not HF_AVAILABLE:
        raise RuntimeError("pip install -U datasets huggingface_hub")
    out = source_dir(root, spec)
    if already_done(root, spec) and not force:
        print(f"  [SKIP] {spec.key} already present")
        return {"key": spec.key, "status": "skipped_exists", "output_dir": str(out)}
    ensure_dir(out)
    cache = ensure_dir(root / "hf_cache")
    os.environ.setdefault("HF_HOME", str(cache))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache / "datasets"))
    os.environ.setdefault("HF_HUB_CACHE", str(cache / "hub"))
    cfg_str = f" [{spec.hf_config}])" if spec.hf_config else ")"
    print(f"\n[HF ] {spec.key}  ({spec.hf_id}{cfg_str}")
    print(f"      output = {out}")
    t0 = time.time()
    try:
        if spec.hf_split:
            ds = load_dataset(spec.hf_id, spec.hf_config,
                              split=spec.hf_split, trust_remote_code=False)
        else:
            ds = load_dataset(spec.hf_id, spec.hf_config,
                              trust_remote_code=False)
        ds.save_to_disk(str(out))
    except Exception as e:
        err_manifest = common_manifest(spec, root, "error", out)
        err_manifest["error"] = repr(e)
        write_json(manifest_path(root, spec), err_manifest)
        log_event(root, {"event": "hf_error", "key": spec.key, "error": repr(e)})
        raise
    dur = time.time() - t0
    splits = ({s: int(len(d)) for s, d in ds.items()}
              if isinstance(ds, DatasetDict) else {"data": int(len(ds))})
    total = sum(splits.values())
    man = common_manifest(spec, root, "ok", out)
    man.update({"hf_id": spec.hf_id, "hf_config": spec.hf_config,
                "hf_split": spec.hf_split, "splits": splits,
                "total_rows": total, "duration_sec": round(dur, 2)})
    write_json(manifest_path(root, spec), man)
    log_event(root, {"event": "hf_ok", "key": spec.key, "rows": total,
                     "dur": round(dur, 2)})
    print(f"      ok {total:,} rows in {dur:.1f}s")
    return man

def acquire_http(spec, root, force=False):
    out = source_dir(root, spec)
    if already_done(root, spec) and not force:
        print(f"  [SKIP] {spec.key} already present")
        return {"key": spec.key, "status": "skipped_exists", "output_dir": str(out)}
    ensure_dir(out)
    print(f"\n[HTTP] {spec.key}")
    print(f"       url    = {spec.url}")
    print(f"       output = {out}")
    artifacts = []
    primary_name = (spec.url.split("?")[0].rstrip("/").split("/")[-1]
                    or f"{spec.key}.dat")
    # avoid name collisions
    if not primary_name or primary_name in ("", "/"):
        primary_name = f"{spec.key}.dat"
    artifacts.append(download_file(spec.url, out / primary_name, force=force))
    for i, extra in enumerate(spec.extra_urls):
        en = (extra.split("?")[0].rstrip("/").split("/")[-1]
              or f"extra_{i}.dat")
        if (out / en).exists():
            en = f"{Path(en).stem}_{i}{Path(en).suffix or '.dat'}"
        artifacts.append(download_file(extra, out / en, force=force))
    man = common_manifest(spec, root, "ok", out)
    man.update({"url": spec.url, "extra_urls": list(spec.extra_urls),
                "artifacts": artifacts})
    write_json(manifest_path(root, spec), man)
    ok = sum(1 for a in artifacts
             if a.get("status") in {"downloaded", "skipped_exists"})
    err = sum(1 for a in artifacts if a.get("status") == "error")
    log_event(root, {"event": "http_ok", "key": spec.key, "ok": ok, "err": err})
    print(f"       ok artifacts ok={ok} err={err}")
    return man


# =====================================================================
# Dispatcher + CLI
# =====================================================================
def acquire(spec, root, force=False):
    if spec.kind.startswith("hf"):
        return acquire_hf(spec, root, force=force)
    if spec.kind.startswith("http"):
        return acquire_http(spec, root, force=force)
    raise ValueError(f"Unknown kind: {spec.kind}")

def build_registry(include_warehouse_only=False):
    if include_warehouse_only:
        return list(EXTENDED_SOURCES)
    return [s for s in EXTENDED_SOURCES if not s.warehouse_only]

def ensure_warehouse_layout(root):
    for n in ["raw", "manifests", "logs", "hf_cache",
              "reports", "registry", "provenance", "governance_logs"]:
        ensure_dir(root / n)
    for cat in CATEGORY_DIRS.values():
        ensure_dir(root / "raw" / cat)

def print_table(rows, headers):
    widths = [max(len(str(r[i])) for r in [headers] + rows)
              for i in range(len(headers))]
    print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print("  ".join(str(c).ljust(w) for c, w in zip(r, widths)))

def list_sources(include_wo, categories=None):
    sources = build_registry(include_wo)
    if categories:
        sources = [s for s in sources if s.category in set(categories)]
    rows = [[s.key,
             "WAREHOUSE-ONLY" if s.warehouse_only else f"P{s.priority}",
             f"{s.category}/{s.subdir}",
             s.license_name[:24],
             f"{s.license_risk:.1f}",
             s.estimated_tokens] for s in sources]
    print_table(rows, ["KEY", "PRIORITY", "CATEGORY/SUBDIR",
                       "LICENSE", "RISK", "EST. TOKENS"])
    print(f"\nTotal: {len(sources)} sources")

def write_summary(root, results):
    s = {
        "summary_generated_at": utc_now(),
        "warehouse_root": str(root),
        "acquisition_manager": "02b_extended",
        "total_attempted": len(results),
        "ok": sum(1 for r in results if r.get("status") == "ok"),
        "skipped": sum(1 for r in results if r.get("status") == "skipped_exists"),
        "errors": sum(1 for r in results if r.get("status") == "error"),
        "results": results,
    }
    stamp = utc_now().replace(":", "-")
    write_json(root / "reports" / f"acquisition_summary_ext_{stamp}.json", s)
    write_json(root / "download_summary_extended.json", s)

def main():
    p = argparse.ArgumentParser(
        description="CognitiveOC EXTENDED corpus acquisition manager (gap-fill).")
    p.add_argument("--root", default=str(DEFAULT_WAREHOUSE))
    p.add_argument("--all", action="store_true")
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument("--list", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--include-warehouse-only", action="store_true")
    p.add_argument("--priority", type=int, choices=[1, 2, 3], default=None)
    p.add_argument("--skip-large", action="store_true")
    p.add_argument("--categories", nargs="*", default=None,
                   help="Limit to specific category letters (e.g. E F I K M)")
    args = p.parse_args()

    root = Path(args.root)
    ensure_warehouse_layout(root)

    if args.list:
        list_sources(args.include_warehouse_only, args.categories)
        return 0
    if not args.all and not args.only:
        p.error("Use --all, --only <keys...>, or --list")

    sources = build_registry(args.include_warehouse_only)
    if args.categories:
        sources = [s for s in sources if s.category in set(args.categories)]
    if args.priority is not None:
        sources = [s for s in sources if s.priority == args.priority]
    if args.skip_large:
        sources = [s for s in sources if s.key not in LARGE_SOURCES]
    if args.only:
        wanted = set(args.only)
        sources = [s for s in sources if s.key in wanted]
        missing = wanted - {s.key for s in sources}
        if missing:
            print(f"WARNING: unknown keys: {sorted(missing)}", file=sys.stderr)
    if not sources:
        print("No sources to acquire.")
        return 1

    print(f"\nWarehouse: {root}")
    print(f"Queued (extended): {len(sources)} sources")
    print("=" * 70)

    results = []
    for spec in sources:
        try:
            r = acquire(spec, root, force=args.force)
        except Exception as e:
            r = {"key": spec.key, "status": "error", "error": repr(e)}
            log_event(root, {"event": "acquisition_error",
                             "key": spec.key, "error": repr(e)})
            print(f"  FAILED: {spec.key}: {e}", file=sys.stderr)
        results.append(r)

    write_summary(root, results)
    print(f"\n{'=' * 70}")
    print(f"  EXTENDED ACQUISITION COMPLETE")
    print(f"  ok={sum(1 for r in results if r.get('status') == 'ok')}  "
          f"skipped={sum(1 for r in results if r.get('status') == 'skipped_exists')}  "
          f"errors={sum(1 for r in results if r.get('status') == 'error')}")
    print(f"{'=' * 70}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
