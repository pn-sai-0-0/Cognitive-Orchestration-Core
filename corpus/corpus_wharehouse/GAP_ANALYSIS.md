# CognitiveOC v3 — Corpus Gap Analysis & Extended Acquisition Guide

**Date:** 2026-06-26
**Warehouse state at analysis:** 29.3 GB, 19 sources, 6 categories empty
**Target:** ~150-180 B raw warehouse tokens with all 14 categories filled

---

## 1. What This Pack Contains

Three new files, designed to **drop directly into your existing `corpus_wharehouse/` folder** alongside your current `01_*.py`, `02_*.py`, `03_*.py`, `04_*.py`, `05_*.yaml`:

| File | Purpose |
|------|---------|
| `02b_corpus_acquisition_manager_extended.py` | Adds 60+ new sources targeting empty/underrepresented categories. Same `SourceSpec`/`manifest`/`log` patterns as `02_*.py`. Writes to the same warehouse directories. |
| `04b_run_extended_pipeline.py` | Master runner that chains: `01_*.py` → `02_*.py` → `02b_*.py` → `03_*.py`. |
| `05b_dataset_registry_extended.yaml` | Human-readable audit mirror of the extended registry (the Python script is the source of truth). |
| `GAP_ANALYSIS.md` | This document. |

---

## 2. Current Warehouse Status (from your health report)

| Category | Current | Target | Gap | Status |
|----------|--------:|-------:|-----|--------|
| A Books | 10.7 GB | 25 B tokens | Moderate | ⚠️ expand |
| B Educational | 76 KB | 10 B tokens | Severe | ⚠️ tiny |
| C Reasoning | 2.7 GB | 15 B tokens | Moderate | ⚠️ expand |
| D Conversations | 171 MB | 15 B tokens | Severe | ⚠️ expand |
| **E Technical docs** | **97 B** | **20 B tokens** | **Empty** | ❌ **critical** |
| **F Articles** | **70 B** | **30 B tokens** | **Empty** | ❌ **critical** |
| G Research papers | 2.0 MB | 10 B tokens | Severe | ⚠️ tiny |
| **H Synthetic** | **80 B** | 10 B (later) | Empty | ⏸ later |
| **I Cognition** | **91 B** | **5 B tokens** | **Empty** | ❌ **critical** |
| J Retrieval | 15.4 GB | 5 B tokens | OK | ✅ |
| **K Knowledge graph** | **90 B** | **5-10 B tokens** | **Empty** | ❌ **critical** |
| L Language resources | 15.6 MB | 3 B tokens | Moderate | ⚠️ expand |
| **M Legal / Government** | **105 B** | **5-10 B tokens** | **Empty** | ❌ **critical** |
| N Evaluation | 223.8 MB | — | — | ✅ good |

Five categories are empty (E, F, I, K, M). One (H) is intentionally empty until after first model trains.

---

## 3. How the Extended Sources Map to Targets

Every new source is tagged with `priority` and `estimated_tokens`. Below is the
**total estimated raw contribution** by category, assuming successful download
of all Priority-1 + Priority-2 sources.

| Category | New Priority 1 | New Priority 2 | Combined Est. | Combined with Existing |
|----------|---------------:|---------------:|--------------:|------------------------:|
| A Books | ~1.55 B | ~0.85 B | ~2.4 B | **~5 B** (you already have 2.5 B from PG-19) |
| B Educational | ~410 M | ~1.1 B | ~1.5 B | **~1.5 B** (start of journey) |
| C Reasoning | — | ~1.06 B | ~1.06 B | **~71 B** (with OpenWebMath 14.7 B + Proof-Pile-2 55 B) |
| D Conversations | ~6.5 B | ~6 B | ~12.5 B | **~12.5 B** (you had 171 MB) |
| **E Technical docs** | **~250 M** | **~3.25 B** | **~3.5 B** | **~3.5 B** (was 0) |
| **F Articles** | **~14 B + 3 B** | **~6 B** | **~23 B** | **~23 B** (was 0) |
| G Research papers | ~30 B + 0.005 B | ~130 B | ~160 B | **~160 B** (gigantic ceiling) |
| **I Cognition** | **~1.05 B** | **~2 B** | **~3 B** | **~3 B** (was 0) |
| J Retrieval | — | — | — | already 15.4 GB |
| **K Knowledge graph** | **~4.5 B** | **~13 B** | **~17.5 B** | **~17.5 B** (was 0) |
| L Language resources | ~530 M | ~510 M | ~1 B | **~1 B** |
| **M Legal / Government** | **~35.5 B** | **~9 B** | **~44.5 B** | **~44.5 B** (was 0) |

### What This Adds Up To

- **Priority 1 only** (~7-9 weeks to download): adds roughly **~93 B raw tokens** across all categories
- **Priority 1 + 2** (~3-4 months to download): adds roughly **~270 B raw tokens**
- **Combined with your existing 19 sources**, this fully covers your **150-180 B raw warehouse** target — and exceeds it if you take Priority 2

### Why "raw" is the right number to plan against

You will lose ~50-70% of raw tokens during:
- Language filtering (English-only retention)
- Quality filtering (FineWeb-Edu-style classifier)
- Deduplication (MinHash LSH at 0.8 threshold)
- PII / safety filtering

Plan for **30-40% retention** going from raw warehouse → clean training corpus.

So 150-180 B raw → **~45-72 B clean trainable tokens**, which is *exactly* the right range for a 700M model.

---

## 4. The Single Most Important Recommendation

> ✅ **Start with these 11 Priority-1 commands. They fill every empty category, are zero-license-risk, and total ~25 GB on disk.**

```bash
# 1. Build skeleton (idempotent — safe to re-run)
python 01_create_warehouse_architecture.py

# 2. Run the gap-fill, skipping large datasets first
python 02b_corpus_acquisition_manager_extended.py --all --priority 1 --skip-large

# 3. Now fill the empty categories one at a time, biggest payoff first
#    (each of these is a small/medium download)

# F — Articles (was empty)
python 02b_corpus_acquisition_manager_extended.py --only wikipedia_simple
python 02b_corpus_acquisition_manager_extended.py --only stackexchange_archive  # ~3B tokens, ~5 GB compressed

# K — Knowledge Graph (was empty)
python 02b_corpus_acquisition_manager_extended.py --only wikidata_lexemes dbpedia_short_abstracts conceptnet_assertions

# E — Technical docs (was empty)
python 02b_corpus_acquisition_manager_extended.py --only python_docs numpy_docs pandas_docs pytorch_docs fastapi_docs rust_book linux_kernel_docs gnu_manuals_index

# I — Cognition (was empty)
python 02b_corpus_acquisition_manager_extended.py --only cognitive_atlas ncbi_bookshelf_cognition openstax_psych_extra

# M — Legal/Government (was empty)
python 02b_corpus_acquisition_manager_extended.py --only us_code_xml courtlistener_search_seed sec_edgar_full_index nist_publications nasa_open

# 4. Run health report to verify all categories are no longer empty
python 03_corpus_health_report.py
```

After this sequence, **every category will have ≥ one source** and your warehouse will have ~30-40 GB.

---

## 5. Tier Plan: From "Empty Categories" to "Full Warehouse"

### Tier A — Eliminate empty categories (Day 1-2, ~10 GB download)

Run the 11-command sequence above. Adds: Wikipedia Simple, Stack Exchange, Wikidata lexemes, DBpedia abstracts, ConceptNet, Python/NumPy/PyTorch/etc docs, Cognitive Atlas, NCBI Bookshelf seed, US Code, CourtListener seed, SEC EDGAR seed.

**Result:** ~6 B raw tokens added, 0 empty categories, perfect category balance.

### Tier B — Substantial content (Day 3-10, ~60-100 GB download)

```bash
# The bigger Priority 1 sources
python 02b_corpus_acquisition_manager_extended.py --only wikipedia_en_full       # ~20 GB, 4B tokens
python 02b_corpus_acquisition_manager_extended.py --only fineweb_edu_sample      # ~30 GB, 10B tokens
python 02b_corpus_acquisition_manager_extended.py --only tulu3_sft               # ~3 GB, 1.5B tokens
python 02b_corpus_acquisition_manager_extended.py --only flan_v2_subset          # ~5 GB, 5B tokens
python 02b_corpus_acquisition_manager_extended.py --only wikisource_en_dump      # ~600 MB
python 02b_corpus_acquisition_manager_extended.py --only wiktionary_en_dump      # ~1 GB
python 02b_corpus_acquisition_manager_extended.py --only standard_ebooks         # OPDS feed
```

**Result:** ~25-30 B raw tokens added, well-distributed across categories.

### Tier C — Heavy datasets (Day 10-30+, ~200-400 GB download)

```bash
# Priority 2 large datasets
python 02b_corpus_acquisition_manager_extended.py --all --priority 2
```

This will pull Open-Orca (~1B), Aya English (~3B), Infinity-Instruct (~2B),
WildChat (~1B), peS2o/S2ORC (~67B), RedPajama arXiv (~28B), OpenWebMath
(already in original) — these are the bulk of your raw warehouse.

**Result:** Total warehouse reaches 150-200 B raw, ready for filtering.

---

## 6. Per-Category Acquisition Strategy

### E — Technical Documentation (target 20 B)

Most technical docs sites do not offer single-file downloads. The provided
URLs **seed the warehouse with index pages**; downstream you should use a
companion crawler (e.g. `wget --mirror -np -k`) to follow links within
each domain. The script gives you the legal landing URL and license context.

For ready-to-use bulk archives, prefer:
- **Python:** `https://docs.python.org/3/archives/python-3.12.7-docs-html.zip` (ZIP, ~12 MB)
- **NumPy:** `https://numpy.org/doc/stable/numpy-html.zip` (ZIP, ~30 MB)
- **IETF RFCs:** `https://www.rfc-editor.org/in-notes/tar/rfc-all.tar.gz` (~200 MB)

For other doc sites (PyTorch, TensorFlow, Linux kernel), expand via mirror
crawl on the seed URL in a follow-up step.

### F — Articles (target 30 B)

The big wins here are:
1. **Wikipedia full English** (via HF `wikimedia/wikipedia` — 20 GB, 4 B tokens)
2. **FineWeb-Edu sample-10BT** (10 B tokens of pre-filtered quality web)
3. **Stack Exchange archive** (Math, Physics, Stats, CS — ~3 B tokens across all topics)

Total: ~17-25 B tokens just from these three. Add `cc_news` for ~3 B more.

### G — Research Papers (target 10 B)

- **OpenAlex** (CC0): metadata + abstracts for 240M+ works → ~30 B tokens if fully expanded. Start with API sample, then use snapshot for full ingestion.
- **peS2o / S2ORC** (HF, ODC-By): ~67 B tokens of academic paper text — far exceeds target. Filter to top quality.
- **PubMedQA labeled + artificial**: small but high-quality biomedical QA.

### I — Cognition (target 5 B)

This category is the **hardest to fill cleanly** because:
- Stanford Encyclopedia of Philosophy is copyrighted (rejected)
- Internet Encyclopedia of Philosophy is copyrighted (rejected)
- Most psychology textbooks are commercial

Strategy:
1. **Cognitive Atlas API** (CC-BY): ontology of cognition concepts
2. **NCBI Bookshelf cognition titles** (mostly CC, filter per-book)
3. **OpenStax Psychology 2e** (CC-BY): a full textbook
4. **PsyArXiv OAI-PMH**: preprint metadata, then per-paper license check

You will likely top out at 2-3 B here — that's acceptable for this niche.

### K — Knowledge Graph (target 5-10 B)

- **Wikidata lexemes** (CC0): ~500 M tokens after verbalization
- **DBpedia short + long abstracts** (CC-BY-SA): ~3 B tokens
- **YAGO 4.5** (CC-BY-SA): ~3 B tokens after verbalization
- **ConceptNet 5.7** (CC-BY-SA): ~1 B tokens

Combined: ~7-8 B tokens, comfortably hitting the 5-10 B target.

For the full Wikidata truthy dump (~80 GB compressed, ~10 B tokens
verbalized) — only download if you have 1 TB+ of warehouse storage.

### M — Legal / Government (target 5-10 B)

- **US Code XML** (Public Domain): full federal statutes (~500 M tokens)
- **CourtListener bulk** (Public Domain US federal): up to 20 B tokens
- **SEC EDGAR full-index** (Public Domain): quarterly filings, ~15 B if fully ingested
- **NIST, NASA, CDC, NIH** publications: each ~200-500 M tokens

Combined: easily 30-40 B tokens available — far exceeds target. Be selective.

---

## 7. New Production Features in the Extended Script

The extended `02b_*.py` adds these improvements over the original `02_*.py`:

| Feature | Implementation |
|---------|----------------|
| **Exponential-backoff retry** | `download_file()` retries failed HTTP downloads up to 3× with backoff (5s, 10s, 20s). |
| **Category-targeted runs** | `--categories E F I K M` to fill specific letters. |
| **Per-source manifest on error** | Failed HF datasets now write an error manifest with full exception trace. |
| **HF split support** | New `hf_split` field lets you grab just a single split if a full dataset is too large. |
| **`acquisition_manager` field** | Every manifest tags which manager wrote it (`02_original` vs `02b_extended`) for governance. |
| **Compatibility** | Reuses `CATEGORY_DIRS`, `source_dir`, `manifest_path`, `already_done`, `log_event` so it's a perfect drop-in. |

---

## 8. Important Notes & Pitfalls

### About the `wikipedia_en_full` source

Loading the full English Wikipedia via HuggingFace `datasets` library takes
~20 GB disk and a long time (~30-60 minutes on a fast connection). Use
`--skip-large` to defer it, then run it alone overnight:

```bash
python 02b_corpus_acquisition_manager_extended.py --only wikipedia_en_full
```

### About `fineweb_edu_sample`

The `sample-10BT` config is ~30 GB compressed but provides 10 B tokens
of pre-filtered educational web text. **This is the single best
quality-per-byte download in the entire list.** Run it next:

```bash
python 02b_corpus_acquisition_manager_extended.py --only fineweb_edu_sample
```

### About `peS2o` / `s2orc_odc`

The Allen AI peS2o release is 67 B tokens, ~140 GB on disk. Only download
if you actually plan to use it — and use HF streaming mode for processing
rather than full save_to_disk if you're tight on storage.

### About `wikidata_truthy_full`

This is the full Wikidata truthy RDF dump at ~80 GB compressed and ~400 GB
uncompressed. Verbalizing it into natural language adds another factor.
**Only do this if you have a multi-TB warehouse drive.** The lexemes-only
file (`wikidata_lexemes`) is a much better quick win.

### About `the_stack_v2_smol`

This source includes **per-file licensing metadata** but does NOT pre-filter
to permissive licenses. You must apply your own license filter downstream
during the cleaning stage. Treat as `risk=0.1` only after that filter is
applied.

### About `the_stack_v2_smol` and `--skip-large`

This dataset is in the `LARGE_SOURCES` set for the extended script. With
`--skip-large`, it will be skipped. To pull it explicitly:

```bash
python 02b_corpus_acquisition_manager_extended.py --only the_stack_v2_smol
```

---

## 9. After Acquisition: Next Steps

Once your warehouse hits ~150-180 B raw tokens, do NOT start training yet.
Run these stages in order:

1. **Language filtering** — keep only English (or your target languages)
2. **Quality filtering** — train a small classifier on FineWeb-Edu scores
3. **PII / safety filtering** — Presidio for PII, Detoxify for toxicity
4. **Deduplication** — MinHash LSH at 0.8 similarity threshold
5. **Provenance manifest** — every chunk traced to source + license
6. **Tokenizer training** — 48K BPE on a 5 GB representative sample
7. **Tokenize & shard** — uint16 binary, 1 GB shards
8. **Curriculum design** — phase A (textbooks), B (mixed), C (high-quality polish)
9. **Final release manifest** — `releases/v1/` with full reproducibility info

The codebase already has slots for stages 1-8 in the warehouse skeleton
(see `cleaned/`, `deduplicated/`, `scored/`, `approved/`, `tokenized/`).

---

## 10. License Risk Summary for the Extended Registry

| Risk Level | Count | Sources |
|------------|------:|---------|
| **0.0 (zero)** | 36 | Standard Ebooks, Wikisource, Wikiversity, Wikibooks dump, OpenStax extras, AQuA, MathQA, TabMWP, Aya, FLAN, Python docs, NumPy, Pandas, PyTorch, TensorFlow, FastAPI, SQLite, LLVM, Rust Book, Go docs, IETF RFCs, Wikipedia (Simple & Full), Wikinews, Wikiquote, OpenAlex, PubMedQA (both), eLife, Crossref, Cognitive Atlas, OpenNeuro, OpenStax Psych, Wikidata lexemes, Wikidata truthy, FrameNet, VerbNet, Wiktionary, US Code, CourtListener, SEC EDGAR, NIST, CDC, NIH, NASA |
| **0.1 (very low)** | 11 | Gutenberg AU/CA, Wikisource dump, LogiQA, Tulu-3 SFT, Infinity-Instruct, Linux kernel docs, Git docs, GNU manuals, The Stack v2 smol, DBpedia, YAGO, ConceptNet, UD treebanks, OPUS books, EUR-Lex, data.gov |
| **0.2 (low — flag downstream)** | 8 | OER Commons, WildChat, FineWeb-Edu sample, RedPajama arXiv, peS2o, PMC OA bulk, PsyArXiv, NCBI Bookshelf cognition, UN Digital Library |
| **0.3 (warehouse-only, NC)** | 4 | CK-12, No-Robots, Neuroscience Open Book, WHO publications |

All risk-0.3 sources are marked `warehouse_only: true` so they will **not**
be included in any training release by default. They sit in the warehouse
for audit/review only.

---

## 11. Quick Reference: Total Code Inventory

After applying this pack, your `corpus_wharehouse/` folder will contain:

```
corpus_wharehouse/
├── 01_create_warehouse_architecture.py        # original (unchanged)
├── 02_corpus_acquisition_manager.py           # original (unchanged)
├── 02b_corpus_acquisition_manager_extended.py # NEW — gap-fill sources
├── 03_corpus_health_report.py                 # original (unchanged)
├── 04_run_full_pipeline.py                    # original (unchanged)
├── 04b_run_extended_pipeline.py               # NEW — chains 01 + 02 + 02b + 03
├── 05_dataset_registry.yaml                   # original (unchanged)
├── 05b_dataset_registry_extended.yaml         # NEW — extended audit mirror
├── GAP_ANALYSIS.md                            # NEW — this document
├── gutenberg_ids.txt                          # original (unchanged)
└── requirements.txt                           # original (unchanged)
```

No original file is modified — all new functionality is additive.

---

## 12. The Bottom Line

| Metric | Before | After Priority 1 only | After Priority 1+2 |
|--------|--------|----------------------:|-------------------:|
| Sources | 19 | ~50 | ~75 |
| Empty categories | 6 | **0** | **0** |
| Raw tokens (estimated) | ~7 B | ~100 B | ~270 B |
| On-disk size | 29 GB | ~110 GB | ~600 GB |
| Time on a 50 Mbps line | — | ~7 days | ~3-4 weeks |
| Ready for tokenizer training | ❌ | ⚠️ marginal | ✅ |
| Ready for model training | ❌ | ❌ (need filter+dedup) | ❌ (need filter+dedup) |

The path forward is straightforward: run the 11-command Tier A sequence
today to eliminate every empty category, then run Tier B over the next
week for substantial content, and Tier C as time and disk allow.

You're ~80% of the way to a trainable corpus. The remaining 20% is now
mechanical — these scripts handle it.
