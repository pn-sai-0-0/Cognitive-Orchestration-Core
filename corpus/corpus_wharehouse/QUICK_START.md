# CognitiveOC Corpus Gap-Fill — Quick Start

> **TL;DR:** Place the 3 new files into your existing `corpus_wharehouse/` folder. Run the 11-command sequence below. Every empty category will be filled.

## 1. File placement

```
D:\projects\CognitiveOC\Final_Versions\cognitiveoc_v3\corpus\corpus_wharehouse\
├── 01_create_warehouse_architecture.py        (existing)
├── 02_corpus_acquisition_manager.py           (existing)
├── 02b_corpus_acquisition_manager_extended.py  ← NEW  drop in
├── 03_corpus_health_report.py                  (existing)
├── 04_run_full_pipeline.py                    (existing)
├── 04b_run_extended_pipeline.py                ← NEW  drop in
├── 05_dataset_registry.yaml                   (existing)
├── 05b_dataset_registry_extended.yaml          ← NEW  drop in
├── GAP_ANALYSIS.md                             ← NEW  reference doc
└── QUICK_START.md                              ← this file
```

## 2. One-line install verification

```bash
cd D:\projects\CognitiveOC\Final_Versions\cognitiveoc_v3\corpus\corpus_wharehouse
python 02b_corpus_acquisition_manager_extended.py --list
```

You should see a table of 75 new sources.

## 3. The 11 commands that fill every empty category

Run them in order. Each is independent and idempotent (already-downloaded sources are skipped).

```bash
# === STEP A: Fill F (Articles) — was empty ===
python 02b_corpus_acquisition_manager_extended.py --only wikipedia_simple
python 02b_corpus_acquisition_manager_extended.py --only stackexchange_archive

# === STEP B: Fill K (Knowledge Graph) — was empty ===
python 02b_corpus_acquisition_manager_extended.py --only wikidata_lexemes dbpedia_short_abstracts conceptnet_assertions

# === STEP C: Fill E (Technical Docs) — was empty ===
python 02b_corpus_acquisition_manager_extended.py --only python_docs numpy_docs pandas_docs pytorch_docs fastapi_docs rust_book linux_kernel_docs gnu_manuals_index

# === STEP D: Fill I (Cognition) — was empty ===
python 02b_corpus_acquisition_manager_extended.py --only cognitive_atlas ncbi_bookshelf_cognition openstax_psych_extra

# === STEP E: Fill M (Legal/Government) — was empty ===
python 02b_corpus_acquisition_manager_extended.py --only us_code_xml courtlistener_search_seed sec_edgar_full_index nist_publications nasa_open

# === STEP F: Verify ===
python 03_corpus_health_report.py
```

After STEP F, your health report should show **0 empty categories**.

## 4. Then expand the major categories

```bash
# Wikipedia full English (~20 GB, ~30 min on fast connection)
python 02b_corpus_acquisition_manager_extended.py --only wikipedia_en_full

# FineWeb-Edu sample (~30 GB, 10B tokens — highest quality web)
python 02b_corpus_acquisition_manager_extended.py --only fineweb_edu_sample

# Tulu-3 SFT (best open instruction data, ~1.5B tokens)
python 02b_corpus_acquisition_manager_extended.py --only tulu3_sft

# FLAN v2 instruction collection
python 02b_corpus_acquisition_manager_extended.py --only flan_v2_subset

# Wikisource + Wiktionary
python 02b_corpus_acquisition_manager_extended.py --only wikisource_en_dump wiktionary_en_dump

# Standard Ebooks (curated public domain books)
python 02b_corpus_acquisition_manager_extended.py --only standard_ebooks
```

## 5. Run everything Priority-1 at once (alternative)

```bash
# Small/medium priority-1 sources only
python 02b_corpus_acquisition_manager_extended.py --all --priority 1 --skip-large

# All priority-1 including large datasets
python 02b_corpus_acquisition_manager_extended.py --all --priority 1
```

## 6. Run everything together (master pipeline)

```bash
# Master pipeline: build skeleton -> original 02 -> extended 02b -> health report
python 04b_run_extended_pipeline.py --priority 1 --skip-large

# Skip the original 02 (already done) and only run the new extended phase
python 04b_run_extended_pipeline.py --only-extended --priority 1
```

## 7. Filter by category letter

```bash
# Only fill empty categories E, F, I, K, M (gap-fill mode)
python 02b_corpus_acquisition_manager_extended.py --all --categories E F I K M

# Only Priority-1 sources in those categories
python 02b_corpus_acquisition_manager_extended.py --all --categories E F I K M --priority 1
```

## 8. Common issues

| Symptom | Fix |
|---------|-----|
| `Dataset 'X' doesn't exist on the Hub` | Check spelling or set `HF_TOKEN` env var |
| Symlinks warning on Windows | Set `HF_HUB_DISABLE_SYMLINKS_WARNING=1` |
| 403 / 429 errors during HTTP downloads | The retry logic handles transient ones; for persistent 429, slow down with `time.sleep` or wait |
| `Connection timeout` on huge datasets | Use `--skip-large` first; download huge ones one-at-a-time with `--only` |
| Disk space full | Run `--only` per source, process+delete cache between runs |
| Need to redownload | Add `--force` flag |

## 9. After acquisition: what you should run next (not included here)

1. Filter pipeline — language + quality + PII filters
2. Deduplication — MinHash LSH
3. Tokenizer training — 48K BPE on a sample
4. Tokenize + shard — uint16 binary
5. Training-release manifest — `releases/v1/`

These are separate stages and are NOT part of acquisition. The warehouse
skeleton you already have (`cleaned/`, `deduplicated/`, `scored/`,
`approved/`, `tokenized/`) is where each of those stages outputs.

## 10. Need to see all 79 new sources?

```bash
python 02b_corpus_acquisition_manager_extended.py --list
```

Or open `05b_dataset_registry_extended.yaml` in a text editor for a human-readable audit view.
