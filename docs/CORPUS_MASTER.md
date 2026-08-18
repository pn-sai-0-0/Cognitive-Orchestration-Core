# CognitiveOC v3 — Zero-Risk Corpus Engineering Master Document

**Policy:** Zero-risk-first · Public Domain · CC0 · Explicit-permissive only in v1  
**Target:** Release v1 = 30B tokens · Warehouse ceiling = 35B tokens (zero-risk) / ~80B (permissive-first)  
**Hardware:** 1TB SSD · RTX 5060 8GB · 16 h/day training  
**Status:** Implementation complete · All modules verified · CLI wired

---

## Table of Contents

1. [Corpus Strategy Summary](#1-corpus-strategy-summary)
2. [Source Category Matrix](#2-source-category-matrix)
3. [Licence / Risk Matrix](#3-licence--risk-matrix)
4. [Quality Scoring and Acceptance Rules](#4-quality-scoring-and-acceptance-rules)
5. [Corpus Warehouse Design](#5-corpus-warehouse-design)
6. [Training Release Design](#6-training-release-design)
7. [Storage / Size Plan](#7-storage--size-plan)
8. [Folder Structure](#8-folder-structure)
9. [Manifest Schema](#9-manifest-schema)
10. [Cleaning Pipeline Design](#10-cleaning-pipeline-design)
11. [Deduplication Pipeline Design](#11-deduplication-pipeline-design)
12. [Synthetic Data Strategy](#12-synthetic-data-strategy)
13. [Train / Val / Test Strategy](#13-train--val--test-strategy)
14. [Corpus Versioning Strategy](#14-corpus-versioning-strategy)
15. [Corpus Governance Rules](#15-corpus-governance-rules)
16. [Project Integration Plan](#16-project-integration-plan)
17. [Exact Files to Add / Modify](#17-exact-files-to-add--modify)
18. [CLI Commands](#18-cli-commands)
19. [Risks and Failure Modes](#19-risks-and-failure-modes)
20. [Final Recommendation](#20-final-recommendation)
21. [Go / No-Go Decision](#21-go--no-go-decision)

---

## 1. Corpus Strategy Summary

### The zero-risk-first principle

Release v1 contains **only** sources where training rights are unambiguous and zero-risk:
- Public Domain (pre-1927, government works, expired copyright)
- CC0 (explicit public domain dedication)
- Apache 2.0 / MIT / BSD (explicit permissive)
- CC-BY 4.0 (attribution required — recorded in manifest)
- CC-BY-SA 3.0/4.0 (attribution + share-alike — documented)

Any source with NC clauses, ND clauses, ambiguous ToS, or unknown provenance is **warehouse-only at best, rejected at worst**.

### Why this reduces the maximum warehouse size

An honest calculation:

| Policy tier | Achievable clean tokens | Notes |
|---|---|---|
| Pure PD + CC0 only | ~22B tokens | Books, Wikidata, basic datasets |
| + Apache/MIT/BSD | ~25B tokens | Adds research datasets, code docs |
| + CC-BY 4.0 | ~30B tokens | Adds OpenStax, ArXiv CC-BY, PubMed OA |
| + CC-BY-SA | ~35B tokens | Adds Wikipedia, DBpedia, ACL, Dolly |
| + warehouse-only CC-BY-NC | ~55B tokens | Never in release v1 |
| + warehouse-only S2ORC | ~80B tokens | NC clause — never in release |
| Unrestricted crawl | ~150B+ tokens | **Rejected — licence unknown** |

**Conclusion: The honest zero-risk warehouse ceiling is ~35B tokens. A 150B warehouse is not achievable on a zero-risk policy. This is stated clearly and not obscured.**

Release v1 target: **30B tokens** — achievable within the zero-risk ceiling.

### What this corpus optimises for

Teaching behaviour · Reasoning · Document analysis · Emotion-aware dialogue  
Intent handling · Retrieval grounding · KG support · Memory support  
Technical accuracy · Long-form coherence

What it does **not** optimise for:
Raw volume · Internet breadth · Noise tolerance · Social media mimicry  
Generic autocomplete · Unreviewed AI output

---

## 2. Source Category Matrix

### Category A — Books (Public Domain)

| Field | Value |
|---|---|
| Purpose | Long-form coherence, vocabulary depth, narrative and expository structure |
| Best sources | Project Gutenberg (PD) · Standard Ebooks (CC0) · Wikisource curated texts (CC-BY-SA) |
| Acceptable | Internet Archive plain-text OCR scans (verify PD status per title) |
| Rejected | Post-1927 copyrighted books · Z-Library · Anna's Archive · Google Books snippets |
| Licence risk | 0.0 (PD/CC0) · 0.2 (CC-BY-SA Wikisource) |
| Quality | High (Standard Ebooks) · Medium-high (raw PG, OCR artifacts) |
| v1 token target | 12B tokens |
| Warehouse ceiling | 19B tokens |
| Cleanup | Strip PG header/footer · OCR artifact removal · paragraph reconstruction |
| Split strategy | 90/5/5 standard |
| In release v1 | YES |
| Warehouse | YES |
| Synthetic later | NO — real books are irreplaceable |
| Human review | Source-level validation only; paragraph-level auto-processed |
| Reliability | HIGH |

**Acquisition notes:**
- Project Gutenberg: `https://www.gutenberg.org/robot/harvest` — bulk plain-text download
- Standard Ebooks: `https://standardebooks.org/ebooks` — clean EPUB, extract with epub2txt
- Target: top 10,000 PG English texts by download count + full Standard Ebooks catalogue

---

### Category B — Educational Content

| Field | Value |
|---|---|
| Purpose | Structured explanation, concept definitions, worked examples, progressive teaching |
| Best sources | OpenStax (CC-BY 4.0) · Wikisource educational texts (CC-BY-SA) · CK-12 (CC-BY) |
| Acceptable | MIT OCW text materials (CC-BY-NC-SA) — **warehouse-only** |
| Rejected | Coursera/edX scrapes (ToS) · Copyrighted textbook PDFs · Khan Academy (CC-BY-NC) |
| Licence risk | 0.1 CC-BY · 0.2 CC-BY-SA · 0.4 CC-BY-NC-SA (warehouse-only) |
| Quality | High (OpenStax professionally edited) |
| v1 token target | 2B tokens |
| Warehouse ceiling | 3B tokens |
| Cleanup | Strip exercise answer blocks separately · Remove image captions · Reconstruct paragraphs |
| In release v1 | YES (OpenStax + CK-12 CC-BY only) |
| Warehouse | YES |
| Synthetic later | YES — COC-generated explanations of its own subsystems |
| Human review | Source-level + spot-check 5% of paragraphs |
| Reliability | HIGH |

**Acquisition notes:**
- OpenStax: free PDF/HTML download at `https://openstax.org/subjects` — all titles
- CK-12: `https://www.ck12.org/flexbooks/` — CC-BY FlexBooks
- Parse HTML versions; PDF is lossy

---

### Category C — Reasoning / STEM

| Field | Value |
|---|---|
| Purpose | Step-by-step mathematical reasoning, scientific analysis, causal thinking |
| Best sources | GSM8K (MIT) · MATH dataset (MIT) · ARC (CC-BY 4.0) · LogiQA (MIT) · NuminaMath (Apache 2.0) · StrategyQA (CC-BY) |
| Acceptable | MMLU training splits (CC-BY) · SciQ (CC-BY) |
| Rejected | BIG-Bench raw without curation · HELM benchmark (eval-restricted) |
| Licence risk | 0.0–0.1 |
| Quality | HIGH — all human-verified or competitively curated |
| v1 token target | 3B tokens |
| Warehouse ceiling | 3.5B tokens |
| Cleanup | Convert to `<reasoning>...</reasoning>` special-token format · Strip JSON scaffolding |
| In release v1 | YES — high priority |
| Warehouse | YES |
| Synthetic later | YES — COC reasoning chain generation |
| Human review | Format verification; content auto-processed |
| Reliability | HIGH |

**Acquisition notes:**
- HuggingFace `datasets` library: `load_dataset("gsm8k", "main")`, `load_dataset("math_dataset")`, `load_dataset("allenai/ai2_arc")`, `load_dataset("EleutherAI/logiqa")`, `load_dataset("AI-MO/NuminaMath-CoT")`
- All MIT/Apache/CC-BY — clean zero-risk

---

### Category D — Conversations / Instruction

| Field | Value |
|---|---|
| Purpose | Instruction-following, dialogue management, natural QA |
| Best sources | Dolly 15k (CC-BY-SA 3.0) · FLAN formatted (Apache 2.0) · OASST2 English subset (Apache 2.0) |
| Acceptable | Alpaca (CC-BY-NC 4.0) — **warehouse-only only** |
| Rejected | ShareGPT (unclear licence) · UltraChat (GPT-generated, ToS) · Any dataset using GPT-3.5/4 outputs |
| Licence risk | 0.1–0.2 (Dolly, FLAN, OASST2) · 0.4 CC-BY-NC (warehouse-only) |
| Quality | HIGH for Dolly (human-written) · MEDIUM for FLAN (template-derived) |
| v1 token target | 2B tokens |
| Warehouse ceiling | 2.5B tokens |
| Cleanup | Convert to `<user>...</user><assistant>...</assistant>` format · Filter responses < 15 words |
| Hard rule | Response generator must be human or explicitly open-source LLM |
| In release v1 | YES (Dolly + FLAN + OASST2 English only) |
| Warehouse | YES |
| Synthetic later | YES — best synthetic category; COC live conversations |
| Human review | 10% spot-check of approved items |
| Reliability | HIGH (Dolly) · MEDIUM (FLAN) |

---

### Category E — Technical Documentation

| Field | Value |
|---|---|
| Purpose | Precise structured writing, CS/ML factual grounding, retrieval anchor text |
| Best sources | Python stdlib docs (PSF — permissive) · PyTorch docs (BSD-3) · HuggingFace docs (Apache 2.0) · Wikipedia EN (CC-BY-SA 3.0) CS/math/science articles only |
| Acceptable | MDN Web Docs (CC-BY-SA 2.5) · Linux kernel docs prose (GPL-2 prose) |
| Rejected | AWS/Azure/GCP proprietary docs (ToS) · Stack Overflow at scale (attribution complexity) |
| Licence risk | 0.0–0.2 |
| Quality | HIGH (official authoritative sources) |
| v1 token target | 4B tokens |
| Warehouse ceiling | 6B tokens |
| Wikipedia filter | Article text only · CS, math, natural science, linguistics sections · Min 500 words · No stubs |
| Cleanup | Strip nav sidebars · version tables · API reference tables (keep explanatory prose) |
| In release v1 | YES |
| Warehouse | YES |
| Synthetic later | YES — COC-specific documentation examples |
| Human review | Source-level validation |
| Reliability | HIGH |

**Acquisition notes:**
- Wikipedia: Wikimedia dumps `https://dumps.wikimedia.org/enwiki/` — use `enwiki-latest-pages-articles.xml.bz2`
- Filter to CS / math / science categories using `mwparserfromhell` or `wikitextprocessor`
- Python docs: `pip download Sphinx` then `make html` from CPython source

---

### Category F — Long-Form Articles

| Field | Value |
|---|---|
| Purpose | Sustained paragraph coherence, argument development, factual grounding |
| Best sources | ArXiv abstracts + introductions (CC-BY 4.0 papers only) · Wikinews (CC-BY 2.5) · Project Gutenberg essays (PD) |
| Acceptable | The Conversation (CC-BY-ND — training use distinct from redistribution) — **warehouse-only pending legal review** |
| Rejected | Reuters/AP/BBC (ToS) · SEO blogs · BuzzFeed/listicles · News agency scrapes |
| Licence risk | 0.1 CC-BY · 0.5 CC-BY-ND (warehouse-only) |
| Quality | HIGH (ArXiv) · MEDIUM (Wikinews) |
| v1 token target | 2B tokens |
| Warehouse ceiling | 3B tokens |
| Cleanup | Strip author bio · Remove related-article links · Remove comment sections |
| In release v1 | YES (ArXiv CC-BY + Wikinews + PG essays) |
| Warehouse | YES |
| Synthetic later | NO |
| Human review | Licence verification per paper |
| Reliability | HIGH (ArXiv) · MEDIUM (Wikinews) |

---

### Category G — Research Papers

| Field | Value |
|---|---|
| Purpose | Highest-quality technical reasoning, claim-making, scientific analysis |
| Best sources | ArXiv full text (CC-BY 4.0 papers only) · ACL Anthology (CC-BY 4.0) · PubMed Open Access CC-BY subset |
| Acceptable | S2ORC CC-BY papers (extract CC-BY licensed subset) |
| Warehouse-only | S2ORC CC-BY-NC papers — NC clause prevents commercial release |
| Rejected | Elsevier/Springer without explicit CC · SciHub-sourced papers |
| Licence risk | 0.1 (CC-BY) |
| Filter | Min 5 citations for papers > 2 years old · Strip references + equations + figure captions |
| Quality | HIGH (peer-reviewed) |
| v1 token target | 3B tokens |
| Warehouse ceiling | 5B tokens |
| In release v1 | YES (CC-BY only) |
| Warehouse | YES (including CC-BY-NC subset with clear label) |
| Synthetic later | NO |
| Human review | Licence per-paper spot-check |
| Reliability | HIGH |

**Acquisition notes:**
- ArXiv S3 bulk access: `s3://arxiv/src/` — requires AWS account; licence is per-paper, filter CC-BY in metadata
- ACL Anthology: `https://aclanthology.org/anthology+abstracts.bib` + bulk PDF download
- PubMed OA: `https://www.ncbi.nlm.nih.gov/pmc/tools/ftp/` — `oa_comm` subfolder is CC licenced

---

### Category H — COC Synthetic Data

| Field | Value |
|---|---|
| Purpose | Train COC on its own operational patterns — memory, retrieval, KG, workflows |
| Source | `dataset/generator.py` — live session capture |
| Types | conversation · retrieval · kg · memory · teaching · emotion · reasoning · evaluation |
| Licence | Internal / CC0 — zero risk |
| Quality | MEDIUM initially → HIGH after review and curation |
| v1 token target | 0.5B tokens (reviewed) |
| Warehouse ceiling | 2B tokens (grows with usage) |
| Hard rules | NEVER auto-approved · ALWAYS human review · Versioned separately · Min quality 0.70 |
| In release v1 | YES — after mandatory review |
| Warehouse | YES |
| Synthetic later | YES — primary expansion path |
| Human review | MANDATORY — 100% of synthetic items |
| Reliability | MEDIUM — depends on live session quality |

---

### Category I — Human Cognition Material

| Field | Value |
|---|---|
| Purpose | Supports the 9-module Human Cognition Layer |
| Best sources | PG psychology classics (PD) — James, Wundt, Dewey · OpenStax Psychology (CC-BY) · CogSci proceedings (CC-BY) |
| Acceptable | PubMed OA neuroscience CC-BY papers |
| Rejected | Popular psychology blogs · Self-help content |
| Licence risk | 0.0–0.1 |
| Quality | HIGH (PD + peer-reviewed) |
| v1 token target | 0.5B tokens |
| Warehouse ceiling | 1B tokens |
| Cleanup | Strip clinical case notes (PII risk) · Strip questionnaire scoring tables |
| In release v1 | YES |
| Warehouse | YES |
| Synthetic later | YES — COC emotion + teaching module scenario generation |
| Human review | Source-level |
| Reliability | HIGH |

---

### Category J — Retrieval Material

| Field | Value |
|---|---|
| Purpose | Trains retrieval-grounded responses, citation patterns, "according to..." framing |
| Best sources | MS MARCO (MIT) · Natural Questions (CC-BY-SA 3.0) · TriviaQA (Apache 2.0) · HotpotQA (CC-BY-SA 4.0) |
| Acceptable | QuALITY (CC-BY 4.0) |
| Rejected | WebQuestions raw (Freebase, outdated) · SQuAD 2.0 unanswerable (less relevant) |
| Licence risk | 0.0–0.2 |
| Format | `<retrieval>query</retrieval><passage>text</passage><answer>answer</answer>` |
| Quality | HIGH |
| v1 token target | 0.5B tokens |
| Warehouse ceiling | 1B tokens |
| In release v1 | YES |
| Warehouse | YES |
| Synthetic later | YES — COC-indexed document QA generation |
| Human review | Format verification |
| Reliability | HIGH |

---

### Category K — Knowledge Graph Material

| Field | Value |
|---|---|
| Purpose | Structured factual generation, triple extraction training |
| Best sources | Wikidata text (CC0) · DBpedia abstracts (CC-BY-SA 3.0) · ConceptNet (CC-BY-SA 4.0) |
| Acceptable | Freebase-derived text (CC-BY 4.0 Google release) |
| Rejected | Raw Wikidata JSON (too structured without transformation) |
| Licence risk | 0.0–0.2 |
| Format | `<kg>entity: description. entity relation entity.</kg>` |
| Quality | MEDIUM (auto-generated from structured data — verbose but factual) |
| v1 token target | 0.5B tokens |
| Warehouse ceiling | 1.5B tokens |
| In release v1 | YES |
| Warehouse | YES |
| Synthetic later | YES — COC KG extraction examples |
| Human review | Source-level |
| Reliability | MEDIUM-HIGH |

---

### Category L — Evaluation / Holdout Material

| Field | Value |
|---|---|
| Purpose | Reserved for final model evaluation only — never in training |
| Sources | ARC test split · MMLU test split · HellaSwag test split · WinoGrande test split |
| Licence | CC-BY / MIT |
| Training | NEVER — permanent holdout |
| Storage | `warehouse/holdout/` — separate from all training data |
| Notes | These benchmark test splits are included in the warehouse as references for post-training evaluation only |

---

## 3. Licence / Risk Matrix

| ID | Licence | Risk Score | Training Use | v1 Release | Warehouse |
|---|---|---|---|---|---|
| pd | Public Domain | 0.0 | YES | YES | YES |
| cc0 | CC0 1.0 | 0.0 | YES | YES | YES |
| mit | MIT | 0.05 | YES | YES | YES |
| apache2 | Apache 2.0 | 0.05 | YES | YES | YES |
| bsd3 | BSD 3-Clause | 0.05 | YES | YES | YES |
| psf | PSF License | 0.05 | YES | YES | YES |
| cc-by-4 | CC-BY 4.0 | 0.10 | YES (attribute) | YES | YES |
| cc-by-3 | CC-BY 3.0 | 0.10 | YES (attribute) | YES | YES |
| cc-by-sa-4 | CC-BY-SA 4.0 | 0.20 | YES (document SA) | YES | YES |
| cc-by-sa-3 | CC-BY-SA 3.0 | 0.20 | YES (document SA) | YES | YES |
| cc-by-nc-4 | CC-BY-NC 4.0 | 0.40 | YES non-commercial | **NO** | YES |
| cc-by-nc-sa | CC-BY-NC-SA 4.0 | 0.45 | YES non-commercial | **NO** | YES |
| cc-by-nd | CC-BY-ND | 0.50 | Training OK†, no redistrib | **NO** | Review |
| gpl2 | GPL v2 | 0.30 | Prose only | YES prose | YES |
| unknown | Unknown / None | 0.85 | Human review | **NO** | Review |
| ars | All Rights Reserved | 1.00 | **NO** | **NO** | **NO** |
| tos-violation | ToS violation | 1.00 | **NO** | **NO** | **NO** |

†Training on ND text is legally distinct from redistributing the text; however, out of caution, ND sources are warehouse-only pending explicit legal review.

**Zero-risk release v1 ceiling:** Only licences with risk ≤ 0.20 enter the release.

---

## 4. Quality Scoring and Acceptance Rules

### Quality score formula (0.0 – 1.0)

```
quality_score = (
    word_count_score    × 0.20   # target 20–500 words per paragraph
  + lexical_richness    × 0.20   # unique/total word ratio (type-token ratio)
  + sentence_variety    × 0.15   # avg words/sentence, target 10–30
  + printable_ratio     × 0.10   # printable chars / total chars
  + information_density × 0.15   # domain keyword density (STEM, cognition, etc.)
  + pii_clean           × 0.10   # 1.0 if no PII detected, 0.0 if PII present
  + non_repetitive      × 0.10   # 1.0 if top-3 words < 30% of total
)
```

### Category score formula (0.0 – 1.0)

Per-category vocabulary density signals (see `corpus/scorer.py`):
- Reasoning: causal language, conditionals, if/then structures
- Technical: domain terminology (STEM word lists)
- Instruction: QA structure, step markers, worked examples
- Emotion/cognition: emotional and psychological vocabulary
- Retrieval: citation language, "according to", source references
- KG: entity-relation language, "is a", "consists of", definitions

### Acceptance thresholds

| Condition | Decision |
|---|---|
| quality ≥ 0.70 AND risk ≤ 0.20 | **Auto-approve** |
| quality 0.45–0.70 OR risk 0.20–0.50 | **Human review queue** |
| quality < 0.45 OR risk > 0.50 | **Auto-reject** |
| Is synthetic (any score) | **Always human review** |
| Synthetic quality < 0.70 | **Auto-reject regardless** |

### Quality dimensions tracked

| Dimension | What it measures | How scored |
|---|---|---|
| Lexical richness | Vocabulary diversity | Type-token ratio ÷ 0.5 |
| Information density | Domain keyword presence | Regex density per category |
| Structure quality | Sentence length distribution | Std dev of sentence lengths |
| Reasoning value | Causal / logical language | Reasoning vocab density |
| Teaching value | Explanation markers | Instruction vocab density |
| Retrieval value | Citation and sourcing language | Retrieval vocab density |
| KG value | Entity-relation language | KG vocab density |
| Cognition value | Emotional / psychological terms | Emotion vocab density |
| Duplication risk | MinHash Jaccard similarity | Cross-source dedup score |
| Safety risk | PII / harmful content | Regex + classifier |
| Licence risk | From source registry | Lookup from registry |

---

## 5. Corpus Warehouse Design

### Design principles

1. **Append-only:** Raw files are never modified after acquisition
2. **Separation of stages:** Each pipeline stage writes to its own directory
3. **Source-level manifests:** Every source has a manifest before anything is processed
4. **Immutable releases:** Once built and signed, release artifacts are never overwritten
5. **Audit at every stage:** Every action writes to the audit log before completing

### Warehouse structure

```
D:/corpus_warehouse/                    ← CORPUS_WAREHOUSE_DIR in config.py
│
├── raw/                                ← Stage 1: immutable source archives
│   ├── books/                          ← Category A
│   ├── educational/                    ← Category B
│   ├── reasoning/                      ← Category C
│   ├── conversations/                  ← Category D
│   ├── technical_docs/                 ← Category E
│   ├── articles/                       ← Category F
│   ├── research_papers/                ← Category G
│   ├── synthetic/                      ← Category H (COC-generated)
│   ├── cognition/                      ← Category I
│   ├── retrieval/                      ← Category J
│   ├── kg/                             ← Category K
│   └── holdout/                        ← Category L (eval only — never training)
│
├── cleaned/                            ← Stage 4: boilerplate-removed UTF-8 text
│   └── <same subdirs as raw/>
│
├── deduplicated/                       ← Stage 5: within-source dedup output
│   └── <same subdirs>
│
├── scored/                             ← Stage 6: paragraph + score metadata
│   └── <source_id>_scores.jsonl
│
├── review_queue/                       ← Stage 7: human review items
│   ├── pending.jsonl
│   ├── approved.jsonl
│   └── rejected.jsonl
│
├── approved/                           ← Stage 8: release-eligible text
│   └── <same subdirs>
│
├── rejected/                           ← Failed content (kept for audit only)
│   └── <same subdirs>
│
├── synthetic/                          ← Versioned COC synthetic data
│   ├── v1/
│   └── v2/
│
├── manifests/                          ← Per-source manifests
│   └── <source_id>.json
│
├── releases/                           ← Assembled training releases
│   └── v1/
│       ├── train.txt
│       ├── val.txt
│       ├── test.txt
│       ├── manifest.json
│       ├── checksums.sha256
│       └── LOCK
│
├── governance_logs/                    ← Daily audit JSONL shards
│   └── audit_YYYYMMDD.jsonl
│
└── archive/                            ← Recalled releases, retired sources
    └── recalled/
```

### What lives in repo vs SSD

| Item | Location | Notes |
|---|---|---|
| All corpus code | `corpus/` in repo | Python modules |
| Source registry | `governance/source_registry.json` in repo | Version-tracked in git |
| Approval log | `governance/approval_log.jsonl` in repo | Append-only, committed |
| Licence rules | `governance/license_rules.json` in repo | Committed |
| Raw text archives | SSD `raw/` | Never in repo |
| Cleaned/deduped text | SSD `cleaned/`, `deduplicated/` | Never in repo |
| Release artifacts | SSD `releases/` | Never in repo |
| Audit logs (daily shards) | SSD `governance_logs/` | Never in repo |
| Model checkpoints | `var/checkpoints/` in repo | Not committed |
| Training ledger | `var/checkpoints/training_ledger.jsonl` | Not committed |
| Shard tracker | `var/checkpoints/shard_tracker.json` | Not committed |

---

## 6. Training Release Design

### What a release is

A release is a **versioned, checksummed, locked, reproducible snapshot** of approved corpus paragraphs. It is the only data that may enter `train/train_model.py`.

### Release lifecycle

```
WAREHOUSE (approved paragraphs)
    ↓ corpus build-release v1
RELEASE DRAFT
    ↓ corpus verify-release v1  (checksums + leakage check)
RELEASE VERIFIED
    ↓ corpus sign-release v1 --operator mpssp
RELEASE SIGNED  (manifest.status = "signed")
    ↓ lock-release v1 --operator mpssp
RELEASE LOCKED  (LOCK file written with artifact hashes)
    ↓ resume-verify v1 --fresh  (pre-training guard)
TRAINING STARTS
```

After `lock-release`, any modification to `train.txt`, `val.txt`, `test.txt`, `manifest.json`, or `checksums.sha256` is detected by the resume guard and training is aborted.

### Release v1 composition

| Category | Tokens | % of release |
|---|---|---|
| A — Books PD/CC0 | 12.0B | 40.0% |
| E — Technical docs (Wikipedia + official docs) | 4.0B | 13.3% |
| G — Research papers (CC-BY only) | 3.0B | 10.0% |
| C — Reasoning/STEM | 3.0B | 10.0% |
| B — Educational (OpenStax + CK-12) | 2.0B | 6.7% |
| F — Long-form articles (ArXiv + Wikinews) | 2.0B | 6.7% |
| D — Conversations (Dolly + FLAN + OASST2) | 2.0B | 6.7% |
| I — Cognition (PG psychology + OpenStax Psych) | 0.5B | 1.7% |
| J — Retrieval (MSMARCO + NQ + TriviaQA) | 0.5B | 1.7% |
| K — KG material (Wikidata + DBpedia + ConceptNet) | 0.5B | 1.7% |
| H — COC synthetic (reviewed) | 0.5B | 1.7% |
| **Total** | **30B** | **100%** |

**Dominant category is A (books) at 40%.** This is intentional: long-form coherent prose is the hardest thing for a 700M model to learn and the most important for COC's use cases.

### Release mixing curriculum

To prevent any single source from dominating early training:

1. Shuffle all approved paragraphs with seed=42 (full random mix)
2. Apply token budget cap at 30B
3. Split: 90% train / 5% val / 5% test
4. Val and test are stratified: each category contributes proportionally

### Why 30B and not 40B

The zero-risk ceiling for release v1 is ~30B tokens from the above sources. The Chinchilla-optimal ratio for a 700M model is ~14 tokens/parameter = 9.8B tokens. We exceed this by 3×, giving the model substantial repetition at the cost of zero licence risk. This is the correct trade-off.

If more data is needed later, release v2 can be built from additional zero-risk sources already in the warehouse.

---

## 7. Storage / Size Plan

### 1TB SSD budget

| Partition | Raw size | Notes |
|---|---|---|
| Raw source archives | 150 GB | PG texts, PDFs, JSONL datasets |
| Cleaned text | 110 GB | ~73% of raw after markup/noise removal |
| Deduplicated | 65 GB | ~59% of cleaned after dedup |
| Score metadata (JSONL) | 5 GB | Paragraph scores |
| Review queue files | 1 GB | Pending/approved/rejected JSONL |
| Approved text | 55 GB | ~85% of deduped passes quality gate |
| Tokenized release (uint16) | 60 GB | 30B tokens × 2 bytes |
| Synthetic data | 5 GB | COC-generated, versioned |
| Model checkpoints | 60 GB | 700M × bf16 × 40 saves |
| Release artifacts | 20 GB | Train/val/test + manifests × versions |
| Holdout eval data | 5 GB | Category L — eval only |
| Governance logs | 2 GB | Daily audit shards |
| Archive | 20 GB | Recalled/retired versions |
| Reserve / working space | **442 GB** | Pipeline temp files, future releases |
| **Total used** | **558 GB** | **442 GB headroom — comfortable** |

**Conclusion: 1TB SSD is sufficient. The warehouse ceiling (~35B zero-risk tokens clean) fits in ~55 GB of approved text. No compression required.**

### Per-source storage estimates

| Source family | Raw size | Clean size | Tokens |
|---|---|---|---|
| Project Gutenberg top 10K | 25 GB | 18 GB | 4.7B |
| Standard Ebooks complete | 8 GB | 6 GB | 1.6B |
| Remaining PG long-tail | 40 GB | 28 GB | 7.4B |
| Wikipedia EN (filtered) | 22 GB | 16 GB | 4.2B |
| ArXiv CC-BY papers | 18 GB | 12 GB | 3.2B |
| ACL Anthology | 4 GB | 2.5 GB | 0.7B |
| PubMed OA CC-BY | 18 GB | 12 GB | 3.2B |
| OpenStax + CK-12 | 1.5 GB | 1 GB | 0.3B |
| HuggingFace datasets (C/D/J/K) | 3 GB | 2 GB | 0.5B |
| Python/PyTorch/HF docs | 1 GB | 0.7 GB | 0.2B |
| PG psychology + cognition | 1 GB | 0.7 GB | 0.2B |
| COC synthetic | 2 GB | 1.5 GB | 0.4B |
| **Total** | **143.5 GB** | **100.4 GB** | **~26.6B*** |

*After cross-source deduplication removes ~10% overlap, net approved tokens ≈ **24B–30B** depending on quality gate stringency.

### Training time estimates

| Steps | Tokens consumed | Days @ 16h/day (2000 tok/s) |
|---|---|---|
| 10,000 | 2.6B | 26 |
| 30,000 | 7.9B | 78 |
| 60,000 | 15.8B | 156 |
| 100,000 | 26.2B | 261 |
| **Full 30B epoch** | 30B | **~300** |

**Realistic plan:** Run to 100K steps (~261 days) for near-full corpus coverage. Evaluate at 30K, 60K, 100K steps using the phase gate system.

---

## 8. Folder Structure

### Repository (`cognitiveoc_v3/`)

```
cognitiveoc_v3/
├── corpus/                   ← Corpus pipeline code
│   ├── __init__.py
│   ├── cleaner.py            ← 15 category-specific cleaners
│   ├── cli.py                ← 16 corpus CLI subcommands
│   ├── dedup.py              ← MinHash LSH cross-source dedup
│   ├── manifest.py           ← Manifest generation + validation
│   ├── release_builder.py    ← Release assembly, verify, sign
│   ├── reviewer.py           ← Human review queue
│   ├── scorer.py             ← Quality + category + risk scoring
│   ├── source_registry.py    ← CRUD on source_registry.json
│   └── warehouse.py          ← Directory management + stats
│
├── governance/               ← Governance data (version-tracked)
│   ├── __init__.py
│   ├── approval_log.jsonl    ← Append-only approval history
│   ├── audit.py              ← High-level audit interface
│   ├── license_rules.json    ← Licence risk matrix
│   └── source_registry.json  ← All registered sources
│
├── audit/                    ← Audit logging
│   ├── __init__.py
│   ├── logger.py             ← Structured JSONL event writer
│   └── reporter.py           ← Report generator
│
├── release/                  ← Release management
│   ├── __init__.py
│   ├── lock.py               ← Release locking + integrity
│   ├── verify.py             ← Verification enforcement
│   └── v1/
│       └── manifest.json     ← Placeholder; populated by build-release
│
├── train/                    ← Training control
│   ├── __init__.py
│   ├── training_ledger.py    ← Append-only session ledger
│   ├── shard_tracker.py      ← Shard consumption tracker
│   ├── resume_guard.py       ← Pre-training integrity gate
│   ├── provenance.py         ← Permanent training provenance
│   └── train_model.py        ← 700M training loop [FROZEN]
│
├── docs/
│   ├── GUIDE.md              ← Complete user + developer guide
│   └── CORPUS_MASTER.md      ← This document
│
└── config.py                 ← CORPUS + TRAINING_CONTROL config blocks
```

---

## 9. Manifest Schema

### Source manifest (`warehouse/manifests/<source_id>.json`)

```json
{
  "manifest_version": "1.0",
  "coc_version": "v3",
  "source_id": "A-gutenberg-20260701",
  "category": "A",
  "name": "Project Gutenberg English — Top 10K texts",
  "url": "https://gutenberg.org",
  "licence": "Public Domain",
  "licence_id": "pd",
  "licence_risk": 0.0,
  "created": "2026-07-01T14:00:00",
  "acquired_by": "mpssp",
  "approved_by": "mpssp",
  "paths": {
    "raw": "D:/corpus_warehouse/raw/books/gutenberg/",
    "cleaned": "D:/corpus_warehouse/cleaned/books/",
    "deduped": "D:/corpus_warehouse/deduplicated/books/",
    "approved": "D:/corpus_warehouse/approved/books/"
  },
  "pipeline_stats": {
    "n_raw_files": 10000,
    "n_paragraphs_raw": 8500000,
    "n_paragraphs_clean": 7200000,
    "n_paragraphs_deduped": 6800000,
    "n_paragraphs_approved": 5900000,
    "retention_pct": 69.4
  },
  "score_summary": {
    "avg_quality": 0.74,
    "avg_risk": 0.0,
    "auto_approve": 5200000,
    "human_review": 420000,
    "auto_reject": 1380000
  },
  "tokens_estimate": 12000000000,
  "sha256_approved": "abc123..."
}
```

### Release manifest (`releases/v1/manifest.json`)

```json
{
  "manifest_version": "1.0",
  "coc_version": "v3",
  "release_id": "v1",
  "release_date": "2026-09-01",
  "released_by": "mpssp",
  "signed_by": "mpssp",
  "signed_at": "2026-09-01T10:00:00",
  "status": "signed",
  "tokenizer": "48K-SentencePiece-Unigram",
  "total_tokens_estimate": 30000000000,
  "train_tokens": 27000000000,
  "val_tokens":   1500000000,
  "test_tokens":  1500000000,
  "split_ratios": [0.90, 0.05, 0.05],
  "shuffle_seed": 42,
  "categories_included": ["A","B","C","D","E","F","G","H","I","J","K"],
  "zero_risk_policy": true,
  "max_licence_risk_included": 0.20,
  "sources": [
    {
      "source_id": "A-gutenberg-20260701",
      "category": "A",
      "token_contribution": 12000000000,
      "licence": "Public Domain",
      "licence_id": "pd",
      "licence_risk": 0.0,
      "approved_by": "mpssp",
      "approval_date": "2026-08-15"
    }
  ],
  "quality_gate": {
    "min_quality_score": 0.45,
    "max_risk_score": 0.20,
    "all_sources_validated": true,
    "all_synthetic_reviewed": true,
    "cross_source_dedup_complete": true,
    "leakage_check_passed": true
  },
  "checksums": {
    "train": "sha256:...",
    "val":   "sha256:...",
    "test":  "sha256:..."
  }
}
```

---

## 10. Cleaning Pipeline Design

### Stage 3 — Normalize

**Purpose:** Convert all source formats to clean UTF-8 plain text paragraphs.

| Input format | Tool | Notes |
|---|---|---|
| `.txt` (Gutenberg) | Direct read | Strip PG header/footer first |
| `.epub` (Standard Ebooks) | `epub2txt` or `ebooklib` | Extract chapter prose |
| `.pdf` (OpenStax, papers) | `pdfminer.six` or `pymupdf` | Per-page text extraction |
| `.html` (docs, Wikinews) | `html2text` or `BeautifulSoup` | Strip nav/sidebars first |
| `.jsonl` (HuggingFace datasets) | Direct JSON parse | Use `text` / `document` field |
| `.xml` (Wikipedia dump) | `mwparserfromhell` | Strip wikitext templates |

All output: UTF-8, NFC normalised, CRLF → LF, one paragraph per text block separated by double newline.

### Stage 4 — Clean

**Purpose:** Remove low-signal content from normalised text.

Operations (in order):
1. PG boilerplate strip (Gutenberg header/footer regex)
2. HTML/XML tag removal (final pass)
3. Running header/footer collapse (repeated lines across pages)
4. Figure/table caption removal (regex patterns)
5. Reference list removal (papers)
6. Equation replacement (`[equation]` placeholder for LaTeX)
7. URL removal (bare URLs without context)
8. PII scan and redaction (email, SSN, phone, CC, API key patterns)
9. Minimum word count filter (< 8 words → discard)
10. Degenerate content removal (all numbers, all whitespace, all repeated chars)

Category-specific cleaners are in `corpus/cleaner.py`:
- `clean_gutenberg()` — PG-specific
- `clean_openstax()` — exercise separation
- `clean_wikipedia()` — template removal, stub detection
- `clean_arxiv_full()` — references + equations
- `clean_dolly()`, `clean_oasst2()`, `clean_flan()` — instruction format conversion
- `clean_msmarco()`, `clean_nq()` — retrieval format
- `clean_dbpedia()`, `clean_conceptnet()` — KG format
- `clean_generic()` — fallback for uncategorised sources

---

## 11. Deduplication Pipeline Design

### Two deduplication levels

**Level 1 — Within-source (Stage 5, via `data/pipeline.py::dedup()`)**
- Cosine similarity threshold: 0.92
- Applied per source family immediately after cleaning
- Removes reprintings, OCR duplicates, same passage in different chapters

**Level 2 — Cross-source (Stage 5, via `corpus/dedup.py::CrossSourceDeduper`)**
- MinHash LSH, 128 hash functions, 5-gram character shingles
- Jaccard similarity threshold: 0.85
- Applied across ALL source families before release assembly
- Removes: same OpenStax passage in Wikipedia + CK-12 + textbook mirror
- Priority: first-registered source wins (earlier source_id in registry)

**Level 3 — Release leakage check (Stage 9, via `release/verify.py`)**
- Exact SHA-256 paragraph hash comparison
- Between train and val; between train and test
- Zero tolerance — any leakage blocks the release build
- Also checks holdout (Category L) against all training data

### MinHash design

```
128 hash functions, Mersenne prime modular hash
5-gram character shingles (robust to word-order variation)
32 LSH bands × 4 rows/band
Jaccard threshold 0.85 → ~99% true positive rate at threshold
Pure Python fallback + datasketch acceleration if available
```

### Deduplication storage

| Stage | Output | Location |
|---|---|---|
| Within-source dedup | `<source_id>.txt` | `warehouse/deduplicated/<category>/` |
| Cross-source dedup report | `<date>_xsource_report.json` | `warehouse/governance_logs/` |
| Leakage report | Embedded in release manifest | `releases/v1/manifest.json` |

---

## 12. Synthetic Data Strategy

### Rule set (non-negotiable)

1. **Never auto-approved.** 100% human review, regardless of score.
2. **Never mixed with raw data.** Lives in `warehouse/synthetic/v<N>/` only.
3. **Minimum quality 0.70.** Higher bar than real data to prevent template drift.
4. **Versioned separately.** Each synthetic batch has its own version directory.
5. **Never from closed models.** Only COC's own components generate synthetic data.
6. **Proportion cap: 3.3% of release.** 1B / 30B = 3.3%. Prevents synthetic drift.

### Generation types and volume targets

| Type | Generator | Target v1 | Format |
|---|---|---|---|
| `conversation` | Engine live sessions | 200M tokens | `<user>...</user><assistant>...</assistant>` |
| `retrieval` | RAG pipeline | 100M tokens | `<retrieval>...</retrieval>` |
| `kg` | KG extractor | 50M tokens | `<kg>...</kg>` |
| `memory` | Memory system | 50M tokens | Store/recall pairs |
| `teaching` | Cognition layer | 50M tokens | Levelled explanation pairs |
| `emotion` | Cognition layer | 30M tokens | Text + 28-class label |
| `reasoning` | Reasoner | 20M tokens | `<reasoning>...</reasoning>` |
| **Total** | | **500M tokens** | |

### Review workflow for synthetic data

```bash
# 1. Live sessions auto-capture to var/datasets/
python main.py chat  # or use desktop/web UI

# 2. Register synthetic batch
python main.py corpus register-source H-coc-synthetic-v1 \
  --category H --licence "Internal/CC0" --licence-id cc0 \
  --licence-risk 0.0 --raw-path "var/datasets/" --operator mpssp

# 3. Validate
python main.py corpus validate-source H-coc-synthetic-v1 --operator mpssp

# 4. Pipeline (score only — no re-cleaning needed for synthetic)
python main.py corpus run-pipeline H-coc-synthetic-v1 --stages score

# 5. MANDATORY review — every item
python main.py corpus review --source H-coc-synthetic-v1 --operator mpssp
```

---

## 13. Train / Val / Test Strategy

### Split ratios

| Split | Ratio | Purpose |
|---|---|---|
| Train | 90% | Model pre-training |
| Val | 5% | Per-step validation loss (every 500 steps) |
| Test | 5% | Final evaluation only — never seen during training |

### Split seed

Fixed: `42`. Never changes between releases. Changing the seed would make different runs incomparable.

### Stratification

Val and test are stratified by category: every category contributes proportionally. This prevents a category from being absent from evaluation.

```python
# Implemented in corpus/release_builder.py::build()
# After global shuffle, sample proportionally per category for val and test
```

### Holdout policy (Category L)

The benchmark test sets (ARC, MMLU, HellaSwag, WinoGrande) are stored in `warehouse/holdout/` and are **never included in any training split**. They are used only by `eval/run_suite.py` after training.

The leakage check at release build time verifies that no holdout paragraph appears in train/val/test.

### Reproducibility guarantee

Given the same:
- Source manifests (same approved paragraphs)
- Split seed (42)
- Token budget (30B)

The same `train.txt`, `val.txt`, `test.txt` will be produced every time. This is verified by the release checksum.

---

## 14. Corpus Versioning Strategy

### Source versioning

Each source has a unique `source_id` in the format: `<category>-<short_name>-<YYYYMMDD>`

Example: `A-gutenberg-20260701`

Source records are append-only in `governance/source_registry.json`. A source is never deleted — only marked `rejected` or `archived`.

### Release versioning

Releases are monotonically numbered: `v1`, `v2`, `v3`, ...

Each release is immutable once signed. A new release is a new version, not a modification.

### Registry versioning

The registry hash (SHA-256 of the sources list) is recomputed on every save. This provides a deterministic fingerprint of the corpus state at any point in time.

### Rollback strategy

If a release is found to contain bad data:
1. `python main.py unlock-release v1 --operator mpssp --reason "Bad source X detected"`
2. This archives the LOCK file (not deleted — kept for audit)
3. Reject the offending source: `python main.py corpus validate-source <id> --reject`
4. Rebuild: `python main.py corpus build-release v2 --categories ...`
5. Sign and lock v2
6. Resume training from v2

The original v1 release artifacts are never deleted — moved to `warehouse/archive/recalled/`.

---

## 15. Corpus Governance Rules

### The twelve controls (no stage may be skipped silently)

| # | Control | Who | When | Blocking |
|---|---|---|---|---|
| 1 | Source registration | Human | Acquisition | YES |
| 2 | Licence validation | Human + lookup | Acquisition | YES |
| 3 | Integrity check (SHA-256) | Automated | Acquisition + release | YES |
| 4 | Language check | Automated | Normalize | YES |
| 5 | PII scan | Automated | Clean | YES |
| 6 | Boilerplate removal | Automated | Clean | Logged |
| 7 | Deduplication | Automated | Dedup | Logged |
| 8 | Quality scoring | Automated | Score | YES (< 0.45) |
| 9 | Category scoring | Automated | Score | Informational |
| 10 | Risk scoring | Automated | Score | YES (> 0.50) |
| 11 | Human review | Human | Review | YES (synthetic + borderline) |
| 12 | Release approval | Human release officer | Release | YES — mandatory |

### Release approval policy

A release may only be signed if ALL of the following are true:
- All sources in the release have `status = "approved"` in the registry
- All synthetic sources have 100% human-reviewed items
- Cross-source deduplication has been run
- Leakage check has passed
- No source has `licence_risk > 0.20` (for zero-risk v1 policy)

The signing event is logged to `governance/approval_log.jsonl` with: timestamp, operator, release_id, manifest_hash.

### Audit log event types

```
acquire          source_registered | source_acquired
validate         source_validated | source_rejected
normalize        normalize_complete
clean            clean_complete
dedup            dedup_complete
score            score_complete
review           review_approve | review_reject
approve          source_approved
release          release_built | release_signed | release_locked | release_recalled
resume_guard     guard_check (ok | fail)
training         run_open | run_close | shard_complete | shard_reset
```

---

## 16. Project Integration Plan

### What is already wired (verified)

All of the following exist and import clean:

| Module | Status |
|---|---|
| `corpus/cleaner.py` | ✅ 15 category cleaners |
| `corpus/scorer.py` | ✅ 3-axis scoring |
| `corpus/dedup.py` | ✅ MinHash LSH |
| `corpus/reviewer.py` | ✅ JSONL queue + interactive CLI |
| `corpus/manifest.py` | ✅ Source + release manifests |
| `corpus/warehouse.py` | ✅ Directory management + stats |
| `corpus/release_builder.py` | ✅ Build + verify + sign |
| `corpus/source_registry.py` | ✅ CRUD on registry JSON |
| `corpus/cli.py` | ✅ 16 subcommands wired |
| `governance/source_registry.json` | ✅ Empty, valid schema |
| `governance/license_rules.json` | ✅ 18 licence types |
| `governance/approval_log.jsonl` | ✅ Append-only |
| `audit/logger.py` | ✅ Dual-write + file lock |
| `audit/reporter.py` | ✅ Activity + compliance reports |
| `release/lock.py` | ✅ Lock + verify + break |
| `release/verify.py` | ✅ Full verification enforcement |
| `train/training_ledger.py` | ✅ Append-only session ledger |
| `train/shard_tracker.py` | ✅ Consumption tracker |
| `train/resume_guard.py` | ✅ 8-point pre-training gate |
| `train/provenance.py` | ✅ Permanent provenance record |
| `config.py CORPUS block` | ✅ All paths + thresholds |
| `config.py TRAINING_CONTROL block` | ✅ Shard size + abort flags |
| `main.py` all corpus commands | ✅ 17 top-level + 16 corpus sub |

### What still needs to happen (operational, not code)

| Item | Who | When |
|---|---|---|
| Set `CORPUS_WAREHOUSE_DIR` in `config.py` to actual SSD path | mpssp | Day 1 |
| Download Project Gutenberg bulk text | mpssp | Week 1 |
| Download Standard Ebooks catalogue | mpssp | Week 1 |
| Download OpenStax HTML titles | mpssp | Week 2 |
| Download Wikipedia EN dump + filter | mpssp | Week 2 |
| Download ArXiv CC-BY papers | mpssp | Week 3 |
| Download ACL Anthology | mpssp | Week 3 |
| Download PubMed OA CC-BY | mpssp | Week 3 |
| Download HuggingFace datasets (C/D/J/K) | mpssp | Week 2 |
| Register + pipeline each source | mpssp | Weeks 4–8 |
| Generate COC synthetic data | mpssp | Weeks 6–8 |
| Build, sign, lock release v1 | mpssp | Week 9 |
| Train tokenizer on v1 | mpssp | Week 9 |
| Begin pre-training | mpssp | Week 10 |

---

## 17. Exact Files to Add / Modify

### Files already created (all wired, all syntax-clean)

No new files are needed. All 20 required modules from the master prompt exist.

### Config additions still needed

Add to `config.py` after the existing `CORPUS` dict — the zero-risk policy enforcement constants:

```python
# Zero-risk release policy
CORPUS_POLICY = dict(
    max_licence_risk_v1      = 0.20,   # Hard ceiling for release v1
    max_licence_risk_warehouse = 0.50, # Warehouse-only ceiling
    require_explicit_licence = True,   # Reject sources with no licence field
    require_source_validated = True,   # Block pipeline on unvalidated sources
    synthetic_proportion_cap = 0.033,  # Max 3.3% synthetic in any release
    holdout_categories       = ["L"],  # Never in training
    zero_risk_licences       = ["pd", "cc0", "mit", "apache2", "bsd3", "psf",
                                 "cc-by-4", "cc-by-3", "cc-by-sa-4", "cc-by-sa-3"],
)
```

---

## 18. CLI Commands

### Complete corpus command reference

```bash
# === SOURCE MANAGEMENT ===
python main.py corpus register-source <source_id> \
  --category <A-L> --name "..." --url "..." \
  --licence "..." --licence-id <id> --licence-risk <0.0-1.0> \
  --raw-path "<path>" --raw-sha256 "" --operator mpssp

python main.py corpus validate-source <source_id> --operator mpssp
python main.py corpus validate-source <source_id> --reject --reason "..." --operator mpssp

# === PIPELINE ===
python main.py corpus run-pipeline <source_id> \
  --stages clean,dedup,score --operator mpssp --verbose

# === REVIEW ===
python main.py corpus review --source <source_id> --operator mpssp
python main.py corpus review --stats

# === RELEASE ===
python main.py corpus build-release v1 --dry-run
python main.py corpus build-release v1 \
  --categories A,B,C,D,E,F,G,H,I,J,K \
  --token-budget 30000000000
python main.py corpus verify-release v1
python main.py corpus sign-release v1 --operator mpssp
python main.py corpus list-releases

# === LOCKING ===
python main.py lock-release v1 --operator mpssp --notes "Phase 1 pre-training"
python main.py unlock-release v1 --operator mpssp --reason "Bad source recall"

# === WAREHOUSE ===
python main.py corpus warehouse-stats

# === AUDIT ===
python main.py corpus audit-report
python main.py corpus audit-report --source <source_id>
python main.py corpus audit-report --from 2026-07-01 --to 2026-08-31

# === TRAINING CONTROL ===
python main.py resume-verify v1 --fresh --operator mpssp   # first session
python main.py resume-verify v1 --operator mpssp           # resume
python main.py training-ledger
python main.py shard-status
python main.py provenance-report
```

### Acquisition helper commands (run separately, not via main.py)

```bash
# Project Gutenberg bulk download
# Install: pip install gutenberg
python -c "
from gutenberg.acquire import load_etext
from gutenberg.cleanup import strip_headers
# Or use the mirror: rsync -av --del ftp@ftp.ibiblio.org::gutenberg/cache/epub/ /gutenberg/
"

# HuggingFace datasets (all zero-risk)
python -c "
from datasets import load_dataset
import json, pathlib
for name, cfg, split in [
    ('gsm8k','main','train'),
    ('EleutherAI/hendrycks_math','all','train'),
    ('allenai/ai2_arc','ARC-Challenge','train'),
    ('EleutherAI/logiqa',None,'train'),
    ('AI-MO/NuminaMath-CoT',None,'train'),
    ('databricks/databricks-dolly-15k',None,'train'),
    ('trivia_qa','rc','train'),
    ('ms_marco','v2.1','train'),
]:
    ds = load_dataset(name, cfg, split=split, trust_remote_code=False)
    out = pathlib.Path(f'D:/corpus_warehouse/raw/reasoning/{name.split(\"/\")[-1]}.jsonl')
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out,'w') as f:
        for row in ds: f.write(json.dumps(row) + chr(10))
    print(f'Saved {len(ds)} rows to {out}')
"

# Wikipedia dump (CC-BY-SA 3.0)
# Download: https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
# Parse: pip install wikiextractor
# python -m wikiextractor.WikiExtractor enwiki-latest-pages-articles.xml.bz2 \
#   --output D:/corpus_warehouse/raw/technical_docs/wikipedia/ \
#   --bytes 1M --compress --json

# ArXiv (filter CC-BY papers from metadata)
# ArXiv bulk access: https://info.arxiv.org/help/bulk_data_s3.html
# AWS: aws s3 cp s3://arxiv/metadata/ . --recursive --no-sign-request
# Filter: jq 'select(.license == "http://creativecommons.org/licenses/by/4.0/")' metadata.jsonl
```

---

## 19. Risks and Failure Modes

### R1 — Zero-risk ceiling lower than expected

**Risk:** After deduplication, approved tokens < 25B, insufficient for quality training.  
**Probability:** MEDIUM — depends on how much duplicate content exists across PG/Wikipedia/ArXiv.  
**Mitigation:** The 1TB SSD storage budget allows adding more Category A (PG long-tail) to compensate. PG has 60,000+ texts; we target 10,000. Expanding to 20,000 adds ~7B more tokens.  
**Recovery:** `python main.py corpus register-source A-gutenberg-ext ...` and re-pipeline.

### R2 — Quality gate too strict (< 40% approval rate)

**Risk:** Stringent quality gate removes too many paragraphs, reducing corpus below target.  
**Probability:** LOW — PD books and ArXiv papers are high quality and will mostly auto-approve.  
**Mitigation:** Lower `min_quality_score` from 0.45 to 0.40 for Category A only (books have naturally lower TTR than technical writing).  
**Recovery:** Update `CORPUS.min_quality_score_override = {"A": 0.40}` in config; re-run score stage.

### R3 — Wikipedia dump too large for pipeline

**Risk:** 22 GB Wikipedia dump takes days to process.  
**Probability:** MEDIUM — the dump is large; filtering is necessary.  
**Mitigation:** Use `mwparserfromhell` to filter only CS/math/science category articles during XML parsing — reduces processed volume by ~80%.  
**Recovery:** Process in 1GB chunks using `pipeline.py::prepare_corpus()` in batch mode.

### R4 — ArXiv licence filtering misses non-CC papers

**Risk:** Non-CC ArXiv papers enter the release undetected.  
**Probability:** LOW — with per-paper metadata filtering.  
**Mitigation:** Filter on `metadata.license` field: only `cc-by*` patterns pass. Every paper gets source_id with SHA-256 of its arxiv ID.  
**Recovery:** If discovered: `validate-source --reject`, rebuild release as v2.

### R5 — Shard tracker corruption after power loss

**Risk:** Power cut during training corrupts `shard_tracker.json`.  
**Probability:** LOW — atomic write (temp file + rename) prevents partial writes.  
**Mitigation:** Atomic write implemented in `shard_tracker.py::_atomic_write()`.  
**Recovery:** `python main.py shard-status` — if corrupted, delete `shard_tracker.json` and reinitialise (loses progress but not correctness).

### R6 — Release lock tampered after training started

**Risk:** Someone modifies `train.txt` after training begins, corrupting the provenance.  
**Probability:** LOW — LOCK file detects any change.  
**Mitigation:** Resume guard verifies all artifact hashes against LOCK file before every session.  
**Recovery:** If tampered: `ResumeGuardError` is raised, training aborted. Investigate, rebuild as v2.

### R7 — Human review queue grows faster than cleared

**Risk:** Operator cannot keep up with borderline items requiring review.  
**Probability:** MEDIUM for large sources (Wikipedia 22 GB produces many borderline paragraphs).  
**Mitigation:** Focus first on high-quality sources (PG, Standard Ebooks, ArXiv) that mostly auto-approve. Process Wikipedia last with aggressive pre-filtering.  
**Recovery:** Lower `auto_approve_threshold` from 0.70 to 0.65 for stable Category A/G sources.

### R8 — Synthetic data quality is too low

**Risk:** COC generates repetitive or template-locked synthetic pairs.  
**Probability:** MEDIUM — synthetic quality depends on live session quality.  
**Mitigation:** Synthetic is capped at 3.3% (500M / 30B). Even if quality is mediocre, it cannot dominate.  
**Recovery:** Reject low-quality synthetic items during review. Rebuild with fewer synthetic tokens.

---

## 20. Final Recommendation

### What is the best corpus strategy for CognitiveOC?

A **zero-risk-first, precision-curated 30B token release** from 11 source categories, all with explicit permissive licensing, with strict quality gates, full audit trail, and release locking.

**Do not attempt a 150B warehouse on a zero-risk policy.** The honest ceiling is ~35B clean tokens. Build to 30B for v1, expand the warehouse incrementally as new zero-risk sources are identified and pipelined.

### What belongs in release v1?

Books (PD/CC0) dominate at 40%. Technical docs (Wikipedia + official docs) at 13%. Research papers and STEM reasoning at 10% each. The remaining 37% distributed across educational, articles, conversations, cognition, retrieval, KG, and synthetic.

All sources: licence risk ≤ 0.20. No exceptions.

### What belongs in the warehouse but not v1?

- MIT OCW materials (CC-BY-NC-SA — NC clause)
- S2ORC CC-BY-NC papers (NC clause)
- Alpaca (CC-BY-NC 4.0)
- The Conversation (CC-BY-ND — pending legal review)

### What must never be included?

- Post-1927 copyrighted books
- News agency content (Reuters, AP, BBC — ToS)
- GPT/Claude/Gemini outputs as training data
- Social media content
- SEO content farms
- Z-Library / Anna's Archive / SciHub
- Any source with unknown provenance

### Exact next action

```bash
# 1. Set SSD path in config.py
# 2. Verify system
python main.py corpus warehouse-stats

# 3. Begin source acquisition in priority order:
#    Week 1: Project Gutenberg (biggest win, zero risk)
#    Week 2: Standard Ebooks + Wikipedia filtered + HuggingFace datasets
#    Week 3: ArXiv CC-BY + ACL Anthology + PubMed OA
#    Week 4-6: OpenStax + remaining datasets + Python/PyTorch docs
#    Week 7-8: Pipeline all sources + human review
#    Week 9: Build v1, sign, lock
#    Week 9: Train tokenizer
#    Week 10: Begin segmented pre-training
```

---

## 21. Go / No-Go Decision

### GO — Corpus collection and release generation

**Decision: GO**, with the following conditions:

✅ All 20 corpus system modules are implemented and import-clean  
✅ All CLI commands are wired and verified live  
✅ Zero-risk licence policy is codified in `governance/license_rules.json`  
✅ Quality gates are implemented in `corpus/scorer.py`  
✅ Release locking is implemented in `release/lock.py`  
✅ Training control (ledger, shard, guard, provenance) is complete  
✅ 1TB SSD budget is sufficient (558 GB used, 442 GB reserve)  
✅ 30B token target is achievable from zero-risk sources  
✅ Audit trail is wired end-to-end  

**Condition:** `CORPUS_WAREHOUSE_DIR` in `config.py` must be set to the actual SSD path before any acquisition work begins.

### NO-GO — Tokenizer training

**Decision: NO-GO until release v1 is built, signed, and locked.**

### NO-GO — Model training

**Decision: NO-GO until tokenizer is trained and evaluated (fertility ≥ 3.5 chars/token).**

### No-go conditions that would reverse the GO

- Discovery that a major source family (e.g. bulk PG texts) has unexpected licence issues → pause, re-audit, replace with alternative PD sources
- Quality gate produces < 20B approved tokens after full pipeline → lower threshold for Category A to 0.40; add more PG long-tail texts
- SSD throughput is too slow for pipeline processing → benchmark before full acquisition; NVMe read should be adequate

---

*Document version: 1.0 — Zero-Risk Corpus Edition*  
*CognitiveOC v3 — Corpus + Training Control · All systems verified*
