#!/usr/bin/env python3
"""
CognitiveOC - Unified Corpus Acquisition Manager
=================================================
Downloads curated zero-risk datasets into the correct warehouse categories.

Usage:
    python 02_corpus_acquisition_manager.py --list
    python 02_corpus_acquisition_manager.py --all
    python 02_corpus_acquisition_manager.py --all --priority 1
    python 02_corpus_acquisition_manager.py --only oasst2 dolly15k gsm8k
    python 02_corpus_acquisition_manager.py --all --skip-large
    python 02_corpus_acquisition_manager.py --all --include-warehouse-only
    python 02_corpus_acquisition_manager.py --only wikipedia_dump --force

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
# Config
# =====================================================================
DEFAULT_WAREHOUSE = Path(os.environ.get(
    "COC_WAREHOUSE_DIR",
    r"D:\projects\CognitiveOC\Final_Versions\cognitiveoc_v3\corpus\corpus_wharehouse",
))
USER_AGENT = "CognitiveOC-CorpusAcquisition/2.0"

CATEGORY_DIRS = {
    "A": "A_books", "B": "B_educational", "C": "C_reasoning",
    "D": "D_conversations", "E": "E_technical_docs", "F": "F_articles",
    "G": "G_research_papers", "H": "H_synthetic", "I": "I_cognition",
    "J": "J_retrieval", "K": "K_knowledge_graph", "L": "L_language_resources",
    "M": "M_legal_government", "N": "N_evaluation",
}

LARGE_SOURCES = {
    "wikipedia_dump", "wikidata_truthy", "cosmopedia_v2",
    "openwebmath", "proof_pile_2",
}

# =====================================================================
# Source Registry
# =====================================================================
@dataclass(frozen=True)
class SourceSpec:
    key: str
    kind: str           # "hf" | "http" | "hf_warehouse_only" | "http_warehouse_only"
    category: str       # A..N
    subdir: str
    license_name: str
    license_risk: float
    notes: str
    priority: int = 2
    hf_id: Optional[str] = None
    hf_config: Optional[str] = None
    url: Optional[str] = None
    extra_urls: tuple = ()
    warehouse_only: bool = False
    estimated_tokens: str = "unknown"


SAFE_SOURCES: list[SourceSpec] = [
    # === D. Conversations / Instruction ===
    SourceSpec("oasst2","hf","D","oasst2","Apache-2.0",0.0,"OpenAssistant trees; 129k train/6.6k val.",1,
        hf_id="OpenAssistant/oasst2",estimated_tokens="~150M"),
    SourceSpec("dolly15k","hf","D","dolly15k","CC-BY-SA-3.0",0.1,"15k Databricks human-written instructions.",1,
        hf_id="databricks/databricks-dolly-15k",estimated_tokens="~10M"),

    # === C. Reasoning / STEM ===
    SourceSpec("numinamath_cot","hf","C","numinamath_cot","Apache-2.0",0.0,"860k math CoT rows.",1,
        hf_id="AI-MO/NuminaMath-CoT",estimated_tokens="~200M"),
    SourceSpec("arc_challenge","hf","C","arc_challenge","CC-BY-SA-4.0",0.1,"ARC-Challenge science reasoning.",1,
        hf_id="allenai/ai2_arc",hf_config="ARC-Challenge",estimated_tokens="~2M"),
    SourceSpec("arc_easy","hf","C","arc_easy","CC-BY-SA-4.0",0.1,"ARC-Easy science reasoning.",1,
        hf_id="allenai/ai2_arc",hf_config="ARC-Easy",estimated_tokens="~3M"),
    SourceSpec("gsm8k","hf","C","gsm8k","MIT",0.0,"Grade-school math word problems.",1,
        hf_id="openai/gsm8k",hf_config="main",estimated_tokens="~3M"),
    SourceSpec("math_lighteval","hf","C","math_dataset","MIT",0.0,"Competition math.",1,
        hf_id="lighteval/MATH",estimated_tokens="~10M"),
    SourceSpec("strategyqa","hf","C","strategyqa","MIT",0.0,"Multi-hop commonsense reasoning.",2,
        hf_id="ChilleD/StrategyQA",estimated_tokens="~1M"),
    SourceSpec("winogrande","hf","C","winogrande","CC-BY",0.1,"Commonsense pronoun resolution.",2,
        hf_id="allenai/winogrande",hf_config="winogrande_xl",estimated_tokens="~5M"),
    SourceSpec("openwebmath","hf","C","openwebmath","ODC-By-1.0",0.2,"Best open math corpus. CC downstream flag.",1,
        hf_id="open-web-math/open-web-math",estimated_tokens="~14.7B"),
    SourceSpec("proof_pile_2","hf","C","proof_pile_2","Mixed (per-source)",0.2,"Math+code reasoning. Filter per-source.",2,
        hf_id="EleutherAI/proof-pile-2",estimated_tokens="~55B"),
    SourceSpec("metamathqa","hf","C","metamathqa","MIT",0.0,"MetaMath augmented Q&A.",1,
        hf_id="meta-math/MetaMathQA",estimated_tokens="~400M"),

    # === J. Retrieval / QA ===
    SourceSpec("hotpotqa","hf","J","hotpotqa","CC-BY-SA-4.0",0.1,"Multi-hop QA.",1,
        hf_id="hotpotqa/hotpot_qa",hf_config="distractor",estimated_tokens="~50M"),
    SourceSpec("trivia_qa","hf","J","trivia_qa","Apache-2.0",0.0,"TriviaQA reading comprehension.",1,
        hf_id="mandarjoshi/trivia_qa",hf_config="rc",estimated_tokens="~100M"),
    SourceSpec("natural_questions","hf","J","natural_questions","CC-BY-SA-3.0",0.1,"Google Natural Questions.",2,
        hf_id="google-research-datasets/natural_questions",hf_config="default",estimated_tokens="~300M"),
    SourceSpec("fever","hf","J","fever","CC-BY-SA-3.0",0.1,"Fact verification.",2,
        hf_id="fever/fever",hf_config="v1.0",estimated_tokens="~30M"),
    SourceSpec("miracl","hf","J","miracl","Apache-2.0",0.0,"Multilingual retrieval (EN subset).",2,
        hf_id="miracl/miracl",hf_config="en",estimated_tokens="~500M"),

    # === N. Evaluation ===
    SourceSpec("mmlu","hf","N","mmlu","MIT",0.0,"MMLU benchmark (eval only).",1,
        hf_id="cais/mmlu",hf_config="all",estimated_tokens="~5M"),
    SourceSpec("hellaswag","hf","N","hellaswag","MIT",0.0,"HellaSwag commonsense (eval only).",1,
        hf_id="Rowan/hellaswag",estimated_tokens="~10M"),
    SourceSpec("truthful_qa","hf","N","truthful_qa","Apache-2.0",0.0,"TruthfulQA (eval only).",1,
        hf_id="truthfulqa/truthful_qa",hf_config="generation",estimated_tokens="~1M"),
    SourceSpec("openbookqa","hf","N","openbookqa","Apache-2.0",0.0,"OpenBookQA science (eval).",2,
        hf_id="allenai/openbookqa",hf_config="main",estimated_tokens="~1M"),

    # === H. Synthetic ===
    SourceSpec("cosmopedia_v2","hf","H","cosmopedia_v2","Apache-2.0",0.0,"Cosmopedia v2 synthetic textbooks.",1,
        hf_id="HuggingFaceTB/smollm-corpus",hf_config="cosmopedia-v2",estimated_tokens="~28B"),

    # === A. Books ===
    SourceSpec("pg19","hf","A","pg19","Apache-2.0 + Public Domain",0.0,"Project Gutenberg pre-1919 books.",1,
        hf_id="emozilla/pg19",estimated_tokens="~2.5B"),
    SourceSpec("gutenberg_catalog","http","A","gutenberg","Public Domain (US)",0.0,"Gutenberg RDF catalog + seed texts.",1,
        url="https://www.gutenberg.org/cache/epub/feeds/rdf-files.tar.bz2",estimated_tokens="catalog+20 books"),

    # === B. Educational ===
    SourceSpec("openstax_subjects","http","B","openstax","CC-BY-4.0",0.0,"OpenStax subjects + book pages.",1,
        url="https://openstax.org/subjects",estimated_tokens="~50M"),
    SourceSpec("wikibooks_index","http","B","wikibooks","CC-BY-SA-4.0",0.1,"Wikibooks index pages.",1,
        url="https://en.wikibooks.org/wiki/Special:AllPages",
        extra_urls=("https://www.wikibooks.org/",),estimated_tokens="~300M"),

    # === F. Articles ===
    SourceSpec("wikipedia_dump","http","F","wikipedia","CC-BY-SA-4.0",0.1,"English Wikipedia dump (~22GB).",1,
        url="https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2",estimated_tokens="~4B"),

    # === G. Research Papers ===
    SourceSpec("arxiv_api","http","G","arxiv","Mixed (filter CC)",0.2,"arXiv API metadata pages.",1,
        url="https://export.arxiv.org/api/query?search_query=cat:cs.*&start=0&max_results=200",
        extra_urls=(
            "https://export.arxiv.org/api/query?search_query=cat:math.*&start=0&max_results=200",
            "https://export.arxiv.org/api/query?search_query=cat:stat.*&start=0&max_results=200",
            "https://export.arxiv.org/api/query?search_query=cat:q-bio.*&start=0&max_results=200",
            "https://export.arxiv.org/api/query?search_query=cat:physics.*&start=0&max_results=200",
        ),estimated_tokens="metadata seed"),
    SourceSpec("pmc_oa_index","http","G","pmc_oa","Mixed CC (filter BY/0)",0.1,"PMC OA file list.",2,
        url="https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_file_list.csv",
        extra_urls=("https://www.ncbi.nlm.nih.gov/pmc/tools/openftlist/",),estimated_tokens="index only"),

    # === K. Knowledge Graph ===
    SourceSpec("wikidata_truthy","http","K","wikidata","CC0",0.0,"Wikidata truthy RDF dump (~50GB).",1,
        url="https://dumps.wikimedia.org/wikidatawiki/entities/latest-truthy.nt.gz",estimated_tokens="~10B verbalized"),
    SourceSpec("conceptnet","http","K","conceptnet","CC-BY-SA-4.0",0.1,"ConceptNet 5.7 assertions.",2,
        url="https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz",estimated_tokens="~1B verbalized"),

    # === L. Language Resources ===
    SourceSpec("wordnet","http","L","wordnet","Princeton WordNet (BSD-like)",0.0,"WordNet 3.1 lexical database.",1,
        url="https://wordnetcode.princeton.edu/wn3.1.dict.tar.gz",estimated_tokens="~50M"),
    SourceSpec("framenet_info","http","L","framenet","CC-BY 3.0",0.0,"FrameNet info page.",2,
        url="https://framenet.icsi.berkeley.edu/fndrupal/framenet_data",estimated_tokens="~30M"),

    # === M. Legal / Government ===
    SourceSpec("courtlistener_info","http","M","courtlistener","Public Domain (US federal)",0.0,"CourtListener bulk data info.",2,
        url="https://www.courtlistener.com/help/api/bulk-data/",estimated_tokens="~20B (full)"),
    SourceSpec("sec_edgar_info","http","M","sec_edgar","Public Domain (US)",0.0,"SEC EDGAR seed page.",3,
        url="https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&type=10-K&dateb=&owner=include&count=40",estimated_tokens="~15B"),
    SourceSpec("ietf_rfc_index","http","M","ietf_rfcs","BSD-like",0.0,"IETF RFC index.",2,
        url="https://www.rfc-editor.org/rfc-index.txt",estimated_tokens="~200M"),

    # === Warehouse-only (excluded from release by default) ===
    SourceSpec("sciq","hf_warehouse_only","C","sciq","CC-BY-NC-3.0",0.4,"NC license - warehouse-only.",3,
        hf_id="allenai/sciq",warehouse_only=True,estimated_tokens="~5M"),
    SourceSpec("ms_marco","hf_warehouse_only","J","ms_marco","Microsoft Research License",0.6,"Manual review needed.",3,
        hf_id="microsoft/ms_marco",hf_config="v2.1",warehouse_only=True,estimated_tokens="~5B"),
]


# =====================================================================
# Utilities
# =====================================================================
def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def ensure_dir(p):
    p.mkdir(parents=True, exist_ok=True)
    return p

def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
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
    for u in ("B","KB","MB","GB","TB"):
        if n < 1024: return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}PB"

def http_stream(url, timeout=180):
    r = requests.get(url, stream=True, timeout=timeout,
        headers={"User-Agent": USER_AGENT, "Accept": "*/*"}, allow_redirects=True)
    r.raise_for_status()
    return r

def download_file(url, dest, force=False):
    ensure_dir(dest.parent)
    if dest.exists() and dest.stat().st_size > 0 and not force:
        return {"status":"skipped_exists","url":url,"path":str(dest),
                "bytes":dest.stat().st_size,"sha256":None}
    tmp = dest.with_suffix(dest.suffix + ".part")
    if tmp.exists(): tmp.unlink()
    print(f"    downloading: {url}")
    t0 = time.time(); total = 0
    try:
        r = http_stream(url)
        with tmp.open("wb") as f:
            for chunk in r.iter_content(1024*1024):
                if chunk: f.write(chunk); total += len(chunk)
        tmp.replace(dest)
    except Exception as e:
        if tmp.exists(): tmp.unlink()
        return {"status":"error","url":url,"path":str(dest),"bytes":0,"error":repr(e)}
    dur = time.time()-t0; sha = sha256_file(dest)
    print(f"    ok ({fmt_bytes(total)} in {dur:.1f}s) sha={sha[:16]}...")
    return {"status":"downloaded","url":url,"path":str(dest),"bytes":total,
            "sha256":sha,"duration_sec":round(dur,2)}

def source_dir(root, spec):
    return root / "raw" / CATEGORY_DIRS[spec.category] / spec.subdir

def manifest_path(root, spec):
    return root / "manifests" / f"{spec.subdir}.json"

def already_done(root, spec):
    d = source_dir(root, spec); m = manifest_path(root, spec)
    if not (d.exists() and m.exists()): return False
    if not any(d.iterdir()): return False
    try: return json.loads(m.read_text(encoding="utf-8")).get("status") == "ok"
    except: return False

def log_event(root, event):
    append_jsonl(root / "logs" / "acquisition_log.jsonl", {"ts": utc_now(), **event})

def common_manifest(spec, root, status, out_dir):
    return {
        "key": spec.key, "kind": spec.kind, "category": spec.category,
        "category_dir": CATEGORY_DIRS[spec.category], "subdir": spec.subdir,
        "license_name": spec.license_name, "license_risk": spec.license_risk,
        "warehouse_only": spec.warehouse_only, "priority": spec.priority,
        "estimated_tokens": spec.estimated_tokens, "notes": spec.notes,
        "status": status, "downloaded_at": utc_now(),
        "warehouse_root": str(root), "output_dir": str(out_dir),
    }


# =====================================================================
# Acquisition handlers
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
    print(f"\n[HF ] {spec.key}  ({spec.hf_id}" + (f" [{spec.hf_config}])" if spec.hf_config else ")"))
    print(f"      output = {out}")
    t0 = time.time()
    ds = load_dataset(spec.hf_id, spec.hf_config, trust_remote_code=False)
    ds.save_to_disk(str(out))
    dur = time.time() - t0
    splits = {s: int(len(d)) for s, d in ds.items()} if isinstance(ds, DatasetDict) else {"data": int(len(ds))}
    total = sum(splits.values())
    man = common_manifest(spec, root, "ok", out)
    man.update({"hf_id": spec.hf_id, "hf_config": spec.hf_config,
                "splits": splits, "total_rows": total, "duration_sec": round(dur, 2)})
    write_json(manifest_path(root, spec), man)
    log_event(root, {"event": "hf_ok", "key": spec.key, "rows": total, "dur": round(dur, 2)})
    print(f"      ok {total:,} rows in {dur:.1f}s")
    return man

DEFAULT_GUTENBERG_IDS = [
    1342,84,11,1661,2701,98,76,46,1952,2600,
    5200,4300,1400,219,6130,2814,2554,1497,2542,1080,
]

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
    name = spec.url.split("?")[0].rstrip("/").split("/")[-1] or f"{spec.key}.dat"
    artifacts.append(download_file(spec.url, out / name, force=force))
    for i, extra in enumerate(spec.extra_urls):
        en = extra.split("?")[0].rstrip("/").split("/")[-1] or f"extra_{i}.dat"
        if (out / en).exists():
            en = f"{Path(en).stem}_{i}{Path(en).suffix}"
        artifacts.append(download_file(extra, out / en, force=force))
    # Special handlers
    if spec.key == "openstax_subjects":
        artifacts.extend(_discover_openstax(out, force))
    elif spec.key == "gutenberg_catalog":
        artifacts.extend(_seed_gutenberg(out, force))
    elif spec.key == "wikibooks_index":
        (out / "README.txt").write_text("Wikibooks index. Extraction later.\n", encoding="utf-8")
    man = common_manifest(spec, root, "ok", out)
    man.update({"url": spec.url, "extra_urls": list(spec.extra_urls), "artifacts": artifacts})
    write_json(manifest_path(root, spec), man)
    ok = sum(1 for a in artifacts if a.get("status") in {"downloaded","skipped_exists"})
    err = sum(1 for a in artifacts if a.get("status") == "error")
    log_event(root, {"event": "http_ok", "key": spec.key, "ok": ok, "err": err})
    print(f"       ok artifacts ok={ok} err={err}")
    return man

def _seed_gutenberg(out_dir, force):
    results = []
    texts = ensure_dir(out_dir / "texts")
    ids_file = Path(__file__).resolve().parent / "gutenberg_ids.txt"
    if ids_file.exists():
        ids = [int(l.strip()) for l in ids_file.read_text(encoding="utf-8").splitlines()
               if l.strip() and not l.strip().startswith("#") and l.strip().isdigit()]
    else:
        ids = DEFAULT_GUTENBERG_IDS
    for gid in ids:
        t = texts / f"{gid}.txt"
        if t.exists() and t.stat().st_size > 0 and not force: continue
        url = f"https://www.gutenberg.org/files/{gid}/{gid}-0.txt"
        try: results.append(download_file(url, t, force=force))
        except Exception as e: results.append({"status":"error","url":url,"error":repr(e)})
    write_json(out_dir / "gutenberg_index.json",
        {"downloaded_at": utc_now(), "ids_count": len(ids),
         "ids_source": str(ids_file) if ids_file.exists() else "builtin"})
    return results

def _discover_openstax(out_dir, force):
    results = []
    candidates = list(out_dir.glob("subjects*"))
    html = ""
    for c in candidates:
        try: html += c.read_text(encoding="utf-8", errors="ignore")
        except: continue
    links = sorted(set(re.findall(r'https://openstax\.org/books/[^"\'>\s]+', html)))
    books = ensure_dir(out_dir / "books")
    for link in links:
        slug = link.rstrip("/").split("/")[-1]
        bd = ensure_dir(books / slug)
        page = bd / "index.html"
        if page.exists() and page.stat().st_size > 0 and not force: continue
        try: results.append(download_file(link, page, force=force))
        except Exception as e: results.append({"status":"error","url":link,"error":repr(e)})
    write_json(out_dir / "openstax_index.json",
        {"downloaded_at": utc_now(), "books_discovered": len(links), "book_links": links})
    return results


# =====================================================================
# Dispatcher + CLI
# =====================================================================
def acquire(spec, root, force=False):
    if spec.kind.startswith("hf"): return acquire_hf(spec, root, force=force)
    if spec.kind.startswith("http"): return acquire_http(spec, root, force=force)
    raise ValueError(f"Unknown kind: {spec.kind}")

def build_registry(include_warehouse_only=False):
    if include_warehouse_only: return list(SAFE_SOURCES)
    return [s for s in SAFE_SOURCES if not s.warehouse_only]

def ensure_warehouse_layout(root):
    for n in ["raw","manifests","logs","hf_cache","reports","registry","provenance","governance_logs"]:
        ensure_dir(root / n)
    for cat in CATEGORY_DIRS.values():
        ensure_dir(root / "raw" / cat)

def print_table(rows, headers):
    widths = [max(len(str(r[i])) for r in [headers]+rows) for i in range(len(headers))]
    print("  ".join(h.ljust(w) for h,w in zip(headers,widths)))
    print("  ".join("-"*w for w in widths))
    for r in rows:
        print("  ".join(str(c).ljust(w) for c,w in zip(r,widths)))

def list_sources(include_wo):
    sources = build_registry(include_wo)
    rows = [[s.key, "WAREHOUSE-ONLY" if s.warehouse_only else f"P{s.priority}",
             f"{s.category}/{s.subdir}", s.license_name[:24],
             f"{s.license_risk:.1f}", s.estimated_tokens] for s in sources]
    print_table(rows, ["KEY","PRIORITY","CATEGORY/SUBDIR","LICENSE","RISK","EST. TOKENS"])
    print(f"\nTotal: {len(sources)} sources")

def write_summary(root, results):
    s = {"summary_generated_at": utc_now(), "warehouse_root": str(root),
         "total_attempted": len(results),
         "ok": sum(1 for r in results if r.get("status")=="ok"),
         "skipped": sum(1 for r in results if r.get("status")=="skipped_exists"),
         "errors": sum(1 for r in results if r.get("status")=="error"),
         "results": results}
    write_json(root / "reports" / f"acquisition_summary_{utc_now().replace(':','-')}.json", s)
    write_json(root / "download_summary.json", s)

def main():
    p = argparse.ArgumentParser(description="CognitiveOC unified corpus acquisition manager.")
    p.add_argument("--root", default=str(DEFAULT_WAREHOUSE))
    p.add_argument("--all", action="store_true")
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument("--list", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--include-warehouse-only", action="store_true")
    p.add_argument("--priority", type=int, choices=[1,2,3], default=None)
    p.add_argument("--skip-large", action="store_true")
    args = p.parse_args()

    root = Path(args.root)
    ensure_warehouse_layout(root)

    if args.list:
        list_sources(args.include_warehouse_only)
        return 0
    if not args.all and not args.only:
        p.error("Use --all, --only <keys...>, or --list")

    sources = build_registry(args.include_warehouse_only)
    if args.priority is not None:
        sources = [s for s in sources if s.priority == args.priority]
    if args.skip_large:
        sources = [s for s in sources if s.key not in LARGE_SOURCES]
    if args.only:
        wanted = set(args.only)
        sources = [s for s in sources if s.key in wanted]
        missing = wanted - {s.key for s in sources}
        if missing: print(f"WARNING: unknown keys: {sorted(missing)}", file=sys.stderr)
    if not sources:
        print("No sources to acquire."); return 1

    print(f"\nWarehouse: {root}")
    print(f"Queued: {len(sources)} sources")
    print("="*70)

    results = []
    for spec in sources:
        try: r = acquire(spec, root, force=args.force)
        except Exception as e:
            r = {"key": spec.key, "status": "error", "error": repr(e)}
            log_event(root, {"event": "acquisition_error", "key": spec.key, "error": repr(e)})
            print(f"  FAILED: {spec.key}: {e}", file=sys.stderr)
        results.append(r)

    write_summary(root, results)
    print(f"\n{'='*70}")
    print(f"  COMPLETE")
    print(f"  ok={sum(1 for r in results if r.get('status')=='ok')}  "
          f"skipped={sum(1 for r in results if r.get('status')=='skipped_exists')}  "
          f"errors={sum(1 for r in results if r.get('status')=='error')}")
    print(f"{'='*70}\n")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
