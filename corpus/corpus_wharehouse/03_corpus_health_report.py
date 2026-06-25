#!/usr/bin/env python3
"""
CognitiveOC - Corpus Health & Statistics
=========================================
Reads the warehouse and reports:
  - Per-category disk usage + source counts
  - License distribution
  - Risk distribution
  - Coverage gaps (empty categories)
  - Source status (ok / error / missing)

Usage:
    python 03_corpus_health_report.py
    python 03_corpus_health_report.py --root D:/custom/path
    python 03_corpus_health_report.py --json
"""
from __future__ import annotations
import argparse, json, os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_WAREHOUSE = Path(os.environ.get(
    "COC_WAREHOUSE_DIR",
    r"D:\projects\CognitiveOC\Final_Versions\cognitiveoc_v3\corpus\corpus_wharehouse",
))

def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def fmt_bytes(n):
    for u in ("B","KB","MB","GB","TB"):
        if n < 1024: return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}PB"

def dir_size(path):
    total = 0
    if not path.exists(): return 0
    for p in path.rglob("*"):
        try:
            if p.is_file(): total += p.stat().st_size
        except: continue
    return total

def collect_manifests(root):
    manifests = []
    d = root / "manifests"
    if not d.exists(): return manifests
    for f in d.glob("*.json"):
        try: manifests.append(json.loads(f.read_text(encoding="utf-8")))
        except Exception as e: manifests.append({"_error": repr(e), "_file": str(f)})
    return manifests

def category_breakdown(root):
    raw = root / "raw"; out = {}
    if not raw.exists(): return out
    for cd in sorted(raw.iterdir()):
        if not cd.is_dir(): continue
        sources = [p.name for p in cd.iterdir() if p.is_dir()]
        size = dir_size(cd)
        out[cd.name] = {"sources_present": len(sources), "sources": sources,
                        "size_bytes": size, "size_human": fmt_bytes(size)}
    return out

def license_dist(manifests):
    return dict(Counter(m.get("license_name","unknown") for m in manifests).most_common())

def risk_dist(manifests):
    b = {"zero (0.0)":0, "very_low (0.1)":0, "low (0.2)":0, "moderate (0.3+)":0}
    for m in manifests:
        r = m.get("license_risk", 0.0)
        if r <= 0.05: b["zero (0.0)"] += 1
        elif r <= 0.15: b["very_low (0.1)"] += 1
        elif r <= 0.25: b["low (0.2)"] += 1
        else: b["moderate (0.3+)"] += 1
    return b

def status_summary(manifests):
    return dict(Counter(m.get("status","unknown") for m in manifests))

def print_report(r):
    print(f"\n{'='*70}")
    print(f"  CognitiveOC Corpus Health Report")
    print(f"  Generated: {r['generated_at']}")
    print(f"  Root:      {r['warehouse_root']}")
    print(f"{'='*70}")
    print(f"\n[Storage]  Total: {r['total_size_human']}")
    print(f"\n[Sources]  Manifests: {r['manifest_count']}")
    for k,v in r["status_summary"].items(): print(f"  {k:<20} {v}")
    print(f"\n[Categories]")
    for cat, info in r["category_breakdown"].items():
        print(f"  {cat:<24} sources={info['sources_present']:<3}  size={info['size_human']}")
    print(f"\n[License Distribution]")
    for lic, n in r["license_distribution"].items(): print(f"  {lic:<40} {n}")
    print(f"\n[Risk Distribution]")
    for b, n in r["risk_distribution"].items(): print(f"  {b:<20} {n}")
    print(f"\n[Coverage Gaps]")
    if not r["empty_categories"]: print("  None - all categories populated.")
    else:
        for c in r["empty_categories"]: print(f"  empty: {c}")
    print(f"\n{'='*70}\n")

def build_report(root):
    mans = collect_manifests(root)
    cats = category_breakdown(root)
    empty = [n for n,i in cats.items() if i["sources_present"]==0]
    total = sum(i["size_bytes"] for i in cats.values())
    return {"generated_at": utc_now(), "warehouse_root": str(root),
            "manifest_count": len(mans), "status_summary": status_summary(mans),
            "category_breakdown": cats, "license_distribution": license_dist(mans),
            "risk_distribution": risk_dist(mans), "empty_categories": empty,
            "total_size_bytes": total, "total_size_human": fmt_bytes(total)}

def main():
    p = argparse.ArgumentParser(description="CognitiveOC warehouse health & stats")
    p.add_argument("--root", default=str(DEFAULT_WAREHOUSE))
    p.add_argument("--json", action="store_true")
    args = p.parse_args()
    root = Path(args.root)
    if not root.exists(): print(f"Warehouse not found: {root}"); return 1
    r = build_report(root)
    if args.json: print(json.dumps(r, indent=2, ensure_ascii=False))
    else: print_report(r)
    save = root / "reports" / f"health_{utc_now().replace(':','-')}.json"
    save.parent.mkdir(parents=True, exist_ok=True)
    save.write_text(json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
    latest = root / "reports" / "health_latest.json"
    latest.write_text(json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
