"""
CognitiveOC v3 — Evaluation Framework
=======================================

Comprehensive evaluation suite covering every subsystem:

  1.  Perplexity          — token-level cross-entropy on held-out corpus
  2.  MRR@K               — Mean Reciprocal Rank for retrieval
  3.  MR                  — Mean Rank for retrieval
  4.  NDCG@K              — Normalised Discounted Cumulative Gain
  5.  Hit@K               — Hit Rate at K for retrieval
  6.  Memory Recall Accuracy — ranked recall precision
  7.  KG Entity F1        — precision/recall/F1 on extracted entities
  8.  KG Triple F1        — precision/recall/F1 on extracted triples
  9.  Reasoning Accuracy  — chain-of-thought accuracy on test cases
  10. Emotion F1          — macro-F1 on emotion classification
  11. Tokenizer Fertility — chars/token (target ≥ 3.5)
  12. Tokenizer Round-trip— encode→decode identity accuracy
  13. Overall Readiness   — weighted composite score across all metrics

Phase gates (eval/gates/phase_N.json):
  Each phase gate defines required metric thresholds.
  A gate PASSES only when ALL required metrics meet their thresholds.

Ratings (strict, per-component):
  Excellent: ≥ 0.90 of target
  Good:      ≥ 0.75 of target
  Fair:      ≥ 0.60 of target
  Poor:      <  0.60 of target

File: eval/run_suite.py
Run:  python main.py eval --all data/corpus/v1/split/val.txt
      python tools/readiness_check.py --phase 1
"""

from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path

try:
    from config import EVALUATION, CHECKPOINT_DIR, EVAL_BASELINE, EVAL_GATES, ensure_dirs
except ImportError:
    EVALUATION    = {}
    CHECKPOINT_DIR= Path("var/checkpoints")
    EVAL_BASELINE = Path("eval/baseline")
    EVAL_GATES    = Path("eval/gates")
    def ensure_dirs(): pass


# ═══════════════════════════════════════════════════════════════════
# Rating helpers
# ═══════════════════════════════════════════════════════════════════

_THRESHOLDS = {
    "perplexity":     dict(target=15.0,  excellent=13.5, good=18.75, fair=25.0,
                           lower_is_better=True),
    "retrieval_mrr":  dict(target=0.85,  excellent=0.765, good=0.6375, fair=0.51),
    "retrieval_mr":   dict(target=3.0,   excellent=3.3,   good=4.0,   fair=5.0,
                           lower_is_better=True),
    "retrieval_ndcg": dict(target=0.80,  excellent=0.72,  good=0.60,  fair=0.48),
    "retrieval_hit":  dict(target=0.90,  excellent=0.81,  good=0.675, fair=0.54),
    "memory_recall":  dict(target=0.90,  excellent=0.81,  good=0.675, fair=0.54),
    "kg_entity_f1":   dict(target=0.80,  excellent=0.72,  good=0.60,  fair=0.48),
    "kg_triple_f1":   dict(target=0.70,  excellent=0.63,  good=0.525, fair=0.42),
    "reasoning_acc":  dict(target=0.75,  excellent=0.675, good=0.5625,fair=0.45),
    "emotion_f1":     dict(target=0.70,  excellent=0.63,  good=0.525, fair=0.42),
    "tok_fertility":  dict(target=3.5,   excellent=3.15,  good=2.625, fair=2.10),
    "tok_roundtrip":  dict(target=1.0,   excellent=1.0,   good=0.999, fair=0.995),
}


def rate(metric_name: str, value: float) -> str:
    """Return Excellent / Good / Fair / Poor rating for a metric value."""
    cfg = _THRESHOLDS.get(metric_name, {})
    if not cfg:
        return "Unknown"
    lib = cfg.get("lower_is_better", False)
    ex  = cfg.get("excellent", cfg.get("target", 1.0) * 0.9)
    gd  = cfg.get("good",      cfg.get("target", 1.0) * 0.75)
    fr  = cfg.get("fair",      cfg.get("target", 1.0) * 0.60)
    if lib:
        if value <= ex: return "Excellent"
        if value <= gd: return "Good"
        if value <= fr: return "Fair"
        return "Poor"
    else:
        if value >= ex: return "Excellent"
        if value >= gd: return "Good"
        if value >= fr: return "Fair"
        return "Poor"


def overall_score(results: dict) -> float:
    """Compute weighted overall readiness score [0, 1].

    Weights from config.py EVALUATION.weights.
    """
    weights = EVALUATION.get("weights", {
        "perplexity":    0.25,
        "retrieval_mrr": 0.20,
        "memory_recall": 0.15,
        "kg_entity_f1":  0.15,
        "reasoning_acc": 0.15,
        "emotion_f1":    0.10,
    })
    total_w, score = 0.0, 0.0
    for metric, w in weights.items():
        val    = results.get(metric)
        if val is None:
            continue
        cfg    = _THRESHOLDS.get(metric, {})
        target = cfg.get("target", 1.0)
        lib    = cfg.get("lower_is_better", False)
        if lib:
            # Normalise: score 1.0 when value ≤ target
            norm = max(0.0, min(1.0, target / max(val, 1e-9)))
        else:
            norm = max(0.0, min(1.0, val / max(target, 1e-9)))
        score   += w * norm
        total_w += w
    return round(score / max(total_w, 1e-9), 4)


# ═══════════════════════════════════════════════════════════════════
# 1. Perplexity
# ═══════════════════════════════════════════════════════════════════

def eval_perplexity(val_corpus_path: str,
                    checkpoint_path: str = None,
                    n_batches: int = 50) -> dict:
    """Compute validation perplexity on a held-out corpus.

    Requires trained checkpoint. Falls back to reporting None if not available.

    Returns:
        {"val_loss": float, "perplexity": float, "rating": str, "n_batches": int}
    """
    try:
        import torch
        from models.transformer import CognitiveDecoder
        from tokenizer.tokenizer import CognitiveTokenizer
        from config import MODEL

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt   = checkpoint_path or str(CHECKPOINT_DIR / "model_700m_best.pt")
        if not Path(ckpt).exists():
            return {"val_loss": None, "perplexity": None,
                    "rating": "N/A", "error": "checkpoint not found"}

        model = CognitiveDecoder.from_checkpoint(ckpt, device=str(device))
        model.eval()
        tok   = CognitiveTokenizer.load_default()

        text   = Path(val_corpus_path).read_text(encoding="utf-8", errors="replace")
        paras  = [p.strip() for p in text.split("\n\n") if p.strip()]
        tokens = []
        for p in paras:
            tokens.extend(tok.encode(p, add_bos=True, add_eos=True))

        block  = MODEL.get("block_size", 8192)
        import numpy as np
        import random
        rng    = random.Random(42)
        arr    = np.array(tokens, dtype=np.int32)
        n      = len(arr)
        losses = []

        with torch.no_grad():
            for _ in range(n_batches):
                s = rng.randint(0, max(n - block - 1, 0))
                x = torch.from_numpy(arr[s:s+block].astype(np.int64)).unsqueeze(0).to(device)
                y = torch.from_numpy(arr[s+1:s+block+1].astype(np.int64)).unsqueeze(0).to(device)
                _, loss, _ = model(x, targets=y)
                if loss and torch.isfinite(loss):
                    losses.append(loss.item())

        val_loss = sum(losses) / max(len(losses), 1)
        ppl      = math.exp(min(val_loss, 20.0))
        return {
            "val_loss":   round(val_loss, 6),
            "perplexity": round(ppl, 4),
            "rating":     rate("perplexity", ppl),
            "n_batches":  len(losses),
            "target":     EVALUATION.get("perplexity_target", 15.0),
        }
    except Exception as e:
        return {"val_loss": None, "perplexity": None,
                "rating": "N/A", "error": str(e)}


# ═══════════════════════════════════════════════════════════════════
# 2. Retrieval Evaluation (MRR, MR, NDCG, Hit@K)
# ═══════════════════════════════════════════════════════════════════

def eval_retrieval(bench_path: str = None, k: int = 10) -> dict:
    """Evaluate retrieval using a benchmark JSON file.

    Benchmark format:
        [{"query": str, "relevant": [str, ...], "documents": [str, ...]}]

    Metrics computed:
        MRR@K  — Mean Reciprocal Rank
        MR     — Mean Rank (of first relevant result)
        NDCG@K — Normalised DCG
        Hit@K  — fraction of queries with relevant doc in top-K

    Returns:
        {"mrr": float, "mr": float, "ndcg": float, "hit_k": float,
         "n_queries": int, "k": int, ratings...}
    """
    bench_path = bench_path or str(
        Path("eval") / "retrieval_bench.json"
    )
    if not Path(bench_path).exists():
        return {"mrr": None, "mr": None, "ndcg": None, "hit_k": None,
                "error": "benchmark not found", "rating_mrr": "N/A"}

    try:
        bench = json.loads(Path(bench_path).read_text())
    except Exception as e:
        return {"error": str(e)}

    from retrieval.rag import RAGPipeline
    rag = RAGPipeline()

    reciprocal_ranks, ranks, ndcg_scores, hits = [], [], [], []

    for item in bench:
        query    = item.get("query", "")
        relevant = set(item.get("relevant", []))
        results  = rag.retrieve(query, k=k)
        texts    = [r.get("text","") for r in results]

        # Find rank of first relevant result
        rank = None
        for i, t in enumerate(texts, 1):
            # Partial match: check if any relevant text is contained
            if any(rel[:60].lower() in t.lower() for rel in relevant):
                rank = i
                break

        if rank:
            reciprocal_ranks.append(1.0 / rank)
            ranks.append(rank)
            hits.append(1)
        else:
            reciprocal_ranks.append(0.0)
            ranks.append(k + 1)
            hits.append(0)

        # NDCG
        dcg, idcg = 0.0, 0.0
        for i, t in enumerate(texts, 1):
            rel = int(any(r[:60].lower() in t.lower() for r in relevant))
            dcg += rel / math.log2(i + 1)
        for i in range(1, min(len(relevant), k) + 1):
            idcg += 1.0 / math.log2(i + 1)
        ndcg_scores.append(dcg / max(idcg, 1e-9))

    n = max(len(bench), 1)
    mrr  = round(sum(reciprocal_ranks) / n, 4)
    mr   = round(sum(ranks) / n, 2)
    ndcg = round(sum(ndcg_scores) / n, 4)
    hit  = round(sum(hits) / n, 4)

    return {
        "mrr":       mrr,
        "mr":        mr,
        "ndcg":      ndcg,
        "hit_k":     hit,
        "k":         k,
        "n_queries": n,
        "rating_mrr":  rate("retrieval_mrr",  mrr),
        "rating_mr":   rate("retrieval_mr",   mr),
        "rating_ndcg": rate("retrieval_ndcg", ndcg),
        "rating_hit":  rate("retrieval_hit",  hit),
        "target_mrr":  EVALUATION.get("retrieval_mrr_target", 0.85),
    }


# ═══════════════════════════════════════════════════════════════════
# 3. Memory Recall Accuracy
# ═══════════════════════════════════════════════════════════════════

def eval_memory(n_samples: int = 50) -> dict:
    """Evaluate memory recall accuracy.

    Inserts N facts, then queries for each, checks if fact is recalled in top-K.

    Returns:
        {"recall_accuracy": float, "n_samples": int, "rating": str}
    """
    try:
        from memory.memory import CognitiveMemory
        mem = CognitiveMemory()

        test_facts = [
            (f"The capital of country_{i} is city_{i}.", f"capital country_{i}")
            for i in range(n_samples)
        ]

        # Store
        ids = []
        for fact, _ in test_facts:
            ids.append(mem.remember(fact, importance=1.0))

        # Query
        hits = 0
        for (fact, query), fid in zip(test_facts, ids):
            results = mem.ranked_recall(query, k=5)
            recalled_ids = [r["id"] for r in results]
            if fid in recalled_ids:
                hits += 1

        # Cleanup
        for fid in ids:
            try:
                mem.forget(fid)
            except Exception:
                pass

        acc = round(hits / max(n_samples, 1), 4)
        return {
            "recall_accuracy": acc,
            "n_samples":       n_samples,
            "hits":            hits,
            "rating":          rate("memory_recall", acc),
            "target":          EVALUATION.get("memory_recall_target", 0.90),
        }
    except Exception as e:
        return {"recall_accuracy": None, "error": str(e), "rating": "N/A"}


# ═══════════════════════════════════════════════════════════════════
# 4. KG Evaluation (Entity F1, Triple F1)
# ═══════════════════════════════════════════════════════════════════

def eval_kg(bench_path: str = None) -> dict:
    """Evaluate KG extraction: entity precision/recall/F1 and triple F1.

    Benchmark format:
        [{"text": str, "entities": [str,...], "triples": [[s,r,o],...]}]

    Returns:
        {"entity_f1": float, "triple_f1": float, "n_samples": int, ratings...}
    """
    bench_path = bench_path or str(Path("eval") / "kg_bench.json")
    if not Path(bench_path).exists():
        return {"entity_f1": None, "triple_f1": None,
                "error": "benchmark not found", "rating_entity": "N/A"}

    try:
        bench = json.loads(Path(bench_path).read_text())
    except Exception as e:
        return {"error": str(e)}

    from memory.summarizer import extract_triples

    entity_tp = entity_fp = entity_fn = 0
    triple_tp = triple_fp = triple_fn = 0

    for item in bench:
        text     = item.get("text", "")
        gold_ent = {e.lower() for e in item.get("entities", [])}
        gold_tri = {(t[0].lower(), t[1].lower(), t[2].lower())
                    for t in item.get("triples", [])}

        # Extract
        raw_triples = extract_triples(text)
        pred_ent    = {s.lower() for s,_,_ in raw_triples} | \
                      {o.lower() for _,_,o in raw_triples}
        pred_tri    = {(s.lower(), r.lower(), o.lower())
                       for s,r,o in raw_triples}

        entity_tp += len(gold_ent & pred_ent)
        entity_fp += len(pred_ent - gold_ent)
        entity_fn += len(gold_ent - pred_ent)

        triple_tp += len(gold_tri & pred_tri)
        triple_fp += len(pred_tri - gold_tri)
        triple_fn += len(gold_tri - pred_tri)

    def _f1(tp, fp, fn) -> float:
        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)
        return round(2 * prec * rec / max(prec + rec, 1e-9), 4)

    ent_f1 = _f1(entity_tp, entity_fp, entity_fn)
    tri_f1 = _f1(triple_tp, triple_fp, triple_fn)

    return {
        "entity_f1":    ent_f1,
        "triple_f1":    tri_f1,
        "n_samples":    len(bench),
        "rating_entity":rate("kg_entity_f1", ent_f1),
        "rating_triple":rate("kg_triple_f1", tri_f1),
        "target_entity":EVALUATION.get("kg_f1_target", 0.80),
    }


# ═══════════════════════════════════════════════════════════════════
# 5. Reasoning Accuracy
# ═══════════════════════════════════════════════════════════════════

def eval_reasoning(bench_path: str = None) -> dict:
    """Evaluate reasoning accuracy on test cases.

    Benchmark format:
        [{"query": str, "expected_type": str, "expected_steps_min": int}]

    Returns:
        {"accuracy": float, "type_accuracy": dict, "n_cases": int, "rating": str}
    """
    bench_path = bench_path or str(Path("eval") / "reasoning_bench.json")
    if not Path(bench_path).exists():
        return {"accuracy": None, "error": "benchmark not found", "rating": "N/A"}

    try:
        bench = json.loads(Path(bench_path).read_text())
    except Exception as e:
        return {"error": str(e)}

    from reasoning.reasoner import Reasoner
    reasoner   = Reasoner()
    correct    = 0
    type_hits: dict[str, list[int]] = {}

    for case in bench:
        query    = case.get("query", "")
        exp_type = case.get("expected_type", "")
        min_steps= case.get("expected_steps_min", 1)

        result = reasoner.assess(query)
        got_type   = result.get("type", "")
        got_steps  = len(result.get("steps", []))
        got_conf   = result.get("confidence", 0.0)

        ok = (got_type == exp_type) and (got_steps >= min_steps)
        if ok:
            correct += 1

        if exp_type not in type_hits:
            type_hits[exp_type] = []
        type_hits[exp_type].append(int(ok))

    n   = max(len(bench), 1)
    acc = round(correct / n, 4)
    type_acc = {t: round(sum(v)/max(len(v),1), 3) for t, v in type_hits.items()}

    return {
        "accuracy":     acc,
        "type_accuracy":type_acc,
        "n_cases":      n,
        "correct":      correct,
        "rating":       rate("reasoning_acc", acc),
        "target":       EVALUATION.get("reasoning_acc_target", 0.75),
    }


# ═══════════════════════════════════════════════════════════════════
# 6. Tokenizer Evaluation
# ═══════════════════════════════════════════════════════════════════

def eval_tokenizer(domain_text: str = None) -> dict:
    """Evaluate the 48K tokenizer: fertility, round-trip, special tokens.

    Returns full validation result dict from CognitiveTokenizer.validate().
    """
    try:
        from tokenizer.tokenizer import CognitiveTokenizer
        tok     = CognitiveTokenizer.load_default()
        results = tok.validate(domain_text)

        fert    = results.get("checks", [{}])
        fert_val= next(
            (c.get("detail","0").split()[0] for c in results.get("checks",[])
             if c.get("name") == "fertility"), "0"
        )
        try:
            fert_float = float(fert_val)
        except Exception:
            fert_float = 0.0

        return {
            "passed":          results.get("passed", False),
            "vocab_size":      results.get("vocab_size", 0),
            "fertility":       fert_float,
            "checks":          results.get("checks", []),
            "rating_fertility":rate("tok_fertility", fert_float),
            "target_fertility":3.5,
        }
    except Exception as e:
        return {"passed": False, "error": str(e), "rating_fertility": "N/A"}


# ═══════════════════════════════════════════════════════════════════
# 7. Emotion Classification F1
# ═══════════════════════════════════════════════════════════════════

def eval_emotion(bench_path: str = None) -> dict:
    """Evaluate emotion classifier macro-F1.

    Benchmark format:
        [{"text": str, "label": str}]

    Returns:
        {"macro_f1": float, "per_class_f1": dict, "n_samples": int, "rating": str}
    """
    bench_path = bench_path or str(Path("eval") / "emotion_bench.json")
    if not Path(bench_path).exists():
        return {"macro_f1": None, "error": "benchmark not found", "rating": "N/A"}

    try:
        bench = json.loads(Path(bench_path).read_text())
    except Exception as e:
        return {"error": str(e)}

    try:
        from encoder.hub import get_hub
        hub = get_hub()
    except Exception as e:
        return {"macro_f1": None, "error": str(e), "rating": "N/A"}

    tp: dict[str,int] = {}
    fp: dict[str,int] = {}
    fn: dict[str,int] = {}

    for item in bench:
        text  = item.get("text", "")
        label = item.get("label", "neutral")
        result = hub.classify_emotion(text)
        pred   = result.get("primary", "neutral")

        tp[label] = tp.get(label, 0) + int(pred == label)
        if pred != label:
            fp[pred]  = fp.get(pred, 0) + 1
            fn[label] = fn.get(label, 0) + 1

    labels = set(list(tp) + list(fp) + list(fn))
    f1s    = []
    per_class = {}
    for lbl in labels:
        p   = tp.get(lbl,0) / max(tp.get(lbl,0)+fp.get(lbl,0), 1)
        r   = tp.get(lbl,0) / max(tp.get(lbl,0)+fn.get(lbl,0), 1)
        f1  = 2*p*r/max(p+r,1e-9)
        f1s.append(f1)
        per_class[lbl] = round(f1, 3)

    macro_f1 = round(sum(f1s)/max(len(f1s),1), 4)
    return {
        "macro_f1":    macro_f1,
        "per_class_f1":per_class,
        "n_samples":   len(bench),
        "rating":      rate("emotion_f1", macro_f1),
        "target":      EVALUATION.get("emotion_f1_target", 0.70),
    }


# ═══════════════════════════════════════════════════════════════════
# Master Evaluation Runner
# ═══════════════════════════════════════════════════════════════════

def run_all(val_corpus_path: str,
            save: bool = True,
            verbose: bool = True) -> dict:
    """Run the complete evaluation suite and produce a report.

    Args:
        val_corpus_path: Path to val.txt corpus for perplexity.
        save:            Save results to eval/baseline/.
        verbose:         Print report to stdout.

    Returns:
        Full results dict with all metric values, ratings, and overall score.
    """
    ensure_dirs()
    t0 = time.time()

    if verbose:
        print("[eval] Running COC v3 evaluation suite...")

    results: dict = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S")}

    # ── Perplexity ─────────────────────────────────────────────────
    if verbose: print("  [1/7] Perplexity...")
    ppl_r = eval_perplexity(val_corpus_path)
    results["perplexity"] = ppl_r.get("perplexity")
    results["val_loss"]   = ppl_r.get("val_loss")
    results["ppl_detail"] = ppl_r

    # ── Tokenizer ──────────────────────────────────────────────────
    if verbose: print("  [2/7] Tokenizer...")
    tok_r = eval_tokenizer()
    results["tok_fertility"]  = tok_r.get("fertility")
    results["tok_roundtrip"]  = 1.0 if tok_r.get("passed") else 0.0
    results["tok_detail"]     = tok_r

    # ── Retrieval ──────────────────────────────────────────────────
    if verbose: print("  [3/7] Retrieval (MRR/MR/NDCG/Hit@10)...")
    ret_r = eval_retrieval()
    results["retrieval_mrr"]  = ret_r.get("mrr")
    results["retrieval_mr"]   = ret_r.get("mr")
    results["retrieval_ndcg"] = ret_r.get("ndcg")
    results["retrieval_hit"]  = ret_r.get("hit_k")
    results["ret_detail"]     = ret_r

    # ── Memory ─────────────────────────────────────────────────────
    if verbose: print("  [4/7] Memory recall...")
    mem_r = eval_memory()
    results["memory_recall"] = mem_r.get("recall_accuracy")
    results["mem_detail"]    = mem_r

    # ── KG ─────────────────────────────────────────────────────────
    if verbose: print("  [5/7] Knowledge graph...")
    kg_r = eval_kg()
    results["kg_entity_f1"]  = kg_r.get("entity_f1")
    results["kg_triple_f1"]  = kg_r.get("triple_f1")
    results["kg_detail"]     = kg_r

    # ── Reasoning ──────────────────────────────────────────────────
    if verbose: print("  [6/7] Reasoning accuracy...")
    rsn_r = eval_reasoning()
    results["reasoning_acc"] = rsn_r.get("accuracy")
    results["rsn_detail"]    = rsn_r

    # ── Emotion ────────────────────────────────────────────────────
    if verbose: print("  [7/7] Emotion F1...")
    em_r = eval_emotion()
    results["emotion_f1"]  = em_r.get("macro_f1")
    results["em_detail"]   = em_r

    # ── Overall score ──────────────────────────────────────────────
    results["overall_score"]   = overall_score(results)
    results["overall_rating"]  = _overall_rating(results["overall_score"])
    results["elapsed_s"]       = round(time.time() - t0, 2)

    # ── Per-metric ratings ─────────────────────────────────────────
    results["ratings"] = {
        "perplexity":    rate("perplexity",    results.get("perplexity") or 999),
        "tok_fertility": rate("tok_fertility", results.get("tok_fertility") or 0),
        "retrieval_mrr": rate("retrieval_mrr", results.get("retrieval_mrr") or 0),
        "retrieval_mr":  rate("retrieval_mr",  results.get("retrieval_mr")  or 999),
        "memory_recall": rate("memory_recall", results.get("memory_recall") or 0),
        "kg_entity_f1":  rate("kg_entity_f1",  results.get("kg_entity_f1")  or 0),
        "kg_triple_f1":  rate("kg_triple_f1",  results.get("kg_triple_f1")  or 0),
        "reasoning_acc": rate("reasoning_acc", results.get("reasoning_acc") or 0),
        "emotion_f1":    rate("emotion_f1",    results.get("emotion_f1")    or 0),
    }

    if verbose:
        _print_report(results)

    # ── Save to baseline ───────────────────────────────────────────
    if save:
        out = Path(str(EVAL_BASELINE)) / "eval_results.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2, default=str))
        if verbose:
            print(f"\n[eval] Results saved → {out}")

    return results


def _overall_rating(score: float) -> str:
    if score >= 0.90: return "Excellent"
    if score >= 0.75: return "Good"
    if score >= 0.60: return "Fair"
    return "Poor"


def _print_report(results: dict):
    W = 62
    print(f"\n{'═'*W}")
    print(f"  COC v3 Evaluation Report  —  {results.get('ts','')}")
    print(f"{'═'*W}")
    rows = [
        ("Perplexity",       results.get("perplexity"),    "perplexity",    ".2f"),
        ("Val Loss",          results.get("val_loss"),      None,            ".4f"),
        ("Tok Fertility",     results.get("tok_fertility"), "tok_fertility", ".2f"),
        ("Retrieval MRR@10", results.get("retrieval_mrr"), "retrieval_mrr", ".4f"),
        ("Retrieval MR",     results.get("retrieval_mr"),  "retrieval_mr",  ".1f"),
        ("Retrieval NDCG",   results.get("retrieval_ndcg"),"retrieval_ndcg",".4f"),
        ("Retrieval Hit@10", results.get("retrieval_hit"), "retrieval_hit", ".4f"),
        ("Memory Recall",    results.get("memory_recall"), "memory_recall", ".4f"),
        ("KG Entity F1",     results.get("kg_entity_f1"),  "kg_entity_f1",  ".4f"),
        ("KG Triple F1",     results.get("kg_triple_f1"),  "kg_triple_f1",  ".4f"),
        ("Reasoning Acc",    results.get("reasoning_acc"), "reasoning_acc", ".4f"),
        ("Emotion F1",       results.get("emotion_f1"),    "emotion_f1",    ".4f"),
    ]
    for label, val, metric, fmt in rows:
        if val is None:
            vstr = "N/A"
            rating = "N/A"
        else:
            vstr   = format(val, fmt)
            rating = rate(metric, val) if metric else ""
        print(f"  {label:<22s}  {vstr:<10s}  {rating}")
    print(f"{'─'*W}")
    score  = results.get("overall_score", 0.0)
    rating = results.get("overall_rating", "N/A")
    print(f"  {'OVERALL READINESS SCORE':<22s}  {score:<10.4f}  {rating}")
    print(f"{'═'*W}\n")


# ═══════════════════════════════════════════════════════════════════
# Phase Gate Checker
# ═══════════════════════════════════════════════════════════════════

def check_gate(phase: int, results: dict = None) -> dict:
    """Check a phase gate definition against current evaluation results.

    Gate file: eval/gates/phase_N.json
    Format:
        {"phase": N, "checks": [{"metric": str, "op": str, "threshold": float}, ...]}

    ops: ">=" | "<=" | "==" | "!=" | ">" | "<"

    Returns:
        {"passed": bool, "phase": int, "checks": list, "score": float}
    """
    gate_path = Path(str(EVAL_GATES)) / f"phase_{phase}.json"
    if not gate_path.exists():
        return {"passed": False, "error": f"Gate file not found: {gate_path}"}

    gate = json.loads(gate_path.read_text())
    if results is None:
        baseline = Path(str(EVAL_BASELINE)) / "eval_results.json"
        if baseline.exists():
            results = json.loads(baseline.read_text())
        else:
            results = {}

    check_results = []
    for chk in gate.get("checks", []):
        metric    = chk["metric"]
        op        = chk["op"]
        threshold = chk["threshold"]
        val       = results.get(metric)

        if val is None:
            passed = False
            detail = "metric not available"
        else:
            ops = {
                ">=": val >= threshold,
                "<=": val <= threshold,
                ">":  val >  threshold,
                "<":  val <  threshold,
                "==": abs(val - threshold) < 1e-6,
                "!=": abs(val - threshold) >= 1e-6,
            }
            passed = bool(ops.get(op, False))
            detail = f"{val:.4f} {op} {threshold:.4f}"

        check_results.append({
            "metric":    metric,
            "op":        op,
            "threshold": threshold,
            "value":     val,
            "passed":    passed,
            "detail":    detail,
        })

    all_passed = all(c["passed"] for c in check_results)
    n_pass     = sum(1 for c in check_results if c["passed"])
    n_total    = len(check_results)

    return {
        "passed":   all_passed,
        "phase":    phase,
        "score":    round(n_pass / max(n_total, 1), 3),
        "n_pass":   n_pass,
        "n_total":  n_total,
        "checks":   check_results,
    }


def write_phase1_gate():
    """Write the Phase 1 gate definition (tokenizer upgrade)."""
    EVAL_GATES.mkdir(parents=True, exist_ok=True)
    gate = {
        "phase": 1,
        "description": "48K SentencePiece tokenizer validated",
        "checks": [
            {"metric": "tok_fertility",  "op": ">=", "threshold": 3.5},
            {"metric": "tok_roundtrip",  "op": ">=", "threshold": 1.0},
            {"metric": "vocab_size_ok",  "op": ">=", "threshold": 1.0},
        ],
    }
    (EVAL_GATES / "phase_1.json").write_text(json.dumps(gate, indent=2))


def write_phase3_gate():
    """Write the Phase 3 gate definition (700M decoder training)."""
    gate = {
        "phase": 3,
        "description": "700M decoder trained to target perplexity",
        "checks": [
            {"metric": "perplexity",    "op": "<=", "threshold": 30.0},
            {"metric": "val_loss",      "op": "<=", "threshold": 3.40},
            {"metric": "memory_recall", "op": ">=", "threshold": 0.80},
            {"metric": "retrieval_mrr", "op": ">=", "threshold": 0.70},
        ],
    }
    (EVAL_GATES / "phase_3.json").write_text(json.dumps(gate, indent=2))


def write_phase6_gate():
    """Write the Phase 6 gate (full system integration)."""
    gate = {
        "phase": 6,
        "description": "Full COC v3 integration: all subsystems at target",
        "checks": [
            {"metric": "perplexity",    "op": "<=", "threshold": 15.0},
            {"metric": "retrieval_mrr", "op": ">=", "threshold": 0.85},
            {"metric": "memory_recall", "op": ">=", "threshold": 0.90},
            {"metric": "kg_entity_f1",  "op": ">=", "threshold": 0.80},
            {"metric": "reasoning_acc", "op": ">=", "threshold": 0.75},
            {"metric": "emotion_f1",    "op": ">=", "threshold": 0.70},
            {"metric": "overall_score", "op": ">=", "threshold": 0.80},
        ],
    }
    (EVAL_GATES / "phase_6.json").write_text(json.dumps(gate, indent=2))
