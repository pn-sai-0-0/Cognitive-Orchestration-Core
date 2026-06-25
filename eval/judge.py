"""
CognitiveOC v3 — LLM Judge (Evaluation-Only)
==============================================

Scores generated responses on faithfulness, consistency, and citation quality.
DISABLED in runtime by default (JUDGE.enabled = False in config.py).
ENABLED only during eval/run_suite.py evaluation runs.

This module uses the judge encoder (cross-encoder/qnli-electra-base) for
scoring. It does NOT call the 700M decoder — that would contaminate eval.

File: eval/judge.py
Called by: validation/validator.py (mode='eval'), eval/run_suite.py
"""

from __future__ import annotations

import time
from typing import Any

try:
    from config import JUDGE
except ImportError:
    JUDGE = dict(enabled=False, score_threshold=0.60,
                 faithfulness_weight=0.40, consistency_weight=0.30,
                 citation_weight=0.30)


def score(query:    str,
          response: str,
          evidence: list[str] = None,
          mode:     str = "encoder") -> dict:
    """Score a response on three dimensions.

    Dimensions:
        faithfulness  — response grounded in evidence?
        consistency   — response internally consistent?
        citation      — cited sources match evidence?

    Args:
        query:    Original user query.
        response: Generated response.
        evidence: List of evidence strings (retrieved chunks, memories).
        mode:     'encoder' (cross-encoder scoring) | 'heuristic' (fast fallback).

    Returns:
        {"score": float, "faithfulness": float, "consistency": float,
         "citation": float, "reasoning": str, "latency_ms": float}
    """
    evidence = evidence or []
    t0       = time.time()

    if mode == "encoder" and evidence:
        return _encoder_score(query, response, evidence, t0)
    return _heuristic_score(query, response, evidence, t0)


def _encoder_score(query: str, response: str,
                   evidence: list[str], t0: float) -> dict:
    """Cross-encoder based scoring."""
    try:
        from encoder.hub import get_hub
        hub = get_hub()

        # Faithfulness: does response entail evidence?
        faith_scores = hub._cross_enc.score(response, evidence) \
            if hub._cross_enc else []
        faith = float(max(faith_scores)) if faith_scores else 0.5

        # Consistency: self-entailment via sentence similarity
        import re
        sentences = re.split(r"(?<=[.!?])\s+", response.strip())
        if len(sentences) >= 2:
            vecs = hub.encode("semantic", sentences[:5])
            import numpy as np
            sims = []
            for i in range(len(vecs)):
                for j in range(i+1, len(vecs)):
                    sims.append(float(vecs[i] @ vecs[j]))
            consist = float(sum(sims)/len(sims)) if sims else 0.8
        else:
            consist = 0.8

        # Citation: cited sources in evidence
        import re as _re
        cited  = _re.findall(r'\[([^\]]{2,40})\]', response)
        if cited and evidence:
            found  = sum(1 for c in cited
                         if any(c.lower() in e.lower() for e in evidence))
            cit    = found / len(cited)
        else:
            cit = 0.5 if not cited else 0.0

        # Weighted composite
        wf = JUDGE.get("faithfulness_weight", 0.40)
        wc = JUDGE.get("consistency_weight",  0.30)
        wr = JUDGE.get("citation_weight",      0.30)
        composite = wf*faith + wc*consist + wr*cit

        return {
            "score":         round(composite, 3),
            "faithfulness":  round(faith, 3),
            "consistency":   round(consist, 3),
            "citation":      round(cit, 3),
            "reasoning":     (f"faithfulness={faith:.2f} "
                              f"consistency={consist:.2f} citation={cit:.2f}"),
            "latency_ms":    round((time.time()-t0)*1000, 1),
        }
    except Exception as e:
        return _heuristic_score(query, response, evidence, t0, note=str(e))


def _heuristic_score(query: str, response: str,
                     evidence: list[str], t0: float,
                     note: str = "") -> dict:
    """Fast heuristic scoring (fallback when encoder unavailable)."""
    import re

    q_toks = set(re.findall(r"[a-z]{3,}", query.lower()))
    r_toks = set(re.findall(r"[a-z]{3,}", response.lower()))

    # Faithfulness: term overlap between response and evidence
    ev_text = " ".join(evidence).lower()
    ev_toks = set(re.findall(r"[a-z]{3,}", ev_text))
    faith   = len(r_toks & ev_toks) / max(len(r_toks), 1) if ev_toks else 0.5

    # Consistency: response length and structure (heuristic)
    words   = response.split()
    consist = min(len(words) / 50, 1.0) if words else 0.0

    # Citation coverage
    cited = re.findall(r'\[([^\]]{2,40})\]', response)
    cit   = 0.5 if not cited else min(len(cited) / 3, 1.0)

    wf = JUDGE.get("faithfulness_weight", 0.40)
    wc = JUDGE.get("consistency_weight",  0.30)
    wr = JUDGE.get("citation_weight",      0.30)
    composite = wf*faith + wc*consist + wr*cit

    return {
        "score":         round(composite, 3),
        "faithfulness":  round(faith, 3),
        "consistency":   round(consist, 3),
        "citation":      round(cit, 3),
        "reasoning":     f"heuristic mode{f' ({note})' if note else ''}",
        "latency_ms":    round((time.time()-t0)*1000, 1),
    }
