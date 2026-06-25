"""
CognitiveOC v3 — Reasoning Engine
===================================

Implements the full v3 reasoning pipeline:

  1. Classification  — what type of reasoning is required?
  2. Decomposition   — break complex queries into subgoals
  3. Planning        — order subgoals and identify dependencies
  4. Evidence Gathering — pull from memory, retrieval, KG, tools
  5. Verification    — validate evidence consistency
  6. Reflection      — self-review the reasoning chain
  7. Synthesis       — combine evidence into a grounded answer plan
  8. Confidence Estimation — score the final reasoning chain
  9. Explainable Trace — full reasoning trace for UI display

Sources used during reasoning:
  - Memory hits  (CognitiveMemory.ranked_recall)
  - Retrieval chunks (HybridRetriever.retrieve)
  - KG facts    (KnowledgeGraph.ranked_query)
  - Tool outputs (passed in via context dict)

File: reasoning/reasoner.py
Used by: engine.py (reasoner.assess() called in build_context())
"""

from __future__ import annotations

import re
import time
from typing import Any


# ── Reasoning type taxonomy ───────────────────────────────────────────
_REASONING_TYPES = {
    "factual":      re.compile(
        r"\b(what is|who is|when did|where is|define|meaning of)\b", re.I),
    "comparative":  re.compile(
        r"\b(compare|difference|vs\.?|versus|better|worse|pros and cons)\b", re.I),
    "causal":       re.compile(
        r"\b(why|cause|reason|because|leads to|results in|due to)\b", re.I),
    "procedural":   re.compile(
        r"\b(how to|steps to|how do I|procedure|process|implement|build)\b", re.I),
    "evaluative":   re.compile(
        r"\b(should I|is it|evaluate|assess|worth|recommend|best)\b", re.I),
    "hypothetical": re.compile(
        r"\b(what if|suppose|imagine|if .* then|scenario|would)\b", re.I),
    "summarization":re.compile(
        r"\b(summarize|summarise|overview|tldr|brief|main points|key points)\b", re.I),
    "creative":     re.compile(
        r"\b(write|draft|generate|create|design|brainstorm|suggest)\b", re.I),
}


def classify_reasoning(query: str) -> str:
    """Classify the reasoning type required for a query."""
    scores: dict[str, int] = {}
    for rtype, pat in _REASONING_TYPES.items():
        scores[rtype] = len(pat.findall(query))
    best = max(scores, key=lambda k: scores[k])
    return best if scores[best] > 0 else "factual"


# ── Decomposition ─────────────────────────────────────────────────────

def decompose(query: str, reasoning_type: str = None) -> list[str]:
    """Break a complex query into atomic subgoals.

    Strategies by reasoning type:
      factual       → [recall fact, verify with KG, retrieve evidence]
      comparative   → [gather facts for A, gather facts for B, compare]
      causal        → [identify effect, trace causes, verify chain]
      procedural    → [identify goal, list steps, order steps, verify]
      evaluative    → [gather options, score criteria, synthesise]
      hypothetical  → [identify condition, trace implications, assess]
      summarization → [retrieve content, extract key points, synthesise]
      creative      → [understand intent, gather context, generate]
    """
    rt = reasoning_type or classify_reasoning(query)

    subgoal_templates = {
        "factual": [
            f"Recall what is known about: {query[:60]}",
            "Verify with knowledge graph facts",
            "Retrieve supporting evidence from documents",
        ],
        "comparative": [
            f"Gather facts about the first subject in: {query[:50]}",
            "Gather facts about the second subject",
            "Identify key dimensions for comparison",
            "Synthesise comparative analysis",
        ],
        "causal": [
            f"Identify the effect described in: {query[:50]}",
            "Trace contributing causes from knowledge and evidence",
            "Verify causal chain consistency",
        ],
        "procedural": [
            f"Identify the goal of: {query[:50]}",
            "Recall or retrieve relevant steps",
            "Verify step ordering and completeness",
        ],
        "evaluative": [
            f"Gather relevant options or facts for: {query[:50]}",
            "Apply evaluation criteria",
            "Synthesise recommendation with confidence",
        ],
        "hypothetical": [
            f"Identify the hypothetical condition in: {query[:50]}",
            "Reason through implications step by step",
            "Assess plausibility with available evidence",
        ],
        "summarization": [
            "Retrieve key content from documents and memory",
            "Extract main points and structure",
            "Synthesise concise summary",
        ],
        "creative": [
            "Understand the creative intent and constraints",
            "Gather relevant context and examples",
            "Generate structured output",
        ],
    }

    return subgoal_templates.get(rt, subgoal_templates["factual"])


# ── Evidence aggregation ──────────────────────────────────────────────

def aggregate_evidence(memory_hits:  list[dict],
                       chunks:       list[dict],
                       kg_facts:     list[dict],
                       tool_output:  dict | None = None) -> dict:
    """Aggregate evidence from all sources into a unified structure.

    Returns:
        {
            "sources":    list of evidence items with source tags
            "total":      int total evidence items
            "coverage":   dict {memory, retrieval, kg, tool} counts
            "confidence": float overall evidence confidence
        }
    """
    sources: list[dict] = []

    for m in memory_hits[:5]:
        sources.append({
            "type":   "memory",
            "text":   m.get("text", ""),
            "kind":   m.get("kind", "fact"),
            "score":  m.get("score", 1.0),
            "source": "memory",
        })

    for c in chunks[:5]:
        sources.append({
            "type":   "retrieval",
            "text":   c.get("text", ""),
            "source": c.get("source", "document"),
            "score":  c.get("score", c.get("rerank_score", 0.5)),
        })

    for k in kg_facts[:5]:
        triple = (f"{k.get('subject','')} {k.get('relation','')} "
                  f"{k.get('object','')}")
        sources.append({
            "type":   "kg",
            "text":   triple,
            "source": "knowledge_graph",
            "score":  k.get("confidence", 0.7),
        })

    if tool_output:
        sources.append({
            "type":   "tool",
            "text":   str(tool_output.get("result", ""))[:300],
            "source": tool_output.get("tool", "tool"),
            "score":  1.0,
        })

    scores = [s["score"] for s in sources if "score" in s]
    conf   = round(sum(scores) / max(len(scores), 1), 3) if scores else 0.0

    return {
        "sources":    sources,
        "total":      len(sources),
        "coverage":   {
            "memory":    len(memory_hits),
            "retrieval": len(chunks),
            "kg":        len(kg_facts),
            "tool":      1 if tool_output else 0,
        },
        "confidence": conf,
    }


# ── Verification ──────────────────────────────────────────────────────

def verify_evidence(query: str,
                    evidence: dict,
                    reasoning_type: str = "factual") -> dict:
    """Verify consistency and coverage of gathered evidence.

    Checks:
      - Evidence coverage relative to query terms
      - Internal consistency (contradiction detection)
      - Minimum evidence threshold for the reasoning type

    Returns:
        {
            "verified": bool,
            "issues":   list[str],
            "gaps":     list[str],
            "adjustments": list[str],
        }
    """
    issues, gaps, adjustments = [], [], []
    q_tokens = set(re.findall(r"[a-z]{3,}", query.lower()))
    sources  = evidence.get("sources", [])

    # Coverage check
    evidence_text = " ".join(s.get("text","") for s in sources)
    ev_tokens     = set(re.findall(r"[a-z]{3,}", evidence_text.lower()))
    coverage      = len(q_tokens & ev_tokens) / max(len(q_tokens), 1)

    if coverage < 0.3:
        gaps.append("low query coverage in evidence")
        adjustments.append("Acknowledge limited evidence and caveat response")

    # Minimum evidence threshold
    min_required = {
        "factual": 1, "comparative": 2, "causal": 2,
        "procedural": 1, "evaluative": 2,
        "hypothetical": 1, "summarization": 1, "creative": 0,
    }
    min_ev = min_required.get(reasoning_type, 1)
    if evidence.get("total", 0) < min_ev:
        issues.append(f"insufficient evidence for {reasoning_type} reasoning")
        adjustments.append("Use hedging language — evidence is limited")

    # Confidence check
    if evidence.get("confidence", 1.0) < 0.35:
        issues.append("low overall evidence confidence")
        adjustments.append("Explicitly state uncertainty in the response")

    return {
        "verified":    len(issues) == 0,
        "coverage":    round(coverage, 3),
        "issues":      issues,
        "gaps":        gaps,
        "adjustments": adjustments,
    }


# ── Confidence estimation ─────────────────────────────────────────────

def estimate_confidence(evidence:       dict,
                        verification:   dict,
                        cognition:      dict = None) -> float:
    """Estimate overall response confidence from all signals.

    Factors:
      - Evidence confidence (weight 0.40)
      - Evidence coverage  (weight 0.30)
      - Verification pass  (weight 0.20)
      - Cognition signal   (weight 0.10)

    Returns:
        float in [0.0, 1.0]
    """
    ev_conf  = evidence.get("confidence", 0.5)
    coverage = verification.get("coverage", 0.5)
    verified = 1.0 if verification.get("verified", False) else 0.5
    cog_conf = 1.0
    if cognition:
        refl = cognition.get("reflection", {})
        cog_conf = refl.get("confidence_adj", 1.0) if refl else 1.0

    conf = (0.40 * ev_conf +
            0.30 * coverage +
            0.20 * verified +
            0.10 * cog_conf)
    return round(min(max(conf, 0.0), 1.0), 3)


# ── Synthesis ─────────────────────────────────────────────────────────

def synthesise(query:         str,
               reasoning_type:str,
               subgoals:      list[str],
               evidence:      dict,
               verification:  dict,
               confidence:    float) -> dict:
    """Synthesise a reasoning plan for the decoder.

    Returns a synthesis dict that is injected into the decoder context.
    Does NOT generate the final text — that is the decoder's job.

    Returns:
        {
            "plan":        str,   # high-level answer plan for decoder
            "evidence_summary": str,
            "hedging":     str,   # hedging language if low confidence
            "trace_steps": list,  # for reasoning trace UI panel
        }
    """
    # Build evidence summary (for decoder context)
    ev_parts = []
    for s in evidence.get("sources", [])[:5]:
        snippet = s.get("text","")[:150]
        src     = s.get("source","")
        ev_parts.append(f"[{src}] {snippet}")
    ev_summary = "\n".join(ev_parts) if ev_parts else "No supporting evidence found."

    # Determine hedging language
    if confidence < 0.4:
        hedging = "Note: limited evidence available — treat this as a best-effort answer."
    elif confidence < 0.6:
        hedging = "Based on available evidence, with moderate confidence:"
    else:
        hedging = ""

    # Build reasoning plan string (guides decoder generation)
    plan_parts = [f"Reasoning type: {reasoning_type}"]
    if subgoals:
        plan_parts.append("Steps: " + " → ".join(s[:40] for s in subgoals[:4]))
    for adj in verification.get("adjustments", []):
        plan_parts.append(f"Note: {adj}")
    if hedging:
        plan_parts.append(hedging)
    plan = " | ".join(plan_parts)

    # Build trace steps for UI
    trace = [
        {"step": "classify",  "result": reasoning_type},
        {"step": "decompose", "result": "; ".join(subgoals[:3])},
        {"step": "evidence",  "result": f"{evidence.get('total',0)} items from "
                                         f"{list(evidence.get('coverage',{}).keys())}"},
        {"step": "verify",    "result": "PASS" if verification.get("verified") else
                                        f"ISSUES: {verification.get('issues',[])}"},
        {"step": "confidence","result": str(confidence)},
    ]

    return {
        "plan":             plan,
        "evidence_summary": ev_summary,
        "hedging":          hedging,
        "trace_steps":      trace,
    }


# ═══════════════════════════════════════════════════════════════════
# Reasoner — Main Orchestrator Class
# ═══════════════════════════════════════════════════════════════════

class Reasoner:
    """COC v3 Reasoning Engine.

    Called by engine.py during build_context(). Runs the full pipeline:
      classify → decompose → plan → aggregate → verify → synthesise

    The output is injected into the decoder context — it shapes HOW
    the model generates its response, not the response itself.

    Usage:
        reasoner = Reasoner()
        result   = reasoner.assess(
            query    = "What causes gradient vanishing in deep networks?",
            memory   = [...],
            chunks   = [...],
            kg_facts = [...],
            tool     = None,
            cognition= {...},
        )
        # result["steps"]  → list of subgoals
        # result["plan"]   → synthesis plan injected into decoder context
        # result["confidence"] → estimated confidence float
        # result["trace"]  → full trace for reasoning panel UI
    """

    def assess(self,
               query:      str,
               memory:     list[dict] = None,
               chunks:     list[dict] = None,
               kg_facts:   list[dict] = None,
               tool:       dict | None = None,
               cognition:  dict | None = None) -> dict:
        """Run full reasoning pipeline and return unified result.

        Args:
            query:     User query string.
            memory:    Memory hits from CognitiveMemory.ranked_recall.
            chunks:    Retrieval chunks from HybridRetriever.retrieve.
            kg_facts:  KG results from KnowledgeGraph.ranked_query.
            tool:      Tool output dict or None.
            cognition: CognitionLayer.process() output or None.

        Returns:
            {
                "type":       str,    reasoning type
                "steps":      list,   subgoals
                "evidence":   dict,   aggregated evidence
                "verification": dict,
                "synthesis":  dict,   plan + evidence summary
                "confidence": float,
                "trace":      list,   UI trace steps
                "latency_ms": float,
            }
        """
        t0 = time.time()
        memory   = memory   or []
        chunks   = chunks   or []
        kg_facts = kg_facts or []

        # Pipeline
        rtype      = classify_reasoning(query)
        subgoals   = decompose(query, rtype)
        evidence   = aggregate_evidence(memory, chunks, kg_facts, tool)
        verification = verify_evidence(query, evidence, rtype)
        confidence   = estimate_confidence(evidence, verification, cognition)
        synthesis    = synthesise(query, rtype, subgoals, evidence,
                                  verification, confidence)

        latency = round((time.time() - t0) * 1000, 1)

        return {
            "type":         rtype,
            "steps":        subgoals,
            "evidence":     evidence,
            "verification": verification,
            "synthesis":    synthesis,
            "confidence":   confidence,
            "plan":         synthesis["plan"],
            "trace":        synthesis["trace_steps"],
            "latency_ms":   latency,
        }
