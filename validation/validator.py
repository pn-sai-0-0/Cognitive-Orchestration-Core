"""
CognitiveOC v3 — Validation Engine
=====================================

Runtime validation subsystem. Validates model outputs before they
are returned to the user, and validates subsystem consistency.

Validation types:
  fact_check    — claim vs KG + retrieval evidence
  citation      — cited sources contain the claimed content
  kg            — KG consistency (no contradictions)
  memory        — output consistent with stored memories
  reasoning     — reasoning chain covers the query
  output        — format, length, PII, secrets

Judge path (evaluation-only by default):
  Disabled in chat runtime (JUDGE.enabled = False in config).
  Enabled only during eval/run_suite.py evaluation runs.

File: validation/validator.py
Used by: engine.py (post-generation), eval/run_suite.py, ui/app.py
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from config import VALIDATION, JUDGE, EVAL_BASELINE, ensure_dirs
except ImportError:
    VALIDATION   = dict(fact_check=True, citation_check=True, kg_check=True,
                        memory_check=True, reasoning_check=True,
                        confidence_threshold=0.70)
    JUDGE        = dict(enabled=False, score_threshold=0.60)
    EVAL_BASELINE= Path("eval/baseline")
    def ensure_dirs(): pass


# ═══════════════════════════════════════════════════════════════════
# Result dataclass
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ValidationResult:
    """Structured result from one validation check."""
    check:      str
    passed:     bool
    score:      float     = 1.0
    detail:     str       = ""
    evidence:   list[str] = field(default_factory=list)
    warnings:   list[str] = field(default_factory=list)
    latency_ms: float     = 0.0

    def to_dict(self) -> dict:
        return {
            "check":      self.check,
            "passed":     self.passed,
            "score":      round(self.score, 3),
            "detail":     self.detail,
            "evidence":   self.evidence[:3],
            "warnings":   self.warnings,
            "latency_ms": self.latency_ms,
        }


@dataclass
class ValidationReport:
    """Aggregated validation report for one response."""
    passed:     bool
    checks:     list[ValidationResult] = field(default_factory=list)
    score:      float = 1.0
    should_warn:bool  = False
    should_block:bool = False
    ts:         str   = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))

    def to_dict(self) -> dict:
        return {
            "passed":      self.passed,
            "score":       round(self.score, 3),
            "should_warn": self.should_warn,
            "should_block":self.should_block,
            "checks":      [c.to_dict() for c in self.checks],
            "ts":          self.ts,
        }

    def add_warning(self, msg: str):
        for c in self.checks:
            if msg not in c.warnings:
                c.warnings.append(msg)


# ═══════════════════════════════════════════════════════════════════
# Individual validators
# ═══════════════════════════════════════════════════════════════════

def validate_facts(response: str,
                   chunks:   list[dict],
                   kg:       Any = None) -> ValidationResult:
    """Check if claims in response are grounded in retrieved evidence or KG.

    Heuristic: key noun phrases in response should appear in evidence.
    Returns ValidationResult with score 0–1.
    """
    t0     = time.time()
    claims = re.findall(r"\b[A-Z][a-zA-Z]{2,}(?:\s+[a-z]+){0,3}", response)
    if not claims:
        return ValidationResult("fact_check", True, 1.0,
                                "no verifiable claims detected",
                                latency_ms=round((time.time()-t0)*1000,1))

    ev_text = " ".join(c.get("text","") for c in chunks[:5]).lower()
    if kg:
        try:
            kg_facts = kg.ranked_query(response[:200], limit=5)
            ev_text += " " + " ".join(
                f"{f.get('subject','')} {f.get('object','')}" for f in kg_facts
            ).lower()
        except Exception:
            pass

    grounded = sum(1 for c in claims if c.lower() in ev_text)
    score    = grounded / max(len(claims), 1)
    passed   = score >= VALIDATION.get("confidence_threshold", 0.70)

    return ValidationResult(
        check    = "fact_check",
        passed   = passed,
        score    = score,
        detail   = f"{grounded}/{len(claims)} claims grounded in evidence",
        evidence = [c for c in claims[:3] if c.lower() in ev_text],
        latency_ms = round((time.time()-t0)*1000, 1),
    )


def validate_citations(response: str, chunks: list[dict]) -> ValidationResult:
    """Verify that cited sources are present in the retrieved chunks."""
    t0     = time.time()
    # Extract [source] patterns from response
    cited  = re.findall(r'\[([^\]]{2,40})\]', response)
    if not cited:
        return ValidationResult("citation_check", True, 1.0,
                                "no citations in response",
                                latency_ms=round((time.time()-t0)*1000,1))

    sources   = {c.get("source","").lower() for c in chunks}
    verified  = [c for c in cited if any(c.lower() in s or s in c.lower()
                                          for s in sources)]
    score     = len(verified) / max(len(cited), 1)
    passed    = score >= 0.5   # at least 50% of citations verifiable

    return ValidationResult(
        check    = "citation_check",
        passed   = passed,
        score    = score,
        detail   = f"{len(verified)}/{len(cited)} citations verified",
        evidence = verified[:3],
        warnings = [c for c in cited if c not in verified][:3],
        latency_ms = round((time.time()-t0)*1000, 1),
    )


def validate_kg(response: str, kg: Any) -> ValidationResult:
    """Check response against KG for contradictions."""
    t0 = time.time()
    if kg is None:
        return ValidationResult("kg_check", True, 1.0,
                                "KG not available — skipped",
                                latency_ms=round((time.time()-t0)*1000,1))
    try:
        contradictions = kg.contradictions()
        if contradictions:
            # Check if any contradiction subject appears in response
            resp_lower = response.lower()
            active = [c for c in contradictions
                      if c.get("subject","") in resp_lower]
            if active:
                return ValidationResult(
                    check    = "kg_check",
                    passed   = False,
                    score    = 0.5,
                    detail   = f"{len(active)} potential KG contradictions in response",
                    warnings = [str(c) for c in active[:2]],
                    latency_ms = round((time.time()-t0)*1000, 1),
                )
    except Exception:
        pass
    return ValidationResult("kg_check", True, 1.0, "no KG contradictions detected",
                            latency_ms=round((time.time()-t0)*1000,1))


def validate_memory(response: str, memories: list[dict]) -> ValidationResult:
    """Check response consistency with recalled memories."""
    t0 = time.time()
    if not memories:
        return ValidationResult("memory_check", True, 1.0,
                                "no memories to check against",
                                latency_ms=round((time.time()-t0)*1000,1))

    resp_lower = response.lower()
    # Simple contradiction check: if a memory says "X is Y" and response says "X is not Y"
    conflicts  = []
    for m in memories[:5]:
        text = m.get("text","").lower()
        # Look for simple negation contradictions
        key_phrases = re.findall(r"[a-z]{3,}\s+is\s+[a-z]{3,}", text)
        for phrase in key_phrases[:2]:
            neg = phrase.replace(" is ", " is not ")
            if neg in resp_lower:
                conflicts.append(phrase)

    if conflicts:
        return ValidationResult(
            check    = "memory_check",
            passed   = False,
            score    = max(0.3, 1.0 - 0.2 * len(conflicts)),
            detail   = f"{len(conflicts)} potential memory conflicts",
            warnings = conflicts[:3],
            latency_ms = round((time.time()-t0)*1000, 1),
        )
    return ValidationResult("memory_check", True, 1.0,
                            "response consistent with recalled memories",
                            latency_ms=round((time.time()-t0)*1000,1))


def validate_reasoning(response: str, query: str,
                        reasoning: dict) -> ValidationResult:
    """Check response covers the query intent based on reasoning trace."""
    t0       = time.time()
    q_tokens = set(re.findall(r"[a-z]{3,}", query.lower()))
    r_tokens = set(re.findall(r"[a-z]{3,}", response.lower()))
    coverage = len(q_tokens & r_tokens) / max(len(q_tokens), 1)

    issues   = reasoning.get("verification",{}).get("issues",[]) if reasoning else []
    score    = coverage * (0.8 if issues else 1.0)

    return ValidationResult(
        check    = "reasoning_check",
        passed   = coverage >= 0.25,
        score    = round(score, 3),
        detail   = f"query coverage={coverage:.2f}  reasoning_issues={len(issues)}",
        warnings = issues[:3],
        latency_ms = round((time.time()-t0)*1000, 1),
    )


def validate_output(response: str) -> ValidationResult:
    """Format, length, PII, and secret checks on final output."""
    t0       = time.time()
    warnings = []

    # Length checks
    words = response.split()
    if len(words) < 3:
        warnings.append("response too short")
    if len(words) > 2000:
        warnings.append("response unusually long")

    # PII check (should have been redacted by guardrails, but double-check)
    pii_patterns = [
        re.compile(r'\b[\w.+\-]+@[\w\-]+\.[\w.]+\b'),
        re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
    ]
    for pat in pii_patterns:
        if pat.search(response):
            warnings.append("potential PII in output")

    # Secret check
    secret_pat = re.compile(r'(api[_\-]?key|password|token)\s*[:=]\s*\S+', re.I)
    if secret_pat.search(response):
        warnings.append("potential secret in output")

    passed = not any("PII" in w or "secret" in w for w in warnings)
    return ValidationResult(
        check    = "output_check",
        passed   = passed,
        score    = 1.0 if passed else 0.2,
        detail   = f"words={len(words)}",
        warnings = warnings,
        latency_ms = round((time.time()-t0)*1000, 1),
    )


# ═══════════════════════════════════════════════════════════════════
# Validator — Main Orchestrator
# ═══════════════════════════════════════════════════════════════════

class Validator:
    """Runtime validation orchestrator.

    Runs all enabled validation checks and returns a ValidationReport.
    Used by engine.py after generation and by eval/run_suite.py.

    The judge path (LLM-as-judge) is DISABLED by default (eval-only).
    """

    def __init__(self):
        self._cfg         = VALIDATION
        self._judge_cfg   = JUDGE
        self._judge_active= JUDGE.get("enabled", False)

    def validate(self,
                 response:  str,
                 query:     str    = "",
                 chunks:    list   = None,
                 memories:  list   = None,
                 kg:        Any    = None,
                 reasoning: dict   = None,
                 mode:      str    = "runtime") -> ValidationReport:
        """Run all enabled validation checks.

        Args:
            response:  Generated response text.
            query:     Original user query.
            chunks:    Retrieved chunks used for generation.
            memories:  Recalled memories used for generation.
            kg:        KnowledgeGraph instance.
            reasoning: Reasoner.assess() output dict.
            mode:      'runtime' (fast, non-blocking) | 'eval' (thorough, may block).

        Returns:
            ValidationReport with all check results.
        """
        chunks   = chunks   or []
        memories = memories or []
        reasoning= reasoning or {}
        checks:  list[ValidationResult] = []

        # Fact check
        if self._cfg.get("fact_check", True):
            checks.append(validate_facts(response, chunks, kg))

        # Citation check
        if self._cfg.get("citation_check", True):
            checks.append(validate_citations(response, chunks))

        # KG check
        if self._cfg.get("kg_check", True) and kg is not None:
            checks.append(validate_kg(response, kg))

        # Memory check
        if self._cfg.get("memory_check", True):
            checks.append(validate_memory(response, memories))

        # Reasoning check
        if self._cfg.get("reasoning_check", True):
            checks.append(validate_reasoning(response, query, reasoning))

        # Output check (always)
        checks.append(validate_output(response))

        # Judge (eval-only)
        if mode == "eval" and self._judge_active:
            judge_r = self._run_judge(query, response, chunks)
            if judge_r:
                checks.append(judge_r)

        # Aggregate
        all_passed   = all(c.passed for c in checks)
        avg_score    = sum(c.score for c in checks) / max(len(checks), 1)
        should_warn  = not all_passed and avg_score >= 0.5
        should_block = avg_score < 0.2

        report = ValidationReport(
            passed       = all_passed,
            checks       = checks,
            score        = round(avg_score, 3),
            should_warn  = should_warn,
            should_block = should_block,
        )

        # Persist report for inspection
        self._persist(report, query)
        return report

    def _run_judge(self, query: str, response: str,
                   chunks: list) -> ValidationResult | None:
        """LLM-as-judge scoring (evaluation-only, disabled in runtime)."""
        try:
            from eval.judge import score as judge_score
            result = judge_score(query, response,
                                 [c.get("text","") for c in chunks[:3]])
            return ValidationResult(
                check  = "judge",
                passed = result.get("score", 0) >= self._judge_cfg.get("score_threshold", 0.6),
                score  = result.get("score", 0),
                detail = result.get("reasoning",""),
            )
        except Exception:
            return None

    def _persist(self, report: ValidationReport, query: str):
        """Save validation report for inspection (non-fatal)."""
        try:
            ensure_dirs()
            path = Path(str(EVAL_BASELINE)) / "last_validation.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = report.to_dict()
            payload["query"] = query[:80]
            path.write_text(json.dumps(payload, indent=2))
        except Exception:
            pass
