"""Claim-level comparison for COC self-consistency.

Claim equivalence is the primary agreement signal. Answer text, traces,
confidence, citations, and tool-use remain supporting signals.
"""

from __future__ import annotations

import re
from collections import Counter
from difflib import SequenceMatcher

from self_consistency.claim_normalizer import Claim, claim_equivalence, normalize_claim
from self_consistency.contracts import (
    ConsistencyMetrics,
    ConsistencyResult,
    ReasoningAttempt,
)

_STOP = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "if",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "to",
        "with",
        "this",
        "these",
        "those",
        "using",
        "used",
        "based",
        "available",
        "evidence",
        "limited",
        "confidence",
        "best",
        "effort",
        "answer",
        "supported",
        "supporting",
        "memory",
        "document",
        "knowledge",
        "graph",
    }
)


def _norm(text: str) -> str:
    value = (text or "").lower()
    value = re.sub(r"limited-confidence answer:\s*", "", value)
    value = re.sub(r"best supported answer from [a-z0-9_.\-/]+:\s*", "", value)
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9._ -]", " ", value)).strip()


def answer_similarity(left: str, right: str) -> float:
    lt = {x for x in _norm(left).split() if len(x) >= 3 and x not in _STOP}
    rt = {x for x in _norm(right).split() if len(x) >= 3 and x not in _STOP}
    if not lt and not rt:
        return 1.0
    jaccard = len(lt & rt) / max(len(lt | rt), 1)
    sequence = SequenceMatcher(a=_norm(left), b=_norm(right)).ratio()
    ln = tuple(re.findall(r"\d+(?:\.\d+)?", left))
    rn = tuple(re.findall(r"\d+(?:\.\d+)?", right))
    if ln or rn:
        numeric = 1.0 if ln == rn else 0.0
        return round(0.40 * jaccard + 0.25 * sequence + 0.35 * numeric, 4)
    return round(0.55 * jaccard + 0.45 * sequence, 4)


def _avg(values: list[float], default: float = 1.0) -> float:
    return round(sum(values) / len(values), 4) if values else default


def _jaccard(left: tuple[str, ...], right: tuple[str, ...]) -> float:
    a, b = set(left), set(right)
    if not a and not b:
        return 1.0
    return len(a & b) / max(len(a | b), 1)


def _low_signal(attempt: ReasoningAttempt, claim: Claim) -> bool:
    return claim.negation or attempt.evidence_strength < 0.45 or bool(attempt.warnings)


def compare_attempts(
    query: str,
    attempts: tuple[ReasoningAttempt, ...],
    answer_threshold: float = 0.72,
    support_threshold: float = 0.65,
    claim_threshold: float = 0.7,
) -> ConsistencyResult:
    claims = [normalize_claim(a.answer) for a in attempts]
    clusters: list[list[int]] = []
    reps: list[Claim] = []
    for index, claim in enumerate(claims):
        for slot, rep in enumerate(reps):
            if claim_equivalence(claim, rep) >= claim_threshold:
                clusters[slot].append(index)
                break
        else:
            clusters.append([index])
            reps.append(claim)

    best = max(
        clusters,
        key=lambda c: (len(c), sum(attempts[i].confidence for i in c) / len(c)),
    )
    consensus_original = max((attempts[i] for i in best), key=lambda a: a.confidence)
    minority_original = tuple(a for i, a in enumerate(attempts) if i not in best)
    agreement_ratio = round(len(best) / len(attempts), 4)

    answer_scores, trace_scores, citation_scores, claim_scores = [], [], [], []
    for i in range(len(attempts)):
        for j in range(i + 1, len(attempts)):
            answer_scores.append(
                answer_similarity(attempts[i].answer, attempts[j].answer)
            )
            trace_scores.append(
                SequenceMatcher(
                    a="|".join(attempts[i].trace), b="|".join(attempts[j].trace)
                ).ratio()
            )
            citation_scores.append(
                _jaccard(attempts[i].citations, attempts[j].citations)
            )
            claim_scores.append(claim_equivalence(claims[i], claims[j]))

    claim_agreement = _avg(claim_scores)
    evidence_strength = round(
        sum(a.evidence_strength for a in attempts) / len(attempts), 4
    )
    confidence_agreement = round(
        1.0
        - (max(a.confidence for a in attempts) - min(a.confidence for a in attempts)),
        4,
    )
    tool_sets = [tuple(sorted(set(a.tool_usage))) for a in attempts]
    tool_agreement = round(Counter(tool_sets).most_common(1)[0][1] / len(tool_sets), 4)

    agreement = agreement_ratio >= 0.6 and claim_agreement >= claim_threshold
    superficial = False
    warning = ""
    agreement_type = "strong_agreement"
    if not agreement:
        agreement_type = "clear_disagreement"
        warning = (
            "claim-level divergence across attempts "
            f"(agreement_ratio={agreement_ratio:.2f}, "
            f"claim_agreement={claim_agreement:.2f})"
        )
    elif all(_low_signal(a, c) for a, c in zip(attempts, claims)):
        agreement_type = "false_agreement"
        warning = "answers align but evidence is thin or caveated"
        superficial = True
    elif (
        _avg(trace_scores) < support_threshold
        or confidence_agreement < 0.8
        or evidence_strength < 0.7
    ):
        agreement_type = "superficial_agreement"
        warning = "answers align semantically but supporting signals are weak"
        superficial = True

    enriched = tuple(
        ReasoningAttempt(
            a.attempt_id,
            a.strategy,
            a.answer,
            a.confidence,
            a.trace,
            a.citations,
            a.tool_usage,
            a.warnings,
            a.evidence_strength,
            a.answer_similarity_key,
            a.reasoning,
            c.to_dict(),
        )
        for a, c in zip(attempts, claims)
    )
    consensus = enriched[attempts.index(consensus_original)]
    minority = tuple(enriched[attempts.index(a)] for a in minority_original)
    metrics = ConsistencyMetrics(
        len(attempts),
        agreement_ratio,
        _avg(answer_scores),
        _avg(trace_scores),
        confidence_agreement,
        _avg(citation_scores),
        tool_agreement,
        evidence_strength,
        claim_agreement,
    )
    return ConsistencyResult(
        query,
        enriched,
        consensus,
        minority,
        agreement and agreement_type == "strong_agreement",
        agreement_type,
        warning,
        superficial,
        consensus.confidence,
        metrics,
    )
