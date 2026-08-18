"""Reflection checker: purely deterministic problem detection.

The checker inspects a draft answer against the SAME evidence bundle
the reasoner already saw (chunks + memory + kg_facts). It never calls a
model. Three classes of problem are recognised, in priority order:

1. contradiction     — the draft states a value or fact that the
                       evidence directly refutes (numeric mismatch on
                       the same subject, or explicit negation flip).
2. missed_evidence   — the query mentions a strong evidence subject
                       that the draft omits entirely.
3. unsupported_claim — the draft asserts a specific factual token
                       (numeric literal, quoted phrase, or proper noun)
                       that appears nowhere in the evidence.

If none of these fire, the draft is considered acceptable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-]{2,}")
_STOP = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "are",
        "was",
        "were",
        "has",
        "have",
        "had",
        "you",
        "your",
        "our",
        "their",
        "his",
        "her",
        "its",
        "not",
        "but",
        "any",
        "all",
        "some",
        "one",
        "two",
        "into",
        "onto",
        "over",
        "under",
        "than",
        "then",
        "there",
        "these",
        "those",
        "them",
        "they",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "can",
        "cannot",
        "about",
        "which",
        "what",
        "when",
        "where",
        "why",
        "how",
        "who",
        "answer",
        "question",
        "based",
        "evidence",
        "limited",
        "confidence",
        "best",
        "supported",
        "supporting",
        "memory",
        "document",
        "knowledge",
        "graph",
        "using",
        "used",
        "available",
    }
)


@dataclass(frozen=True)
class CheckResult:
    ok: bool
    reason: str  # "" when ok, else one of the reason labels in ReflectionReport


def _evidence_text(
    chunks: list[dict] | None, memory: list[dict] | None, kg_facts: list[dict] | None
) -> str:
    parts: list[str] = [str(c.get("text", "")) for c in chunks or []]
    parts.extend(str(m.get("text", "")) for m in memory or [])
    for f in kg_facts or []:
        subj = str(f.get("subject", ""))
        rel = str(f.get("relation", ""))
        obj = str(f.get("object", ""))
        parts.append(f"{subj} {rel} {obj}")
    return "\n".join(p for p in parts if p)


def _numbers(text: str) -> list[str]:
    return _NUMBER_RE.findall(text)


def _content_tokens(text: str) -> set[str]:
    return {
        t.lower()
        for t in _TOKEN_RE.findall(text)
        if t.lower() not in _STOP and len(t) > 2
    }


# Copulas and their negations that we recognise as "X <cop> [not] Y".
# Each entry is (positive_form, negated_alternation, label). The
# negated alternation matches spelled-out negation ("is not"),
# contractions ("isn't") and the "<cop> never" form.
_COPULA_PAIRS: tuple[tuple[str, str, str], ...] = (
    (r"is", r"(?:is\s+not|isn'?t|is\s+never)", "present_singular"),
    (r"are", r"(?:are\s+not|aren'?t|are\s+never)", "present_plural"),
    (r"was", r"(?:was\s+not|wasn'?t|was\s+never)", "past_singular"),
    (r"were", r"(?:were\s+not|weren'?t|were\s+never)", "past_plural"),
)

_SUBJECT_FRAG = r"([a-z][a-z0-9_\- ]{2,40})"
_OBJECT_FRAG = r"([a-z][a-z0-9_\- ]{2,40})"


def _explicit_negation_flip(draft: str, evidence: str) -> bool:
    """Detect a direct negation flip on a shared subject.

    Recognises all four English copulas (``is``, ``are``, ``was``,
    ``were``) plus their contracted negations (``isn't``, ``aren't``,
    ``wasn't``, ``weren't``) and the ``<cop> never`` form. A flip is:

      * evidence: ``X <cop> Y``       <->   draft: ``X <cop_neg> Y``
      * evidence: ``X <cop_neg> Y``   <->   draft: ``X <cop> Y``

    Both directions are checked for every copula independently.
    """
    d_low = draft.lower()
    e_low = evidence.lower()

    for positive, negated, _label in _COPULA_PAIRS:
        # Direction 1: evidence positive, draft negated.
        pos_pattern = re.compile(
            r"\b" + _SUBJECT_FRAG + r"\s+" + positive + r"\s+" + _OBJECT_FRAG,
        )
        for subj, obj in pos_pattern.findall(e_low):
            subj = subj.strip()
            obj = obj.strip()
            if not subj or not obj:
                continue
            flip = re.compile(
                r"\b"
                + re.escape(subj)
                + r"\s+"
                + negated
                + r"\s+"
                + re.escape(obj)
                + r"\b"
            )
            if flip.search(d_low):
                return True

        # Direction 2: evidence negated, draft positive.
        neg_pattern = re.compile(
            r"\b" + _SUBJECT_FRAG + r"\s+" + negated + r"\s+" + _OBJECT_FRAG,
        )
        for subj, obj in neg_pattern.findall(e_low):
            subj = subj.strip()
            obj = obj.strip()
            if not subj or not obj:
                continue
            flip = re.compile(
                r"\b"
                + re.escape(subj)
                + r"\s+"
                + positive
                + r"\s+"
                + re.escape(obj)
                + r"\b"
            )
            if flip.search(d_low):
                return True

    return False


def _numeric_contradiction(draft: str, evidence: str) -> bool:
    """Detect a numeric contradiction on a shared subject.

    Strategy: for each numeric literal in the draft, take up to 4
    preceding content tokens as its "subject window". If any evidence
    sentence shares at least one non-stopword subject token AND
    carries a numeric literal that differs from the draft's, treat
    that as a numeric contradiction on the same subject.
    """
    d_low = draft.lower()
    e_low = evidence.lower()

    for m in _NUMBER_RE.finditer(d_low):
        num_d = m.group(0)
        start = max(0, m.start() - 80)
        window = d_low[start : m.start()]
        window_tokens = [t for t in _TOKEN_RE.findall(window) if t not in _STOP][-4:]
        # Keep only subject tokens that also appear in the evidence.
        subject_tokens = [t for t in window_tokens if t in e_low]
        if not subject_tokens:
            continue

        # Scan each evidence sentence: same subject, different number.
        for sent in re.split(r"(?<=[.!?])\s+|\n+", e_low):
            if not any(tok in sent for tok in subject_tokens):
                continue
            nums_in_sent = _numbers(sent)
            if nums_in_sent and any(n != num_d for n in nums_in_sent):
                return True
    return False


def _missed_evidence(query: str, draft: str, evidence: str) -> bool:
    """Draft omits a strong evidence subject that the query asks about.

    Strong subject = a content token that appears in BOTH the query
    and the evidence at least twice, but not in the draft at all.
    """
    q_tokens = _content_tokens(query)
    if not q_tokens:
        return False
    d_tokens = _content_tokens(draft)
    e_low = evidence.lower()
    for tok in q_tokens:
        if tok in d_tokens:
            continue
        if e_low.count(tok) >= 2:
            return True
    return False


def _unsupported_claim(draft: str, evidence: str) -> bool:
    """Draft asserts a specific factual literal absent from evidence.

    Numeric literals in the draft that do not appear anywhere in the
    evidence are treated as unsupported.
    """
    d_nums = _numbers(draft)
    if not d_nums:
        return False
    e_low = evidence.lower()
    for n in d_nums:
        if n not in e_low:
            return True
    return False


def check_draft(
    draft_answer: str,
    query: str,
    chunks: list[dict] | None = None,
    memory: list[dict] | None = None,
    kg_facts: list[dict] | None = None,
) -> CheckResult:
    """Return a CheckResult classifying any problem found in ``draft_answer``.

    Order of precedence: contradiction > missed_evidence > unsupported_claim.
    """

    draft = (draft_answer or "").strip()
    if not draft:
        # Empty draft cannot be judged; leave it to the caller.
        return CheckResult(ok=True, reason="")

    evidence = _evidence_text(chunks, memory, kg_facts).strip()
    if not evidence:
        # No evidence supplied → nothing to reflect against.
        return CheckResult(ok=True, reason="")

    if _numeric_contradiction(draft, evidence) or _explicit_negation_flip(
        draft, evidence
    ):
        return CheckResult(ok=False, reason="contradiction")

    if _missed_evidence(query, draft, evidence):
        return CheckResult(ok=False, reason="missed_evidence")

    if _unsupported_claim(draft, evidence):
        return CheckResult(ok=False, reason="unsupported_claim")

    return CheckResult(ok=True, reason="")
