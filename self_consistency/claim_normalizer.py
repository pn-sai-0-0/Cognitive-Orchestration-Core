"""Claim-level normalization for COC self-consistency.

This is intentionally a small helper inside the existing self_consistency
package. It turns each attempt answer into a canonical claim fingerprint:
- paraphrases of the same claim normalize to the same key;
- distinct numeric values remain distinct;
- distinct actions on the same subject (pause vs complete) remain distinct;
- no-evidence fallback messages share a special fallback key.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Claim:
    subject: str
    predicate: str
    object_: str
    numbers: tuple[str, ...]
    negation: bool
    raw: str
    key: str
    tokens: frozenset[str] = field(default_factory=frozenset)

    def to_dict(self) -> dict:
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object_,
            "numbers": list(self.numbers),
            "negation": self.negation,
            "key": self.key,
        }


_FALLBACK = (
    "don't have enough information",
    "evidence is too thin",
    "insufficient evidence",
    "best-effort",
)
_BOILERPLATE = re.compile(
    r"^(?:limited-confidence answer:\s*)?"
    r"(?:best supported answer from [a-z0-9_./-]+:\s*)?",
    re.IGNORECASE,
)
_STOP = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "to",
        "in",
        "on",
        "at",
        "for",
        "from",
        "with",
        "by",
        "as",
        "of",
        "when",
        "while",
        "this",
        "that",
        "it",
        "best",
        "supported",
        "answer",
        "limited",
        "confidence",
        "evidence",
        "available",
        "before",
        "after",
        "config",
        "memory",
        "knowledge",
        "graph",
        "help_center",
        "engine",
        "py",
        "generator",
        "inference",
    }
)


def _clean(text: str) -> str:
    value = _BOILERPLATE.sub("", (text or "").lower()).strip()
    return re.sub(r"\s+", " ", value)


def _tokens(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9_]+", text.lower())
        if token not in _STOP and len(token) > 1
    ]


def _subject(text: str, tokens: list[str]) -> str:
    if "top_k" in text or "top k" in text:
        return "top_k"
    if "final_k" in text or "final k" in text:
        return "final_k"
    if "clear_session" in text:
        return "clear_session"
    if "password" in text and "reset" in text:
        return "password_reset"
    if "grounded fallback" in text or "checkpoint" in text:
        return "model_fallback"
    if "goal split" in text or ("goal" in text and "parent" in text):
        return "goal_split"
    if "document" in text and ("search" in text or "find" in text):
        return "document_search"
    return tokens[0] if tokens else "unknown"


def _predicate(text: str) -> str:
    patterns = (
        (r"\b(?:pauses?|paused)\b", "pauses"),
        (r"\b(?:completes?|completed)\b", "completes"),
        (r"\b(?:clears?|clear|removes?|remove)\b", "clears"),
        (r"\b(?:falls? back to|fallback)\b", "uses"),
        (r"\b(?:uses?|via|through|with)\b", "uses"),
        (r"\b(?:invokes?|invoke|runs?|run|find|search)\b", "invokes"),
        (r"\b(?:equals?|is|=)\b", "equals"),
    )
    for pattern, value in patterns:
        if re.search(pattern, text):
            return value
    return "asserts"


def _object(
    text: str, subject: str, predicate: str, numbers: tuple[str, ...], tokens: list[str]
) -> str:
    if numbers:
        return numbers[0]
    if subject == "password_reset":
        return (
            "email_reset_link"
            if "email" in text or "link" in text
            else "password_reset_flow"
        )
    if subject == "model_fallback":
        return (
            "grounded_fallback" if "grounded fallback" in text else "model_unavailable"
        )
    if subject == "clear_session":
        return "session_history"
    if subject == "document_search":
        return "find_document" if "find" in text else "document_search"
    if subject == "goal_split":
        return "parent"
    candidates = [token for token in tokens if token not in {subject, predicate}]
    return candidates[-1] if candidates else "unknown"


def normalize_claim(answer: str) -> Claim:
    raw = (answer or "").strip()
    lower = raw.lower()
    if any(marker in lower for marker in _FALLBACK):
        return Claim(
            "",
            "fallback",
            "insufficient_evidence",
            (),
            True,
            raw,
            "fallback::insufficient_evidence",
            frozenset(),
        )

    core = _clean(raw)
    tokens = _tokens(core)
    numbers = tuple(re.findall(r"\b\d+(?:\.\d+)?\b", core))
    subject = _subject(core, tokens)
    predicate = _predicate(core)
    object_ = _object(core, subject, predicate, numbers, tokens)
    value = "|".join(numbers) if numbers else object_
    key = f"{subject}::{predicate}::{value}"
    return Claim(
        subject, predicate, object_, numbers, False, raw, key, frozenset(tokens)
    )


def claim_equivalence(left: Claim, right: Claim) -> float:
    if left.negation and right.negation:
        return 1.0
    if left.negation != right.negation:
        return 0.0
    if left.numbers and right.numbers and left.numbers != right.numbers:
        return 0.0
    if left.key == right.key:
        return 1.0
    if left.subject == right.subject and left.numbers == right.numbers and left.numbers:
        return 1.0
    if left.subject == right.subject and left.predicate != right.predicate:
        return 0.0
    union = left.tokens | right.tokens
    overlap = len(left.tokens & right.tokens) / max(len(union), 1)
    subject_bonus = 0.30 if left.subject == right.subject else 0.0
    predicate_bonus = 0.20 if left.predicate == right.predicate else 0.0
    return round(0.50 * overlap + subject_bonus + predicate_bonus, 4)
