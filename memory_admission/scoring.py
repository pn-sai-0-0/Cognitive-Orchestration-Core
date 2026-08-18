"""Deterministic scoring helpers for the admission policy.

All functions are pure. They return floats in [0.0, 1.0] wherever a
score is described as a score.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-]*")
_FILLER_TOKENS = frozenset(
    {
        "ok",
        "okay",
        "yes",
        "no",
        "sure",
        "cool",
        "nice",
        "thanks",
        "thank",
        "hi",
        "hello",
        "hey",
        "hmm",
        "uh",
        "um",
        "lol",
        "haha",
        "yeah",
        "yep",
        "nope",
        "meh",
    }
)
_IMPORTANT_MARKERS = (
    "prefer",
    "preference",
    "always",
    "never",
    "must",
    "must not",
    "should",
    "should not",
    "important",
    "note that",
    "remember that",
    "actually",
    "correction:",
    "correct:",
    "instead",
    "not ",
    "no,",
    "wrong",
    "mistake",
    "update:",
    "changed",
    "change my",
    "changed my",
    "goal",
    "deadline",
    "budget",
    "phone",
    "email",
    "address",
    "name is",
    "birthday",
    "birthdate",
    "password",
    "credential",
    "policy",
)
_PREFERENCE_MARKERS = (
    "prefer",
    "preference",
    "like",
    "favourite",
    "favorite",
    "hate",
    "dislike",
    "always use",
    "never use",
    "changed my mind",
    "instead of",
)
_CORRECTION_MARKERS = (
    "actually",
    "correction",
    "correct:",
    "wrong",
    "not ",
    "no,",
    "instead of",
    "update:",
    "was mistaken",
    "meant to say",
)

_PREFERENCE_ATTRIBUTES = frozenset(
    {
        "color",
        "colour",
        "language",
        "tone",
        "style",
        "font",
        "theme",
        "editor",
        "shell",
        "browser",
        "framework",
        "timezone",
        "unit",
        "units",
        "currency",
        "notation",
    }
)


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def similarity(a: str, b: str) -> float:
    a_norm = (a or "").strip().lower()
    b_norm = (b or "").strip().lower()
    if not a_norm and not b_norm:
        return 1.0
    if not a_norm or not b_norm:
        return 0.0
    return float(SequenceMatcher(None, a_norm, b_norm).ratio())


def importance_score(
    text: str, importance_hint: float = 1.0, kind: str = "fact"
) -> float:
    toks = tokenize(text)
    if not toks:
        return 0.0
    hint = max(0.0, min(1.0, float(importance_hint) / 2.0))
    kind_bias = {
        "preference": 0.85,
        "goal": 0.75,
        "learning": 0.70,
        "correction": 0.85,
        "project": 0.55,
        "task": 0.55,
        "fact": 0.50,
        "semantic": 0.50,
        "workspace": 0.50,
        "episodic": 0.35,
    }.get(kind, 0.45)
    low = text.lower()
    marker_hits = sum(1 for m in _IMPORTANT_MARKERS if m in low)
    marker_score = min(1.0, 0.15 * marker_hits)
    length_score = min(1.0, max(0.0, (len(toks) - 3) / 27.0))
    score = 0.35 * hint + 0.25 * kind_bias + 0.20 * marker_score + 0.20 * length_score
    return max(0.0, min(1.0, round(score, 4)))


def duplicate_score(text: str, history_texts: list[str]) -> tuple[float, int]:
    if not history_texts:
        return 0.0, -1
    best = 0.0
    best_idx = -1
    for i, h in enumerate(history_texts):
        s = similarity(text, h)
        if s > best:
            best = s
            best_idx = i
            if best >= 1.0:
                break
    return round(best, 4), best_idx


def novelty_from_duplicate(duplicate: float) -> float:
    return round(max(0.0, min(1.0, 1.0 - duplicate)), 4)


def is_low_value_filler(text: str, min_tokens: int = 2, min_chars: int = 4) -> bool:
    stripped = (text or "").strip()
    if len(stripped) < min_chars:
        return True
    toks = tokenize(stripped)
    if len(toks) < min_tokens:
        return True
    non_filler = [t for t in toks if t not in _FILLER_TOKENS]
    return len(non_filler) == 0


def is_preference_replacement(text: str, history_text: str) -> bool:
    t_toks = set(tokenize(text))
    h_toks = set(tokenize(history_text))
    shared_attr = t_toks & h_toks & _PREFERENCE_ATTRIBUTES
    if not shared_attr:
        return False
    low = text.lower()
    if (
        not any(m in low for m in _PREFERENCE_MARKERS)
        and " is " not in low
        and " are " not in low
    ):
        return False
    sim = similarity(text, history_text)
    return 0.35 <= sim < 0.98


def is_correction(text: str) -> bool:
    low = " " + (text or "").lower() + " "
    return any(m in low for m in _CORRECTION_MARKERS)


def is_preference_signal(text: str) -> bool:
    low = (text or "").lower()
    return any(m in low for m in _PREFERENCE_MARKERS)
