"""Contracts exchanged by the admission policy.

Tiny frozen dataclasses. Callers pass in a ``MemoryCandidate``, the
policy returns an ``AdmissionDecision``. Neither depends on the memory
backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MemoryCandidate:
    """A single write candidate."""

    text: str
    kind: str = "fact"
    importance_hint: float = 1.0
    session: str = "default"
    source: str = "unspecified"
    now: float = 0.0


@dataclass(frozen=True)
class AdmissionDecision:
    """The policy's verdict for a single candidate.

    Stable ``reason`` values:

        "admitted_high_importance"       -- stored: strong signal
        "admitted_novel"                 -- stored: nothing similar
        "admitted_preference_change"     -- stored: replaces prior
        "admitted_correction"            -- stored: fixes prior
        "admitted_repeated_important"    -- stored: repeat but valuable
        "admitted_stale_duplicate"       -- stored: duplicate but old
        "rejected_exact_duplicate"       -- rejected: identical exists
        "rejected_near_duplicate_recent" -- rejected: similar and fresh
        "rejected_low_value"             -- rejected: filler
        "rejected_empty"                 -- rejected: empty text
    """

    admitted: bool
    reason: str
    importance_score: float
    novelty_score: float
    duplicate_score: float

    def to_dict(self) -> dict:
        return {
            "admitted": self.admitted,
            "reason": self.reason,
            "importance_score": round(self.importance_score, 4),
            "novelty_score": round(self.novelty_score, 4),
            "duplicate_score": round(self.duplicate_score, 4),
        }


@dataclass(frozen=True)
class HistoryEntry:
    """A minimal view of an existing memory used for duplicate scoring."""

    text: str
    kind: str = "fact"
    ts: float = 0.0
    importance: float = 1.0


@dataclass(frozen=True)
class PolicyConfig:
    """Tunable knobs for the admission policy."""

    importance_admit_threshold: float = 0.60
    exact_duplicate_threshold: float = 0.98
    near_duplicate_threshold: float = 0.85
    recent_duplicate_window_s: float = 24.0 * 60.0 * 60.0  # 24h
    min_content_tokens: int = 2
    min_text_chars: int = 4
    max_history_scan: int = 500

    metadata: dict = field(default_factory=dict)
