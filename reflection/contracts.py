"""Reflection contracts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReflectionReport:
    """Outcome of a single reflection pass.

    Attributes:
        revised: True iff the reflector revised the draft answer.
        reason:  Short human-readable label for why revision happened,
                 or "" when ``revised`` is False. Stable set of values:
                 ""                          — no revision needed
                 "unsupported_claim"         — draft claim not backed by evidence
                 "contradiction"             — draft contradicts the evidence
                 "missed_evidence"           — draft ignored strong evidence
                 "revision_failed"           — a revision was attempted but
                                               produced no usable text; the
                                               original draft is kept.
    """

    revised: bool
    reason: str

    def to_dict(self) -> dict:
        return {"revised": self.revised, "reason": self.reason}
