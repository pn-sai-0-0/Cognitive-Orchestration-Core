"""The admission policy itself.

Deterministic and side-effect free. The CALLER performs the write when
``admitted`` is True.
"""

from __future__ import annotations

import time
from collections.abc import Iterable

from memory_admission.contracts import (
    AdmissionDecision,
    HistoryEntry,
    MemoryCandidate,
    PolicyConfig,
)
from memory_admission.scoring import (
    duplicate_score,
    importance_score,
    is_correction,
    is_low_value_filler,
    is_preference_replacement,
    is_preference_signal,
    novelty_from_duplicate,
)


class AdmissionPolicy:
    """Gate memory writes."""

    def __init__(self, config: PolicyConfig | None = None) -> None:
        self._cfg = config or PolicyConfig()

    @property
    def config(self) -> PolicyConfig:
        return self._cfg

    def decide(
        self, candidate: MemoryCandidate, history: Iterable[HistoryEntry] | None = None
    ) -> AdmissionDecision:
        text = (candidate.text or "").strip()
        if not text:
            return AdmissionDecision(False, "rejected_empty", 0.0, 0.0, 0.0)

        entries: list[HistoryEntry] = []
        if history is not None:
            for i, e in enumerate(history):
                if i >= self._cfg.max_history_scan:
                    break
                entries.append(e)
        history_texts = [e.text for e in entries]

        dup, dup_idx = duplicate_score(text, history_texts)
        nov = novelty_from_duplicate(dup)
        best = entries[dup_idx] if dup_idx >= 0 else None

        imp = importance_score(
            text,
            importance_hint=candidate.importance_hint,
            kind=candidate.kind,
        )

        # Low-value conversational filler is rejected first.
        if is_low_value_filler(
            text,
            min_tokens=self._cfg.min_content_tokens,
            min_chars=self._cfg.min_text_chars,
        ):
            return AdmissionDecision(False, "rejected_low_value", imp, nov, dup)

        # Exact-duplicate branch (with correction / preference-change override).
        if dup >= self._cfg.exact_duplicate_threshold and best is not None:
            if is_correction(text) or is_preference_replacement(text, best.text):
                reason = (
                    "admitted_correction"
                    if is_correction(text)
                    else "admitted_preference_change"
                )
                return AdmissionDecision(True, reason, imp, nov, dup)
            return AdmissionDecision(False, "rejected_exact_duplicate", imp, nov, dup)

        # Explicit correction (below the exact-dup ceiling) beats
        # every remaining duplicate/importance branch: a correction
        # is by definition new information.
        if is_correction(text):
            return AdmissionDecision(True, "admitted_correction", imp, nov, dup)

        # Near-duplicate branch.
        if dup >= self._cfg.near_duplicate_threshold and best is not None:
            if is_preference_replacement(text, best.text):
                return AdmissionDecision(
                    True, "admitted_preference_change", imp, nov, dup
                )
            now = candidate.now if candidate.now > 0.0 else time.time()
            age = max(0.0, now - float(best.ts or 0.0))
            if age < self._cfg.recent_duplicate_window_s:
                return AdmissionDecision(
                    False, "rejected_near_duplicate_recent", imp, nov, dup
                )
            # Stale near-duplicate: admit when the candidate carries
            # a strong signal -- either moderate importance OR a
            # strong-signal kind (goal/preference/correction/learning).
            strong_kind = candidate.kind in {
                "goal",
                "preference",
                "correction",
                "learning",
            }
            if imp >= (self._cfg.importance_admit_threshold * 0.66) or strong_kind:
                return AdmissionDecision(
                    True, "admitted_stale_duplicate", imp, nov, dup
                )
            return AdmissionDecision(
                False, "rejected_near_duplicate_recent", imp, nov, dup
            )

        # Not a duplicate.
        if imp >= self._cfg.importance_admit_threshold:
            reason = "admitted_high_importance"
            if is_preference_signal(text) and entries:
                reason = "admitted_preference_change"
            return AdmissionDecision(True, reason, imp, nov, dup)

        if dup == 0.0:
            return AdmissionDecision(True, "admitted_novel", imp, nov, dup)

        if dup < self._cfg.near_duplicate_threshold and imp >= (
            self._cfg.importance_admit_threshold * 0.66
        ):
            return AdmissionDecision(True, "admitted_repeated_important", imp, nov, dup)

        return AdmissionDecision(False, "rejected_low_value", imp, nov, dup)
