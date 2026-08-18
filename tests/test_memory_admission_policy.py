"""Unit tests for the Memory Admission Policy."""

from __future__ import annotations

import time

from memory_admission import AdmissionPolicy, MemoryCandidate
from memory_admission.contracts import HistoryEntry, PolicyConfig


def _hist(text: str, kind: str = "fact",
          ts: float | None = None, importance: float = 1.0) -> HistoryEntry:
    return HistoryEntry(
        text=text, kind=kind,
        ts=ts if ts is not None else time.time(),
        importance=importance,
    )


def test_new_important_memory_is_admitted() -> None:
    policy = AdmissionPolicy()
    candidate = MemoryCandidate(
        text="User prefers dark mode in the editor and never wants light mode.",
        kind="preference",
        importance_hint=1.3,
        session="s",
        source="remember_intent",
    )
    decision = policy.decide(candidate, history=[])
    assert decision.admitted is True
    assert decision.reason in {"admitted_high_importance", "admitted_novel"}
    assert 0.0 <= decision.importance_score <= 1.0
    assert decision.duplicate_score == 0.0
    assert decision.novelty_score == 1.0


def test_exact_duplicate_is_rejected() -> None:
    policy = AdmissionPolicy()
    prior = "User prefers dark mode in the editor."
    candidate = MemoryCandidate(text=prior, kind="preference", importance_hint=1.0)
    decision = policy.decide(candidate, history=[_hist(prior)])
    assert decision.admitted is False
    assert decision.reason == "rejected_exact_duplicate"
    assert decision.duplicate_score >= 0.98


def test_near_duplicate_recent_is_rejected() -> None:
    policy = AdmissionPolicy()
    prior = "The tests were green on the release branch last night."
    candidate = MemoryCandidate(
        text="The tests were green on the release branch yesterday.",
        kind="fact",
        importance_hint=0.7,
    )
    decision = policy.decide(candidate, history=[_hist(prior)])
    assert decision.admitted is False
    assert decision.reason == "rejected_near_duplicate_recent"
    assert 0.85 <= decision.duplicate_score < 0.98


def test_old_duplicate_after_sufficient_time_is_admitted_if_important() -> None:
    cfg = PolicyConfig(recent_duplicate_window_s=1.0)
    policy = AdmissionPolicy(cfg)
    prior = "Deadline is 2026-09-01 for the quarterly release."
    old_ts = time.time() - 10.0
    candidate = MemoryCandidate(
        text="Deadline is 2026-09-01 for the quarterly release please.",
        kind="goal",
        importance_hint=1.4,
        now=time.time(),
    )
    decision = policy.decide(
        candidate, history=[_hist(prior, ts=old_ts, importance=1.4)],
    )
    assert decision.admitted is True
    assert decision.reason == "admitted_stale_duplicate"


def test_old_duplicate_low_importance_stays_rejected() -> None:
    cfg = PolicyConfig(recent_duplicate_window_s=1.0)
    policy = AdmissionPolicy(cfg)
    prior = "some very average note about the meeting."
    old_ts = time.time() - 10.0
    candidate = MemoryCandidate(
        text="some very average note about the meeting today.",
        kind="episodic",
        importance_hint=0.2,
        now=time.time(),
    )
    decision = policy.decide(
        candidate, history=[_hist(prior, ts=old_ts, importance=0.2)],
    )
    assert decision.admitted is False
    assert decision.reason == "rejected_near_duplicate_recent"


def test_low_value_conversational_filler_is_rejected() -> None:
    policy = AdmissionPolicy()
    for filler in ["ok", "yeah sure", "hi", "hmm", "thanks"]:
        candidate = MemoryCandidate(text=filler, kind="episodic")
        decision = policy.decide(candidate, history=[])
        assert decision.admitted is False, filler
        assert decision.reason == "rejected_low_value", filler


def test_preference_change_blue_to_green_is_admitted() -> None:
    policy = AdmissionPolicy()
    prior = "My favourite color is blue."
    candidate = MemoryCandidate(
        text="My favourite color is green.",
        kind="preference",
        importance_hint=1.2,
    )
    decision = policy.decide(candidate, history=[_hist(prior)])
    assert decision.admitted is True
    assert decision.reason == "admitted_preference_change"


def test_critical_correction_overrides_duplicate_rejection() -> None:
    policy = AdmissionPolicy()
    prior = "My phone number is 555-0100."
    candidate = MemoryCandidate(
        text="Actually, my phone number is 555-0101, not 555-0100.",
        kind="correction",
        importance_hint=1.4,
    )
    decision = policy.decide(candidate, history=[_hist(prior)])
    assert decision.admitted is True
    assert decision.reason == "admitted_correction"


def test_repeated_important_fact_is_admitted_as_reinforcement() -> None:
    policy = AdmissionPolicy()
    prior = "The project deadline is next Friday."
    candidate = MemoryCandidate(
        text="Remember that the deadline for the project falls on Friday next week.",
        kind="goal",
        importance_hint=1.2,
    )
    decision = policy.decide(candidate, history=[_hist(prior)])
    assert decision.admitted is True
    assert decision.reason in {
        "admitted_repeated_important",
        "admitted_high_importance",
    }
    assert decision.duplicate_score < 0.85


def test_empty_candidate_is_rejected() -> None:
    policy = AdmissionPolicy()
    decision = policy.decide(MemoryCandidate(text=""), history=None)
    assert decision.admitted is False
    assert decision.reason == "rejected_empty"


def test_decision_to_dict_shape() -> None:
    policy = AdmissionPolicy()
    decision = policy.decide(
        MemoryCandidate(text="User prefers vim over emacs.",
                        kind="preference",
                        importance_hint=1.2),
        history=[],
    )
    d = decision.to_dict()
    assert set(d.keys()) == {
        "admitted", "reason",
        "importance_score", "novelty_score", "duplicate_score",
    }
    assert isinstance(d["admitted"], bool)
    assert isinstance(d["reason"], str)
    for key in ("importance_score", "novelty_score", "duplicate_score"):
        v = d[key]
        assert isinstance(v, float)
        assert 0.0 <= v <= 1.0
