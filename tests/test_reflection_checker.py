"""Tests for reflection.checker (deterministic, no reasoner)."""

from __future__ import annotations

from reflection.checker import check_draft


def _chunks(*texts: str) -> list[dict]:
    return [{"source": f"doc{i}", "text": t} for i, t in enumerate(texts)]


def test_checker_passes_when_draft_is_supported() -> None:
    query = "What is the default retrieval top_k?"
    evidence = _chunks("RETRIEVAL top_k = 8 before reranking is applied.")
    draft = "The default retrieval top_k is 8 before reranking."
    result = check_draft(draft, query, chunks=evidence)
    assert result.ok is True
    assert result.reason == ""


def test_checker_flags_numeric_contradiction() -> None:
    query = "What is the default retrieval top_k?"
    evidence = _chunks("RETRIEVAL top_k = 8 before reranking is applied.")
    draft = "The default retrieval top_k is 3 before reranking."
    result = check_draft(draft, query, chunks=evidence)
    assert result.ok is False
    assert result.reason == "contradiction"


def test_checker_flags_missed_evidence() -> None:
    query = "How does splitting a goal affect the parent goal?"
    evidence = _chunks(
        "Splitting a goal pauses the parent goal automatically.",
        "The parent goal is paused when a goal is split into children.",
    )
    draft = "Children get created."
    result = check_draft(draft, query, chunks=evidence)
    assert result.ok is False
    assert result.reason == "missed_evidence"


def test_checker_flags_unsupported_numeric_claim() -> None:
    query = "How many retries does the client attempt?"
    evidence = _chunks(
        "The client attempts a retry on transient failures before giving up.")
    draft = "The client attempts 7 retries before giving up."
    result = check_draft(draft, query, chunks=evidence)
    assert result.ok is False
    assert result.reason == "unsupported_claim"


def test_checker_is_silent_without_evidence() -> None:
    result = check_draft("Some draft answer.", "any question", chunks=[])
    assert result.ok is True
    assert result.reason == ""

# -- Negation-flip generalisation (was: only "is"; now: all four copulas) --

def test_negation_flip_detected_for_are_plural() -> None:
    evidence = _chunks("Guardrails are applied to every request.")
    draft = "Guardrails are not applied to every request."
    result = check_draft(draft, "Are guardrails applied?", chunks=evidence)
    assert result.ok is False
    assert result.reason == "contradiction"


def test_negation_flip_detected_for_arent_contraction() -> None:
    evidence = _chunks("Guardrails are applied to every request.")
    draft = "Guardrails aren't applied to every request."
    result = check_draft(draft, "Are guardrails applied?", chunks=evidence)
    assert result.ok is False
    assert result.reason == "contradiction"


def test_negation_flip_detected_for_was_past_singular() -> None:
    evidence = _chunks("The migration was successful last night.")
    draft = "The migration was not successful last night."
    result = check_draft(draft, "Was the migration successful?", chunks=evidence)
    assert result.ok is False
    assert result.reason == "contradiction"


def test_negation_flip_detected_for_wasnt_contraction() -> None:
    evidence = _chunks("The migration was successful last night.")
    draft = "The migration wasn't successful last night."
    result = check_draft(draft, "Was the migration successful?", chunks=evidence)
    assert result.ok is False
    assert result.reason == "contradiction"


def test_negation_flip_detected_for_were_past_plural() -> None:
    evidence = _chunks("The tests were green on the release branch.")
    draft = "The tests were not green on the release branch."
    result = check_draft(draft, "Were the tests green?", chunks=evidence)
    assert result.ok is False
    assert result.reason == "contradiction"


def test_negation_flip_detected_for_werent_contraction() -> None:
    evidence = _chunks("The tests were green on the release branch.")
    draft = "The tests weren't green on the release branch."
    result = check_draft(draft, "Were the tests green?", chunks=evidence)
    assert result.ok is False
    assert result.reason == "contradiction"


def test_negation_flip_detected_in_the_reverse_direction() -> None:
    evidence = _chunks("Guardrails are not applied to every request.")
    draft = "Guardrails are applied to every request."
    result = check_draft(draft, "Are guardrails applied?", chunks=evidence)
    assert result.ok is False
    assert result.reason == "contradiction"


def test_negation_never_form_detected() -> None:
    evidence = _chunks("Guardrails are applied to every request.")
    draft = "Guardrails are never applied to every request."
    result = check_draft(draft, "Are guardrails applied?", chunks=evidence)
    assert result.ok is False
    assert result.reason == "contradiction"
