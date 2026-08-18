"""Tests for reflection.reflector.

These use a tiny stub reasoner so the reflector logic is exercised
without dragging in the full reasoning pipeline.
"""

from __future__ import annotations

from reflection.reflector import Reflector


class _StubReasoner:
    def __init__(self, answer: str, raise_exc: bool = False) -> None:
        self._answer = answer
        self._raise = raise_exc
        self.calls: list[dict] = []

    def assess_contract(self, **kwargs: object) -> dict:
        self.calls.append(dict(kwargs))
        if self._raise:
            raise RuntimeError("simulated reasoner failure")
        return {"answer": self._answer}


def _chunks(*texts: str) -> list[dict]:
    return [{"source": f"doc{i}", "text": t} for i, t in enumerate(texts)]


def test_reflector_no_revision_when_draft_is_supported() -> None:
    reasoner = _StubReasoner(answer="should-not-be-used")
    reflector = Reflector(reasoner)
    draft = "The default retrieval top_k is 8 before reranking."
    text, report = reflector.reflect(
        draft_answer=draft,
        query="What is the default retrieval top_k?",
        chunks=_chunks("RETRIEVAL top_k = 8 before reranking is applied."),
    )
    assert text == draft
    assert report.revised is False
    assert report.reason == ""
    assert reasoner.calls == []


def test_reflector_revises_once_on_contradiction() -> None:
    good_answer = "The default retrieval top_k is 8 before reranking."
    reasoner = _StubReasoner(answer=good_answer)
    reflector = Reflector(reasoner)
    bad_draft = "The default retrieval top_k is 3 before reranking."
    text, report = reflector.reflect(
        draft_answer=bad_draft,
        query="What is the default retrieval top_k?",
        chunks=_chunks("RETRIEVAL top_k = 8 before reranking is applied."),
    )
    assert text == good_answer
    assert report.revised is True
    assert report.reason == "contradiction"
    # Must be called exactly once — reflection never loops.
    assert len(reasoner.calls) == 1


def test_reflector_marks_revision_failed_when_reasoner_returns_empty() -> None:
    reasoner = _StubReasoner(answer="")
    reflector = Reflector(reasoner)
    bad_draft = "The default retrieval top_k is 3 before reranking."
    text, report = reflector.reflect(
        draft_answer=bad_draft,
        query="What is the default retrieval top_k?",
        chunks=_chunks("RETRIEVAL top_k = 8 before reranking is applied."),
    )
    assert text == bad_draft
    assert report.revised is False
    assert report.reason == "revision_failed"


def test_reflector_marks_revision_failed_when_reasoner_raises() -> None:
    reasoner = _StubReasoner(answer="unused", raise_exc=True)
    reflector = Reflector(reasoner)
    bad_draft = "The default retrieval top_k is 3 before reranking."
    text, report = reflector.reflect(
        draft_answer=bad_draft,
        query="What is the default retrieval top_k?",
        chunks=_chunks("RETRIEVAL top_k = 8 before reranking is applied."),
    )
    assert text == bad_draft
    assert report.revised is False
    assert report.reason == "revision_failed"
