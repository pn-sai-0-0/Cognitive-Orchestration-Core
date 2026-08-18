"""End-to-end wiring test for the reflection pass on the engine.

The generator, reasoner, memory, and other subsystems are stubbed so
this test verifies exactly ONE thing: that ``Engine.process`` on a
normal (non-goal-command) request invokes reflection, records a
``reflection`` dict on the ``EngineResult``, and replaces the draft
answer with the reasoner-derived revision when the checker detects a
contradiction with the provided evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from engine import Engine


@dataclass
class _StubGeneratorBackend:
    backend: str = "stub"


class _StubGenerator:
    backend = "stub"

    def __init__(self, output: str) -> None:
        self._output = output

    def generate(self, message: str, context: dict) -> str:
        return self._output


class _StubReasoner:
    def __init__(self, revised_answer: str) -> None:
        self._revised = revised_answer
        self.contract_calls = 0
        self.assess_calls = 0

    # Full pipeline call issued by build_context; return an empty-shaped
    # reasoning dict — the test does not care about the reasoning trace.
    def assess(self, **kwargs: Any) -> dict:
        self.assess_calls += 1
        return {
            "type": "factual", "steps": [], "evidence": {"sources": []},
            "verification": {"verified": True, "coverage": 1.0,
                             "issues": [], "gaps": [], "adjustments": []},
            "synthesis": {"plan": "", "trace_steps": []},
            "confidence": 0.9,
            "plan": "", "trace": [], "latency_ms": 0.0,
        }

    def assess_contract(self, **kwargs: Any) -> dict:
        self.contract_calls += 1
        return {"answer": self._revised}


def _install_stubs(engine: Engine, *, draft: str, revised: str,
                   chunks: list[dict]) -> _StubReasoner:
    """Skip real init and inject just enough scaffolding to run process()."""
    engine._init_done = True
    engine._generator = _StubGenerator(draft)  # type: ignore[assignment]
    reasoner = _StubReasoner(revised_answer=revised)
    engine._reasoner = reasoner  # type: ignore[assignment]

    # Neutralise everything build_context might reach for; each of these
    # subsystems is optional and guarded by ``if self._X`` in engine.py.
    engine._memory = None      # type: ignore[assignment]
    engine._rag = None         # type: ignore[assignment]
    engine._cag = None         # type: ignore[assignment]
    engine._kg = None          # type: ignore[assignment]
    engine._hybrid = None      # type: ignore[assignment]
    engine._cognition = None   # type: ignore[assignment]
    engine._obs = None         # type: ignore[assignment]
    engine._tools = None       # type: ignore[assignment]
    engine._self_consistency = None  # type: ignore[assignment]

    # Force build_context to see our evidence chunks by monkey-patching it.
    real_build = engine.build_context

    def build_context(message: str, session: str = "default",
                      ip: str = "127.0.0.1",
                      active_doc: str | None = None) -> dict | None:
        return {
            "intent": "chat",
            "message": message,
            "clean_msg": message,
            "chunks": chunks,
            "memory": [],
            "kg": [],
            "reasoning": {},
            "self_consistency": {},
            "cognition": {},
            "tool": None,
        }

    # Preserve for potential debug but override for this test.
    engine.build_context = build_context  # type: ignore[assignment]
    _ = real_build  # keep reference to satisfy linters
    return reasoner


def test_engine_reflection_revises_contradicting_draft() -> None:
    """A draft that contradicts the evidence is revised by reflection."""
    engine = Engine()
    chunks = [{
        "source": "config.py",
        "text": "RETRIEVAL top_k = 8 before reranking is applied.",
    }]
    reasoner = _install_stubs(
        engine,
        draft="The default retrieval top_k is 3 before reranking.",
        revised="The default retrieval top_k is 8 before reranking.",
        chunks=chunks,
    )

    result = engine.process(
        "What is the default retrieval top_k?",
        session="test-reflect",
    )

    # Revised text replaced the draft.
    assert result.text == "The default retrieval top_k is 8 before reranking."
    # EngineResult carries a reflection dict with the expected shape.
    assert result.reflection == {"revised": True, "reason": "contradiction"}
    # Exactly one reasoner revision call — never more.
    assert reasoner.contract_calls == 1


def test_engine_reflection_no_revision_when_draft_is_supported() -> None:
    """A draft consistent with evidence is left untouched."""
    engine = Engine()
    chunks = [{
        "source": "config.py",
        "text": "RETRIEVAL top_k = 8 before reranking is applied.",
    }]
    reasoner = _install_stubs(
        engine,
        draft="The default retrieval top_k is 8 before reranking.",
        revised="should-not-appear",
        chunks=chunks,
    )

    result = engine.process(
        "What is the default retrieval top_k?",
        session="test-reflect-ok",
    )

    assert result.text == "The default retrieval top_k is 8 before reranking."
    assert result.reflection == {"revised": False, "reason": ""}
    assert reasoner.contract_calls == 0
