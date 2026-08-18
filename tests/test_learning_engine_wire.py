"""Engine wiring tests for Learning v1.

Learning is stored ONLY through the existing Memory Admission gate and the
existing learning write path. These tests cover the two allowed triggers:
reflection revisions and strong self-consistency agreement above the configured
confidence threshold.
"""

from __future__ import annotations

from typing import Any

from engine import Engine
from memory_admission import AdmissionPolicy
from memory_admission.contracts import AdmissionDecision, PolicyConfig


class _StubMemory:
    def __init__(self, recall_hits: list[dict] | None = None) -> None:
        self.messages: list[tuple[str, str, str]] = []
        self.learning_calls: list[dict] = []
        self.episodic_calls: list[dict] = []
        self._recall_hits = list(recall_hits or [])
        self.recall_calls: list[tuple[str, int]] = []

    def add_message(self, session: str, role: str, text: str) -> None:
        self.messages.append((session, role, text))

    def remember_learning(self, text: str, importance: float = 1.1) -> int:
        self.learning_calls.append({"text": text, "importance": importance})
        self._recall_hits.append(
            {
                "text": text,
                "kind": "learning",
                "ts": 1e12,
                "importance": importance,
            }
        )
        return len(self.learning_calls)

    def remember_episodic(
        self,
        text: str,
        session: str = "default",
        importance: float = 0.7,
        **kw: object,
    ) -> int:
        self.episodic_calls.append(
            {"text": text, "session": session, "importance": importance, **kw}
        )
        return len(self.episodic_calls)

    def ranked_recall(self, query: str, k: int = 8, **kw: object) -> list[dict]:
        self.recall_calls.append((query, k))
        return list(self._recall_hits)


class _StubGenerator:
    backend = "stub"

    def __init__(self, output: str) -> None:
        self._output = output

    def generate(self, message: str, context: dict) -> str:
        return self._output

    def generate_stream(self, message: str, context: dict):
        midpoint = max(1, len(self._output) // 2)
        yield self._output[:midpoint]
        yield self._output[midpoint:]
        yield ""


class _ReflectionReasoner:
    def __init__(self, revised_answer: str) -> None:
        self._revised = revised_answer
        self.contract_calls = 0

    def assess_contract(self, **kwargs: Any) -> dict:
        self.contract_calls += 1
        return {"answer": self._revised}


class _RejectAllPolicy:
    def decide(self, candidate: Any, history: list[Any]) -> AdmissionDecision:
        _ = candidate, history
        return AdmissionDecision(
            admitted=False,
            reason="rejected_exact_duplicate",
            importance_score=0.0,
            novelty_score=0.0,
            duplicate_score=1.0,
        )


def _base_context(
    *,
    message: str,
    session: str,
    chunks: list[dict] | None = None,
    self_consistency: dict | None = None,
) -> dict:
    return {
        "intent": "chat",
        "message": message,
        "session": session,
        "chunks": list(chunks or []),
        "memory": [],
        "kg": [],
        "reasoning": {},
        "self_consistency": dict(self_consistency or {}),
        "cognition": {},
        "tool": None,
        "retrieval_mode": "none",
        "retrieval_hops": 1,
        "retrieval_score": 0.0,
        "latency_build_ms": 0.0,
    }


def _sc_payload(answer: str, confidence: float) -> dict:
    return {
        "agreement": True,
        "agreement_type": "strong_agreement",
        "warning_reason": "",
        "confidence": confidence,
        "consensus": {
            "attempt_id": "attempt-1",
            "strategy": "baseline",
            "answer": answer,
            "confidence": confidence,
            "trace": [],
            "citations": [],
            "tool_usage": [],
            "warnings": [],
            "evidence_strength": 0.9,
            "claim": {},
        },
        "attempts": [],
        "minority": [],
        "metrics": {},
        "query": answer,
        "superficial": False,
    }


def _install_process_engine(
    *,
    draft: str,
    memory: _StubMemory,
    reflection_chunks: list[dict] | None = None,
    revised_answer: str = "",
    self_consistency: dict | None = None,
    admission_policy: Any | None = None,
) -> Engine:
    engine = Engine()
    engine._init_done = True
    engine._memory = memory  # type: ignore[assignment]
    engine._generator = _StubGenerator(draft)  # type: ignore[assignment]
    engine._reasoner = (
        _ReflectionReasoner(revised_answer) if revised_answer else None
    )  # type: ignore[assignment]
    engine._admission_policy = admission_policy  # type: ignore[assignment]
    engine._goal_store = None  # type: ignore[assignment]
    engine._rag = None  # type: ignore[assignment]
    engine._cag = None  # type: ignore[assignment]
    engine._kg = None  # type: ignore[assignment]
    engine._hybrid = None  # type: ignore[assignment]
    engine._cognition = None  # type: ignore[assignment]
    engine._obs = None  # type: ignore[assignment]
    engine._tools = None  # type: ignore[assignment]
    engine._self_consistency = None  # type: ignore[assignment]

    def build_context(
        message: str,
        session: str = "default",
        ip: str = "127.0.0.1",
        active_doc: str | None = None,
    ) -> dict:
        _ = ip, active_doc
        return _base_context(
            message=message,
            session=session,
            chunks=reflection_chunks,
            self_consistency=self_consistency,
        )

    engine.build_context = build_context  # type: ignore[assignment]
    return engine


def _install_stream_engine(
    *,
    output: str,
    memory: _StubMemory,
    self_consistency: dict,
    admission_policy: Any | None = None,
) -> Engine:
    engine = Engine()
    engine._init_done = True
    engine._memory = memory  # type: ignore[assignment]
    engine._generator = _StubGenerator(output)  # type: ignore[assignment]
    engine._reasoner = object()  # type: ignore[assignment]
    engine._admission_policy = admission_policy  # type: ignore[assignment]
    engine._goal_store = None  # type: ignore[assignment]
    engine._rag = None  # type: ignore[assignment]
    engine._cag = None  # type: ignore[assignment]
    engine._kg = None  # type: ignore[assignment]
    engine._hybrid = None  # type: ignore[assignment]
    engine._cognition = None  # type: ignore[assignment]
    engine._obs = None  # type: ignore[assignment]
    engine._tools = None  # type: ignore[assignment]
    engine._self_consistency = None  # type: ignore[assignment]

    def build_context(
        message: str,
        session: str = "default",
        ip: str = "127.0.0.1",
        active_doc: str | None = None,
    ) -> dict:
        _ = ip, active_doc
        return _base_context(
            message=message,
            session=session,
            self_consistency=self_consistency,
        )

    engine.build_context = build_context  # type: ignore[assignment]
    return engine


def test_reflection_revision_creates_learning_memory_candidate() -> None:
    memory = _StubMemory()
    engine = _install_process_engine(
        draft="The default retrieval top_k is 3.",
        revised_answer="The default retrieval top_k is 8.",
        reflection_chunks=[{"source": "config.py", "text": "top_k is 8."}],
        memory=memory,
        admission_policy=AdmissionPolicy(PolicyConfig()),
    )

    result = engine.process("What is the default retrieval top_k?", session="s1")

    assert result.reflection == {"revised": True, "reason": "contradiction"}
    assert len(memory.learning_calls) == 1
    assert memory.learning_calls[0]["text"] == (
        "Reflection revised answer: changed from "
        "'The default retrieval top_k is 3.' to "
        "'The default retrieval top_k is 8.' because contradiction."
    )


def test_reflection_without_revision_creates_no_learning_memory() -> None:
    memory = _StubMemory()
    engine = _install_process_engine(
        draft="The default retrieval top_k is 8.",
        revised_answer="unused",
        reflection_chunks=[{"source": "config.py", "text": "top_k is 8."}],
        memory=memory,
        admission_policy=AdmissionPolicy(PolicyConfig()),
    )

    result = engine.process("What is the default retrieval top_k?", session="s2")

    assert result.reflection == {"revised": False, "reason": ""}
    assert memory.learning_calls == []


def test_strong_agreement_above_threshold_creates_learning_memory() -> None:
    memory = _StubMemory()
    engine = _install_process_engine(
        draft="Stable answer.",
        memory=memory,
        self_consistency=_sc_payload("Mars is the fourth planet from the Sun", 0.91),
        admission_policy=AdmissionPolicy(PolicyConfig()),
    )

    result = engine.process("Which planet is fourth from the Sun?", session="s3")

    assert result.self_consistency["agreement_type"] == "strong_agreement"
    assert len(memory.learning_calls) == 1
    assert memory.learning_calls[0]["text"] == (
        "Self-consistency confirmed: Mars is the fourth planet from the Sun."
    )


def test_low_confidence_creates_no_learning_memory() -> None:
    memory = _StubMemory()
    engine = _install_process_engine(
        draft="Stable answer.",
        memory=memory,
        self_consistency=_sc_payload("Mars is the fourth planet from the Sun", 0.70),
        admission_policy=AdmissionPolicy(PolicyConfig()),
    )

    engine.process("Which planet is fourth from the Sun?", session="s4")

    assert memory.learning_calls == []


def test_admission_rejection_prevents_storage() -> None:
    memory = _StubMemory()
    engine = _install_process_engine(
        draft="Stable answer.",
        memory=memory,
        self_consistency=_sc_payload("Mars is the fourth planet from the Sun", 0.91),
        admission_policy=_RejectAllPolicy(),
    )

    engine.process("Which planet is fourth from the Sun?", session="s5")

    assert memory.learning_calls == []


def test_engine_process_stream_creates_learning_memory() -> None:
    memory = _StubMemory()
    engine = _install_stream_engine(
        output="Streaming answer.",
        memory=memory,
        self_consistency=_sc_payload("Saturn has prominent rings", 0.95),
        admission_policy=AdmissionPolicy(PolicyConfig()),
    )

    events = list(engine.process_stream("Tell me about Saturn", session="s6"))

    assert events[-1]["done"] is True
    assert len(memory.learning_calls) == 1
    assert memory.learning_calls[0]["text"] == (
        "Self-consistency confirmed: Saturn has prominent rings."
    )


def test_deterministic_repeated_execution_stores_learning_once() -> None:
    memory = _StubMemory()
    engine = _install_process_engine(
        draft="Stable answer.",
        memory=memory,
        self_consistency=_sc_payload("Neptune is the eighth planet", 0.92),
        admission_policy=AdmissionPolicy(PolicyConfig()),
    )

    engine.process("Which planet is eighth from the Sun?", session="s7")
    engine.process("Which planet is eighth from the Sun?", session="s7")

    assert [call["text"] for call in memory.learning_calls] == [
        "Self-consistency confirmed: Neptune is the eighth planet."
    ]
