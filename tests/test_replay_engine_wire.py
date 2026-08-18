"""Replay v1 engine wiring tests.

Replay is a deterministic, read-only execution trace attached only to
``result.trace["replay"]``. These tests verify that replay metadata records what
already happened, without mutating the underlying subsystems.
"""

from __future__ import annotations

from typing import Any

from engine import Engine
from goal.store import GoalStore
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
        _ = message, context
        return self._output

    def generate_stream(self, message: str, context: dict):
        _ = message, context
        midpoint = max(1, len(self._output) // 2)
        yield self._output[:midpoint]
        yield self._output[midpoint:]
        yield ""


class _ReflectionReasoner:
    def __init__(self, revised_answer: str) -> None:
        self._revised = revised_answer
        self.contract_calls = 0

    def assess_contract(self, **kwargs: Any) -> dict:
        _ = kwargs
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
        "citations": [],
    }


def _sc_payload(answer: str, confidence: float, warning_reason: str = "") -> dict:
    return {
        "agreement": True,
        "agreement_type": "strong_agreement",
        "warning_reason": warning_reason,
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


def _normalise_replay(replay: dict) -> dict:
    clone = {
        "mode": replay["mode"],
        "read_only": replay["read_only"],
        "stages": replay["stages"],
        "warnings": replay["warnings"],
        "errors": replay["errors"],
    }
    return clone


def test_normal_conversation_trace() -> None:
    memory = _StubMemory()
    engine = _install_process_engine(
        draft="Short grounded reply.",
        memory=memory,
        admission_policy=AdmissionPolicy(PolicyConfig()),
    )

    result = engine.process("Explain the setting.", session="s1")
    replay = result.trace["replay"]

    assert replay["mode"] == "process"
    assert replay["read_only"] is True
    assert replay["stages"]["input"]["intent"] == "chat"
    assert replay["stages"]["planning"]["executed"] is True
    assert replay["stages"]["reflection"]["executed"] is False
    assert replay["stages"]["learning"]["generated"] == []
    assert replay["stages"]["output"]["short_circuit"] is False
    assert replay["errors"] == []
    assert replay["warnings"] == []


def test_goal_command_short_circuit_trace(tmp_path) -> None:
    engine = Engine()
    engine._init_done = True
    engine._goal_store = GoalStore(tmp_path / "replay-goals.db")
    try:
        result = engine.process("/goal create Keep scope narrow", session="g1")
    finally:
        engine._goal_store.close()

    replay = result.trace["replay"]
    assert result.intent == "goal_command"
    assert replay["stages"]["input"]["intent"] == "goal_command"
    assert replay["stages"]["input"]["short_circuit"] is True
    assert replay["stages"]["planning"]["executed"] is False
    assert replay["stages"]["output"]["short_circuit"] is True
    assert replay["stages"]["learning"]["generated"] == []


def test_reflection_revised_trace() -> None:
    memory = _StubMemory()
    engine = _install_process_engine(
        draft="The default retrieval top_k is 3.",
        revised_answer="The default retrieval top_k is 8.",
        reflection_chunks=[{"source": "config.py", "text": "top_k is 8."}],
        memory=memory,
        admission_policy=AdmissionPolicy(PolicyConfig()),
    )

    result = engine.process("What is the default retrieval top_k?", session="s2")
    replay = result.trace["replay"]

    assert replay["stages"]["reflection"] == {
        "executed": True,
        "revised": True,
        "reason": "contradiction",
        "changed_output": True,
    }
    assert replay["stages"]["learning"]["generated"][0]["source"] == (
        "reflection_learning"
    )


def test_learning_generated_trace() -> None:
    memory = _StubMemory()
    engine = _install_process_engine(
        draft="Stable answer.",
        memory=memory,
        self_consistency=_sc_payload("Mars is the fourth planet from the Sun", 0.91),
        admission_policy=AdmissionPolicy(PolicyConfig()),
    )

    result = engine.process("Which planet is fourth from the Sun?", session="s3")
    replay = result.trace["replay"]

    learning_entry = replay["stages"]["learning"]["generated"][0]
    assert learning_entry["source"] == "self_consistency_learning"
    assert learning_entry["stored"] is True
    assert learning_entry["admitted"] is True
    assert replay["stages"]["memory_admission"]["events"][0]["kind"] == "learning"


def test_memory_admission_rejected_trace() -> None:
    memory = _StubMemory()
    engine = _install_process_engine(
        draft="Stable answer.",
        memory=memory,
        self_consistency=_sc_payload("Mars is the fourth planet from the Sun", 0.91),
        admission_policy=_RejectAllPolicy(),
    )

    result = engine.process("Which planet is fourth from the Sun?", session="s4")
    replay = result.trace["replay"]

    learning_entry = replay["stages"]["learning"]["generated"][0]
    admission_entry = replay["stages"]["memory_admission"]["events"][0]
    assert learning_entry["stored"] is False
    assert learning_entry["reason"] == "rejected_exact_duplicate"
    assert admission_entry["admitted"] is False
    assert admission_entry["reason"] == "rejected_exact_duplicate"


def test_deterministic_repeated_trace_generation() -> None:
    engine_one = _install_process_engine(
        draft="Stable answer.",
        memory=_StubMemory(),
        self_consistency=_sc_payload("Neptune is the eighth planet", 0.92),
        admission_policy=AdmissionPolicy(PolicyConfig()),
    )
    engine_two = _install_process_engine(
        draft="Stable answer.",
        memory=_StubMemory(),
        self_consistency=_sc_payload("Neptune is the eighth planet", 0.92),
        admission_policy=AdmissionPolicy(PolicyConfig()),
    )

    replay_one = engine_one.process(
        "Which planet is eighth from the Sun?", session="s5"
    ).trace["replay"]
    replay_two = engine_two.process(
        "Which planet is eighth from the Sun?", session="s5"
    ).trace["replay"]

    assert _normalise_replay(replay_one) == _normalise_replay(replay_two)


def test_replay_does_not_mutate_existing_subsystems() -> None:
    memory = _StubMemory()
    engine = _install_process_engine(
        draft="Short grounded reply.",
        memory=memory,
        admission_policy=AdmissionPolicy(PolicyConfig()),
    )

    result = engine.process("Explain the setting.", session="s6")

    assert result.trace["replay"]["read_only"] is True
    assert memory.messages == [
        ("s6", "user", "Explain the setting."),
        ("s6", "assistant", "Short grounded reply."),
    ]
    assert memory.learning_calls == []
    assert memory.episodic_calls == []
    assert memory.recall_calls == []
