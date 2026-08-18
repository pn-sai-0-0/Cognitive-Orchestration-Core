"""Engine wiring tests for Adaptation v1.

Adaptation consumes replay output only and attaches deterministic metadata to
``result.trace["adaptation"]`` without mutating existing subsystems.
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

    def assess_contract(self, **kwargs: Any) -> dict:
        _ = kwargs
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


def test_normal_adaptation_flow() -> None:
    memory = _StubMemory()
    engine = _install_process_engine(
        draft="Short grounded reply.",
        memory=memory,
        admission_policy=AdmissionPolicy(PolicyConfig()),
    )

    result = engine.process("Explain the setting.", session="a1")
    adaptation = result.trace["adaptation"]

    assert adaptation == {
        "source": "replay",
        "read_only": True,
        "deterministic": True,
        "valid": True,
        "mode": "process",
        "classification": "stable",
        "signals": {
            "short_circuit": False,
            "reflection_revised": False,
            "learning_generated": 0,
            "memory_admission_rejections": 0,
            "warnings": 0,
            "errors": 0,
        },
    }


def test_replay_integration_uses_reflection_and_learning_signals() -> None:
    memory = _StubMemory()
    engine = _install_process_engine(
        draft="The default retrieval top_k is 3.",
        revised_answer="The default retrieval top_k is 8.",
        reflection_chunks=[{"source": "config.py", "text": "top_k is 8."}],
        memory=memory,
        self_consistency=_sc_payload("The default retrieval top_k is 8", 0.91),
        admission_policy=AdmissionPolicy(PolicyConfig()),
    )

    result = engine.process("What is the default retrieval top_k?", session="a2")
    replay = result.trace["replay"]
    adaptation = result.trace["adaptation"]

    assert replay["stages"]["reflection"]["revised"] is True
    assert adaptation["classification"] == "reflection_revision"
    assert adaptation["signals"]["reflection_revised"] is True
    assert adaptation["signals"]["learning_generated"] == 2


def test_deterministic_output_for_identical_replay() -> None:
    engine = Engine()
    replay = {
        "mode": "process",
        "read_only": True,
        "stages": {
            "input": {"short_circuit": False},
            "planning": {"executed": True},
            "reflection": {"revised": False},
            "learning": {"generated": [{"source": "x"}]},
            "memory_admission": {"events": []},
            "output": {"short_circuit": False},
        },
        "warnings": [],
        "errors": [],
    }

    assert engine._build_adaptation_trace(replay) == engine._build_adaptation_trace(
        replay
    )


def test_repeated_identical_runs_match() -> None:
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

    adaptation_one = engine_one.process(
        "Which planet is eighth from the Sun?", session="a3"
    ).trace["adaptation"]
    adaptation_two = engine_two.process(
        "Which planet is eighth from the Sun?", session="a3"
    ).trace["adaptation"]

    assert adaptation_one == adaptation_two


def test_empty_replay_returns_unavailable_metadata() -> None:
    engine = Engine()

    assert engine._build_adaptation_trace({}) == {
        "source": "replay",
        "read_only": True,
        "deterministic": True,
        "valid": False,
        "mode": "none",
        "classification": "unavailable",
        "signals": {
            "short_circuit": False,
            "reflection_revised": False,
            "learning_generated": 0,
            "memory_admission_rejections": 0,
            "warnings": 0,
            "errors": 0,
        },
        "reason": "missing_replay",
    }


def test_malformed_replay_returns_invalid_metadata() -> None:
    engine = Engine()
    malformed = {"mode": "process", "stages": [], "warnings": [], "errors": []}

    assert engine._build_adaptation_trace(malformed) == {
        "source": "replay",
        "read_only": True,
        "deterministic": True,
        "valid": False,
        "mode": "none",
        "classification": "unavailable",
        "signals": {
            "short_circuit": False,
            "reflection_revised": False,
            "learning_generated": 0,
            "memory_admission_rejections": 0,
            "warnings": 0,
            "errors": 0,
        },
        "reason": "malformed_replay",
    }


def test_engine_process_stream_attaches_adaptation() -> None:
    memory = _StubMemory()
    engine = _install_stream_engine(
        output="Streaming answer.",
        memory=memory,
        self_consistency=_sc_payload("Saturn has prominent rings", 0.95),
        admission_policy=AdmissionPolicy(PolicyConfig()),
    )

    events = list(engine.process_stream("Tell me about Saturn", session="a4"))
    adaptation = events[-1]["trace"]["adaptation"]

    assert events[-1]["done"] is True
    assert adaptation["mode"] == "stream"
    assert adaptation["classification"] == "learning_signal"
    assert adaptation["signals"]["learning_generated"] == 1


def test_engine_wiring_preserves_no_mutation() -> None:
    memory = _StubMemory()
    engine = _install_process_engine(
        draft="Stable answer.",
        memory=memory,
        self_consistency=_sc_payload("Mars is the fourth planet from the Sun", 0.91),
        admission_policy=_RejectAllPolicy(),
    )

    result = engine.process("Which planet is fourth from the Sun?", session="a5")
    adaptation = result.trace["adaptation"]

    assert adaptation["classification"] == "admission_rejection"
    assert memory.learning_calls == []
    assert memory.messages == [
        ("a5", "user", "Which planet is fourth from the Sun?"),
        ("a5", "assistant", "Stable answer."),
    ]
