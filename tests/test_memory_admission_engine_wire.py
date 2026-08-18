"""Engine-level test: the admission gate is actually applied at the two
memory write sites and RETRIEVAL is untouched.
"""

from __future__ import annotations

from engine import Engine


class _StubMemory:
    """Records every call so we can assert what actually got written."""

    def __init__(self, recall_hits: list[dict] | None = None) -> None:
        self.messages: list[tuple[str, str, str]] = []
        self.remember_auto_calls: list[dict] = []
        self.remember_episodic_calls: list[dict] = []
        self._recall_hits = recall_hits or []
        self.recall_calls: list[tuple[str, int]] = []

    def add_message(self, session: str, role: str, text: str) -> None:
        self.messages.append((session, role, text))

    def remember_auto(self, text: str, session: str = "default", **kw: object) -> int:
        self.remember_auto_calls.append({"text": text, "session": session, **kw})
        return 1

    def remember_episodic(self, text: str, session: str = "default",
                          importance: float = 0.7, **kw: object) -> int:
        self.remember_episodic_calls.append(
            {"text": text, "session": session, "importance": importance, **kw}
        )
        return 1

    def ranked_recall(self, query: str, k: int = 8, **kw: object) -> list[dict]:
        self.recall_calls.append((query, k))
        return list(self._recall_hits)


def _install(engine: Engine, *, memory: _StubMemory) -> None:
    engine._init_done = True
    engine._memory = memory  # type: ignore[assignment]
    from memory_admission import AdmissionPolicy
    from memory_admission.contracts import PolicyConfig
    engine._admission_policy = AdmissionPolicy(PolicyConfig())  # type: ignore[assignment]
    engine._rag = None            # type: ignore[assignment]
    engine._cag = None            # type: ignore[assignment]
    engine._kg = None             # type: ignore[assignment]
    engine._hybrid = None         # type: ignore[assignment]
    engine._reasoner = None       # type: ignore[assignment]
    engine._cognition = None      # type: ignore[assignment]
    engine._generator = None      # type: ignore[assignment]
    engine._obs = None            # type: ignore[assignment]
    engine._tools = None          # type: ignore[assignment]
    engine._self_consistency = None  # type: ignore[assignment]


def test_remember_intent_admits_novel_preference() -> None:
    memory = _StubMemory(recall_hits=[])
    engine = Engine()
    _install(engine, memory=memory)
    result = engine.process("remember: I prefer dark mode in the editor",
                            session="s1")
    assert result.intent == "remember"
    assert len(memory.remember_auto_calls) == 1
    assert "dark mode" in memory.remember_auto_calls[0]["text"]


def test_remember_intent_rejects_exact_duplicate() -> None:
    memory = _StubMemory(recall_hits=[
        {"text": "I prefer dark mode in the editor",
         "kind": "preference", "ts": 1e12, "importance": 1.2},
    ])
    engine = Engine()
    _install(engine, memory=memory)
    result = engine.process("remember: I prefer dark mode in the editor",
                            session="s2")
    assert result.intent == "remember"
    assert memory.remember_auto_calls == [], (
        "Policy should have rejected the duplicate write."
    )


def test_retrieval_is_untouched_by_admission() -> None:
    hits = [
        {"text": "unrelated memory", "kind": "fact",
         "ts": 1e12, "importance": 1.0},
    ]
    memory = _StubMemory(recall_hits=hits)
    engine = Engine()
    _install(engine, memory=memory)
    engine.process("remember: totally different novel content", session="s3")
    assert len(memory.recall_calls) == 1
    assert memory._recall_hits == hits
