from __future__ import annotations

from engine import Engine
from goal.store import GoalStore


class _StubGenerator:
    backend = "stub"

    def __init__(self, output: str) -> None:
        self._output = output

    def generate(self, message: str, context: dict) -> str:
        return self._output

    def generate_stream(self, message: str, context: dict):
        yield self._output[:5]
        yield self._output[5:]
        yield ""


def _install(
    engine: Engine, store: GoalStore, *, output: str = "planner response"
) -> None:
    engine._init_done = True
    engine._goal_store = store  # type: ignore[assignment]
    engine._generator = _StubGenerator(output)  # type: ignore[assignment]
    engine._memory = None  # type: ignore[assignment]
    engine._rag = None  # type: ignore[assignment]
    engine._cag = None  # type: ignore[assignment]
    engine._kg = None  # type: ignore[assignment]
    engine._hybrid = None  # type: ignore[assignment]
    engine._reasoner = None  # type: ignore[assignment]
    engine._cognition = None  # type: ignore[assignment]
    engine._obs = None  # type: ignore[assignment]
    engine._tools = None  # type: ignore[assignment]
    engine._self_consistency = None  # type: ignore[assignment]
    engine._admission_policy = None  # type: ignore[assignment]


def test_process_attaches_planning_trace_without_mutating_goals(tmp_path) -> None:
    store = GoalStore(tmp_path / "goals.db")
    try:
        goal = store.create("Ship phase one", session="s1")
        before = [item.to_dict() for item in store.list_session("s1")]

        engine = Engine()
        _install(engine, store)
        result = engine.process("continue", session="s1")

        assert result.text == "planner response"
        assert result.trace["planning"]["selected_goal"]["goal_id"] == goal.goal_id
        assert result.trace["planning"]["execution_order"] == [goal.goal_id]
        after = [item.to_dict() for item in store.list_session("s1")]
        assert before == after
    finally:
        store.close()


def test_process_stream_attaches_planning_trace(tmp_path) -> None:
    store = GoalStore(tmp_path / "goals.db")
    try:
        goal = store.create("Write CI", session="s1")
        engine = Engine()
        _install(engine, store, output="streamed text")

        events = list(engine.process_stream("continue", session="s1"))
        assert events[-1]["done"] is True
        assert (
            events[-1]["trace"]["planning"]["selected_goal"]["goal_id"] == goal.goal_id
        )
    finally:
        store.close()


def test_goal_command_still_short_circuits_before_planning(tmp_path) -> None:
    store = GoalStore(tmp_path / "goals.db")
    engine = Engine()
    engine._init_done = True
    engine._goal_store = store  # type: ignore[assignment]
    try:
        result = engine.process("/goal create Keep scope narrow", session="s2")
        assert result.intent == "goal_command"
        assert result.text.startswith("Goal created: goal-")
        assert "planning" not in result.trace
    finally:
        store.close()
