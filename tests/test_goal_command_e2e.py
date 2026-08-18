from __future__ import annotations

import re

from engine import Engine
from goal.store import GoalStore, handle_goal_command


def _goal_id(response: str) -> str:
    match = re.search(r"goal-[a-f0-9]{12}", response)
    assert match
    return match.group(0)


def test_goal_command_flow(tmp_path):
    store = GoalStore(tmp_path / "goals.db")
    try:
        created = handle_goal_command(
            store, "/goal create Ship phase one", session="s1"
        )
        goal_id = _goal_id(created)
        assert "Goal created" in created

        paused = handle_goal_command(store, f"/goal pause {goal_id}", session="s1")
        assert paused == f"Goal paused: {goal_id}"

        resumed = handle_goal_command(store, f"/goal resume {goal_id}", session="s1")
        assert resumed == f"Goal resumed: {goal_id}"

        split = handle_goal_command(
            store, f"/goal split {goal_id}: Write CI | Add constraints", session="s1"
        )
        assert split.startswith(f"Goal split: {goal_id} -> ")
        assert "Write CI" in split
        assert "Add constraints" in split

        completed = handle_goal_command(
            store, f"/goal complete {goal_id}", session="s1"
        )
        assert completed == f"Goal completed: {goal_id}"
    finally:
        store.close()


def test_engine_short_circuits_goal_command(tmp_path):
    engine = Engine()
    engine._init_done = True
    engine._goal_store = GoalStore(tmp_path / "engine-goals.db")
    try:
        result = engine.process("/goal create Keep scope narrow", session="s2")
        assert result.intent == "goal_command"
        assert result.text.startswith("Goal created: goal-")
    finally:
        engine._goal_store.close()
