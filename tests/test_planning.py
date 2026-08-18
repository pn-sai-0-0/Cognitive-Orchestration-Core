from __future__ import annotations

from typing import Literal

from goal.contracts import Goal
from planning import normalize_title, plan_turn


def _goal(
    goal_id: str,
    title: str,
    *,
    status: Literal["active", "paused", "completed"] = "active",
    parent_goal_id: str | None = None,
    created_at: float = 1.0,
    priority: int = 0,
) -> Goal:
    return Goal(
        goal_id=goal_id,
        session="s1",
        title=title,
        status=status,
        parent_goal_id=parent_goal_id,
        created_at=created_at,
        updated_at=created_at,
        completed_at=created_at if status == "completed" else None,
        paused_at=created_at if status == "paused" else None,
        priority=priority,
    )


def test_single_active_goal_is_selected() -> None:
    goal = _goal("goal-000000000001", "Ship phase one")
    result = plan_turn("what should we do next", [goal])
    assert result.selected_goal is not None
    assert result.selected_goal["goal_id"] == goal.goal_id
    assert result.execution_order == [goal.goal_id]
    assert result.blocked_goals == []


def test_multiple_active_goals_select_by_priority() -> None:
    low = _goal("goal-000000000001", "Low", created_at=2.0, priority=1)
    high = _goal("goal-000000000002", "High", created_at=3.0, priority=9)
    result = plan_turn("continue", [low, high])
    assert result.selected_goal is not None
    assert result.selected_goal["goal_id"] == high.goal_id
    assert result.execution_order == [high.goal_id, low.goal_id]


def test_created_at_orders_identical_priority_goals() -> None:
    earlier = _goal("goal-000000000002", "Earlier", created_at=1.0)
    later = _goal("goal-000000000001", "Later", created_at=5.0)
    result = plan_turn("continue", [later, earlier])
    assert result.selected_goal is not None
    assert result.selected_goal["goal_id"] == earlier.goal_id


def test_goal_id_breaks_full_ties_deterministically() -> None:
    a = _goal("goal-000000000001", "Alpha", created_at=1.0, priority=3)
    b = _goal("goal-000000000002", "Beta", created_at=1.0, priority=3)
    result = plan_turn("continue", [b, a])
    assert result.selected_goal is not None
    assert result.selected_goal["goal_id"] == a.goal_id
    assert result.execution_order == [a.goal_id, b.goal_id]


def test_goal_id_override_wins() -> None:
    first = _goal("goal-000000000001", "First", priority=1)
    second = _goal("goal-000000000002", "Second", priority=9)
    result = plan_turn("focus on goal-000000000001 now", [first, second])
    assert result.selected_goal is not None
    assert result.selected_goal["goal_id"] == first.goal_id
    assert result.planning_reason == "explicit_goal_id_override"


def test_title_override_uses_normalized_title() -> None:
    low = _goal("goal-000000000001", "Write CI", priority=1)
    high = _goal("goal-000000000002", "Add constraints", priority=9)
    result = plan_turn("please focus on write, ci first", [low, high])
    assert normalize_title("Write, CI!") == "write ci"
    assert result.selected_goal is not None
    assert result.selected_goal["goal_id"] == low.goal_id
    assert result.planning_reason == "explicit_goal_title_override"


def test_dependency_blocking_and_release() -> None:
    parent = _goal("goal-000000000001", "Parent", status="active")
    child = _goal("goal-000000000002", "Child", parent_goal_id=parent.goal_id)
    blocked = plan_turn("continue", [parent, child])
    assert blocked.selected_goal is not None
    assert blocked.selected_goal["goal_id"] == parent.goal_id
    assert blocked.blocked_goals[0]["goal_id"] == child.goal_id
    assert blocked.blocked_goals[0]["block_reason"] == "blocked_incomplete_parent"

    parent_done = _goal(
        "goal-000000000001",
        "Parent",
        status="completed",
        created_at=1.0,
    )
    child_again = _goal(
        "goal-000000000002",
        "Child",
        parent_goal_id=parent_done.goal_id,
        created_at=2.0,
    )
    released = plan_turn("continue", [parent_done, child_again])
    assert released.selected_goal is not None
    assert released.selected_goal["goal_id"] == child_again.goal_id
    assert released.blocked_goals == []


def test_paused_parent_blocks_active_child() -> None:
    parent = _goal("goal-000000000001", "Parent", status="paused")
    child = _goal("goal-000000000002", "Child", parent_goal_id=parent.goal_id)
    result = plan_turn("continue", [parent, child])
    assert result.selected_goal is not None
    assert result.selected_goal["goal_id"] == parent.goal_id
    assert result.blocked_goals[0]["block_reason"] == "blocked_paused_parent"


def test_missing_parent_blocks_active_child() -> None:
    child = _goal("goal-000000000002", "Child", parent_goal_id="goal-missing")
    result = plan_turn("continue", [child])
    assert result.selected_goal is None
    assert result.blocked_goals[0]["block_reason"] == "blocked_missing_parent"
    assert "missing_parent_dependencies" in result.warnings


def test_no_active_goals_falls_back_to_paused_then_completed() -> None:
    paused = _goal("goal-000000000002", "Paused", status="paused", created_at=2.0)
    done = _goal("goal-000000000001", "Done", status="completed", created_at=1.0)
    result = plan_turn("continue", [done, paused])
    assert result.selected_goal is not None
    assert result.selected_goal["goal_id"] == paused.goal_id
    assert result.planning_reason == "automatic_paused_goal_selection"


def test_chain_blocks_multiple_descendants() -> None:
    parent = _goal("goal-000000000001", "Parent", status="active", created_at=1.0)
    child = _goal(
        "goal-000000000002", "Child", parent_goal_id=parent.goal_id, created_at=2.0
    )
    grandchild = _goal(
        "goal-000000000003", "Grandchild", parent_goal_id=child.goal_id, created_at=3.0
    )
    result = plan_turn("continue", [grandchild, child, parent])
    blocked_ids = [item["goal_id"] for item in result.blocked_goals]
    assert result.selected_goal is not None
    assert result.selected_goal["goal_id"] == parent.goal_id
    assert blocked_ids == [child.goal_id, grandchild.goal_id]


def test_cycles_are_blocked_and_warned() -> None:
    a = _goal("goal-000000000001", "A", parent_goal_id="goal-000000000002")
    b = _goal("goal-000000000002", "B", parent_goal_id="goal-000000000001")
    self_cycle = _goal("goal-000000000003", "Self", parent_goal_id="goal-000000000003")
    result = plan_turn("continue", [a, b, self_cycle])
    assert result.selected_goal is None
    assert len(result.blocked_goals) == 3
    assert "dependency_cycle_detected" in result.warnings
    assert all(item["block_reason"] == "blocked_cycle" for item in result.blocked_goals)


def test_explicit_selection_of_blocked_goal_is_visible() -> None:
    parent = _goal("goal-000000000001", "Parent", status="active")
    child = _goal("goal-000000000002", "Blocked child", parent_goal_id=parent.goal_id)
    result = plan_turn("please work on blocked child", [parent, child])
    assert result.selected_goal is not None
    assert result.selected_goal["goal_id"] == child.goal_id
    assert result.selected_goal["blocked"] is True
    assert "selected_goal_blocked:blocked_incomplete_parent" in result.warnings


def test_deterministic_repeated_execution_is_identical() -> None:
    goals = [
        _goal("goal-000000000001", "Parent", status="completed", created_at=1.0),
        _goal(
            "goal-000000000002",
            "Child",
            parent_goal_id="goal-000000000001",
            created_at=2.0,
        ),
        _goal("goal-000000000003", "Paused item", status="paused", created_at=3.0),
    ]
    first = plan_turn(
        "continue with child", goals, reflection={"revised": True}
    ).to_dict()
    second = plan_turn(
        "continue with child", goals, reflection={"revised": True}
    ).to_dict()
    assert first == second
    assert "reflection_revised" in first["warnings"]
