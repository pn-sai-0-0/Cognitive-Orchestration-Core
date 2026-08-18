from __future__ import annotations

import re
from collections.abc import Iterable

from goal.contracts import Goal
from planning.contracts import PlanningResult

_STATUS_RANK = {"active": 0, "paused": 1, "completed": 2}
_TITLE_TOKEN_RE = re.compile(r"[a-z0-9]+")


def normalize_title(text: str) -> str:
    return " ".join(_TITLE_TOKEN_RE.findall((text or "").lower()))


def _priority(goal: Goal) -> int:
    return int(getattr(goal, "priority", 0) or 0)


def _sort_key(goal: Goal) -> tuple[int, int, float, str]:
    return (
        _STATUS_RANK.get(goal.status, 99),
        -_priority(goal),
        float(goal.created_at),
        goal.goal_id,
    )


def _goal_item(goal: Goal, *, block_reason: str | None = None) -> dict:
    item = dict(goal.to_dict())
    item["priority"] = _priority(goal)
    item["blocked"] = block_reason is not None
    if block_reason is not None:
        item["block_reason"] = block_reason
    return item


def _has_ancestor_cycle(goal: Goal, goal_map: dict[str, Goal]) -> bool:
    seen: set[str] = {goal.goal_id}
    current = goal
    while current.parent_goal_id is not None:
        parent_id = current.parent_goal_id
        if parent_id in seen:
            return True
        seen.add(parent_id)
        parent = goal_map.get(parent_id)
        if parent is None:
            return False
        current = parent
    return False


def _block_reason(goal: Goal, goal_map: dict[str, Goal]) -> str | None:
    if _has_ancestor_cycle(goal, goal_map):
        return "blocked_cycle"
    if goal.status != "active":
        return None
    parent_id = goal.parent_goal_id
    if parent_id is None:
        return None
    parent = goal_map.get(parent_id)
    if parent is None:
        return "blocked_missing_parent"
    if parent.status == "completed":
        return None
    if parent.status == "paused":
        return "blocked_paused_parent"
    return "blocked_incomplete_parent"


def _match_goal_id(message: str, ordered_goals: list[Goal]) -> Goal | None:
    lower_message = (message or "").lower()
    matches = [goal for goal in ordered_goals if goal.goal_id.lower() in lower_message]
    if not matches:
        return None
    return min(matches, key=_sort_key)


def _match_goal_title(message: str, ordered_goals: list[Goal]) -> Goal | None:
    normalized_message = normalize_title(message)
    matches = []
    for goal in ordered_goals:
        normalized_goal = normalize_title(goal.title)
        if normalized_goal and normalized_goal in normalized_message:
            matches.append((len(normalized_goal), goal))
    if not matches:
        return None
    matches.sort(key=lambda item: (-item[0], *_sort_key(item[1])))
    return matches[0][1]


def _auto_reason(goal: Goal | None, goals_present: bool) -> str:
    if goal is None:
        return "no_eligible_goals" if goals_present else "no_goals"
    if goal.status == "active":
        return "automatic_active_goal_selection"
    if goal.status == "paused":
        return "automatic_paused_goal_selection"
    return "automatic_completed_goal_selection"


def _confidence(
    selected_goal: Goal | None,
    override_reason: str | None,
    selected_block_reason: str | None,
) -> float:
    if selected_goal is None:
        return 0.0
    if override_reason == "explicit_goal_id_override":
        return 1.0 if selected_block_reason is None else 0.65
    if override_reason == "explicit_goal_title_override":
        return 0.95 if selected_block_reason is None else 0.6
    base = {
        "active": 0.9,
        "paused": 0.75,
        "completed": 0.6,
    }.get(selected_goal.status, 0.5)
    if selected_block_reason is not None:
        base = min(base, 0.5)
    return round(base, 2)


def plan_turn(
    message: str, goals: Iterable[Goal], reflection: dict | None = None
) -> PlanningResult:
    ordered_goals = sorted(goals, key=_sort_key)
    goal_map = {goal.goal_id: goal for goal in ordered_goals}
    block_map = {
        goal.goal_id: reason
        for goal in ordered_goals
        if (reason := _block_reason(goal, goal_map)) is not None
    }

    override_reason: str | None = None
    selected_goal = _match_goal_id(message, ordered_goals)
    if selected_goal is not None:
        override_reason = "explicit_goal_id_override"
    else:
        selected_goal = _match_goal_title(message, ordered_goals)
        if selected_goal is not None:
            override_reason = "explicit_goal_title_override"

    eligible_goals = [goal for goal in ordered_goals if goal.goal_id not in block_map]
    if selected_goal is None and eligible_goals:
        selected_goal = eligible_goals[0]

    selected_block_reason = None
    if selected_goal is not None:
        selected_block_reason = block_map.get(selected_goal.goal_id)

    considered_goals = [
        _goal_item(goal, block_reason=block_map.get(goal.goal_id))
        for goal in ordered_goals
    ]
    blocked_goals = [
        _goal_item(goal, block_reason=block_map[goal.goal_id])
        for goal in ordered_goals
        if goal.goal_id in block_map
    ]
    deferred_goals = [
        _goal_item(goal)
        for goal in eligible_goals
        if selected_goal is None or goal.goal_id != selected_goal.goal_id
    ]
    execution_order = [goal.goal_id for goal in eligible_goals]

    warnings: list[str] = []
    if any(item.get("block_reason") == "blocked_cycle" for item in blocked_goals):
        warnings.append("dependency_cycle_detected")
    if any(
        item.get("block_reason") == "blocked_missing_parent" for item in blocked_goals
    ):
        warnings.append("missing_parent_dependencies")
    if reflection and reflection.get("revised"):
        warnings.append("reflection_revised")
    if selected_block_reason is not None:
        warnings.append(f"selected_goal_blocked:{selected_block_reason}")

    planning_reason = override_reason or _auto_reason(
        selected_goal, bool(ordered_goals)
    )
    selected_item = None
    if selected_goal is not None:
        selected_item = _goal_item(selected_goal, block_reason=selected_block_reason)

    return PlanningResult(
        selected_goal=selected_item,
        considered_goals=considered_goals,
        blocked_goals=blocked_goals,
        deferred_goals=deferred_goals,
        execution_order=execution_order,
        planning_reason=planning_reason,
        confidence=_confidence(selected_goal, override_reason, selected_block_reason),
        warnings=warnings,
    )
