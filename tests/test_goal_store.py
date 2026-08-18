from __future__ import annotations

import sqlite3

import pytest

from goal.store import GoalStore


@pytest.fixture()
def store(tmp_path):
    db_path = tmp_path / "goals.db"
    goal_store = GoalStore(db_path)
    try:
        yield goal_store
    finally:
        goal_store.close()


def test_goal_store_create_pause_resume_complete_split(store: GoalStore):
    goal = store.create("Ship phase one", session="s1")
    assert goal.status == "active"

    paused = store.pause(goal.goal_id)
    assert paused.status == "paused"
    assert paused.paused_at is not None

    resumed = store.resume(goal.goal_id)
    assert resumed.status == "active"
    assert resumed.paused_at is None

    parent, children = store.split(goal.goal_id, ["Write CI", "Add constraints"])
    assert parent.status == "paused"
    assert len(children) == 2
    assert all(child.parent_goal_id == goal.goal_id for child in children)

    completed = store.complete(goal.goal_id)
    assert completed.status == "completed"
    assert completed.completed_at is not None


def test_goal_store_constraints_fire(store: GoalStore):
    with pytest.raises(sqlite3.IntegrityError):
        store.create("   ", session="s1")

    with pytest.raises(sqlite3.IntegrityError):
        store.create("Missing session", session=None)  # type: ignore[arg-type]

    with pytest.raises(sqlite3.IntegrityError):
        store.create("Orphan child", session="s1", parent_goal_id="goal-missing")

    parent = store.create("Parent", session="s1")
    store.create("Duplicate child", session="s1", parent_goal_id=parent.goal_id)
    with pytest.raises(sqlite3.IntegrityError):
        store.create("Duplicate child", session="s1", parent_goal_id=parent.goal_id)
