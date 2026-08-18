from __future__ import annotations

import sqlite3
import time
import uuid
from pathlib import Path

from goal.contracts import Goal

_ALLOWED_STATUSES = ("active", "paused", "completed")


class GoalStore:
    def __init__(self, db_path: str | Path):
        self._db_path = str(db_path)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS goals (
                goal_id         TEXT PRIMARY KEY,
                session         TEXT NOT NULL,
                title           TEXT NOT NULL CHECK (length(trim(title)) > 0),
                status          TEXT NOT NULL CHECK (
                    status IN ('active', 'paused', 'completed')
                ),
                parent_goal_id  TEXT REFERENCES goals(goal_id) ON DELETE RESTRICT,
                created_at      REAL NOT NULL,
                updated_at      REAL NOT NULL,
                completed_at    REAL,
                paused_at       REAL,
                UNIQUE(parent_goal_id, title)
            );
            CREATE INDEX IF NOT EXISTS idx_goals_session_status
                ON goals(session, status, updated_at DESC);
            CREATE INDEX IF NOT EXISTS idx_goals_parent ON goals(parent_goal_id);
            """
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def _row_to_goal(self, row: sqlite3.Row) -> Goal:
        return Goal(
            goal_id=row["goal_id"],
            session=row["session"],
            title=row["title"],
            status=row["status"],
            parent_goal_id=row["parent_goal_id"],
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
            completed_at=float(row["completed_at"])
            if row["completed_at"] is not None
            else None,
            paused_at=(
                float(row["paused_at"]) if row["paused_at"] is not None else None
            ),
            priority=0,
        )

    def _generate_goal_id(self) -> str:
        return f"goal-{uuid.uuid4().hex[:12]}"

    def get(self, goal_id: str) -> Goal:
        row = self._conn.execute(
            (
                "SELECT goal_id, session, title, status, parent_goal_id, "
                "created_at, updated_at, completed_at, paused_at "
                "FROM goals WHERE goal_id=?"
            ),
            (goal_id,),
        ).fetchone()
        if not row:
            raise KeyError(f"Unknown goal id: {goal_id}")
        return self._row_to_goal(row)

    def create(
        self, title: str, session: str = "default", parent_goal_id: str | None = None
    ) -> Goal:
        now = time.time()
        goal_id = self._generate_goal_id()
        self._conn.execute(
            (
                "INSERT INTO goals("
                "goal_id, session, title, status, parent_goal_id, "
                "created_at, updated_at, completed_at, paused_at"
                ") VALUES(?,?,?,?,?,?,?,?,?)"
            ),
            (goal_id, session, title, "active", parent_goal_id, now, now, None, None),
        )
        self._conn.commit()
        return self.get(goal_id)

    def _set_status(self, goal_id: str, status: str) -> Goal:
        if status not in _ALLOWED_STATUSES:
            raise ValueError(f"Unsupported status: {status}")
        current = self.get(goal_id)
        now = time.time()
        paused_at = current.paused_at
        completed_at = current.completed_at
        if status == "paused":
            paused_at = now
        elif status == "active":
            paused_at = None
        elif status == "completed":
            completed_at = now
        self._conn.execute(
            (
                "UPDATE goals SET status=?, updated_at=?, paused_at=?, "
                "completed_at=? WHERE goal_id=?"
            ),
            (status, now, paused_at, completed_at, goal_id),
        )
        self._conn.commit()
        return self.get(goal_id)

    def pause(self, goal_id: str) -> Goal:
        return self._set_status(goal_id, "paused")

    def resume(self, goal_id: str) -> Goal:
        return self._set_status(goal_id, "active")

    def complete(self, goal_id: str) -> Goal:
        return self._set_status(goal_id, "completed")

    def split(
        self, goal_id: str, child_titles: list[str]
    ) -> tuple[Goal, tuple[Goal, ...]]:
        parent = self.pause(goal_id)
        cleaned = [title.strip() for title in child_titles if title and title.strip()]
        if not cleaned:
            raise ValueError("split requires at least one child title")
        children = tuple(
            self.create(
                title=title, session=parent.session, parent_goal_id=parent.goal_id
            )
            for title in cleaned
        )
        return self.get(goal_id), children

    def list_active(self, session: str = "default") -> list[Goal]:
        rows = self._conn.execute(
            (
                "SELECT goal_id, session, title, status, parent_goal_id, "
                "created_at, updated_at, completed_at, paused_at "
                "FROM goals WHERE session=? AND status='active' "
                "ORDER BY updated_at DESC"
            ),
            (session,),
        ).fetchall()
        return [self._row_to_goal(row) for row in rows]

    def list_session(self, session: str = "default") -> list[Goal]:
        rows = self._conn.execute(
            (
                "SELECT goal_id, session, title, status, parent_goal_id, "
                "created_at, updated_at, completed_at, paused_at "
                "FROM goals WHERE session=? "
                "ORDER BY created_at ASC, goal_id ASC"
            ),
            (session,),
        ).fetchall()
        return [self._row_to_goal(row) for row in rows]


def handle_goal_command(
    store: GoalStore, command: str, session: str = "default"
) -> str:
    text = (command or "").strip()
    if not text.startswith("/goal"):
        raise ValueError("goal command must start with /goal")
    body = text[len("/goal") :].strip()
    if not body:
        raise ValueError("missing /goal action")
    action, _, remainder = body.partition(" ")
    action = action.strip().lower()
    remainder = remainder.strip()

    if action == "create":
        goal = store.create(remainder, session=session)
        return f"Goal created: {goal.goal_id} - {goal.title}"
    if action == "pause":
        goal = store.pause(remainder)
        return f"Goal paused: {goal.goal_id}"
    if action == "resume":
        goal = store.resume(remainder)
        return f"Goal resumed: {goal.goal_id}"
    if action == "complete":
        goal = store.complete(remainder)
        return f"Goal completed: {goal.goal_id}"
    if action == "split":
        goal_id, sep, children_blob = remainder.partition(":")
        if not sep:
            raise ValueError(
                "split syntax: /goal split <goal_id>: child one | child two"
            )
        child_titles = [item.strip() for item in children_blob.split("|")]
        parent, children = store.split(goal_id.strip(), child_titles)
        child_text = ", ".join(f"{child.goal_id} - {child.title}" for child in children)
        return f"Goal split: {parent.goal_id} -> {child_text}"
    raise ValueError(f"unsupported /goal action: {action}")
