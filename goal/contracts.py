from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

GoalStatus = Literal["active", "paused", "completed"]


@dataclass(frozen=True)
class Goal:
    goal_id: str
    session: str
    title: str
    status: GoalStatus
    parent_goal_id: str | None
    created_at: float
    updated_at: float
    completed_at: float | None = None
    paused_at: float | None = None
    priority: int = 0

    def to_dict(self) -> dict:
        return {
            "goal_id": self.goal_id,
            "session": self.session,
            "title": self.title,
            "status": self.status,
            "parent_goal_id": self.parent_goal_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at,
            "paused_at": self.paused_at,
            "priority": self.priority,
        }
