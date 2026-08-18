from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PlanningResult:
    selected_goal: dict | None
    considered_goals: list[dict] = field(default_factory=list)
    blocked_goals: list[dict] = field(default_factory=list)
    deferred_goals: list[dict] = field(default_factory=list)
    execution_order: list[str] = field(default_factory=list)
    planning_reason: str = ""
    confidence: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "selected_goal": self.selected_goal,
            "considered_goals": list(self.considered_goals),
            "blocked_goals": list(self.blocked_goals),
            "deferred_goals": list(self.deferred_goals),
            "execution_order": list(self.execution_order),
            "planning_reason": self.planning_reason,
            "confidence": self.confidence,
            "warnings": list(self.warnings),
        }
