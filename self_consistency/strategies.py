from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StrategyInput:
    query: str
    memory: list[dict]
    chunks: list[dict]
    kg_facts: list[dict]
    tool: dict | None
    cognition: dict | None


class ReasoningStrategy:
    name = "baseline"

    def prepare(self, payload: StrategyInput) -> StrategyInput:
        return StrategyInput(
            payload.query,
            [dict(x) for x in payload.memory],
            [dict(x) for x in payload.chunks],
            [dict(x) for x in payload.kg_facts],
            dict(payload.tool) if payload.tool else None,
            dict(payload.cognition) if payload.cognition else None,
        )


class BaselineStrategy(ReasoningStrategy):
    name = "baseline"


class RetrievalFocusedStrategy(ReasoningStrategy):
    name = "retrieval_focused"

    def prepare(self, payload: StrategyInput) -> StrategyInput:
        p = super().prepare(payload)
        return StrategyInput(
            p.query,
            [] if p.chunks else p.memory,
            p.chunks,
            [] if p.chunks else p.kg_facts,
            p.tool,
            p.cognition,
        )


class MemoryFocusedStrategy(ReasoningStrategy):
    name = "memory_focused"

    def prepare(self, payload: StrategyInput) -> StrategyInput:
        p = super().prepare(payload)
        return StrategyInput(
            p.query,
            p.memory,
            [] if p.memory else p.chunks,
            [] if p.memory else p.kg_facts,
            p.tool,
            p.cognition,
        )


class KnowledgeFocusedStrategy(ReasoningStrategy):
    name = "knowledge_focused"

    def prepare(self, payload: StrategyInput) -> StrategyInput:
        p = super().prepare(payload)
        return StrategyInput(
            p.query,
            [] if p.kg_facts else p.memory,
            [] if p.kg_facts else p.chunks,
            p.kg_facts,
            p.tool,
            p.cognition,
        )


class ThinEvidenceStrategy(ReasoningStrategy):
    name = "thin_evidence"

    def prepare(self, payload: StrategyInput) -> StrategyInput:
        p = super().prepare(payload)
        return StrategyInput(
            p.query, p.memory[:1], p.chunks[:1], p.kg_facts[:1], p.tool, p.cognition
        )


def default_strategies() -> tuple[ReasoningStrategy, ...]:
    return (
        BaselineStrategy(),
        RetrievalFocusedStrategy(),
        MemoryFocusedStrategy(),
        KnowledgeFocusedStrategy(),
        ThinEvidenceStrategy(),
    )
