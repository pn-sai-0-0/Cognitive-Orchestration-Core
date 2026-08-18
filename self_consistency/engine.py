from __future__ import annotations

from self_consistency.comparison import compare_attempts
from self_consistency.contracts import ReasoningAttempt
from self_consistency.strategies import StrategyInput, default_strategies


class SelfConsistencyRunner:
    def __init__(
        self,
        reasoner,
        attempt_count: int = 5,
        answer_threshold: float = 0.72,
        support_threshold: float = 0.65,
        claim_threshold: float = 0.7,
        strategies=None,
    ):
        self._reasoner = reasoner
        self._answer_threshold = answer_threshold
        self._support_threshold = support_threshold
        self._claim_threshold = claim_threshold
        self._strategies = tuple(strategies or default_strategies())[:attempt_count]

    def run(
        self,
        query: str,
        memory: list[dict] | None = None,
        chunks: list[dict] | None = None,
        kg_facts: list[dict] | None = None,
        tool: dict | None = None,
        cognition: dict | None = None,
    ):
        payload = StrategyInput(
            query,
            list(memory or []),
            list(chunks or []),
            list(kg_facts or []),
            dict(tool) if tool else None,
            dict(cognition) if cognition else None,
        )
        attempts = []
        for index, strategy in enumerate(self._strategies, start=1):
            prepared = strategy.prepare(payload)
            contract = self._reasoner.assess_contract(
                query=prepared.query,
                memory=prepared.memory,
                chunks=prepared.chunks,
                kg_facts=prepared.kg_facts,
                tool=prepared.tool,
                cognition=prepared.cognition,
            )
            attempts.append(
                ReasoningAttempt(
                    f"attempt-{index}",
                    strategy.name,
                    contract["answer"],
                    float(contract["confidence"]),
                    tuple(contract["trace"]),
                    tuple(contract["citations"]),
                    tuple(contract["tool_usage"]),
                    tuple(contract["warnings"]),
                    float(contract["evidence_strength"]),
                    contract["answer_similarity_key"],
                    contract["reasoning"],
                )
            )
        return compare_attempts(
            query,
            tuple(attempts),
            self._answer_threshold,
            self._support_threshold,
            self._claim_threshold,
        )
