from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ReasoningAttempt:
    attempt_id: str
    strategy: str
    answer: str
    confidence: float
    trace: tuple[str, ...]
    citations: tuple[str, ...]
    tool_usage: tuple[str, ...]
    warnings: tuple[str, ...]
    evidence_strength: float
    answer_similarity_key: str
    reasoning: dict = field(default_factory=dict)
    claim: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ConsistencyMetrics:
    attempt_count: int
    agreement_ratio: float
    answer_agreement: float
    trace_agreement: float
    confidence_agreement: float
    citation_agreement: float
    tool_agreement: float
    evidence_strength: float
    claim_agreement: float = 0.0


@dataclass(frozen=True)
class ConsistencyResult:
    query: str
    attempts: tuple[ReasoningAttempt, ...]
    consensus: ReasoningAttempt
    minority: tuple[ReasoningAttempt, ...]
    agreement: bool
    agreement_type: str
    warning_reason: str
    superficial: bool
    confidence: float
    metrics: ConsistencyMetrics

    def to_dict(self) -> dict:
        def serialize(attempt: ReasoningAttempt) -> dict:
            return {
                "attempt_id": attempt.attempt_id,
                "strategy": attempt.strategy,
                "answer": attempt.answer,
                "confidence": attempt.confidence,
                "trace": list(attempt.trace),
                "citations": list(attempt.citations),
                "tool_usage": list(attempt.tool_usage),
                "warnings": list(attempt.warnings),
                "evidence_strength": attempt.evidence_strength,
                "claim": attempt.claim,
            }

        return {
            "query": self.query,
            "attempts": [serialize(attempt) for attempt in self.attempts],
            "consensus": serialize(self.consensus),
            "minority": [serialize(attempt) for attempt in self.minority],
            "agreement": self.agreement,
            "agreement_type": self.agreement_type,
            "warning_reason": self.warning_reason,
            "superficial": self.superficial,
            "confidence": self.confidence,
            "metrics": self.metrics.__dict__,
        }
