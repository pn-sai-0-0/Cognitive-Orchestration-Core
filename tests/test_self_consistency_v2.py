from __future__ import annotations

from reasoning.reasoner import Reasoner
from self_consistency.engine import SelfConsistencyRunner


def test_runner_produces_attempts_and_result():
    result = SelfConsistencyRunner(Reasoner()).run(
        "What is the default retrieval top_k?",
        memory=[{"text": "top_k is 8", "score": 0.98}],
        chunks=[{"text": "top_k = 8", "source": "config.py", "score": 0.98}],
        kg_facts=[
            {
                "subject": "retrieval",
                "relation": "top_k",
                "object": "8",
                "confidence": 0.98,
            }
        ],
    )
    assert len(result.attempts) == 5
    assert result.consensus.answer
    assert result.agreement_type == "strong_agreement"
