from __future__ import annotations

from reasoning.reasoner import Reasoner
from self_consistency.claim_normalizer import claim_equivalence, normalize_claim
from self_consistency.engine import SelfConsistencyRunner


def test_paraphrase_numeric_claims_normalize_equally():
    left = normalize_claim(
        "Best supported answer from config.py: retrieval top_k is 8 before reranking"
    )
    right = normalize_claim(
        "Best supported answer from memory: the default retrieval top_k is 8"
    )
    assert left.key == right.key
    assert claim_equivalence(left, right) == 1.0


def test_numeric_conflicts_stay_distinct():
    seven = normalize_claim("Best supported answer from memory: top_k is 7")
    eight = normalize_claim("Best supported answer from config.py: top_k = 8")
    assert seven.key != eight.key
    assert claim_equivalence(seven, eight) == 0.0


def test_action_conflicts_stay_distinct():
    paused = normalize_claim(
        "Best supported answer from goal/store.py: goal split pauses the parent goal"
    )
    completed = normalize_claim(
        "Best supported answer from memory: splitting a goal completes the parent"
    )
    assert paused.subject == completed.subject == "goal_split"
    assert paused.predicate != completed.predicate
    assert claim_equivalence(paused, completed) == 0.0


def test_fallback_claims_have_shared_dedicated_key():
    left = normalize_claim(
        "I don't have enough information to answer this confidently. "
        "The available evidence is too thin."
    )
    right = normalize_claim(
        "I don't have enough information to answer this confidently. "
        "The available evidence is too thin."
    )
    assert left.negation is True
    assert left.key == right.key


def test_paraphrase_equivalent_answers_become_agreement():
    result = SelfConsistencyRunner(Reasoner()).run(
        "How do I reset my forgotten password?",
        memory=[
            {
                "text": "use the email reset link to reset a forgotten password",
                "score": 0.90,
            }
        ],
        chunks=[
            {
                "text": "Reset a forgotten password via the email reset link.",
                "source": "help_center",
                "score": 0.92,
            }
        ],
        kg_facts=[
            {
                "subject": "password reset",
                "relation": "uses",
                "object": "email reset link",
                "confidence": 0.90,
            }
        ],
    )
    assert result.agreement_type == "strong_agreement"
    assert result.metrics.claim_agreement >= 0.7


def test_conflicting_claims_remain_clear_disagreement():
    result = SelfConsistencyRunner(Reasoner()).run(
        "What is the default retrieval top_k?",
        memory=[{"text": "the default retrieval top_k is 7", "score": 0.95}],
        chunks=[
            {
                "text": "RETRIEVAL top_k = 8 before reranking",
                "source": "config.py",
                "score": 0.98,
            }
        ],
        kg_facts=[
            {
                "subject": "retrieval",
                "relation": "top_k",
                "object": "9",
                "confidence": 0.97,
            }
        ],
    )
    assert result.agreement_type == "clear_disagreement"


def test_thin_evidence_remains_false_agreement():
    result = SelfConsistencyRunner(Reasoner()).run(
        "Unknown configuration", memory=[], chunks=[], kg_facts=[]
    )
    assert result.agreement_type == "false_agreement"
