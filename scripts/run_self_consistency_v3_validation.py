"""Hard mixed validation set for self-consistency claim normalization."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reasoning.reasoner import Reasoner
from self_consistency.engine import SelfConsistencyRunner
from self_consistency.metrics import summarise_results


def case(query: str, memory: list[dict], chunks: list[dict], kg_facts: list[dict]) -> dict:
    return {"query": query, "memory": memory, "chunks": chunks, "kg_facts": kg_facts}


def main() -> None:
    cases = [
        case("How do I reset my forgotten password?", [{"text": "use the email reset link to reset a forgotten password", "score": .90}], [{"text": "Reset a forgotten password via the email reset link.", "source": "help_center", "score": .92}], [{"subject": "password reset", "relation": "uses", "object": "email reset link", "confidence": .90}]),
        case("What command should I use to search for a document?", [{"text": "run find document to search for a document", "score": .86}], [{"text": "You can search for a document by invoking find document.", "source": "engine.py", "score": .88}], [{"subject": "document search", "relation": "invokes", "object": "find document", "confidence": .90}]),
        case("What is the default retrieval top_k?", [{"text": "the default retrieval top_k is 8", "score": .95}], [{"text": "RETRIEVAL top_k = 8 before reranking", "source": "config.py", "score": .98}], [{"subject": "retrieval", "relation": "top_k", "object": "8", "confidence": .97}]),
        case("What is the final chunk count fed to the decoder?", [{"text": "final_k equals 5 after reranking", "score": .92}], [{"text": "final_k = 5 post-rerank selections passed to context", "source": "config.py", "score": .96}], [{"subject": "retrieval", "relation": "final_k", "object": "5", "confidence": .95}]),
        case("Why is the reasoner returning limited-confidence answers so often?", [{"text": "reasoner marks answers limited-confidence when verification issues are found", "score": .72}], [{"text": "verify_evidence adds an issues entry when query coverage in evidence is low.", "source": "reasoning/reasoner.py", "score": .70}], [{"subject": "reasoner", "relation": "hedges", "object": "when verification issues", "confidence": .72}]),
        case("Which port does COC listen on by default?", [], [], []),
        case("What is the default retrieval top_k? (conflict case)", [{"text": "the default retrieval top_k is 7", "score": .95}], [{"text": "RETRIEVAL top_k = 8 before reranking", "source": "config.py", "score": .98}], [{"subject": "retrieval", "relation": "top_k", "object": "9", "confidence": .97}]),
        case("How is the parent goal state affected when a goal is split?", [{"text": "splitting a goal completes the parent", "score": .90}], [{"text": "goal split pauses the parent goal", "source": "goal/store.py", "score": .93}], [{"subject": "goal split", "relation": "pauses", "object": "parent", "confidence": .92}]),
        case("What happens if no model checkpoint is available?", [{"text": "the system falls back to grounded fallback when no checkpoint is available", "score": .90}], [{"text": "with no checkpoint the generator uses grounded fallback", "source": "inference/generator.py", "score": .95}], [{"subject": "generator", "relation": "falls back to", "object": "grounded fallback", "confidence": .94}]),
        case("How do I clean up my session history?", [{"text": "clear_session clears conversation history for a session", "score": .88}], [{"text": "Call clear_session to remove session messages.", "source": "engine.py", "score": .90}], [{"subject": "clear_session", "relation": "clears", "object": "session messages", "confidence": .90}]),
    ]
    runner = SelfConsistencyRunner(Reasoner())
    results = [runner.run(item["query"], item["memory"], item["chunks"], item["kg_facts"]) for item in cases]
    report = {"summary": summarise_results(results), "queries": []}
    for result in results:
        report["queries"].append({
            "query": result.query,
            "attempts": len(result.attempts),
            "consensus_answer": result.consensus.answer,
            "minority_answer": result.minority[0].answer if result.minority else "",
            "agreement_type": result.agreement_type,
            "confidence": result.confidence,
            "warning_reason": result.warning_reason,
            "claim_agreement": result.metrics.claim_agreement,
        })
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
