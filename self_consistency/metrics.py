from __future__ import annotations

from collections import Counter


def summarise_results(results: list) -> dict:
    if not results:
        return {"queries": 0, "agreement_types": {}, "avg_confidence": 0.0}
    return {
        "queries": len(results),
        "agreement_types": dict(Counter(result.agreement_type for result in results)),
        "avg_confidence": round(
            sum(result.confidence for result in results) / len(results), 4
        ),
    }
