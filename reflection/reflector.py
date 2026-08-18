"""Reflector: run one reflection pass over a draft answer."""

from __future__ import annotations

from typing import Any

from reflection.checker import check_draft
from reflection.contracts import ReflectionReport


class Reflector:
    """One-shot reflection over a completed draft answer.

    Usage::

        reflector = Reflector(reasoner)
        revised_text, report = reflector.reflect(
            draft_answer=text,
            query=query,
            chunks=chunks,
            memory=memory,
            kg_facts=kg_facts,
        )

    Contract:
      * ``reflect`` runs the deterministic checker exactly once.
      * If the checker returns ``ok=True``, the draft is returned
        unchanged with ``ReflectionReport(revised=False, reason="")``.
      * If the checker returns ``ok=False``, the reasoner is asked
        once (via ``assess_contract``) for a re-derived answer from
        the same evidence. That revised text replaces the draft and
        ``ReflectionReport(revised=True, reason=<label>)`` is
        returned. If the reasoner call fails or produces an empty
        answer, the original draft is kept and ``reason`` becomes
        ``"revision_failed"`` (but ``revised`` remains False so
        downstream consumers see a truthful record).
    """

    def __init__(self, reasoner: Any):
        self._reasoner = reasoner

    def reflect(
        self,
        draft_answer: str,
        query: str,
        chunks: list[dict] | None = None,
        memory: list[dict] | None = None,
        kg_facts: list[dict] | None = None,
        tool: dict | None = None,
        cognition: dict | None = None,
    ) -> tuple[str, ReflectionReport]:
        check = check_draft(
            draft_answer=draft_answer,
            query=query,
            chunks=chunks,
            memory=memory,
            kg_facts=kg_facts,
        )

        if check.ok:
            return draft_answer, ReflectionReport(revised=False, reason="")

        # Attempt exactly one revision pass via the reasoner's stable
        # attempt-level contract. We do NOT loop.
        try:
            attempt = self._reasoner.assess_contract(
                query=query,
                memory=memory or [],
                chunks=chunks or [],
                kg_facts=kg_facts or [],
                tool=tool,
                cognition=cognition,
            )
        except Exception:  # noqa: BLE001 - reflection revision is intentionally fail-open.
            return draft_answer, ReflectionReport(
                revised=False, reason="revision_failed"
            )

        revised = (
            str(attempt.get("answer", "")).strip() if isinstance(attempt, dict) else ""
        )
        if not revised:
            return draft_answer, ReflectionReport(
                revised=False, reason="revision_failed"
            )

        return revised, ReflectionReport(revised=True, reason=check.reason)
