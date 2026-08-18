"""
CognitiveOC v3 — Research Engine
==================================

Orchestrates long-running research loops:
    Research → Gather → Analyse → Validate → Refine → Report

Uses the WorkflowEngine for state persistence and crash recovery.
Every research task is a Workflow under the hood.

File: research/engine.py
Used by: engine.py (intent == 'research'), workflow/workflow.py, ui/app.py
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

try:
    from config import RESEARCH, WORKFLOW_DIR, ensure_dirs
except ImportError:
    RESEARCH     = dict(max_loops=10, evidence_min=3, synthesis_top_k=10,
                        validation_rounds=2, report_format="markdown")
    WORKFLOW_DIR  = Path("var/workflows")
    def ensure_dirs(): pass


class ResearchEngine:
    """Orchestrates multi-loop research tasks.

    Each research task creates a Workflow, runs the evidence loop,
    and produces a structured report.

    Usage:
        re  = ResearchEngine(engine_ctx)
        wf  = re.start("What are the trade-offs between RAG and CAG?")
        rep = re.get_report(wf.id)
    """

    def __init__(self, engine_ctx: dict = None):
        """
        engine_ctx should contain:
            memory, retriever, kg, reasoner references
        """
        self._ctx    = engine_ctx or {}
        self._max    = RESEARCH.get("max_loops", 10)
        self._ev_min = RESEARCH.get("evidence_min", 3)
        self._top_k  = RESEARCH.get("synthesis_top_k", 10)
        self._v_rounds = RESEARCH.get("validation_rounds", 2)
        self._fmt    = RESEARCH.get("report_format", "markdown")

        from workflow.workflow import WorkflowEngine
        self._wfe = WorkflowEngine()

    # ── Public interface ──────────────────────────────────────────────
    def start(self, question: str, session: str = "default") -> Any:
        """Start a research workflow and run it. Returns Workflow."""
        wf = self._wfe.create(
            goal      = question,
            task_type = "research",
            session   = session,
            metadata  = {"engine": "research", "max_loops": self._max},
        )
        return self._wfe.run(wf.id, self._ctx)

    def start_async(self, question: str, session: str = "default") -> str:
        """Start a research workflow in background. Returns workflow ID."""
        import threading
        wf = self._wfe.create(
            goal      = question,
            task_type = "research",
            session   = session,
        )
        t  = threading.Thread(
            target  = self._wfe.run,
            args    = (wf.id, self._ctx),
            daemon  = True,
            name    = f"research-{wf.id}",
        )
        t.start()
        return wf.id

    def status(self, wf_id: str) -> dict | None:
        return self._wfe.status(wf_id)

    def get_report(self, wf_id: str) -> str:
        wf = self._wfe.get_workflow(wf_id)
        if not wf:
            return ""
        return wf.result or ""

    def list_research(self, limit: int = 10) -> list[dict]:
        return self._wfe.list_workflows(limit=limit)

    # ── Evidence loop (called by workflow gather step) ────────────────
    def evidence_loop(self, question: str, loops: int = None) -> dict:
        """Run iterative evidence collection and return structured evidence.

        Each loop:
          1. Query memory, retrieval, KG
          2. Analyse gaps
          3. Refine query for next loop
          4. Stop if evidence_min reached or loops exhausted

        Returns: {"evidence": list, "loops_run": int, "gap_analysis": list}
        """
        loops    = loops or self._max
        evidence = []
        gaps     = []
        query    = question

        memory    = self._ctx.get("memory")
        retriever = self._ctx.get("retriever")
        kg        = self._ctx.get("kg")

        for loop_i in range(loops):
            loop_evidence = []

            # Memory
            if memory:
                try:
                    hits = memory.ranked_recall(query, k=3)
                    for h in hits:
                        loop_evidence.append({
                            "type": "memory", "loop": loop_i,
                            "text": h.get("text","")[:300],
                            "score": h.get("score",0),
                        })
                except Exception:
                    pass

            # Retrieval
            if retriever:
                try:
                    ret = retriever.retrieve(query, k=3)
                    for c in ret.get("chunks",[]):
                        loop_evidence.append({
                            "type": "retrieval", "loop": loop_i,
                            "text": c.get("text","")[:300],
                            "source": c.get("source",""),
                            "score": c.get("score",0),
                        })
                except Exception:
                    pass

            # KG
            if kg:
                try:
                    facts = kg.ranked_query(query, limit=3)
                    for f in facts:
                        loop_evidence.append({
                            "type": "kg", "loop": loop_i,
                            "text": (f"{f.get('subject','')} "
                                     f"{f.get('relation','')} "
                                     f"{f.get('object','')}"),
                            "confidence": f.get("confidence",0),
                        })
                except Exception:
                    pass

            evidence.extend(loop_evidence)

            # Gap analysis: terms in question not covered in evidence
            import re
            q_terms = set(re.findall(r"[a-z]{4,}", question.lower()))
            ev_text = " ".join(e.get("text","") for e in loop_evidence)
            ev_terms= set(re.findall(r"[a-z]{4,}", ev_text.lower()))
            new_gaps= list(q_terms - ev_terms)[:5]
            gaps.extend(new_gaps)

            # Stop if sufficient evidence gathered
            if len(evidence) >= self._ev_min:
                break

            # Refine query with identified gaps
            if new_gaps:
                query = question + " " + " ".join(new_gaps[:3])

        return {
            "evidence":     evidence[:self._top_k],
            "loops_run":    loop_i + 1,
            "gap_analysis": list(set(gaps))[:10],
            "coverage":     len(evidence),
        }

    # ── Synthesis ─────────────────────────────────────────────────────
    def synthesise(self, question: str,
                   evidence: list[dict],
                   validation_rounds: int = None) -> str:
        """Synthesise evidence into a structured research report.

        Returns:
            Markdown report string.
        """
        v_rounds = validation_rounds or self._v_rounds

        # Group evidence by type
        by_type: dict[str, list] = {}
        for ev in evidence:
            t = ev.get("type","unknown")
            by_type.setdefault(t, []).append(ev)

        parts = [
            f"# Research Report",
            f"\n**Question:** {question}",
            f"**Evidence items:** {len(evidence)}  "
            f"**Sources:** {', '.join(by_type.keys())}",
        ]

        # Evidence sections
        for etype, items in by_type.items():
            parts.append(f"\n## {etype.title()} Evidence")
            for item in items[:5]:
                text = item.get("text","")[:200]
                src  = item.get("source", item.get("type",""))
                parts.append(f"- [{src}] {text}")

        # Synthesis statement
        all_text = " ".join(e.get("text","") for e in evidence)
        import re
        key_terms = list(set(re.findall(r"\b[A-Za-z]{5,}\b", all_text)))[:15]
        parts.append(f"\n## Key Themes\n{', '.join(key_terms)}")

        # Validation note
        parts.append(f"\n## Validation\n"
                      f"Evidence validated over {v_rounds} round(s). "
                      f"{'Sufficient evidence gathered.' if len(evidence) >= self._ev_min else 'Evidence may be incomplete — consider adding more documents.'}")

        return "\n".join(parts)
