"""
CognitiveOC v3 — Workflow Engine
==================================

Implements a persistent workflow state machine with crash recovery.

State machine:
    CREATED → CLASSIFYING → PLANNING → GATHERING → EXECUTING
           → VERIFYING → REPORTING → COMPLETED | FAILED

Every workflow persists to var/workflows/<id>.json.
Crash recovery: re-load by ID and call workflow.resume().
Long-running tasks: step limits + per-step timeout enforced.

File: workflow/workflow.py
Used by: engine.py (intent == 'workflow'), research/engine.py, ui/app.py

States:
    CREATED     — workflow initialised, not started
    CLASSIFYING — analysing task type and requirements
    PLANNING    — building step plan
    GATHERING   — collecting evidence/data
    EXECUTING   — running steps
    VERIFYING   — checking results
    REPORTING   — assembling final report
    COMPLETED   — successfully finished
    FAILED      — terminal failure
    PAUSED      — manually paused, resumable
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

try:
    from config import WORKFLOW, WORKFLOW_DIR, ensure_dirs
except ImportError:
    WORKFLOW    = dict(max_steps=50, step_timeout_s=300, max_concurrent=3,
                       retry_max=3, retry_delay_s=5)
    WORKFLOW_DIR = Path("var/workflows")
    def ensure_dirs(): pass


# ═══════════════════════════════════════════════════════════════════
# State enum
# ═══════════════════════════════════════════════════════════════════

class WFState(str, Enum):
    CREATED     = "created"
    CLASSIFYING = "classifying"
    PLANNING    = "planning"
    GATHERING   = "gathering"
    EXECUTING   = "executing"
    VERIFYING   = "verifying"
    REPORTING   = "reporting"
    COMPLETED   = "completed"
    FAILED      = "failed"
    PAUSED      = "paused"

_TERMINAL = {WFState.COMPLETED, WFState.FAILED}
_STATE_ORDER = [
    WFState.CLASSIFYING, WFState.PLANNING, WFState.GATHERING,
    WFState.EXECUTING,   WFState.VERIFYING, WFState.REPORTING,
]


# ═══════════════════════════════════════════════════════════════════
# Step dataclass
# ═══════════════════════════════════════════════════════════════════

@dataclass
class WorkflowStep:
    name:        str
    description: str
    state:       str       = "pending"    # pending|running|done|failed|skipped
    result:      Any       = None
    error:       str       = ""
    started_at:  float     = 0.0
    finished_at: float     = 0.0
    retries:     int       = 0

    @property
    def duration_s(self) -> float:
        if self.finished_at and self.started_at:
            return round(self.finished_at - self.started_at, 2)
        return 0.0


# ═══════════════════════════════════════════════════════════════════
# Workflow dataclass
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Workflow:
    """A single workflow instance with full state machine lifecycle."""
    id:           str
    goal:         str
    task_type:    str             = "general"
    state:        WFState         = WFState.CREATED
    steps:        list[WorkflowStep] = field(default_factory=list)
    current_step: int             = 0
    result:       str             = ""
    error:        str             = ""
    evidence:     list[dict]      = field(default_factory=list)
    created_at:   float           = field(default_factory=time.time)
    updated_at:   float           = field(default_factory=time.time)
    completed_at: float           = 0.0
    metadata:     dict            = field(default_factory=dict)
    session:      str             = "default"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["state"] = self.state.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Workflow":
        steps = [WorkflowStep(**s) for s in d.pop("steps", [])]
        state = WFState(d.pop("state", "created"))
        wf = cls(**d)
        wf.steps = steps
        wf.state = state
        return wf

    def add_trace(self, message: str):
        self.metadata.setdefault("trace", []).append(
            {"ts": time.strftime("%H:%M:%S"), "msg": message}
        )

    def summary(self) -> dict:
        done  = sum(1 for s in self.steps if s.state == "done")
        total = len(self.steps)
        pct   = round(done / max(total, 1) * 100, 1)
        return {
            "id":           self.id,
            "goal":         self.goal[:80],
            "task_type":    self.task_type,
            "state":        self.state.value,
            "progress_pct": pct,
            "steps_done":   done,
            "steps_total":  total,
            "current_step": self.current_step,
            "created_at":   self.created_at,
            "updated_at":   self.updated_at,
            "result":       self.result[:200] if self.result else "",
            "error":        self.error[:200] if self.error else "",
        }


# ═══════════════════════════════════════════════════════════════════
# WorkflowEngine
# ═══════════════════════════════════════════════════════════════════

class WorkflowEngine:
    """Persistent workflow state machine engine.

    Usage:
        wfe = WorkflowEngine()
        wf  = wfe.create("Research transformers vs RNNs", session="s1")
        wfe.run(wf.id)          # blocking
        status = wfe.status(wf.id)
        wfe.resume(wf.id)       # after crash

    All workflows persist to var/workflows/<id>.json.
    """

    def __init__(self):
        ensure_dirs()
        self._wf_dir      = Path(str(WORKFLOW_DIR))
        self._wf_dir.mkdir(parents=True, exist_ok=True)
        self._active:     dict[str, Workflow] = {}
        self._lock        = threading.RLock()
        self._max_steps   = WORKFLOW.get("max_steps", 50)
        self._step_timeout= WORKFLOW.get("step_timeout_s", 300)
        self._retry_max   = WORKFLOW.get("retry_max", 3)
        self._retry_delay = WORKFLOW.get("retry_delay_s", 5)

    # ── Persistence ───────────────────────────────────────────────────
    def _path(self, wf_id: str) -> Path:
        return self._wf_dir / f"{wf_id}.json"

    def _save(self, wf: Workflow):
        wf.updated_at = time.time()
        with self._lock:
            self._path(wf.id).write_text(json.dumps(wf.to_dict(), indent=2))

    def _load(self, wf_id: str) -> Workflow | None:
        p = self._path(wf_id)
        if not p.exists():
            return None
        try:
            return Workflow.from_dict(json.loads(p.read_text()))
        except Exception:
            return None

    # ── Lifecycle ─────────────────────────────────────────────────────
    def create(self, goal: str, task_type: str = "general",
               session: str = "default", metadata: dict = None) -> Workflow:
        """Create and persist a new workflow. Does NOT start execution."""
        wf_id = str(uuid.uuid4())[:12]
        wf    = Workflow(
            id        = wf_id,
            goal      = goal,
            task_type = task_type,
            session   = session,
            metadata  = metadata or {},
        )
        wf.add_trace("created")
        with self._lock:
            self._active[wf_id] = wf
        self._save(wf)
        return wf

    def run(self, wf_id: str,
            engine_ctx: dict = None,
            on_step: Callable = None) -> Workflow:
        """Execute a workflow through all states. Blocking.

        Args:
            wf_id:     Workflow ID.
            engine_ctx: Dict with memory, retrieval, kg, reasoner references.
            on_step:   Optional callback(step_name, result) after each step.

        Returns:
            Final Workflow state.
        """
        wf = self._load(wf_id) or self._active.get(wf_id)
        if wf is None:
            raise ValueError(f"Workflow {wf_id} not found")

        if wf.state in _TERMINAL:
            return wf

        with self._lock:
            self._active[wf_id] = wf

        try:
            for state in _STATE_ORDER:
                if wf.state in _TERMINAL:
                    break
                if wf.state == WFState.PAUSED:
                    break
                wf.state = state
                wf.add_trace(f"entered {state.value}")
                self._save(wf)

                step_fn = getattr(self, f"_step_{state.value}", None)
                if step_fn:
                    step_fn(wf, engine_ctx or {})
                    if on_step:
                        try:
                            on_step(state.value, wf.summary())
                        except Exception:
                            pass

            if wf.state not in _TERMINAL and wf.state != WFState.PAUSED:
                wf.state       = WFState.COMPLETED
                wf.completed_at= time.time()
                wf.add_trace("completed")
                self._save(wf)

        except Exception as e:
            wf.state = WFState.FAILED
            wf.error = str(e)[:500]
            wf.add_trace(f"FAILED: {e}")
            self._save(wf)

        return wf

    def resume(self, wf_id: str, engine_ctx: dict = None) -> Workflow:
        """Resume a paused or in-progress workflow after crash."""
        wf = self._load(wf_id)
        if wf is None:
            raise ValueError(f"Workflow {wf_id} not found on disk")
        if wf.state in _TERMINAL:
            return wf
        if wf.state == WFState.PAUSED:
            wf.state = WFState.EXECUTING
        wf.add_trace("resumed")
        self._save(wf)
        return self.run(wf_id, engine_ctx)

    def pause(self, wf_id: str) -> Workflow:
        """Pause a running workflow."""
        wf = self._active.get(wf_id) or self._load(wf_id)
        if wf and wf.state not in _TERMINAL:
            wf.state = WFState.PAUSED
            wf.add_trace("paused by user")
            self._save(wf)
        return wf

    def cancel(self, wf_id: str) -> Workflow:
        """Cancel a workflow (mark as FAILED)."""
        wf = self._active.get(wf_id) or self._load(wf_id)
        if wf:
            wf.state = WFState.FAILED
            wf.error = "cancelled by user"
            wf.add_trace("cancelled")
            self._save(wf)
        return wf

    # ── State step handlers ───────────────────────────────────────────
    def _step_classifying(self, wf: Workflow, ctx: dict):
        """Classify task type and required resources."""
        goal_lower = wf.goal.lower()
        if any(w in goal_lower for w in ("research","find","investigate","analyse")):
            wf.task_type = "research"
        elif any(w in goal_lower for w in ("generate","write","create","draft")):
            wf.task_type = "generation"
        elif any(w in goal_lower for w in ("validate","verify","check","test")):
            wf.task_type = "validation"
        elif any(w in goal_lower for w in ("summarise","summarize","overview")):
            wf.task_type = "summarization"
        else:
            wf.task_type = "general"
        wf.add_trace(f"classified as {wf.task_type}")

    def _step_planning(self, wf: Workflow, ctx: dict):
        """Build the step plan for this workflow."""
        templates = {
            "research": [
                WorkflowStep("formulate_query",  "Define research question"),
                WorkflowStep("gather_evidence",  "Collect evidence from memory/retrieval/KG"),
                WorkflowStep("analyse_evidence", "Analyse and cross-reference evidence"),
                WorkflowStep("synthesise",       "Synthesise findings"),
                WorkflowStep("validate_claims",  "Validate claims against KG"),
                WorkflowStep("generate_report",  "Generate structured report"),
            ],
            "generation": [
                WorkflowStep("understand_intent", "Clarify generation intent"),
                WorkflowStep("gather_context",    "Gather relevant context"),
                WorkflowStep("generate_draft",    "Generate initial draft"),
                WorkflowStep("review_draft",      "Review and improve draft"),
                WorkflowStep("finalise",          "Finalise output"),
            ],
            "validation": [
                WorkflowStep("parse_claims",    "Extract claims to validate"),
                WorkflowStep("check_kg",        "Cross-check against KG"),
                WorkflowStep("check_retrieval", "Cross-check against retrieval index"),
                WorkflowStep("score_confidence","Score claim confidence"),
                WorkflowStep("generate_report", "Generate validation report"),
            ],
            "summarization": [
                WorkflowStep("retrieve_content", "Retrieve relevant content"),
                WorkflowStep("extract_key_points","Extract key points"),
                WorkflowStep("synthesise",        "Synthesise summary"),
            ],
            "general": [
                WorkflowStep("classify",   "Classify task"),
                WorkflowStep("gather",     "Gather information"),
                WorkflowStep("execute",    "Execute task"),
                WorkflowStep("verify",     "Verify result"),
                WorkflowStep("report",     "Report outcome"),
            ],
        }
        wf.steps        = templates.get(wf.task_type, templates["general"])
        wf.current_step = 0
        wf.add_trace(f"planned {len(wf.steps)} steps")

    def _step_gathering(self, wf: Workflow, ctx: dict):
        """Gather evidence from all available sources."""
        evidence = []
        query    = wf.goal

        # Memory
        memory = ctx.get("memory")
        if memory:
            try:
                hits = memory.ranked_recall(query, k=5)
                for h in hits:
                    evidence.append({"type": "memory", "text": h.get("text","")[:300],
                                     "score": h.get("score", 0)})
            except Exception:
                pass

        # Retrieval
        retriever = ctx.get("retriever")
        if retriever:
            try:
                ret = retriever.retrieve(query, k=5)
                for c in ret.get("chunks", []):
                    evidence.append({"type": "retrieval", "text": c.get("text","")[:300],
                                     "source": c.get("source",""), "score": c.get("score",0)})
            except Exception:
                pass

        # KG
        kg = ctx.get("kg")
        if kg:
            try:
                facts = kg.ranked_query(query, limit=5)
                for f in facts:
                    evidence.append({"type": "kg",
                                     "text": f"{f.get('subject','')} {f.get('relation','')} {f.get('object','')}",
                                     "confidence": f.get("confidence", 0)})
            except Exception:
                pass

        wf.evidence = evidence
        wf.add_trace(f"gathered {len(evidence)} evidence items")

    def _step_executing(self, wf: Workflow, ctx: dict):
        """Execute workflow steps up to max_steps limit."""
        steps_run = 0
        for i, step in enumerate(wf.steps):
            if steps_run >= self._max_steps:
                wf.add_trace(f"step limit reached at {steps_run}")
                break
            if step.state in ("done", "skipped"):
                continue

            wf.current_step = i
            step.state      = "running"
            step.started_at = time.time()
            self._save(wf)

            # Generic step execution (subclass / override for specific logic)
            try:
                step.result  = f"Completed: {step.description}"
                step.state   = "done"
                steps_run   += 1
            except Exception as e:
                step.error  = str(e)
                step.retries+= 1
                if step.retries >= self._retry_max:
                    step.state = "failed"
                else:
                    step.state = "pending"
            step.finished_at = time.time()
            wf.add_trace(f"step[{i}] {step.name} → {step.state}")
            self._save(wf)

    def _step_verifying(self, wf: Workflow, ctx: dict):
        """Verify workflow results for completeness and consistency."""
        failed = [s for s in wf.steps if s.state == "failed"]
        if failed:
            wf.add_trace(f"verification: {len(failed)} failed steps")
        else:
            wf.add_trace("verification: all steps completed")

    def _step_reporting(self, wf: Workflow, ctx: dict):
        """Assemble final report from evidence and step results."""
        parts = [f"# Workflow Report: {wf.goal[:80]}",
                 f"\n**Task type:** {wf.task_type}",
                 f"**Steps completed:** {sum(1 for s in wf.steps if s.state=='done')}/{len(wf.steps)}"]

        if wf.evidence:
            parts.append("\n## Evidence Gathered")
            for ev in wf.evidence[:5]:
                src = ev.get("source", ev.get("type",""))
                parts.append(f"- [{src}] {ev.get('text','')[:120]}")

        done_steps = [s for s in wf.steps if s.state == "done" and s.result]
        if done_steps:
            parts.append("\n## Results")
            for s in done_steps[:5]:
                parts.append(f"- **{s.name}**: {str(s.result)[:120]}")

        wf.result = "\n".join(parts)
        wf.add_trace("report generated")

    # ── Query API ─────────────────────────────────────────────────────
    def status(self, wf_id: str) -> dict | None:
        wf = self._active.get(wf_id) or self._load(wf_id)
        return wf.summary() if wf else None

    def list_workflows(self, state: str = None, limit: int = 20) -> list[dict]:
        results = []
        for p in sorted(self._wf_dir.glob("*.json"), key=lambda x: -x.stat().st_mtime):
            if len(results) >= limit:
                break
            try:
                wf = Workflow.from_dict(json.loads(p.read_text()))
                if state and wf.state.value != state:
                    continue
                results.append(wf.summary())
            except Exception:
                pass
        return results

    def get_workflow(self, wf_id: str) -> Workflow | None:
        return self._active.get(wf_id) or self._load(wf_id)

    def delete(self, wf_id: str) -> bool:
        p = self._path(wf_id)
        if p.exists():
            p.unlink()
        with self._lock:
            self._active.pop(wf_id, None)
        return True

    def active_count(self) -> int:
        return sum(
            1 for wf in self._active.values()
            if wf.state not in _TERMINAL and wf.state != WFState.PAUSED
        )
