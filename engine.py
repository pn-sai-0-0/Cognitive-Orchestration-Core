"""
CognitiveOC v3 — Cognitive Orchestration Core (Engine)
=======================================================

The Engine is the single entry point for every request.
No subsystem calls another directly at runtime — the engine sequences
everything, passes outputs as inputs to the next stage, and assembles
the final context before generation.

Full pipeline per request (build_context → generate):

  1.  check_input()          guardrails: injection, rate limit, length
  2.  _perception()          normalise text, detect language
  3.  _route()               intent classification (15 patterns)
  4.  short_circuit check    memory/workflow intents handled without model
  5.  encoder_stack()        run intent+emotion+goal encoders
  6.  cognition.process()    Human Cognition Layer (9 modules)
  7.  _tool_dispatch()       calculator/code/file/search tools
  8.  memory.ranked_recall() 4-factor memory recall
  9.  kg.enrich_recall()     annotate memories with KG context
  10. hybrid.retrieve()      semantic+BM25+reranker hybrid retrieval
  11. sanitise_chunks()      retrieval sanitisation guardrail
  12. kg.ranked_query()      KG fact retrieval
  13. reasoner.assess()      full reasoning pipeline
  14. generator.generate()   700M decoder or grounded fallback
  15. filter_output()        PII redaction, secret redaction
  16. memory.remember()      persist exchange
  17. kg.extract_from_text() extract KG triples from response
  18. observability.record() log metrics

File: engine.py
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterator
from dataclasses import dataclass, field

try:
    from config import (
        COGNITION,
        GOALS_DB,
        GUARDRAILS,
        INFERENCE,
        KNOWLEDGE_GRAPH,
        MEMORY,
        MEMORY_ADMISSION,
        MODEL,
        REFLECTION,
        RETRIEVAL,
        SELF_CONSISTENCY,
        VALIDATION,
        ensure_dirs,
    )
except ImportError:
    MODEL = {}
    INFERENCE = {}
    MEMORY = {}
    RETRIEVAL = {}
    KNOWLEDGE_GRAPH = {}
    VALIDATION = {"confidence_threshold": 0.70}
    COGNITION = {}
    GUARDRAILS = {}
    GOALS_DB = "goals.db"
    SELF_CONSISTENCY = {
        "enabled": False,
        "attempt_count": 5,
        "answer_threshold": 0.72,
        "support_threshold": 0.65,
    }
    REFLECTION = {"enabled": False}
    MEMORY_ADMISSION = {"enabled": False}

    def ensure_dirs():
        pass


# ═══════════════════════════════════════════════════════════════════
# Result dataclass
# ═══════════════════════════════════════════════════════════════════


@dataclass
class EngineResult:
    """Structured result from engine.process()."""

    text: str
    session: str
    intent: str
    trace: dict = field(default_factory=dict)
    ok: bool = True
    error: str = ""
    self_consistency: dict = field(default_factory=dict)
    reflection: dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════
# Intent patterns (15 patterns, matched in order)
# ═══════════════════════════════════════════════════════════════════

_INTENT_PATTERNS = [
    (
        "calculate",
        re.compile(
            r"^(calc|calculate|compute|what\s+is\s+\d|evaluate)\b", re.IGNORECASE
        ),
    ),
    (
        "code",
        re.compile(
            r"\b(run|execute)\s+(this\s+)?(python|code|script)\b", re.IGNORECASE
        ),
    ),
    (
        "remember",
        re.compile(
            r"^(remember|note\s+that|save\s+this|memo|don'?t\s+forget)\b", re.IGNORECASE
        ),
    ),
    (
        "recall_profile",
        re.compile(
            r"\b(what\s+do\s+you\s+know\s+about\s+me|my\s+(preferences?|profile"
            r"|settings?|style))\b",
            re.IGNORECASE,
        ),
    ),
    (
        "search_files",
        re.compile(
            r"\b(find|search)\s+(a\s+|my\s+|the\s+)?"
            r"(file|document|pdf|doc)\b",
            re.IGNORECASE,
        ),
    ),
    ("goal_command", re.compile(r"^/goal\b", re.IGNORECASE)),
    ("kg_query", re.compile(r"^(who|what|where|when)\s+.{3,40}\?$", re.IGNORECASE)),
    (
        "workflow",
        re.compile(
            r"^(run\s+workflow|execute\s+task|agent:|start\s+task)\b", re.IGNORECASE
        ),
    ),
    (
        "ingest",
        re.compile(
            r"\b(ingest|index|upload|add\s+document|load\s+file)\b", re.IGNORECASE
        ),
    ),
    (
        "goal_add",
        re.compile(
            r"\b(add\s+goal|set\s+goal|new\s+goal|create\s+goal)\b", re.IGNORECASE
        ),
    ),
    (
        "goal_list",
        re.compile(
            r"\b(show\s+(my\s+)?goals?|list\s+goals?|what\s+are\s+my\s+goals?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "memory_list",
        re.compile(
            r"\b(show\s+(my\s+)?memories?|list\s+memories?|what\s+do\s+you\s+remember)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "eval",
        re.compile(r"^(run\s+eval|evaluate\s+model|run\s+evaluation)\b", re.IGNORECASE),
    ),
    (
        "metrics",
        re.compile(
            r"^(show\s+metrics?|system\s+status|observability)\b", re.IGNORECASE
        ),
    ),
    (
        "teach",
        re.compile(
            r"\b(teach\s+me|explain\s+step\s+by\s+step|quiz\s+me|lesson)\b",
            re.IGNORECASE,
        ),
    ),
    ("chat", re.compile(r".*", re.IGNORECASE)),  # catch-all
]


def _route(text: str) -> str:
    for intent, pat in _INTENT_PATTERNS:
        if pat.search(text.strip()):
            return intent
    return "chat"


# ═══════════════════════════════════════════════════════════════════
# Engine
# ═══════════════════════════════════════════════════════════════════


class Engine:
    """COC v3 Cognitive Orchestration Core.

    All subsystems are initialised lazily on first use.
    Thread-safe: each request uses its own local state dict.

    Usage:
        eng = Engine()
        result = eng.process("What is attention?")
        print(result.text)
        print(result.trace["citations"])

        # Streaming
        for fragment in eng.process_stream("Explain transformers"):
            print(fragment, end="", flush=True)
    """

    def __init__(self):
        ensure_dirs()
        # Lazy-loaded subsystems
        self._memory = None
        self._rag = None
        self._cag = None
        self._hybrid = None
        self._kg = None
        self._reasoner = None
        self._cognition = None
        self._generator = None
        self._obs = None
        self._guardrails = None
        self._tools = None
        self._goal_store = None
        self._self_consistency = None
        self._admission_policy = None
        self._init_done = False
        self._replay_state = None

    # ── Lazy initialisation ───────────────────────────────────────────
    def _init(self):
        if self._init_done:
            return
        try:
            from cognition.cognition import CognitionLayer
            from goal.store import GoalStore
            from inference.generator import ResponseGenerator
            from knowledge.graph import KnowledgeGraph
            from memory.memory import CognitiveMemory
            from observability.metrics import Observability
            from reasoning.reasoner import Reasoner
            from retrieval.rag import CAGManager, HybridRetriever, RAGPipeline
            from self_consistency.engine import SelfConsistencyRunner

            self._memory = CognitiveMemory()
            self._rag = RAGPipeline()
            self._cag = CAGManager()
            self._kg = KnowledgeGraph()
            self._hybrid = HybridRetriever(
                rag=self._rag, cag=self._cag, memory=self._memory, kg=self._kg
            )
            self._reasoner = Reasoner()
            self._cognition = CognitionLayer()
            self._generator = ResponseGenerator()
            self._obs = Observability()
            self._goal_store = GoalStore(GOALS_DB)
            if SELF_CONSISTENCY.get("enabled", False) and self._reasoner:
                self._self_consistency = SelfConsistencyRunner(
                    reasoner=self._reasoner,
                    attempt_count=SELF_CONSISTENCY.get("attempt_count", 5),
                    answer_threshold=SELF_CONSISTENCY.get("answer_threshold", 0.72),
                    support_threshold=SELF_CONSISTENCY.get("support_threshold", 0.65),
                )
            if MEMORY_ADMISSION.get("enabled", False):
                from memory_admission import AdmissionPolicy
                from memory_admission.contracts import PolicyConfig

                self._admission_policy = AdmissionPolicy(
                    PolicyConfig(
                        importance_admit_threshold=float(
                            MEMORY_ADMISSION.get("importance_admit_threshold", 0.60)
                        ),
                        exact_duplicate_threshold=float(
                            MEMORY_ADMISSION.get("exact_duplicate_threshold", 0.98)
                        ),
                        near_duplicate_threshold=float(
                            MEMORY_ADMISSION.get("near_duplicate_threshold", 0.85)
                        ),
                        recent_duplicate_window_s=float(
                            MEMORY_ADMISSION.get(
                                "recent_duplicate_window_s", 24.0 * 60.0 * 60.0
                            )
                        ),
                        min_content_tokens=int(
                            MEMORY_ADMISSION.get("min_content_tokens", 2)
                        ),
                        min_text_chars=int(MEMORY_ADMISSION.get("min_text_chars", 4)),
                        max_history_scan=int(
                            MEMORY_ADMISSION.get("max_history_scan", 500)
                        ),
                    )
                )
            self._init_done = True

        except Exception as e:  # noqa: BLE001 - best-effort lazy init fallback keeps engine usable.
            print(f"[engine] Init warning: {e}")
            self._init_done = True

    # ── Public properties (for ui/app.py access) ──────────────────────
    @property
    def rag(self):
        self._init()
        return self._rag

    @property
    def memory(self):
        self._init()
        return self._memory

    @property
    def kg(self):
        self._init()
        return self._kg

    @property
    def observability(self):
        self._init()
        return self._obs

    # ── Status ────────────────────────────────────────────────────────
    def status(self) -> dict:
        """Full system status dict (used by /api/status)."""
        self._init()
        gen_backend = self._generator.backend if self._generator else "not loaded"
        mem_stats = self._memory.stats() if self._memory else {}
        rag_stats = self._rag.stats() if self._rag else {}
        kg_stats = {}
        if self._kg:
            kg_stats = {
                "triples": len(self._kg),
                "entities": self._kg._conn.execute(
                    "SELECT COUNT(*) FROM entities"
                ).fetchone()[0],
            }
        obs_snap = self._obs.snapshot() if self._obs else {}

        from cognition.cognition import get_state as cog_state
        from safety.guardrails_state import get as gs_get

        return {
            "version": "3.0",
            "backend": gen_backend,
            "memory": mem_stats,
            "retrieval": rag_stats,
            "kg": kg_stats,
            "guardrails": gs_get(),
            "cognition": cog_state(),
            "metrics": obs_snap.get("requests", {}),
            "hardware": obs_snap.get("hardware", {}),
            "training": obs_snap.get("training", {}),
        }

    # ── Document ingestion ────────────────────────────────────────────
    def ingest(self, path: str) -> dict:
        """Ingest a document: parse, chunk, embed, index + KG extract."""
        self._init()
        from safety.guardrails import safe_file

        ok, reason = safe_file(path)
        if not ok:
            return {"ok": False, "error": reason}

        result = self._rag.ingest(path) if self._rag else {"ok": False}
        if result.get("ok") and self._kg:
            try:
                from pathlib import Path

                text = Path(path).read_text(encoding="utf-8", errors="replace")[:8000]
                added = self._kg.extract_from_text(text, source=path)
                result["kg_triples"] = len(added)
                if self._obs:
                    self._obs.record_kg_extract(len(added))
            except Exception:  # noqa: BLE001, S110 - optional KG enrichment must not block ingest.
                pass
        return result

    # ── Context assembly (full pipeline without generation) ───────────
    def build_context(
        self,
        message: str,
        session: str = "default",
        ip: str = "127.0.0.1",
        active_doc: str | None = None,
    ) -> dict | None:
        """Run full pre-generation pipeline. Returns context dict or None on block."""
        self._init()

        t0 = time.time()

        # ── Step 1: Input guardrails ───────────────────────────────────
        from safety.guardrails import check_input, sanitise_chunks

        ok, _reason, clean_msg = check_input(message, ip)
        if not ok:
            return None  # blocked

        # ── Step 2: Intent routing ─────────────────────────────────────
        intent = _route(clean_msg)

        # ── Step 3: Short-circuit intents (no model needed) ───────────
        if intent == "goal_command":
            try:
                from goal.store import handle_goal_command

                response = handle_goal_command(
                    self._goal_store,
                    clean_msg,
                    session=session,
                )
                return {"_short_circuit": True, "intent": intent, "response": response}
            except Exception as exc:  # noqa: BLE001 - goal commands intentionally surface backend errors verbatim.
                return {
                    "_short_circuit": True,
                    "intent": intent,
                    "response": f"Goal command error: {exc}",
                }

        if intent == "remember":
            text = re.sub(
                r"^(remember|note\s+that|save\s+this|memo)\s*[:\-]?\s*",
                "",
                clean_msg,
                flags=re.IGNORECASE,
            ).strip()
            if text and self._memory:
                from safety.guardrails import check_memory_write

                ok_w, _ = check_memory_write(text)
                if ok_w:
                    decision = self._admit_memory(
                        text=text,
                        kind="preference",
                        importance_hint=1.0,
                        session=session,
                        source="remember_intent",
                    )
                    if decision is None or decision.admitted:
                        self._memory.remember_auto(text, session=session)
            return {
                "_short_circuit": True,
                "intent": intent,
                "response": f"Remembered: {text[:80]}",
            }

        if intent == "recall_profile":
            profile = self._memory.profile() if self._memory else {}
            resp = "Here's what I know about you:\n"
            if profile:
                resp += "\n".join(f"- {k}: {v}" for k, v in list(profile.items())[:10])
            return {"_short_circuit": True, "intent": intent, "response": resp}

        if intent == "goal_list":
            goals = self._cognition.active_goals() if self._cognition else []
            resp = (
                (
                    "Active goals:\n"
                    + "\n".join(
                        f"- {g['title']} ({int(g['progress'] * 100)}%)" for g in goals
                    )
                )
                if goals
                else "No active goals."
            )
            return {"_short_circuit": True, "intent": intent, "response": resp}

        if intent == "goal_add":
            title = re.sub(
                r"^(add|set|new|create)\s+goal\s*[:\-]?\s*",
                "",
                clean_msg,
                flags=re.IGNORECASE,
            ).strip()
            if title and self._cognition:
                gid = self._cognition.add_goal(title)
                return {
                    "_short_circuit": True,
                    "intent": intent,
                    "response": f"Goal added (id={gid}): {title}",
                }

        if intent == "memory_list":
            mems = self._memory.list_memories(limit=10) if self._memory else []
            resp = (
                (
                    "Recent memories:\n"
                    + "\n".join(f"- [{m['kind']}] {m['text'][:80]}" for m in mems)
                )
                if mems
                else "No memories stored."
            )
            return {"_short_circuit": True, "intent": intent, "response": resp}

        if intent == "metrics":
            snap = self._obs.snapshot() if self._obs else {}
            import json

            return {
                "_short_circuit": True,
                "intent": intent,
                "response": json.dumps(snap, indent=2, default=str)[:2000],
            }

        # ── Step 4: Tool dispatch ──────────────────────────────────────
        tool_result = None
        if intent in ("calculate", "code", "search_files"):
            tool_result = self._run_tool(intent, clean_msg)

        # ── Step 5: Memory recall ──────────────────────────────────────
        memories: list[dict] = []
        if self._memory:
            try:
                memories = self._memory.ranked_recall(
                    clean_msg, k=MEMORY.get("recall_k", 8)
                )
            except Exception:  # noqa: BLE001, S110 - fail-open preserves responses when recall backends fail.
                pass

        # ── Step 6: Memory → KG enrichment ────────────────────────────
        if memories and self._kg:
            try:
                from memory.summarizer import enrich_recall

                memories = enrich_recall(memories, self._kg)
            except Exception:  # noqa: BLE001, S110 - fail-open preserves baseline recall results.
                pass

        # ── Step 7: Hybrid retrieval ───────────────────────────────────
        retrieval_result: dict = {}
        if self._hybrid:
            try:
                history = (
                    self._memory.recent(session, MEMORY.get("session_window", 30))
                    if self._memory
                    else []
                )
                retrieval_result = self._hybrid.retrieve(
                    query=clean_msg,
                    session=session,
                    history=history,
                    active_doc=active_doc,
                    k=RETRIEVAL.get("final_k", 5),
                    rerank=True,
                )
            except Exception:  # noqa: BLE001, S110 - fail-open preserves non-retrieval responses.
                pass

        chunks = retrieval_result.get("chunks", [])
        citations = retrieval_result.get("citations", "")

        # Sanitise retrieved chunks
        chunks = sanitise_chunks(chunks)

        # ── Step 8: KG query ───────────────────────────────────────────
        kg_facts: list[dict] = []
        if self._kg:
            try:
                kg_facts = self._kg.ranked_query(clean_msg, limit=10)
            except Exception:  # noqa: BLE001, S110 - fail-open preserves responses without KG facts.
                pass

        # ── Step 9: Human Cognition Layer ─────────────────────────────
        cog_result: dict = {}
        if self._cognition:
            try:
                history = self._memory.recent(session, 10) if self._memory else []
                cog_result = self._cognition.process(
                    text=clean_msg,
                    session=session,
                    history=history,
                    chunks=chunks,
                    memories=memories,
                    kg_facts=[
                        f.get("subject", "")
                        + " "
                        + f.get("relation", "")
                        + " "
                        + f.get("object", "")
                        for f in kg_facts
                    ],
                )
            except Exception:  # noqa: BLE001, S110 - fail-open preserves baseline reasoning inputs.
                pass

        # ── Step 10: Reasoning ─────────────────────────────────────────
        reasoning: dict = {}
        self_consistency: dict = {}
        if self._reasoner:
            try:
                reasoning = self._reasoner.assess(
                    query=clean_msg,
                    memory=memories,
                    chunks=chunks,
                    kg_facts=kg_facts,
                    tool=tool_result,
                    cognition=cog_result,
                )
            except Exception:  # noqa: BLE001, S110 - fail-open preserves response generation without reasoning metadata.
                pass

        if self._self_consistency and self._reasoner:
            try:
                sc_result = self._self_consistency.run(
                    query=clean_msg,
                    memory=memories,
                    chunks=chunks,
                    kg_facts=kg_facts,
                    tool=tool_result,
                    cognition=cog_result,
                )
                self_consistency = sc_result.to_dict()
                if sc_result.agreement_type in {
                    "strong_agreement",
                    "superficial_agreement",
                    "false_agreement",
                }:
                    reasoning = dict(sc_result.consensus.reasoning)
            except Exception:  # noqa: BLE001 - self-consistency is optional and intentionally fail-open.
                self_consistency = {}

        # ── Step 11: Session history ───────────────────────────────────
        history_turns = (
            self._memory.recent(session, MEMORY.get("session_window", 30))
            if self._memory
            else []
        )

        context = {
            "message": clean_msg,
            "session": session,
            "intent": intent,
            "history": history_turns,
            "memory": memories,
            "chunks": chunks,
            "kg": kg_facts,
            "tool": tool_result,
            "citations": citations,
            "reasoning": reasoning,
            "self_consistency": self_consistency,
            "cognition": cog_result,
            "cognition_addendum": cog_result.get("prompt_addendum", ""),
            "goals_context": cog_result.get("goals", ""),
            "retrieval_mode": retrieval_result.get("mode", "none"),
            "retrieval_hops": retrieval_result.get("hops", 1),
            "retrieval_score": retrieval_result.get("confidence", 0.0),
            "latency_build_ms": round((time.time() - t0) * 1000, 1),
        }
        return context

    # ── Tool dispatcher ───────────────────────────────────────────────
    def _run_tool(self, intent: str, message: str) -> dict | None:
        """Dispatch to the appropriate tool and return result dict."""
        try:
            if intent == "calculate":
                from tools.calculator import calculate

                expr = re.sub(
                    r"^(calc|calculate|compute|what\s+is)\s*",
                    "",
                    message,
                    flags=re.IGNORECASE,
                ).strip()
                result = calculate(expr)
                return {
                    "tool": "calculator",
                    "input": expr,
                    "result": str(result),
                    "success": True,
                }

            if intent == "code":
                from tools.code_exec import execute

                code = re.sub(
                    r"^(run|execute)\s+(this\s+)?(python|code|script)\s*",
                    "",
                    message,
                    flags=re.IGNORECASE,
                ).strip()
                result = execute(code)
                return {
                    "tool": "code_exec",
                    "input": code[:200],
                    "result": str(result)[:1000],
                    "success": True,
                }

            if intent == "search_files":
                from tools.file_search import search as file_search

                query = re.sub(
                    r"^(find|search)\s+(a\s+|my\s+|the\s+)?"
                    r"(file|document|pdf|doc)\s*",
                    "",
                    message,
                    flags=re.IGNORECASE,
                ).strip()
                result = file_search(query)
                return {
                    "tool": "file_search",
                    "input": query,
                    "result": str(result)[:1000],
                    "success": True,
                }
        except Exception as e:  # noqa: BLE001 - tool dispatch intentionally returns backend errors verbatim.
            return {"tool": intent, "error": str(e), "success": False}
        return None

    # ── Blocking process ──────────────────────────────────────────────
    def process(
        self,
        message: str,
        session: str = "default",
        ip: str = "127.0.0.1",
        active_doc: str | None = None,
    ) -> EngineResult:
        """Full blocking pipeline: build_context → generate → persist → log."""
        t0 = time.time()
        self._init()
        self._start_replay(
            message=message,
            session=session,
            mode="process",
            active_doc=active_doc,
        )

        context = self.build_context(message, session, ip, active_doc)

        # Blocked by guardrails
        if context is None:
            self._mark_replay_input(
                intent="blocked",
                short_circuit=False,
                blocked=True,
            )
            self._record_replay_error("input_guardrail")
            latency = round((time.time() - t0) * 1000, 1)
            replay_trace = self._consume_replay_trace(
                intent="blocked",
                response="Blocked: request did not pass input validation.",
                latency_ms=latency,
                build_ms=0.0,
                short_circuit=False,
            )
            return EngineResult(
                text="Blocked: request did not pass input validation.",
                session=session,
                intent="blocked",
                trace={
                    "replay": replay_trace,
                    "adaptation": self._build_adaptation_trace(replay_trace),
                },
                ok=False,
                error="input_guardrail",
            )

        # Short-circuit response (no model needed)
        if context.get("_short_circuit"):
            self._mark_replay_input(
                intent=context["intent"],
                short_circuit=True,
                blocked=False,
            )
            latency = round((time.time() - t0) * 1000, 1)
            replay_trace = self._consume_replay_trace(
                intent=context["intent"],
                response=context["response"],
                latency_ms=latency,
                build_ms=0.0,
                short_circuit=True,
            )
            return EngineResult(
                text=context["response"],
                session=session,
                intent=context["intent"],
                trace={
                    "pipeline": ["guardrails", "intent", "short_circuit"],
                    "replay": replay_trace,
                    "adaptation": self._build_adaptation_trace(replay_trace),
                },
                self_consistency={},
            )

        # Generate
        intent = context.get("intent", "chat")
        self._mark_replay_input(intent=intent, short_circuit=False, blocked=False)
        self._record_replay_context_warnings(context)
        if not self._generator:
            self._record_replay_error("generator_not_loaded")
            latency = round((time.time() - t0) * 1000, 1)
            replay_trace = self._consume_replay_trace(
                intent=intent,
                response="Generator not available.",
                latency_ms=latency,
                build_ms=context.get("latency_build_ms", 0.0),
                short_circuit=False,
            )
            return EngineResult(
                text="Generator not available.",
                session=session,
                intent=intent,
                trace={
                    "replay": replay_trace,
                    "adaptation": self._build_adaptation_trace(replay_trace),
                },
                ok=False,
                error="generator_not_loaded",
            )

        raw = self._generator.generate(message, context)

        # Output guardrails
        from safety.guardrails import filter_output

        text = filter_output(raw)
        pre_reflection_text = text

        # ── Reflection pass (single-shot, non-goal path only) ─────────
        reflection_report: dict = {}
        if REFLECTION.get("enabled", False) and self._reasoner:
            try:
                from reflection import Reflector

                reflector = Reflector(self._reasoner)
                revised_text, report = reflector.reflect(
                    draft_answer=text,
                    query=message,
                    chunks=context.get("chunks", []),
                    memory=context.get("memory", []),
                    kg_facts=context.get("kg", []),
                    tool=context.get("tool"),
                    cognition=context.get("cognition"),
                )
                if report.revised:
                    text = revised_text
                reflection_report = report.to_dict()
            except Exception as exc:  # noqa: BLE001 - reflection is optional and intentionally fail-open.
                reflection_report = {
                    "revised": False,
                    "reason": f"reflection_error: {exc}",
                }

        self._record_replay_reflection(
            reflection_report,
            draft_text=pre_reflection_text,
            final_text=text,
        )

        planning_trace = self._build_planning_trace(
            message=message,
            session=session,
            reflection=reflection_report,
        )
        self._record_replay_planning(planning_trace)

        self._maybe_store_learning_memories(
            message=message,
            session=session,
            draft_text=pre_reflection_text,
            final_text=text,
            reflection=reflection_report,
            self_consistency=context.get("self_consistency", {}),
        )

        # Persist exchange
        self._finish(message, text, context)

        # Build trace
        latency = round((time.time() - t0) * 1000, 1)
        trace = self._build_trace(context, text, latency)
        trace["planning"] = planning_trace
        replay_trace = self._consume_replay_trace(
            intent=intent,
            response=text,
            latency_ms=latency,
            build_ms=context.get("latency_build_ms", 0.0),
            short_circuit=False,
        )
        trace["replay"] = replay_trace
        trace["adaptation"] = self._build_adaptation_trace(replay_trace)

        # Record observability
        if self._obs:
            self._obs.record(
                {
                    "latency_ms": latency,
                    "tokens_in": len(message.split()),
                    "tokens_out": len(text.split()),
                    "intent": intent,
                    "backend": self._generator.backend,
                    "retrieval_mode": context.get("retrieval_mode", "none"),
                    "retrieval_score": context.get("retrieval_score", 0.0),
                    "retrieval_hops": context.get("retrieval_hops", 1),
                    "memory_hits": len(context.get("memory", [])),
                    "kg_facts": len(context.get("kg", [])),
                    "emotion": context.get("cognition", {})
                    .get("emotion", {})
                    .get("primary", ""),
                    "tool_name": context.get("tool", {}).get("tool", "")
                    if context.get("tool")
                    else "",
                    "tool_success": context.get("tool", {}).get("success", False)
                    if context.get("tool")
                    else False,
                }
            )

        return EngineResult(
            text=text,
            session=session,
            intent=intent,
            trace=trace,
            self_consistency=context.get("self_consistency", {}),
            reflection=reflection_report,
        )

    # ── Streaming process ─────────────────────────────────────────────
    def process_stream(
        self,
        message: str,
        session: str = "default",
        ip: str = "127.0.0.1",
        active_doc: str | None = None,
    ) -> Iterator[dict]:
        """Streaming pipeline. Yields SSE-compatible dicts.

        Each dict: {"fragment": str, "done": bool, "trace": dict|None}
        Final dict: {"fragment": "", "done": True, "trace": {...}}
        """
        self._init()
        t0 = time.time()
        self._start_replay(
            message=message,
            session=session,
            mode="stream",
            active_doc=active_doc,
        )

        context = self.build_context(message, session, ip, active_doc)

        if context is None:
            self._mark_replay_input(
                intent="blocked",
                short_circuit=False,
                blocked=True,
            )
            self._record_replay_error("input_guardrail")
            latency = round((time.time() - t0) * 1000, 1)
            replay_trace = self._consume_replay_trace(
                intent="blocked",
                response="Blocked: request did not pass input validation.",
                latency_ms=latency,
                build_ms=0.0,
                short_circuit=False,
            )
            yield {
                "fragment": "Blocked: request did not pass input validation.",
                "done": True,
                "trace": {
                    "replay": replay_trace,
                    "adaptation": self._build_adaptation_trace(replay_trace),
                },
            }
            return

        if context.get("_short_circuit"):
            self._mark_replay_input(
                intent=context["intent"],
                short_circuit=True,
                blocked=False,
            )
            latency = round((time.time() - t0) * 1000, 1)
            replay_trace = self._consume_replay_trace(
                intent=context["intent"],
                response=context["response"],
                latency_ms=latency,
                build_ms=0.0,
                short_circuit=True,
            )
            yield {
                "fragment": context["response"],
                "done": True,
                "trace": {
                    "intent": context["intent"],
                    "replay": replay_trace,
                    "adaptation": self._build_adaptation_trace(replay_trace),
                },
            }
            return

        intent = context.get("intent", "chat")
        self._mark_replay_input(intent=intent, short_circuit=False, blocked=False)
        self._record_replay_context_warnings(context)
        if not self._generator:
            self._record_replay_error("generator_not_loaded")
            latency = round((time.time() - t0) * 1000, 1)
            replay_trace = self._consume_replay_trace(
                intent=intent,
                response="Generator not available.",
                latency_ms=latency,
                build_ms=context.get("latency_build_ms", 0.0),
                short_circuit=False,
            )
            yield {
                "fragment": "Generator not available.",
                "done": True,
                "trace": {
                    "replay": replay_trace,
                    "adaptation": self._build_adaptation_trace(replay_trace),
                },
            }
            return

        from safety.guardrails import filter_output

        collected: list[str] = []

        for fragment in self._generator.generate_stream(message, context):
            if fragment == "":  # terminal sentinel
                break
            collected.append(fragment)
            yield {"fragment": fragment, "done": False, "trace": None}

        full_text = filter_output("".join(collected))
        planning_trace = self._build_planning_trace(
            message=message,
            session=session,
            reflection=None,
        )
        self._record_replay_planning(planning_trace)
        self._maybe_store_learning_memories(
            message=message,
            session=session,
            draft_text=full_text,
            final_text=full_text,
            reflection={},
            self_consistency=context.get("self_consistency", {}),
        )
        self._finish(message, full_text, context)

        latency = round((time.time() - t0) * 1000, 1)
        trace = self._build_trace(context, full_text, latency)
        trace["planning"] = planning_trace
        replay_trace = self._consume_replay_trace(
            intent=intent,
            response=full_text,
            latency_ms=latency,
            build_ms=context.get("latency_build_ms", 0.0),
            short_circuit=False,
        )
        trace["replay"] = replay_trace
        trace["adaptation"] = self._build_adaptation_trace(replay_trace)

        if self._obs:
            intent = context.get("intent", "chat")
            self._obs.record(
                {
                    "latency_ms": latency,
                    "tokens_in": len(message.split()),
                    "tokens_out": len(full_text.split()),
                    "intent": intent,
                    "backend": self._generator.backend,
                    "retrieval_mode": context.get("retrieval_mode", "none"),
                    "retrieval_score": context.get("retrieval_score", 0.0),
                }
            )

        yield {"fragment": "", "done": True, "trace": trace}

    def _start_replay(
        self,
        *,
        message: str,
        session: str,
        mode: str,
        active_doc: str | None,
    ) -> None:
        """Initialise a deterministic, read-only replay buffer for one request."""
        self._replay_state = {
            "mode": mode,
            "input": {
                "session": session,
                "message_chars": len(message),
                "message_words": len(message.split()),
                "active_doc": bool(active_doc),
                "intent": "",
                "blocked": False,
                "short_circuit": False,
            },
            "planning": {"executed": False},
            "reflection": {
                "executed": False,
                "revised": False,
                "reason": "",
                "changed_output": False,
            },
            "learning": {"generated": []},
            "memory_admission": {"events": []},
            "output": {"short_circuit": False},
            "warnings": [],
            "errors": [],
        }

    def _mark_replay_input(
        self,
        *,
        intent: str,
        short_circuit: bool,
        blocked: bool,
    ) -> None:
        """Record stable input-stage metadata for replay."""
        if self._replay_state is None:
            return
        self._replay_state["input"]["intent"] = intent
        self._replay_state["input"]["short_circuit"] = short_circuit
        self._replay_state["input"]["blocked"] = blocked

    def _record_replay_planning(self, planning_trace: dict) -> None:
        """Record planning-stage metadata without mutating planning."""
        if self._replay_state is None:
            return
        self._replay_state["planning"] = {
            "executed": True,
            "selected_goal": planning_trace.get("selected_goal"),
            "considered_goals": list(planning_trace.get("considered_goals", [])),
            "blocked_goals": list(planning_trace.get("blocked_goals", [])),
            "deferred_goals": list(planning_trace.get("deferred_goals", [])),
            "reflection": dict(planning_trace.get("reflection", {})),
        }

    def _record_replay_reflection(
        self,
        reflection_report: dict | None,
        *,
        draft_text: str,
        final_text: str,
    ) -> None:
        """Record reflection metadata without re-running reflection."""
        if self._replay_state is None:
            return
        report = reflection_report or {}
        reason = str(report.get("reason", ""))
        self._replay_state["reflection"] = {
            "executed": bool(report),
            "revised": bool(report.get("revised", False)),
            "reason": reason,
            "changed_output": draft_text != final_text,
        }
        if reason.startswith("reflection_error:"):
            self._record_replay_error(reason)

    def _record_replay_learning(self, summary: str, source: str, decision) -> None:
        """Record learning-stage metadata for generated summaries only."""
        if self._replay_state is None:
            return
        admitted = True if decision is None else bool(decision.admitted)
        reason = "policy_unavailable" if decision is None else str(decision.reason)
        self._replay_state["learning"]["generated"].append(
            {
                "source": source,
                "summary": summary,
                "admitted": admitted,
                "reason": reason,
                "stored": admitted,
            }
        )

    def _record_replay_memory_admission(self, kind: str, source: str, decision) -> None:
        """Record read-only admission metadata in call order."""
        if self._replay_state is None:
            return
        if decision is None:
            event = {
                "kind": kind,
                "source": source,
                "admitted": True,
                "reason": "policy_unavailable",
            }
        else:
            payload = decision.to_dict()
            event = {
                "kind": kind,
                "source": source,
                "admitted": bool(payload["admitted"]),
                "reason": str(payload["reason"]),
                "importance_score": float(payload["importance_score"]),
                "novelty_score": float(payload["novelty_score"]),
                "duplicate_score": float(payload["duplicate_score"]),
            }
        self._replay_state["memory_admission"]["events"].append(event)

    def _record_replay_warning(self, warning: str) -> None:
        """Append a stable warning once."""
        if self._replay_state is None:
            return
        text = str(warning).strip()
        if text and text not in self._replay_state["warnings"]:
            self._replay_state["warnings"].append(text)

    def _record_replay_error(self, error: str) -> None:
        """Append a stable error once."""
        if self._replay_state is None:
            return
        text = str(error).strip()
        if text and text not in self._replay_state["errors"]:
            self._replay_state["errors"].append(text)

    def _record_replay_context_warnings(self, context: dict) -> None:
        """Capture warning metadata already produced by existing subsystems."""
        warning_reason = str(
            context.get("self_consistency", {}).get("warning_reason", "")
        ).strip()
        if warning_reason:
            self._record_replay_warning(warning_reason)

    def _consume_replay_trace(
        self,
        *,
        intent: str,
        response: str,
        latency_ms: float,
        build_ms: float,
        short_circuit: bool,
    ) -> dict:
        """Build the final replay trace from recorded metadata only."""
        state = self._replay_state or {}
        input_stage = dict(state.get("input", {}))
        input_stage["intent"] = intent
        input_stage["short_circuit"] = short_circuit
        output_stage = {
            "short_circuit": short_circuit,
            "response_chars": len(response),
            "response_words": len(response.split()),
        }
        replay = {
            "mode": state.get("mode", "unknown"),
            "read_only": True,
            "stages": {
                "input": input_stage,
                "planning": dict(state.get("planning", {"executed": False})),
                "reflection": dict(
                    state.get(
                        "reflection",
                        {
                            "executed": False,
                            "revised": False,
                            "reason": "",
                            "changed_output": False,
                        },
                    )
                ),
                "learning": {
                    "generated": list(
                        state.get("learning", {}).get("generated", [])
                    )
                },
                "memory_admission": {
                    "events": list(
                        state.get("memory_admission", {}).get("events", [])
                    )
                },
                "output": output_stage,
            },
            "warnings": list(state.get("warnings", [])),
            "errors": list(state.get("errors", [])),
            "duration": {
                "total_ms": latency_ms,
                "build_ms": build_ms,
            },
        }
        self._replay_state = None
        return replay

    def _build_adaptation_trace(self, replay: dict | None) -> dict:
        """Compute deterministic adaptation metadata from replay only."""
        base = {
            "source": "replay",
            "read_only": True,
            "deterministic": True,
            "valid": False,
            "mode": "none",
            "classification": "unavailable",
            "signals": {
                "short_circuit": False,
                "reflection_revised": False,
                "learning_generated": 0,
                "memory_admission_rejections": 0,
                "warnings": 0,
                "errors": 0,
            },
        }
        if not replay:
            return {**base, "reason": "missing_replay"}
        if not isinstance(replay, dict):
            return {**base, "reason": "malformed_replay"}

        stages = replay.get("stages")
        warnings = replay.get("warnings")
        errors = replay.get("errors")
        if not isinstance(stages, dict):
            return {**base, "reason": "malformed_replay"}
        if not isinstance(warnings, list) or not isinstance(errors, list):
            return {**base, "reason": "malformed_replay"}

        input_stage = stages.get("input", {})
        reflection_stage = stages.get("reflection", {})
        learning_stage = stages.get("learning", {})
        admission_stage = stages.get("memory_admission", {})
        output_stage = stages.get("output", {})

        if not all(
            isinstance(stage, dict)
            for stage in (
                input_stage,
                reflection_stage,
                learning_stage,
                admission_stage,
                output_stage,
            )
        ):
            return {**base, "reason": "malformed_replay"}

        generated = learning_stage.get("generated", [])
        events = admission_stage.get("events", [])
        if not isinstance(generated, list) or not isinstance(events, list):
            return {**base, "reason": "malformed_replay"}

        rejection_count = sum(
            1
            for event in events
            if isinstance(event, dict) and not bool(event.get("admitted", True))
        )
        signals = {
            "short_circuit": bool(
                input_stage.get(
                    "short_circuit", output_stage.get("short_circuit", False)
                )
            ),
            "reflection_revised": bool(reflection_stage.get("revised", False)),
            "learning_generated": len(generated),
            "memory_admission_rejections": rejection_count,
            "warnings": len(warnings),
            "errors": len(errors),
        }
        if signals["errors"] > 0:
            classification = "error"
        elif signals["short_circuit"]:
            classification = "short_circuit"
        elif signals["memory_admission_rejections"] > 0:
            classification = "admission_rejection"
        elif signals["reflection_revised"]:
            classification = "reflection_revision"
        elif signals["learning_generated"] > 0:
            classification = "learning_signal"
        elif signals["warnings"] > 0:
            classification = "warning"
        else:
            classification = "stable"

        return {
            "source": "replay",
            "read_only": True,
            "deterministic": True,
            "valid": True,
            "mode": str(replay.get("mode", "unknown")),
            "classification": classification,
            "signals": signals,
        }

    # ── Post-generation persistence ───────────────────────────────────
    def _finish(self, message: str, response: str, context: dict):
        """Persist exchange to memory, extract KG triples, log event."""
        session = context.get("session", "default")

        if self._memory:
            try:
                self._memory.add_message(session, "user", message)
                self._memory.add_message(session, "assistant", response)
                # Auto-store high-information responses as episodic memory,
                # gated by the Memory Admission Policy.
                if len(response.split()) > 20:
                    snippet = response[:200]
                    candidate_text = f"Q: {message[:80]} A: {snippet}"
                    decision = self._admit_memory(
                        text=candidate_text,
                        kind="episodic",
                        importance_hint=0.7,
                        session=session,
                        source="exchange",
                    )
                    if decision is None or decision.admitted:
                        self._memory.remember_episodic(
                            candidate_text,
                            session=session,
                            importance=0.7,
                        )
            except Exception:  # noqa: BLE001, S110 - persistence failures must not alter the delivered response.
                pass

        if self._kg and len(response) > 50:
            try:
                added = self._kg.extract_from_text(
                    response, source=f"session:{session}"
                )
                if self._obs and added:
                    self._obs.record_kg_extract(len(added))
            except Exception:  # noqa: BLE001, S110 - post-response enrichment must not alter the delivered response.
                pass

    def _maybe_store_learning_memories(
        self,
        message: str,
        session: str,
        draft_text: str,
        final_text: str,
        reflection: dict | None,
        self_consistency: dict | None,
    ) -> None:
        """Store Learning v1 memories via the existing admission gate only."""
        if not self._memory:
            return

        reflection_summary = self._build_reflection_learning_summary(
            draft_text=draft_text,
            final_text=final_text,
            reflection=reflection,
        )
        if reflection_summary:
            self._store_learning_memory(
                summary=reflection_summary,
                session=session,
                source="reflection_learning",
            )

        self_consistency_summary = self._build_self_consistency_learning_summary(
            message=message,
            self_consistency=self_consistency,
        )
        if self_consistency_summary:
            self._store_learning_memory(
                summary=self_consistency_summary,
                session=session,
                source="self_consistency_learning",
            )

    def _store_learning_memory(self, summary: str, session: str, source: str) -> None:
        """Submit a learning summary through Memory Admission and store on admit."""
        if not self._memory or not summary:
            return
        decision = self._admit_memory(
            text=summary,
            kind="learning",
            importance_hint=1.1,
            session=session,
            source=source,
        )
        self._record_replay_learning(summary, source, decision)
        if decision is None or decision.admitted:
            self._memory.remember_learning(summary, importance=1.1)

    def _build_reflection_learning_summary(
        self,
        draft_text: str,
        final_text: str,
        reflection: dict | None,
    ) -> str | None:
        """Return a deterministic learning summary for reflection revisions."""
        report = reflection or {}
        if not report.get("revised", False):
            return None
        reason = str(report.get("reason", "revision")).strip() or "revision"
        before = self._summarize_learning_text(draft_text)
        after = self._summarize_learning_text(final_text)
        return (
            "Reflection revised answer: changed from "
            f"'{before}' to '{after}' because {reason}."
        )

    def _build_self_consistency_learning_summary(
        self,
        message: str,
        self_consistency: dict | None,
    ) -> str | None:
        """Return a learning summary only for strong high-confidence agreement."""
        data = self_consistency or {}
        if data.get("agreement_type") != "strong_agreement":
            return None
        confidence = float(data.get("confidence", 0.0) or 0.0)
        if confidence <= self._learning_confidence_threshold():
            return None
        consensus = data.get("consensus", {})
        confirmed = self._summarize_learning_text(
            str(consensus.get("answer", "")).strip() or message
        )
        return f"Self-consistency confirmed: {confirmed}."

    def _learning_confidence_threshold(self) -> float:
        """Configured confidence threshold for learning writes."""
        try:
            return float(VALIDATION.get("confidence_threshold", 0.70))
        except Exception:  # noqa: BLE001 - invalid config must fall back to default threshold.
            return 0.70

    @staticmethod
    def _summarize_learning_text(text: str, limit: int = 120) -> str:
        """Normalise whitespace and cap summaries deterministically."""
        compact = " ".join(str(text).split())
        if len(compact) <= limit:
            return compact
        return compact[: limit - 3].rstrip() + "..."

    # ── Trace builder ─────────────────────────────────────────────────
    # -- Memory admission helper (single write gate) --
    def _admit_memory(
        self, text: str, kind: str, importance_hint: float, session: str, source: str
    ):
        """Consult the admission policy for a proposed memory write.

        Returns the ``AdmissionDecision`` or ``None`` when the policy
        is disabled / uninitialised. Callers MUST treat ``None`` as
        "admit" so behaviour is unchanged when admission is off.
        Retrieval is not touched here.
        """
        if not self._admission_policy or not self._memory:
            self._record_replay_memory_admission(kind, source, None)
            return None
        try:
            from memory_admission.contracts import HistoryEntry, MemoryCandidate

            candidate = MemoryCandidate(
                text=text,
                kind=kind,
                importance_hint=float(importance_hint),
                session=session,
                source=source,
            )
            history_dicts = self._recent_history_for_admission(text)
            history_entries = [HistoryEntry(**h) for h in history_dicts]
            decision = self._admission_policy.decide(
                candidate,
                history=history_entries,
            )
            self._record_replay_memory_admission(kind, source, decision)
            if self._obs is not None:
                try:
                    self._obs.record(
                        {
                            "memory_admission": decision.to_dict(),
                            "session": session,
                            "source": source,
                        }
                    )
                except Exception:  # noqa: BLE001, S110 - observability failures must not block memory admission.
                    pass
            return decision
        except Exception:  # noqa: BLE001 - admission lookup is intentionally fail-open when policy helpers fail.
            return None

    def _recent_history_for_admission(self, text: str) -> list[dict]:
        """Return a newest-first slice of recent memories.

        This uses the EXISTING recall API and does not alter
        ranking or retrieval. Only ``text``, ``kind``, ``ts``,
        and ``importance`` are exposed to the policy.
        """
        if not self._memory:
            return []
        k = 25
        try:
            k = int(MEMORY_ADMISSION.get("history_recall_k", 25))
        except Exception:  # noqa: BLE001, S110 - invalid config must fall back to the default recall window.
            pass
        try:
            hits = self._memory.ranked_recall(query=text, k=k)
        except Exception:  # noqa: BLE001 - fallback to the legacy recall API preserves existing behavior.
            try:
                hits = self._memory.recall(text, k=k)
            except Exception:  # noqa: BLE001 - missing recall backends fall back to empty history.
                hits = []
        return [
            {
                "text": str(h.get("text", "")),
                "kind": str(h.get("kind", "fact")),
                "ts": float(h.get("ts", 0.0) or 0.0),
                "importance": float(h.get("importance", 1.0) or 1.0),
            }
            for h in hits or []
        ]

    def _build_planning_trace(
        self, message: str, session: str, reflection: dict | None
    ) -> dict:
        if not self._goal_store:
            return {
                "selected_goal": None,
                "considered_goals": [],
                "blocked_goals": [],
                "deferred_goals": [],
                "execution_order": [],
                "planning_reason": "no_goal_store",
                "confidence": 0.0,
                "warnings": [],
            }
        try:
            from planning import plan_turn

            goals = self._goal_store.list_session(session=session)
            return plan_turn(message, goals, reflection=reflection).to_dict()
        except Exception as exc:  # noqa: BLE001 - planning trace is intentionally fail-open.
            return {
                "selected_goal": None,
                "considered_goals": [],
                "blocked_goals": [],
                "deferred_goals": [],
                "execution_order": [],
                "planning_reason": "planning_error",
                "confidence": 0.0,
                "warnings": [str(exc)],
            }

    def _build_trace(self, context: dict, response: str, latency_ms: float) -> dict:
        """Build diagnostic trace dict for API response and UI panels."""
        pipeline = [
            "guardrails",
            "intent_router",
            "human_cognition",
            "memory",
            "retrieval",
            "knowledge_graph",
            "reasoning",
            "decoder",
            "output_filter",
        ]
        if context.get("tool"):
            pipeline.insert(4, f"tool:{context['tool'].get('tool', '')}")

        memory_snippets = [
            m.get("text", "")[:60] for m in context.get("memory", [])[:3]
        ]
        chunk_sources = list({c.get("source", "") for c in context.get("chunks", [])})
        kg_triples = [
            f"{f.get('subject', '')} {f.get('relation', '')} {f.get('object', '')}"
            for f in context.get("kg", [])[:5]
        ]
        reasoning_trace = context.get("reasoning", {}).get("trace", [])
        self_consistency = context.get("self_consistency", {})
        cognition_mode = context.get("cognition", {}).get("mode", "")
        emotion = (
            context.get("cognition", {}).get("emotion", {}).get("primary", "neutral")
        )
        intent_det = context.get("cognition", {}).get("intent", {}).get("intent", "")
        personality = (
            context.get("cognition", {}).get("personality", {}).get("mode", "")
        )

        return {
            "pipeline": pipeline,
            "intent": context.get("intent", ""),
            "backend": (self._generator.backend if self._generator else "unknown"),
            "retrieval_mode": context.get("retrieval_mode", "none"),
            "retrieval_hops": context.get("retrieval_hops", 1),
            "retrieval_conf": context.get("retrieval_score", 0.0),
            "citations": context.get("citations", ""),
            "memory_hits": memory_snippets,
            "chunk_sources": chunk_sources,
            "knowledge_graph": kg_triples,
            "reasoning_trace": reasoning_trace,
            "reasoning_conf": context.get("reasoning", {}).get("confidence", 0.0),
            "self_consistency": {
                "agreement": self_consistency.get("agreement", False),
                "agreement_type": self_consistency.get("agreement_type", ""),
                "warning_reason": self_consistency.get("warning_reason", ""),
                "confidence": self_consistency.get("confidence", 0.0),
            },
            "cognition_mode": cognition_mode,
            "emotion": emotion,
            "detected_intent": intent_det,
            "personality": personality,
            "latency_ms": latency_ms,
            "build_ms": context.get("latency_build_ms", 0.0),
        }

    # ── Convenience: session management ──────────────────────────────
    def clear_session(self, session: str):
        """Clear conversation history for a session."""
        if self._memory:
            try:
                self._memory._conn.execute(
                    "DELETE FROM messages WHERE session=?", (session,)
                )
                self._memory._conn.commit()
            except Exception:  # noqa: BLE001, S110 - clear-session failures are intentionally suppressed.
                pass

    def set_active_doc(self, path: str):
        """Open a document as the active CAG session."""
        self._init()
        if self._cag:
            self._cag.open(path, self._rag)

    def workspaces(self):
        """Return WorkspaceManager instance."""
        self._init()
        from retrieval.rag import WorkspaceManager

        return WorkspaceManager()
