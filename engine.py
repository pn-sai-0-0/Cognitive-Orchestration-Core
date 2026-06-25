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
from dataclasses import dataclass, field
from typing import Any, Iterator

try:
    from config import (MODEL, INFERENCE, MEMORY, RETRIEVAL,
                        KNOWLEDGE_GRAPH, COGNITION, GUARDRAILS,
                        ensure_dirs)
except ImportError:
    MODEL = {}; INFERENCE = {}; MEMORY = {}; RETRIEVAL = {}
    KNOWLEDGE_GRAPH = {}; COGNITION = {}; GUARDRAILS = {}
    def ensure_dirs(): pass


# ═══════════════════════════════════════════════════════════════════
# Result dataclass
# ═══════════════════════════════════════════════════════════════════

@dataclass
class EngineResult:
    """Structured result from engine.process()."""
    text:    str
    session: str
    intent:  str
    trace:   dict = field(default_factory=dict)
    ok:      bool = True
    error:   str  = ""


# ═══════════════════════════════════════════════════════════════════
# Intent patterns (15 patterns, matched in order)
# ═══════════════════════════════════════════════════════════════════

_INTENT_PATTERNS = [
    ("calculate",     re.compile(
        r"^(calc|calculate|compute|what\s+is\s+\d|evaluate)\b", re.I)),
    ("code",          re.compile(
        r"\b(run|execute)\s+(this\s+)?(python|code|script)\b", re.I)),
    ("remember",      re.compile(
        r"^(remember|note\s+that|save\s+this|memo|don'?t\s+forget)\b", re.I)),
    ("recall_profile",re.compile(
        r"\b(what\s+do\s+you\s+know\s+about\s+me|my\s+(preferences?|profile"
        r"|settings?|style))\b", re.I)),
    ("search_files",  re.compile(
        r"\b(find|search)\s+(a\s+|my\s+|the\s+)?(file|document|pdf|doc)\b", re.I)),
    ("kg_query",      re.compile(
        r"^(who|what|where|when)\s+.{3,40}\?$", re.I)),
    ("workflow",      re.compile(
        r"^(run\s+workflow|execute\s+task|agent:|start\s+task)\b", re.I)),
    ("ingest",        re.compile(
        r"\b(ingest|index|upload|add\s+document|load\s+file)\b", re.I)),
    ("goal_add",      re.compile(
        r"\b(add\s+goal|set\s+goal|new\s+goal|create\s+goal)\b", re.I)),
    ("goal_list",     re.compile(
        r"\b(show\s+(my\s+)?goals?|list\s+goals?|what\s+are\s+my\s+goals?)\b",
        re.I)),
    ("memory_list",   re.compile(
        r"\b(show\s+(my\s+)?memories?|list\s+memories?|what\s+do\s+you\s+remember)\b",
        re.I)),
    ("eval",          re.compile(
        r"^(run\s+eval|evaluate\s+model|run\s+evaluation)\b", re.I)),
    ("metrics",       re.compile(
        r"^(show\s+metrics?|system\s+status|observability)\b", re.I)),
    ("teach",         re.compile(
        r"\b(teach\s+me|explain\s+step\s+by\s+step|quiz\s+me|lesson)\b", re.I)),
    ("chat",          re.compile(r".*", re.I)),   # catch-all
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
        self._memory      = None
        self._rag         = None
        self._cag         = None
        self._hybrid      = None
        self._kg          = None
        self._reasoner    = None
        self._cognition   = None
        self._generator   = None
        self._obs         = None
        self._guardrails  = None
        self._tools       = None
        self._init_done   = False

    # ── Lazy initialisation ───────────────────────────────────────────
    def _init(self):
        if self._init_done:
            return
        try:
            from memory.memory      import CognitiveMemory
            from retrieval.rag      import RAGPipeline, CAGManager, HybridRetriever
            from knowledge.graph    import KnowledgeGraph
            from reasoning.reasoner import Reasoner
            from cognition.cognition import CognitionLayer
            from inference.generator import ResponseGenerator
            from observability.metrics import Observability

            self._memory    = CognitiveMemory()
            self._rag       = RAGPipeline()
            self._cag       = CAGManager()
            self._kg        = KnowledgeGraph()
            self._hybrid    = HybridRetriever(
                rag=self._rag, cag=self._cag,
                memory=self._memory, kg=self._kg
            )
            self._reasoner  = Reasoner()
            self._cognition = CognitionLayer()
            self._generator = ResponseGenerator()
            self._obs       = Observability()
            self._init_done = True

        except Exception as e:
            print(f"[engine] Init warning: {e}")
            self._init_done = True

    # ── Public properties (for ui/app.py access) ──────────────────────
    @property
    def rag(self):
        self._init(); return self._rag

    @property
    def memory(self):
        self._init(); return self._memory

    @property
    def kg(self):
        self._init(); return self._kg

    @property
    def observability(self):
        self._init(); return self._obs

    # ── Status ────────────────────────────────────────────────────────
    def status(self) -> dict:
        """Full system status dict (used by /api/status)."""
        self._init()
        gen_backend = (self._generator.backend
                       if self._generator else "not loaded")
        mem_stats   = self._memory.stats() if self._memory else {}
        rag_stats   = self._rag.stats()    if self._rag    else {}
        kg_stats    = {}
        if self._kg:
            kg_stats = {"triples": len(self._kg),
                        "entities": self._kg._conn.execute(
                            "SELECT COUNT(*) FROM entities").fetchone()[0]}
        obs_snap = self._obs.snapshot() if self._obs else {}

        from safety.guardrails_state import get as gs_get
        from cognition.cognition import get_state as cog_state

        return {
            "version":    "3.0",
            "backend":    gen_backend,
            "memory":     mem_stats,
            "retrieval":  rag_stats,
            "kg":         kg_stats,
            "guardrails": gs_get(),
            "cognition":  cog_state(),
            "metrics":    obs_snap.get("requests", {}),
            "hardware":   obs_snap.get("hardware", {}),
            "training":   obs_snap.get("training", {}),
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
                text  = Path(path).read_text(encoding="utf-8", errors="replace")[:8000]
                added = self._kg.extract_from_text(text, source=path)
                result["kg_triples"] = len(added)
                if self._obs:
                    self._obs.record_kg_extract(len(added))
            except Exception:
                pass
        return result

    # ── Context assembly (full pipeline without generation) ───────────
    def build_context(self,
                      message:    str,
                      session:    str = "default",
                      ip:         str = "127.0.0.1",
                      active_doc: str = None) -> dict | None:
        """Run full pre-generation pipeline. Returns context dict or None on block."""
        self._init()

        t0 = time.time()

        # ── Step 1: Input guardrails ───────────────────────────────────
        from safety.guardrails import check_input, sanitise_chunks
        ok, reason, clean_msg = check_input(message, ip)
        if not ok:
            return None  # blocked

        # ── Step 2: Intent routing ─────────────────────────────────────
        intent = _route(clean_msg)

        # ── Step 3: Short-circuit intents (no model needed) ───────────
        if intent == "remember":
            text = re.sub(r"^(remember|note\s+that|save\s+this|memo)\s*[:\-]?\s*",
                          "", clean_msg, flags=re.I).strip()
            if text and self._memory:
                from safety.guardrails import check_memory_write
                ok_w, _ = check_memory_write(text)
                if ok_w:
                    self._memory.remember_auto(text, session=session)
            return {"_short_circuit": True, "intent": intent,
                    "response": f"Remembered: {text[:80]}"}

        if intent == "recall_profile":
            profile = self._memory.profile() if self._memory else {}
            summary = self._memory.get_user_summary() if hasattr(
                self._memory, "get_user_summary") else {}
            resp = "Here's what I know about you:\n"
            if profile:
                resp += "\n".join(f"- {k}: {v}" for k, v in list(profile.items())[:10])
            return {"_short_circuit": True, "intent": intent, "response": resp}

        if intent == "goal_list":
            goals = (self._cognition.active_goals()
                     if self._cognition else [])
            resp  = ("Active goals:\n" +
                     "\n".join(f"- {g['title']} ({int(g['progress']*100)}%)"
                               for g in goals)
                     ) if goals else "No active goals."
            return {"_short_circuit": True, "intent": intent, "response": resp}

        if intent == "goal_add":
            title = re.sub(
                r"^(add|set|new|create)\s+goal\s*[:\-]?\s*", "",
                clean_msg, flags=re.I
            ).strip()
            if title and self._cognition:
                gid = self._cognition.add_goal(title)
                return {"_short_circuit": True, "intent": intent,
                        "response": f"Goal added (id={gid}): {title}"}

        if intent == "memory_list":
            mems = self._memory.list_memories(limit=10) if self._memory else []
            resp = ("Recent memories:\n" +
                    "\n".join(f"- [{m['kind']}] {m['text'][:80]}" for m in mems)
                    ) if mems else "No memories stored."
            return {"_short_circuit": True, "intent": intent, "response": resp}

        if intent == "metrics":
            snap = self._obs.snapshot() if self._obs else {}
            import json
            return {"_short_circuit": True, "intent": intent,
                    "response": json.dumps(snap, indent=2, default=str)[:2000]}

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
            except Exception:
                pass

        # ── Step 6: Memory → KG enrichment ────────────────────────────
        if memories and self._kg:
            try:
                from memory.summarizer import enrich_recall
                memories = enrich_recall(memories, self._kg)
            except Exception:
                pass

        # ── Step 7: Hybrid retrieval ───────────────────────────────────
        retrieval_result: dict = {}
        if self._hybrid:
            try:
                history = (self._memory.recent(session,
                           MEMORY.get("session_window", 30))
                           if self._memory else [])
                retrieval_result = self._hybrid.retrieve(
                    query      = clean_msg,
                    session    = session,
                    history    = history,
                    active_doc = active_doc,
                    k          = RETRIEVAL.get("final_k", 5),
                    rerank     = True,
                )
            except Exception:
                pass

        chunks    = retrieval_result.get("chunks", [])
        citations = retrieval_result.get("citations", "")

        # Sanitise retrieved chunks
        from safety.guardrails import sanitise_chunks
        chunks = sanitise_chunks(chunks)

        # ── Step 8: KG query ───────────────────────────────────────────
        kg_facts: list[dict] = []
        if self._kg:
            try:
                kg_facts = self._kg.ranked_query(clean_msg, limit=10)
            except Exception:
                pass

        # ── Step 9: Human Cognition Layer ─────────────────────────────
        cog_result: dict = {}
        if self._cognition:
            try:
                history = (self._memory.recent(session, 10)
                           if self._memory else [])
                cog_result = self._cognition.process(
                    text      = clean_msg,
                    session   = session,
                    history   = history,
                    chunks    = chunks,
                    memories  = memories,
                    kg_facts  = [f.get("subject","")+" "+f.get("relation","")
                                 +" "+f.get("object","") for f in kg_facts],
                )
            except Exception:
                pass

        # ── Step 10: Reasoning ─────────────────────────────────────────
        reasoning: dict = {}
        if self._reasoner:
            try:
                reasoning = self._reasoner.assess(
                    query    = clean_msg,
                    memory   = memories,
                    chunks   = chunks,
                    kg_facts = kg_facts,
                    tool     = tool_result,
                    cognition= cog_result,
                )
            except Exception:
                pass

        # ── Step 11: Session history ───────────────────────────────────
        history_turns = (self._memory.recent(session,
                         MEMORY.get("session_window", 30))
                         if self._memory else [])

        context = {
            "message":           clean_msg,
            "session":           session,
            "intent":            intent,
            "history":           history_turns,
            "memory":            memories,
            "chunks":            chunks,
            "kg":                kg_facts,
            "tool":              tool_result,
            "citations":         citations,
            "reasoning":         reasoning,
            "cognition":         cog_result,
            "cognition_addendum":cog_result.get("prompt_addendum", ""),
            "goals_context":     cog_result.get("goals", ""),
            "retrieval_mode":    retrieval_result.get("mode", "none"),
            "retrieval_hops":    retrieval_result.get("hops", 1),
            "retrieval_score":   retrieval_result.get("confidence", 0.0),
            "latency_build_ms":  round((time.time() - t0) * 1000, 1),
        }
        return context

    # ── Tool dispatcher ───────────────────────────────────────────────
    def _run_tool(self, intent: str, message: str) -> dict | None:
        """Dispatch to the appropriate tool and return result dict."""
        try:
            if intent == "calculate":
                from tools.calculator import calculate
                expr   = re.sub(r"^(calc|calculate|compute|what\s+is)\s*",
                                "", message, flags=re.I).strip()
                result = calculate(expr)
                return {"tool": "calculator", "input": expr,
                        "result": str(result), "success": True}

            if intent == "code":
                from tools.code_exec import execute
                code   = re.sub(r"^(run|execute)\s+(this\s+)?(python|code|script)\s*",
                                "", message, flags=re.I).strip()
                result = execute(code)
                return {"tool": "code_exec", "input": code[:200],
                        "result": str(result)[:1000], "success": True}

            if intent == "search_files":
                from tools.file_search import search as file_search
                query  = re.sub(r"^(find|search)\s+(a\s+|my\s+|the\s+)?(file|document|pdf|doc)\s*",
                                "", message, flags=re.I).strip()
                result = file_search(query)
                return {"tool": "file_search", "input": query,
                        "result": str(result)[:1000], "success": True}
        except Exception as e:
            return {"tool": intent, "error": str(e), "success": False}
        return None

    # ── Blocking process ──────────────────────────────────────────────
    def process(self,
                message:    str,
                session:    str = "default",
                ip:         str = "127.0.0.1",
                active_doc: str = None) -> EngineResult:
        """Full blocking pipeline: build_context → generate → persist → log."""
        t0 = time.time()
        self._init()

        context = self.build_context(message, session, ip, active_doc)

        # Blocked by guardrails
        if context is None:
            return EngineResult(
                text    = "Blocked: request did not pass input validation.",
                session = session,
                intent  = "blocked",
                ok      = False,
                error   = "input_guardrail",
            )

        # Short-circuit response (no model needed)
        if context.get("_short_circuit"):
            return EngineResult(
                text    = context["response"],
                session = session,
                intent  = context["intent"],
                trace   = {"pipeline": ["guardrails","intent","short_circuit"]},
            )

        # Generate
        intent = context.get("intent", "chat")
        if not self._generator:
            return EngineResult(
                text    = "Generator not available.",
                session = session,
                intent  = intent,
                ok      = False,
                error   = "generator_not_loaded",
            )

        raw = self._generator.generate(message, context)

        # Output guardrails
        from safety.guardrails import filter_output
        text = filter_output(raw)

        # Persist exchange
        self._finish(message, text, context)

        # Build trace
        latency = round((time.time() - t0) * 1000, 1)
        trace   = self._build_trace(context, text, latency)

        # Record observability
        if self._obs:
            self._obs.record({
                "latency_ms":         latency,
                "tokens_in":          len(message.split()),
                "tokens_out":         len(text.split()),
                "intent":             intent,
                "backend":            self._generator.backend,
                "retrieval_mode":     context.get("retrieval_mode", "none"),
                "retrieval_score":    context.get("retrieval_score", 0.0),
                "retrieval_hops":     context.get("retrieval_hops", 1),
                "memory_hits":        len(context.get("memory", [])),
                "kg_facts":           len(context.get("kg", [])),
                "emotion":            context.get("cognition",{}).get(
                    "emotion",{}).get("primary",""),
                "tool_name":          context.get("tool",{}).get("tool","")
                                      if context.get("tool") else "",
                "tool_success":       context.get("tool",{}).get("success",False)
                                      if context.get("tool") else False,
            })

        return EngineResult(
            text=text, session=session, intent=intent, trace=trace
        )

    # ── Streaming process ─────────────────────────────────────────────
    def process_stream(self,
                       message:    str,
                       session:    str = "default",
                       ip:         str = "127.0.0.1",
                       active_doc: str = None) -> Iterator[dict]:
        """Streaming pipeline. Yields SSE-compatible dicts.

        Each dict: {"fragment": str, "done": bool, "trace": dict|None}
        Final dict: {"fragment": "", "done": True, "trace": {...}}
        """
        self._init()
        t0 = time.time()

        context = self.build_context(message, session, ip, active_doc)

        if context is None:
            yield {"fragment": "Blocked: request did not pass input validation.",
                   "done": True, "trace": None}
            return

        if context.get("_short_circuit"):
            yield {"fragment": context["response"], "done": True,
                   "trace": {"intent": context["intent"]}}
            return

        if not self._generator:
            yield {"fragment": "Generator not available.", "done": True, "trace": None}
            return

        from safety.guardrails import filter_output
        collected: list[str] = []

        for fragment in self._generator.generate_stream(message, context):
            if fragment == "":   # terminal sentinel
                break
            collected.append(fragment)
            yield {"fragment": fragment, "done": False, "trace": None}

        full_text = filter_output("".join(collected))
        self._finish(message, full_text, context)

        latency = round((time.time() - t0) * 1000, 1)
        trace   = self._build_trace(context, full_text, latency)

        if self._obs:
            intent = context.get("intent", "chat")
            self._obs.record({
                "latency_ms":     latency,
                "tokens_in":      len(message.split()),
                "tokens_out":     len(full_text.split()),
                "intent":         intent,
                "backend":        self._generator.backend,
                "retrieval_mode": context.get("retrieval_mode", "none"),
                "retrieval_score":context.get("retrieval_score", 0.0),
            })

        yield {"fragment": "", "done": True, "trace": trace}

    # ── Post-generation persistence ───────────────────────────────────
    def _finish(self, message: str, response: str, context: dict):
        """Persist exchange to memory, extract KG triples, log event."""
        session = context.get("session", "default")

        if self._memory:
            try:
                self._memory.add_message(session, "user",      message)
                self._memory.add_message(session, "assistant", response)
                # Auto-store high-information responses as episodic memory
                if len(response.split()) > 20:
                    snippet = response[:200]
                    self._memory.remember_episodic(
                        f"Q: {message[:80]} A: {snippet}",
                        session=session, importance=0.7
                    )
            except Exception:
                pass

        if self._kg and len(response) > 50:
            try:
                added = self._kg.extract_from_text(
                    response, source=f"session:{session}"
                )
                if self._obs and added:
                    self._obs.record_kg_extract(len(added))
            except Exception:
                pass

    # ── Trace builder ─────────────────────────────────────────────────
    def _build_trace(self, context: dict, response: str,
                     latency_ms: float) -> dict:
        """Build diagnostic trace dict for API response and UI panels."""
        pipeline = [
            "guardrails", "intent_router",
            "human_cognition", "memory", "retrieval",
            "knowledge_graph", "reasoning", "decoder", "output_filter",
        ]
        if context.get("tool"):
            pipeline.insert(4, f"tool:{context['tool'].get('tool','')}")

        memory_snippets = [
            m.get("text","")[:60] for m in context.get("memory",[])[:3]
        ]
        chunk_sources   = list({
            c.get("source","") for c in context.get("chunks",[])
        })
        kg_triples      = [
            f"{f.get('subject','')} {f.get('relation','')} {f.get('object','')}"
            for f in context.get("kg",[])[:5]
        ]
        reasoning_trace = context.get("reasoning",{}).get("trace",[])
        cognition_mode  = context.get("cognition",{}).get("mode","")
        emotion         = context.get("cognition",{}).get(
            "emotion",{}).get("primary","neutral")
        intent_det      = context.get("cognition",{}).get(
            "intent",{}).get("intent","")
        personality     = context.get("cognition",{}).get(
            "personality",{}).get("mode","")

        return {
            "pipeline":         pipeline,
            "intent":           context.get("intent",""),
            "backend":          (self._generator.backend
                                 if self._generator else "unknown"),
            "retrieval_mode":   context.get("retrieval_mode","none"),
            "retrieval_hops":   context.get("retrieval_hops",1),
            "retrieval_conf":   context.get("retrieval_score",0.0),
            "citations":        context.get("citations",""),
            "memory_hits":      memory_snippets,
            "chunk_sources":    chunk_sources,
            "knowledge_graph":  kg_triples,
            "reasoning_trace":  reasoning_trace,
            "reasoning_conf":   context.get("reasoning",{}).get("confidence",0.0),
            "cognition_mode":   cognition_mode,
            "emotion":          emotion,
            "detected_intent":  intent_det,
            "personality":      personality,
            "latency_ms":       latency_ms,
            "build_ms":         context.get("latency_build_ms",0.0),
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
            except Exception:
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
