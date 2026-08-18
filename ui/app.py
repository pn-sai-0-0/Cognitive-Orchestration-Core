"""
CognitiveOC v3 — Web UI Server
================================

Pure-Python HTTP server. No external web framework.
Standard library only: http.server, json, pathlib.

Endpoints:
  GET  /                        → index.html
  GET  /api/auth-key            → {key}
  GET  /api/status              → full system status
  GET  /api/metrics             → observability snapshot
  GET  /api/memories            → memory list
  GET  /api/kg                  → KG analytics
  GET  /api/retrieval-stats     → retrieval self-improvement stats
  GET  /api/workflows           → workflow list
  GET  /api/cognition-state     → current cognition mode/modules
  GET  /api/guardrail-state     → current guardrail profile/guards
  GET  /api/dataset/queue       → review queue
  POST /api/chat                → blocking chat
  POST /api/stream              → SSE streaming chat
  POST /api/upload              → document ingestion
  POST /api/feedback            → rating submission
  POST /api/ingest              → explicit file ingest
  POST /api/memory              → memory actions (search, forget, link)
  POST /api/kg                  → KG actions (query, merge, export)
  POST /api/workspace           → workspace actions
  POST /api/workflow            → workflow actions (create, status, cancel)
  POST /api/research            → research engine actions
  POST /api/validate            → validation actions
  POST /api/dataset             → dataset export actions
  POST /api/guardrail-profile   → switch guardrail profile
  POST /api/guardrail-toggle    → toggle one cognitive guard
  POST /api/cognition-mode      → switch cognition mode
  POST /api/cognition-toggle    → toggle one cognition module
  POST /api/eval                → run evaluation suite
  POST /api/retrieval-feedback  → record retrieval hit/miss

Authentication: X-CognitiveOC-Key header required for all POST endpoints
                and GET endpoints except /api/auth-key.

File: ui/app.py
"""

from __future__ import annotations

import json
import socketserver
import http.server
from pathlib import Path

try:
    from api.auth import check as auth_check, get_secret
except ImportError:
    def get_secret(): return "dev"
    def auth_check(h): return True

try:
    from config import UI, UPLOAD_DIR, ensure_dirs
except ImportError:
    UI = dict(host="127.0.0.1", port=8765)
    UPLOAD_DIR = Path("var/uploads")
    def ensure_dirs(): pass

_engine = None


def get_engine():
    global _engine
    if _engine is None:
        from engine import Engine
        _engine = Engine()
    return _engine


class Handler(http.server.BaseHTTPRequestHandler):

    # ── Suppress request logs (optional, comment out for debugging) ───
    def log_message(self, fmt, *args):
        pass

    # ── GET ──────────────────────────────────────────────────────────
    def do_GET(self):
        path = self.path.split("?")[0]

        # Public endpoints (no auth)
        if path in ("/", "/index.html"):
            html = (Path(__file__).parent / "static" / "index.html").read_bytes()
            self._raw(html, "text/html; charset=utf-8")
            return
        if path == "/api/auth-key":
            self._json({"key": get_secret()})
            return

        # Auth-required GET endpoints
        if not auth_check(self):
            self._json({"error": "Unauthorized"}, 401)
            return

        eng = get_engine()
        if path == "/api/status":
            self._json(eng.status())
        elif path == "/api/metrics":
            self._json(eng.observability.snapshot())
        elif path == "/api/memories":
            self._json(eng.memory.list_memories(50))
        elif path == "/api/kg":
            self._json(eng.kg.analytics())
        elif path == "/api/retrieval-stats":
            from retrieval.self_improve import stats
            self._json(stats())
        elif path == "/api/workflows":
            from workflow.workflow import WorkflowEngine
            self._json(WorkflowEngine().list_workflows(limit=20))
        elif path == "/api/cognition-state":
            from cognition.cognition import get_state
            self._json(get_state())
        elif path == "/api/guardrail-state":
            from safety.guardrails_state import get
            self._json(get())
        elif path == "/api/dataset/queue":
            from dataset.generator import DatasetGenerator
            self._json(DatasetGenerator().review_queue())
        elif path == "/api/validation/last":
            try:
                from config import EVAL_BASELINE
                p = Path(str(EVAL_BASELINE)) / "last_validation.json"
                self._json(json.loads(p.read_text()) if p.exists() else {})
            except Exception:
                self._json({})
        else:
            self.send_response(404)
            self.end_headers()

    # ── POST ─────────────────────────────────────────────────────────
    def do_POST(self):
        if self.path != "/api/auth-key" and not auth_check(self):
            self._json({"error": "Unauthorized — X-CognitiveOC-Key required"}, 401)
            return

        length = int(self.headers.get("Content-Length", 0))
        body   = json.loads(self.rfile.read(length) or b"{}")
        ip     = (self.client_address[0] if self.client_address else "127.0.0.1")
        p      = self.path
        eng    = get_engine()

        # ── Chat ──────────────────────────────────────────────────────
        if p == "/api/chat":
            msg = body.get("message", "")
            ses = body.get("session", "default")
            res = eng.process(msg, session=ses, ip=ip)
            self._json({"text": res.text, "trace": res.trace, "ok": res.ok})

        # ── Streaming chat ────────────────────────────────────────────
        elif p == "/api/stream":
            self._stream(body.get("message",""), session=body.get("session","default"), ip=ip)

        # ── Upload / ingest ───────────────────────────────────────────
        elif p in ("/api/upload", "/api/ingest"):
            file_path = body.get("path", body.get("file",""))
            if not file_path:
                self._json({"error": "path required"})
            else:
                self._json(eng.ingest(file_path))

        # ── Feedback ──────────────────────────────────────────────────
        elif p == "/api/feedback":
            try:
                from dataset.generator import LearningStore
                store = LearningStore()
                store.record_feedback(
                    question   = body.get("question",""),
                    answer     = body.get("answer",""),
                    rating     = int(body.get("rating",0)),
                    session    = body.get("session","default"),
                    confidence = body.get("confidence"),
                    emotion    = body.get("emotion",""),
                    intent     = body.get("intent",""),
                )
                # Also record retrieval feedback if provided
                if body.get("chunk_text"):
                    from retrieval.self_improve import record_hit
                    record_hit(body.get("question",""),
                               body.get("chunk_text",""),
                               body.get("chunk_source",""),
                               useful=int(body.get("rating",0)) > 0)
            except Exception as e:
                self._json({"ok": False, "error": str(e)}); return
            self._json({"ok": True})

        # ── Retrieval feedback ────────────────────────────────────────
        elif p == "/api/retrieval-feedback":
            from retrieval.self_improve import record_hit
            record_hit(body.get("query",""), body.get("chunk_text",""),
                       body.get("chunk_source",""), useful=bool(body.get("useful",True)))
            self._json({"ok": True})

        # ── Memory actions ────────────────────────────────────────────
        elif p == "/api/memory":
            action = body.get("action","")
            mem    = eng.memory
            if action == "search":
                self._json(mem.search(body.get("query",""), limit=20))
            elif action == "forget":
                mem.forget(int(body.get("id",0)))
                self._json({"ok": True})
            elif action == "link":
                mem.link_memories(int(body.get("id_a")), int(body.get("id_b")))
                self._json({"ok": True})
            elif action == "stats":
                self._json(mem.stats())
            elif action == "consolidate":
                n = mem.consolidate()
                self._json({"removed": n})
            elif action == "decay":
                n = mem.decay()
                self._json({"archived": n})
            else:
                self._json({"error": f"unknown action: {action}"})

        # ── KG actions ────────────────────────────────────────────────
        elif p == "/api/kg":
            action = body.get("action","")
            kg     = eng.kg
            if action == "query":
                self._json(kg.query(entity=body.get("entity"),
                                    relation=body.get("relation")))
            elif action == "search":
                self._json(kg.fts_search(body.get("query",""), limit=20))
            elif action == "analytics":
                self._json(kg.analytics())
            elif action == "contradictions":
                self._json(kg.contradictions())
            elif action == "neighbourhood":
                self._json(kg.neighbourhood(body.get("entity",""),
                                             hops=int(body.get("hops",2))))
            elif action == "merge":
                n = kg.merge_entities(body.get("canonical",""),
                                      body.get("alias",""))
                self._json({"updated": n})
            elif action == "export":
                path = kg.export_json()
                self._json({"path": path})
            elif action == "cluster":
                self._json(kg.cluster())
            else:
                self._json({"error": f"unknown action: {action}"})

        # ── Workspace ─────────────────────────────────────────────────
        elif p == "/api/workspace":
            action = body.get("action","")
            wm     = eng.workspaces()
            name   = body.get("name","default")
            if action == "list":
                self._json(wm.list_workspaces())
            elif action == "create":
                self._json(wm.create(name))
            elif action == "add_doc":
                self._json(wm.add_document(name, body.get("path","")))
            elif action == "search":
                self._json(wm.search(name, body.get("query",""),
                                     k=int(body.get("k",5))))
            else:
                self._json({"error": f"unknown action: {action}"})

        # ── Workflow ──────────────────────────────────────────────────
        elif p == "/api/workflow":
            action = body.get("action","")
            from workflow.workflow import WorkflowEngine
            wfe = WorkflowEngine()
            if action == "create":
                wf = wfe.create(body.get("goal",""),
                                session=body.get("session","default"))
                self._json(wf.summary())
            elif action == "run":
                import threading
                wf_id = body.get("id","")
                t = threading.Thread(target=wfe.run, args=(wf_id,), daemon=True)
                t.start()
                self._json({"ok": True, "id": wf_id, "status": "running"})
            elif action == "status":
                self._json(wfe.status(body.get("id","")) or {"error":"not found"})
            elif action == "list":
                self._json(wfe.list_workflows(limit=20))
            elif action == "cancel":
                wf = wfe.cancel(body.get("id",""))
                self._json(wf.summary() if wf else {"error":"not found"})
            elif action == "resume":
                import threading
                wf_id = body.get("id","")
                t = threading.Thread(target=wfe.resume, args=(wf_id,), daemon=True)
                t.start()
                self._json({"ok": True, "id": wf_id, "status": "resuming"})
            else:
                self._json({"error": f"unknown action: {action}"})

        # ── Research ──────────────────────────────────────────────────
        elif p == "/api/research":
            action = body.get("action","")
            from research.engine import ResearchEngine
            re_eng = ResearchEngine({"memory": eng.memory,
                                     "retriever": eng._hybrid,
                                     "kg": eng.kg})
            if action == "start":
                wf_id = re_eng.start_async(body.get("question",""),
                                           body.get("session","default"))
                self._json({"ok": True, "id": wf_id})
            elif action == "status":
                self._json(re_eng.status(body.get("id","")) or {"error":"not found"})
            elif action == "report":
                self._json({"report": re_eng.get_report(body.get("id",""))})
            elif action == "list":
                self._json(re_eng.list_research())
            else:
                self._json({"error": f"unknown action: {action}"})

        # ── Validation ────────────────────────────────────────────────
        elif p == "/api/validate":
            from validation.validator import Validator
            v = Validator()
            result = v.validate(
                response  = body.get("response",""),
                query     = body.get("query",""),
                chunks    = body.get("chunks",[]),
                memories  = body.get("memories",[]),
                kg        = eng.kg,
            )
            self._json(result.to_dict())

        # ── Dataset export ────────────────────────────────────────────
        elif p == "/api/dataset":
            action = body.get("action","")
            from dataset.generator import DatasetGenerator, LearningStore
            gen = DatasetGenerator(LearningStore())
            if action == "export_all":
                self._json(gen.export_all(eng.memory, eng.kg))
            elif action == "export_conversations":
                self._json(gen.export_conversations())
            elif action == "export_retrieval":
                self._json(gen.export_retrieval())
            elif action == "export_kg":
                self._json(gen.export_kg(eng.kg))
            elif action == "export_memory":
                self._json(gen.export_memory(eng.memory))
            elif action == "hard_examples":
                self._json(gen.hard_examples())
            elif action == "queue":
                self._json(gen.review_queue())
            elif action == "analytics":
                from dataset.generator import LearningStore as LS
                self._json(LS().analytics())
            else:
                self._json({"error": f"unknown action: {action}"})

        # ── Guardrail profile ─────────────────────────────────────────
        elif p == "/api/guardrail-profile":
            from safety.guardrails_state import set_profile
            try:
                self._json(set_profile(body.get("profile","standard")))
            except ValueError as e:
                self._json({"error": str(e)})

        # ── Guardrail toggle ──────────────────────────────────────────
        elif p == "/api/guardrail-toggle":
            from safety.guardrails_state import set_guard
            try:
                self._json(set_guard(body.get("guard",""),
                                     bool(body.get("enabled",True))))
            except ValueError as e:
                self._json({"error": str(e)})

        # ── Cognition mode ────────────────────────────────────────────
        elif p == "/api/cognition-mode":
            from cognition.cognition import set_mode
            try:
                self._json(set_mode(body.get("mode","full")))
            except ValueError as e:
                self._json({"error": str(e)})

        # ── Cognition module toggle ───────────────────────────────────
        elif p == "/api/cognition-toggle":
            from cognition.cognition import set_module
            try:
                self._json(set_module(body.get("module",""),
                                      bool(body.get("enabled",True))))
            except ValueError as e:
                self._json({"error": str(e)})

        # ── Evaluation ────────────────────────────────────────────────
        elif p == "/api/eval":
            action = body.get("action","")
            if action == "run_all":
                import threading
                corpus = body.get("corpus","data/corpus/v1/split/val.txt")
                def _run():
                    from eval.run_suite import run_all
                    run_all(corpus, save=True, verbose=False)
                threading.Thread(target=_run, daemon=True).start()
                self._json({"ok": True, "status": "running_in_background"})
            elif action == "tokenizer":
                from eval.run_suite import eval_tokenizer
                self._json(eval_tokenizer())
            elif action == "gate":
                from eval.run_suite import check_gate
                self._json(check_gate(int(body.get("phase",0))))
            else:
                self._json({"error": f"unknown eval action: {action}"})

        # ── Self-improving retrieval ───────────────────────────────────
        elif p == "/api/retrieval-improve":
            action = body.get("action","")
            if action == "stats":
                from retrieval.self_improve import stats
                self._json(stats())
            elif action == "hard_examples":
                from retrieval.self_improve import mine_hard_examples
                self._json(mine_hard_examples())
            elif action == "export_rewrites":
                from retrieval.self_improve import export_query_rewrites
                self._json(export_query_rewrites())
            elif action == "export_reranker":
                from retrieval.self_improve import export_reranker_examples
                self._json(export_reranker_examples())
            else:
                self._json({"error": f"unknown action: {action}"})

        else:
            self.send_response(404)
            self.end_headers()

    # ── SSE streaming ─────────────────────────────────────────────────
    def _stream(self, message: str, session: str = "default", ip: str = "127.0.0.1"):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        eng = get_engine()
        try:
            for event in eng.process_stream(message, session=session, ip=ip):
                data = json.dumps(event)
                self.wfile.write(f"data: {data}\n\n".encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    # ── Response helpers ──────────────────────────────────────────────
    def _json(self, payload: dict, code: int = 200):
        body = json.dumps(payload, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _raw(self, body: bytes, content_type: str, code: int = 200):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class ThreadedServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


def serve(host: str = None, port: int = None):
    """Start the COC v3 web server."""
    ensure_dirs()
    h = host or UI.get("host", "127.0.0.1")
    p = port or UI.get("port", 8765)

    # Pre-warm engine so first request is not slow
    print("[ui] Initialising engine…")
    get_engine()
    print("[ui] Engine ready.")

    # Print auth key on startup
    key = get_secret()
    print(f"[ui] CognitiveOC v3 web server")
    print(f"[ui] URL: http://{h}:{p}")
    print(f"[ui] Auth key: {key}")

    with ThreadedServer((h, p), Handler) as server:
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\n[ui] Stopped.")
