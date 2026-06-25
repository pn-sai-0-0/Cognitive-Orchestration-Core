"""
CognitiveOC v3 — Native PySide6 Desktop Application
======================================================

Tier-1 primary interface. Calls Python backend directly — no HTTP layer,
no QWebEngineView wrapper. All panels use native Qt widgets.

Architecture:
  QMainWindow
  ├── MenuBar          — File / System / View / Help
  ├── ToolBar          — quick-action buttons
  ├── QSplitter (H)
  │   ├── Left Panel   — Chat  (QTextEdit + QLineEdit)
  │   └── Right Tabs   — Memory | KG | Workflow | Guardrails | Cognition
  │                      Metrics | Eval | Dataset | Training
  └── StatusBar        — backend, mode, guardrail profile, message count

Direct backend calls (no HTTP):
  engine.process()         → chat response
  engine.process_stream()  → streaming chat (token-by-token)
  engine.status()          → system status
  engine.ingest()          → document ingestion
  engine.memory.*          → memory browser
  engine.kg.*              → KG panel
  guardrails_state.get()   → guardrail controls
  cognition.get_state()    → cognition controls

Requires: pip install PySide6
Does NOT require: PySide6-WebEngine, browser, HTTP server

File: ui/desktop.py
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from typing import Any


# ═══════════════════════════════════════════════════════════════════
# Qt availability guard
# ═══════════════════════════════════════════════════════════════════

def _qt_available() -> bool:
    try:
        from PySide6.QtWidgets import QApplication
        return True
    except ImportError:
        return False


# ═══════════════════════════════════════════════════════════════════
# Worker thread for non-blocking backend calls
# ═══════════════════════════════════════════════════════════════════

if _qt_available():
    from PySide6.QtCore import QThread, Signal, QObject

    class Worker(QThread):
        """Runs a backend call in a background thread; emits result signal."""
        result  = Signal(object)
        error   = Signal(str)

        def __init__(self, fn, *args, **kwargs):
            super().__init__()
            self._fn   = fn
            self._args = args
            self._kw   = kwargs

        def run(self):
            try:
                r = self._fn(*self._args, **self._kw)
                self.result.emit(r)
            except Exception as e:
                self.error.emit(str(e))

    class StreamWorker(QThread):
        """Runs streaming generation; emits fragment signals."""
        fragment = Signal(str)
        done     = Signal(dict)

        def __init__(self, engine, message, session):
            super().__init__()
            self._engine  = engine
            self._message = message
            self._session = session

        def run(self):
            trace = {}
            for event in self._engine.process_stream(self._message,
                                                      session=self._session):
                if event.get("fragment"):
                    self.fragment.emit(event["fragment"])
                if event.get("done"):
                    trace = event.get("trace", {})
                    break
            self.done.emit(trace)


# ═══════════════════════════════════════════════════════════════════
# Main Application Window
# ═══════════════════════════════════════════════════════════════════

def build_main_window(engine):
    """Build and return the QMainWindow instance."""
    from PySide6.QtWidgets import (
        QMainWindow, QWidget, QSplitter, QTabWidget,
        QVBoxLayout, QHBoxLayout, QTextEdit, QLineEdit,
        QPushButton, QLabel, QListWidget, QListWidgetItem,
        QComboBox, QCheckBox, QGroupBox, QScrollArea,
        QStatusBar, QToolBar, QMenuBar, QMenu, QFileDialog,
        QMessageBox, QFrame, QSizePolicy, QProgressBar,
        QGridLayout, QFormLayout, QSpinBox, QTableWidget,
        QTableWidgetItem, QHeaderView, QDialog, QDialogButtonBox,
    )
    from PySide6.QtCore import Qt, QTimer, Signal, Slot
    from PySide6.QtGui  import (
        QAction, QColor, QPalette, QFont, QIcon, QTextCursor,
    )

    # ── Dark palette ────────────────────────────────────────────────
    def _apply_dark(app):
        pal = QPalette()
        pal.setColor(QPalette.Window,          QColor(15,  17,  23))
        pal.setColor(QPalette.WindowText,      QColor(226, 232, 240))
        pal.setColor(QPalette.Base,            QColor(26,  29,  39))
        pal.setColor(QPalette.AlternateBase,   QColor(34,  38,  58))
        pal.setColor(QPalette.Text,            QColor(226, 232, 240))
        pal.setColor(QPalette.Button,          QColor(34,  38,  58))
        pal.setColor(QPalette.ButtonText,      QColor(226, 232, 240))
        pal.setColor(QPalette.Highlight,       QColor(108, 141, 250))
        pal.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
        pal.setColor(QPalette.ToolTipBase,     QColor(26,  29,  39))
        pal.setColor(QPalette.ToolTipText,     QColor(226, 232, 240))
        pal.setColor(QPalette.PlaceholderText, QColor(100, 116, 139))
        from PySide6.QtWidgets import QApplication
        QApplication.instance().setPalette(pal)

    class CognitiveOCDesktop(QMainWindow):
        def __init__(self, eng):
            super().__init__()
            self._engine  = eng
            self._session = f"desktop_{int(time.time())}"
            self._workers: list[QThread] = []   # keep references alive

            self.setWindowTitle("CognitiveOC v3 — Native Desktop")
            self.setMinimumSize(1280, 780)

            _apply_dark(self)
            self._build_menu()
            self._build_toolbar()
            self._build_central()
            self._build_statusbar()

            # Auto-refresh timer (metrics + status)
            self._refresh_timer = QTimer(self)
            self._refresh_timer.timeout.connect(self._refresh_status)
            self._refresh_timer.start(8000)
            self._refresh_status()

        # ── Menu bar ───────────────────────────────────────────────
        def _build_menu(self):
            mb = self.menuBar()

            # File
            m = mb.addMenu("File")
            a_ingest = QAction("Ingest Document…", self)
            a_ingest.triggered.connect(self._ingest_file_dialog)
            m.addAction(a_ingest)
            m.addSeparator()
            a_quit = QAction("Quit", self)
            a_quit.triggered.connect(self.close)
            m.addAction(a_quit)

            # System
            m2 = mb.addMenu("System")
            a_status = QAction("Refresh Status", self)
            a_status.triggered.connect(self._refresh_status)
            m2.addAction(a_status)
            a_export_kg = QAction("Export KG to JSON", self)
            a_export_kg.triggered.connect(self._export_kg)
            m2.addAction(a_export_kg)
            a_clear = QAction("Clear Session", self)
            a_clear.triggered.connect(self._clear_session)
            m2.addAction(a_clear)

            # View
            m3 = mb.addMenu("View")
            a_web = QAction("Open Web UI in Browser", self)
            a_web.triggered.connect(self._open_web_ui)
            m3.addAction(a_web)

            # Help
            m4 = mb.addMenu("Help")
            a_about = QAction("About", self)
            a_about.triggered.connect(self._about)
            m4.addAction(a_about)

        # ── Toolbar ────────────────────────────────────────────────
        def _build_toolbar(self):
            tb = QToolBar("Main", self)
            tb.setMovable(False)
            self.addToolBar(tb)

            for label, slot in [
                ("📎 Ingest",      self._ingest_file_dialog),
                ("🔄 Refresh",     self._refresh_status),
                ("🗑 Clear Chat",   self._clear_chat),
                ("📊 Metrics",     lambda: self._right_tabs.setCurrentIndex(5)),
                ("🛡 Guardrails",  lambda: self._right_tabs.setCurrentIndex(3)),
                ("🧩 Cognition",   lambda: self._right_tabs.setCurrentIndex(4)),
            ]:
                btn = QPushButton(label)
                btn.setMaximumHeight(28)
                btn.clicked.connect(slot)
                tb.addWidget(btn)

        # ── Central widget ─────────────────────────────────────────
        def _build_central(self):
            central  = QWidget()
            h_layout = QHBoxLayout(central)
            h_layout.setContentsMargins(0, 0, 0, 0)
            self.setCentralWidget(central)

            splitter = QSplitter(Qt.Horizontal)
            h_layout.addWidget(splitter)

            # LEFT: Chat panel
            splitter.addWidget(self._build_chat_panel())

            # RIGHT: Tab panel
            self._right_tabs = QTabWidget()
            self._right_tabs.setMinimumWidth(420)
            self._right_tabs.addTab(self._build_memory_panel(),     "🧠 Memory")
            self._right_tabs.addTab(self._build_kg_panel(),         "🕸 KG")
            self._right_tabs.addTab(self._build_workflow_panel(),   "⚙ Workflow")
            self._right_tabs.addTab(self._build_guardrail_panel(),  "🛡 Guardrails")
            self._right_tabs.addTab(self._build_cognition_panel(),  "🧩 Cognition")
            self._right_tabs.addTab(self._build_metrics_panel(),    "📊 Metrics")
            self._right_tabs.addTab(self._build_eval_panel(),       "🎯 Eval")
            self._right_tabs.addTab(self._build_dataset_panel(),    "📦 Dataset")
            self._right_tabs.addTab(self._build_training_panel(),   "🏋 Training")
            splitter.addWidget(self._right_tabs)
            splitter.setSizes([740, 460])

        # ── Status bar ─────────────────────────────────────────────
        def _build_statusbar(self):
            sb = QStatusBar()
            self.setStatusBar(sb)
            self._lbl_backend   = QLabel("backend: —")
            self._lbl_guardrail = QLabel("guardrails: —")
            self._lbl_cognition = QLabel("cognition: —")
            self._lbl_memory    = QLabel("memory: —")
            self._lbl_kg        = QLabel("kg: —")
            for lbl in [self._lbl_backend, self._lbl_guardrail,
                        self._lbl_cognition, self._lbl_memory, self._lbl_kg]:
                sb.addPermanentWidget(lbl)

        # ══════════════════════════════════════════════════════════
        # CHAT PANEL
        # ══════════════════════════════════════════════════════════
        def _build_chat_panel(self) -> QWidget:
            w   = QWidget()
            lay = QVBoxLayout(w)
            lay.setContentsMargins(8, 4, 4, 4)

            hdr = QLabel("💬  Chat")
            hdr.setStyleSheet("font-size:14px;font-weight:bold;padding:4px;")
            lay.addWidget(hdr)

            self._chat_display = QTextEdit()
            self._chat_display.setReadOnly(True)
            self._chat_display.setFont(QFont("Consolas", 11))
            self._chat_display.setPlaceholderText("Conversation will appear here…")
            lay.addWidget(self._chat_display, stretch=3)

            # Trace display
            self._trace_display = QTextEdit()
            self._trace_display.setReadOnly(True)
            self._trace_display.setMaximumHeight(90)
            self._trace_display.setFont(QFont("Consolas", 9))
            self._trace_display.setStyleSheet("background:#0a0c14;color:#6b7280;")
            self._trace_display.setPlaceholderText("Reasoning trace…")
            lay.addWidget(self._trace_display)

            # Input row
            inp_row = QHBoxLayout()
            self._chat_input = QLineEdit()
            self._chat_input.setPlaceholderText("Message CognitiveOC…  (Enter to send)")
            self._chat_input.returnPressed.connect(self._send_message)
            self._chat_input.setMinimumHeight(34)
            inp_row.addWidget(self._chat_input)

            send_btn = QPushButton("Send")
            send_btn.setMinimumHeight(34)
            send_btn.setMinimumWidth(70)
            send_btn.clicked.connect(self._send_message)
            inp_row.addWidget(send_btn)
            lay.addLayout(inp_row)

            # Options row
            opt_row = QHBoxLayout()
            self._show_trace = QCheckBox("Show trace")
            opt_row.addWidget(self._show_trace)
            self._stream_mode = QCheckBox("Stream")
            self._stream_mode.setChecked(True)
            opt_row.addWidget(self._stream_mode)
            self._chat_status = QLabel("")
            self._chat_status.setStyleSheet("color:#94a3b8;font-size:11px;")
            opt_row.addWidget(self._chat_status)
            opt_row.addStretch()
            lay.addLayout(opt_row)
            return w

        # ── Send message ────────────────────────────────────────────
        def _send_message(self):
            msg = self._chat_input.text().strip()
            if not msg:
                return
            self._chat_input.clear()
            self._append_chat("You", msg, "#6c8dfa")
            self._chat_status.setText("generating…")

            if self._stream_mode.isChecked():
                worker = StreamWorker(self._engine, msg, self._session)
                self._cur_assistant_pos = None
                worker.fragment.connect(self._on_stream_fragment)
                worker.done.connect(self._on_stream_done)
                self._workers.append(worker)
                worker.start()
            else:
                worker = Worker(self._engine.process, msg, session=self._session)
                worker.result.connect(self._on_process_result)
                worker.error.connect(lambda e: self._chat_status.setText(f"Error: {e}"))
                self._workers.append(worker)
                worker.start()

        def _append_chat(self, role: str, text: str, color: str = "#e2e8f0"):
            cur = self._chat_display.textCursor()
            cur.movePosition(QTextCursor.End)
            self._chat_display.setTextCursor(cur)
            self._chat_display.append(
                f'<span style="color:{color};font-weight:bold">{role}:</span> '
                f'<span style="color:#e2e8f0">{text.replace(chr(10),"<br>")}</span><br>'
            )
            self._chat_display.verticalScrollBar().setValue(
                self._chat_display.verticalScrollBar().maximum()
            )

        def _on_stream_fragment(self, fragment: str):
            if not hasattr(self, "_stream_buf"):
                self._stream_buf = ""
                self._append_chat("COC", "")
            self._stream_buf += fragment
            # Update last paragraph
            cur = self._chat_display.textCursor()
            cur.movePosition(QTextCursor.End)
            self._chat_display.setTextCursor(cur)

        def _on_stream_done(self, trace: dict):
            if hasattr(self, "_stream_buf"):
                self._append_chat("COC", self._stream_buf)
                del self._stream_buf
            if trace and self._show_trace.isChecked():
                self._trace_display.setPlainText(json.dumps(trace, indent=2))
            self._chat_status.setText(
                f"intent={trace.get('intent','?')} | "
                f"{trace.get('latency_ms',0):.0f}ms"
            )

        def _on_process_result(self, result):
            self._append_chat("COC", result.text)
            if result.trace and self._show_trace.isChecked():
                self._trace_display.setPlainText(json.dumps(result.trace, indent=2))
            self._chat_status.setText(
                f"intent={result.intent} | {result.trace.get('latency_ms',0):.0f}ms"
            )

        def _clear_chat(self):
            self._chat_display.clear()
            self._trace_display.clear()
            self._session = f"desktop_{int(time.time())}"

        # ══════════════════════════════════════════════════════════
        # MEMORY PANEL
        # ══════════════════════════════════════════════════════════
        def _build_memory_panel(self) -> QWidget:
            w   = QWidget()
            lay = QVBoxLayout(w)

            hdr_row = QHBoxLayout()
            hdr_row.addWidget(QLabel("Search:"))
            self._mem_search = QLineEdit()
            self._mem_search.setPlaceholderText("Search memories…")
            self._mem_search.textChanged.connect(self._load_memories)
            hdr_row.addWidget(self._mem_search)
            refresh_btn = QPushButton("Refresh")
            refresh_btn.clicked.connect(self._load_memories)
            hdr_row.addWidget(refresh_btn)
            cons_btn = QPushButton("Consolidate")
            cons_btn.clicked.connect(self._consolidate_memory)
            hdr_row.addWidget(cons_btn)
            lay.addLayout(hdr_row)

            self._mem_list = QListWidget()
            self._mem_list.setAlternatingRowColors(True)
            lay.addWidget(self._mem_list)

            self._mem_stats = QLabel("")
            self._mem_stats.setStyleSheet("color:#94a3b8;font-size:11px;")
            lay.addWidget(self._mem_stats)
            return w

        def _load_memories(self):
            q   = self._mem_search.text().strip() if hasattr(self, "_mem_search") else ""
            mem = self._engine.memory
            if not mem:
                return
            items = (mem.search(q, limit=50) if q
                     else mem.list_memories(limit=50))
            self._mem_list.clear()
            for m in items:
                text = f"[{m.get('kind','?')}] {m.get('text','')[:120]}"
                item = QListWidgetItem(text)
                self._mem_list.addItem(item)
            stats = mem.stats()
            self._mem_stats.setText(
                f"total={stats.get('total',0)}  active={stats.get('active',0)}  "
                f"archived={stats.get('archived',0)}"
            )

        def _consolidate_memory(self):
            mem = self._engine.memory
            if mem:
                n = mem.consolidate()
                QMessageBox.information(self, "Consolidate", f"Removed {n} duplicate memories")
                self._load_memories()

        # ══════════════════════════════════════════════════════════
        # KG PANEL
        # ══════════════════════════════════════════════════════════
        def _build_kg_panel(self) -> QWidget:
            w   = QWidget()
            lay = QVBoxLayout(w)

            hdr_row = QHBoxLayout()
            self._kg_search = QLineEdit()
            self._kg_search.setPlaceholderText("Search entities or triples…")
            self._kg_search.returnPressed.connect(self._kg_search_action)
            hdr_row.addWidget(self._kg_search)
            for label, slot in [("Search", self._kg_search_action),
                                  ("Analytics", self._kg_analytics),
                                  ("Export JSON", self._export_kg)]:
                b = QPushButton(label)
                b.clicked.connect(slot)
                hdr_row.addWidget(b)
            lay.addLayout(hdr_row)

            self._kg_result = QTextEdit()
            self._kg_result.setReadOnly(True)
            self._kg_result.setFont(QFont("Consolas", 10))
            lay.addWidget(self._kg_result)
            return w

        def _kg_search_action(self):
            q  = self._kg_search.text().strip()
            if not q:
                return
            kg = self._engine.kg
            if not kg:
                return
            results = kg.fts_search(q, limit=20)
            lines   = [f"{r.get('subject')} → {r.get('relation')} → {r.get('object')} "
                       f"[conf={r.get('confidence',0):.2f}]" for r in results]
            self._kg_result.setPlainText(
                "\n".join(lines) if lines else "No results."
            )

        def _kg_analytics(self):
            kg = self._engine.kg
            if not kg:
                return
            a = kg.analytics()
            self._kg_result.setPlainText(json.dumps(a, indent=2))

        def _export_kg(self):
            kg = self._engine.kg
            if not kg:
                return
            path = kg.export_json()
            QMessageBox.information(self, "Export KG", f"Exported → {path}")

        # ══════════════════════════════════════════════════════════
        # WORKFLOW PANEL
        # ══════════════════════════════════════════════════════════
        def _build_workflow_panel(self) -> QWidget:
            w   = QWidget()
            lay = QVBoxLayout(w)

            inp_row = QHBoxLayout()
            self._wf_goal = QLineEdit()
            self._wf_goal.setPlaceholderText("Workflow goal…")
            inp_row.addWidget(self._wf_goal)
            create_btn = QPushButton("Create & Run")
            create_btn.clicked.connect(self._create_workflow)
            inp_row.addWidget(create_btn)
            refresh_btn = QPushButton("Refresh")
            refresh_btn.clicked.connect(self._load_workflows)
            inp_row.addWidget(refresh_btn)
            lay.addLayout(inp_row)

            self._wf_list = QListWidget()
            self._wf_list.setAlternatingRowColors(True)
            lay.addWidget(self._wf_list)

            self._wf_detail = QTextEdit()
            self._wf_detail.setReadOnly(True)
            self._wf_detail.setMaximumHeight(120)
            self._wf_detail.setFont(QFont("Consolas", 9))
            lay.addWidget(self._wf_detail)
            self._wf_list.itemClicked.connect(self._wf_item_clicked)
            self._wf_items_data: list[dict] = []
            return w

        def _create_workflow(self):
            goal = self._wf_goal.text().strip()
            if not goal:
                return
            self._wf_goal.clear()
            from workflow.workflow import WorkflowEngine
            wfe = WorkflowEngine()
            wf  = wfe.create(goal, session=self._session)
            worker = Worker(wfe.run, wf.id)
            worker.result.connect(lambda _: self._load_workflows())
            self._workers.append(worker)
            worker.start()
            self._load_workflows()

        def _load_workflows(self):
            from workflow.workflow import WorkflowEngine
            items = WorkflowEngine().list_workflows(limit=20)
            self._wf_items_data = items
            self._wf_list.clear()
            for wf in items:
                state = wf.get("state","?")
                color_map = {"completed":"✅","failed":"❌","running":"⏳",
                             "paused":"⏸","executing":"⚙","created":"🆕"}
                icon  = color_map.get(state, "•")
                self._wf_list.addItem(f"{icon} {wf.get('goal','')[:60]}  [{state}]")

        def _wf_item_clicked(self, item):
            idx = self._wf_list.row(item)
            if idx < len(self._wf_items_data):
                self._wf_detail.setPlainText(
                    json.dumps(self._wf_items_data[idx], indent=2, default=str)
                )

        # ══════════════════════════════════════════════════════════
        # GUARDRAIL PANEL
        # ══════════════════════════════════════════════════════════
        def _build_guardrail_panel(self) -> QWidget:
            w   = QWidget()
            lay = QVBoxLayout(w)

            # Hard integrity notice
            hard_lbl = QLabel(
                "⚠  Hard Integrity Guards: ALWAYS ON\n"
                "File | DB | Checkpoint | Memory | Path | Permission | Schema | Stability | Crash"
            )
            hard_lbl.setStyleSheet(
                "background:#16213a;color:#93c5fd;border:1px solid #3b4f8a;"
                "padding:8px;border-radius:4px;font-size:11px;"
            )
            hard_lbl.setWordWrap(True)
            lay.addWidget(hard_lbl)

            # Profile selector
            prof_row = QHBoxLayout()
            prof_row.addWidget(QLabel("Profile:"))
            self._guard_profile = QComboBox()
            self._guard_profile.addItems(["strict","standard","research","developer","off"])
            self._guard_profile.currentTextChanged.connect(self._set_guardrail_profile)
            prof_row.addWidget(self._guard_profile)
            prof_row.addStretch()
            lay.addLayout(prof_row)

            # Per-guard toggles
            lay.addWidget(QLabel("Cognitive Guards:"))
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            inner = QWidget()
            self._guard_toggle_layout = QVBoxLayout(inner)
            scroll.setWidget(inner)
            lay.addWidget(scroll)

            self._guard_checkboxes: dict[str, QCheckBox] = {}
            self._load_guardrail_toggles()
            return w

        def _load_guardrail_toggles(self):
            from safety.guardrails_state import get as gs_get
            state = gs_get()

            # Update profile combo without triggering signal
            self._guard_profile.blockSignals(True)
            idx = self._guard_profile.findText(state.get("_profile","standard"))
            if idx >= 0:
                self._guard_profile.setCurrentIndex(idx)
            self._guard_profile.blockSignals(False)

            # Clear and rebuild toggles
            while self._guard_toggle_layout.count():
                child = self._guard_toggle_layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
            self._guard_checkboxes.clear()

            for key, val in state.items():
                if key.startswith("_"):
                    continue
                cb = QCheckBox(key.replace("_"," "))
                cb.setChecked(bool(val))
                cb.stateChanged.connect(lambda v, k=key: self._toggle_guard(k, bool(v)))
                self._guard_checkboxes[key] = cb
                self._guard_toggle_layout.addWidget(cb)
            self._guard_toggle_layout.addStretch()

        def _set_guardrail_profile(self, profile: str):
            from safety.guardrails_state import set_profile
            set_profile(profile)
            self._load_guardrail_toggles()
            self._refresh_status()

        def _toggle_guard(self, guard: str, enabled: bool):
            from safety.guardrails_state import set_guard
            set_guard(guard, enabled)

        # ══════════════════════════════════════════════════════════
        # COGNITION PANEL
        # ══════════════════════════════════════════════════════════
        def _build_cognition_panel(self) -> QWidget:
            w   = QWidget()
            lay = QVBoxLayout(w)

            isolation_lbl = QLabel(
                "ℹ  OFF mode only disables cognition modules.\n"
                "Memory · Retrieval · KG · Reasoning · Guardrails · Tools remain active."
            )
            isolation_lbl.setStyleSheet(
                "background:#162032;color:#7dd3fc;border:1px solid #1e4063;"
                "padding:8px;border-radius:4px;font-size:11px;"
            )
            isolation_lbl.setWordWrap(True)
            lay.addWidget(isolation_lbl)

            # Mode selector
            mode_row = QHBoxLayout()
            mode_row.addWidget(QLabel("Mode:"))
            self._cog_mode = QComboBox()
            self._cog_mode.addItems(["full","partial","custom","off"])
            self._cog_mode.currentTextChanged.connect(self._set_cognition_mode)
            mode_row.addWidget(self._cog_mode)
            mode_row.addStretch()
            lay.addLayout(mode_row)

            # Per-module toggles
            lay.addWidget(QLabel("Modules:"))
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            inner = QWidget()
            self._cog_toggle_layout = QVBoxLayout(inner)
            scroll.setWidget(inner)
            lay.addWidget(scroll)

            self._cog_checkboxes: dict[str, QCheckBox] = {}
            self._load_cognition_toggles()

            # Current state display
            self._cog_state_lbl = QLabel("")
            self._cog_state_lbl.setStyleSheet("color:#94a3b8;font-size:10px;")
            lay.addWidget(self._cog_state_lbl)
            return w

        def _load_cognition_toggles(self):
            from cognition.cognition import get_state
            state   = get_state()
            modules = state.get("modules", {})

            self._cog_mode.blockSignals(True)
            idx = self._cog_mode.findText(state.get("mode","full"))
            if idx >= 0:
                self._cog_mode.setCurrentIndex(idx)
            self._cog_mode.blockSignals(False)

            while self._cog_toggle_layout.count():
                child = self._cog_toggle_layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
            self._cog_checkboxes.clear()

            for mod, enabled in modules.items():
                cb = QCheckBox(mod.replace("_"," "))
                cb.setChecked(bool(enabled))
                cb.stateChanged.connect(
                    lambda v, m=mod: self._toggle_cognition_module(m, bool(v))
                )
                self._cog_checkboxes[mod] = cb
                self._cog_toggle_layout.addWidget(cb)
            self._cog_toggle_layout.addStretch()

            if hasattr(self, "_cog_state_lbl"):
                self._cog_state_lbl.setText(
                    f"mode={state.get('mode')}  modules={sum(v for v in modules.values())}/{len(modules)} active"
                )

        def _set_cognition_mode(self, mode: str):
            from cognition.cognition import set_mode
            set_mode(mode)
            self._load_cognition_toggles()
            self._refresh_status()

        def _toggle_cognition_module(self, module: str, enabled: bool):
            from cognition.cognition import set_module
            set_module(module, enabled)

        # ══════════════════════════════════════════════════════════
        # METRICS PANEL
        # ══════════════════════════════════════════════════════════
        def _build_metrics_panel(self) -> QWidget:
            w   = QWidget()
            lay = QVBoxLayout(w)

            hdr_row = QHBoxLayout()
            refresh_btn = QPushButton("Refresh")
            refresh_btn.clicked.connect(self._load_metrics)
            hdr_row.addWidget(refresh_btn)
            self._auto_refresh_cb = QCheckBox("Auto-refresh (8s)")
            self._auto_refresh_cb.stateChanged.connect(
                lambda v: self._refresh_timer.start(8000) if v else None
            )
            hdr_row.addWidget(self._auto_refresh_cb)
            hdr_row.addStretch()
            lay.addLayout(hdr_row)

            self._metrics_display = QTextEdit()
            self._metrics_display.setReadOnly(True)
            self._metrics_display.setFont(QFont("Consolas", 10))
            lay.addWidget(self._metrics_display)
            return w

        def _load_metrics(self):
            try:
                snap = self._engine.observability.snapshot()
                hw   = snap.get("hardware",{})
                req  = snap.get("requests",{})
                ret  = snap.get("retrieval",{})
                tr   = snap.get("training",{})

                lines = [
                    "── Hardware ──────────────────────────",
                    f"  CPU:         {hw.get('cpu_pct',0)}%",
                    f"  RAM:         {hw.get('ram_used_gb',0):.1f} / {hw.get('ram_total_gb',0):.1f} GB",
                    f"  GPU VRAM:    {hw.get('gpu_used_mb',0):.0f} / {hw.get('gpu_total_mb',0):.0f} MB",
                    f"  GPU util:    {hw.get('gpu_util_pct',0)}%  temp={hw.get('gpu_temp_c',0)}°C",
                    "",
                    "── Requests ──────────────────────────",
                    f"  Total:       {req.get('total',0)}",
                    f"  Errors:      {req.get('errors',0)}  ({req.get('error_rate',0)*100:.1f}%)",
                    f"  Avg latency: {req.get('avg_latency_ms',0):.1f} ms",
                    f"  Tokens in:   {req.get('total_tokens_in',0)}",
                    f"  Tokens out:  {req.get('total_tokens_out',0)}",
                    "",
                    "── Retrieval ─────────────────────────",
                    f"  Hit rate:    {ret.get('hit_rate',0):.3f}",
                    f"  Cache hit:   {ret.get('cache_hit_rate',0):.3f}",
                    f"  Multi-hop:   {ret.get('multi_hop_rate',0):.3f}",
                    f"  Avg score:   {ret.get('avg_score',0):.3f}",
                ]
                if tr.get("step"):
                    lines += [
                        "",
                        "── Live Training ─────────────────────",
                        f"  Step:        {tr.get('step',0)}",
                        f"  Train loss:  {tr.get('train_loss','—')}",
                        f"  Val loss:    {tr.get('val_loss','—')}",
                        f"  Perplexity:  {tr.get('perplexity','—')}",
                        f"  Grad norm:   {tr.get('grad_norm','—')}",
                        f"  Tok/s:       {tr.get('tokens_per_sec',0)}",
                    ]
                self._metrics_display.setPlainText("\n".join(lines))
            except Exception as e:
                self._metrics_display.setPlainText(f"Error: {e}")

        # ══════════════════════════════════════════════════════════
        # EVAL PANEL
        # ══════════════════════════════════════════════════════════
        def _build_eval_panel(self) -> QWidget:
            w   = QWidget()
            lay = QVBoxLayout(w)

            btn_row = QHBoxLayout()
            for label, slot in [
                ("Run All",     self._run_eval_all),
                ("Tokenizer",   self._eval_tokenizer),
                ("Gate 1",      lambda: self._check_gate(1)),
                ("Gate 3",      lambda: self._check_gate(3)),
                ("Gate 6",      lambda: self._check_gate(6)),
            ]:
                b = QPushButton(label)
                b.clicked.connect(slot)
                btn_row.addWidget(b)
            lay.addLayout(btn_row)

            self._eval_display = QTextEdit()
            self._eval_display.setReadOnly(True)
            self._eval_display.setFont(QFont("Consolas", 10))
            self._eval_display.setPlainText(
                "Click Run All to execute the complete evaluation suite.\n"
                "Results appear here."
            )
            lay.addWidget(self._eval_display)
            return w

        def _run_eval_all(self):
            self._eval_display.setPlainText("Running evaluation suite…")
            def _run():
                from eval.run_suite import run_all
                r = run_all("data/corpus/v1/split/val.txt", save=True, verbose=False)
                return r
            worker = Worker(_run)
            worker.result.connect(lambda r: self._eval_display.setPlainText(
                json.dumps(r, indent=2, default=str)
            ))
            self._workers.append(worker)
            worker.start()

        def _eval_tokenizer(self):
            def _run():
                from eval.run_suite import eval_tokenizer
                return eval_tokenizer()
            worker = Worker(_run)
            worker.result.connect(lambda r: self._eval_display.setPlainText(
                json.dumps(r, indent=2, default=str)
            ))
            self._workers.append(worker)
            worker.start()

        def _check_gate(self, phase: int):
            from eval.run_suite import check_gate
            r = check_gate(phase)
            passed = r.get("passed", False)
            lines = [f"Phase {phase} Gate: {'PASS ✅' if passed else 'FAIL ❌'}",
                     f"Score: {r.get('n_pass',0)}/{r.get('n_total',0)}", ""]
            for c in r.get("checks",[]):
                mark = "✓" if c["passed"] else "✗"
                lines.append(f"  [{mark}] {c['metric']} {c['op']} {c['threshold']} → {c.get('detail','')}")
            self._eval_display.setPlainText("\n".join(lines))

        # ══════════════════════════════════════════════════════════
        # DATASET PANEL
        # ══════════════════════════════════════════════════════════
        def _build_dataset_panel(self) -> QWidget:
            w   = QWidget()
            lay = QVBoxLayout(w)
            lay.addWidget(QLabel("Dataset Export (human review required before training):"))

            btn_row = QHBoxLayout()
            for label, action in [
                ("Conversations",  "export_conversations"),
                ("Retrieval",      "export_retrieval"),
                ("Memory",         "export_memory"),
                ("KG",             "export_kg"),
                ("Hard Examples",  "hard_examples"),
                ("Export All",     "export_all"),
            ]:
                b = QPushButton(label)
                b.clicked.connect(lambda _=False, a=action: self._dataset_action(a))
                btn_row.addWidget(b)
            lay.addLayout(btn_row)

            self._ds_display = QTextEdit()
            self._ds_display.setReadOnly(True)
            self._ds_display.setFont(QFont("Consolas", 10))
            lay.addWidget(self._ds_display)
            return w

        def _dataset_action(self, action: str):
            from dataset.generator import DatasetGenerator, LearningStore
            store = LearningStore()
            gen   = DatasetGenerator(store)
            mem   = self._engine.memory
            kg    = self._engine.kg
            methods = {
                "export_conversations": gen.export_conversations,
                "export_retrieval":     gen.export_retrieval,
                "export_memory":        lambda: gen.export_memory(mem),
                "export_kg":            lambda: gen.export_kg(kg),
                "hard_examples":        gen.hard_examples,
                "export_all":           lambda: gen.export_all(mem, kg),
            }
            fn = methods.get(action)
            if fn:
                result = fn()
                self._ds_display.setPlainText(json.dumps(result, indent=2, default=str))

        # ══════════════════════════════════════════════════════════
        # TRAINING PANEL
        # ══════════════════════════════════════════════════════════
        def _build_training_panel(self) -> QWidget:
            w   = QWidget()
            lay = QVBoxLayout(w)

            lbl = QLabel(
                "Training is launched from the command line.\n"
                "This panel shows live metrics from the training log.\n\n"
                "Commands:\n"
                "  python main.py train-tokenizer data/corpus/v1/split/train.txt\n"
                "  python main.py train-model data/corpus/v1/split/train.txt \\\n"
                "    --steps 100000 --batch 2 --accum 16 --precision bf16\n"
                "  python main.py eval --all data/corpus/v1/split/val.txt"
            )
            lbl.setFont(QFont("Consolas", 10))
            lbl.setWordWrap(True)
            lay.addWidget(lbl)

            refresh_btn = QPushButton("Refresh Training Metrics")
            refresh_btn.clicked.connect(self._load_training_metrics)
            lay.addWidget(refresh_btn)

            self._train_display = QTextEdit()
            self._train_display.setReadOnly(True)
            self._train_display.setFont(QFont("Consolas", 10))
            lay.addWidget(self._train_display)
            return w

        def _load_training_metrics(self):
            try:
                snap = self._engine.observability.snapshot()
                tr   = snap.get("training", {})
                if not tr.get("step"):
                    self._train_display.setPlainText("No training in progress.")
                    return
                lines = [
                    f"Step:        {tr.get('step')} / {tr.get('steps','?')}",
                    f"Train loss:  {tr.get('train_loss','—')}",
                    f"Val loss:    {tr.get('val_loss','—')}",
                    f"Perplexity:  {tr.get('perplexity','—')}",
                    f"Grad norm:   {tr.get('grad_norm','—')}",
                    f"LR:          {tr.get('lr','—')}",
                    f"Tok/s:       {tr.get('tokens_per_sec','—')}",
                    f"GPU mem:     {tr.get('gpu_mem_gb','—')} GB",
                    f"Elapsed:     {tr.get('elapsed_s','—')} s",
                    f"Timestamp:   {tr.get('ts','—')}",
                ]
                self._train_display.setPlainText("\n".join(lines))
            except Exception as e:
                self._train_display.setPlainText(f"Error: {e}")

        # ══════════════════════════════════════════════════════════
        # Status / helpers
        # ══════════════════════════════════════════════════════════
        def _refresh_status(self):
            try:
                status = self._engine.status()
                self._lbl_backend.setText(f"backend: {status.get('backend','?')[:30]}")
                gs = status.get("guardrails",{})
                self._lbl_guardrail.setText(f"guardrails: {gs.get('_profile','?')}")
                cs = status.get("cognition",{})
                self._lbl_cognition.setText(f"cognition: {cs.get('mode','?')}")
                ms = status.get("memory",{})
                self._lbl_memory.setText(f"memory: {ms.get('active',0)}")
                ks = status.get("kg",{})
                self._lbl_kg.setText(f"kg: {ks.get('triples',0)}")
                # Reload panels that auto-update
                if hasattr(self,"_mem_list"):
                    self._load_memories()
                if hasattr(self,"_metrics_display"):
                    self._load_metrics()
                if hasattr(self,"_train_display"):
                    self._load_training_metrics()
            except Exception:
                pass

        def _ingest_file_dialog(self):
            path, _ = QFileDialog.getOpenFileName(
                self, "Ingest Document", "",
                "Supported (*.pdf *.docx *.txt *.md *.csv *.xlsx *.png *.jpg)"
            )
            if path:
                result = self._engine.ingest(path)
                QMessageBox.information(
                    self, "Ingest",
                    f"OK: {result.get('ok')}\n"
                    f"Added: {result.get('added',0)} chunks\n"
                    f"KG triples: {result.get('kg_triples',0)}"
                )

        def _open_web_ui(self):
            try:
                from config import UI
                import webbrowser
                webbrowser.open(f"http://{UI.get('host','127.0.0.1')}:{UI.get('port',8765)}")
            except Exception:
                pass

        def _about(self):
            QMessageBox.about(
                self, "CognitiveOC v3",
                "CognitiveOC v3\n"
                "Local Cognitive Orchestration Core\n\n"
                "700M Decoder · 48K Tokenizer · 13 Encoders\n"
                "Fully local · No cloud APIs\n\n"
                "Native PySide6 Desktop — Primary Interface"
            )

        def _clear_session(self):
            self._session = f"desktop_{int(time.time())}"
            self._clear_chat()

    return CognitiveOCDesktop(engine)


# ═══════════════════════════════════════════════════════════════════
# Entry points
# ═══════════════════════════════════════════════════════════════════

def launch(start_web_server: bool = True):
    """Launch the native PySide6 desktop application.

    Args:
        start_web_server: If True, also starts the web UI server in a
                          background thread so the browser interface remains
                          available as a secondary access point.
    """
    if not _qt_available():
        print("[desktop] PySide6 not installed.")
        print("[desktop] Install: pip install PySide6")
        print("[desktop] Note: PySide6-WebEngine is NOT required for the native app.")
        sys.exit(1)

    from PySide6.QtWidgets import QApplication
    from engine import Engine

    # Optionally start web server for secondary access
    if start_web_server:
        try:
            from config import UI
            def _serve():
                from ui.app import serve
                serve(UI.get("host","127.0.0.1"), UI.get("port",8765))
            t = threading.Thread(target=_serve, daemon=True, name="coc-web")
            t.start()
            print(f"[desktop] Web UI also available at "
                  f"http://{UI.get('host','127.0.0.1')}:{UI.get('port',8765)}")
        except Exception as e:
            print(f"[desktop] Web server skipped: {e}")

    app    = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("CognitiveOC v3")
    app.setApplicationVersion("3.0")
    app.setStyle("Fusion")

    print("[desktop] Initialising engine…")
    engine = Engine()

    window = build_main_window(engine)
    window.show()
    print("[desktop] CognitiveOC v3 native desktop started.")

    sys.exit(app.exec())


def launch_headless_test() -> dict:
    """Test desktop module imports and engine without opening a window.
    Used by runtime audit scripts.
    """
    from engine import Engine
    eng = Engine()
    return {
        "import":         "ok",
        "engine_status":  eng.status().get("version"),
        "backend":        eng.status().get("backend"),
        "qt_available":   _qt_available(),
        "note": ("Native window requires PySide6 installed and display. "
                 "All backend logic verified without display."),
    }


# ═══════════════════════════════════════════════════════════════════
# Patch: Add Research, Validation, Retrieval panels to desktop
# These are injected into build_main_window after the class definition
# via monkey-patching at import time.
# ═══════════════════════════════════════════════════════════════════

def _patch_desktop_panels():
    """Inject research, validation, and retrieval panels into
    CognitiveOCDesktop if they are not already present.
    Called automatically at module import time.
    """
    if not _qt_available():
        return

    from PySide6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QTextEdit,
        QLineEdit, QPushButton, QLabel,
    )
    from PySide6.QtGui import QFont

    # ── Research panel ───────────────────────────────────────────
    def _build_research_panel(self) -> QWidget:
        w   = QWidget()
        lay = QVBoxLayout(w)
        lay.addWidget(QLabel("Research Engine — iterative evidence loop:"))

        inp = QHBoxLayout()
        self._re_question = QLineEdit()
        self._re_question.setPlaceholderText("Research question…")
        inp.addWidget(self._re_question)
        start_btn = QPushButton("Start")
        start_btn.clicked.connect(self._start_research)
        inp.addWidget(start_btn)
        refresh_btn = QPushButton("List")
        refresh_btn.clicked.connect(self._load_research)
        inp.addWidget(refresh_btn)
        lay.addLayout(inp)

        self._re_display = QTextEdit()
        self._re_display.setReadOnly(True)
        self._re_display.setFont(QFont("Consolas", 10))
        lay.addWidget(self._re_display)
        return w

    def _start_research(self):
        q = self._re_question.text().strip()
        if not q:
            return
        self._re_question.clear()
        from research.engine import ResearchEngine
        re_eng = ResearchEngine({
            "memory":    self._engine.memory,
            "retriever": self._engine._hybrid,
            "kg":        self._engine.kg,
        })
        wf_id = re_eng.start_async(q, session=self._session)
        self._re_display.setPlainText(f"Research started: {wf_id}\nRefresh to see status.")

    def _load_research(self):
        from research.engine import ResearchEngine
        re_eng = ResearchEngine({})
        items  = re_eng.list_research(limit=10)
        import json
        self._re_display.setPlainText(
            json.dumps(items, indent=2, default=str) if items else "No research tasks."
        )

    # ── Validation panel ─────────────────────────────────────────
    def _build_validation_panel(self) -> QWidget:
        w   = QWidget()
        lay = QVBoxLayout(w)
        lay.addWidget(QLabel("Validate a response against evidence:"))

        self._val_response = QTextEdit()
        self._val_response.setPlaceholderText("Response text to validate…")
        self._val_response.setMaximumHeight(90)
        lay.addWidget(self._val_response)

        self._val_query = QLineEdit()
        self._val_query.setPlaceholderText("Original query…")
        lay.addWidget(self._val_query)

        run_btn = QPushButton("Validate")
        run_btn.clicked.connect(self._run_desktop_validation)
        lay.addWidget(run_btn)

        self._val_display = QTextEdit()
        self._val_display.setReadOnly(True)
        self._val_display.setFont(QFont("Consolas", 10))
        lay.addWidget(self._val_display)
        return w

    def _run_desktop_validation(self):
        response = self._val_response.toPlainText().strip()
        query    = self._val_query.text().strip()
        if not response:
            return
        from validation.validator import Validator
        vld = Validator()
        vr  = vld.validate(response, query=query, kg=self._engine.kg)
        import json
        lines = [
            f"Result: {'PASS ✅' if vr.passed else 'FAIL/WARN ⚠'}  Score: {vr.score}",
            "",
        ]
        for c in vr.checks:
            mark = "✓" if c.passed else "✗"
            lines.append(f"[{mark}] {c.check:<20s}  score={c.score:.2f}  {c.detail}")
        self._val_display.setPlainText("\n".join(lines))

    # ── Retrieval panel ──────────────────────────────────────────
    def _build_retrieval_panel(self) -> QWidget:
        w   = QWidget()
        lay = QVBoxLayout(w)
        lay.addWidget(QLabel("Retrieval Inspector — search the document index:"))

        inp = QHBoxLayout()
        self._ret_query = QLineEdit()
        self._ret_query.setPlaceholderText("Search query…")
        self._ret_query.returnPressed.connect(self._run_retrieval)
        inp.addWidget(self._ret_query)
        search_btn = QPushButton("Search")
        search_btn.clicked.connect(self._run_retrieval)
        inp.addWidget(search_btn)
        stats_btn = QPushButton("Stats")
        stats_btn.clicked.connect(self._retrieval_stats)
        inp.addWidget(stats_btn)
        lay.addLayout(inp)

        self._ret_display = QTextEdit()
        self._ret_display.setReadOnly(True)
        self._ret_display.setFont(QFont("Consolas", 10))
        lay.addWidget(self._ret_display)
        return w

    def _run_retrieval(self):
        q = self._ret_query.text().strip()
        if not q or not self._engine._rag:
            return
        self._engine._init()
        chunks = self._engine._rag.retrieve(q, k=8)
        lines  = []
        for c in chunks:
            lines.append(
                f"[{c.get('source','?')}] score={c.get('score',0):.3f}"
            )
            lines.append(f"  {c.get('text','')[:200]}")
            lines.append("")
        self._ret_display.setPlainText(
            "\n".join(lines) if lines else "No results — ingest documents first."
        )

    def _retrieval_stats(self):
        self._engine._init()
        if self._engine._rag:
            import json
            s = self._engine._rag.stats()
            from retrieval.self_improve import stats as si_stats
            s["self_improve"] = si_stats()
            self._ret_display.setPlainText(json.dumps(s, indent=2))

    # ── Inject into class ────────────────────────────────────────
    import types

    # Only patch if build_main_window has been defined
    # We patch the CognitiveOCDesktop class's methods directly
    # This is safe because build_main_window is called at runtime
    # and the class is created inside the function scope.
    # Instead, we extend build_main_window to add the extra tabs.

    # Store patch methods for use in patched build_main_window
    _EXTRA_PANELS = {
        "research":   (_build_research_panel,  _start_research,
                       _load_research),
        "validation": (_build_validation_panel, _run_desktop_validation),
        "retrieval":  (_build_retrieval_panel,  _run_retrieval, _retrieval_stats),
    }

    # Expose globally so the patched build can access them
    globals()["_EXTRA_PANELS"]          = _EXTRA_PANELS
    globals()["_build_research_panel"]  = _build_research_panel
    globals()["_start_research"]        = _start_research
    globals()["_load_research"]         = _load_research
    globals()["_build_validation_panel"]= _build_validation_panel
    globals()["_run_desktop_validation"]= _run_desktop_validation
    globals()["_build_retrieval_panel"] = _build_retrieval_panel
    globals()["_run_retrieval"]         = _run_retrieval
    globals()["_retrieval_stats"]       = _retrieval_stats


_patch_desktop_panels()


# ── Patched build_main_window that includes extra panels ─────────────
_orig_build = build_main_window


def build_main_window(engine):
    """Build QMainWindow with ALL panels including research, validation, retrieval."""
    window = _orig_build(engine)

    if not _qt_available():
        return window

    import types

    # Bind extra methods to the window instance
    extra_methods = {
        "_build_research_panel":   _build_research_panel,
        "_start_research":         _start_research,
        "_load_research":          _load_research,
        "_build_validation_panel": _build_validation_panel,
        "_run_desktop_validation": _run_desktop_validation,
        "_build_retrieval_panel":  _build_retrieval_panel,
        "_run_retrieval":          _run_retrieval,
        "_retrieval_stats":        _retrieval_stats,
    }
    for name, fn in extra_methods.items():
        setattr(window, name, types.MethodType(fn, window))

    # Add the three missing tabs to the right tab widget
    if hasattr(window, "_right_tabs"):
        from PySide6.QtWidgets import QTabWidget
        tabs = window._right_tabs
        tabs.addTab(window._build_research_panel(),  "🔬 Research")
        tabs.addTab(window._build_validation_panel(),"✅ Validation")
        tabs.addTab(window._build_retrieval_panel(), "🔍 Retrieval")

    return window
