"""
CognitiveOC v3 — Human Cognition Layer
=======================================

Implements the 9-module Human Cognition Orchestrator that sits between
the Encoder Intelligence Stack and the Orchestration Core (engine.py).

Architecture position (top-down workflow):
  Input & Perception
    → Encoder Intelligence Stack (emotion/intent/goal/safety signals)
    → Human Cognition Layer   ← THIS FILE
    → Orchestration Core (engine.py)
    → Memory / Retrieval / KG / Reasoning
    → Foundation Decoder (700M)
    → Output Governance

Modules (each independently toggleable via cognition_state.json):
  1. Emotion Understanding       — maps encoder emotion state → response tone
  2. Intent Understanding        — classifies user intent across 8 categories
  3. User Modeling               — persistent skill/preference/history profile
  4. Goal Tracking               — creates, updates, blocks goals; milestones
  5. Teaching Intelligence       — gap detection, curriculum, adaptive difficulty
  6. Decision Support            — option/tradeoff/risk/evidence analysis
  7. Reflection Engine           — self-review before final response
  8. Personality Adaptation      — 7 response style modes, auto-detected
  9. Context Awareness           — session/memory/workspace/goal context fusion

Control modes (hot-reload from cognition_state.json):
  full    — all 9 modules active
  partial — emotion + intent + context only
  custom  — per-module toggle dict
  off     — no cognition processing (memory/retrieval/guardrails unaffected)

CRITICAL: Turning cognition OFF does NOT disable:
  - Memory system
  - Retrieval system
  - Knowledge graph
  - Reasoning system
  - Guardrails

File: cognition/cognition.py
Used by: engine.py (CognitionLayer.process() called in build_context())
Persists: var/cognition_state.json (module toggles + mode)
          var/user_model.db (SQLite: profile, goals, skill levels, history)
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

try:
    from config import COGNITION, COGNITION_STATE, USER_MODEL_DB, ensure_dirs
except ImportError:
    COGNITION       = {}
    COGNITION_STATE = Path("var/cognition_state.json")
    USER_MODEL_DB   = Path("var/user_model.db")
    def ensure_dirs(): pass


# ═══════════════════════════════════════════════════════════════════
# State management (hot-reload safe)
# ═══════════════════════════════════════════════════════════════════

_DEFAULT_STATE = {
    "mode": "full",
    "modules": {
        "emotion":         True,
        "intent":          True,
        "user_modeling":   True,
        "goal_tracking":   True,
        "teaching":        True,
        "decision_support":True,
        "reflection":      True,
        "personality":     True,
        "context_aware":   True,
    },
    "updated": None,
}

_state_lock   = threading.RLock()
_cached_state: dict | None = None
_state_mtime: float = 0.0


def _load_state() -> dict:
    """Load cognition state from disk (hot-reload safe)."""
    global _cached_state, _state_mtime
    with _state_lock:
        path = Path(str(COGNITION_STATE))
        if path.exists():
            mtime = path.stat().st_mtime
            if _cached_state is None or mtime > _state_mtime:
                try:
                    _cached_state = json.loads(path.read_text())
                    _state_mtime  = mtime
                except Exception:
                    _cached_state = dict(_DEFAULT_STATE)
        else:
            _cached_state = dict(_DEFAULT_STATE)
        return _cached_state


def _save_state(state: dict):
    global _cached_state, _state_mtime
    with _state_lock:
        path = Path(str(COGNITION_STATE))
        path.parent.mkdir(parents=True, exist_ok=True)
        state["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        path.write_text(json.dumps(state, indent=2))
        _cached_state = state
        _state_mtime  = path.stat().st_mtime


def get_state() -> dict:
    return dict(_load_state())


def set_mode(mode: str) -> dict:
    """Set global cognition mode: full | partial | custom | off."""
    valid = {"full", "partial", "custom", "off"}
    if mode not in valid:
        raise ValueError(f"Invalid mode '{mode}'. Valid: {valid}")
    state = dict(_load_state())
    state["mode"] = mode
    _save_state(state)
    return get_state()


def set_module(name: str, enabled: bool) -> dict:
    """Toggle a single cognition module on/off.
    Automatically switches mode to 'custom' if not already.
    """
    state = dict(_load_state())
    if name not in state.get("modules", {}):
        raise ValueError(
            f"Unknown module '{name}'. Valid: {list(_DEFAULT_STATE['modules'])}"
        )
    state["modules"][name] = bool(enabled)
    if state["mode"] not in ("custom", "off"):
        state["mode"] = "custom"
    _save_state(state)
    return get_state()


def is_module_active(name: str) -> bool:
    """Check if a specific module should run given current mode/toggles."""
    state  = _load_state()
    mode   = state.get("mode", "full")
    if mode == "off":
        return False
    if mode == "full":
        return True
    if mode == "partial":
        return name in ("emotion", "intent", "context_aware")
    # custom
    return bool(state.get("modules", {}).get(name, True))


def reset_state() -> dict:
    """Reset cognition state to defaults."""
    _save_state(dict(_DEFAULT_STATE))
    return get_state()


# ═══════════════════════════════════════════════════════════════════
# 1. Emotion Understanding Module
# ═══════════════════════════════════════════════════════════════════

class EmotionModule:
    """Detect and interpret emotional signals from user text.

    Uses:
      - Emotion encoder (go_emotions model) via EncoderHub
      - Keyword heuristic fallback (zero deps)

    Output informs:
      - Response tone adjustment
      - Personality mode selection
      - Pacing / verbosity tuning
    """

    _TONE_MAP = {
        "frustration":    dict(tone="patient", verbosity="concise",
                               prefix="I understand this can be tricky. "),
        "confusion":      dict(tone="clear",   verbosity="detailed",
                               prefix="Let me break this down clearly. "),
        "curiosity":      dict(tone="engaged", verbosity="rich",
                               prefix=""),
        "anger":          dict(tone="calm",    verbosity="concise",
                               prefix=""),
        "sadness":        dict(tone="warm",    verbosity="gentle",
                               prefix=""),
        "excitement":     dict(tone="energetic", verbosity="rich",
                               prefix=""),
        "anxiety":        dict(tone="reassuring", verbosity="structured",
                               prefix=""),
        "neutral":        dict(tone="standard",  verbosity="standard",
                               prefix=""),
        "joy":            dict(tone="positive",  verbosity="standard",
                               prefix=""),
    }

    def __init__(self, cfg: dict):
        self._cfg   = cfg
        self._hub   = None   # lazy import to avoid circular deps

    def _get_hub(self):
        if self._hub is None:
            try:
                from encoder.hub import get_hub
                self._hub = get_hub()
            except Exception:
                self._hub = False
        return self._hub

    def process(self, text: str) -> dict:
        """Detect emotion in user text.

        Returns:
            {
                "primary":    str,
                "emotions":   [(label, score), ...],
                "tone":       str,
                "verbosity":  str,
                "prefix":     str,
                "strategy":   str,
                "confidence": float,
            }
        """
        hub = self._get_hub()
        if hub:
            try:
                result = hub.classify_emotion(text)
            except Exception:
                result = self._keyword_fallback(text)
        else:
            result = self._keyword_fallback(text)

        primary = result.get("primary", "neutral")
        tone_cfg = self._TONE_MAP.get(primary, self._TONE_MAP["neutral"])
        return {**result, **tone_cfg}

    def _keyword_fallback(self, text: str) -> dict:
        text_l = text.lower()
        signals = {
            "frustration":  ["frustrat","annoy","ugh","wrong","not working","broken","stuck"],
            "confusion":    ["confus","don't understand","unclear","what does","huh","?"],
            "curiosity":    ["how does","why","what is","explain","tell me","curious"],
            "excitement":   ["amazing","love","great","awesome","fantastic","can't wait"],
            "sadness":      ["sad","unfortunately","fail","disappoint","sorry","wrong"],
            "anger":        ["angry","terrible","worst","useless","nonsense","hate"],
        }
        best, best_score = "neutral", 0.0
        for emotion, keywords in signals.items():
            hits = sum(1 for k in keywords if k in text_l)
            score = min(0.95, hits * 0.3)
            if score > best_score:
                best, best_score = emotion, score
        return {"primary": best, "emotions": [(best, best_score)],
                "confidence": best_score, "strategy": "standard"}


# ═══════════════════════════════════════════════════════════════════
# 2. Intent Understanding Module
# ═══════════════════════════════════════════════════════════════════

class IntentModule:
    """Classify user intent across 8 high-level categories.

    Categories (from architecture spec):
      learning, research, planning, decision, coding, writing, support, exploration

    Also maps to personality mode and decoder persona.
    """

    _PATTERNS = {
        "learning":    re.compile(
            r"\b(learn|study|understand|teach|explain|how does|what is|tutorial|guide)\b", re.I),
        "research":    re.compile(
            r"\b(research|investigate|analyze|find|search|evidence|sources|compare)\b", re.I),
        "planning":    re.compile(
            r"\b(plan|schedule|organize|roadmap|timeline|steps|agenda|next steps)\b", re.I),
        "decision":    re.compile(
            r"\b(decide|choose|should I|which|best option|recommendation|pros and cons)\b", re.I),
        "coding":      re.compile(
            r"\b(code|program|function|bug|debug|implement|script|class|api|error)\b", re.I),
        "writing":     re.compile(
            r"\b(write|draft|essay|report|summarize|compose|document|article|email)\b", re.I),
        "support":     re.compile(
            r"\b(help|support|problem|issue|not working|broken|fix|error|trouble)\b", re.I),
        "exploration": re.compile(
            r"\b(tell me|curious|wonder|explore|ideas|brainstorm|what if|possibilities)\b", re.I),
    }

    _PERSONA_MAP = {
        "learning":    "teacher",
        "research":    "researcher",
        "planning":    "mentor",
        "decision":    "coach",
        "coding":      "engineer",
        "writing":     "assistant",
        "support":     "supportive",
        "exploration": "mentor",
    }

    def process(self, text: str, history: list[dict] = None) -> dict:
        """Classify intent from user text and conversation history.

        Returns:
            {
                "intent":   str,      # one of 8 categories
                "persona":  str,      # suggested personality mode
                "scores":   dict,     # {category: score}
                "confidence": float,
            }
        """
        combined = text
        if history:
            # Include last 2 turns for context
            for turn in history[-2:]:
                combined += " " + turn.get("content", "")

        scores: dict[str, float] = {}
        for name, pat in self._PATTERNS.items():
            hits = len(pat.findall(combined))
            scores[name] = min(1.0, hits * 0.35)

        best        = max(scores, key=lambda k: scores[k]) if scores else "exploration"
        best_score  = scores.get(best, 0.0)
        if best_score < 0.2:
            best = "exploration"   # default when no clear signal

        return {
            "intent":     best,
            "persona":    self._PERSONA_MAP.get(best, "assistant"),
            "scores":     scores,
            "confidence": round(best_score, 3),
        }


# ═══════════════════════════════════════════════════════════════════
# 3. User Modeling Engine
# ═══════════════════════════════════════════════════════════════════

class UserModelEngine:
    """Persistent user model — skills, preferences, communication style, history.

    Storage: var/user_model.db (SQLite)
    Tables:
      profile      (key TEXT PRIMARY KEY, value TEXT, updated_at REAL)
      skill_levels (domain TEXT PRIMARY KEY, level TEXT, evidence_count INT)
      preferences  (key TEXT PRIMARY KEY, value TEXT)
      history_stats(intent TEXT PRIMARY KEY, count INT, last_seen REAL)
    """

    _LEVELS = ["beginner", "intermediate", "advanced", "expert"]

    def __init__(self):
        ensure_dirs()
        self._db_path = str(USER_MODEL_DB)
        self._conn    = None
        self._lock    = threading.Lock()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        return self._conn

    def _init_db(self):
        with self._lock:
            conn = self._get_conn()
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS profile (
                    key        TEXT PRIMARY KEY,
                    value      TEXT,
                    updated_at REAL DEFAULT (strftime('%s','now'))
                );
                CREATE TABLE IF NOT EXISTS skill_levels (
                    domain         TEXT PRIMARY KEY,
                    level          TEXT DEFAULT 'intermediate',
                    evidence_count INTEGER DEFAULT 0,
                    last_seen      REAL DEFAULT (strftime('%s','now'))
                );
                CREATE TABLE IF NOT EXISTS preferences (
                    key        TEXT PRIMARY KEY,
                    value      TEXT,
                    updated_at REAL DEFAULT (strftime('%s','now'))
                );
                CREATE TABLE IF NOT EXISTS history_stats (
                    intent   TEXT PRIMARY KEY,
                    count    INTEGER DEFAULT 0,
                    last_seen REAL DEFAULT (strftime('%s','now'))
                );
            """)
            conn.commit()

    def set_profile(self, key: str, value: str):
        with self._lock:
            self._get_conn().execute(
                "INSERT OR REPLACE INTO profile (key, value, updated_at) VALUES (?,?,?)",
                (key, str(value), time.time())
            )
            self._get_conn().commit()

    def get_profile(self, key: str = None) -> dict:
        with self._lock:
            if key:
                row = self._get_conn().execute(
                    "SELECT value FROM profile WHERE key=?", (key,)
                ).fetchone()
                return {key: row[0]} if row else {}
            rows = self._get_conn().execute(
                "SELECT key, value FROM profile"
            ).fetchall()
            return {k: v for k, v in rows}

    def set_preference(self, key: str, value: str):
        with self._lock:
            self._get_conn().execute(
                "INSERT OR REPLACE INTO preferences (key, value, updated_at) VALUES (?,?,?)",
                (key, str(value), time.time())
            )
            self._get_conn().commit()

    def get_preference(self, key: str, default: str = None) -> str | None:
        with self._lock:
            row = self._get_conn().execute(
                "SELECT value FROM preferences WHERE key=?", (key,)
            ).fetchone()
            return row[0] if row else default

    def update_skill(self, domain: str, level: str = None):
        """Record that user demonstrated knowledge in a domain.
        If level provided, set it explicitly; else increment evidence_count.
        """
        with self._lock:
            if level and level in self._LEVELS:
                self._get_conn().execute("""
                    INSERT OR REPLACE INTO skill_levels
                        (domain, level, evidence_count, last_seen)
                    VALUES (?, ?, COALESCE(
                        (SELECT evidence_count+1 FROM skill_levels WHERE domain=?), 1
                    ), ?)
                """, (domain, level, domain, time.time()))
            else:
                self._get_conn().execute("""
                    INSERT INTO skill_levels (domain, level, evidence_count, last_seen)
                    VALUES (?, 'intermediate', 1, ?)
                    ON CONFLICT(domain) DO UPDATE SET
                        evidence_count = evidence_count + 1,
                        last_seen = excluded.last_seen
                """, (domain, time.time()))
            self._get_conn().commit()

    def get_skill(self, domain: str) -> dict:
        with self._lock:
            row = self._get_conn().execute(
                "SELECT level, evidence_count FROM skill_levels WHERE domain=?",
                (domain,)
            ).fetchone()
            return {"level": row[0], "evidence": row[1]} if row else {
                "level": "intermediate", "evidence": 0
            }

    def record_intent(self, intent: str):
        with self._lock:
            self._get_conn().execute("""
                INSERT INTO history_stats (intent, count, last_seen) VALUES (?,1,?)
                ON CONFLICT(intent) DO UPDATE SET
                    count = count + 1, last_seen = excluded.last_seen
            """, (intent, time.time()))
            self._get_conn().commit()

    def get_summary(self) -> dict:
        with self._lock:
            conn = self._get_conn()
            profile = {k: v for k, v in
                       conn.execute("SELECT key,value FROM profile").fetchall()}
            skills  = {d: l for d, l, in
                       conn.execute("SELECT domain,level FROM skill_levels").fetchall()}
            intents = {i: c for i, c in
                       conn.execute("SELECT intent,count FROM history_stats").fetchall()}
            return {
                "profile": profile,
                "skills":  skills,
                "top_intents": sorted(intents.items(), key=lambda x: -x[1])[:5],
            }


# ═══════════════════════════════════════════════════════════════════
# 4. Goal Tracking Engine
# ═══════════════════════════════════════════════════════════════════

class GoalTracker:
    """Track user goals, milestones, blockers, and progress.

    Goals are stored in SQLite. Active goals surface in context assembly
    so the model can reference them in responses.
    """

    def __init__(self, user_model: UserModelEngine):
        self._db   = user_model._get_conn
        self._lock = user_model._lock
        self._init_tables()

    def _init_tables(self):
        with self._lock:
            self._db().executescript("""
                CREATE TABLE IF NOT EXISTS goals (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    title       TEXT NOT NULL,
                    description TEXT,
                    status      TEXT DEFAULT 'active',
                    progress    REAL DEFAULT 0.0,
                    created_at  REAL DEFAULT (strftime('%s','now')),
                    updated_at  REAL DEFAULT (strftime('%s','now'))
                );
                CREATE TABLE IF NOT EXISTS milestones (
                    id       INTEGER PRIMARY KEY AUTOINCREMENT,
                    goal_id  INTEGER REFERENCES goals(id),
                    text     TEXT NOT NULL,
                    done     INTEGER DEFAULT 0,
                    created_at REAL DEFAULT (strftime('%s','now'))
                );
                CREATE TABLE IF NOT EXISTS blockers (
                    id       INTEGER PRIMARY KEY AUTOINCREMENT,
                    goal_id  INTEGER REFERENCES goals(id),
                    text     TEXT NOT NULL,
                    resolved INTEGER DEFAULT 0,
                    created_at REAL DEFAULT (strftime('%s','now'))
                );
            """)
            self._db().commit()

    def add_goal(self, title: str, description: str = "") -> int:
        with self._lock:
            cur = self._db().execute(
                "INSERT INTO goals (title, description) VALUES (?,?)",
                (title, description)
            )
            self._db().commit()
            return cur.lastrowid

    def update_progress(self, goal_id: int, progress: float):
        progress = max(0.0, min(1.0, progress))
        with self._lock:
            self._db().execute(
                "UPDATE goals SET progress=?, updated_at=? WHERE id=?",
                (progress, time.time(), goal_id)
            )
            if progress >= 1.0:
                self._db().execute(
                    "UPDATE goals SET status='completed' WHERE id=?", (goal_id,)
                )
            self._db().commit()

    def add_milestone(self, goal_id: int, text: str) -> int:
        with self._lock:
            cur = self._db().execute(
                "INSERT INTO milestones (goal_id, text) VALUES (?,?)",
                (goal_id, text)
            )
            self._db().commit()
            return cur.lastrowid

    def add_blocker(self, goal_id: int, text: str) -> int:
        with self._lock:
            cur = self._db().execute(
                "INSERT INTO blockers (goal_id, text) VALUES (?,?)",
                (goal_id, text)
            )
            self._db().commit()
            return cur.lastrowid

    def active_goals(self, limit: int = 5) -> list[dict]:
        with self._lock:
            rows = self._db().execute("""
                SELECT id, title, description, progress, created_at
                FROM goals WHERE status='active'
                ORDER BY updated_at DESC LIMIT ?
            """, (limit,)).fetchall()
            return [
                {"id": r[0], "title": r[1], "description": r[2],
                 "progress": r[3], "created_at": r[4]}
                for r in rows
            ]

    def context_snippet(self) -> str:
        """Return a brief context string for injection into system prompt."""
        goals = self.active_goals(3)
        if not goals:
            return ""
        parts = [f"Active goals: " + "; ".join(
            f"{g['title']} ({int(g['progress']*100)}%)" for g in goals
        )]
        return " | ".join(parts)


# ═══════════════════════════════════════════════════════════════════
# 5. Teaching Intelligence Engine
# ═══════════════════════════════════════════════════════════════════

class TeachingEngine:
    """Adaptive teaching system.

    Capabilities:
      - Knowledge gap detection from user questions
      - Adaptive difficulty based on user skill profile
      - Curriculum generation (topic → subtopics → exercises)
      - Quiz generation
      - Progress tracking
    """

    _DIFFICULTY_INDICATORS = {
        "beginner":     ["what is", "how do i", "explain", "basic", "simple", "start"],
        "intermediate": ["how does", "why does", "compare", "difference", "implement"],
        "advanced":     ["optimize", "internals", "architecture", "design", "trade-off"],
        "expert":       ["formal proof", "complexity", "benchmark", "cutting edge"],
    }

    def __init__(self, cfg: dict, user_model: UserModelEngine):
        self._cfg        = cfg
        self._user_model = user_model

    def detect_level(self, text: str, domain: str = "general") -> str:
        """Infer user knowledge level from their question/text."""
        text_l = text.lower()
        for level in ["expert", "advanced", "intermediate", "beginner"]:
            indicators = self._DIFFICULTY_INDICATORS[level]
            if any(ind in text_l for ind in indicators):
                return level
        # Fall back to stored skill level for this domain
        return self._user_model.get_skill(domain).get("level", "intermediate")

    def detect_gaps(self, text: str) -> list[str]:
        """Identify apparent knowledge gaps from user text."""
        gaps = []
        gap_signals = [
            (r"what (?:is|are|does?) (.{3,50})\?", "concept"),
            (r"don'?t understand (.{3,50})", "topic"),
            (r"confused about (.{3,50})", "topic"),
            (r"how (?:do|does|can) (.{3,50})\?", "process"),
        ]
        for pattern, gap_type in gap_signals:
            matches = re.findall(pattern, text, re.I)
            for m in matches:
                gaps.append(m.strip()[:50])
        return gaps[:5]

    def build_teaching_context(self, intent: str, text: str,
                                user_level: str) -> dict:
        """Build teaching context to inject into decoder prompt."""
        gaps   = self.detect_gaps(text)
        level  = user_level or self.detect_level(text)
        cfg_lv = self._cfg.get("levels", [])
        idx    = cfg_lv.index(level) if level in cfg_lv else 1

        return {
            "mode":       "teaching",
            "level":      level,
            "level_idx":  idx,
            "gaps":       gaps,
            "adaptive":   self._cfg.get("adaptive", True),
            "directive":  self._level_directive(level),
        }

    def _level_directive(self, level: str) -> str:
        directives = {
            "beginner":     "Explain clearly using simple language and concrete examples.",
            "intermediate": "Assume foundational knowledge. Use examples and analogies.",
            "advanced":     "Assume solid understanding. Focus on nuance and edge cases.",
            "expert":       "Be precise and technical. Skip basics entirely.",
        }
        return directives.get(level, directives["intermediate"])

    def generate_quiz(self, topic: str, level: str = "intermediate") -> list[dict]:
        """Generate quiz questions for a topic (template-based for now).
        Full neural generation enabled after model is trained.
        """
        templates = {
            "beginner": [
                f"What is the basic definition of {topic}?",
                f"Can you give an example of {topic} in everyday terms?",
            ],
            "intermediate": [
                f"How does {topic} work internally?",
                f"What are the key trade-offs when using {topic}?",
            ],
            "advanced": [
                f"Compare the different approaches to {topic} and their complexities.",
                f"When would you NOT use {topic}, and why?",
            ],
            "expert": [
                f"What are the theoretical limits of {topic}?",
                f"Describe a non-obvious failure mode of {topic}.",
            ],
        }
        questions = templates.get(level, templates["intermediate"])
        return [{"question": q, "topic": topic, "level": level}
                for q in questions]


# ═══════════════════════════════════════════════════════════════════
# 6. Decision Support Engine
# ═══════════════════════════════════════════════════════════════════

class DecisionSupportEngine:
    """Structured decision-making support.

    When intent == "decision", this engine:
      - Extracts options from user text
      - Frames trade-offs and risks
      - Structures evidence for comparison
      - Formats recommendation context for the decoder
    """

    _OPTION_PATTERNS = [
        re.compile(r"\b(?:option|choice|alternative|approach)\s+\w*[:\-]\s*(.{5,80})", re.I),
        re.compile(r"\b(?:should I|whether to)\s+(.{5,60})\s+or\s+(.{5,60})", re.I),
    ]

    def extract_options(self, text: str) -> list[str]:
        """Extract decision options from user text."""
        options = []
        for pat in self._OPTION_PATTERNS:
            for match in pat.findall(text):
                if isinstance(match, tuple):
                    options.extend([m.strip() for m in match if m.strip()])
                else:
                    options.append(match.strip())
        return options[:6]

    def build_decision_context(self, text: str,
                                chunks: list[dict] = None) -> dict:
        """Build structured decision context for decoder prompt."""
        options   = self.extract_options(text)
        evidence  = []
        if chunks:
            for chunk in chunks[:3]:
                evidence.append(chunk.get("text", "")[:200])

        return {
            "mode":     "decision_support",
            "options":  options,
            "evidence": evidence,
            "directive": (
                "Structure your response as: 1) Analyse each option, "
                "2) Key trade-offs, 3) Risks, 4) Recommendation with reasoning."
            ) if options else "Provide a structured analysis.",
        }


# ═══════════════════════════════════════════════════════════════════
# 7. Reflection Engine
# ═══════════════════════════════════════════════════════════════════

class ReflectionEngine:
    """Pre-response self-review (runs before final generation).

    Checks:
      - Consistency with recent memory
      - Citation coverage (are retrieved chunks referenced?)
      - Confidence calibration (low evidence → hedge)
      - Response completeness (does it address the query?)
    """

    def __init__(self, cfg: dict):
        self._cfg         = cfg
        self._max_rounds  = cfg.get("max_rounds", 2)

    def review(self, query: str, planned_response: str,
               chunks: list[dict], memories: list[dict],
               kg_facts: list[str], confidence: float) -> dict:
        """Review a planned response before generation.

        Returns:
            {
                "should_revise":  bool,
                "issues":         list[str],
                "directives":     list[str],
                "confidence_adj": float,   # adjusted confidence
            }
        """
        issues     = []
        directives = []

        # Consistency check — query terms present in response?
        q_toks = set(re.findall(r"\b[a-z]{3,}\b", query.lower()))
        r_toks = set(re.findall(r"\b[a-z]{3,}\b", planned_response.lower()))
        overlap = len(q_toks & r_toks) / max(len(q_toks), 1)
        if overlap < 0.25:
            issues.append("low_query_coverage")
            directives.append("Ensure the response directly addresses the user's question.")

        # Citation check
        if chunks and self._cfg.get("citation_check", True):
            chunk_terms = set()
            for c in chunks[:3]:
                chunk_terms.update(re.findall(r"\b[a-z]{4,}\b",
                                              c.get("text", "").lower()))
            cited = len(chunk_terms & r_toks) / max(len(chunk_terms), 1)
            if cited < 0.15 and len(chunks) > 0:
                issues.append("low_grounding")
                directives.append("Ground the response in the retrieved evidence.")

        # Confidence calibration
        conf_adj = confidence
        if not chunks and not memories and not kg_facts:
            conf_adj = min(confidence, 0.4)
            if confidence > 0.5:
                issues.append("overconfident_no_evidence")
                directives.append(
                    "Acknowledge uncertainty — no supporting evidence was retrieved."
                )

        return {
            "should_revise":  len(issues) > 0,
            "issues":         issues,
            "directives":     directives,
            "confidence_adj": round(conf_adj, 3),
        }


# ═══════════════════════════════════════════════════════════════════
# 8. Personality Adaptation Engine
# ═══════════════════════════════════════════════════════════════════

class PersonalityEngine:
    """Adapt response style to 7 personality modes.

    Modes (from architecture spec):
      teacher, mentor, engineer, researcher, coach, assistant, supportive

    Each mode sets:
      - System prefix text
      - Response structure preference
      - Verbosity setting
      - Formality level
    """

    _MODE_PROFILES = {
        "teacher": dict(
            prefix="I'll explain this clearly step by step.",
            structure="numbered_steps",
            verbosity="detailed",
            formality="friendly",
        ),
        "mentor": dict(
            prefix="",
            structure="prose_with_questions",
            verbosity="balanced",
            formality="warm",
        ),
        "engineer": dict(
            prefix="",
            structure="technical_precise",
            verbosity="concise",
            formality="technical",
        ),
        "researcher": dict(
            prefix="",
            structure="evidence_based",
            verbosity="comprehensive",
            formality="formal",
        ),
        "coach": dict(
            prefix="",
            structure="action_oriented",
            verbosity="motivating",
            formality="direct",
        ),
        "assistant": dict(
            prefix="",
            structure="natural",
            verbosity="standard",
            formality="friendly",
        ),
        "supportive": dict(
            prefix="I hear you. ",
            structure="empathetic",
            verbosity="gentle",
            formality="warm",
        ),
    }

    def __init__(self, cfg: dict):
        self._cfg     = cfg
        self._default = cfg.get("default_mode", "assistant")

    def get_profile(self, mode: str) -> dict:
        return dict(self._MODE_PROFILES.get(mode, self._MODE_PROFILES["assistant"]))

    def resolve_mode(self, intent_persona: str,
                     emotion_primary: str,
                     user_preference: str = None) -> str:
        """Resolve final personality mode.

        Priority:
          1. Explicit user preference (stored in user model)
          2. Emotion override (e.g., support mode on distress)
          3. Intent-based auto-detection
          4. Default mode from config
        """
        if user_preference and user_preference in self._MODE_PROFILES:
            return user_preference
        if emotion_primary in ("sadness", "anger", "fear", "anxiety"):
            return "supportive"
        valid = self._cfg.get("available_modes", list(self._MODE_PROFILES))
        if intent_persona in valid:
            return intent_persona
        return self._default

    def format_directive(self, mode: str) -> str:
        """Generate a concise formatting directive for the decoder prompt."""
        profile = self.get_profile(mode)
        parts   = []
        if profile["prefix"]:
            parts.append(f"Start with: '{profile['prefix']}'")
        struct_map = {
            "numbered_steps":    "Use numbered steps.",
            "prose_with_questions": "Use warm prose; end with a guiding question.",
            "technical_precise": "Be precise and technical. Use code blocks when relevant.",
            "evidence_based":    "Cite evidence. Structure: claim → evidence → conclusion.",
            "action_oriented":   "Focus on actionable next steps.",
            "empathetic":        "Acknowledge feelings before providing information.",
        }
        struct = struct_map.get(profile["structure"], "")
        if struct:
            parts.append(struct)
        return " ".join(parts)


# ═══════════════════════════════════════════════════════════════════
# 9. Context Awareness Engine
# ═══════════════════════════════════════════════════════════════════

class ContextAwarenessEngine:
    """Fuse session, memory, workspace, goal, and project context.

    Produces a concise context summary string for injection into the
    decoder's system prompt. Keeps token budget tight.
    """

    def build(self,
              session_id:   str,
              history:      list[dict],
              memories:     list[dict],
              workspace:    str = None,
              goals_snippet: str = "",
              intent:       str = "",
              emotion:      str = "neutral") -> dict:
        """Build context awareness summary.

        Returns:
            {
                "session_summary": str,
                "memory_count":    int,
                "has_workspace":   bool,
                "goals_active":    bool,
                "context_tags":    list[str],
            }
        """
        tags = []
        if len(history) > 5:
            tags.append("ongoing_conversation")
        if memories:
            tags.append("memory_grounded")
        if workspace:
            tags.append(f"workspace:{workspace}")
        if goals_snippet:
            tags.append("goal_tracking_active")
        if intent != "exploration":
            tags.append(f"intent:{intent}")
        if emotion not in ("neutral", ""):
            tags.append(f"emotion:{emotion}")

        return {
            "session_summary": f"Turn {len(history)+1} | intent={intent} | emotion={emotion}",
            "memory_count":    len(memories),
            "has_workspace":   bool(workspace),
            "goals_active":    bool(goals_snippet),
            "context_tags":    tags,
        }


# ═══════════════════════════════════════════════════════════════════
# CognitionLayer — Main Orchestrator
# ═══════════════════════════════════════════════════════════════════

class CognitionLayer:
    """Human Cognition Layer orchestrator.

    Called by engine.py as part of build_context().
    All modules are lazy-initialized; missing dependencies degrade gracefully.

    Usage:
        layer  = CognitionLayer()
        result = layer.process(
            text     = user_message,
            session  = session_id,
            history  = conversation_history,
            chunks   = retrieval_chunks,
            memories = memory_hits,
            kg_facts = kg_results,
            workspace= active_workspace,
        )
        # result["prompt_addendum"] → inject into decoder system prompt
        # result["emotion"]        → detected emotion dict
        # result["intent"]         → detected intent dict
        # result["personality"]    → resolved personality mode + profile
    """

    def __init__(self, cfg: dict = None):
        self._cfg       = cfg or COGNITION
        self._user_model= UserModelEngine()
        self._emotion   = EmotionModule(self._cfg.get("emotion", {}))
        self._intent    = IntentModule()
        self._user_mod  = self._user_model   # alias
        self._goals     = GoalTracker(self._user_model)
        self._teaching  = TeachingEngine(
            self._cfg.get("teaching", {}), self._user_model
        )
        self._decision  = DecisionSupportEngine()
        self._reflection= ReflectionEngine(self._cfg.get("reflection", {}))
        self._personality= PersonalityEngine(self._cfg.get("personality", {}))
        self._context   = ContextAwarenessEngine()

    def process(self,
                text:      str,
                session:   str = "default",
                history:   list[dict] = None,
                chunks:    list[dict] = None,
                memories:  list[dict] = None,
                kg_facts:  list[str]  = None,
                workspace: str        = None,
                confidence: float     = 0.5,
                ) -> dict:
        """Run all active cognition modules and return unified result dict.

        Safe to call even when mode == 'off' — returns empty neutral result.
        """
        history  = history  or []
        chunks   = chunks   or []
        memories = memories or []
        kg_facts = kg_facts or []

        # Neutral result (returned when mode == 'off')
        result: dict[str, Any] = {
            "mode":           _load_state().get("mode", "full"),
            "emotion":        {"primary": "neutral", "tone": "standard",
                               "verbosity": "standard", "prefix": "",
                               "confidence": 0.0},
            "intent":         {"intent": "exploration", "persona": "assistant",
                               "confidence": 0.0},
            "user_level":     "intermediate",
            "personality":    {"mode": "assistant"},
            "teaching":       {},
            "decision":       {},
            "goals":          "",
            "context":        {},
            "reflection":     {"should_revise": False, "issues": [],
                               "directives": [], "confidence_adj": confidence},
            "prompt_addendum": "",
        }

        if result["mode"] == "off":
            return result

        # ── Module 1: Emotion ─────────────────────────────────────────
        if is_module_active("emotion"):
            result["emotion"] = self._emotion.process(text)

        # ── Module 2: Intent ──────────────────────────────────────────
        if is_module_active("intent"):
            result["intent"] = self._intent.process(text, history)

        # ── Module 3: User Modeling ───────────────────────────────────
        if is_module_active("user_modeling"):
            intent_str = result["intent"]["intent"]
            self._user_model.record_intent(intent_str)
            # Infer domain from intent + text
            domain = self._infer_domain(text, intent_str)
            level  = self._user_model.get_skill(domain).get("level", "intermediate")
            result["user_level"] = level

        # ── Module 4: Goal Tracking ───────────────────────────────────
        if is_module_active("goal_tracking"):
            result["goals"] = self._goals.context_snippet()

        # ── Module 5: Teaching Intelligence ──────────────────────────
        if is_module_active("teaching") and result["intent"]["intent"] == "learning":
            result["teaching"] = self._teaching.build_teaching_context(
                result["intent"]["intent"], text, result["user_level"]
            )

        # ── Module 6: Decision Support ────────────────────────────────
        if is_module_active("decision_support") \
                and result["intent"]["intent"] == "decision":
            result["decision"] = self._decision.build_decision_context(
                text, chunks
            )

        # ── Module 8: Personality ─────────────────────────────────────
        if is_module_active("personality"):
            pref_mode  = self._user_model.get_preference("personality_mode")
            mode       = self._personality.resolve_mode(
                result["intent"]["persona"],
                result["emotion"]["primary"],
                pref_mode,
            )
            profile    = self._personality.get_profile(mode)
            directive  = self._personality.format_directive(mode)
            result["personality"] = {"mode": mode, **profile,
                                     "directive": directive}

        # ── Module 9: Context Awareness ───────────────────────────────
        if is_module_active("context_aware"):
            result["context"] = self._context.build(
                session_id    = session,
                history       = history,
                memories      = memories,
                workspace     = workspace,
                goals_snippet = result["goals"],
                intent        = result["intent"]["intent"],
                emotion       = result["emotion"]["primary"],
            )

        # ── Module 7: Reflection (runs last, before generation) ───────
        if is_module_active("reflection"):
            placeholder = ""   # pre-generation; no response text yet
            result["reflection"] = self._reflection.review(
                text, placeholder, chunks, memories, kg_facts, confidence
            )

        # ── Build prompt addendum ─────────────────────────────────────
        result["prompt_addendum"] = self._build_addendum(result, text)

        return result

    def _infer_domain(self, text: str, intent: str) -> str:
        """Infer knowledge domain from text + intent for skill tracking."""
        domain_signals = {
            "coding":      ["python","javascript","code","function","class","api"],
            "mathematics": ["equation","formula","calculus","algebra","statistics"],
            "science":     ["physics","chemistry","biology","quantum","molecule"],
            "writing":     ["essay","paragraph","grammar","style","prose"],
            "ml":          ["machine learning","neural","transformer","model","training"],
        }
        text_l = text.lower()
        for domain, signals in domain_signals.items():
            if any(s in text_l for s in signals):
                return domain
        return intent if intent != "exploration" else "general"

    def _build_addendum(self, result: dict, query: str) -> str:
        """Build the system-prompt addendum injected into the decoder context.
        Stays within cognition_budget_tokens (128 tokens ≈ ~500 chars).
        """
        parts = []

        # Emotion prefix
        prefix = result["emotion"].get("prefix", "")
        if prefix:
            parts.append(f"[Response tone: {result['emotion'].get('tone','standard')}. "
                         f"Open with: {prefix.strip()}]")

        # Personality directive
        persona_dir = result.get("personality", {}).get("directive", "")
        if persona_dir:
            parts.append(f"[Style: {persona_dir}]")

        # Teaching directive
        if result.get("teaching"):
            parts.append(f"[Teaching: {result['teaching'].get('directive','')}]")

        # Decision support directive
        if result.get("decision"):
            parts.append(f"[Decision: {result['decision'].get('directive','')}]")

        # Reflection directives
        for d in result.get("reflection", {}).get("directives", []):
            parts.append(f"[Reflection: {d}]")

        # Goals context
        if result.get("goals"):
            parts.append(f"[Context: {result['goals']}]")

        addendum = " ".join(parts)
        return addendum[:500]   # hard cap at ~500 chars / ~128 tokens

    # ── State management API ─────────────────────────────────────────
    def set_mode(self, mode: str) -> dict:
        return set_mode(mode)

    def set_module(self, name: str, enabled: bool) -> dict:
        return set_module(name, enabled)

    def get_state(self) -> dict:
        return get_state()

    def reset(self) -> dict:
        return reset_state()

    # ── User model convenience API ────────────────────────────────────
    def set_user_preference(self, key: str, value: str):
        self._user_model.set_preference(key, value)

    def get_user_summary(self) -> dict:
        return self._user_model.get_summary()

    def add_goal(self, title: str, description: str = "") -> int:
        return self._goals.add_goal(title, description)

    def active_goals(self) -> list[dict]:
        return self._goals.active_goals()
