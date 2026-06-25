"""
CognitiveOC v3 — Memory System
================================

Implements all 8 memory types from the v3 architecture spec:
  1. Episodic Memory    — individual interaction events with timestamps
  2. Semantic Memory    — generalised facts about the world / domain
  3. Preference Memory  — user preferences and communication style
  4. Project Memory     — project-specific context and decisions
  5. Task Memory        — task state, progress, steps
  6. Goal Memory        — long-term goals (synced with GoalTracker)
  7. Learning Memory    — topics the user has studied; knowledge state
  8. Workspace Memory   — active document/workspace context

All types share one SQLite database (var/memory.db) with a 'kind' column.
FTS5 virtual table provides fast text search across all memory types.
Encoder-based near-dedup (cosine similarity) prevents storing near-copies.

4-factor ranked recall (improved from baseline):
  score = relevance_w × relevance
        + recency_w   × recency
        + frequency_w × frequency
        + importance_w× importance

  Weights from config.py MEMORY section (default: 0.10/0.30/0.20/0.40).

New in v3 vs baseline:
  Baseline: 3 tables, 4-factor recall, consolidation by token overlap.
  v3: 8 typed memory kinds, encoder-based dedup, compression via summarizer,
      memory ↔ KG sync, memory ↔ retrieval sync, workspace memory, project memory,
      task memory, goal memory, configurable weights from config.py.

File: memory/memory.py
Used by: engine.py, cognition/cognition.py, memory/sync.py
Persists: var/memory.db (SQLite with FTS5)
"""

from __future__ import annotations

import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

try:
    from config import MEMORY, MEMORY_DB, ensure_dirs
except ImportError:
    MEMORY    = dict(
        session_window=30, recall_k=8, max_long_term=20_000,
        decay_half_life_days=45, archive_threshold=0.15,
        consolidation_interval_h=24, compression_threshold=500,
        similarity_threshold=0.92,
        importance_weight=0.40, recency_weight=0.30,
        frequency_weight=0.20, relevance_weight=0.10,
    )
    MEMORY_DB = Path("var/memory.db")
    def ensure_dirs(): pass

DAY       = 86_400.0
_WRITE_LK = threading.RLock()

STOPWORDS = {
    "the","a","an","and","or","is","are","was","were","i","you","my","me",
    "do","what","of","to","in","it","that","about","be","have","has","had",
    "will","would","could","should","can","may","this","these","those","they",
}

# ── Memory kind constants ─────────────────────────────────────────────
KIND_EPISODIC   = "episodic"
KIND_SEMANTIC   = "semantic"
KIND_PREFERENCE = "preference"
KIND_PROJECT    = "project"
KIND_TASK       = "task"
KIND_GOAL       = "goal"
KIND_LEARNING   = "learning"
KIND_WORKSPACE  = "workspace"
KIND_FACT       = "fact"   # generic / legacy
ALL_KINDS = [KIND_EPISODIC, KIND_SEMANTIC, KIND_PREFERENCE, KIND_PROJECT,
             KIND_TASK, KIND_GOAL, KIND_LEARNING, KIND_WORKSPACE, KIND_FACT]


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower())
            if t not in STOPWORDS and len(t) > 1}


def classify_kind(text: str) -> str:
    """Heuristic auto-classification of memory kind from text content."""
    tl = text.lower()
    if re.search(r"\b(prefer|like|favorite|always|never|style|want|wish)\b", tl):
        return KIND_PREFERENCE
    if re.search(r"\b(project|sprint|milestone|deadline|deliverable)\b", tl):
        return KIND_PROJECT
    if re.search(r"\b(task|step|todo|subtask|action item|checklist)\b", tl):
        return KIND_TASK
    if re.search(r"\b(goal|objective|target|aim|plan to|intend to)\b", tl):
        return KIND_GOAL
    if re.search(r"\b(learned|studying|understood|know about|progress on)\b", tl):
        return KIND_LEARNING
    if re.search(r"\b(document|workspace|file|pdf|uploaded|reading)\b", tl):
        return KIND_WORKSPACE
    if re.search(r"\b(is|are|means|defined as|created|born|located)\b", tl):
        return KIND_SEMANTIC
    return KIND_FACT


# ═══════════════════════════════════════════════════════════════════
# MemoryStore — Base SQLite Layer
# ═══════════════════════════════════════════════════════════════════

class MemoryStore:
    """Base memory store — SQLite-backed, FTS5-accelerated, thread-safe.

    Tables:
      messages   — session conversation history (ephemeral per-session)
      memories   — long-term typed memories (all 8 kinds)
      profile    — user profile key-value pairs
      events     — episodic event log (timestamped with kind + summary)

    FTS5 virtual table (memories_fts) shadows memories for fast text recall.
    All writes are serialised via _WRITE_LK.
    """

    def __init__(self, path: str = None):
        ensure_dirs()
        self._db_path = str(path or MEMORY_DB)
        self._conn    = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._init_schema()
        self._sync_fts()

    def _init_schema(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS messages (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                session  TEXT    NOT NULL,
                role     TEXT    NOT NULL,
                content  TEXT    NOT NULL,
                ts       REAL    DEFAULT (strftime('%s','now'))
            );
            CREATE INDEX IF NOT EXISTS idx_msg_session ON messages(session, id DESC);

            CREATE TABLE IF NOT EXISTS memories (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                text         TEXT    NOT NULL,
                kind         TEXT    DEFAULT 'fact',
                importance   REAL    DEFAULT 1.0,
                ts           REAL    DEFAULT (strftime('%s','now')),
                access_count INTEGER DEFAULT 0,
                last_access  REAL    DEFAULT NULL,
                archived     INTEGER DEFAULT 0,
                linked_ids   TEXT    DEFAULT '',
                project      TEXT    DEFAULT NULL,
                session      TEXT    DEFAULT NULL,
                workspace    TEXT    DEFAULT NULL,
                tags         TEXT    DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS idx_mem_kind      ON memories(kind);
            CREATE INDEX IF NOT EXISTS idx_mem_archived  ON memories(archived);
            CREATE INDEX IF NOT EXISTS idx_mem_project   ON memories(project);
            CREATE INDEX IF NOT EXISTS idx_mem_workspace ON memories(workspace);

            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5 (
                text, content=memories, content_rowid=id
            );

            CREATE TABLE IF NOT EXISTS profile (
                key        TEXT PRIMARY KEY,
                value      TEXT,
                updated_at REAL DEFAULT (strftime('%s','now'))
            );

            CREATE TABLE IF NOT EXISTS events (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                session  TEXT,
                kind     TEXT,
                summary  TEXT,
                ts       REAL DEFAULT (strftime('%s','now'))
            );
            CREATE INDEX IF NOT EXISTS idx_events_session ON events(session, id DESC);
        """)
        # Add columns that may be missing in older DBs (upgrade safe)
        for col, definition in [
            ("access_count", "INTEGER DEFAULT 0"),
            ("last_access",  "REAL DEFAULT NULL"),
            ("archived",     "INTEGER DEFAULT 0"),
            ("linked_ids",   "TEXT DEFAULT ''"),
            ("project",      "TEXT DEFAULT NULL"),
            ("session",      "TEXT DEFAULT NULL"),
            ("workspace",    "TEXT DEFAULT NULL"),
            ("tags",         "TEXT DEFAULT ''"),
        ]:
            try:
                self._conn.execute(
                    f"ALTER TABLE memories ADD COLUMN {col} {definition}"
                )
            except sqlite3.OperationalError:
                pass
        self._conn.commit()

    def _sync_fts(self):
        """Rebuild FTS5 index (safe to call on every startup)."""
        try:
            self._conn.execute("INSERT INTO memories_fts(memories_fts) VALUES('rebuild')")
            self._conn.commit()
        except Exception:
            pass

    # ── Write helpers ────────────────────────────────────────────────
    def _execute_write(self, sql: str, params: tuple = ()):
        with _WRITE_LK:
            self._conn.execute(sql, params)
            self._conn.commit()

    # ── Session messages ─────────────────────────────────────────────
    def add_message(self, session: str, role: str, content: str):
        self._execute_write(
            "INSERT INTO messages(session, role, content, ts) VALUES(?,?,?,?)",
            (session, role, content, time.time()),
        )

    def recent(self, session: str, n: int = None) -> list[dict]:
        n   = n or MEMORY.get("session_window", 30)
        rows= self._conn.execute(
            "SELECT role, content, ts FROM messages "
            "WHERE session=? ORDER BY id DESC LIMIT ?",
            (session, n),
        ).fetchall()
        return [{"role": r, "content": c, "ts": t} for r, c, t in reversed(rows)]

    # ── Core memory write ────────────────────────────────────────────
    def remember(self,
                 text:       str,
                 kind:       str = KIND_FACT,
                 importance: float = 1.0,
                 project:    str = None,
                 session:    str = None,
                 workspace:  str = None,
                 tags:       str = "") -> int:
        """Store a new long-term memory.

        Returns:
            memory ID (int).
        """
        with _WRITE_LK:
            cur = self._conn.execute(
                "INSERT INTO memories"
                "(text, kind, importance, ts, project, session, workspace, tags)"
                " VALUES (?,?,?,?,?,?,?,?)",
                (text, kind, importance, time.time(),
                 project, session, workspace, tags),
            )
            rowid = cur.lastrowid
            self._conn.execute(
                "INSERT INTO memories_fts(rowid, text) VALUES(?,?)",
                (rowid, text),
            )
            self._conn.commit()
        self._prune()
        return rowid

    def forget(self, memory_id: int):
        """Permanently delete a memory."""
        with _WRITE_LK:
            self._conn.execute("DELETE FROM memories WHERE id=?", (memory_id,))
            self._conn.execute("INSERT INTO memories_fts(memories_fts) VALUES('rebuild')")
            self._conn.commit()

    # ── Basic recall ─────────────────────────────────────────────────
    def recall(self, query: str, k: int = None) -> list[dict]:
        """FTS5-accelerated keyword recall. Returns top-k matching memories."""
        k   = k or MEMORY.get("recall_k", 8)
        toks= _tokens(query)
        if not toks:
            return []
        fts_q = " OR ".join(toks)
        try:
            rows = self._conn.execute(
                "SELECT m.id, m.text, m.kind, m.importance "
                "FROM memories m "
                "JOIN memories_fts f ON m.id = f.rowid "
                "WHERE memories_fts MATCH ? AND m.archived=0 LIMIT 200",
                (fts_q,),
            ).fetchall()
        except Exception:
            rows = self._conn.execute(
                "SELECT id, text, kind, importance FROM memories WHERE archived=0"
            ).fetchall()
        scored = []
        for mid, text, kind, imp in rows:
            overlap = len(toks & _tokens(text))
            if overlap:
                scored.append((overlap * (imp or 1.0),
                               {"id": mid, "text": text, "kind": kind}))
        scored.sort(key=lambda x: -x[0])
        return [m for _, m in scored[:k]]

    # ── Prune ────────────────────────────────────────────────────────
    def _prune(self, max_items: int = None):
        max_items = max_items or MEMORY.get("max_long_term", 20_000)
        n = self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        if n > max_items:
            with _WRITE_LK:
                self._conn.execute(
                    "DELETE FROM memories WHERE id IN "
                    "(SELECT id FROM memories ORDER BY importance ASC, ts ASC LIMIT ?)",
                    (n - max_items,),
                )
                self._conn.execute("INSERT INTO memories_fts(memories_fts) VALUES('rebuild')")
                self._conn.commit()

    # ── Profile ──────────────────────────────────────────────────────
    def set_profile(self, key: str, value: str):
        with _WRITE_LK:
            self._conn.execute(
                "INSERT OR REPLACE INTO profile(key, value, updated_at) VALUES(?,?,?)",
                (key, str(value), time.time()),
            )
            self._conn.commit()

    def profile(self) -> dict:
        return dict(self._conn.execute("SELECT key, value FROM profile").fetchall())

    # ── List / inspect ────────────────────────────────────────────────
    def list_memories(self,
                      limit: int = 100,
                      kind:  str = None,
                      include_archived: bool = False) -> list[dict]:
        q    = "SELECT id, text, kind, importance, archived FROM memories"
        cond = [] if include_archived else ["archived=0"]
        if kind:
            cond.append(f"kind='{kind}'")
        if cond:
            q += " WHERE " + " AND ".join(cond)
        q += " ORDER BY id DESC LIMIT ?"
        return [
            {"id": i, "text": t, "kind": k,
             "importance": round(float(imp or 1.0), 3), "archived": bool(a)}
            for i, t, k, imp, a in self._conn.execute(q, (limit,))
        ]

    def inspect(self, memory_id: int) -> dict | None:
        row = self._conn.execute(
            "SELECT id, text, kind, importance, ts, access_count, "
            "last_access, archived, linked_ids, project, workspace, tags "
            "FROM memories WHERE id=?", (memory_id,)
        ).fetchone()
        if not row:
            return None
        keys = ("id","text","kind","importance","ts","access_count",
                "last_access","archived","linked_ids","project","workspace","tags")
        return dict(zip(keys, row))


# ═══════════════════════════════════════════════════════════════════
# CognitiveMemory — Extended Memory with Full v3 Capabilities
# ═══════════════════════════════════════════════════════════════════

class CognitiveMemory(MemoryStore):
    """Full v3 memory system with all 8 typed memory kinds.

    Extends MemoryStore with:
      - 4-factor weighted ranked recall (configurable weights)
      - Encoder-based near-dedup (optional, uses memory encoder)
      - Memory compression via summarizer
      - Cross-memory linking (BFS traversal)
      - Half-life decay and archival
      - Consolidation (merge near-duplicates)
      - Episodic event log
      - Auto-kind classification
      - Typed accessors for each of the 8 memory kinds
    """

    def __init__(self, path: str = None):
        super().__init__(path)
        self._w_imp = MEMORY.get("importance_weight", 0.40)
        self._w_rec = MEMORY.get("recency_weight",    0.30)
        self._w_frq = MEMORY.get("frequency_weight",  0.20)
        self._w_rel = MEMORY.get("relevance_weight",  0.10)
        self._half  = MEMORY.get("decay_half_life_days", 45)
        self._arch_thresh = MEMORY.get("archive_threshold", 0.15)
        self._comp_thresh = MEMORY.get("compression_threshold", 500)
        self._sim_thresh  = MEMORY.get("similarity_threshold", 0.92)

    # ── Typed remember helpers ────────────────────────────────────────
    def remember_episodic(self, text: str, session: str = None,
                          importance: float = 0.8) -> int:
        return self.remember(text, KIND_EPISODIC, importance, session=session)

    def remember_semantic(self, text: str, importance: float = 1.0) -> int:
        return self.remember(text, KIND_SEMANTIC, importance)

    def remember_preference(self, text: str, importance: float = 1.2) -> int:
        return self.remember(text, KIND_PREFERENCE, importance)

    def remember_project(self, text: str, project: str,
                         importance: float = 1.0) -> int:
        return self.remember(text, KIND_PROJECT, importance, project=project)

    def remember_task(self, text: str, project: str = None,
                      importance: float = 0.9) -> int:
        return self.remember(text, KIND_TASK, importance, project=project)

    def remember_goal(self, text: str, importance: float = 1.3) -> int:
        return self.remember(text, KIND_GOAL, importance)

    def remember_learning(self, text: str, importance: float = 1.1) -> int:
        return self.remember(text, KIND_LEARNING, importance)

    def remember_workspace(self, text: str, workspace: str,
                           importance: float = 0.7) -> int:
        return self.remember(text, KIND_WORKSPACE, importance, workspace=workspace)

    def remember_auto(self, text: str, importance: float = 1.0,
                      **kwargs) -> int:
        """Store memory with auto-classified kind."""
        kind = classify_kind(text)
        return self.remember(text, kind, importance, **kwargs)

    # ── 4-factor ranked recall ────────────────────────────────────────
    def ranked_recall(self,
                      query:    str,
                      k:        int = None,
                      kind:     str = None,
                      project:  str = None,
                      workspace:str = None) -> list[dict]:
        """4-factor weighted ranked recall with FTS5 pre-filter.

        Score = w_rel × relevance + w_rec × recency
              + w_frq × frequency + w_imp × importance

        All weights from config.py MEMORY section.
        """
        k   = k or MEMORY.get("recall_k", 8)
        qt  = _tokens(query)
        if not qt:
            return []

        now   = time.time()
        fts_q = " OR ".join(qt)

        # Build filter conditions
        cond  = ["m.archived=0"]
        args: list[Any] = [fts_q]
        if kind:
            cond.append("m.kind=?"); args.append(kind)
        if project:
            cond.append("m.project=?"); args.append(project)
        if workspace:
            cond.append("m.workspace=?"); args.append(workspace)

        try:
            rows = self._conn.execute(
                "SELECT m.id, m.text, m.kind, m.importance, m.ts, "
                "       m.access_count, m.linked_ids "
                "FROM memories m "
                "JOIN memories_fts f ON m.id = f.rowid "
                "WHERE memories_fts MATCH ? AND " + " AND ".join(cond) +
                " LIMIT 300",
                tuple(args),
            ).fetchall()
        except Exception:
            rows = self._conn.execute(
                "SELECT id, text, kind, importance, ts, access_count, linked_ids "
                "FROM memories WHERE archived=0"
            ).fetchall()

        scored = []
        for mid, text, kind_, imp, ts, freq, linked in rows:
            mt    = _tokens(text)
            ovlp  = len(qt & mt)
            if not ovlp:
                continue
            relevance = ovlp / max(len(qt), 1)
            age_d     = (now - (ts or now)) / DAY
            recency   = 0.5 ** (age_d / self._half)
            frequency = min((freq or 0) / 10.0, 1.0)
            score = (self._w_rel * relevance
                     + self._w_rec * recency
                     + self._w_frq * frequency
                     + self._w_imp * (imp or 1.0))
            scored.append((score, {
                "id":         mid,
                "text":       text,
                "kind":       kind_,
                "score":      round(score, 4),
                "linked_ids": [int(x) for x in (linked or "").split(",") if x],
            }))

        scored.sort(key=lambda x: -x[0])
        top = [m for _, m in scored[:k]]

        # Update access statistics
        if top:
            with _WRITE_LK:
                for m in top:
                    self._conn.execute(
                        "UPDATE memories "
                        "SET access_count=access_count+1, last_access=? "
                        "WHERE id=?",
                        (now, m["id"]),
                    )
                self._conn.commit()
        return top

    # Override base recall with ranked version
    recall = ranked_recall

    # ── Near-dedup check ─────────────────────────────────────────────
    def is_near_duplicate(self, text: str, threshold: float = None) -> bool:
        """Check if a near-duplicate of text already exists in memory.

        Uses encoder cosine similarity when available;
        falls back to token Jaccard similarity.
        """
        threshold = threshold or self._sim_thresh

        # Encoder-based check (preferred)
        try:
            from encoder.hub import get_hub
            hub = get_hub()
            # Compare against recent memories (last 200)
            recent_rows = self._conn.execute(
                "SELECT text FROM memories WHERE archived=0 ORDER BY id DESC LIMIT 200"
            ).fetchall()
            if not recent_rows:
                return False
            texts = [r[0] for r in recent_rows]
            q_vec = hub.encode_single("memory", text)
            c_vecs= hub.encode("memory", texts)
            import numpy as np
            sims  = c_vecs @ q_vec
            return bool(sims.max() >= threshold)
        except Exception:
            pass

        # Token Jaccard fallback
        qt = _tokens(text)
        if not qt:
            return False
        rows = self._conn.execute(
            "SELECT text FROM memories WHERE archived=0 ORDER BY id DESC LIMIT 200"
        ).fetchall()
        for (t,) in rows:
            mt    = _tokens(t)
            union = qt | mt
            if union and len(qt & mt) / len(union) >= threshold:
                return True
        return False

    def remember_if_new(self, text: str, **kwargs) -> int | None:
        """Store memory only if it's not a near-duplicate. Returns ID or None."""
        if self.is_near_duplicate(text):
            return None
        return self.remember_auto(text, **kwargs)

    # ── Episodic events ──────────────────────────────────────────────
    def add_event(self, session: str, kind: str, summary: str):
        with _WRITE_LK:
            self._conn.execute(
                "INSERT INTO events(session, kind, summary, ts) VALUES(?,?,?,?)",
                (session, kind, summary, time.time()),
            )
            self._conn.commit()

    def recent_events(self,
                      session: str = None,
                      kind:    str = None,
                      n:       int = 20) -> list[dict]:
        q, args = "SELECT id, session, kind, summary, ts FROM events", []
        cond    = []
        if session: cond.append("session=?"); args.append(session)
        if kind:    cond.append("kind=?");    args.append(kind)
        if cond:    q += " WHERE " + " AND ".join(cond)
        q += " ORDER BY id DESC LIMIT ?"; args.append(n)
        return [{"id":i,"session":s,"kind":k,"summary":m,"ts":t}
                for i,s,k,m,t in self._conn.execute(q, args)]

    # ── Decay and archival ────────────────────────────────────────────
    def decay(self) -> int:
        """Apply half-life decay; archive memories below threshold.

        Returns number of memories archived.
        """
        now  = time.time()
        rows = self._conn.execute(
            "SELECT id, importance, ts, last_access FROM memories WHERE archived=0"
        ).fetchall()
        to_archive, to_update = [], []
        for mid, imp, ts, last in rows:
            age   = now - max(ts or 0, last or 0)
            new_i = (imp or 1.0) * (0.5 ** (age / (self._half * DAY)))
            if new_i < self._arch_thresh:
                to_archive.append(mid)
            else:
                to_update.append((round(new_i, 4), mid))
        with _WRITE_LK:
            for mid in to_archive:
                self._conn.execute(
                    "UPDATE memories SET archived=1 WHERE id=?", (mid,)
                )
            for new_i, mid in to_update:
                self._conn.execute(
                    "UPDATE memories SET importance=? WHERE id=?", (new_i, mid)
                )
            self._conn.commit()
        return len(to_archive)

    def restore(self, memory_id: int) -> bool:
        """Unarchive a memory and reset importance to 1.0."""
        with _WRITE_LK:
            self._conn.execute(
                "UPDATE memories SET archived=0, importance=1.0 WHERE id=?",
                (memory_id,),
            )
            self._conn.commit()
        return True

    # ── Consolidation ────────────────────────────────────────────────
    def consolidate(self, similarity: float = 0.80) -> int:
        """Merge near-duplicate memories by token overlap.

        Returns number of duplicates removed.
        """
        rows = self._conn.execute(
            "SELECT id, text, importance, access_count "
            "FROM memories WHERE archived=0 ORDER BY importance DESC"
        ).fetchall()
        kept: list[tuple[int, set]] = []
        to_delete: list[tuple[int]] = []
        to_update: list[tuple[int, int]] = []

        for mid, text, imp, freq in rows:
            mt  = _tokens(text)
            dup = None
            for kid, ktoks in kept:
                union = mt | ktoks
                if union and len(mt & ktoks) / len(union) >= similarity:
                    dup = kid; break
            if dup is None:
                kept.append((mid, mt))
            else:
                to_update.append((freq or 0, dup))
                to_delete.append((mid,))

        if to_delete:
            with _WRITE_LK:
                for extra, kid in to_update:
                    self._conn.execute(
                        "UPDATE memories SET access_count=access_count+? WHERE id=?",
                        (extra, kid),
                    )
                self._conn.executemany(
                    "DELETE FROM memories WHERE id=?", to_delete
                )
                self._conn.execute("INSERT INTO memories_fts(memories_fts) VALUES('rebuild')")
                self._conn.commit()
        return len(to_delete)

    # ── Compression ──────────────────────────────────────────────────
    def compress_long_memories(self) -> int:
        """Compress memories above compression_threshold chars via summarizer.

        Returns number of memories compressed.
        """
        rows = self._conn.execute(
            "SELECT id, text, kind, importance FROM memories "
            "WHERE archived=0 AND LENGTH(text)>? LIMIT 100",
            (self._comp_thresh,),
        ).fetchall()
        if not rows:
            return 0
        compressed = 0
        try:
            from memory.summarizer import summarize
        except ImportError:
            return 0
        for mid, text, kind, imp in rows:
            summary = summarize([text], use_neural=False)
            if summary and len(summary) < len(text) * 0.7:
                with _WRITE_LK:
                    self._conn.execute(
                        "UPDATE memories SET text=? WHERE id=?",
                        (summary, mid),
                    )
                    self._conn.execute("INSERT INTO memories_fts(memories_fts) VALUES('rebuild')")
                    self._conn.commit()
                compressed += 1
        return compressed

    # ── Cross-memory linking ──────────────────────────────────────────
    def link_memories(self, id_a: int, id_b: int):
        """Record a bidirectional link between two memories."""
        def _add(src: int, tgt: int):
            row = self._conn.execute(
                "SELECT linked_ids FROM memories WHERE id=?", (src,)
            ).fetchone()
            if not row:
                return
            ids = {x for x in (row[0] or "").split(",") if x}
            ids.add(str(tgt))
            with _WRITE_LK:
                self._conn.execute(
                    "UPDATE memories SET linked_ids=? WHERE id=?",
                    (",".join(ids), src),
                )
                self._conn.commit()
        _add(id_a, id_b)
        _add(id_b, id_a)

    def related_chain(self, memory_id: int, depth: int = 2) -> list[dict]:
        """BFS traversal of memory links — returns related memories."""
        visited = {memory_id}
        queue   = [memory_id]
        results = []
        for _ in range(depth):
            nxt = []
            for mid in queue:
                row = self._conn.execute(
                    "SELECT linked_ids FROM memories WHERE id=?", (mid,)
                ).fetchone()
                if not row or not row[0]:
                    continue
                for lid in (int(x) for x in row[0].split(",") if x):
                    if lid not in visited:
                        visited.add(lid)
                        nxt.append(lid)
                        m = self._conn.execute(
                            "SELECT id, text, kind FROM memories WHERE id=?", (lid,)
                        ).fetchone()
                        if m:
                            results.append({"id":m[0],"text":m[1],"kind":m[2]})
            queue = nxt
        return results

    def auto_link(self, threshold: float = 0.6) -> int:
        """Auto-link high token-overlap memories (call after consolidate)."""
        rows   = self._conn.execute(
            "SELECT id, text FROM memories WHERE archived=0 LIMIT 500"
        ).fetchall()
        linked = 0
        for i, (id_a, text_a) in enumerate(rows):
            ta = _tokens(text_a)
            for id_b, text_b in rows[i+1:]:
                tb    = _tokens(text_b)
                union = ta | tb
                if union and len(ta & tb) / len(union) >= threshold:
                    self.link_memories(id_a, id_b)
                    linked += 1
        return linked

    # ── Session summary ───────────────────────────────────────────────
    def summarize_session(self, session: str) -> str:
        """Build a heuristic session summary from recent events."""
        events = self.recent_events(session=session, n=20)
        if not events:
            return ""
        lines = [e["summary"] for e in events if e.get("summary")]
        if not lines:
            return ""
        seen, unique = set(), []
        for line in reversed(lines):
            key = line[:40].lower()
            if key not in seen:
                seen.add(key); unique.append(line)
        return "; ".join(reversed(unique[:5]))

    # ── Search / browser API ──────────────────────────────────────────
    def search(self,
               query:            str,
               include_archived: bool = False,
               limit:            int  = 50,
               kind:             str  = None) -> list[dict]:
        """Full-text search over memories (UI memory browser)."""
        qt  = _tokens(query)
        q   = "SELECT id, text, kind, importance, archived FROM memories"
        cnd = [] if include_archived else ["archived=0"]
        if kind:
            cnd.append(f"kind='{kind}'")
        if cnd:
            q += " WHERE " + " AND ".join(cnd)
        out = []
        for mid, text, k, imp, arch in self._conn.execute(q):
            if not qt or qt & _tokens(text):
                out.append({
                    "id": mid, "text": text, "kind": k,
                    "importance": round(float(imp or 1.0), 3),
                    "archived": bool(arch),
                })
        return out[:limit]

    # ── Stats ─────────────────────────────────────────────────────────
    def stats(self) -> dict:
        rows = self._conn.execute(
            "SELECT kind, COUNT(*), AVG(importance) FROM memories "
            "WHERE archived=0 GROUP BY kind"
        ).fetchall()
        total = self._conn.execute(
            "SELECT COUNT(*) FROM memories"
        ).fetchone()[0]
        archived = self._conn.execute(
            "SELECT COUNT(*) FROM memories WHERE archived=1"
        ).fetchone()[0]
        return {
            "total":    total,
            "active":   total - archived,
            "archived": archived,
            "by_kind":  {k: {"count":c,"avg_importance":round(a or 0,3)}
                         for k,c,a in rows},
        }
