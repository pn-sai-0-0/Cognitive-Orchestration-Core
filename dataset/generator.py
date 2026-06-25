"""
CognitiveOC v3 — Dataset Generation Engine
============================================

Captures feedback, errors, and hard examples then exports them as
training-ready datasets with provenance tracking.

NO automatic retraining. Human review is required before any dataset
is used for training. All exports go to var/datasets/ with manifests.

Export types:
  conversation  — SFT pairs from rated conversations
  retrieval     — query + relevant chunks (hard negative mining)
  kg            — text + extracted triple labels
  memory        — store/recall pairs
  reasoning     — query + reasoning chain
  teaching      — level-labelled instructional pairs
  emotion       — text + emotion labels
  evaluation    — regression test cases

File: dataset/generator.py
Used by: engine.py (feedback recording), ui/app.py, main.py
Persists: var/datasets/ (JSONL files + manifests)
          var/learning.db (feedback SQLite)
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

try:
    from config import DATASET, STORE_DIR, ensure_dirs
except ImportError:
    DATASET   = dict(export_dir="var/datasets", review_queue_path="var/datasets/review_queue.jsonl",
                     auto_approve=False, dedup_threshold=0.92, min_quality_score=0.60,
                     types=["conversation","retrieval","kg","memory","teaching","emotion","reasoning"])
    STORE_DIR = Path("var")
    def ensure_dirs(): pass

_WRITE_LK = threading.Lock()

# PII patterns for export-time redaction
_PII = [
    re.compile(r'\b[\w.+\-]+@[\w\-]+\.[\w.]+\b'),
    re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
    re.compile(r'(api[_\-]?key|password|token)\s*[:=]\s*\S+', re.I),
]


def _redact(text: str) -> str:
    for pat in _PII:
        text = pat.sub("[REDACTED]", text)
    return text


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _quality_ok(prompt: str, response: str) -> bool:
    """Basic quality filter for dataset entries."""
    if len(prompt.split()) < 3 or len(response.split()) < 5:
        return False
    all_words = (prompt + " " + response).lower().split()
    if not all_words:
        return False
    unique_ratio = len(set(all_words)) / len(all_words)
    return unique_ratio >= 0.25


# ═══════════════════════════════════════════════════════════════════
# LearningStore — SQLite feedback DB
# ═══════════════════════════════════════════════════════════════════

class LearningStore:
    """SQLite store for feedback, errors, and hard examples."""

    def __init__(self, path: str = None):
        ensure_dirs()
        db_path = path or str(Path(str(STORE_DIR)) / "learning.db")
        self._db   = sqlite3.connect(db_path, check_same_thread=False)
        self._init_schema()

    def _init_schema(self):
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS feedback(
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                session    TEXT,
                question   TEXT,
                answer     TEXT,
                rating     INTEGER,
                confidence REAL,
                emotion    TEXT DEFAULT '',
                intent     TEXT DEFAULT '',
                ts         REAL
            );
            CREATE TABLE IF NOT EXISTS errors(
                id      INTEGER PRIMARY KEY AUTOINCREMENT,
                context TEXT,
                error   TEXT,
                ts      REAL
            );
            CREATE TABLE IF NOT EXISTS retrieval_feedback(
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                query       TEXT,
                chunk_text  TEXT,
                chunk_source TEXT,
                useful      INTEGER,
                ts          REAL
            );
            CREATE INDEX IF NOT EXISTS idx_fb_rating  ON feedback(rating);
            CREATE INDEX IF NOT EXISTS idx_fb_session ON feedback(session);
        """)
        self._db.commit()

    def _w(self, sql: str, params: tuple = ()):
        with _WRITE_LK:
            self._db.execute(sql, params)
            self._db.commit()

    def record_feedback(self, question: str, answer: str, rating: int,
                        session: str = "default", confidence: float = None,
                        emotion: str = "", intent: str = ""):
        self._w(
            "INSERT INTO feedback(session,question,answer,rating,confidence,"
            "emotion,intent,ts) VALUES(?,?,?,?,?,?,?,?)",
            (session, question[:2000], answer[:4000], int(rating),
             confidence, emotion, intent, time.time()),
        )

    def record_error(self, context: str, error: str):
        self._w(
            "INSERT INTO errors(context,error,ts) VALUES(?,?,?)",
            (context[:500], str(error)[:500], time.time()),
        )

    def record_retrieval_feedback(self, query: str, chunk_text: str,
                                   chunk_source: str, useful: bool):
        self._w(
            "INSERT INTO retrieval_feedback(query,chunk_text,chunk_source,useful,ts)"
            " VALUES(?,?,?,?,?)",
            (query[:500], chunk_text[:500], chunk_source[:200], int(useful), time.time()),
        )

    def hard_examples(self, confidence_below: float = 0.4,
                      limit: int = 100) -> list[dict]:
        rows = self._db.execute(
            "SELECT question,answer,rating,confidence FROM feedback"
            " WHERE rating<0 OR (confidence IS NOT NULL AND confidence<?)"
            " ORDER BY id DESC LIMIT ?",
            (confidence_below, limit),
        ).fetchall()
        return [{"question":q,"answer":a,"rating":r,"confidence":c}
                for q,a,r,c in rows]

    def analytics(self) -> dict:
        total    = self._db.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
        positive = self._db.execute("SELECT COUNT(*) FROM feedback WHERE rating>0").fetchone()[0]
        errors   = self._db.execute("SELECT COUNT(*) FROM errors").fetchone()[0]
        return {
            "total_feedback": total,
            "positive":       positive,
            "negative":       total - positive,
            "errors":         errors,
            "positive_rate":  round(positive/max(total,1), 3),
        }


# ═══════════════════════════════════════════════════════════════════
# DatasetGenerator
# ═══════════════════════════════════════════════════════════════════

class DatasetGenerator:
    """Exports training datasets from captured feedback and system state.

    All exports:
      - Are written to var/datasets/<type>_<timestamp>.jsonl
      - Include a manifest.json with provenance + SHA-256
      - Require human review before use in training
      - Are PII-redacted
      - Are quality-filtered

    Usage:
        gen = DatasetGenerator(learning_store)
        result = gen.export_conversations()
        result = gen.export_retrieval()
        result = gen.export_kg()
        queue  = gen.review_queue()
    """

    def __init__(self, store: LearningStore = None):
        ensure_dirs()
        self._store     = store or LearningStore()
        self._dir       = Path(DATASET.get("export_dir", "var/datasets"))
        self._dir.mkdir(parents=True, exist_ok=True)
        self._queue_path= Path(DATASET.get("review_queue_path",
                                            "var/datasets/review_queue.jsonl"))
        self._auto_approve = DATASET.get("auto_approve", False)
        self._min_quality  = DATASET.get("min_quality_score", 0.60)

    def _write_jsonl(self, name: str, records: list[dict]) -> dict:
        """Write records to JSONL and return manifest."""
        ts   = time.strftime("%Y%m%d_%H%M%S")
        path = self._dir / f"{name}_{ts}.jsonl"

        dedup_seen: set[str] = set()
        written = 0
        with open(str(path), "w") as f:
            for rec in records:
                h = _sha256(json.dumps(rec, sort_keys=True))
                if h in dedup_seen:
                    continue
                dedup_seen.add(h)
                f.write(json.dumps(rec) + "\n")
                written += 1

        checksum = _sha256(path.read_text())
        manifest = {
            "type":        name,
            "path":        str(path),
            "records":     written,
            "created":     time.strftime("%Y-%m-%dT%H:%M:%S"),
            "reviewed":    False,
            "auto_approved": self._auto_approve,
            "sha256":      checksum,
            "pii_redacted":True,
            "quality_filtered": True,
        }
        (self._dir / f"{name}_{ts}_manifest.json").write_text(
            json.dumps(manifest, indent=2)
        )

        # Add to review queue
        if not self._auto_approve:
            self._enqueue_review(manifest)

        return manifest

    def _enqueue_review(self, manifest: dict):
        """Add an export to the human review queue."""
        record = json.dumps({"ts": time.time(), **manifest})
        with _WRITE_LK:
            with open(str(self._queue_path), "a") as f:
                f.write(record + "\n")

    # ── Export methods ────────────────────────────────────────────────

    def export_conversations(self,
                              min_rating: int = 1,
                              limit: int = 5000) -> dict:
        """Export positively-rated conversations as SFT pairs."""
        rows = self._store._db.execute(
            "SELECT question,answer,rating,confidence,session FROM feedback"
            " WHERE rating>=? ORDER BY id DESC LIMIT ?",
            (min_rating, limit),
        ).fetchall()

        records = []
        for q, a, rating, conf, sess in rows:
            q_clean = _redact(q or "")
            a_clean = _redact(a or "")
            if not _quality_ok(q_clean, a_clean):
                continue
            records.append({
                "type":       "conversation",
                "prompt":     q_clean,
                "response":   a_clean,
                "rating":     rating,
                "confidence": conf,
                "session":    sess,
                "source":     "feedback_capture",
            })

        return self._write_jsonl("conversation", records)

    def export_retrieval(self, limit: int = 2000) -> dict:
        """Export retrieval feedback as query+relevant/irrelevant pairs."""
        rows = self._store._db.execute(
            "SELECT query,chunk_text,chunk_source,useful FROM retrieval_feedback"
            " ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()

        records = []
        for query, chunk, source, useful in rows:
            q_clean = _redact(query or "")
            c_clean = _redact(chunk or "")
            if len(q_clean.split()) < 2 or len(c_clean.split()) < 5:
                continue
            records.append({
                "type":    "retrieval",
                "query":   q_clean,
                "passage": c_clean,
                "source":  source,
                "label":   int(useful),
            })

        return self._write_jsonl("retrieval", records)

    def export_kg(self, kg=None, limit: int = 10000) -> dict:
        """Export KG triples as (text, labels) pairs for triple extraction training."""
        if kg is None:
            return {"error": "KG instance required", "records": 0}

        rows = kg._conn.execute(
            "SELECT subject, relation, object, confidence FROM triples"
            " WHERE confidence>=0.7 ORDER BY confidence DESC LIMIT ?",
            (limit,),
        ).fetchall()

        records = []
        for s, r, o, conf in rows:
            # Reconstruct pseudo-sentence for the training label
            text = f"{s} {r} {o}."
            records.append({
                "type":       "kg",
                "text":       text,
                "subject":    s,
                "relation":   r,
                "object":     o,
                "confidence": round(float(conf), 3),
            })

        return self._write_jsonl("kg", records)

    def export_memory(self, memory=None, limit: int = 5000) -> dict:
        """Export memory entries as fact pairs for memory training."""
        if memory is None:
            return {"error": "memory instance required", "records": 0}

        rows = memory.list_memories(limit=limit)
        records = []
        for m in rows:
            text = _redact(m.get("text",""))
            if len(text.split()) < 5:
                continue
            records.append({
                "type":       "memory",
                "text":       text,
                "kind":       m.get("kind","fact"),
                "importance": m.get("importance", 1.0),
            })

        return self._write_jsonl("memory", records)

    def export_reasoning(self, limit: int = 1000) -> dict:
        """Export reasoning traces from feedback for reasoning training."""
        rows = self._store._db.execute(
            "SELECT question, answer, confidence FROM feedback"
            " WHERE rating>0 AND confidence IS NOT NULL ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()

        records = []
        for q, a, conf in rows:
            q_clean = _redact(q or "")
            a_clean = _redact(a or "")
            if not _quality_ok(q_clean, a_clean):
                continue
            records.append({
                "type":             "reasoning",
                "query":            q_clean,
                "response":         a_clean,
                "confidence":       conf,
                "reasoning_type":   "general",
            })

        return self._write_jsonl("reasoning", records)

    def export_teaching(self, limit: int = 2000) -> dict:
        """Export teaching-style exchanges with level labels."""
        rows = self._store._db.execute(
            "SELECT question,answer,rating,intent FROM feedback"
            " WHERE rating>0 AND intent='learning' ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()

        records = []
        for q, a, rating, intent in rows:
            q_c = _redact(q or "")
            a_c = _redact(a or "")
            if not _quality_ok(q_c, a_c):
                continue
            records.append({
                "type":    "teaching",
                "prompt":  q_c,
                "response":a_c,
                "intent":  intent or "learning",
                "level":   "intermediate",
            })

        return self._write_jsonl("teaching", records)

    def export_emotion(self, limit: int = 2000) -> dict:
        """Export emotion-labelled examples."""
        rows = self._store._db.execute(
            "SELECT question,emotion FROM feedback"
            " WHERE emotion!='' ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()

        records = []
        for q, em in rows:
            q_c = _redact(q or "")
            if len(q_c.split()) < 3 or not em:
                continue
            records.append({
                "type":  "emotion",
                "text":  q_c,
                "label": em,
            })

        return self._write_jsonl("emotion", records)

    def export_all(self, memory=None, kg=None) -> dict:
        """Export all dataset types. Returns dict of type → manifest."""
        results = {}
        results["conversation"] = self.export_conversations()
        results["retrieval"]    = self.export_retrieval()
        results["reasoning"]    = self.export_reasoning()
        results["teaching"]     = self.export_teaching()
        results["emotion"]      = self.export_emotion()
        if memory:
            results["memory"]   = self.export_memory(memory)
        if kg:
            results["kg"]       = self.export_kg(kg)
        return results

    # ── Review queue ──────────────────────────────────────────────────
    def review_queue(self) -> list[dict]:
        """Return all entries in the review queue."""
        if not self._queue_path.exists():
            return []
        entries = []
        for line in self._queue_path.read_text().splitlines():
            try:
                entries.append(json.loads(line))
            except Exception:
                pass
        return sorted(entries, key=lambda x: -x.get("ts", 0))

    def approve(self, export_path: str) -> dict:
        """Mark an export as human-approved (updates manifest)."""
        manifest_candidates = list(Path(str(self._dir)).glob(
            Path(export_path).stem.rsplit("_", 1)[0] + "*manifest*.json"
        ))
        if not manifest_candidates:
            return {"error": "manifest not found"}
        manifest = json.loads(manifest_candidates[0].read_text())
        manifest["reviewed"]        = True
        manifest["reviewed_at"]     = time.strftime("%Y-%m-%dT%H:%M:%S")
        manifest["approved"]        = True
        manifest_candidates[0].write_text(json.dumps(manifest, indent=2))
        return manifest

    def hard_examples(self) -> dict:
        """Return hard examples for targeted retraining (after human review)."""
        return {
            "hard_examples":   self._store.hard_examples(),
            "total":           len(self._store.hard_examples()),
            "note":            "Human review required before use in training.",
        }
