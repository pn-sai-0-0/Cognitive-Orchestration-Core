"""
CognitiveOC v3 — Knowledge Graph System
=========================================

Stores factual knowledge as (subject, relation, object, confidence, source) triples.
Backed by SQLite (var/knowledge_graph.db) — replaces the baseline JSON file.

Why SQLite over JSON?
  - Fast queries by entity, relation, or object (indexed)
  - Atomic writes (no corrupted JSON on crash)
  - Supports 500K+ triples without memory pressure
  - FTS5 for text search across all triples
  - Scales to 1M+ triples on 1TB SSD without issue

Architecture position:
  Encoder Stack → extraction → KnowledgeGraph.add_triple()
  Engine         → query     → KnowledgeGraph.ranked_query()
  Memory sync    → batch     → sync_memory_to_kg()  (in summarizer.py)

New in v3 vs baseline:
  Baseline: JSON file, 12 patterns, no FTS, no confidence update, no analytics.
  v3: SQLite, 20 patterns, FTS5, confidence Bayesian update, graph analytics,
      entity resolution, clustering, contradiction detection, 24 relation types,
      neural extraction flag (enabled after model trained), provenance tracking.

File: knowledge/graph.py
Used by: engine.py, memory/summarizer.py, retrieval/rag.py, reasoning/reasoner.py
Persists: var/knowledge_graph.db
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from pathlib import Path

try:
    from config import KNOWLEDGE_GRAPH, KG_PATH, KG_BACKUP, BASE_DIR, ensure_dirs
except ImportError:
    KG_PATH    = Path("var/knowledge_graph.db")
    KG_BACKUP  = Path("var/knowledge_graph_backup.json")
    BASE_DIR   = Path(".")
    ensure_dirs = lambda: None
    KNOWLEDGE_GRAPH = dict(
        confidence_threshold=0.50, max_triples=500_000, max_entities=100_000,
        enable_neural_extraction=False, cluster_min_size=3,
        contradiction_resolution="confidence",
        relation_types=[
            "is","has","uses","created","works_at","located_in","belongs_to",
            "contains","born_in","named","means","causes","depends_on","produces",
            "relates_to","contradicts","precedes","follows","similar_to",
            "instance_of","part_of","derived_from","enables","inhibits",
        ],
    )

_WRITE_LK = threading.RLock()


# ═══════════════════════════════════════════════════════════════════
# KnowledgeGraph
# ═══════════════════════════════════════════════════════════════════

class KnowledgeGraph:
    """COC v3 Knowledge Graph — SQLite-backed triple store.

    Core operations:
      add_triple()      — insert or update a (s, r, o) triple
      query()           — filter triples by subject/relation/object
      ranked_query()    — score triples by relevance to a query string
      extract_from_text()— extract + add triples from raw text
      merge()           — merge two entities (entity resolution)
      contradictions()  — find (s,r,o1) and (s,r,o2) conflicts
      cluster()         — group entities by shared relations
      analytics()       — degree, density, top entities
      export_json()     — backup to JSON
      import_json()     — restore from JSON backup
    """

    def __init__(self, db_path: str = None):
        ensure_dirs()
        self._db_path = str(db_path or KG_PATH)
        self._conn    = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conf_threshold = KNOWLEDGE_GRAPH.get("confidence_threshold", 0.50)
        self._max_triples    = KNOWLEDGE_GRAPH.get("max_triples", 500_000)
        self._relation_types = set(KNOWLEDGE_GRAPH.get("relation_types", []))
        self._init_schema()

    # ── Schema ───────────────────────────────────────────────────────
    def _init_schema(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS triples (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                subject     TEXT    NOT NULL,
                relation    TEXT    NOT NULL,
                object      TEXT    NOT NULL,
                confidence  REAL    DEFAULT 1.0,
                source      TEXT    DEFAULT '',
                ts          REAL    DEFAULT (strftime('%s','now')),
                updated_at  REAL    DEFAULT (strftime('%s','now')),
                access_cnt  INTEGER DEFAULT 0,
                verified    INTEGER DEFAULT 0,
                UNIQUE(subject, relation, object)
            );
            CREATE INDEX IF NOT EXISTS idx_kg_subject  ON triples(subject);
            CREATE INDEX IF NOT EXISTS idx_kg_relation ON triples(relation);
            CREATE INDEX IF NOT EXISTS idx_kg_object   ON triples(object);
            CREATE INDEX IF NOT EXISTS idx_kg_conf     ON triples(confidence DESC);

            CREATE VIRTUAL TABLE IF NOT EXISTS triples_fts USING fts5(
                subject, relation, object,
                content=triples, content_rowid=id
            );

            CREATE TABLE IF NOT EXISTS entities (
                name        TEXT PRIMARY KEY,
                kind        TEXT DEFAULT '',
                aliases     TEXT DEFAULT '',
                mention_cnt INTEGER DEFAULT 1,
                first_seen  REAL DEFAULT (strftime('%s','now')),
                last_seen   REAL DEFAULT (strftime('%s','now'))
            );
            CREATE INDEX IF NOT EXISTS idx_ent_kind ON entities(kind);
        """)
        self._conn.commit()
        self._rebuild_fts()

    def _rebuild_fts(self):
        try:
            self._conn.execute(
                "INSERT INTO triples_fts(triples_fts) VALUES('rebuild')"
            )
            self._conn.commit()
        except Exception:
            pass

    def _w(self, sql: str, params: tuple = ()):
        with _WRITE_LK:
            self._conn.execute(sql, params)
            self._conn.commit()

    # ── Core triple operations ────────────────────────────────────────
    def add_triple(self,
                   subject:    str,
                   relation:   str,
                   object_:    str,
                   confidence: float = 1.0,
                   source:     str   = "",
                   verified:   bool  = False) -> int:
        """Insert or update a triple. Uses Bayesian confidence update on conflict.

        Confidence update on existing triple:
          new_conf = old_conf + (1 - old_conf) × new_conf × 0.5
          (Converges toward 1.0 as evidence accumulates.)

        Returns:
            Triple row ID.
        """
        if confidence < self._conf_threshold:
            return -1

        subject  = subject.strip().lower()[:200]
        relation = relation.strip().lower()[:100]
        object_  = object_.strip().lower()[:200]

        if not subject or not relation or not object_:
            return -1

        now = time.time()
        with _WRITE_LK:
            existing = self._conn.execute(
                "SELECT id, confidence FROM triples "
                "WHERE subject=? AND relation=? AND object=?",
                (subject, relation, object_),
            ).fetchone()

            if existing:
                tid     = existing[0]
                old_c   = existing[1]
                new_c   = min(1.0, old_c + (1.0 - old_c) * confidence * 0.5)
                self._conn.execute(
                    "UPDATE triples SET confidence=?, updated_at=?, "
                    "access_cnt=access_cnt+1, source=?, verified=? "
                    "WHERE id=?",
                    (round(new_c, 4), now, source or "", int(verified), tid),
                )
            else:
                cur = self._conn.execute(
                    "INSERT INTO triples"
                    "(subject, relation, object, confidence, source, ts, updated_at, verified)"
                    " VALUES(?,?,?,?,?,?,?,?)",
                    (subject, relation, object_,
                     round(confidence, 4), source or "", now, now, int(verified)),
                )
                tid = cur.lastrowid
                self._conn.execute(
                    "INSERT INTO triples_fts(rowid, subject, relation, object) "
                    "VALUES(?,?,?,?)",
                    (tid, subject, relation, object_),
                )

            # Update entity registry
            for ent in (subject, object_):
                self._conn.execute(
                    "INSERT INTO entities(name, mention_cnt, last_seen) VALUES(?,1,?)"
                    " ON CONFLICT(name) DO UPDATE SET "
                    "mention_cnt=mention_cnt+1, last_seen=excluded.last_seen",
                    (ent, now),
                )
            self._conn.commit()

        self._prune()
        return tid

    def remove_triple(self, subject: str, relation: str, object_: str) -> bool:
        subject  = subject.strip().lower()
        relation = relation.strip().lower()
        object_  = object_.strip().lower()
        with _WRITE_LK:
            cur = self._conn.execute(
                "DELETE FROM triples WHERE subject=? AND relation=? AND object=?",
                (subject, relation, object_),
            )
            self._conn.commit()
        self._rebuild_fts()
        return cur.rowcount > 0

    def _prune(self):
        n = self._conn.execute("SELECT COUNT(*) FROM triples").fetchone()[0]
        if n > self._max_triples:
            with _WRITE_LK:
                self._conn.execute(
                    "DELETE FROM triples WHERE id IN "
                    "(SELECT id FROM triples ORDER BY confidence ASC, ts ASC LIMIT ?)",
                    (n - self._max_triples,),
                )
                self._conn.commit()
            self._rebuild_fts()

    # ── Query ────────────────────────────────────────────────────────
    def query(self,
              entity:   str = None,
              relation: str = None,
              object_:  str = None,
              limit:    int = 30,
              min_conf: float = None) -> list[dict]:
        """Filter triples by subject entity, relation, and/or object.

        Any combination of filters can be combined.
        Results ordered by confidence descending.
        """
        min_conf = min_conf or self._conf_threshold
        q        = "SELECT id, subject, relation, object, confidence, source FROM triples WHERE confidence>=?"
        args: list = [min_conf]

        if entity:
            ent = entity.strip().lower()
            q  += " AND (subject=? OR object=?)"
            args += [ent, ent]
        if relation:
            q  += " AND relation=?"
            args.append(relation.strip().lower())
        if object_:
            q  += " AND object=?"
            args.append(object_.strip().lower())

        q    += " ORDER BY confidence DESC LIMIT ?"
        args.append(limit)
        rows  = self._conn.execute(q, tuple(args)).fetchall()
        return [
            {"id":i, "subject":s, "relation":r, "object":o,
             "confidence":round(float(c),3), "source":src}
            for i,s,r,o,c,src in rows
        ]

    def ranked_query(self,
                     query_text: str,
                     limit:      int = 20,
                     min_conf:   float = None) -> list[dict]:
        """Score triples by text similarity to query_text.

        Scoring:
          - FTS5 match on (subject, relation, object)
          - Token overlap with query terms (weight 0.6)
          - Triple confidence (weight 0.4)

        Returns top-limit results sorted by composite score.
        """
        min_conf = min_conf or self._conf_threshold
        tokens   = set(re.findall(r"[a-z0-9]+", query_text.lower()))
        fts_q    = " OR ".join(tokens) if tokens else ""

        if fts_q:
            try:
                rows = self._conn.execute(
                    "SELECT t.id, t.subject, t.relation, t.object, t.confidence, t.source "
                    "FROM triples t "
                    "JOIN triples_fts f ON t.id=f.rowid "
                    "WHERE triples_fts MATCH ? AND t.confidence>=? LIMIT 500",
                    (fts_q, min_conf),
                ).fetchall()
            except Exception:
                rows = self._conn.execute(
                    "SELECT id, subject, relation, object, confidence, source "
                    "FROM triples WHERE confidence>=? LIMIT 200",
                    (min_conf,),
                ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT id, subject, relation, object, confidence, source "
                "FROM triples WHERE confidence>=? LIMIT 200",
                (min_conf,),
            ).fetchall()

        scored = []
        for tid, s, r, o, c, src in rows:
            triple_tokens = set(re.findall(r"[a-z0-9]+", f"{s} {r} {o}"))
            overlap       = len(tokens & triple_tokens) / max(len(tokens), 1)
            score         = 0.6 * overlap + 0.4 * float(c)
            if score > 0.1:
                scored.append((score, {
                    "id":tid, "subject":s, "relation":r, "object":o,
                    "confidence":round(float(c),3), "source":src,
                    "score":round(score,4),
                }))

        scored.sort(key=lambda x: -x[0])
        # Update access counts
        if scored:
            with _WRITE_LK:
                for _, t in scored[:limit]:
                    self._conn.execute(
                        "UPDATE triples SET access_cnt=access_cnt+1 WHERE id=?",
                        (t["id"],)
                    )
                self._conn.commit()
        return [t for _, t in scored[:limit]]

    def fts_search(self, query: str, limit: int = 20) -> list[dict]:
        """Full-text search across all triple fields."""
        tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
        if not tokens:
            return []
        fts_q  = " OR ".join(tokens)
        try:
            rows = self._conn.execute(
                "SELECT t.id, t.subject, t.relation, t.object, t.confidence "
                "FROM triples t JOIN triples_fts f ON t.id=f.rowid "
                "WHERE triples_fts MATCH ? ORDER BY t.confidence DESC LIMIT ?",
                (fts_q, limit),
            ).fetchall()
        except Exception:
            return []
        return [{"id":i,"subject":s,"relation":r,"object":o,"confidence":round(float(c),3)}
                for i,s,r,o,c in rows]

    # ── Extraction ───────────────────────────────────────────────────
    def extract_from_text(self,
                          text:       str,
                          source:     str = "",
                          confidence: float = 0.70) -> list[dict]:
        """Extract triples from text and add them to the graph.

        Returns list of added triple dicts.
        """
        from memory.summarizer import extract_triples
        raw     = extract_triples(text)
        added   = []
        for subj, rel, obj in raw:
            if rel not in self._relation_types:
                continue   # only store known relation types
            tid = self.add_triple(subj, rel, obj,
                                  confidence=confidence, source=source)
            if tid > 0:
                added.append({"subject":subj, "relation":rel,
                               "object":obj, "confidence":confidence,
                               "id":tid})
        return added

    def extract_and_link(self,
                         text:         str,
                         source:       str = "",
                         memory_id:    int = None) -> list[dict]:
        """Extract triples and optionally link to a memory entry.

        High-level convenience method used by engine.py after ingestion.
        """
        added = self.extract_from_text(text, source=source)
        # Future: memory ↔ KG linkage table (Phase 6)
        return added

    # ── Entity operations ────────────────────────────────────────────
    def entities(self, kind: str = None, limit: int = 100) -> list[dict]:
        """List entities from the entity registry."""
        q    = "SELECT name, kind, mention_cnt FROM entities"
        args = []
        if kind:
            q    += " WHERE kind=?"
            args.append(kind)
        q    += " ORDER BY mention_cnt DESC LIMIT ?"
        args.append(limit)
        return [{"name":n,"kind":k,"mentions":c}
                for n,k,c in self._conn.execute(q, args)]

    def set_entity_kind(self, name: str, kind: str):
        """Manually set entity type (person, org, concept, place, etc.)"""
        with _WRITE_LK:
            self._conn.execute(
                "UPDATE entities SET kind=? WHERE name=?",
                (kind.lower(), name.strip().lower()),
            )
            self._conn.commit()

    def merge_entities(self, canonical: str, alias: str) -> int:
        """Entity resolution: replace all occurrences of alias with canonical.

        Returns number of triples updated.
        """
        c  = canonical.strip().lower()
        a  = alias.strip().lower()
        with _WRITE_LK:
            self._conn.execute(
                "UPDATE triples SET subject=? WHERE subject=?", (c, a)
            )
            self._conn.execute(
                "UPDATE triples SET object=? WHERE object=?", (c, a)
            )
            # Update aliases in entity registry
            row = self._conn.execute(
                "SELECT aliases FROM entities WHERE name=?", (c,)
            ).fetchone()
            existing = (row[0] or "") if row else ""
            aliases  = set(existing.split(",")) | {a}
            aliases.discard("")
            self._conn.execute(
                "INSERT INTO entities(name, aliases) VALUES(?,?)"
                " ON CONFLICT(name) DO UPDATE SET aliases=excluded.aliases",
                (c, ",".join(aliases)),
            )
            self._conn.execute("DELETE FROM entities WHERE name=?", (a,))
            n = self._conn.execute(
                "SELECT changes()"
            ).fetchone()[0]
            self._conn.commit()
        self._rebuild_fts()
        return n

    # ── Graph Analytics ──────────────────────────────────────────────
    def contradictions(self) -> list[dict]:
        """Find contradicting triples: same (subject, relation) with different objects.

        Returns list of conflict groups.
        """
        rows = self._conn.execute(
            "SELECT subject, relation, GROUP_CONCAT(object,';'), "
            "GROUP_CONCAT(confidence,';') "
            "FROM triples "
            "GROUP BY subject, relation "
            "HAVING COUNT(*)>1 AND COUNT(DISTINCT object)>1"
        ).fetchall()
        conflicts = []
        for subj, rel, objs, confs in rows:
            obj_list  = objs.split(";")
            conf_list = [float(c) for c in confs.split(";")]
            conflicts.append({
                "subject":  subj,
                "relation": rel,
                "objects":  obj_list,
                "confs":    conf_list,
                "resolution": self._resolve_contradiction(rel, obj_list, conf_list),
            })
        return conflicts

    def _resolve_contradiction(self,
                                relation:  str,
                                objects:   list[str],
                                confs:     list[float]) -> str:
        """Pick the preferred object in a contradiction."""
        strategy = KNOWLEDGE_GRAPH.get("contradiction_resolution", "confidence")
        if strategy == "confidence":
            best = objects[confs.index(max(confs))]
        elif strategy == "recency":
            # Would need timestamp — fall back to confidence
            best = objects[confs.index(max(confs))]
        else:
            best = objects[0]
        return best

    def cluster(self, min_size: int = None) -> list[dict]:
        """Cluster entities by shared relations.

        Two entities are in the same cluster if they share at least one
        (relation, object) pair with the same object.

        Returns list of cluster dicts: {entities, shared_relations, size}.
        """
        min_size = min_size or KNOWLEDGE_GRAPH.get("cluster_min_size", 3)
        # Build entity → (relation+object) sets
        rows = self._conn.execute(
            "SELECT subject, relation, object FROM triples WHERE confidence>=?",
            (self._conf_threshold,)
        ).fetchall()
        from collections import defaultdict
        ent_rels: dict[str, set] = defaultdict(set)
        for s, r, o in rows:
            ent_rels[s].add((r, o))
            ent_rels[o].add((r, s))

        entities = list(ent_rels.keys())
        clusters: list[set] = []
        assigned: set       = set()
        for e in entities:
            if e in assigned:
                continue
            cluster = {e}
            for other in entities:
                if other == e or other in assigned:
                    continue
                shared = ent_rels[e] & ent_rels[other]
                if len(shared) >= 1:
                    cluster.add(other)
            if len(cluster) >= min_size:
                clusters.append(cluster)
                assigned.update(cluster)

        return [
            {
                "entities":        sorted(c),
                "size":            len(c),
                "shared_relations": list(
                    set(r for e in c for r, _ in ent_rels[e])
                )[:10],
            }
            for c in clusters
        ]

    def analytics(self) -> dict:
        """Compute graph-level analytics.

        Returns:
            {
                total_triples, total_entities, avg_confidence,
                top_subjects, top_objects, top_relations,
                density, verified_count
            }
        """
        n_triples  = self._conn.execute("SELECT COUNT(*) FROM triples").fetchone()[0]
        n_entities = self._conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        avg_conf   = self._conn.execute(
            "SELECT AVG(confidence) FROM triples"
        ).fetchone()[0] or 0.0
        verified   = self._conn.execute(
            "SELECT COUNT(*) FROM triples WHERE verified=1"
        ).fetchone()[0]

        top_subj   = [
            {"entity":s, "count":c}
            for s,c in self._conn.execute(
                "SELECT subject, COUNT(*) FROM triples "
                "GROUP BY subject ORDER BY COUNT(*) DESC LIMIT 10"
            )
        ]
        top_rel    = [
            {"relation":r, "count":c}
            for r,c in self._conn.execute(
                "SELECT relation, COUNT(*) FROM triples "
                "GROUP BY relation ORDER BY COUNT(*) DESC LIMIT 10"
            )
        ]
        density    = (n_triples / max(n_entities ** 2, 1)) if n_entities else 0

        return {
            "total_triples":   n_triples,
            "total_entities":  n_entities,
            "avg_confidence":  round(float(avg_conf), 3),
            "verified_count":  verified,
            "top_subjects":    top_subj,
            "top_relations":   top_rel,
            "density":         round(density, 6),
        }

    def neighbourhood(self, entity: str, hops: int = 2,
                      limit: int = 50) -> list[dict]:
        """Return N-hop neighbourhood of an entity (BFS over triples).

        Useful for context assembly and reasoning traces.
        """
        entity    = entity.strip().lower()
        visited   = {entity}
        frontier  = [entity]
        results   = []

        for _ in range(hops):
            next_frontier = []
            for ent in frontier:
                rows = self._conn.execute(
                    "SELECT subject, relation, object, confidence "
                    "FROM triples WHERE (subject=? OR object=?) "
                    "AND confidence>=? LIMIT 20",
                    (ent, ent, self._conf_threshold),
                ).fetchall()
                for s, r, o, c in rows:
                    results.append({"subject":s,"relation":r,"object":o,
                                    "confidence":round(float(c),3)})
                    for neighbour in (s, o):
                        if neighbour not in visited:
                            visited.add(neighbour)
                            next_frontier.append(neighbour)
                if len(results) >= limit:
                    return results[:limit]
            frontier = next_frontier

        return results[:limit]

    def confidence_update(self, triple_id: int, delta: float):
        """Manually adjust confidence of a specific triple."""
        with _WRITE_LK:
            self._conn.execute(
                "UPDATE triples SET "
                "confidence=MIN(1.0,MAX(0.0,confidence+?)), updated_at=? "
                "WHERE id=?",
                (delta, time.time(), triple_id),
            )
            self._conn.commit()

    def verify_triple(self, triple_id: int):
        """Mark a triple as human-verified."""
        with _WRITE_LK:
            self._conn.execute(
                "UPDATE triples SET verified=1, confidence=MIN(1.0, confidence+0.15) "
                "WHERE id=?", (triple_id,)
            )
            self._conn.commit()

    # ── Import / Export ──────────────────────────────────────────────
    def export_json(self, path: str = None) -> str:
        """Export all triples to JSON backup file."""
        path  = path or str(KG_BACKUP)
        rows  = self._conn.execute(
            "SELECT subject, relation, object, confidence, source, verified "
            "FROM triples"
        ).fetchall()
        data  = [
            {"subject":s,"relation":r,"object":o,
             "confidence":round(float(c),4),"source":src,"verified":bool(v)}
            for s,r,o,c,src,v in rows
        ]
        with open(path, "w") as f:
            json.dump({"triples": data, "exported_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                       "count": len(data)}, f, indent=2)
        return path

    def import_json(self, path: str = None, clear_first: bool = False) -> int:
        """Import triples from JSON backup file."""
        path = path or str(KG_BACKUP)
        if not Path(path).exists():
            raise FileNotFoundError(path)
        with open(path) as f:
            data = json.load(f)
        triples = data.get("triples", data) if isinstance(data, dict) else data
        if clear_first:
            with _WRITE_LK:
                self._conn.execute("DELETE FROM triples")
                self._conn.commit()
        added = 0
        for t in triples:
            tid = self.add_triple(
                t["subject"], t["relation"], t["object"],
                confidence=t.get("confidence", 0.7),
                source=t.get("source", "import"),
                verified=bool(t.get("verified", False)),
            )
            if tid > 0:
                added += 1
        self._rebuild_fts()
        return added

    # ── Stats / info ─────────────────────────────────────────────────
    def __len__(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM triples").fetchone()[0]

    def __repr__(self) -> str:
        n = len(self)
        e = self._conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        return f"KnowledgeGraph(triples={n:,}, entities={e:,}, db={self._db_path})"
