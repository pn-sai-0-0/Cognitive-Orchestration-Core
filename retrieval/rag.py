"""
CognitiveOC v3 — Retrieval System
===================================

Implements the full v3 retrieval stack:

  RAGPipeline     — persistent document index (semantic + BM25 hybrid)
  CAGManager      — in-memory active-document cache (Cache-Augmented Generation)
  HybridRetriever — orchestrates RAG + CAG + memory + KG with query rewrite,
                    expansion, multi-hop, cross-encoder reranking, citation tracking
  WorkspaceManager— multi-document workspace sessions
  CitationEngine  — tracks chunk provenance for grounded citations

Architecture position (top-down):
  Engine.build_context()
    → HybridRetriever.retrieve()
      → query rewrite (history-aware)
      → query expansion (synonyms + encoder)
      → RAGPipeline.retrieve()   (semantic cosine + BM25 fusion)
      → CAGManager.retrieve()    (active document cache)
      → CrossEncoder.rerank()    (precision reranking)
      → CitationEngine.track()   (provenance)
      → multi-hop loop (if low confidence)

New in v3 vs baseline:
  Baseline: one BM25+cosine retriever, LRU cache, query rewrite, synonym expand.
  v3: 768-dim BGE encoder (vs 384-dim MiniLM), cross-encoder reranking,
      multi-hop up to 3 hops (configurable), citation engine with chunk IDs,
      retrieval analytics with self-improvement hooks, workspace manager,
      CAG with stale-file detection, retrieval confidence scoring.

File: retrieval/rag.py
Used by: engine.py, workflow/workflow.py, research/engine.py
Persists: var/index/ (embedding vectors + metadata)
          var/cache/ (retrieval cache + analytics)
          var/workspaces/ (workspace sessions)
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

try:
    import numpy as np
    _NP = True
except ImportError:
    _NP = False

try:
    from config import (RETRIEVAL, EMBEDDING, INDEX_DIR, CACHE_DIR,
                        WORKSPACE_DIR, SYNONYMS_PATH, ensure_dirs)
except ImportError:
    INDEX_DIR     = Path("var/index")
    CACHE_DIR     = Path("var/cache")
    WORKSPACE_DIR = Path("var/workspaces")
    SYNONYMS_PATH = Path("var/synonyms.json")
    ensure_dirs   = lambda: None
    RETRIEVAL     = dict(
        chunk_size=512, chunk_overlap=64, top_k=8, rerank_k=24, final_k=5,
        cache_ttl_s=3600, cache_max_entries=512, multi_hop_max=3,
        min_score_threshold=0.25, bm25_weight=0.30, semantic_weight=0.70,
        query_expansion=True, query_rewrite=True, self_improve=True,
    )
    EMBEDDING = dict(model="sentence-transformers/all-MiniLM-L6-v2",
                     dim=384, device="auto", batch=32)


# ═══════════════════════════════════════════════════════════════════
# Embedding helper — uses EncoderHub semantic encoder
# ═══════════════════════════════════════════════════════════════════

def _embed(texts: list[str]) -> "np.ndarray":
    """Embed texts using the semantic encoder from EncoderHub.

    Falls back to hash-TF if encoder unavailable.
    """
    try:
        from encoder.hub import get_hub
        return get_hub().encode("semantic", texts)
    except Exception:
        from encoder.hub import _hash_tf_encode
        dim = EMBEDDING.get("dim", 384)
        return _hash_tf_encode(texts, dim)


def _embed_single(text: str) -> "np.ndarray":
    return _embed([text])[0]


def _cosine_matrix(qv: "np.ndarray", mv: "np.ndarray") -> "np.ndarray":
    """Cosine similarity: qv (dim,) vs mv (N, dim) → (N,)."""
    qn = qv / (np.linalg.norm(qv) + 1e-9)
    mn = mv / (np.linalg.norm(mv, axis=1, keepdims=True) + 1e-9)
    return mn @ qn


# ═══════════════════════════════════════════════════════════════════
# Chunker
# ═══════════════════════════════════════════════════════════════════

def chunk_text(text: str,
               chunk_size: int = None,
               overlap:    int = None) -> list[str]:
    """Split text into overlapping chunks at sentence boundaries.

    Splits at sentence boundaries (. ! ?) first; falls back to word boundaries.
    Never splits mid-word.
    """
    chunk_size = chunk_size or RETRIEVAL.get("chunk_size", 512)
    overlap    = overlap    or RETRIEVAL.get("chunk_overlap", 64)

    sentences  = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    cur  = ""

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        if len(cur) + len(sent) + 1 <= chunk_size:
            cur = (cur + " " + sent).strip()
        else:
            if cur:
                chunks.append(cur)
            # Overlap: carry last `overlap` chars into next chunk
            if overlap and cur:
                tail = cur[-overlap:].rsplit(" ", 1)[-1]
                cur  = (tail + " " + sent).strip()
            else:
                cur  = sent

    if cur:
        chunks.append(cur)

    return [c for c in chunks if len(c.split()) >= 4]


# ═══════════════════════════════════════════════════════════════════
# BM25 (Okapi BM25, pure Python, zero deps)
# ═══════════════════════════════════════════════════════════════════

class BM25:
    """Okapi BM25 scorer over a list of tokenised documents."""

    def __init__(self, corpus: list[list[str]],
                 k1: float = 1.5, b: float = 0.75):
        self.k1     = k1
        self.b      = b
        self.N      = len(corpus)
        self.df: dict[str, int] = {}
        self.tf: list[dict[str,int]] = []
        self.dl: list[int] = []

        for doc in corpus:
            freq: dict[str,int] = {}
            for w in doc:
                freq[w] = freq.get(w, 0) + 1
            self.tf.append(freq)
            self.dl.append(len(doc))
            for w in set(doc):
                self.df[w] = self.df.get(w, 0) + 1

        self.avg_dl = sum(self.dl) / max(self.N, 1)

    def score(self, query_tokens: list[str]) -> list[float]:
        scores = [0.0] * self.N
        for q in query_tokens:
            idf   = math.log((self.N - self.df.get(q, 0) + 0.5)
                              / (self.df.get(q, 0) + 0.5) + 1)
            for i, freq in enumerate(self.tf):
                f  = freq.get(q, 0)
                tf = f * (self.k1 + 1) / (
                    f + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avg_dl)
                )
                scores[i] += idf * tf
        return scores


def _tokenise(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


# ═══════════════════════════════════════════════════════════════════
# LRU Cache
# ═══════════════════════════════════════════════════════════════════

class RetrievalCache:
    """Thread-safe LRU cache for retrieval results."""

    def __init__(self, max_entries: int = None, ttl_s: int = None):
        self._max   = max_entries or RETRIEVAL.get("cache_max_entries", 512)
        self._ttl   = ttl_s       or RETRIEVAL.get("cache_ttl_s", 3600)
        self._cache: OrderedDict[str, tuple[float, list]] = OrderedDict()
        self._lock  = threading.Lock()

    def _key(self, query: str, k: int) -> str:
        return hashlib.md5(f"{query}||{k}".encode()).hexdigest()

    def get(self, query: str, k: int) -> list | None:
        key = self._key(query, k)
        with self._lock:
            if key in self._cache:
                ts, val = self._cache[key]
                if time.time() - ts < self._ttl:
                    self._cache.move_to_end(key)
                    return val
                del self._cache[key]
        return None

    def set(self, query: str, k: int, results: list):
        key = self._key(query, k)
        with self._lock:
            self._cache[key] = (time.time(), results)
            self._cache.move_to_end(key)
            while len(self._cache) > self._max:
                self._cache.popitem(last=False)

    def invalidate(self):
        with self._lock:
            self._cache.clear()


# ═══════════════════════════════════════════════════════════════════
# Citation Engine
# ═══════════════════════════════════════════════════════════════════

class CitationEngine:
    """Track chunk provenance for grounded response citations.

    Maps chunk_id → (source_file, page/section, text_snippet).
    Serialised to var/cache/citations.json.
    """

    def __init__(self):
        ensure_dirs()
        self._path  = Path(str(CACHE_DIR)) / "citations.json"
        self._data: dict[str, dict] = {}
        self._lock  = threading.Lock()
        self._load()

    def _load(self):
        if self._path.exists():
            try:
                self._data = json.loads(self._path.read_text())
            except Exception:
                self._data = {}

    def _save(self):
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(self._data))
        except Exception:
            pass

    def register(self, chunk_id: str, source: str,
                 location: str = "", snippet: str = "") -> str:
        """Register a chunk and its provenance."""
        with self._lock:
            self._data[chunk_id] = {
                "source":   source,
                "location": location,
                "snippet":  snippet[:120],
                "ts":       time.time(),
            }
        return chunk_id

    def resolve(self, chunk_id: str) -> dict:
        """Return provenance for a chunk ID."""
        return self._data.get(chunk_id, {})

    def format_citations(self, chunks: list[dict]) -> str:
        """Format a human-readable citation string from retrieved chunks."""
        seen, parts = set(), []
        for c in chunks:
            src = c.get("source", "unknown")
            if src not in seen:
                seen.add(src)
                loc = c.get("location", "")
                parts.append(f"[{src}{': ' + loc if loc else ''}]")
        return " ".join(parts)

    def flush(self):
        with self._lock:
            self._save()


# ═══════════════════════════════════════════════════════════════════
# RAGPipeline — Persistent Document Index
# ═══════════════════════════════════════════════════════════════════

class RAGPipeline:
    """Persistent semantic + BM25 hybrid retrieval index.

    Index files:
      var/index/index_meta.json — chunk text, source, location metadata
      var/index/index_vecs.bin  — float32 embedding matrix (N × dim)
      var/index/embed_dim.txt   — embedding dimension (for mismatch detection)
    """

    def __init__(self):
        ensure_dirs()
        self._meta_path = Path(str(INDEX_DIR)) / "index_meta.json"
        self._vecs_path = Path(str(INDEX_DIR)) / "index_vecs.bin"
        self._dim_path  = Path(str(INDEX_DIR)) / "embed_dim.txt"
        self._meta: list[dict]    = []
        self._vecs: "np.ndarray | None" = None
        self._bm25: BM25 | None   = None
        self._cache = RetrievalCache()
        self._citations = CitationEngine()
        self._lock  = threading.RLock()
        self._load_index()

    def _load_index(self):
        """Load existing index from disk."""
        if not _NP:
            return
        if self._meta_path.exists():
            try:
                self._meta = json.loads(self._meta_path.read_text())
            except Exception:
                self._meta = []
        if self._vecs_path.exists() and self._meta:
            try:
                import numpy as np
                raw   = np.frombuffer(
                    self._vecs_path.read_bytes(), dtype=np.float32
                )
                dim   = int(self._dim_path.read_text().strip()) \
                        if self._dim_path.exists() else EMBEDDING.get("dim", 384)
                n     = len(self._meta)
                if raw.size == n * dim:
                    self._vecs = raw.reshape(n, dim)
                    self._rebuild_bm25()
                else:
                    self._meta = []
                    self._vecs = None
            except Exception:
                self._meta = []
                self._vecs = None

    def _save_index(self):
        """Persist index to disk."""
        if not _NP or self._vecs is None:
            return
        try:
            INDEX_DIR.mkdir(parents=True, exist_ok=True)
            self._meta_path.write_text(json.dumps(self._meta))
            self._vecs_path.write_bytes(self._vecs.tobytes())
            dim = self._vecs.shape[1] if self._vecs.ndim == 2 else EMBEDDING.get("dim", 384)
            self._dim_path.write_text(str(dim))
            self._citations.flush()
        except Exception:
            pass

    def _rebuild_bm25(self):
        corpus = [_tokenise(m.get("text", "")) for m in self._meta]
        if corpus:
            self._bm25 = BM25(corpus)

    # ── Ingestion ─────────────────────────────────────────────────────
    def ingest(self, path) -> dict:
        """Parse, chunk, embed, and index a document file.

        Supported formats: .txt .md .pdf .docx .xlsx .csv .json
        Returns: {ok, added, skipped, source, chunks}
        """
        p = Path(path)
        if not p.exists():
            return {"ok": False, "error": f"file not found: {path}"}

        # Parse file
        try:
            from vision.documents import parse_file
            sections = parse_file(str(p))
            text     = "\n\n".join(t for _, t in sections if t.strip())
        except Exception:
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                return {"ok": False, "error": str(e)}

        return self.add_text(text, source=p.name)

    def add_text(self, text: str, source: str = "inline",
                 location: str = "") -> dict:
        """Chunk, embed, and add raw text to the index."""
        if not _NP:
            return {"ok": False, "error": "numpy not available"}

        chunks = chunk_text(text)
        if not chunks:
            return {"ok": True, "added": 0, "skipped": 0, "source": source}

        # Embed all chunks
        vecs = _embed(chunks)

        added, skipped = 0, 0
        with self._lock:
            existing_texts = {m.get("text", "") for m in self._meta}
            new_meta, new_vecs = [], []
            for i, (chunk, vec) in enumerate(zip(chunks, vecs)):
                if chunk in existing_texts:
                    skipped += 1
                    continue
                chunk_id = hashlib.md5(f"{source}:{i}:{chunk[:40]}".encode()).hexdigest()[:12]
                self._citations.register(
                    chunk_id, source, location or f"chunk_{i}", chunk[:120]
                )
                new_meta.append({"text": chunk, "source": source,
                                  "location": location or f"chunk_{i}",
                                  "chunk_id": chunk_id})
                new_vecs.append(vec)
                existing_texts.add(chunk)
                added += 1

            if new_meta:
                import numpy as np
                self._meta.extend(new_meta)
                new_arr = np.array(new_vecs, dtype=np.float32)
                self._vecs = (
                    np.vstack([self._vecs, new_arr])
                    if self._vecs is not None else new_arr
                )
                self._rebuild_bm25()
                self._save_index()

        self._cache.invalidate()
        return {"ok": True, "added": added, "skipped": skipped, "source": source,
                "chunks": len(chunks)}

    # ── Retrieval ─────────────────────────────────────────────────────
    def retrieve(self,
                 query:    str,
                 k:        int = None,
                 rerank:   bool = False) -> list[dict]:
        """Hybrid BM25 + semantic retrieval with optional reranking.

        Fusion: score = semantic_weight × cosine + bm25_weight × bm25_norm

        Returns up to k chunks sorted by fusion score, above min_score_threshold.
        """
        k = k or RETRIEVAL.get("top_k", 8)

        cached = self._cache.get(query, k)
        if cached is not None:
            return cached

        if not _NP or not self._meta or self._vecs is None:
            return []

        import numpy as np
        q_vec    = _embed_single(query)
        cos_sims = _cosine_matrix(q_vec, self._vecs)

        bm25_scores = np.zeros(len(self._meta))
        if self._bm25:
            raw_bm25    = self._bm25.score(_tokenise(query))
            mx          = max(raw_bm25) if raw_bm25 else 1.0
            bm25_scores = np.array(raw_bm25) / max(mx, 1e-9)

        sw      = RETRIEVAL.get("semantic_weight", 0.70)
        bw      = RETRIEVAL.get("bm25_weight",     0.30)
        thresh  = RETRIEVAL.get("min_score_threshold", 0.25)
        fusion  = sw * cos_sims + bw * bm25_scores

        top_idx = fusion.argsort()[::-1][:RETRIEVAL.get("rerank_k", 24)]
        results = []
        for idx in top_idx:
            score = float(fusion[idx])
            if score < thresh and results:
                break
            chunk = dict(self._meta[idx])
            chunk["score"]  = round(score, 4)
            chunk["cosine"] = round(float(cos_sims[idx]), 4)
            chunk["bm25"]   = round(float(bm25_scores[idx]), 4)
            results.append(chunk)

        # Optional cross-encoder reranking
        if rerank and results:
            try:
                from encoder.hub import get_hub
                results = get_hub().rerank(
                    query, results, text_key="text",
                    top_k=RETRIEVAL.get("final_k", 5)
                )
            except Exception:
                results = results[:k]
        else:
            results = results[:k]

        self._cache.set(query, k, results)
        return results

    # ── Stats ─────────────────────────────────────────────────────────
    def stats(self) -> dict:
        dim = (self._vecs.shape[1]
               if self._vecs is not None and self._vecs.ndim == 2
               else EMBEDDING.get("dim", 384))
        return {
            "chunks":  len(self._meta),
            "dim":     dim,
            "sources": list({m.get("source","") for m in self._meta}),
            "backend": "hybrid-bm25-semantic",
        }

    def clear(self):
        """Remove all indexed documents."""
        with self._lock:
            self._meta = []
            self._vecs = None
            self._bm25 = None
        self._cache.invalidate()
        self._save_index()

    @property
    def index(self) -> list[dict]:
        return self._meta


# ═══════════════════════════════════════════════════════════════════
# CAGManager — Cache-Augmented Generation
# ═══════════════════════════════════════════════════════════════════

class CAGManager:
    """In-memory document cache for active CAG sessions.

    When a document is opened, its chunks + embeddings are loaded into
    memory for fast multi-turn Q&A without re-embedding.
    Automatically detects stale files (mtime check) and refreshes.
    """

    def __init__(self):
        self._sessions: dict[str, dict] = {}   # path → session
        self._lock = threading.Lock()

    def open(self, source_path: str, rag: RAGPipeline = None) -> dict:
        """Open a document for CAG session.

        Returns session dict with chunks and embeddings.
        """
        p = Path(source_path)
        with self._lock:
            existing = self._sessions.get(str(p))
            if existing:
                # Stale check
                if p.exists() and p.stat().st_mtime > existing.get("mtime", 0):
                    del self._sessions[str(p)]
                else:
                    return existing

        # Parse and embed
        try:
            from vision.documents import parse_file
            sections = parse_file(str(p))
            text     = "\n\n".join(t for _, t in sections if t.strip())
        except Exception:
            text = p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""

        chunks = chunk_text(text)
        vecs   = _embed(chunks) if chunks and _NP else None

        session = {
            "source": str(p),
            "mtime":  p.stat().st_mtime if p.exists() else 0,
            "chunks": chunks,
            "vecs":   vecs,
            "opened": time.time(),
        }
        with self._lock:
            self._sessions[str(p)] = session
        return session

    def retrieve(self,
                 source_path: str,
                 query:       str,
                 k:           int = 5) -> list[dict]:
        """Retrieve from an active CAG session.

        Returns [] if source_path is not in the cache.
        """
        if not _NP:
            return []
        with self._lock:
            session = self._sessions.get(str(Path(source_path)))
        if not session or not session["chunks"] or session["vecs"] is None:
            return []

        import numpy as np
        q_vec = _embed_single(query)
        sims  = _cosine_matrix(q_vec, session["vecs"])
        idx   = sims.argsort()[::-1][:k]
        return [
            {
                "text":   session["chunks"][i],
                "source": Path(source_path).name,
                "score":  round(float(sims[i]), 4),
                "mode":   "cag",
            }
            for i in idx if float(sims[i]) > 0.1
        ]

    def close(self, source_path: str):
        with self._lock:
            self._sessions.pop(str(Path(source_path)), None)

    def active_sources(self) -> list[str]:
        with self._lock:
            return list(self._sessions.keys())


# ═══════════════════════════════════════════════════════════════════
# WorkspaceManager — Multi-document Workspace
# ═══════════════════════════════════════════════════════════════════

class WorkspaceManager:
    """Manages named multi-document workspace sessions.

    Each workspace persists to var/workspaces/<name>.json.
    Supports multi-document retrieval with reuse-score bonus.
    """

    def __init__(self):
        ensure_dirs()
        self._ws_dir = Path(str(WORKSPACE_DIR))
        self._ws_dir.mkdir(parents=True, exist_ok=True)
        self._active: dict[str, dict] = {}
        self._rag    = RAGPipeline()
        self._lock   = threading.Lock()

    def _ws_path(self, name: str) -> Path:
        return self._ws_dir / f"{name}.json"

    def create(self, name: str) -> dict:
        ws = {"name": name, "docs": [], "created": time.time(), "notes": ""}
        self._ws_path(name).write_text(json.dumps(ws, indent=2))
        with self._lock:
            self._active[name] = ws
        return ws

    def load(self, name: str) -> dict | None:
        p = self._ws_path(name)
        if p.exists():
            ws = json.loads(p.read_text())
            with self._lock:
                self._active[name] = ws
            return ws
        return None

    def get(self, name: str) -> dict | None:
        with self._lock:
            if name in self._active:
                return self._active[name]
        return self.load(name)

    def add_document(self, workspace: str, path: str) -> dict:
        """Add a document to a workspace and index it."""
        ws = self.get(workspace) or self.create(workspace)
        if str(path) not in ws.get("docs", []):
            ws["docs"].append(str(path))
            result = self._rag.ingest(path)
            self._ws_path(workspace).write_text(json.dumps(ws, indent=2))
            with self._lock:
                self._active[workspace] = ws
            return result
        return {"ok": True, "added": 0, "skipped": 0, "source": str(path)}

    def search(self, workspace: str, query: str, k: int = 5) -> list[dict]:
        """Search across all documents in a workspace."""
        ws = self.get(workspace)
        if not ws:
            return []
        # Filter RAG results to docs in this workspace
        results = self._rag.retrieve(query, k=k * 3)
        ws_docs = set(Path(d).name for d in ws.get("docs", []))
        filtered = [r for r in results if r.get("source","") in ws_docs]
        return filtered[:k]

    def list_workspaces(self) -> list[str]:
        return [p.stem for p in self._ws_dir.glob("*.json")]


# ═══════════════════════════════════════════════════════════════════
# Query Rewriter and Expander
# ═══════════════════════════════════════════════════════════════════

def rewrite_query(query: str, history: list[dict]) -> str:
    """Resolve pronouns and context references using conversation history.

    Simple but effective: replaces pronouns (it, this, that, they, he, she)
    with the most recent significant noun from history.
    """
    if not history or not re.search(r"\b(it|this|that|they|he|she|them|its)\b",
                                    query, re.I):
        return query

    recent_content = " ".join(
        h.get("content", "") for h in history[-3:]
    )
    nouns = re.findall(r"\b[A-Z][a-z]{2,}\b", recent_content)
    if not nouns:
        return query

    target = nouns[-1]
    rewritten = re.sub(
        r"\b(it|this|that)\b", target, query, flags=re.I, count=1
    )
    return rewritten


def expand_query(query: str) -> str:
    """Expand query with synonyms from var/synonyms.json.

    File format: {"term": ["syn1", "syn2"], ...}
    Hot-reload safe: file is read on every call (small file, fast IO).
    """
    if not RETRIEVAL.get("query_expansion", True):
        return query
    syns_path = Path(str(SYNONYMS_PATH))
    if not syns_path.exists():
        return query
    try:
        syns = json.loads(syns_path.read_text())
    except Exception:
        return query

    terms   = query.lower().split()
    extras  = []
    for term in terms:
        if term in syns:
            extras.extend(syns[term][:2])
    if extras:
        return query + " " + " ".join(extras)
    return query


# ═══════════════════════════════════════════════════════════════════
# HybridRetriever — Main Retrieval Orchestrator
# ═══════════════════════════════════════════════════════════════════

class HybridRetriever:
    """Orchestrates the full v3 retrieval pipeline.

    Pipeline for each query:
      1. Query rewrite (history-aware pronoun resolution)
      2. Query expansion (synonym + domain terms)
      3. RAG retrieval (semantic + BM25 fusion)
      4. CAG retrieval (active document session)
      5. Cross-encoder reranking
      6. Multi-hop extension (if confidence below threshold)
      7. Analytics tracking
      8. Self-improvement miss logging

    All steps configurable via config.py RETRIEVAL section.
    """

    def __init__(self,
                 rag:     RAGPipeline     = None,
                 cag:     CAGManager      = None,
                 memory   = None,
                 kg       = None):
        self._rag    = rag     or RAGPipeline()
        self._cag    = cag     or CAGManager()
        self._memory = memory
        self._kg     = kg
        self._stats  = {
            "queries": 0, "cache_hits": 0, "multi_hop_used": 0,
            "avg_score": 0.0, "misses": 0,
        }
        self._stats_path = Path(str(CACHE_DIR)) / "retrieval_stats.json"
        self._load_stats()

    def _load_stats(self):
        if self._stats_path.exists():
            try:
                self._stats = json.loads(self._stats_path.read_text())
            except Exception:
                pass

    def _save_stats(self):
        try:
            self._stats_path.parent.mkdir(parents=True, exist_ok=True)
            self._stats_path.write_text(json.dumps(self._stats))
        except Exception:
            pass

    def retrieve(self,
                 query:       str,
                 session:     str = "default",
                 history:     list[dict] = None,
                 active_doc:  str = None,
                 k:           int = None,
                 rerank:      bool = True) -> dict:
        """Full hybrid retrieval pipeline.

        Args:
            query:      User query string.
            session:    Session ID (for analytics tracking).
            history:    Conversation history for query rewriting.
            active_doc: Path to active CAG document (optional).
            k:          Number of final results.
            rerank:     Enable cross-encoder reranking.

        Returns:
            {
                "chunks":    list[dict],   # retrieved chunks
                "mode":      str,          # "cag" | "hybrid-rag" | "empty"
                "query":     str,          # original query
                "rewritten": str,          # rewritten query
                "citations": str,          # formatted citation string
                "hops":      int,          # number of retrieval hops
                "confidence":float,        # avg score of top results
            }
        """
        k       = k or RETRIEVAL.get("final_k", 5)
        history = history or []
        self._stats["queries"] = self._stats.get("queries", 0) + 1

        # Step 1: Query rewrite
        rewritten = rewrite_query(query, history) \
            if RETRIEVAL.get("query_rewrite", True) else query

        # Step 2: Query expansion
        expanded = expand_query(rewritten)

        # Step 3: CAG retrieval (active document takes priority)
        cag_chunks: list[dict] = []
        if active_doc:
            cag_chunks = self._cag.retrieve(active_doc, expanded, k=k)

        # Step 4: RAG retrieval
        rag_chunks = self._rag.retrieve(expanded, k=k * 3, rerank=False)

        # Step 5: Merge (CAG first, then RAG deduped)
        seen_texts  = {c["text"][:80] for c in cag_chunks}
        merged      = list(cag_chunks)
        for c in rag_chunks:
            if c["text"][:80] not in seen_texts:
                merged.append(c)
                seen_texts.add(c["text"][:80])

        # Step 6: Cross-encoder reranking
        if rerank and merged:
            try:
                from encoder.hub import get_hub
                merged = get_hub().rerank(
                    expanded, merged, text_key="text",
                    top_k=RETRIEVAL.get("final_k", 5)
                )
            except Exception:
                merged = merged[:k]
        else:
            merged = merged[:k]

        # Step 7: Multi-hop extension
        hops = 1
        max_hops  = RETRIEVAL.get("multi_hop_max", 3)
        min_score = RETRIEVAL.get("min_score_threshold", 0.25)
        if merged:
            top_score = merged[0].get("score", merged[0].get("rerank_score", 0.0))
            while (hops < max_hops and
                   float(top_score) < min_score + 0.1 * hops):
                # Extract key terms from top result and re-query
                top_text = merged[0].get("text", "")
                hop_q    = expand_query(
                    " ".join(re.findall(r"[a-zA-Z]{4,}", top_text)[:8])
                )
                hop_res  = self._rag.retrieve(hop_q, k=3, rerank=False)
                for c in hop_res:
                    if c["text"][:80] not in seen_texts:
                        merged.append(c)
                        seen_texts.add(c["text"][:80])
                hops += 1
                self._stats["multi_hop_used"] = self._stats.get("multi_hop_used",0)+1

        # Step 8: Final truncation
        final = merged[:k]

        # Confidence + mode
        scores     = [c.get("score", c.get("rerank_score", 0.0)) for c in final]
        confidence = round(sum(scores) / max(len(scores), 1), 3) if scores else 0.0

        if not final:
            self._stats["misses"] = self._stats.get("misses",0) + 1
            if RETRIEVAL.get("self_improve", True):
                self._log_miss(query)

        mode = "cag" if cag_chunks and final and final[0].get("mode") == "cag" \
               else "hybrid-rag" if final else "empty"

        try:
            cit_engine = CitationEngine()
            citations  = cit_engine.format_citations(final)
        except Exception:
            citations = ""

        self._save_stats()
        return {
            "chunks":     final,
            "mode":       mode,
            "query":      query,
            "rewritten":  rewritten,
            "expanded":   expanded,
            "citations":  citations,
            "hops":       hops,
            "confidence": confidence,
        }

    def _log_miss(self, query: str):
        """Log a retrieval miss for self-improvement analysis."""
        try:
            miss_path = Path(str(CACHE_DIR)) / "retrieval_misses.jsonl"
            with open(str(miss_path), "a") as f:
                f.write(json.dumps({
                    "ts": time.time(), "query": query[:200]
                }) + "\n")
        except Exception:
            pass

    def invalidate_cache(self):
        self._rag._cache.invalidate()

    def analytics(self) -> dict:
        return dict(self._stats)
