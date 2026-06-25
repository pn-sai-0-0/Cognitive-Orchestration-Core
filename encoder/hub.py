"""
CognitiveOC v3 — Encoder Intelligence Stack (Hub)
===================================================

Manages all 13 encoders as a unified hub.  Each encoder is:
  • A sentence-transformer-compatible model (HuggingFace)
  • ONNX-exportable for AMD NPU (DirectML / OnnxRuntime)
  • Loaded lazily — only allocated when first called
  • Device-routed: NPU > CUDA > CPU in priority order

Hardware execution map (from architecture spec):
  NPU (AMD 15GB) : semantic, cross_encoder, emotion, summarization,
                   judge, kg  — heavy embedding workloads
  CPU            : intent, memory, safety, teaching, goal, planning,
                   dataset  — lightweight / high-frequency

Why a Hub?
  Single import point for all encoder calls.  Prevents duplicate model
  loading when multiple subsystems (retrieval, cognition, memory) need
  the same encoder.  LRU-manages GPU/NPU memory across all 13 encoders.

Why separate encoders instead of one large model?
  Each domain (emotion, intent, safety …) benefits from specialised
  training objectives.  A single generic encoder produces suboptimal
  representations for e.g. emotion detection vs semantic retrieval.
  Separate ~22-110M models cost far less total memory than one
  monolithic 700M encoder, and run on NPU/CPU in parallel.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any

try:
    import numpy as np
    _NP = True
except ImportError:
    _NP = False

try:
    from config import ENCODERS, ONNX_DIR, BASE_DIR
except ImportError:
    ENCODERS  = {}
    ONNX_DIR  = Path("var/onnx")
    BASE_DIR  = Path(".")


# ── Device priority ───────────────────────────────────────────────────
def _best_device(preferred: str) -> str:
    """Resolve device string respecting what is actually available."""
    if preferred == "npu":
        # AMD NPU via ONNX Runtime DirectML
        try:
            import onnxruntime as ort
            if "DmlExecutionProvider" in ort.get_available_providers():
                return "npu"
        except ImportError:
            pass
        return "cpu"
    if preferred == "cuda":
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"
    return "cpu"


# ═══════════════════════════════════════════════════════════════════
# Single Encoder Wrapper
# ═══════════════════════════════════════════════════════════════════

class _Encoder:
    """Lazy-loading wrapper around one sentence-transformer encoder.

    Priority: ONNX (NPU/CPU) → sentence-transformers → hash-TF fallback.
    Thread-safe: _lock serialises model loading; encode() is thread-safe
    once loaded (inference-only, no mutable state).
    """

    def __init__(self, name: str, cfg: dict):
        self.name     = name
        self.cfg      = cfg
        self.device   = _best_device(cfg.get("device", "cpu"))
        self._model   = None
        self._lock    = threading.Lock()
        self._backend = "unloaded"
        self._last_used = 0.0
        self.dim      = cfg.get("dim", 384)

    def _load(self):
        """Load encoder model (called once, lazily)."""
        # 1. Try ONNX Runtime (NPU / CPU)
        onnx_path = self.cfg.get("onnx_path")
        if onnx_path and Path(onnx_path).exists():
            try:
                import onnxruntime as ort
                providers = (
                    ["DmlExecutionProvider"] if self.device == "npu"
                    else ["CPUExecutionProvider"]
                )
                sess = ort.InferenceSession(str(onnx_path),
                                            providers=providers)
                self._model   = ("onnx", sess)
                self._backend = f"onnx({self.device})"
                return
            except Exception:
                pass

        # 2. Try sentence-transformers (CPU or CUDA)
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            st_device = "cuda" if self.device == "cuda" else "cpu"
            model_id  = self.cfg.get("model", "sentence-transformers/all-MiniLM-L6-v2")
            m = SentenceTransformer(model_id, device=st_device)
            self._model   = ("st", m)
            self._backend = f"sentence_transformers({st_device})"
            self.dim      = m.get_sentence_embedding_dimension() or self.dim
            return
        except Exception:
            pass

        # 3. Hash-TF fallback (zero dependencies)
        self._model   = ("hash", None)
        self._backend = "hash_tf_fallback"

    def _ensure(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    self._load()
        self._last_used = time.time()

    def encode(self,
               texts: list[str],
               normalize: bool = True,
               batch_size: int = None) -> "np.ndarray":
        """Encode a list of texts → numpy float32 matrix (N, dim).

        Falls back gracefully through ONNX → sentence-transformers → hash-TF.
        Always returns float32 numpy array regardless of backend.
        """
        if not _NP:
            raise RuntimeError("numpy is required for encoders.")
        self._ensure()
        bs = batch_size or self.cfg.get("batch", 32)

        kind, model = self._model

        if kind == "onnx":
            return self._encode_onnx(texts, model, normalize, bs)
        if kind == "st":
            return self._encode_st(texts, model, normalize, bs)
        return self._encode_hash(texts, normalize)

    def _encode_onnx(self, texts, sess, normalize, batch_size):
        import numpy as np
        # Requires tokenizer alongside the ONNX model
        try:
            from transformers import AutoTokenizer  # type: ignore
            model_id = self.cfg.get("model", "sentence-transformers/all-MiniLM-L6-v2")
            tok = AutoTokenizer.from_pretrained(model_id)
        except Exception:
            return self._encode_hash(texts, normalize)

        all_vecs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            enc   = tok(batch, padding=True, truncation=True,
                        max_length=self.cfg.get("max_len", 512),
                        return_tensors="np")
            ort_inputs = {k: v for k, v in enc.items()
                          if k in [inp.name for inp in sess.get_inputs()]}
            out = sess.run(None, ort_inputs)
            # Mean-pool last hidden state
            hidden = out[0]                         # (B, T, D)
            mask   = enc.get("attention_mask", np.ones(hidden.shape[:2]))
            mask_e = mask[:, :, np.newaxis].astype(np.float32)
            vec    = (hidden * mask_e).sum(1) / mask_e.sum(1).clip(min=1e-9)
            all_vecs.append(vec.astype(np.float32))

        vecs = np.concatenate(all_vecs, axis=0)
        if normalize:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True).clip(min=1e-9)
            vecs  = vecs / norms
        return vecs

    def _encode_st(self, texts, model, normalize, batch_size):
        import numpy as np
        vecs = model.encode(
            texts, batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vecs.astype(np.float32)

    def _encode_hash(self, texts, normalize) -> "np.ndarray":
        """Hash-TF fallback — deterministic, zero deps, low quality."""
        import numpy as np
        dim  = self.dim
        vecs = np.zeros((len(texts), dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in text.lower().split():
                h = hash(tok) % dim
                vecs[i, h] += 1.0
        if normalize:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True).clip(min=1e-9)
            vecs  = vecs / norms
        return vecs

    # ── ONNX Export ──────────────────────────────────────────────────
    def export_onnx(self, output_path: str = None) -> str:
        """Export this encoder to ONNX for NPU deployment.

        Requires: optimum, transformers, torch.
        Output: <output_path>/<name>_encoder.onnx
        """
        out = output_path or str(ONNX_DIR / f"{self.name}_encoder.onnx")
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        model_id = self.cfg.get("model")
        if not model_id:
            raise ValueError(f"No model ID configured for encoder '{self.name}'")

        try:
            from optimum.onnxruntime import ORTModelForFeatureExtraction  # type: ignore
            from transformers import AutoTokenizer                         # type: ignore
            tok = AutoTokenizer.from_pretrained(model_id)
            m   = ORTModelForFeatureExtraction.from_pretrained(
                model_id, export=True
            )
            m.save_pretrained(str(Path(out).parent / self.name))
            print(f"[encoder] ONNX exported: {self.name} → {out}")
            return out
        except Exception as e:
            raise RuntimeError(f"ONNX export failed for '{self.name}': {e}")

    def __repr__(self) -> str:
        return (f"_Encoder(name={self.name!r}, "
                f"backend={self._backend!r}, dim={self.dim})")


# ═══════════════════════════════════════════════════════════════════
# Cross-Encoder Reranker
# ═══════════════════════════════════════════════════════════════════

class _CrossEncoder:
    """Cross-encoder reranker for precision ranking after retrieval.

    Takes (query, passage) pairs and produces relevance scores.
    Higher score = more relevant.
    """

    def __init__(self, cfg: dict):
        self.cfg      = cfg
        self._model   = None
        self._lock    = threading.Lock()
        self._backend = "unloaded"
        self.device   = _best_device(cfg.get("device", "npu"))

    def _load(self):
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
            model_id = self.cfg.get("model",
                                    "cross-encoder/ms-marco-MiniLM-L-6-v2")
            device = "cuda" if self.device == "cuda" else "cpu"
            self._model   = CrossEncoder(model_id, device=device)
            self._backend = f"sentence_transformers_cross({device})"
            return
        except Exception:
            pass
        # Fallback: score by token overlap
        self._model   = "fallback"
        self._backend = "token_overlap_fallback"

    def _ensure(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    self._load()

    def score(self, query: str, passages: list[str]) -> list[float]:
        """Score (query, passage) pairs → list of relevance floats."""
        self._ensure()
        if self._model == "fallback":
            return self._overlap_score(query, passages)
        try:
            pairs  = [(query, p) for p in passages]
            scores = self._model.predict(
                pairs, batch_size=self.cfg.get("batch", 16)
            )
            return [float(s) for s in scores]
        except Exception:
            return self._overlap_score(query, passages)

    @staticmethod
    def _overlap_score(query: str, passages: list[str]) -> list[float]:
        """Token overlap fallback scoring."""
        import re
        q_toks = set(re.findall(r"[a-z0-9]+", query.lower()))
        scores = []
        for p in passages:
            p_toks = set(re.findall(r"[a-z0-9]+", p.lower()))
            scores.append(
                len(q_toks & p_toks) / max(len(q_toks | p_toks), 1)
            )
        return scores

    def rerank(self,
               query: str,
               candidates: list[dict],
               text_key: str = "text",
               top_k: int = 5) -> list[dict]:
        """Rerank a list of candidate dicts by cross-encoder relevance.

        Args:
            query:      The original user query.
            candidates: List of dicts, each with a text_key field.
            text_key:   Key in each dict that contains the passage text.
            top_k:      Number of top results to return.

        Returns:
            Top-k candidates sorted by relevance, each with a
            "rerank_score" field added.
        """
        if not candidates:
            return []
        texts  = [c.get(text_key, "") for c in candidates]
        scores = self.score(query, texts)
        for c, s in zip(candidates, scores):
            c["rerank_score"] = round(float(s), 4)
        return sorted(candidates, key=lambda c: -c["rerank_score"])[:top_k]


# ═══════════════════════════════════════════════════════════════════
# Emotion Classifier (special-purpose, uses emotion encoder labels)
# ═══════════════════════════════════════════════════════════════════

class _EmotionClassifier:
    """Emotion detection from text — returns top-N emotion labels + scores.

    Uses the go_emotions label set (28 classes) by default.
    Falls back to a simple keyword heuristic if encoder not available.
    """

    _KEYWORD_MAP = {
        "anger":       ["angry","frustrated","annoyed","mad","furious"],
        "confusion":   ["confused","unsure","don't understand","unclear"],
        "curiosity":   ["curious","wondering","interested","want to know"],
        "fear":        ["scared","afraid","worried","anxious","nervous"],
        "joy":         ["happy","great","excited","awesome","love"],
        "sadness":     ["sad","unhappy","disappointed","upset","depressed"],
        "neutral":     [],
    }

    def __init__(self, encoder: _Encoder, cfg: dict):
        self._enc    = encoder
        self._cfg    = cfg
        self._labels = cfg.get("labels", list(self._KEYWORD_MAP.keys()))

    def detect(self, text: str, top_n: int = 3) -> list[dict]:
        """Return top-N emotion predictions for the input text.

        Returns:
            [{"emotion": str, "score": float}, …]  sorted by score desc.
        """
        if not text.strip():
            return [{"emotion": "neutral", "score": 1.0}]

        # Try sentence-transformers zero-shot via encoder
        try:
            if _NP and self._enc._model is not None:
                text_vec   = self._enc.encode([text])[0]
                label_vecs = self._enc.encode(self._labels)
                import numpy as np
                scores = (label_vecs @ text_vec).tolist()
                ranked = sorted(zip(self._labels, scores),
                                key=lambda x: -x[1])
                threshold = self._cfg.get("confidence_threshold", 0.50)
                return [
                    {"emotion": em, "score": round(float(sc), 4)}
                    for em, sc in ranked[:top_n]
                    if float(sc) >= threshold
                ] or [{"emotion": "neutral", "score": 1.0}]
        except Exception:
            pass

        # Keyword fallback
        return self._keyword_detect(text, top_n)

    def _keyword_detect(self, text: str, top_n: int) -> list[dict]:
        tl = text.lower()
        scores: dict[str, float] = {}
        for emotion, keywords in self._KEYWORD_MAP.items():
            hit = sum(1 for kw in keywords if kw in tl)
            if hit:
                scores[emotion] = hit / max(len(keywords), 1)
        if not scores:
            return [{"emotion": "neutral", "score": 1.0}]
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return [{"emotion": e, "score": round(s, 4)} for e, s in ranked[:top_n]]


# ═══════════════════════════════════════════════════════════════════
# Encoder Hub
# ═══════════════════════════════════════════════════════════════════

class EncoderHub:
    """Central hub for all COC v3 encoders.

    Usage:
        hub = EncoderHub()

        # Semantic embedding (retrieval)
        vecs = hub.encode("semantic", ["text one", "text two"])

        # Emotion detection
        emotions = hub.detect_emotion("I am really frustrated right now")

        # Cross-encoder reranking
        ranked = hub.rerank("my query", [{"text": "passage 1"}, ...])

        # Status
        print(hub.status())

    All encoders are lazy-loaded — no memory is allocated until the
    encoder is first called.
    """

    def __init__(self, cfg: dict = None):
        self._cfg   = cfg or ENCODERS
        self._enc: dict[str, _Encoder] = {}
        self._cross: _CrossEncoder | None = None
        self._emotion: _EmotionClassifier | None = None
        self._lock  = threading.Lock()
        self._init_registry()

    def _init_registry(self):
        """Register all encoders from config (lazy — no models loaded yet)."""
        for name, ecfg in self._cfg.items():
            if name == "cross_encoder":
                continue   # handled separately
            self._enc[name] = _Encoder(name, ecfg)

    def _get(self, name: str) -> _Encoder:
        """Get encoder by name; raise clear error if unknown."""
        if name not in self._enc:
            available = list(self._enc.keys())
            raise ValueError(
                f"Unknown encoder '{name}'. Available: {available}"
            )
        return self._enc[name]

    # ── Primary encode API ────────────────────────────────────────────
    def encode(self,
               encoder_name: str,
               texts: list[str],
               normalize: bool = True) -> "np.ndarray":
        """Encode texts with the named encoder.

        Args:
            encoder_name: One of: semantic, intent, emotion, memory,
                          safety, summarization, judge, teaching, goal,
                          kg, dataset, planning.
            texts:        List of input strings.
            normalize:    L2-normalise output vectors (default True).

        Returns:
            numpy float32 array of shape (N, dim).
        """
        return self._get(encoder_name).encode(texts, normalize=normalize)

    def encode_single(self, encoder_name: str, text: str) -> "np.ndarray":
        """Convenience: encode a single string → 1-D vector."""
        return self.encode(encoder_name, [text])[0]

    # ── Cross-encoder reranking ───────────────────────────────────────
    def rerank(self,
               query: str,
               candidates: list[dict],
               text_key: str = "text",
               top_k: int = 5) -> list[dict]:
        """Rerank candidates with cross-encoder.  See _CrossEncoder.rerank."""
        if self._cross is None:
            with self._lock:
                if self._cross is None:
                    ce_cfg = self._cfg.get("cross_encoder", {})
                    self._cross = _CrossEncoder(ce_cfg)
        return self._cross.rerank(query, candidates, text_key, top_k)

    # ── Emotion detection ─────────────────────────────────────────────
    def detect_emotion(self, text: str, top_n: int = 3) -> list[dict]:
        """Detect top-N emotions from text.  See _EmotionClassifier."""
        if self._emotion is None:
            with self._lock:
                if self._emotion is None:
                    em_enc     = self._get("emotion")
                    em_cfg     = self._cfg.get("emotion", {})
                    self._emotion = _EmotionClassifier(em_enc, em_cfg)
        return self._emotion.detect(text, top_n)

    # ── Similarity utilities ──────────────────────────────────────────
    def cosine_similarity(self,
                          encoder_name: str,
                          query: str,
                          candidates: list[str]) -> list[float]:
        """Compute cosine similarity of query vs all candidates."""
        if not _NP:
            return [0.0] * len(candidates)
        import numpy as np
        q_vec  = self.encode(encoder_name, [query])[0]
        c_vecs = self.encode(encoder_name, candidates)
        scores = (c_vecs @ q_vec).tolist()
        return [round(float(s), 4) for s in scores]

    def top_k_similar(self,
                      encoder_name: str,
                      query: str,
                      candidates: list[str],
                      k: int = 5) -> list[tuple[int, float]]:
        """Return indices and scores of top-k most similar candidates."""
        if not candidates:
            return []
        scores   = self.cosine_similarity(encoder_name, query, candidates)
        indexed  = sorted(enumerate(scores), key=lambda x: -x[1])
        return [(i, s) for i, s in indexed[:k]]

    # ── ONNX Export ──────────────────────────────────────────────────
    def export_onnx(self, encoder_name: str, output_path: str = None) -> str:
        """Export a named encoder to ONNX for NPU deployment."""
        return self._get(encoder_name).export_onnx(output_path)

    def export_all_onnx(self, output_dir: str = None) -> dict[str, str]:
        """Export all encoders to ONNX.  Skips those that fail."""
        results = {}
        for name in self._enc:
            try:
                path = self._get(name).export_onnx(output_dir)
                results[name] = path
            except Exception as e:
                results[name] = f"FAILED: {e}"
        return results

    # ── Status ────────────────────────────────────────────────────────
    def status(self) -> dict:
        """Return loaded/unloaded status of all encoders."""
        return {
            name: {
                "backend":    enc._backend,
                "device":     enc.device,
                "dim":        enc.dim,
                "loaded":     enc._model is not None,
                "last_used":  enc._last_used,
            }
            for name, enc in self._enc.items()
        }

    def warm_up(self, encoder_names: list[str] = None):
        """Pre-load encoders to avoid latency on first real request.

        Args:
            encoder_names: subset to warm up; None = all encoders.
        """
        names = encoder_names or list(self._enc.keys())
        for name in names:
            try:
                enc = self._get(name)
                enc._ensure()
                print(f"[encoder_hub] Warmed up {name} → {enc._backend}")
            except Exception as e:
                print(f"[encoder_hub] Warm-up failed for {name}: {e}")

    def __repr__(self) -> str:
        loaded = sum(1 for e in self._enc.values() if e._model is not None)
        return (f"EncoderHub(encoders={len(self._enc)}, "
                f"loaded={loaded}, cross={'yes' if self._cross else 'no'})")


# ═══════════════════════════════════════════════════════════════════
# Module-level singleton (avoids re-loading across imports)
# ═══════════════════════════════════════════════════════════════════
_HUB: EncoderHub | None = None
_HUB_LOCK = threading.Lock()


def get_hub() -> EncoderHub:
    """Return the module-level singleton EncoderHub."""
    global _HUB
    if _HUB is None:
        with _HUB_LOCK:
            if _HUB is None:
                _HUB = EncoderHub()
    return _HUB
