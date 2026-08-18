"""
CognitiveOC v3 — Observability System
=======================================

Tracks all runtime metrics across every subsystem:

Hardware metrics (polled every hardware_poll_s seconds):
  CPU:  usage %, per-core, frequency
  GPU:  VRAM used/total, utilisation %, temperature
  NPU:  memory used (AMD Ryzen AI via DirectML)
  RAM:  used/total GB, available GB

Request metrics (recorded per request):
  latency_ms, tokens_in, tokens_out, tokens_per_sec,
  intent, backend (model/fallback), retrieval_mode

Subsystem metrics (accumulated):
  retrieval:  hit_rate, cache_hit_rate, avg_score, multi_hop_rate
  memory:     recall_count, store_count, avg_score
  kg:         query_count, extract_count, triple_count
  cognition:  mode, emotion_detected, intent_detected
  reasoning:  avg_confidence, type_distribution
  workflow:   active_count, completed_count, failed_count

Training metrics (read from train_log.jsonl, live during training):
  step, train_loss, val_loss, perplexity, grad_norm, lr, tokens_per_sec

All metrics exposed via:
  GET /api/metrics          — current snapshot
  GET /api/metrics/history  — recent history (last N entries)
  GET /api/metrics/hardware — hardware-only snapshot
  Observability.snapshot()  — Python dict

File: observability/metrics.py
Used by: engine.py, ui/app.py, train/train_model.py
Persists: var/metrics.json, var/logs/runtime.jsonl
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from pathlib import Path

try:
    from config import OBSERVABILITY, LOG_DIR, CHECKPOINT_DIR, ensure_dirs
except ImportError:
    OBSERVABILITY  = {}
    LOG_DIR        = Path("var/logs")
    CHECKPOINT_DIR = Path("var/checkpoints")
    def ensure_dirs(): pass


# ═══════════════════════════════════════════════════════════════════
# Hardware Poller
# ═══════════════════════════════════════════════════════════════════

class HardwarePoller:
    """Poll CPU / GPU / NPU / RAM metrics in a background thread.

    Metrics are stored in self.latest and updated every poll_interval_s.
    All metrics are floats/ints — no external dependencies required
    (psutil used if available, falls back to /proc on Linux).
    """

    def __init__(self, poll_interval_s: float = None):
        self._interval = (poll_interval_s or
                          OBSERVABILITY.get("hardware_poll_s", 2))
        self.latest: dict = self._empty()
        self._thread: threading.Thread | None = None
        self._stop   = threading.Event()
        self._lock   = threading.Lock()

    def _empty(self) -> dict:
        return {
            "cpu_pct":      0.0,
            "cpu_freq_mhz": 0.0,
            "ram_used_gb":  0.0,
            "ram_total_gb": 0.0,
            "ram_avail_gb": 0.0,
            "gpu_used_mb":  0.0,
            "gpu_total_mb": 0.0,
            "gpu_util_pct": 0.0,
            "gpu_temp_c":   0.0,
            "npu_used_mb":  0.0,
            "ts":           time.time(),
        }

    def _poll_once(self) -> dict:
        d = self._empty()

        # ── CPU + RAM (psutil preferred, /proc fallback) ──────────────
        try:
            import psutil
            d["cpu_pct"]      = psutil.cpu_percent(interval=0.1)
            freq = psutil.cpu_freq()
            d["cpu_freq_mhz"] = round(freq.current, 1) if freq else 0.0
            vm = psutil.virtual_memory()
            d["ram_used_gb"]  = round(vm.used  / 1e9, 2)
            d["ram_total_gb"] = round(vm.total / 1e9, 2)
            d["ram_avail_gb"] = round(vm.available / 1e9, 2)
        except ImportError:
            # /proc fallback (Linux only)
            try:
                with open("/proc/meminfo") as f:
                    mem = {k: int(v.split()[0]) for k, v in
                           (line.split(":", 1) for line in f if ":" in line)}
                total = mem.get("MemTotal", 0)
                avail = mem.get("MemAvailable", 0)
                d["ram_total_gb"] = round(total / 1e6, 2)
                d["ram_avail_gb"] = round(avail / 1e6, 2)
                d["ram_used_gb"]  = round((total - avail) / 1e6, 2)
            except Exception:
                pass
        except Exception:
            pass

        # ── GPU (PyTorch CUDA) ────────────────────────────────────────
        try:
            import torch
            if torch.cuda.is_available():
                d["gpu_used_mb"]  = round(
                    torch.cuda.memory_allocated() / 1e6, 1)
                d["gpu_total_mb"] = round(
                    torch.cuda.get_device_properties(0).total_memory / 1e6, 1)
        except Exception:
            pass

        # ── GPU utilisation + temp (nvidia-smi) ───────────────────────
        try:
            import subprocess
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=utilization.gpu,temperature.gpu",
                 "--format=csv,noheader,nounits"],
                timeout=2, stderr=subprocess.DEVNULL
            ).decode().strip()
            if out:
                parts = out.split(",")
                d["gpu_util_pct"] = float(parts[0].strip()) if parts else 0.0
                d["gpu_temp_c"]   = float(parts[1].strip()) if len(parts)>1 else 0.0
        except Exception:
            pass

        d["ts"] = time.time()
        return d

    def start(self):
        """Start background polling thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="hw-poller", daemon=True
        )
        self._thread.start()

    def _loop(self):
        while not self._stop.is_set():
            data = self._poll_once()
            with self._lock:
                self.latest = data
            self._stop.wait(self._interval)

    def stop(self):
        self._stop.set()

    def get(self) -> dict:
        with self._lock:
            return dict(self.latest)


# ═══════════════════════════════════════════════════════════════════
# Observability
# ═══════════════════════════════════════════════════════════════════

class Observability:
    """Central metrics hub for COC v3.

    Records every request and subsystem event.
    Exposes snapshot(), history(), and hardware().
    Persists to var/metrics.json and var/logs/runtime.jsonl.

    Usage:
        obs = Observability()
        obs.record(request_dict)       # called by engine after each request
        snap = obs.snapshot()          # full current metrics dict
    """

    def __init__(self):
        ensure_dirs()
        self._metrics_path = Path(str(OBSERVABILITY.get(
            "metrics_path", "var/metrics.json"
        )))
        self._log_path     = Path(str(OBSERVABILITY.get(
            "log_path", "var/logs/runtime.jsonl"
        )))
        self._train_log    = Path(str(CHECKPOINT_DIR)) / "train_log.jsonl"
        self._max_history  = OBSERVABILITY.get("history_max", 10_000)
        self._lock         = threading.RLock()
        self._hw           = HardwarePoller()
        self._hw.start()

        # Accumulated metrics
        self._total_requests   = 0
        self._total_tokens_in  = 0
        self._total_tokens_out = 0
        self._total_latency_ms = 0.0
        self._errors           = 0
        self._intent_counts: dict[str,int] = {}
        self._history: deque  = deque(maxlen=self._max_history)

        # Subsystem accumulators
        self._retrieval  = {"hits":0,"misses":0,"cache_hits":0,"multi_hops":0,
                            "total_score":0.0,"queries":0}
        self._memory_acc = {"recalls":0,"stores":0,"total_score":0.0}
        self._kg_acc     = {"queries":0,"extracts":0,"triples_added":0}
        self._cognition  = {"mode":"full","emotions":{},"intents":{}}
        self._reasoning  = {"queries":0,"total_conf":0.0,"types":{}}
        self._workflow   = {"active":0,"completed":0,"failed":0}
        self._tool_acc   = {"calls":0,"success":0,"errors":0}

        self._load()

    def _load(self):
        """Load persisted metrics on startup."""
        if self._metrics_path.exists():
            try:
                data = json.loads(self._metrics_path.read_text())
                self._total_requests   = data.get("total_requests",   0)
                self._total_tokens_in  = data.get("total_tokens_in",  0)
                self._total_tokens_out = data.get("total_tokens_out", 0)
                self._errors           = data.get("errors",           0)
                self._intent_counts    = data.get("intent_counts",    {})
            except Exception:
                pass

    def _save(self):
        """Persist summary metrics to disk (non-blocking best-effort)."""
        try:
            self._metrics_path.parent.mkdir(parents=True, exist_ok=True)
            self._metrics_path.write_text(json.dumps({
                "total_requests":   self._total_requests,
                "total_tokens_in":  self._total_tokens_in,
                "total_tokens_out": self._total_tokens_out,
                "errors":           self._errors,
                "intent_counts":    self._intent_counts,
                "updated":          time.strftime("%Y-%m-%dT%H:%M:%S"),
            }))
        except Exception:
            pass

    # ── Record a request ─────────────────────────────────────────────
    def record(self, req: dict):
        """Record one completed request.

        Expected keys in req:
          latency_ms, tokens_in, tokens_out, intent, backend,
          retrieval_mode, retrieval_score, memory_hits,
          kg_facts, tool_name, error (optional)
        """
        with self._lock:
            self._total_requests   += 1
            self._total_tokens_in  += req.get("tokens_in",  0)
            self._total_tokens_out += req.get("tokens_out", 0)
            lat = req.get("latency_ms", 0.0)
            self._total_latency_ms += lat
            if req.get("error"):
                self._errors += 1

            intent = req.get("intent", "chat")
            self._intent_counts[intent] = self._intent_counts.get(intent, 0) + 1

            # Retrieval accumulator
            ret_score = req.get("retrieval_score", 0.0)
            if req.get("retrieval_mode") not in (None, "empty"):
                self._retrieval["hits"]  += 1
                self._retrieval["total_score"] += float(ret_score)
            else:
                self._retrieval["misses"] += 1
            if req.get("retrieval_cache_hit"):
                self._retrieval["cache_hits"] += 1
            if req.get("retrieval_hops", 1) > 1:
                self._retrieval["multi_hops"]  += 1
            self._retrieval["queries"] += 1

            # Memory accumulator
            n_mem = req.get("memory_hits", 0)
            if n_mem:
                self._memory_acc["recalls"] += n_mem
            if req.get("memory_stored"):
                self._memory_acc["stores"] += 1

            # KG accumulator
            n_kg = req.get("kg_facts", 0)
            if n_kg:
                self._kg_acc["queries"] += 1

            # Tool accumulator
            if req.get("tool_name"):
                self._tool_acc["calls"] += 1
                if req.get("tool_success"):
                    self._tool_acc["success"] += 1
                else:
                    self._tool_acc["errors"] += 1

            # Cognition
            em = req.get("emotion", "")
            if em:
                self._cognition["emotions"][em] = \
                    self._cognition["emotions"].get(em, 0) + 1
            intn = req.get("intent", "")
            if intn:
                self._cognition["intents"][intn] = \
                    self._cognition["intents"].get(intn, 0) + 1

            # History ring-buffer
            self._history.append({
                "ts":          time.strftime("%Y-%m-%dT%H:%M:%S"),
                "latency_ms":  round(lat, 1),
                "tokens_in":   req.get("tokens_in", 0),
                "tokens_out":  req.get("tokens_out", 0),
                "intent":      intent,
                "backend":     req.get("backend", ""),
                "error":       bool(req.get("error")),
            })

        # Async log + persist (every 10 requests)
        self._log_entry(req)
        if self._total_requests % 10 == 0:
            self._save()

    def _log_entry(self, req: dict):
        """Append one request record to runtime.jsonl."""
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            record = json.dumps({
                "ts":         time.strftime("%Y-%m-%dT%H:%M:%S"),
                "latency_ms": req.get("latency_ms", 0),
                "tokens_in":  req.get("tokens_in", 0),
                "tokens_out": req.get("tokens_out", 0),
                "intent":     req.get("intent", ""),
                "backend":    req.get("backend", ""),
                "error":      req.get("error", ""),
            })
            with open(str(self._log_path), "a") as f:
                f.write(record + "\n")
        except Exception:
            pass

    # ── Snapshot ──────────────────────────────────────────────────────
    def snapshot(self) -> dict:
        """Return current full metrics snapshot.

        Returns:
            dict covering hardware, request metrics, subsystem metrics,
            and latest training metrics.
        """
        with self._lock:
            n = max(self._total_requests, 1)
            ret_q = max(self._retrieval["queries"], 1)

            snap = {
                "ts":              time.strftime("%Y-%m-%dT%H:%M:%S"),
                # Request metrics
                "requests": {
                    "total":           self._total_requests,
                    "errors":          self._errors,
                    "error_rate":      round(self._errors / n, 4),
                    "avg_latency_ms":  round(self._total_latency_ms / n, 1),
                    "total_tokens_in": self._total_tokens_in,
                    "total_tokens_out":self._total_tokens_out,
                    "intent_counts":   dict(self._intent_counts),
                },
                # Hardware
                "hardware":        self._hw.get(),
                # Retrieval metrics
                "retrieval": {
                    "total_queries":   self._retrieval["queries"],
                    "hit_rate":        round(self._retrieval["hits"] / ret_q, 3),
                    "miss_rate":       round(self._retrieval["misses"] / ret_q, 3),
                    "cache_hit_rate":  round(self._retrieval["cache_hits"] / ret_q, 3),
                    "multi_hop_rate":  round(self._retrieval["multi_hops"] / ret_q, 3),
                    "avg_score":       round(
                        self._retrieval["total_score"] /
                        max(self._retrieval["hits"], 1), 3
                    ),
                },
                # Memory metrics
                "memory": {
                    "total_recalls": self._memory_acc["recalls"],
                    "total_stores":  self._memory_acc["stores"],
                },
                # KG metrics
                "knowledge_graph": {
                    "total_queries":   self._kg_acc["queries"],
                    "triples_added":   self._kg_acc["triples_added"],
                },
                # Cognition
                "cognition": {
                    "mode":     self._cognition["mode"],
                    "top_emotions": sorted(
                        self._cognition["emotions"].items(),
                        key=lambda x: -x[1]
                    )[:5],
                    "top_intents": sorted(
                        self._cognition["intents"].items(),
                        key=lambda x: -x[1]
                    )[:5],
                },
                # Tool metrics
                "tools": {
                    "total_calls":   self._tool_acc["calls"],
                    "success_rate":  round(
                        self._tool_acc["success"] /
                        max(self._tool_acc["calls"], 1), 3
                    ),
                },
                # Training (latest step from log)
                "training":     self._latest_training_step(),
            }
        return snap

    def _latest_training_step(self) -> dict:
        """Read the last line of train_log.jsonl for live training metrics."""
        if not self._train_log.exists():
            return {}
        try:
            with open(str(self._train_log), "rb") as f:
                f.seek(0, 2)
                end = f.tell()
                if end == 0:
                    return {}
                # Read last ~512 bytes to find last line
                f.seek(max(0, end - 512))
                chunk = f.read().decode(errors="replace")
            lines = [l for l in chunk.strip().split("\n") if l.strip()]
            if lines:
                return json.loads(lines[-1])
        except Exception:
            pass
        return {}

    def history(self, n: int = 100) -> list[dict]:
        """Return the last n request records."""
        with self._lock:
            return list(self._history)[-n:]

    def hardware(self) -> dict:
        """Return current hardware metrics only."""
        return self._hw.get()

    def record_cognition_mode(self, mode: str):
        with self._lock:
            self._cognition["mode"] = mode

    def record_kg_extract(self, n_triples: int):
        with self._lock:
            self._kg_acc["triples_added"] += n_triples
            self._kg_acc["extracts"] += 1

    def record_workflow(self, event: str):
        """event: 'started' | 'completed' | 'failed'"""
        with self._lock:
            if event == "started":
                self._workflow["active"] += 1
            elif event == "completed":
                self._workflow["active"]    = max(0, self._workflow["active"]-1)
                self._workflow["completed"] += 1
            elif event == "failed":
                self._workflow["active"]  = max(0, self._workflow["active"]-1)
                self._workflow["failed"] += 1

    def reset(self):
        """Reset all accumulated metrics (does not delete log files)."""
        with self._lock:
            self._total_requests   = 0
            self._total_tokens_in  = 0
            self._total_tokens_out = 0
            self._total_latency_ms = 0.0
            self._errors           = 0
            self._intent_counts    = {}
            self._history.clear()
            self._retrieval  = {"hits":0,"misses":0,"cache_hits":0,"multi_hops":0,
                                "total_score":0.0,"queries":0}
            self._memory_acc = {"recalls":0,"stores":0,"total_score":0.0}
            self._kg_acc     = {"queries":0,"extracts":0,"triples_added":0}

    def stop(self):
        self._hw.stop()
        self._save()
