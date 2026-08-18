"""
CognitiveOC v3 — Perception Layer
===================================

First-class input normalisation subsystem. Every request enters the
system through Perceiver.process() before reaching the encoder stack.

Responsibilities:
  1. Identify input type:  text | voice | document | image | workspace | task
  2. Extract raw content from non-text inputs (OCR, document parsing, ASR)
  3. Detect language (heuristic; extend with langdetect if installed)
  4. Normalise encoding and whitespace
  5. Extract safe metadata (filename, mime, page count, word count)
  6. Chunk long inputs for encoder routing
  7. Return a PerceptionResult with normalised text + metadata + trace

Architecture position (top-down workflow):
  Raw User Input
    → Perceiver.process()    ← THIS FILE
    → Encoder Intelligence Stack
    → Human Cognition Layer
    → Orchestration Core (engine.py)

Supported input types:
  text      — plain string from chat UI
  voice     — audio bytes (ASR stub; extend with whisper.cpp)
  document  — file path (.pdf .docx .txt .md .csv .xlsx)
  image     — file path (.png .jpg .jpeg)
  workspace — workspace name (already-indexed multi-doc session)
  task      — structured task dict

File: perception/perception.py
Used by: engine.py (build_context → _perception())
         ui/app.py (upload endpoint)
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ═══════════════════════════════════════════════════════════════════
# Result dataclass
# ═══════════════════════════════════════════════════════════════════

@dataclass
class PerceptionResult:
    """Normalised output from the Perception Layer.

    All downstream subsystems (encoder, cognition, memory, retrieval)
    operate on the `text` field plus `metadata`.
    """
    text:        str                       # Normalised UTF-8 text
    input_type:  str                       # text|voice|document|image|workspace|task
    language:    str           = "en"      # ISO 639-1 code (best-effort)
    chunks:      list[str]     = field(default_factory=list)  # chunked segments
    metadata:    dict          = field(default_factory=dict)   # safe metadata
    trace:       list[str]     = field(default_factory=list)   # processing steps
    ok:          bool          = True
    error:       str           = ""
    latency_ms:  float         = 0.0


# ═══════════════════════════════════════════════════════════════════
# Language detection (heuristic + optional langdetect)
# ═══════════════════════════════════════════════════════════════════

_LANG_SIGNALS = {
    "en": re.compile(r"\b(the|is|are|was|were|and|or|in|of|to|a|an)\b", re.I),
    "es": re.compile(r"\b(el|la|los|las|es|son|y|en|de|que|por)\b", re.I),
    "fr": re.compile(r"\b(le|la|les|est|sont|et|en|de|que|pour|je)\b", re.I),
    "de": re.compile(r"\b(der|die|das|ist|sind|und|in|von|für|ich)\b", re.I),
}


def detect_language(text: str) -> str:
    """Best-effort language detection. Uses langdetect if installed."""
    try:
        from langdetect import detect  # type: ignore
        return detect(text[:500]) or "en"
    except ImportError:
        pass
    # Heuristic fallback
    scores = {lang: len(pat.findall(text[:500])) for lang, pat in _LANG_SIGNALS.items()}
    return max(scores, key=lambda k: scores[k]) if any(scores.values()) else "en"


# ═══════════════════════════════════════════════════════════════════
# Input type detection
# ═══════════════════════════════════════════════════════════════════

_FILE_EXTS_DOC   = {'.pdf', '.docx', '.txt', '.md', '.csv', '.xlsx', '.xls',
                     '.json', '.yaml', '.toml', '.log', '.py'}
_FILE_EXTS_IMAGE = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
_FILE_EXTS_AUDIO = {'.wav', '.mp3', '.m4a', '.ogg', '.flac'}


def detect_input_type(raw_input: Any) -> str:
    """Classify input into one of: text|voice|document|image|workspace|task."""
    if isinstance(raw_input, bytes):
        return "voice"
    if isinstance(raw_input, dict):
        return raw_input.get("type", "task")
    if isinstance(raw_input, str):
        p = Path(raw_input)
        if p.exists():
            ext = p.suffix.lower()
            if ext in _FILE_EXTS_DOC:
                return "document"
            if ext in _FILE_EXTS_IMAGE:
                return "image"
            if ext in _FILE_EXTS_AUDIO:
                return "voice"
        return "text"
    return "text"


# ═══════════════════════════════════════════════════════════════════
# Text normalisation
# ═══════════════════════════════════════════════════════════════════

def normalise_text(text: str) -> str:
    """Normalise encoding and whitespace."""
    import unicodedata
    text = unicodedata.normalize("NFC", text)
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'\n{4,}', '\n\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    # Strip non-printable except newline/tab
    text = ''.join(c for c in text if c.isprintable() or c in '\n\t')
    return text.strip()


# ═══════════════════════════════════════════════════════════════════
# Chunker for long inputs
# ═══════════════════════════════════════════════════════════════════

def chunk_for_encoding(text: str, max_chars: int = 512) -> list[str]:
    """Split text into encoder-sized chunks at sentence boundaries."""
    if len(text) <= max_chars:
        return [text] if text.strip() else []
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, cur = [], ""
    for sent in sentences:
        if len(cur) + len(sent) + 1 <= max_chars:
            cur = (cur + " " + sent).strip()
        else:
            if cur:
                chunks.append(cur)
            cur = sent
    if cur:
        chunks.append(cur)
    return [c for c in chunks if c.strip()]


# ═══════════════════════════════════════════════════════════════════
# Perceiver — Main Perception Pipeline
# ═══════════════════════════════════════════════════════════════════

class Perceiver:
    """First-class input perception pipeline for COC v3.

    Handles every input type uniformly so all downstream subsystems
    receive normalised text + structured metadata.

    Usage:
        perceiver = Perceiver()
        result = perceiver.process("What is attention?")
        result = perceiver.process("/path/to/paper.pdf")
        result = perceiver.process(audio_bytes)
    """

    def process(self,
                raw_input: Any,
                session:   str = "default",
                context:   dict = None) -> PerceptionResult:
        """Process any input type and return a PerceptionResult.

        Args:
            raw_input: str (text or file path) | bytes (audio) | dict (task)
            session:   session identifier for metadata
            context:   optional context dict (workspace name, etc.)

        Returns:
            PerceptionResult with normalised text, metadata, and trace.
        """
        t0    = time.time()
        trace = []
        context = context or {}

        # ── Detect input type ─────────────────────────────────────────
        input_type = detect_input_type(raw_input)
        trace.append(f"detected_type={input_type}")

        # ── Dispatch to handler ───────────────────────────────────────
        if input_type == "text":
            return self._handle_text(str(raw_input), trace, t0)

        if input_type == "document":
            return self._handle_document(str(raw_input), trace, t0)

        if input_type == "image":
            return self._handle_image(str(raw_input), trace, t0)

        if input_type == "voice":
            return self._handle_voice(raw_input, trace, t0)

        if input_type == "task":
            return self._handle_task(raw_input, trace, t0)

        # Fallback: treat as text
        return self._handle_text(str(raw_input), trace, t0)

    # ── Text handler ──────────────────────────────────────────────────
    def _handle_text(self, text: str,
                     trace: list, t0: float) -> PerceptionResult:
        normalised = normalise_text(text)
        language   = detect_language(normalised)
        chunks     = chunk_for_encoding(normalised)
        trace.extend([f"lang={language}", f"chunks={len(chunks)}",
                       f"chars={len(normalised)}"])
        return PerceptionResult(
            text       = normalised,
            input_type = "text",
            language   = language,
            chunks     = chunks,
            metadata   = {"char_count": len(normalised),
                          "word_count": len(normalised.split())},
            trace      = trace,
            latency_ms = round((time.time() - t0) * 1000, 1),
        )

    # ── Document handler ──────────────────────────────────────────────
    def _handle_document(self, path: str,
                         trace: list, t0: float) -> PerceptionResult:
        try:
            from vision.documents import parse_file, _clean
        except ImportError:
            return PerceptionResult(
                text="", input_type="document", ok=False,
                error="vision.documents not available", trace=trace,
                latency_ms=round((time.time()-t0)*1000, 1),
            )

        p        = Path(path)
        sections = parse_file(path)
        errors   = [t for loc, t in sections if loc == 'error']
        content  = [_clean(t) for loc, t in sections
                    if loc != 'error' and t.strip()]

        if errors and not content:
            return PerceptionResult(
                text="", input_type="document", ok=False,
                error="; ".join(errors), trace=trace,
                latency_ms=round((time.time()-t0)*1000, 1),
            )

        text       = '\n\n'.join(content)
        normalised = normalise_text(text)
        language   = detect_language(normalised[:1000])
        chunks     = chunk_for_encoding(normalised)
        meta       = {
            "filename":   p.name,
            "extension":  p.suffix.lower(),
            "size_bytes": p.stat().st_size if p.exists() else 0,
            "sections":   len(sections),
            "pages":      sum(1 for loc, _ in sections if loc.startswith('page')),
            "char_count": len(normalised),
            "word_count": len(normalised.split()),
        }
        trace.extend([f"lang={language}", f"sections={len(sections)}",
                       f"chunks={len(chunks)}", f"chars={len(normalised)}"])
        return PerceptionResult(
            text       = normalised,
            input_type = "document",
            language   = language,
            chunks     = chunks,
            metadata   = meta,
            trace      = trace,
            latency_ms = round((time.time() - t0) * 1000, 1),
        )

    # ── Image handler ─────────────────────────────────────────────────
    def _handle_image(self, path: str,
                      trace: list, t0: float) -> PerceptionResult:
        try:
            from vision.ocr import analyze_image
            results = analyze_image(path)
            text    = '\n'.join(t for loc, t in results
                                if loc not in ('error', 'image_meta'))
            meta    = {loc: t for loc, t in results if loc == 'image_meta'}
            errors  = [t for loc, t in results if loc == 'error']
            normalised = normalise_text(text)
            language   = detect_language(normalised[:500]) if normalised else "en"
            chunks     = chunk_for_encoding(normalised)
            trace.extend([f"ocr_chars={len(normalised)}", f"lang={language}"])
            return PerceptionResult(
                text       = normalised,
                input_type = "image",
                language   = language,
                chunks     = chunks,
                metadata   = {"filename": Path(path).name, **meta},
                trace      = trace,
                ok         = not (errors and not normalised),
                error      = "; ".join(errors) if errors and not normalised else "",
                latency_ms = round((time.time() - t0) * 1000, 1),
            )
        except Exception as e:
            return PerceptionResult(
                text="", input_type="image", ok=False,
                error=str(e), trace=trace,
                latency_ms=round((time.time()-t0)*1000, 1),
            )

    # ── Voice handler (ASR stub) ──────────────────────────────────────
    def _handle_voice(self, audio: bytes,
                      trace: list, t0: float) -> PerceptionResult:
        """ASR stub. Extend with whisper.cpp or faster-whisper when available."""
        # Try whisper if installed
        try:
            import whisper  # type: ignore
            import tempfile, os
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                f.write(audio if isinstance(audio, bytes) else b'')
                tmp = f.name
            model  = whisper.load_model("base")
            result = model.transcribe(tmp)
            os.unlink(tmp)
            text  = result.get("text", "").strip()
            trace.append(f"asr=whisper  chars={len(text)}")
            return PerceptionResult(
                text=normalise_text(text), input_type="voice",
                language=result.get("language","en"),
                chunks=chunk_for_encoding(text),
                metadata={"asr_backend": "whisper"},
                trace=trace,
                latency_ms=round((time.time()-t0)*1000, 1),
            )
        except ImportError:
            pass

        trace.append("asr=unavailable")
        return PerceptionResult(
            text       = "",
            input_type = "voice",
            ok         = False,
            error      = "ASR not available — install openai-whisper for voice support",
            trace      = trace,
            latency_ms = round((time.time() - t0) * 1000, 1),
        )

    # ── Task handler ──────────────────────────────────────────────────
    def _handle_task(self, task: dict,
                     trace: list, t0: float) -> PerceptionResult:
        """Handle structured task dicts from workflow engine."""
        description = task.get("description", task.get("goal", str(task)))
        normalised  = normalise_text(description)
        trace.append(f"task_type={task.get('type','unknown')}")
        return PerceptionResult(
            text       = normalised,
            input_type = "task",
            language   = "en",
            chunks     = chunk_for_encoding(normalised),
            metadata   = {k: str(v) for k, v in task.items()
                          if k in ('type', 'id', 'goal', 'priority')},
            trace      = trace,
            latency_ms = round((time.time() - t0) * 1000, 1),
        )

    # ── Metadata extraction (safe) ────────────────────────────────────
    def extract_metadata(self, path: str) -> dict:
        """Extract safe file metadata without parsing content."""
        p = Path(path)
        if not p.exists():
            return {"error": "file not found"}
        return {
            "filename":   p.name,
            "extension":  p.suffix.lower(),
            "size_bytes": p.stat().st_size,
            "size_mb":    round(p.stat().st_size / 1e6, 3),
            "modified":   p.stat().st_mtime,
        }
