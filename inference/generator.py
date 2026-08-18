"""
CognitiveOC v3 — Response Generator (Inference Layer)
======================================================

Wraps the 700M CognitiveDecoder for inference.

Two generation paths:
  1. local-transformer(kv-cache)  — trained checkpoint loaded; full model inference
  2. grounded-fallback            — no checkpoint; assembles answer from retrieval,
                                    memory, and KG without model generation

Context assembly (called before generation):
  Formats the context dict from engine.py into a structured decoder prompt:

    [SYSTEM] <system_prompt + cognition_addendum>
    [MEMORY] <ranked memory hits>
    [RETRIEVAL] <retrieved chunks with citations>
    [KG] <knowledge graph facts>
    [TOOL] <tool output if any>
    [REASONING] <reasoning plan>
    <user>: <message>
    <assistant>:

Token budget (from config.py INFERENCE):
  system_prompt_tokens  = 256
  memory_budget_tokens  = 512
  retrieval_budget_tokens = 2048
  kg_budget_tokens      = 256
  tool_budget_tokens    = 512
  history_budget_tokens = 512
  cognition_budget_tokens = 128
  generation_budget     = 1024
  Total budget          ≤ 8192 (block_size)

File: inference/generator.py
Used by: engine.py
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Iterator

try:
    from config import MODEL, INFERENCE, CHECKPOINT_DIR, VOCAB_SIZE
except ImportError:
    MODEL = {}; INFERENCE = {}; VOCAB_SIZE = 48_000
    CHECKPOINT_DIR = Path("var/checkpoints")

_SYSTEM_PROMPT = (
    "You are CognitiveOC — a local cognitive orchestration assistant. "
    "You answer questions using the evidence provided. "
    "You are precise, honest, and grounded in evidence. "
    "If evidence is insufficient, say so explicitly."
)


class ResponseGenerator:
    """Manages model loading and response generation for COC v3.

    Thread-safe: a single lock guards model loading. Generation itself
    is stateless (uses KV cache per call, not shared state).
    """

    def __init__(self):
        self._model       = None
        self._tokenizer   = None
        self._device      = None
        self._lock        = threading.Lock()
        self._load_error  = None
        self._loaded      = False

    # ── Lazy load ─────────────────────────────────────────────────────
    def _ensure_loaded(self):
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            try:
                import torch
                from models.transformer import CognitiveDecoder
                from tokenizer.tokenizer import CognitiveTokenizer

                device = "cuda" if torch.cuda.is_available() else "cpu"
                ckpt   = CHECKPOINT_DIR / "model_700m_best.pt"
                if not ckpt.exists():
                    ckpt = CHECKPOINT_DIR / "model_700m.pt"

                if ckpt.exists():
                    self._model = CognitiveDecoder.from_checkpoint(
                        str(ckpt), device=device
                    )
                    # Apply quantisation if configured
                    quant = MODEL.get("quant", "none")
                    if quant == "int8" and device == "cuda":
                        try:
                            import bitsandbytes as bnb  # type: ignore
                            self._model = bnb.nn.quantize_8bit(self._model)
                        except ImportError:
                            pass
                    self._model.eval()
                    self._device = device
                    print(f"[generator] Loaded model → {ckpt.name}  "
                          f"device={device}  "
                          f"params={self._model.num_params():,}")
                else:
                    print("[generator] No checkpoint found — using grounded fallback")

                # Load tokenizer
                tok_paths = [
                    CHECKPOINT_DIR.parent / "tokenizer" / "spm48k.model",
                    Path("tokenizer/48K/spm48k.model"),
                ]
                for p in tok_paths:
                    if p.exists():
                        self._tokenizer = CognitiveTokenizer.load(str(p))
                        break

                self._loaded     = True
                self._load_error = None

            except ImportError as e:
                self._load_error = str(e)
                self._loaded     = True   # don't retry on every call
            except Exception as e:
                self._load_error = str(e)
                self._loaded     = True

    @property
    def backend(self) -> str:
        self._ensure_loaded()
        if self._model is not None:
            return f"local-transformer(kv-cache,{self._device})"
        return "grounded-fallback"

    # ── Context assembly ──────────────────────────────────────────────
    def _assemble(self, message: str, context: dict) -> str:
        """Format the full decoder prompt from the context dict.

        Respects token budgets from INFERENCE config.
        Falls back to character estimation if tokenizer unavailable.
        """
        CHARS_PER_TOK = 4   # conservative estimate for budget without tokenizer

        def _budget_chars(key: str) -> int:
            return INFERENCE.get(key, 512) * CHARS_PER_TOK

        parts: list[str] = []

        # System prompt + cognition addendum
        sys_text = _SYSTEM_PROMPT
        addendum = context.get("cognition_addendum", "")
        if addendum:
            sys_text += "\n" + addendum
        parts.append(f"[SYSTEM] {sys_text[:_budget_chars('system_prompt_tokens')]}")

        # Goal context
        goals = context.get("goals_context", "")
        if goals:
            parts.append(f"[GOALS] {goals[:128]}")

        # Conversation history (most recent turns within budget)
        history = context.get("history", [])
        if history:
            hist_budget = _budget_chars("history_budget_tokens")
            hist_lines, hist_used = [], 0
            for turn in reversed(history[-10:]):
                role    = turn.get("role", "user")
                content = turn.get("content", "").strip()
                line    = f"{role.upper()}: {content}"
                if hist_used + len(line) > hist_budget:
                    break
                hist_lines.insert(0, line)
                hist_used += len(line)
            if hist_lines:
                parts.append("[HISTORY]\n" + "\n".join(hist_lines))

        # Memory hits
        memories = context.get("memory", [])
        if memories:
            mem_budget = _budget_chars("memory_budget_tokens")
            mem_lines, mem_used = [], 0
            for m in memories:
                text = m.get("text", "").strip()
                kind = m.get("kind", "fact")
                line = f"  [{kind}] {text}"
                if mem_used + len(line) > mem_budget:
                    break
                mem_lines.append(line)
                mem_used += len(line)
            if mem_lines:
                parts.append("[MEMORY]\n" + "\n".join(mem_lines))

        # Retrieval chunks
        chunks = context.get("chunks", [])
        if chunks:
            ret_budget = _budget_chars("retrieval_budget_tokens")
            ret_lines, ret_used = [], 0
            for c in chunks:
                text   = c.get("text", "").strip()
                source = c.get("source", "doc")
                score  = c.get("rerank_score", c.get("score", 0.0))
                line   = f"  [{source} score={score:.2f}] {text}"
                if ret_used + len(line) > ret_budget:
                    break
                ret_lines.append(line)
                ret_used += len(line)
            if ret_lines:
                parts.append("[RETRIEVAL]\n" + "\n".join(ret_lines))
            # Citations
            cites = context.get("citations", "")
            if cites:
                parts.append(f"[CITATIONS] {cites}")

        # KG facts
        kg = context.get("kg", [])
        if kg:
            kg_budget = _budget_chars("kg_budget_tokens")
            kg_lines, kg_used = [], 0
            for f in kg:
                line = f"  {f.get('subject','')} → {f.get('relation','')} → {f.get('object','')}"
                if kg_used + len(line) > kg_budget:
                    break
                kg_lines.append(line)
                kg_used += len(line)
            if kg_lines:
                parts.append("[KNOWLEDGE]\n" + "\n".join(kg_lines))

        # Tool output
        tool = context.get("tool")
        if tool:
            tool_budget = _budget_chars("tool_budget_tokens")
            result = str(tool.get("result", ""))[:tool_budget]
            parts.append(f"[TOOL:{tool.get('tool','tool')}] {result}")

        # Reasoning plan
        reasoning = context.get("reasoning", {})
        plan = reasoning.get("plan", "") if isinstance(reasoning, dict) else ""
        if plan:
            parts.append(f"[REASONING] {plan[:400]}")

        # User message
        parts.append(f"<user> {message}")
        parts.append("<assistant>")

        return "\n".join(parts)

    # ── Blocking generation ───────────────────────────────────────────
    def generate(self, message: str, context: dict) -> str:
        """Generate a response (blocking).

        Returns:
            Generated text string.
        """
        self._ensure_loaded()

        if self._model is None or self._tokenizer is None:
            return self._grounded_fallback(context)

        prompt = self._assemble(message, context)
        try:
            import torch
            ids = self._tokenizer.encode(prompt, add_bos=True)
            max_len = INFERENCE.get("max_new_tokens", 1024)
            max_ctx = MODEL.get("block_size", 8192) - max_len
            ids     = ids[-max_ctx:]   # truncate prompt if needed

            input_ids = torch.tensor([ids], device=self._device)
            with torch.inference_mode():
                output = self._model.generate(
                    input_ids,
                    max_new_tokens     = max_len,
                    temperature        = INFERENCE.get("temperature", 0.7),
                    top_k              = INFERENCE.get("top_k", 50),
                    top_p              = INFERENCE.get("top_p", 0.9),
                    repetition_penalty = INFERENCE.get("repetition_penalty", 1.1),
                    eos_id             = self._tokenizer.token_to_id("<eos>"),
                )
            # Decode only newly generated tokens
            new_ids = output[0, len(ids):].tolist()
            text    = self._tokenizer.decode(new_ids, skip_special=True).strip()
            return text if text else self._grounded_fallback(context)

        except Exception as e:
            return self._grounded_fallback(context, error=str(e))

    # ── Streaming generation ──────────────────────────────────────────
    def generate_stream(self,
                        message: str,
                        context: dict) -> Iterator[str]:
        """Streaming token-by-token generation.

        Yields string fragments. Final chunk ends with empty string ''.
        """
        self._ensure_loaded()

        if self._model is None or self._tokenizer is None:
            yield self._grounded_fallback(context)
            yield ""
            return

        prompt = self._assemble(message, context)
        try:
            import torch
            ids     = self._tokenizer.encode(prompt, add_bos=True)
            max_len = INFERENCE.get("max_new_tokens", 1024)
            max_ctx = MODEL.get("block_size", 8192) - max_len
            ids     = ids[-max_ctx:]
            eos_id  = self._tokenizer.token_to_id("<eos>")
            temp    = INFERENCE.get("temperature", 0.7)
            top_k   = INFERENCE.get("top_k", 50)
            top_p   = INFERENCE.get("top_p", 0.9)
            rep_pen = INFERENCE.get("repetition_penalty", 1.1)

            cur_ids  = torch.tensor([ids], device=self._device)
            past     = None
            seen     = set(ids)
            import torch.nn.functional as F

            self._model.eval()
            with torch.inference_mode():
                for _ in range(max_len):
                    logits, _, past = self._model(cur_ids, past=past)
                    logits = logits[:, -1, :].float()

                    # Repetition penalty
                    if rep_pen != 1.0:
                        for tok in seen:
                            logits[:, tok] /= rep_pen

                    logits = logits / max(temp, 1e-8)

                    if top_k:
                        kval = torch.topk(logits, min(top_k, logits.size(-1)))[0][:,-1:]
                        logits[logits < kval] = float("-inf")

                    if top_p < 1.0:
                        sorted_l, sorted_i = torch.sort(logits, descending=True)
                        cum = torch.cumsum(F.softmax(sorted_l, dim=-1), dim=-1)
                        sorted_l[cum - F.softmax(sorted_l, dim=-1) > top_p] = float("-inf")
                        logits.scatter_(1, sorted_i, sorted_l)

                    probs  = F.softmax(logits, dim=-1)
                    nxt    = torch.multinomial(probs, 1)
                    tok_id = nxt.item()
                    seen.add(tok_id)

                    if tok_id == eos_id:
                        break

                    fragment = self._tokenizer.decode([tok_id], skip_special=True)
                    if fragment:
                        yield fragment

                    cur_ids = nxt

        except Exception as e:
            yield self._grounded_fallback(context, error=str(e))

        yield ""   # terminal sentinel

    # ── Grounded fallback ─────────────────────────────────────────────
    def _grounded_fallback(self,
                           context: dict,
                           error:   str = None) -> str:
        """Assemble an answer from retrieval/memory/KG without model generation.

        Used when:
          - No trained checkpoint exists
          - Model load failed
          - Torch not installed
        """
        parts: list[str] = []

        # Tool output (highest priority)
        tool = context.get("tool")
        if tool and tool.get("result"):
            parts.append(f"**Tool result ({tool.get('tool','')}):**\n{tool['result']}")

        # Retrieval chunks
        chunks = context.get("chunks", [])
        if chunks:
            parts.append("**Retrieved evidence:**")
            for c in chunks[:3]:
                src   = c.get("source", "document")
                text  = c.get("text", "").strip()[:300]
                parts.append(f"[{src}] {text}")

        # Memory hits
        memories = context.get("memory", [])
        if memories:
            parts.append("**From memory:**")
            for m in memories[:2]:
                parts.append(f"- {m.get('text','').strip()[:200]}")

        # KG facts
        kg = context.get("kg", [])
        if kg:
            parts.append("**Known facts:**")
            for f in kg[:3]:
                parts.append(
                    f"- {f.get('subject','')} {f.get('relation','')} "
                    f"{f.get('object','')}"
                )

        if not parts:
            return (
                "I don't have enough information to answer this confidently. "
                "Try ingesting relevant documents or providing more context."
            )

        result = "\n".join(parts)
        if error:
            result += f"\n\n*(Note: model generation unavailable — {error})*"
        return result
