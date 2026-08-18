"""
CognitiveOC v3 — Model Training Script
========================================

Trains the 700M CognitiveDecoder from scratch or resumes from checkpoint.
Designed for RTX 5060 8GB with bf16 precision and gradient checkpointing.

Live TUI display (updates every display_interval_s seconds):
  ┌─────────────────────────────────────────────────────────────┐
  │ COC v3 Training  step=1200/100000  epoch≈0.01              │
  ├─────────────────────────────────────────────────────────────┤
  │ train_loss  │ val_loss  │ ppl    │ grad_norm │ tokens/sec  │
  │    2.4312   │  2.5801   │ 13.22  │  0.823    │  1847       │
  ├─────────────────────────────────────────────────────────────┤
  │ lr=0.000285  elapsed=00:12:34  eta=17:22:11  best_ppl=12.8 │
  └─────────────────────────────────────────────────────────────┘

Metrics tracked and logged to var/checkpoints/train_log.jsonl:
  step, train_loss, val_loss, perplexity, grad_norm, lr,
  tokens_per_sec, elapsed_s, gpu_mem_gb, best_val_loss

File: train/train_model.py
Run:  python main.py train-model data/corpus/v1/split/train.txt
  or: python train/train_model.py --corpus data/corpus/v1/split/train.txt
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

# Add project root to path so all imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import torch
    import torch.nn as nn
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    _TORCH = True
except ImportError:
    _TORCH = False

try:
    from config import (MODEL, TRAIN, VOCAB_SIZE, CHECKPOINT_DIR,
                        TOKENIZER_DIR, ensure_dirs)
except ImportError:
    MODEL = {}; TRAIN = {}; VOCAB_SIZE = 48_000
    CHECKPOINT_DIR = Path("var/checkpoints")
    TOKENIZER_DIR  = Path("var/tokenizer")
    def ensure_dirs(): pass


# ═══════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════

def load_tokens(corpus_path: str, tokenizer) -> list[int]:
    """Load and tokenise the training corpus.

    Reads paragraphs separated by double newlines, encodes each,
    and concatenates into one flat token list.
    Returns list of int token IDs.
    """
    text = Path(corpus_path).read_text(encoding="utf-8", errors="replace")
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    tokens: list[int] = []
    for p in paragraphs:
        ids = tokenizer.encode(p, add_bos=True, add_eos=True)
        tokens.extend(ids)
    return tokens


def make_batches(tokens: list[int],
                 block_size: int,
                 batch_size: int,
                 device,
                 seed: int = 42):
    """Yield (x, y) tensor batches for next-token prediction.

    x: (batch_size, block_size)  — input tokens
    y: (batch_size, block_size)  — target tokens (x shifted left by 1)
    """
    import torch, random
    rng = random.Random(seed)
    n   = len(tokens)
    min_len = block_size + 1
    if n < min_len:
        raise ValueError(
            f"Corpus too small after tokenisation: {n} tokens. "
            f"Need at least {min_len}. Add more training data."
        )

    import numpy as np
    arr = np.array(tokens, dtype=np.int32)
    while True:
        starts = [rng.randint(0, n - block_size - 1) for _ in range(batch_size)]
        x = torch.stack([
            torch.from_numpy(arr[s: s + block_size].astype(np.int64))
            for s in starts
        ]).to(device)
        y = torch.stack([
            torch.from_numpy(arr[s + 1: s + block_size + 1].astype(np.int64))
            for s in starts
        ]).to(device)
        yield x, y


# ═══════════════════════════════════════════════════════════════════
# Live TUI Display
# ═══════════════════════════════════════════════════════════════════

class LiveDisplay:
    """Simple ANSI terminal display for live training metrics.

    Updates in-place using ANSI escape codes.
    Safe to use without curses — just prints/overwrites lines.
    """

    _W = 65   # display width

    def __init__(self, total_steps: int, display_interval_s: float = 5.0):
        self._total   = total_steps
        self._interval= display_interval_s
        self._last_t  = 0.0
        self._lines   = 0
        self._start   = time.time()
        self._best_ppl= float("inf")
        self._enabled = sys.stdout.isatty()

    def _clear(self):
        if self._enabled and self._lines:
            sys.stdout.write(f"\033[{self._lines}A\033[J")

    def _fmt_time(self, seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def update(self,
               step:         int,
               train_loss:   float,
               val_loss:     float = None,
               grad_norm:    float = 0.0,
               lr:           float = 0.0,
               tokens_per_sec: float = 0.0,
               gpu_mem_gb:   float = 0.0,
               force:        bool  = False):
        """Refresh the display if interval has elapsed."""
        now = time.time()
        if not force and now - self._last_t < self._interval:
            return
        self._last_t = now

        ppl = math.exp(min(val_loss or train_loss, 20.0))
        if val_loss and ppl < self._best_ppl:
            self._best_ppl = ppl

        elapsed = now - self._start
        eta     = (elapsed / max(step, 1)) * (self._total - step)
        pct     = 100 * step / max(self._total, 1)

        # Progress bar
        bar_w  = self._W - 12
        filled = int(bar_w * step / max(self._total, 1))
        bar    = "█" * filled + "░" * (bar_w - filled)

        lines = [
            f"┌{'─' * self._W}┐",
            f"│ COC v3 Training  step={step:,}/{self._total:,}  ({pct:.1f}%)"
            .ljust(self._W + 1) + "│",
            f"│ [{bar}] ".ljust(self._W + 1) + "│",
            f"├{'─' * self._W}┤",
            f"│  {'train_loss':10s}  {'val_loss':10s}  {'perplexity':10s}  "
            f"{'grad_norm':10s}  {'tok/s':6s}".ljust(self._W + 1) + "│",
            f"│  {train_loss:<10.4f}  "
            f"{(val_loss or 0.0):<10.4f}  "
            f"{ppl:<10.2f}  "
            f"{grad_norm:<10.3f}  "
            f"{tokens_per_sec:<6.0f}".ljust(self._W + 1) + "│",
            f"├{'─' * self._W}┤",
            f"│  lr={lr:.2e}  elapsed={self._fmt_time(elapsed)}  "
            f"eta={self._fmt_time(eta)}  "
            f"best_ppl={self._best_ppl:.2f}  "
            f"gpu={gpu_mem_gb:.1f}GB".ljust(self._W + 1) + "│",
            f"└{'─' * self._W}┘",
        ]

        self._clear()
        output = "\n".join(lines) + "\n"
        sys.stdout.write(output)
        sys.stdout.flush()
        self._lines = len(lines)


# ═══════════════════════════════════════════════════════════════════
# Evaluation (perplexity on validation set)
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model, val_tokens: list[int],
             block_size: int, batch_size: int,
             device, n_batches: int = 20) -> float:
    """Compute validation perplexity over n_batches random batches.

    Returns:
        float — validation loss (cross-entropy, not perplexity).
                perplexity = exp(val_loss).
    """
    model.eval()
    import random, numpy as np
    rng    = random.Random(99)
    n      = len(val_tokens)
    arr    = np.array(val_tokens, dtype=np.int32)
    losses = []
    for _ in range(n_batches):
        starts = [rng.randint(0, n - block_size - 1) for _ in range(batch_size)]
        x = torch.stack([
            torch.from_numpy(arr[s: s + block_size].astype(np.int64))
            for s in starts
        ]).to(device)
        y = torch.stack([
            torch.from_numpy(arr[s + 1: s + block_size + 1].astype(np.int64))
            for s in starts
        ]).to(device)
        _, loss, _ = model(x, targets=y)
        if loss is not None and torch.isfinite(loss):
            losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses) if losses else float("inf")


# ═══════════════════════════════════════════════════════════════════
# Main Training Function
# ═══════════════════════════════════════════════════════════════════

def train(corpus_path: str,
          steps:       int  = None,
          batch:       int  = None,
          accum:       int  = None,
          lr:          float = None,
          precision:   str  = None,
          seed:        int  = None,
          resume:      bool = True,
          verbose:     bool = True):
    """Train the COC v3 700M decoder.

    Args:
        corpus_path: Path to train.txt (double-newline separated paragraphs).
        steps:       Total gradient steps (default from config).
        batch:       Micro-batch size (default from config).
        accum:       Gradient accumulation steps (default from config).
        lr:          Peak learning rate (default from config).
        precision:   'bf16' | 'fp32' (default from config).
        seed:        Random seed (default from config).
        resume:      Auto-resume from checkpoint if it exists.
        verbose:     Print progress (True) or silent (False).

    Returns:
        dict with final metrics.
    """
    if not _TORCH:
        raise RuntimeError(
            "PyTorch is not installed.\n"
            "Install: pip install torch --index-url "
            "https://download.pytorch.org/whl/cu124"
        )

    ensure_dirs()

    # ── Config ────────────────────────────────────────────────────────
    steps    = steps    or TRAIN.get("steps",       100_000)
    batch    = batch    or TRAIN.get("batch",        2)
    accum    = accum    or TRAIN.get("accum_steps",  16)
    lr_peak  = lr       or TRAIN.get("lr",           3e-4)
    lr_min   = TRAIN.get("lr_min",          3e-5)
    precision= precision or TRAIN.get("precision",   "bf16")
    seed_val = seed     or TRAIN.get("seed",         1337)
    val_every= TRAIN.get("val_every",      500)
    save_every=TRAIN.get("save_every",     1000)
    log_every = TRAIN.get("log_every",     50)
    disp_int  = TRAIN.get("display_interval_s", 5)
    block     = MODEL.get("block_size",    8192)
    wd        = TRAIN.get("weight_decay",  0.1)
    clip      = TRAIN.get("grad_clip",     1.0)
    warmup    = TRAIN.get("warmup_steps",  2000)

    torch.manual_seed(seed_val)

    # ── Device & dtype ────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.bfloat16 if precision == "bf16" else torch.float32
    use_amp= (device.type == "cuda" and precision == "bf16")
    if verbose:
        print(f"[train] device={device}  dtype={dtype}  "
              f"batch={batch}  accum={accum}  eff_batch={batch*accum}")

    # ── Tokenizer ─────────────────────────────────────────────────────
    from tokenizer.tokenizer import CognitiveTokenizer
    tok_candidates = [
        TOKENIZER_DIR / "spm48k.model",
        Path("tokenizer/48K/spm48k.model"),
    ]
    tokenizer = None
    for p in tok_candidates:
        if p.exists():
            tokenizer = CognitiveTokenizer.load(str(p))
            break
    if tokenizer is None:
        raise FileNotFoundError(
            "Tokenizer not found. Run: python main.py train-tokenizer <corpus>"
        )

    # ── Load tokens ───────────────────────────────────────────────────
    if verbose:
        print(f"[train] Loading corpus: {corpus_path}")
    all_tokens  = load_tokens(corpus_path, tokenizer)
    n_val       = max(int(len(all_tokens) * TRAIN.get("val_frac", 0.05)), 512)
    val_tokens  = all_tokens[-n_val:]
    train_tokens= all_tokens[:-n_val]
    if verbose:
        print(f"[train] Train tokens: {len(train_tokens):,}  "
              f"Val tokens: {len(val_tokens):,}")

    # ── Model ─────────────────────────────────────────────────────────
    from models.transformer import CognitiveDecoder
    model = CognitiveDecoder(MODEL).to(device)
    if use_amp:
        model = model.to(dtype)
    if verbose:
        print(f"[train] Model: {model.num_params():,} parameters")

    # ── Optimiser ─────────────────────────────────────────────────────
    param_groups = model.param_groups(wd)
    optimiser    = AdamW(param_groups, lr=lr_peak, betas=(0.9, 0.95), eps=1e-8)
    scheduler    = CosineAnnealingLR(optimiser, T_max=steps, eta_min=lr_min)
    scaler       = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ── Checkpoint resume ─────────────────────────────────────────────
    ckpt_path  = CHECKPOINT_DIR / "model_700m.pt"
    best_path  = CHECKPOINT_DIR / "model_700m_best.pt"
    log_path   = CHECKPOINT_DIR / "train_log.jsonl"
    start_step = 0

    if resume and ckpt_path.exists():
        try:
            ckpt       = torch.load(str(ckpt_path), map_location=device)
            model.load_state_dict(ckpt["model"], strict=False)
            if "optimizer" in ckpt:
                optimiser.load_state_dict(ckpt["optimizer"])
            start_step = ckpt.get("step", 0)
            if verbose:
                print(f"[train] Resumed from step {start_step:,}")
        except Exception as e:
            if verbose:
                print(f"[train] Resume failed ({e}), starting fresh")
            start_step = 0

    # ── Batch generator ───────────────────────────────────────────────
    batch_gen = make_batches(train_tokens, block, batch, device, seed_val)

    # ── Live display ──────────────────────────────────────────────────
    display   = LiveDisplay(steps, disp_int) if TRAIN.get("live_display", True) else None

    # ── Training state ────────────────────────────────────────────────
    best_val_loss = float("inf")
    train_loss    = 0.0
    val_loss      = 0.0
    grad_norm_val = 0.0
    tokens_total  = 0
    t_step_start  = time.time()
    model.train()

    if verbose:
        print(f"[train] Starting training: steps={steps:,}  "
              f"resume_from={start_step:,}")
        if display:
            print()

    # ── Training loop ─────────────────────────────────────────────────
    for step in range(start_step + 1, steps + 1):
        t0 = time.time()

        # Gradient accumulation
        accum_loss = 0.0
        optimiser.zero_grad(set_to_none=True)

        for micro in range(accum):
            x, y = next(batch_gen)
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=dtype):
                _, loss, _ = model(x, targets=y)
            if loss is None or not torch.isfinite(loss):
                continue
            loss_scaled = loss / accum
            scaler.scale(loss_scaled).backward()
            accum_loss += loss.item()

        # Gradient clip + optimiser step
        scaler.unscale_(optimiser)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        grad_norm_val = float(gn)
        scaler.step(optimiser)
        scaler.update()
        scheduler.step()

        train_loss  = accum_loss / accum
        tokens_total += batch * accum * block
        tok_per_s    = batch * accum * block / max(time.time() - t0, 1e-6)

        # GPU memory
        gpu_mem = (torch.cuda.memory_allocated(device) / 1e9
                   if device.type == "cuda" else 0.0)

        # ── Validation ────────────────────────────────────────────────
        if step % val_every == 0:
            val_loss = evaluate(model, val_tokens, block, batch, device)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model.save_checkpoint(
                    str(best_path), step=step,
                    opt_state=optimiser.state_dict(),
                    extra={"val_loss": val_loss,
                           "perplexity": math.exp(min(val_loss, 20.0))},
                )
                from safety.guardrails import _integrity
                _integrity.save_checkpoint_hash(str(best_path))

        # ── Log ───────────────────────────────────────────────────────
        if step % log_every == 0:
            record = {
                "step":          step,
                "train_loss":    round(train_loss, 6),
                "val_loss":      round(val_loss, 6) if val_loss else None,
                "perplexity":    round(math.exp(min(val_loss or train_loss, 20.0)), 4),
                "grad_norm":     round(grad_norm_val, 4),
                "lr":            round(scheduler.get_last_lr()[0], 8),
                "tokens_per_sec":round(tok_per_s, 1),
                "gpu_mem_gb":    round(gpu_mem, 3),
                "elapsed_s":     round(time.time() - t_step_start, 1),
                "ts":            time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            with open(str(log_path), "a") as f:
                f.write(json.dumps(record) + "\n")

        # ── Checkpoint ────────────────────────────────────────────────
        if step % save_every == 0:
            model.save_checkpoint(
                str(ckpt_path), step=step,
                opt_state=optimiser.state_dict(),
                extra={"train_loss": train_loss},
            )

        # ── Live display ──────────────────────────────────────────────
        if display:
            display.update(
                step         = step,
                train_loss   = train_loss,
                val_loss     = val_loss if val_loss else None,
                grad_norm    = grad_norm_val,
                lr           = scheduler.get_last_lr()[0],
                tokens_per_sec = tok_per_s,
                gpu_mem_gb   = gpu_mem,
            )

    # ── Final save ───────────────────────────────────────────────────
    model.save_checkpoint(
        str(ckpt_path), step=steps,
        opt_state=optimiser.state_dict(),
        extra={"train_loss": train_loss, "val_loss": val_loss},
    )
    if display:
        display.update(steps, train_loss, val_loss,
                       grad_norm_val, 0.0, 0.0, 0.0, force=True)

    final = {
        "steps":          steps,
        "train_loss":     round(train_loss, 6),
        "val_loss":       round(val_loss, 6),
        "perplexity":     round(math.exp(min(val_loss or train_loss, 20.0)), 4),
        "best_val_loss":  round(best_val_loss, 6),
        "best_perplexity":round(math.exp(min(best_val_loss, 20.0)), 4),
        "checkpoint":     str(ckpt_path),
        "best_checkpoint":str(best_path),
        "log":            str(log_path),
    }
    if verbose:
        print(f"\n[train] Complete. perplexity={final['perplexity']:.2f}  "
              f"best_ppl={final['best_perplexity']:.2f}")
    return final


# ═══════════════════════════════════════════════════════════════════
# Tokenizer Training Entry Point
# ═══════════════════════════════════════════════════════════════════

def train_tokenizer(corpus_path: str,
                    model_prefix: str = None,
                    vocab_size:   int = None) -> dict:
    """Train the 48K SentencePiece Unigram tokenizer.

    Args:
        corpus_path:  Path to train.txt.
        model_prefix: Output prefix (default from config).
        vocab_size:   Vocabulary size (default 48000).

    Returns:
        dict with validation results.
    """
    from tokenizer.tokenizer import CognitiveTokenizer
    tok = CognitiveTokenizer.train(
        corpus_path  = corpus_path,
        model_prefix = model_prefix,
        vocab_size   = vocab_size,
    )
    # Copy to canonical runtime path
    import shutil
    src  = Path(str(model_prefix or "tokenizer/48K/spm48k") + ".model")
    dest = TOKENIZER_DIR / "spm48k.model"
    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
    if src.exists():
        shutil.copy2(str(src), str(dest))

    # Validate
    results = tok.validate()
    print(f"[tokenizer] Validation: {'PASS' if results['passed'] else 'FAIL'}")
    for c in results["checks"]:
        mark = "PASS" if c["passed"] else "FAIL"
        print(f"  [{mark}] {c['name']}: {c.get('detail','')}")
    return results


# ═══════════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="COC v3 Training")
    parser.add_argument("--corpus", required=True,
                        help="Path to training corpus (train.txt)")
    parser.add_argument("--steps",  type=int,   default=None)
    parser.add_argument("--batch",  type=int,   default=None)
    parser.add_argument("--accum",  type=int,   default=None)
    parser.add_argument("--lr",     type=float, default=None)
    parser.add_argument("--precision", choices=["bf16","fp32"], default=None)
    parser.add_argument("--seed",   type=int,   default=None)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    result = train(
        corpus_path = args.corpus,
        steps       = args.steps,
        batch       = args.batch,
        accum       = args.accum,
        lr          = args.lr,
        precision   = args.precision,
        seed        = args.seed,
        resume      = not args.no_resume,
    )
    print(json.dumps(result, indent=2))
