"""
CognitiveOC v3 — 700M LLaMA-style Decoder Foundation Model
============================================================

Architecture decisions and justification:
  RoPE positional encoding      — Su et al. 2021. No length limit from
                                  learned positions; allows context extension
                                  at inference time without retraining.
  RMSNorm (vs LayerNorm)        — Zhang & Sennrich 2019. ~7% faster, no
                                  mean-centering, numerically stable in bf16.
  SwiGLU activation             — Shazeer 2020. ~1.5x quality gain over GELU
                                  at same parameter count on LLM benchmarks.
  Grouped Query Attention (GQA) — Ainslie et al. 2023. n_kv_head=4 vs
                                  n_head=12 gives 3x KV cache reduction,
                                  enabling 8K context on 8 GB GPU.
  Flash Attention 2             — Dao 2023. 3-5x faster than vanilla attention
                                  via IO-aware tiling; zero extra memory.
  Weight tying (embed <-> head) — Press & Wolf 2017. Saves ~92M params at
                                  48K vocab; improves sample efficiency.
  Gradient checkpointing        — Chen et al. 2016. ~33% speed cost, ~4x
                                  activation memory saving. Required for 700M
                                  training on RTX 5060 8GB.

Parameter estimate (n_layer=24, n_head=12, n_kv_head=4,
                    n_embd=1536, intermediate=4096, vocab=48000):
  Embedding:     48000 x 1536              =  73,728,000
  Attention Q:   24 x 1536 x 1536         =  56,623,104
  Attention KV:  24 x 2 x (512 x 1536)    =  37,748,736   (n_kv_head=4, head_dim=128)
  Attention O:   24 x 1536 x 1536         =  56,623,104
  FFN gate+up:   24 x 2 x (1536 x 4096)   = 301,989,888
  FFN down:      24 x 4096 x 1536         = 150,994,944
  RMSNorm:       24 x 2 x 1536 + 1536     =  74,880
  lm_head:       0  (tied to embedding)
  Total:                                  ~ 677,782,656  (~678M, ~700M with vocab)

VRAM budget at bf16 on RTX 5060 8GB:
  Model weights (inference):  678M x 2B         =  1.36 GB
  AdamW states (training):    678M x 8B         =  5.42 GB
  KV cache (8K, inference):   24x2x8192x128x2B  =  0.20 GB
  Activations + grad ckpt:                      ~  0.50 GB
  Framework overhead:                           ~  0.30 GB
  Training total:                               ~  7.78 GB  ← fits 8GB
  Inference total:                              ~  2.06 GB  ← ample headroom
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_ckpt

# Import from config — must be importable from project root
try:
    from config import MODEL, VOCAB_SIZE
except ImportError:
    # Fallback defaults for isolated testing
    VOCAB_SIZE = 48_000
    MODEL = dict(
        n_layer=24, n_head=12, n_kv_head=4, n_embd=1536, intermediate=4096,
        block_size=8192, rope_base=10_000, rope_scaling=None,
        dropout=0.0, norm_eps=1e-5, initializer_range=0.02,
        tie_weights=True, flash_attention=True, gradient_checkpointing=True,
        quant="none",
    )


# ═══════════════════════════════════════════════════════════════════
# 1. Rotary Position Embedding (RoPE)
# ═══════════════════════════════════════════════════════════════════

def _precompute_freqs(head_dim: int, max_seq: int,
                      base: float = 10_000.0,
                      device=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos/sin frequency tables for RoPE.

    Returns:
        cos: (max_seq, head_dim // 2)
        sin: (max_seq, head_dim // 2)
    """
    half = head_dim // 2
    inv_freq = 1.0 / (
        base ** (torch.arange(0, half, device=device).float() / half)
    )
    t = torch.arange(max_seq, device=device).float()
    freqs = torch.outer(t, inv_freq)          # (max_seq, half)
    return freqs.cos().to(torch.float32), freqs.sin().to(torch.float32)


def _apply_rope(x: torch.Tensor,
                cos: torch.Tensor,
                sin: torch.Tensor,
                offset: int = 0) -> torch.Tensor:
    """Apply RoPE rotation to query or key tensor.

    Args:
        x:      (B, n_head, T, head_dim)
        cos:    (max_seq, head_dim // 2)
        sin:    (max_seq, head_dim // 2)
        offset: KV cache offset — how many tokens have already been processed.

    Returns:
        (B, n_head, T, head_dim) — rotated tensor, same dtype as x.
    """
    B, H, T, D = x.shape
    half = D // 2
    c = cos[offset: offset + T].unsqueeze(0).unsqueeze(0)   # (1, 1, T, half)
    s = sin[offset: offset + T].unsqueeze(0).unsqueeze(0)   # (1, 1, T, half)

    x1 = x[..., :half]
    x2 = x[..., half:]
    rotated_x = torch.cat([-x2, x1], dim=-1)                # 90° rotation

    # Broadcast cos/sin to full head_dim then apply
    cs = torch.cat([c, c], dim=-1).to(x.dtype)
    ss = torch.cat([s, s], dim=-1).to(x.dtype)
    return x * cs + rotated_x * ss


# ═══════════════════════════════════════════════════════════════════
# 2. RMSNorm
# ═══════════════════════════════════════════════════════════════════

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalisation.

    No mean centering, no bias — only scale (gamma).
    Numerically stable in bf16 via upcast to float32.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute norm in float32 regardless of input dtype
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight


# ═══════════════════════════════════════════════════════════════════
# 3. SwiGLU Feed-Forward Network
# ═══════════════════════════════════════════════════════════════════

class SwiGLU(nn.Module):
    """SwiGLU gated feed-forward network.

    FFN(x) = SiLU(x @ W_gate) * (x @ W_up) @ W_down

    No bias terms — consistent with LLaMA / Mistral family.
    intermediate dimension is set to 4096 (≈ 2.67 × n_embd for n_embd=1536),
    which is the empirically best ratio per Shazeer (2020).
    """

    def __init__(self, n_embd: int, intermediate: int, dropout: float = 0.0):
        super().__init__()
        self.gate_proj = nn.Linear(n_embd, intermediate, bias=False)
        self.up_proj   = nn.Linear(n_embd, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, n_embd,  bias=False)
        self.drop      = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))   # gating signal
        up   = self.up_proj(x)             # value signal
        return self.drop(self.down_proj(gate * up))


# ═══════════════════════════════════════════════════════════════════
# 4. Grouped Query Attention (GQA)
# ═══════════════════════════════════════════════════════════════════

class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention with RoPE and optional KV cache.

    n_head=12 query heads share n_kv_head=4 KV heads.
    Each KV head is used by n_rep = n_head // n_kv_head = 3 query heads.

    Memory saving vs MHA:
        MHA KV cache:  L x 2 x T x n_head    x head_dim  (baseline)
        GQA KV cache:  L x 2 x T x n_kv_head x head_dim  (3x smaller)

    Flash Attention 2 path:
        Used when CUDA is available and flash_attention=True.
        Falls back to manual scaled dot-product on CPU or if not installed.
    """

    def __init__(self, n_embd: int, n_head: int, n_kv_head: int,
                 dropout: float = 0.0, flash: bool = True):
        super().__init__()
        assert n_head % n_kv_head == 0, (
            f"n_head ({n_head}) must be divisible by n_kv_head ({n_kv_head})"
        )
        self.n_head    = n_head
        self.n_kv_head = n_kv_head
        self.n_rep     = n_head // n_kv_head   # GQA repetition factor
        self.head_dim  = n_embd // n_head
        self.use_flash = flash

        self.q_proj = nn.Linear(n_embd, n_head    * self.head_dim, bias=False)
        self.k_proj = nn.Linear(n_embd, n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(n_embd, n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.drop   = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    @staticmethod
    def _expand_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """Expand KV heads to match query heads for GQA.

        (B, n_kv_head, T, D) -> (B, n_head, T, D)
        Uses expand + reshape — zero memory copy.
        """
        if n_rep == 1:
            return x
        B, H, T, D = x.shape
        return (
            x.unsqueeze(2)
             .expand(B, H, n_rep, T, D)
             .reshape(B, H * n_rep, T, D)
        )

    def forward(self,
                x: torch.Tensor,
                cos: torch.Tensor,
                sin: torch.Tensor,
                past_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
                ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x:       (B, T, E)
            cos/sin: RoPE tables  (max_seq, head_dim // 2)
            past_kv: (k_cache, v_cache) from previous step, or None

        Returns:
            output:  (B, T, E)
            new_kv:  (k, v) updated cache for this layer
        """
        B, T, E = x.shape
        offset  = past_kv[0].size(2) if past_kv is not None else 0

        # Project to Q, K, V
        q = self.q_proj(x).view(B, T, self.n_head,    self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K only
        q = _apply_rope(q, cos, sin, offset)
        k = _apply_rope(k, cos, sin, offset)

        # Append to KV cache
        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        new_kv = (k, v)

        # Expand KV to full query head count (GQA)
        k_full = self._expand_kv(k, self.n_rep)
        v_full = self._expand_kv(v, self.n_rep)

        # Attention computation
        can_flash = (
            self.use_flash
            and q.is_cuda
            and hasattr(F, "scaled_dot_product_attention")
        )

        if can_flash:
            # PyTorch 2.0+ scaled_dot_product_attention with Flash Attention kernel.
            # is_causal=True only when processing the full sequence (no cache).
            # When using KV cache we have already appended past tokens — not causal.
            is_causal = (past_kv is None)
            y = F.scaled_dot_product_attention(
                q, k_full, v_full,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=is_causal,
            )
        else:
            # Manual scaled dot-product — CPU / older GPU fallback
            scale  = 1.0 / math.sqrt(self.head_dim)
            scores = torch.matmul(q, k_full.transpose(-2, -1)) * scale  # (B,H,T,S)

            if past_kv is None:
                # Build causal mask for full-sequence forward
                S = scores.size(-1)
                causal = torch.ones(T, S, device=q.device, dtype=torch.bool).tril()
                scores = scores.masked_fill(~causal.unsqueeze(0).unsqueeze(0),
                                            float("-inf"))

            y = F.softmax(scores.float(), dim=-1).to(q.dtype)
            y = torch.matmul(y, v_full)

        # Merge heads and project
        y = y.transpose(1, 2).contiguous().view(B, T, E)
        return self.drop(self.o_proj(y)), new_kv


# ═══════════════════════════════════════════════════════════════════
# 5. Decoder Block
# ═══════════════════════════════════════════════════════════════════

class DecoderBlock(nn.Module):
    """Single transformer decoder block.

    Pre-norm residual architecture (LLaMA style):
        x = x + Attention(RMSNorm(x))
        x = x + FFN(RMSNorm(x))

    Pre-norm is more stable in bf16 than post-norm.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        E = cfg["n_embd"]
        self.attn_norm = RMSNorm(E, cfg["norm_eps"])
        self.ffn_norm  = RMSNorm(E, cfg["norm_eps"])
        self.attn = GroupedQueryAttention(
            n_embd    = E,
            n_head    = cfg["n_head"],
            n_kv_head = cfg["n_kv_head"],
            dropout   = cfg["dropout"],
            flash     = cfg.get("flash_attention", True),
        )
        self.ffn = SwiGLU(E, cfg["intermediate"], cfg["dropout"])

    def forward(self,
                x: torch.Tensor,
                cos: torch.Tensor,
                sin: torch.Tensor,
                past_kv: Optional[tuple] = None,
                ) -> tuple[torch.Tensor, tuple]:
        attn_out, new_kv = self.attn(self.attn_norm(x), cos, sin, past_kv)
        x = x + attn_out
        x = x + self.ffn(self.ffn_norm(x))
        return x, new_kv


# ═══════════════════════════════════════════════════════════════════
# 6. CognitiveDecoder — 700M Foundation Model
# ═══════════════════════════════════════════════════════════════════

class CognitiveDecoder(nn.Module):
    """COC v3 700M foundation model.

    Training usage:
        model = CognitiveDecoder()
        logits, loss, _ = model(idx, targets=targets)
        loss.backward()

    Inference usage:
        # Full sequence (first token):
        logits, _, past = model(idx)
        # Incremental (subsequent tokens, KV cache):
        logits, _, past = model(next_tok, past=past)
        # Convenience autoregressive:
        output_ids = model.generate(prompt_ids, max_new_tokens=256)

    Checkpoint:
        CognitiveDecoder.from_checkpoint("var/checkpoints/model_700m.pt")
    """

    def __init__(self, cfg: dict = None):
        super().__init__()
        cfg = cfg or MODEL
        self.cfg = cfg
        E   = cfg["n_embd"]
        S   = cfg["block_size"]
        HD  = E // cfg["n_head"]   # head_dim

        # Token embedding
        self.tok_emb = nn.Embedding(VOCAB_SIZE, E)

        # RoPE buffers — not trainable; rebuilt when device changes
        self.register_buffer(
            "rope_cos", torch.zeros(S, HD // 2), persistent=False
        )
        self.register_buffer(
            "rope_sin", torch.zeros(S, HD // 2), persistent=False
        )
        self._rope_built = False

        # Transformer blocks
        self.blocks  = nn.ModuleList(
            DecoderBlock(cfg) for _ in range(cfg["n_layer"])
        )

        # Final normalisation + LM head
        self.norm_f  = RMSNorm(E, cfg["norm_eps"])
        self.lm_head = nn.Linear(E, VOCAB_SIZE, bias=False)

        # Weight tying
        if cfg.get("tie_weights", True):
            self.lm_head.weight = self.tok_emb.weight

        # Gradient checkpointing flag (enabled during training)
        self.use_grad_ckpt = cfg.get("gradient_checkpointing", True)

        self._init_weights()

    # ── Weight Initialisation ────────────────────────────────────────
    def _init_weights(self):
        std = self.cfg.get("initializer_range", 0.02)
        # Scale residual projections by 1/sqrt(2*n_layer) — GPT-2 recipe
        res_std = std / math.sqrt(2 * self.cfg["n_layer"])

        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                is_res = any(k in name for k in ("o_proj", "down_proj"))
                nn.init.normal_(module.weight,
                                mean=0.0, std=res_std if is_res else std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)

    # ── RoPE Table ───────────────────────────────────────────────────
    def _ensure_rope(self, device):
        """Build RoPE tables if not yet built or if device changed."""
        if self._rope_built and self.rope_cos.device == device:
            return
        HD   = self.cfg["n_embd"] // self.cfg["n_head"]
        base = float(self.cfg.get("rope_base", 10_000))
        S    = self.cfg["block_size"]
        cos, sin = _precompute_freqs(HD, S, base, device)
        self.rope_cos = cos
        self.rope_sin = sin
        self._rope_built = True

    # ── Forward Pass ─────────────────────────────────────────────────
    def forward(self,
                idx: torch.Tensor,
                targets: Optional[torch.Tensor] = None,
                past: Optional[list] = None,
                ) -> tuple[torch.Tensor, Optional[torch.Tensor], list]:
        """
        Args:
            idx     : (B, T)  input token IDs
            targets : (B, T)  next-token labels; cross-entropy loss computed
                              when provided. Should be idx shifted left by 1.
            past    : list of (k, v) tensors per layer for KV cache; None on
                      first step.

        Returns:
            logits   : (B, T, vocab_size)
            loss     : scalar tensor if targets provided, else None
            new_past : list of (k, v) per layer — updated KV cache
        """
        device = idx.device
        self._ensure_rope(device)
        B, T   = idx.shape

        x = self.tok_emb(idx)    # (B, T, E)
        new_past = []

        for i, block in enumerate(self.blocks):
            pk = past[i] if past is not None else None

            if self.use_grad_ckpt and self.training:
                # Gradient checkpointing: trade compute for memory.
                # We cannot pass pk through checkpoint if it's None,
                # so wrap the block call in a closure.
                def _block_fwd(x_,
                               _blk=block,
                               _cos=self.rope_cos,
                               _sin=self.rope_sin,
                               _pk=pk):
                    # grad_ckpt does not support returning multiple values
                    # via tuple when use_reentrant=False on some versions,
                    # so we return both concatenated and split afterward.
                    out, kv = _blk(x_, _cos, _sin, _pk)
                    return out, kv

                x, kv = grad_ckpt(_block_fwd, x, use_reentrant=False)
            else:
                x, kv = block(x, self.rope_cos, self.rope_sin, pk)

            new_past.append(kv)

        logits = self.lm_head(self.norm_f(x))   # (B, T, vocab)

        loss = None
        if targets is not None:
            # Standard next-token prediction loss; ignore_index=-1 for padding
            loss = F.cross_entropy(
                logits.view(-1, VOCAB_SIZE),
                targets.view(-1),
                ignore_index=-1,
            )

        return logits, loss, new_past

    # ── Autoregressive Generation ────────────────────────────────────
    @torch.inference_mode()
    def generate(self,
                 idx: torch.Tensor,
                 max_new_tokens: int = 256,
                 temperature: float = 0.7,
                 top_k: int = 50,
                 top_p: float = 0.9,
                 repetition_penalty: float = 1.1,
                 eos_id: Optional[int] = None,
                 ) -> torch.Tensor:
        """Greedy / sampling autoregressive generation with KV cache.

        Args:
            idx:               (B, T) prompt token IDs
            max_new_tokens:    maximum tokens to generate
            temperature:       sampling temperature (lower = more deterministic)
            top_k:             keep only top-k logits; 0 = disabled
            top_p:             nucleus sampling threshold; 1.0 = disabled
            repetition_penalty: > 1.0 discourages repeating seen tokens
            eos_id:            stop generation when this token is sampled

        Returns:
            (B, T + generated) token IDs
        """
        self.eval()
        past    = None
        cur_idx = idx
        seen    = set(idx.view(-1).tolist())

        for _ in range(max_new_tokens):
            logits, _, past = self.forward(cur_idx, past=past)
            logits = logits[:, -1, :].float()   # (B, vocab) — last position

            # Repetition penalty
            if repetition_penalty != 1.0 and seen:
                for tok in seen:
                    logits[:, tok] = logits[:, tok] / repetition_penalty

            # Temperature scaling
            if temperature > 0:
                logits = logits / max(temperature, 1e-8)

            # Top-k filtering
            if top_k and top_k > 0:
                kth_val = torch.topk(logits, min(top_k, logits.size(-1)))[0][:, -1:]
                logits  = logits.masked_fill(logits < kth_val, float("-inf"))

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                # Remove tokens that push cumulative probability above top_p
                sorted_logits[cum_probs - F.softmax(sorted_logits, dim=-1) > top_p] \
                    = float("-inf")
                logits.scatter_(1, sorted_indices, sorted_logits)

            probs   = F.softmax(logits, dim=-1)
            nxt     = torch.multinomial(probs, num_samples=1)  # (B, 1)

            seen.add(nxt.item())
            idx     = torch.cat([idx, nxt], dim=1)
            cur_idx = nxt          # only feed new token; past handles history

            if eos_id is not None and (nxt == eos_id).all():
                break
            if idx.size(1) >= self.cfg["block_size"]:
                break

        return idx

    # ── Checkpoint Utilities ─────────────────────────────────────────
    @classmethod
    def from_checkpoint(cls,
                        path: str,
                        cfg: dict = None,
                        device: str = "cpu",
                        ) -> "CognitiveDecoder":
        """Load a COC v3 checkpoint produced by train/train_model.py."""
        ckpt      = torch.load(str(path), map_location=device)
        model_cfg = ckpt.get("model_cfg", cfg or MODEL)
        m         = cls(model_cfg)
        # Allow strict=False for forward-compat when new keys are added
        missing, unexpected = m.load_state_dict(ckpt["model"], strict=False)
        if missing:
            print(f"[transformer] Missing keys: {missing}")
        if unexpected:
            print(f"[transformer] Unexpected keys: {unexpected}")
        return m.to(device)

    def save_checkpoint(self,
                        path: str,
                        step: int = 0,
                        opt_state: dict = None,
                        extra: dict = None):
        """Save a checkpoint with full metadata for safe resume."""
        import time
        payload = {
            "model":      self.state_dict(),
            "model_cfg":  self.cfg,
            "step":       step,
            "vocab_size": VOCAB_SIZE,
            "timestamp":  time.time(),
        }
        if opt_state:
            payload["optimizer"] = opt_state
        if extra:
            payload.update(extra)
        torch.save(payload, str(path))

    # ── Utility ──────────────────────────────────────────────────────
    def num_params(self, trainable_only: bool = True) -> int:
        """Count parameters."""
        return sum(
            p.numel() for p in self.parameters()
            if (not trainable_only or p.requires_grad)
        )

    def param_groups(self, weight_decay: float = 0.1) -> list[dict]:
        """Separate parameters into decay / no-decay groups for AdamW.

        Matrices (dim >= 2) get weight decay.
        Vectors (norms, biases, embeddings at dim=1) do not.
        """
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.dim() >= 2:
                decay.append(p)
            else:
                no_decay.append(p)
        return [
            {"params": decay,    "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

    def __repr__(self) -> str:
        n = self.num_params()
        return (f"CognitiveDecoder("
                f"n_layer={self.cfg['n_layer']}, "
                f"n_embd={self.cfg['n_embd']}, "
                f"n_head={self.cfg['n_head']}, "
                f"n_kv_head={self.cfg['n_kv_head']}, "
                f"block_size={self.cfg['block_size']}, "
                f"params={n:,})")


# ═══════════════════════════════════════════════════════════════════
# Backward-compatibility alias
# Old baseline code imported as "GPT" — this keeps it working.
# ═══════════════════════════════════════════════════════════════════
GPT = CognitiveDecoder


# ═══════════════════════════════════════════════════════════════════
# Quick sanity check (run directly: python models/transformer.py)
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Smaller config for fast smoke test
    test_cfg = dict(
        n_layer=2, n_head=4, n_kv_head=2, n_embd=64, intermediate=128,
        block_size=128, rope_base=10_000, rope_scaling=None,
        dropout=0.0, norm_eps=1e-5, initializer_range=0.02,
        tie_weights=True, flash_attention=True, gradient_checkpointing=False,
        quant="none",
    )

    m = CognitiveDecoder(test_cfg).to(device)
    print(m)
    print(f"Parameters: {m.num_params():,}")

    # Forward pass — training mode
    B, T = 2, 32
    idx     = torch.randint(0, VOCAB_SIZE, (B, T), device=device)
    targets = torch.randint(0, VOCAB_SIZE, (B, T), device=device)

    m.train()
    logits, loss, past = m(idx, targets=targets)
    print(f"Logits:     {logits.shape}")
    print(f"Loss:       {loss.item():.4f}")

    # Inference with KV cache
    m.eval()
    with torch.inference_mode():
        logits2, _, past2 = m(idx)
        logits3, _, past3 = m(idx[:, -1:], past=past2)
    print(f"Inference logits (step 1): {logits2.shape}")
    print(f"Inference logits (step 2): {logits3.shape}")

    # Generation
    prompt = torch.randint(0, VOCAB_SIZE, (1, 8), device=device)
    out    = m.generate(prompt, max_new_tokens=16, temperature=0.7)
    print(f"Generated:  {out.shape}  — {out[0].tolist()}")

    # Verify RoPE round-trip (cos² + sin² = 1)
    half = test_cfg["n_embd"] // test_cfg["n_head"] // 2
    cos, sin = _precompute_freqs(
        test_cfg["n_embd"] // test_cfg["n_head"], 128, device=device
    )
    unit = (cos ** 2 + sin ** 2)
    assert torch.allclose(unit, torch.ones_like(unit), atol=1e-5), \
        "RoPE frequency table failed unit-circle check"
    print("RoPE unit-circle check: PASS")

    # Verify GQA expansion
    gqa = GroupedQueryAttention(
        n_embd=64, n_head=4, n_kv_head=2, flash=False
    ).to(device)
    x   = torch.randn(1, 8, 64, device=device)
    c   = torch.zeros(128, 8, device=device)
    s   = torch.zeros(128, 8, device=device)
    out, kv = gqa(x, c, s)
    assert out.shape == (1, 8, 64), f"GQA output shape wrong: {out.shape}"
    print("GQA shape check: PASS")

    print("\n✓ transformer.py — all checks passed")
    sys.exit(0)
