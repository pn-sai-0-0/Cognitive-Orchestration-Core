"""
CognitiveOC / COC v3 — Master Configuration
Single source of truth. Restart required after changes unless marked hot-reload.

Hardware budget:
  GPU  : RTX 5060 8 GB   (CUDA, bf16, Flash Attention 2)
  NPU  : AMD Ryzen AI 15 GB  (DirectML / ONNX Runtime)
  RAM  : 32 GB
  SSD  : 1 TB NVMe
"""
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════════
BASE_DIR        = Path(__file__).resolve().parent
STORE_DIR       = BASE_DIR / "var"
TOKENIZER_DIR   = STORE_DIR / "tokenizer"
CHECKPOINT_DIR  = STORE_DIR / "checkpoints"
INDEX_DIR       = STORE_DIR / "index"
CACHE_DIR       = STORE_DIR / "cache"
UPLOAD_DIR      = STORE_DIR / "uploads"
WORKSPACE_DIR   = STORE_DIR / "workspaces"
DATASET_DIR     = STORE_DIR / "datasets"
WORKFLOW_DIR    = STORE_DIR / "workflows"
LOG_DIR         = STORE_DIR / "logs"
ONNX_DIR        = STORE_DIR / "onnx"
USER_MODEL_DB   = STORE_DIR / "user_model.db"
MEMORY_DB       = STORE_DIR / "memory.db"
KG_PATH         = STORE_DIR / "knowledge_graph.db"   # SQLite for v3 (was .json)
KG_BACKUP       = STORE_DIR / "knowledge_graph_backup.json"
SYNONYMS_PATH   = STORE_DIR / "synonyms.json"
COGNITION_STATE = STORE_DIR / "cognition_state.json"
GUARDRAIL_STATE = STORE_DIR / "guardrails_state.json"
CORPUS_DIR      = BASE_DIR / "data" / "corpus"
EVAL_BASELINE   = BASE_DIR / "eval" / "baseline"
EVAL_GATES      = BASE_DIR / "eval" / "gates"

# ═══════════════════════════════════════════════════════════════════
# TOKENIZER  — 48K SentencePiece Unigram
# Upgrade from 8K BPE baseline.
# Rationale: 48K vocab → ~3.8 chars/token on English prose (vs ~1.7 for 8K).
# Fits full context more densely; less compute per semantic unit.
# byte_fallback=True → no <unk> for any Unicode input.
# ═══════════════════════════════════════════════════════════════════
VOCAB_SIZE = 48_000

TOKENIZER = dict(
    type               = "sentencepiece",
    model_type         = "unigram",       # better OOV than BPE at same vocab size
    vocab_size         = VOCAB_SIZE,
    character_coverage = 0.9995,
    byte_fallback      = True,
    pad_id=0, bos_id=1, eos_id=2, unk_id=3,
    # Structural special tokens injected post-training
    extra_tokens=[
        "<memory>", "</memory>",
        "<retrieval>", "</retrieval>",
        "<kg>", "</kg>",
        "<reasoning>", "</reasoning>",
        "<tool>", "</tool>",
        "<emotion>", "</emotion>",
        "<intent>", "</intent>",
        "<teaching>", "</teaching>",
        "<decision>", "</decision>",
        "<plan>", "</plan>",
        "<user>", "</user>",
        "<assistant>", "</assistant>",
        "<system>", "</system>",
        "<cite>", "</cite>",
        "<confidence>", "</confidence>",
        "<reflection>", "</reflection>",
        "<goal>", "</goal>",
    ],
    model_prefix = "tokenizer/48K/spm48k",
)

# ═══════════════════════════════════════════════════════════════════
# FOUNDATION MODEL — 700M Decoder (LLaMA-style)
#
# Architecture choices and rationale:
#   RoPE positional encoding   — no length limit from learned positions;
#                                allows context extension at inference
#   RMSNorm (vs LayerNorm)     — 7% faster, numerically stable
#   SwiGLU activation          — ~1.5x better than GELU on LLM benchmarks
#   Grouped Query Attention     — n_kv_head=4 vs n_head=12 → 3× KV cache
#                                 reduction; enables 8K context on 8 GB GPU
#   Flash Attention 2           — 3–5x faster than vanilla attention
#   Weight tying (embed↔head)   — saves ~92 M params at 48K vocab
#   Gradient checkpointing      — required for training on 8 GB; ~33%
#                                 speed cost, ~4× memory saving
#
# VRAM budget (RTX 5060 8 GB, bf16):
#   Model weights  : 700M × 2 B  =  1.40 GB
#   AdamW states   : 700M × 8 B  =  5.60 GB  (training only)
#   KV cache 8K    : 24×2×8192×128×4×2 B ≈ 0.20 GB  (inference)
#   Activations    : ~0.50 GB  (gradient checkpointing)
#   Overhead       : ~0.30 GB
#   Training total : ~7.80 GB  ← fits with gradient_checkpointing=True
#   Inference total: ~2.10 GB  ← ample room for concurrent encoders
# ═══════════════════════════════════════════════════════════════════
MODEL = dict(
    # Dimensions
    n_layer           = 24,       # transformer depth
    n_head            = 12,       # attention heads (query)
    n_kv_head         = 4,        # key/value heads — GQA 3:1
    n_embd            = 1536,     # embedding dimension
    intermediate      = 4096,     # SwiGLU up/gate projection dim
    block_size        = 8192,     # max context tokens
    # Positional encoding
    rope_base         = 10_000,   # RoPE θ base frequency
    rope_scaling      = None,     # None | {"type":"linear","factor":2.0}
    # Regularisation
    dropout           = 0.0,      # 0 at inference; set to 0.05 for SFT
    # Normalisation
    norm_eps          = 1e-5,
    # Weight init
    initializer_range = 0.02,
    # Efficiency
    tie_weights       = True,     # lm_head weight = tok_embedding weight
    flash_attention   = True,     # requires flash_attn package on CUDA
    gradient_checkpointing = True,
    # Quantisation (inference only)
    quant             = "none",   # "none"|"int8"|"int4"|"bnb_nf4"
)

# ═══════════════════════════════════════════════════════════════════
# ENCODER INTELLIGENCE STACK
# All encoders: sentence-transformer compatible, ONNX exportable.
# NPU target: AMD Ryzen AI (DirectML/ONNX Runtime).
# GPU stays 100% free for the 700M decoder during inference.
#
# RAM budget for all encoders (NPU + CPU):
#   semantic (BGE-base):   ~440 MB
#   cross_encoder:         ~90  MB
#   Others (MiniLM):       ~22  MB each × 9 = ~200 MB
#   Total encoder RAM:     ~730 MB  ← well within 32 GB
# ═══════════════════════════════════════════════════════════════════
ENCODERS = dict(
    semantic=dict(
        model="BAAI/bge-base-en-v1.5",   # 768-dim, MTEB top retrieval model
        dim=768, device="npu", batch=32, max_len=512, normalize=True,
        onnx_path="var/onnx/semantic_encoder.onnx",
    ),
    cross_encoder=dict(
        model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        device="npu", batch=16, max_len=512,
        onnx_path="var/onnx/cross_encoder.onnx",
    ),
    intent=dict(
        model="sentence-transformers/all-MiniLM-L6-v2",
        dim=384, device="cpu", batch=64, max_len=128,
        onnx_path="var/onnx/intent_encoder.onnx",
    ),
    emotion=dict(
        model="SamLowe/roberta-base-go_emotions",
        dim=768, device="npu", batch=32, max_len=128,
        onnx_path="var/onnx/emotion_encoder.onnx",
        labels=[
            "admiration","amusement","anger","annoyance","approval",
            "caring","confusion","curiosity","desire","disappointment",
            "disapproval","disgust","embarrassment","excitement","fear",
            "gratitude","grief","joy","love","nervousness","optimism",
            "pride","realization","relief","remorse","sadness","surprise","neutral",
        ],
    ),
    memory=dict(
        model="sentence-transformers/all-MiniLM-L6-v2",
        dim=384, device="cpu", batch=64,
        onnx_path="var/onnx/memory_encoder.onnx",
    ),
    safety=dict(
        model="sentence-transformers/all-MiniLM-L6-v2",
        dim=384, device="cpu", batch=64,
        onnx_path="var/onnx/safety_encoder.onnx",
    ),
    summarization=dict(
        model="sentence-transformers/all-mpnet-base-v2",
        dim=768, device="npu", batch=16,
        onnx_path="var/onnx/summarization_encoder.onnx",
    ),
    judge=dict(
        model="cross-encoder/qnli-electra-base",
        device="npu", batch=16,
        onnx_path="var/onnx/judge_encoder.onnx",
    ),
    teaching=dict(
        model="sentence-transformers/all-MiniLM-L6-v2",
        dim=384, device="cpu", batch=64,
        onnx_path="var/onnx/teaching_encoder.onnx",
    ),
    goal=dict(
        model="sentence-transformers/all-MiniLM-L6-v2",
        dim=384, device="cpu", batch=64,
        onnx_path="var/onnx/goal_encoder.onnx",
    ),
    kg=dict(
        model="BAAI/bge-small-en-v1.5",
        dim=384, device="npu", batch=64,
        onnx_path="var/onnx/kg_encoder.onnx",
    ),
    dataset=dict(
        model="sentence-transformers/all-MiniLM-L6-v2",
        dim=384, device="cpu", batch=128,
        onnx_path="var/onnx/dataset_encoder.onnx",
    ),
    planning=dict(
        model="sentence-transformers/all-MiniLM-L6-v2",
        dim=384, device="cpu", batch=64,
        onnx_path="var/onnx/planning_encoder.onnx",
    ),
)

# ═══════════════════════════════════════════════════════════════════
# INFERENCE
# Context budget (must sum to ≤ block_size = 8192)
# ═══════════════════════════════════════════════════════════════════
INFERENCE = dict(
    max_new_tokens        = 1024,
    temperature           = 0.7,
    top_p                 = 0.9,
    top_k                 = 50,
    repetition_penalty    = 1.1,
    kv_cache              = True,
    dtype                 = "bfloat16",
    device                = "cuda",
    # Context window budget (tokens)
    system_prompt_tokens  = 256,
    memory_budget_tokens  = 512,
    retrieval_budget_tokens = 2048,
    kg_budget_tokens      = 256,
    tool_budget_tokens    = 512,
    history_budget_tokens = 512,
    cognition_budget_tokens = 128,
    generation_budget     = 1024,
    # 256+512+2048+256+512+512+128+1024 = 5248 ← well inside 8192
)

# ═══════════════════════════════════════════════════════════════════
# RETRIEVAL
# ═══════════════════════════════════════════════════════════════════
RETRIEVAL = dict(
    chunk_size            = 512,
    chunk_overlap         = 64,
    top_k                 = 8,          # candidates before reranking
    rerank_k              = 24,         # wider pool for cross-encoder
    final_k               = 5,          # post-rerank selections passed to context
    cache_ttl_s           = 3600,
    cache_max_entries     = 512,
    multi_hop_max         = 3,
    min_score_threshold   = 0.25,
    bm25_weight           = 0.30,
    semantic_weight       = 0.70,
    query_expansion       = True,
    query_rewrite         = True,
    self_improve          = True,
)

# ═══════════════════════════════════════════════════════════════════
# EMBEDDING  (primary encoder, RAG)
# ═══════════════════════════════════════════════════════════════════
EMBEDDING = dict(
    model     = "BAAI/bge-base-en-v1.5",
    dim       = 768,
    device    = "auto",   # auto → npu→cuda→cpu priority
    onnx_path = None,
    batch     = 32,
)

# ═══════════════════════════════════════════════════════════════════
# MEMORY
# ═══════════════════════════════════════════════════════════════════
MEMORY = dict(
    session_window          = 30,
    recall_k                = 8,
    max_long_term           = 20_000,
    decay_half_life_days    = 45,
    archive_threshold       = 0.15,
    consolidation_interval_h = 24,
    compression_threshold   = 500,     # chars — memories above this get compressed
    similarity_threshold    = 0.92,    # encoder cosine threshold for near-dedup
    # Composite score weights
    importance_weight       = 0.40,
    recency_weight          = 0.30,
    frequency_weight        = 0.20,
    relevance_weight        = 0.10,
)

# ═══════════════════════════════════════════════════════════════════
# KNOWLEDGE GRAPH
# ═══════════════════════════════════════════════════════════════════
KNOWLEDGE_GRAPH = dict(
    confidence_threshold    = 0.50,
    max_triples             = 500_000,
    max_entities            = 100_000,
    relation_types=[
        "is","has","uses","created","works_at","located_in","belongs_to",
        "contains","born_in","named","means","causes","depends_on","produces",
        "relates_to","contradicts","precedes","follows","similar_to",
        "instance_of","part_of","derived_from","enables","inhibits",
    ],
    enable_neural_extraction = False,   # enable after model checkpoint exists
    cluster_min_size        = 3,
    contradiction_resolution = "confidence",   # "confidence"|"recency"|"manual"
)

# ═══════════════════════════════════════════════════════════════════
# HUMAN COGNITION LAYER
# hot-reload: mode and per-module toggles read from cognition_state.json
# ═══════════════════════════════════════════════════════════════════
COGNITION = dict(
    mode = "full",      # "full"|"partial"|"off"|"custom"
    modules=dict(
        emotion=True, intent=True, user_modeling=True, goal_tracking=True,
        teaching=True, decision_support=True, reflection=True,
        personality=True, context_aware=True,
    ),
    emotion=dict(
        confidence_threshold=0.50,
        top_emotions=3,
        map_to_response=True,       # adjust tone based on detected emotion
    ),
    teaching=dict(
        default_level="intermediate",
        levels=["beginner","intermediate","advanced","expert"],
        adaptive=True,
        quiz_frequency=5,
        gap_detection=True,
    ),
    user_model=dict(
        persistence=True,
        profile_max_entries=1_000,
    ),
    goals=dict(
        max_active=20,
        persistence=True,
        reminder_interval_h=24,
    ),
    reflection=dict(
        enabled=True,
        max_rounds=2,
        consistency_check=True,
        citation_check=True,
    ),
    personality=dict(
        default_mode="assistant",
        available_modes=["teacher","mentor","engineer","researcher",
                         "coach","assistant","supportive"],
        auto_detect=True,
    ),
)

# ═══════════════════════════════════════════════════════════════════
# GUARDRAIL SYSTEM
# Hard integrity guards: ALWAYS ON — cannot be disabled under any mode.
# Cognitive guardrails: user-controllable via profiles and per-toggle.
# ═══════════════════════════════════════════════════════════════════
GUARDRAILS = dict(
    # ── Hard integrity — always on ───────────────────────────────────
    integrity=dict(
        file_integrity=True, db_integrity=True, checkpoint_integrity=True,
        memory_integrity=True, path_validation=True, permission_validation=True,
        schema_validation=True, process_stability=True, crash_protection=True,
    ),
    # ── Cognitive guardrails — toggleable ────────────────────────────
    cognitive=dict(
        injection_check=True, jailbreak_detection=True,
        pii_detection=True, pii_redaction=True,
        output_filtering=True, tool_safety=True,
        retrieval_sanitization=True, memory_safety=True,
        kg_validation=True, workspace_validation=True,
        judge_enforcement=False,   # requires judge encoder
        policy_enforcement=True,
    ),
    profiles=["strict","standard","research","developer","custom","off"],
    default_profile="standard",
    # Runtime limits
    max_upload_mb=100,
    allowed_extensions={
        ".pdf",".docx",".txt",".md",".csv",".xlsx",
        ".png",".jpg",".jpeg",".json",".py",".yaml",".toml",".log",
    },
    rate_limit_rpm=120,
    max_input_chars=50_000,
    audit_logging=True,
    audit_log_path="var/logs/guardrail_audit.jsonl",
)

# Legacy alias — keeps old code working
SAFETY = dict(
    max_upload_mb      = GUARDRAILS["max_upload_mb"],
    redact_pii         = GUARDRAILS["cognitive"]["pii_redaction"],
    allowed_extensions = GUARDRAILS["allowed_extensions"],
    rate_limit_rpm     = GUARDRAILS["rate_limit_rpm"],
)

# ═══════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════
TRAIN = dict(
    # Tokenizer
    tokenizer_max_chars = 0,       # 0 = full corpus
    # Pre-training (700M)
    steps               = 100_000,
    warmup_steps        = 2_000,
    batch               = 2,       # micro-batch (gradient checkpointing required)
    accum_steps         = 16,      # effective batch = 32
    lr                  = 3e-4,
    lr_min              = 3e-5,
    weight_decay        = 0.1,
    grad_clip           = 1.0,
    precision           = "bf16",
    seed                = 1337,
    val_frac            = 0.05,
    val_every           = 500,
    save_every          = 1_000,
    log_every           = 50,
    # SFT / fine-tuning
    sft_lr              = 1e-4,
    sft_steps           = 5_000,
    sft_accum           = 8,
    sft_dropout         = 0.05,
    # Display
    live_display        = True,    # TUI metrics during training
    display_interval_s  = 5,
)

# ═══════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════
EVALUATION = dict(
    perplexity_target   = 15.0,
    retrieval_mrr_target= 0.85,
    memory_recall_target= 0.90,
    kg_f1_target        = 0.80,
    reasoning_acc_target= 0.75,
    emotion_f1_target   = 0.70,
    # Metric weights for overall readiness score
    weights=dict(
        perplexity=0.25, retrieval_mrr=0.20, memory_recall=0.15,
        kg_f1=0.15, reasoning_acc=0.15, emotion_f1=0.10,
    ),
)

# ═══════════════════════════════════════════════════════════════════
# OBSERVABILITY
# ═══════════════════════════════════════════════════════════════════
OBSERVABILITY = dict(
    metrics_path       = "var/metrics.json",
    log_path           = "var/logs/runtime.jsonl",
    training_log       = "var/checkpoints/train_log.jsonl",
    hardware_poll_s    = 2,
    history_max        = 10_000,
)

# ═══════════════════════════════════════════════════════════════════
# WORKFLOW ENGINE
# ═══════════════════════════════════════════════════════════════════
WORKFLOW = dict(
    max_steps        = 50,
    step_timeout_s   = 300,
    max_concurrent   = 3,
    retry_max        = 3,
    retry_delay_s    = 5,
)

# ═══════════════════════════════════════════════════════════════════
# DATASET GENERATION
# ═══════════════════════════════════════════════════════════════════
DATASET = dict(
    export_dir          = "var/datasets",
    review_queue_path   = "var/datasets/review_queue.jsonl",
    auto_approve        = False,
    dedup_threshold     = 0.92,
    min_quality_score   = 0.60,
    types=["conversation","retrieval","kg","memory","teaching","emotion","reasoning"],
)

# ═══════════════════════════════════════════════════════════════════
# RESEARCH ENGINE
# ═══════════════════════════════════════════════════════════════════
RESEARCH = dict(
    max_loops       = 10,
    evidence_min    = 3,
    synthesis_top_k = 10,
    validation_rounds = 2,
    report_format   = "markdown",
)

# ═══════════════════════════════════════════════════════════════════
# VALIDATION ENGINE
# ═══════════════════════════════════════════════════════════════════
VALIDATION = dict(
    fact_check=True, citation_check=True, kg_check=True,
    memory_check=True, reasoning_check=True,
    confidence_threshold=0.70,
)

# ═══════════════════════════════════════════════════════════════════
# LLM JUDGE
# ═══════════════════════════════════════════════════════════════════
JUDGE = dict(
    enabled=False, model_path=None, always_score=False,
    score_threshold=0.60,
    faithfulness_weight=0.40, consistency_weight=0.30, citation_weight=0.30,
)

# ═══════════════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════════════
UI = dict(
    host="127.0.0.1", port=8765,
    streaming=True, heartbeat_ms=500,
    auth=True, auth_key_path="var/auth_key.txt",
)

STREAMING = dict(enabled=True, heartbeat_ms=500)


# ═══════════════════════════════════════════════════════════════════
# CORPUS ENGINEERING
# ═══════════════════════════════════════════════════════════════════
# Adjust CORPUS_WAREHOUSE_DIR to the path of your 1TB SSD.
# All other corpus paths are relative to the repo BASE_DIR.
CORPUS_WAREHOUSE_DIR = Path("D:/corpus_warehouse")   # ← set to your 1TB SSD path
CORPUS_RELEASE_DIR   = CORPUS_WAREHOUSE_DIR / "releases"
CORPUS_REGISTRY_PATH = BASE_DIR / "governance" / "source_registry.json"
CORPUS_AUDIT_LOG_DIR = CORPUS_WAREHOUSE_DIR / "governance_logs"
CORPUS_APPROVAL_LOG  = BASE_DIR / "governance" / "approval_log.jsonl"
CORPUS_LICENCE_RULES = BASE_DIR / "governance" / "license_rules.json"
CORPUS_REVIEW_QUEUE  = CORPUS_WAREHOUSE_DIR / "review_queue" / "pending.jsonl"

# Training-control paths (inside repo CHECKPOINT_DIR on main SSD)
TRAINING_LEDGER_PATH  = CHECKPOINT_DIR / "training_ledger.jsonl"
SHARD_TRACKER_PATH    = CHECKPOINT_DIR / "shard_tracker.json"
PROVENANCE_PATH       = CHECKPOINT_DIR / "provenance.json"

CORPUS = dict(
    warehouse_dir          = str(CORPUS_WAREHOUSE_DIR),
    release_dir            = str(CORPUS_RELEASE_DIR),
    # Quality gate thresholds
    min_quality_score      = 0.45,   # below this = auto-reject
    auto_approve_threshold = 0.70,   # above this AND risk <= risk_human_review = auto-approve
    risk_human_review      = 0.20,   # above this risk score = human queue
    risk_reject            = 0.50,   # above this risk score = auto-reject
    # Deduplication thresholds
    dedup_exact_threshold  = 1.0,    # SHA-256 exact match (always on)
    dedup_near_threshold   = 0.85,   # MinHash Jaccard — cross-source near-dedup
    dedup_within_threshold = 0.92,   # cosine — within-source (data/pipeline.py)
    # Release configuration
    split_ratios           = (0.90, 0.05, 0.05),   # train / val / test
    split_seed             = 42,                   # fixed for reproducibility
    target_v1_tokens       = 30_000_000_000,        # 30B tokens in first release
    target_warehouse_tokens= 65_000_000_000,        # 65B tokens warehouse target
    # Synthetic data rules
    synthetic_min_quality  = 0.70,   # higher bar than real data
    auto_approve_synthetic = False,  # NEVER auto-approve synthetic — always human review
    # Token fertility estimate (chars per token, 48K SentencePiece Unigram)
    chars_per_token        = 3.8,
)

# ═══════════════════════════════════════════════════════════════════
# TRAINING CONTROL
# ═══════════════════════════════════════════════════════════════════
TRAINING_CONTROL = dict(
    # Shard configuration
    shard_size             = 10_000_000,   # tokens per shard (~1 training hour)
    # Resume behaviour
    allow_cross_release_resume = False,    # block resuming onto different release
    abort_on_lock_mismatch     = True,     # abort training if lock hash fails
    abort_on_shard_dup         = True,     # abort if shard already consumed
    # Provenance
    record_provenance          = True,
    # Training ledger
    ledger_path            = str(TRAINING_LEDGER_PATH),
    shard_tracker_path     = str(SHARD_TRACKER_PATH),
    provenance_path        = str(PROVENANCE_PATH),
)

# ═══════════════════════════════════════════════════════════════════
# CORPUS ZERO-RISK POLICY
# ═══════════════════════════════════════════════════════════════════
# Hard policy enforced at release build time.
# max_licence_risk_v1 = 0.20 means only PD/CC0/Apache/MIT/CC-BY/CC-BY-SA
# are permitted in release v1. CC-BY-NC and above are warehouse-only.
CORPUS_POLICY = dict(
    max_licence_risk_v1          = 0.20,   # HARD CEILING for release v1
    max_licence_risk_warehouse   = 0.50,   # Warehouse-only ceiling (NC etc.)
    require_explicit_licence     = True,   # Reject sources with no licence
    require_source_validated     = True,   # Block pipeline on unvalidated sources
    synthetic_proportion_cap     = 0.033,  # Max 3.3% synthetic in any release
    holdout_categories           = ["L"],  # Never in any training split
    zero_risk_licence_ids        = [       # Licences allowed in release v1
        "pd", "cc0", "mit", "apache2", "bsd3", "psf",
        "cc-by-4", "cc-by-3", "cc-by-sa-4", "cc-by-sa-3", "gpl2",
    ],
    warehouse_only_licence_ids   = [       # Allowed in warehouse, NOT in release v1
        "cc-by-nc-4", "cc-by-nc-sa", "cc-by-nd",
    ],
    hard_reject_licence_ids      = [       # Never allowed anywhere
        "ars", "tos-violation", "unknown",
    ],
    # Release v1 token targets per category (sum = 30B)
    category_token_targets_v1 = {
        "A": 12_000_000_000,   # Books — dominant category (40%)
        "B":  2_000_000_000,   # Educational
        "C":  3_000_000_000,   # Reasoning/STEM
        "D":  2_000_000_000,   # Conversations
        "E":  4_000_000_000,   # Technical docs (Wikipedia + official)
        "F":  2_000_000_000,   # Long-form articles
        "G":  3_000_000_000,   # Research papers (CC-BY only)
        "H":    500_000_000,   # COC synthetic (reviewed)
        "I":    500_000_000,   # Human cognition
        "J":    500_000_000,   # Retrieval material
        "K":    500_000_000,   # KG material
        "L":          0,       # HOLDOUT — never in training
    },
    target_v1_total              = 30_000_000_000,
    warehouse_zero_risk_ceiling  = 35_000_000_000,  # Honest max on zero-risk policy
    warehouse_permissive_ceiling = 80_000_000_000,  # With warehouse-only NC sources
)

# ═══════════════════════════════════════════════════════════════════
# DIRECTORY BOOTSTRAP
# ═══════════════════════════════════════════════════════════════════
def ensure_dirs():
    for d in [
        STORE_DIR, TOKENIZER_DIR, CHECKPOINT_DIR, INDEX_DIR, CACHE_DIR,
        UPLOAD_DIR, WORKSPACE_DIR, DATASET_DIR, WORKFLOW_DIR, LOG_DIR,
        ONNX_DIR, CORPUS_DIR, EVAL_BASELINE, EVAL_GATES,
        STORE_DIR / "onnx",
    ]:
        Path(d).mkdir(parents=True, exist_ok=True)
