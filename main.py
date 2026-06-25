"""
CognitiveOC v3 — Main CLI Entry Point
======================================

Commands:
  python main.py train-tokenizer <corpus>          Train 48K SentencePiece tokenizer
  python main.py train-model <corpus>              Train 700M decoder
  python main.py prepare-corpus <input> <output>   Run corpus pipeline
  python main.py generate-corpus <output>          Generate synthetic corpus
  python main.py ingest <file>                     Index a document
  python main.py chat                              Interactive CLI chat
  python main.py ui                                Start web UI
  python main.py eval --all <val_corpus>           Run full evaluation suite
  python main.py eval --perplexity <val_corpus>    Perplexity only
  python main.py eval --tokenizer                  Tokenizer validation
  python main.py eval --retrieval                  Retrieval MRR/MR/NDCG
  python main.py eval --memory                     Memory recall accuracy
  python main.py eval --kg                         KG entity/triple F1
  python main.py eval --reasoning                  Reasoning accuracy
  python main.py status                            System status
  python main.py validate-corpus <corpus>          Validate a corpus file
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure project root in path
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from config import ensure_dirs
    ensure_dirs()
except ImportError:
    pass


# ═══════════════════════════════════════════════════════════════════
# Command handlers
# ═══════════════════════════════════════════════════════════════════

def cmd_train_tokenizer(args):
    from train.train_model import train_tokenizer
    result = train_tokenizer(
        corpus_path  = args.corpus,
        model_prefix = getattr(args, "prefix", None),
        vocab_size   = getattr(args, "vocab_size", None),
    )
    print(json.dumps(result, indent=2, default=str))


def cmd_train_model(args):
    from train.train_model import train
    result = train(
        corpus_path = args.corpus,
        steps       = getattr(args, "steps",     None),
        batch       = getattr(args, "batch",     None),
        accum       = getattr(args, "accum",     None),
        lr          = getattr(args, "lr",        None),
        precision   = getattr(args, "precision", None),
        seed        = getattr(args, "seed",      None),
        resume      = not getattr(args, "no_resume", False),
    )
    print(json.dumps(result, indent=2, default=str))


def cmd_prepare_corpus(args):
    from data.pipeline import prepare_corpus
    result = prepare_corpus(
        input_path  = args.input,
        output_dir  = args.output,
        train_ratio = getattr(args, "train", 0.90),
        val_ratio   = getattr(args, "val",   0.05),
        near_dedup  = getattr(args, "near_dedup", False),
    )
    print(json.dumps(result, indent=2))


def cmd_generate_corpus(args):
    from data.pipeline import generate_corpus
    out = generate_corpus(
        output_dir = args.output,
        n          = getattr(args, "n", 1000),
    )
    print(f"[corpus] Generated → {out}")


def cmd_validate_corpus(args):
    from data.pipeline import validate_corpus
    result = validate_corpus(args.corpus)
    for c in result.get("checks", []):
        mark = "PASS" if c["passed"] else "FAIL"
        print(f"  [{mark}] {c['name']}: {c.get('detail','')}")
    print(f"\nOVERALL: {'PASS' if result['passed'] else 'FAIL'}  "
          f"paragraphs={result.get('paragraphs','?')}")


def cmd_ingest(args):
    from engine import Engine
    eng    = Engine()
    result = eng.ingest(args.file)
    print(json.dumps(result, indent=2))


def cmd_chat(args):
    """Interactive CLI chat session."""
    from engine import Engine
    print("CognitiveOC v3 — Interactive Chat")
    print("Type 'exit' or 'quit' to stop.\n")
    eng     = Engine()
    session = getattr(args, "session", "cli_default")
    while True:
        try:
            msg = input("you> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break
        if not msg:
            continue
        if msg.lower() in ("exit", "quit", "q"):
            break
        result = eng.process(msg, session=session)
        print(f"oc>  {result.text}\n")


def cmd_ui(args):
    from ui.app import serve
    port = getattr(args, "port", 8765)
    print(f"[ui] Starting web UI at http://127.0.0.1:{port}")
    serve(port=port)


def cmd_eval(args):
    from eval.run_suite import (
        run_all, eval_perplexity, eval_tokenizer,
        eval_retrieval, eval_memory, eval_kg,
        eval_reasoning, eval_emotion,
    )

    if getattr(args, "all", False):
        corpus = getattr(args, "corpus", "data/corpus/v1/split/val.txt")
        run_all(corpus)
        return

    if getattr(args, "perplexity", False):
        corpus = getattr(args, "corpus", "data/corpus/v1/split/val.txt")
        r = eval_perplexity(corpus)
        print(f"val_loss={r.get('val_loss')}  "
              f"perplexity={r.get('perplexity')}  "
              f"rating={r.get('rating')}")
        return

    if getattr(args, "tokenizer", False):
        r = eval_tokenizer()
        for c in r.get("checks", []):
            print(f"  [{'PASS' if c['passed'] else 'FAIL'}] "
                  f"{c['name']}: {c.get('detail','')}")
        print(f"Fertility rating: {r.get('rating_fertility','N/A')}")
        return

    if getattr(args, "retrieval", False):
        r = eval_retrieval()
        print(f"MRR={r.get('mrr')}  MR={r.get('mr')}  "
              f"NDCG={r.get('ndcg')}  Hit@10={r.get('hit_k')}  "
              f"rating_mrr={r.get('rating_mrr')}")
        return

    if getattr(args, "memory", False):
        r = eval_memory()
        print(f"recall_accuracy={r.get('recall_accuracy')}  "
              f"rating={r.get('rating')}")
        return

    if getattr(args, "kg", False):
        r = eval_kg()
        print(f"entity_f1={r.get('entity_f1')}  "
              f"triple_f1={r.get('triple_f1')}  "
              f"rating={r.get('rating_entity')}")
        return

    if getattr(args, "reasoning", False):
        r = eval_reasoning()
        print(f"accuracy={r.get('accuracy')}  rating={r.get('rating')}")
        return

    if getattr(args, "emotion", False):
        r = eval_emotion()
        print(f"macro_f1={r.get('macro_f1')}  rating={r.get('rating')}")
        return

    # Default: run all
    corpus = getattr(args, "corpus", "data/corpus/v1/split/val.txt")
    run_all(corpus)


def cmd_status(args):
    from engine import Engine
    eng  = Engine()
    snap = eng.status()
    print(json.dumps(snap, indent=2, default=str))


def cmd_gate(args):
    from eval.run_suite import check_gate
    result = check_gate(args.phase)
    print(f"Phase {args.phase} gate: "
          f"{'PASS' if result['passed'] else 'FAIL'}  "
          f"({result['n_pass']}/{result['n_total']})")
    for c in result.get("checks", []):
        print(f"  [{'PASS' if c['passed'] else 'FAIL'}] "
              f"{c['metric']} {c['op']} {c['threshold']} → {c.get('detail')}")


# ═══════════════════════════════════════════════════════════════════
# CLI parser
# ═══════════════════════════════════════════════════════════════════

def cmd_corpus(args):
    """Route all 'corpus' subcommands to corpus/cli.py."""
    from corpus.cli import run_corpus_command
    run_corpus_command(args)


def cmd_training_ledger(args):
    from train.training_ledger import print_ledger_summary
    print_ledger_summary()


def cmd_shard_status(args):
    from train.shard_tracker import print_shard_status
    print_shard_status()


def cmd_resume_verify(args):
    from train.resume_guard import ResumeGuard, ResumeGuardError
    import sys
    release_id = getattr(args, "release_id", None) or "v1"
    resume     = not getattr(args, "fresh", False)
    operator   = getattr(args, "operator", "system")
    try:
        g = ResumeGuard(release_id=release_id, resume=resume, operator=operator)
        g.run()
    except ResumeGuardError as e:
        print("Resume guard failed: " + str(e), file=sys.stderr)
        sys.exit(1)


def cmd_provenance_report(args):
    from train.provenance import generate_report
    generate_report()


def cmd_lock_release(args):
    from release.lock import lock_release, ReleaseLockError
    import sys
    try:
        lock_release(args.release_id, operator=args.operator,
                     notes=getattr(args, "notes", ""))
    except ReleaseLockError as e:
        print("Lock failed: " + str(e), file=sys.stderr)
        sys.exit(1)


def cmd_unlock_release(args):
    from release.lock import break_lock
    reason = getattr(args, "reason", "admin override")
    break_lock(args.release_id, operator=args.operator, reason=reason)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cognitiveoc",
        description="CognitiveOC v3 — Cognitive Orchestration Core",
    )
    sub = p.add_subparsers(dest="command")

    # train-tokenizer
    pt = sub.add_parser("train-tokenizer")
    pt.add_argument("corpus")
    pt.add_argument("--prefix",     default=None)
    pt.add_argument("--vocab-size", type=int, default=None)

    # train-model
    pm = sub.add_parser("train-model")
    pm.add_argument("corpus")
    pm.add_argument("--steps",     type=int,   default=None)
    pm.add_argument("--batch",     type=int,   default=None)
    pm.add_argument("--accum",     type=int,   default=None)
    pm.add_argument("--lr",        type=float, default=None)
    pm.add_argument("--precision", choices=["bf16","fp32"], default=None)
    pm.add_argument("--seed",      type=int,   default=None)
    pm.add_argument("--no-resume", action="store_true")

    # prepare-corpus
    pc = sub.add_parser("prepare-corpus")
    pc.add_argument("input")
    pc.add_argument("output")
    pc.add_argument("--train",      type=float, default=0.90)
    pc.add_argument("--val",        type=float, default=0.05)
    pc.add_argument("--near-dedup", action="store_true")

    # generate-corpus
    pg = sub.add_parser("generate-corpus")
    pg.add_argument("output")
    pg.add_argument("--n", type=int, default=1000)

    # validate-corpus
    pv = sub.add_parser("validate-corpus")
    pv.add_argument("corpus")

    # ingest
    pi = sub.add_parser("ingest")
    pi.add_argument("file")

    # chat
    pch = sub.add_parser("chat")
    pch.add_argument("--session", default="cli_default")

    # ui
    pui = sub.add_parser("ui")
    pui.add_argument("--port", type=int, default=8765)

    # eval
    pe = sub.add_parser("eval")
    pe.add_argument("corpus", nargs="?",
                    default="data/corpus/v1/split/val.txt")
    pe.add_argument("--all",        action="store_true")
    pe.add_argument("--perplexity", action="store_true")
    pe.add_argument("--tokenizer",  action="store_true")
    pe.add_argument("--retrieval",  action="store_true")
    pe.add_argument("--memory",     action="store_true")
    pe.add_argument("--kg",         action="store_true")
    pe.add_argument("--reasoning",  action="store_true")
    pe.add_argument("--emotion",    action="store_true")

    # status
    sub.add_parser("status")

    # gate
    pgat = sub.add_parser("gate")
    pgat.add_argument("phase", type=int)

    # corpus — governed corpus engineering subsystem
    from corpus.cli import build_corpus_parser
    build_corpus_parser(sub)

    # training-ledger
    sub.add_parser("training-ledger", help="Print training session ledger")

    # shard-status
    sub.add_parser("shard-status", help="Print shard tracker status")

    # resume-verify
    prv = sub.add_parser("resume-verify", help="Run resume guard pre-flight check")
    prv.add_argument("release_id", help="Release version, e.g. v1")
    prv.add_argument("--fresh",    action="store_true", help="Fresh start (not a resume)")
    prv.add_argument("--operator", default="system")

    # provenance-report
    sub.add_parser("provenance-report", help="Print training provenance report")

    # lock-release
    plr = sub.add_parser("lock-release", help="Lock a signed release for training")
    plr.add_argument("release_id")
    plr.add_argument("--operator", required=True)
    plr.add_argument("--notes",    default="")

    # unlock-release (admin)
    pur = sub.add_parser("unlock-release", help="Admin: break a release lock")
    pur.add_argument("release_id")
    pur.add_argument("--operator", required=True)
    pur.add_argument("--reason",   required=True)

    return p


def main():
    parser  = build_parser()
    args    = parser.parse_args()
    command = args.command

    handlers = {
        "train-tokenizer": cmd_train_tokenizer,
        "train-model":     cmd_train_model,
        "prepare-corpus":  cmd_prepare_corpus,
        "generate-corpus": cmd_generate_corpus,
        "validate-corpus": cmd_validate_corpus,
        "ingest":          cmd_ingest,
        "chat":            cmd_chat,
        "ui":              cmd_ui,
        "eval":            cmd_eval,
        "status":          cmd_status,
        "gate":            cmd_gate,
        "corpus":          cmd_corpus,
        "training-ledger":  cmd_training_ledger,
        "shard-status":     cmd_shard_status,
        "resume-verify":    cmd_resume_verify,
        "provenance-report":cmd_provenance_report,
        "lock-release":     cmd_lock_release,
        "unlock-release":   cmd_unlock_release,
    }

    handler = handlers.get(command)
    if not handler:
        parser.print_help()
        sys.exit(0)

    try:
        handler(args)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
