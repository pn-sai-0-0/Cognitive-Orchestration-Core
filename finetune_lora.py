"""
QLoRA fine-tuning for CognitiveOC — proof-of-concept path.

This is deliberately separate from train/train_model.py. That pipeline
pretrains a new model from scratch (custom tokenizer, custom architecture);
this script fine-tunes an existing, already-pretrained open-weight model
with LoRA adapters. They are not interchangeable, and this one is the
fast path to something that actually answers questions.

Sized for an 8GB-class consumer GPU (matches the RTX 5060 8GB target this
project already documents) via 4-bit quantization (QLoRA). If you have
more VRAM, raise BASE_MODEL to a larger size and/or drop --load_in_4bit.

Requires, beyond requirements.txt: peft, accelerate (accelerate is
already in requirements.txt; add `pip install peft` — it wasn't in the
scanned import list because this script is new).

Usage:
    python finetune_lora.py --data my_data.jsonl --out ./adapters/run1
    python finetune_lora.py --data my_data.jsonl --out ./adapters/run1 --smoke-test

Data format (JSONL, one object per line):
    {"instruction": "What does the goal engine do?", "response": "..."}

The existing dataset-capture feature (README Section 4 — "Datasets are
captured automatically during chat sessions") already produces data in a
compatible shape under the dataset store; point --data at an export from
that if you want to fine-tune on your own real usage rather than a
hand-written file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BASE_MODEL = "microsoft/Phi-4-mini-instruct"  # ~3.8B, MIT license, fits 8GB in 4-bit
MAX_SEQ_LEN = 1024
PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: invalid JSON — {e}") from e
            if "instruction" not in row or "response" not in row:
                raise ValueError(
                    f"{path}:{line_no}: each row needs 'instruction' and 'response' keys, "
                    f"got {list(row.keys())}"
                )
            records.append(row)
    if not records:
        raise ValueError(f"{path}: no usable rows found")
    return records


def build_dataset(records: list[dict], tokenizer, max_len: int):
    from datasets import Dataset

    def _format(example: dict) -> dict:
        prompt = PROMPT_TEMPLATE.format(instruction=example["instruction"])
        full_text = prompt + example["response"] + tokenizer.eos_token
        tokenized = tokenizer(
            full_text, truncation=True, max_length=max_len, padding="max_length"
        )
        prompt_len = len(
            tokenizer(prompt, truncation=True, max_length=max_len)["input_ids"]
        )
        labels = list(tokenized["input_ids"])
        # mask the prompt portion so loss is only computed on the response
        for i in range(min(prompt_len, len(labels))):
            labels[i] = -100
        tokenized["labels"] = labels
        return tokenized

    ds = Dataset.from_list(records)
    return ds.map(_format, remove_columns=ds.column_names)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="JSONL file of instruction/response pairs")
    parser.add_argument("--out", type=Path, required=True, help="Directory to save the LoRA adapter")
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Validate the data pipeline (loading, tokenization, formatting) without "
        "downloading the base model or touching a GPU. Run this first.",
    )
    args = parser.parse_args()

    records = load_jsonl(args.data)
    print(f"Loaded {len(records)} training examples from {args.data}")

    if args.smoke_test:
        # Exercises the real formatting/masking logic in build_dataset() without
        # downloading anything. Earlier draft of this script used
        # AutoTokenizer.from_pretrained("gpt2") here, which looked network-free
        # but isn't — verified by actually running it: it fails with
        # "couldn't connect to huggingface.co" on a machine with no route there.
        # This mock tokenizer exercises the identical code path (whitespace
        # tokenization instead of BPE) so the smoke test is genuinely offline.
        class _MockTokenizer:
            eos_token = "<eos>"

            def __call__(self, text, truncation=True, max_length=128, padding=None):
                ids = list(range(len(text.split())))[:max_length]
                if padding == "max_length":
                    ids = ids + [0] * (max_length - len(ids))
                return {"input_ids": ids}

        tok = _MockTokenizer()
        ds = build_dataset(records, tok, MAX_SEQ_LEN)
        row = ds[0]
        n_masked = sum(1 for t in row["labels"] if t == -100)
        n_total = len(row["labels"])
        print(f"OK: formatted {len(ds)} examples. First row: {n_total} tokens, "
              f"{n_masked} masked (prompt), {n_total - n_masked} trainable (response).")
        print("Data pipeline is sound. Re-run without --smoke-test on a machine "
              "with a GPU and network access to the model hub to actually train.")
        return 0

    # Real path — needs a GPU and network access to the model hub.
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
    )

    print(f"Loading base model: {args.base_model} (4-bit)")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, quantization_config=bnb_config, device_map="auto"
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = build_dataset(records, tokenizer, MAX_SEQ_LEN)

    training_args = TrainingArguments(
        output_dir=str(args.out / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        bf16=True,
        logging_steps=5,
        save_strategy="epoch",
        report_to=[],
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
    trainer.train()

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.out))
    tokenizer.save_pretrained(str(args.out))
    print(f"Adapter saved to {args.out}")
    print(f"Load it later with: PeftModel.from_pretrained(base_model, '{args.out}')")
    return 0


if __name__ == "__main__":
    sys.exit(main())
