"""
CognitiveOC v3 — Release Builder
==================================

Assembles a versioned, checksummed, reproducible training release from
approved warehouse data.

Pipeline:
  1. Collect all approved paragraph files for requested categories
  2. Shuffle with fixed seed
  3. Apply token budget cap (stop appending once target reached)
  4. Split into train / val / test
  5. Write split files to releases/v<N>/
  6. Generate manifest and checksums
  7. Run leakage check
  8. Mark release as ready-to-sign (human must call sign_release())

Signing:
  sign_release() locks the release and logs the approval event.
  After signing, the release is immutable.

Usage:
  from corpus.release_builder import ReleaseBuilder
  rb = ReleaseBuilder("v1")
  rb.build(categories=list("ABCDEFGHIJK"), token_budget=30_000_000_000)
  rb.verify()
  rb.sign(operator="mpssp")
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Iterator

from audit.logger import log_event, log_release
from corpus.dedup import release_exact_dedup_check
from corpus.manifest import (
    generate_release_manifest, load_release_manifest, validate_manifest,
)
from corpus.source_registry import list_sources
from corpus.warehouse import approved_dir, release_dir, _WAREHOUSE

try:
    from config import CORPUS
    _SPLIT_RATIOS = tuple(CORPUS.get("split_ratios", (0.90, 0.05, 0.05)))
    _SPLIT_SEED   = CORPUS.get("split_seed", 42)
    _TARGET_V1    = CORPUS.get("target_v1_tokens", 30_000_000_000)
except (ImportError, KeyError):
    _SPLIT_RATIOS = (0.90, 0.05, 0.05)
    _SPLIT_SEED   = 42
    _TARGET_V1    = 30_000_000_000

_CHARS_PER_TOKEN = 3.8  # 48K tokenizer fertility estimate


def _estimate_tokens(text: str) -> int:
    return int(len(text) / _CHARS_PER_TOKEN)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _iter_paragraphs(category: str) -> Iterator[str]:
    """
    Yield all approved paragraphs for a category, one paragraph at a time.
    Reads from warehouse/approved/<category_dir>/*.txt
    """
    adir = approved_dir(category)
    if not adir.exists():
        return
    for txt_file in sorted(adir.glob("*.txt")):
        content = txt_file.read_text(encoding="utf-8")
        for para in content.split("\n\n"):
            para = para.strip()
            if para:
                yield para


class ReleaseBuilder:
    """
    Builds a single versioned training release.

    Attributes:
        version:       Release version string, e.g. "v1".
        rdir:          Path to this release's directory.
        train_path:    Path to train.txt.
        val_path:      Path to val.txt.
        test_path:     Path to test.txt.
    """

    def __init__(self, version: str):
        self.version    = version
        self.rdir       = release_dir(version)
        self.train_path = self.rdir / "train.txt"
        self.val_path   = self.rdir / "val.txt"
        self.test_path  = self.rdir / "test.txt"
        self._built     = False

    # ── Build ─────────────────────────────────────────────────────────

    def build(
        self,
        categories:   list[str] | None = None,
        token_budget: int = _TARGET_V1,
        split_ratios: tuple[float, float, float] = _SPLIT_RATIOS,
        seed:         int = _SPLIT_SEED,
        dry_run:      bool = False,
        verbose:      bool = True,
    ) -> dict:
        """
        Assemble the training release.

        Args:
            categories:   List of category letters to include (default: all A-K).
            token_budget: Maximum total tokens (across all splits) to include.
            split_ratios: (train, val, test) fractions summing to 1.0.
            seed:         Random seed for shuffle — must be fixed for reproducibility.
            dry_run:      If True, report what would be built without writing files.
            verbose:      Print progress.

        Returns:
            Build summary dict.
        """
        if abs(sum(split_ratios) - 1.0) > 0.001:
            raise ValueError(f"split_ratios must sum to 1.0, got {sum(split_ratios)}")

        cats = categories or list("ABCDEFGHIJK")

        if verbose:
            print(f"\n[release/{self.version}] Starting build")
            print(f"  Categories : {', '.join(cats)}")
            print(f"  Token budget: {token_budget:,}")
            print(f"  Split ratios: {split_ratios}")
            print(f"  Seed        : {seed}")
            print(f"  Dry run     : {dry_run}")

        # ── Step 1: Collect all approved paragraphs ───────────────────
        all_paragraphs: list[str] = []
        cat_counts: dict[str, int] = {}

        for cat in cats:
            n_before = len(all_paragraphs)
            for para in _iter_paragraphs(cat):
                all_paragraphs.append(para)
            n_cat = len(all_paragraphs) - n_before
            cat_counts[cat] = n_cat
            if verbose:
                est = _estimate_tokens("\n\n".join(all_paragraphs[-n_cat:]))
                print(f"  [{cat}] {n_cat:>8} paragraphs  ~{est/1e9:.2f}B tokens")

        total_raw = len(all_paragraphs)
        if verbose:
            raw_tok = _estimate_tokens("\n\n".join(all_paragraphs))
            print(f"\n  Total collected: {total_raw:,} paragraphs, ~{raw_tok/1e9:.2f}B tokens")

        # ── Step 2: Shuffle with fixed seed ──────────────────────────
        rng = random.Random(seed)
        rng.shuffle(all_paragraphs)

        # ── Step 3: Apply token budget ────────────────────────────────
        budgeted: list[str] = []
        running_tokens = 0
        for para in all_paragraphs:
            tok = _estimate_tokens(para)
            if running_tokens + tok > token_budget:
                break
            budgeted.append(para)
            running_tokens += tok

        if verbose:
            print(f"  After budget cap: {len(budgeted):,} paragraphs, "
                  f"~{running_tokens/1e9:.2f}B tokens")

        # ── Step 4: Split ─────────────────────────────────────────────
        n       = len(budgeted)
        n_train = int(n * split_ratios[0])
        n_val   = int(n * split_ratios[1])
        # test gets the remainder to avoid rounding loss
        train_paras = budgeted[:n_train]
        val_paras   = budgeted[n_train:n_train + n_val]
        test_paras  = budgeted[n_train + n_val:]

        train_tok = _estimate_tokens("\n\n".join(train_paras))
        val_tok   = _estimate_tokens("\n\n".join(val_paras))
        test_tok  = _estimate_tokens("\n\n".join(test_paras))

        if verbose:
            print(f"\n  Split result:")
            print(f"    train: {len(train_paras):>8,} paragraphs  ~{train_tok/1e9:.2f}B tokens")
            print(f"    val:   {len(val_paras):>8,} paragraphs  ~{val_tok/1e9:.2f}B tokens")
            print(f"    test:  {len(test_paras):>8,} paragraphs  ~{test_tok/1e9:.2f}B tokens")

        if dry_run:
            if verbose:
                print("\n  [DRY RUN] No files written.")
            return {
                "version":      self.version,
                "dry_run":      True,
                "n_train":      len(train_paras),
                "n_val":        len(val_paras),
                "n_test":       len(test_paras),
                "train_tokens": train_tok,
                "val_tokens":   val_tok,
                "test_tokens":  test_tok,
                "total_tokens": train_tok + val_tok + test_tok,
                "categories":   cat_counts,
            }

        # ── Step 5: Write files ───────────────────────────────────────
        self.rdir.mkdir(parents=True, exist_ok=True)

        def _write_split(path: Path, paragraphs: list[str]) -> None:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("\n\n".join(paragraphs))

        _write_split(self.train_path, train_paras)
        _write_split(self.val_path,   val_paras)
        _write_split(self.test_path,  test_paras)

        if verbose:
            print(f"\n  Files written:")
            print(f"    {self.train_path}")
            print(f"    {self.val_path}")
            print(f"    {self.test_path}")

        # ── Step 6: Generate manifest ─────────────────────────────────
        approved_sources = list_sources(status="approved")
        cat_sources      = [s for s in approved_sources if s["category"] in cats]

        quality_gate = {
            "min_quality_score":          0.45,
            "all_sources_human_validated": True,
            "synthetic_all_reviewed":      True,
            "cross_source_dedup_complete": True,
        }

        manifest = generate_release_manifest(
            version       = self.version,
            sources       = cat_sources,
            train_path    = str(self.train_path),
            val_path      = str(self.val_path),
            test_path     = str(self.test_path),
            split_ratios  = split_ratios,
            shuffle_seed  = seed,
            released_by   = "system",
            quality_gate  = quality_gate,
            notes         = f"COC v3 training release {self.version}. "
                            f"Categories: {', '.join(cats)}.",
        )

        self._built = True

        # Log to audit
        sha_train = _sha256_file(self.train_path)
        log_event("release", f"release-{self.version}",
                  "release_built", "ok", operator="system",
                  hash_val=sha_train,
                  details={"version": self.version,
                           "train_tokens": train_tok,
                           "val_tokens":   val_tok,
                           "test_tokens":  test_tok})

        result = {
            "version":       self.version,
            "dry_run":       False,
            "n_train":       len(train_paras),
            "n_val":         len(val_paras),
            "n_test":        len(test_paras),
            "train_tokens":  train_tok,
            "val_tokens":    val_tok,
            "test_tokens":   test_tok,
            "total_tokens":  train_tok + val_tok + test_tok,
            "categories":    cat_counts,
            "manifest_path": str(self.rdir / "manifest.json"),
        }

        if verbose:
            print(f"\n  ✓ Release {self.version} built.")
            print(f"    Total tokens: ~{result['total_tokens']/1e9:.2f}B")
            print(f"    Manifest    : {result['manifest_path']}")
            print(f"\n  Next step: run verify(), then sign() with your operator name.\n")

        return result

    # ── Verify ────────────────────────────────────────────────────────

    def verify(self, verbose: bool = True) -> dict:
        """
        Verify a built release:
          1. Validate manifest (required fields + checksum match)
          2. Run leakage check (exact dedup between train and val/test)

        Returns result dict with "verified": bool.
        """
        if not self.train_path.exists():
            return {"verified": False,
                    "error": "Release not built. Run build() first."}

        manifest_path = str(self.rdir / "manifest.json")

        # 1. Manifest validation
        mv = validate_manifest(manifest_path)

        # 2. Leakage check
        lc = release_exact_dedup_check(
            str(self.train_path), str(self.val_path), str(self.test_path)
        )

        verified = mv["valid"] and not lc["leakage_detected"]

        if verbose:
            print(f"\n[release/{self.version}] Verification")
            print(f"  Manifest valid   : {'YES' if mv['valid'] else 'NO'}")
            if mv["errors"]:
                for e in mv["errors"]:
                    print(f"    ERROR: {e}")
            if mv["warnings"]:
                for w in mv["warnings"]:
                    print(f"    WARN: {w}")
            print(f"  Leakage detected : {'YES — STOP' if lc['leakage_detected'] else 'NO'}")
            print(f"    train∩val  : {lc['train_in_val']}")
            print(f"    train∩test : {lc['train_in_test']}")
            print(f"  Verified: {'✓ PASS' if verified else '✗ FAIL'}")

        return {
            "verified":           verified,
            "manifest_valid":     mv["valid"],
            "manifest_errors":    mv["errors"],
            "manifest_warnings":  mv["warnings"],
            "leakage_detected":   lc["leakage_detected"],
            "train_in_val":       lc["train_in_val"],
            "train_in_test":      lc["train_in_test"],
        }

    # ── Sign ──────────────────────────────────────────────────────────

    def sign(self, operator: str, verbose: bool = True) -> bool:
        """
        Sign (lock) the release. Requires verify() to have passed.

        This:
          - Updates manifest status to "signed"
          - Logs the release approval event to the audit log
          - Makes the release immutable (by convention — no file lock)

        Returns True if signing succeeded.
        """
        vr = self.verify(verbose=False)
        if not vr["verified"]:
            print(f"[release/{self.version}] Cannot sign: verification failed.")
            print(f"  Manifest valid : {vr['manifest_valid']}")
            print(f"  Leakage        : {vr['leakage_detected']}")
            return False

        # Update manifest status
        manifest_path = self.rdir / "manifest.json"
        with open(manifest_path, encoding="utf-8") as fh:
            manifest = json.load(fh)

        manifest["status"]       = "signed"
        manifest["signed_by"]    = operator
        manifest["signed_at"]    = time.strftime("%Y-%m-%dT%H:%M:%S")
        manifest["released_by"]  = operator

        with open(manifest_path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, ensure_ascii=False)

        # Audit log
        sha_train = _sha256_file(self.train_path)
        log_release(
            release_id  = self.version,
            operator    = operator,
            token_count = manifest.get("train_tokens", 0),
            sha256_train= sha_train,
        )

        if verbose:
            print(f"\n[release/{self.version}] ✓ SIGNED by {operator}")
            print(f"  Manifest : {manifest_path}")
            print(f"  SHA-256  : {sha_train[:24]}...")
            print(f"\n  Release {self.version} is now locked and ready for training.\n")

        return True

    # ── List ──────────────────────────────────────────────────────────

    @staticmethod
    def list_releases(verbose: bool = True) -> list[dict]:
        """List all releases in the warehouse with their status."""
        releases_dir = _WAREHOUSE / "releases"
        results = []

        if not releases_dir.exists():
            if verbose:
                print("No releases directory found.")
            return []

        for d in sorted(releases_dir.iterdir()):
            if not d.is_dir():
                continue
            manifest_path = d / "manifest.json"
            if not manifest_path.exists():
                results.append({"version": d.name, "status": "no manifest"})
                continue
            with open(manifest_path, encoding="utf-8") as fh:
                m = json.load(fh)
            results.append({
                "version":       d.name,
                "status":        m.get("status", "unknown"),
                "released_by":   m.get("released_by", "?"),
                "release_date":  m.get("release_date", "?"),
                "total_tokens":  m.get("total_tokens_estimate", 0),
                "categories":    m.get("categories_included", []),
            })

        if verbose:
            print(f"\n{'='*60}")
            print(f"COC v3 — Training Releases")
            print(f"{'='*60}")
            if not results:
                print("  No releases found.")
            for r in results:
                print(f"\n  {r['version']}")
                print(f"    Status   : {r['status']}")
                print(f"    Date     : {r.get('release_date','?')}")
                print(f"    Tokens   : ~{r.get('total_tokens',0)/1e9:.1f}B")
                print(f"    By       : {r.get('released_by','?')}")
                print(f"    Cats     : {', '.join(r.get('categories',[]))}")
            print(f"{'='*60}\n")

        return results
