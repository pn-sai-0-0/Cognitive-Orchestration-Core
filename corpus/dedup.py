"""
CognitiveOC v3 — Cross-Source Deduplication
=============================================

Implements two deduplication levels beyond what data/pipeline.py handles:

  Level 1 (within-source):   Already handled by pipeline.py::dedup().
                              Not re-implemented here.

  Level 2 (cross-source):    MinHash LSH over 5-gram shingles.
                              Identifies near-duplicate paragraphs that
                              originated from different source families
                              (e.g. same OpenStax passage appearing in
                              Gutenberg anthology AND Wikipedia).

  Level 3 (release-level):   Exact SHA-256 match verification across
                              the full assembled release before signing.

MinHash implementation:
  - 128 hash functions (good balance of speed vs accuracy)
  - 5-gram character shingles (robust to minor word-order changes)
  - Jaccard threshold: 0.85 (configurable, see CORPUS config)
  - Pure Python — no datasketch dependency required.
    Will use datasketch if available for ~5x speedup.

Usage:
  from corpus.dedup import CrossSourceDeduper
  deduper = CrossSourceDeduper()
  deduper.add_source("A-gutenberg", paragraphs_A)
  deduper.add_source("B-openstax", paragraphs_B)
  kept_A, kept_B = deduper.run()
"""

from __future__ import annotations

import hashlib
import random
import re
import struct
from collections import defaultdict
from pathlib import Path
from typing import Iterable

try:
    from config import CORPUS
    _NEAR_THRESHOLD  = CORPUS.get("dedup_near_threshold", 0.85)
    _EXACT_THRESHOLD = CORPUS.get("dedup_exact_threshold", 1.0)
except (ImportError, KeyError):
    _NEAR_THRESHOLD  = 0.85
    _EXACT_THRESHOLD = 1.0

_NUM_HASHES   = 128
_SHINGLE_SIZE = 5       # characters
_BAND_SIZE    = 4       # rows per LSH band → 32 bands
_NUM_BANDS    = _NUM_HASHES // _BAND_SIZE  # 32


# ── Shingle utilities ─────────────────────────────────────────────────

def _normalise_for_dedup(text: str) -> str:
    """Normalise text for dedup comparison: lowercase, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _shingles(text: str, k: int = _SHINGLE_SIZE) -> set[str]:
    """Produce character k-gram shingles from text."""
    n = text if isinstance(text, str) else str(text)
    if len(n) < k:
        return {n}
    return {n[i:i + k] for i in range(len(n) - k + 1)}


# ── MinHash ───────────────────────────────────────────────────────────

class MinHash:
    """
    Lightweight MinHash implementation (pure Python).
    Uses xxhash-style integer hashing via struct.

    For datasets > 1M paragraphs, install datasketch for C-backend speed:
      pip install datasketch
    """

    _HASH_PARAMS: list[tuple[int, int]] | None = None

    def __init__(self, num_hashes: int = _NUM_HASHES, seed: int = 42):
        self.num_hashes = num_hashes
        self._seed      = seed
        if MinHash._HASH_PARAMS is None:
            rng = random.Random(seed)
            p   = (1 << 61) - 1  # Mersenne prime
            MinHash._HASH_PARAMS = [
                (rng.randint(1, p - 1), rng.randint(0, p - 1))
                for _ in range(num_hashes)
            ]
        self.signature = [float("inf")] * num_hashes

    def update(self, shingles: Iterable[str]) -> None:
        """Add shingles to the signature."""
        p = (1 << 61) - 1
        for shingle in shingles:
            # Hash shingle to integer
            h = int(hashlib.md5(shingle.encode()).hexdigest(), 16)
            for i, (a, b) in enumerate(MinHash._HASH_PARAMS):
                v = (a * h + b) % p
                if v < self.signature[i]:
                    self.signature[i] = v

    @staticmethod
    def jaccard(sig1: list[float], sig2: list[float]) -> float:
        """Estimate Jaccard similarity from two signatures."""
        if len(sig1) != len(sig2):
            raise ValueError("Signatures must have the same length")
        equal = sum(1 for a, b in zip(sig1, sig2) if a == b)
        return equal / len(sig1)


def _try_datasketch(num_hashes: int):
    """Return a datasketch MinHash object if datasketch is installed, else None."""
    try:
        from datasketch import MinHash as DSMinHash, MinHashLSH as DSLSH
        return DSMinHash(num_perm=num_hashes)
    except ImportError:
        return None


# ── LSH Bucket Deduplicator ───────────────────────────────────────────

class CrossSourceDeduper:
    """
    Cross-source MinHash LSH deduplicator.

    Usage:
      deduper = CrossSourceDeduper(threshold=0.85)
      deduper.add_source("A-gutenberg", para_list_A)
      deduper.add_source("B-openstax", para_list_B)
      results = deduper.run()
      # results["A-gutenberg"] = list of (paragraph, kept: bool) tuples
    """

    def __init__(
        self,
        threshold:  float = _NEAR_THRESHOLD,
        num_hashes: int   = _NUM_HASHES,
        band_size:  int   = _BAND_SIZE,
    ):
        self.threshold  = threshold
        self.num_hashes = num_hashes
        self.band_size  = band_size
        self.num_bands  = num_hashes // band_size

        # {source_id: [(original_para, norm_para, signature)]}
        self._sources: dict[str, list[tuple[str, str, list[float]]]] = {}

    def add_source(self, source_id: str, paragraphs: list[str]) -> int:
        """
        Compute MinHash signatures for all paragraphs in a source.

        Returns the number of paragraphs added.
        """
        entries = []
        for para in paragraphs:
            norm = _normalise_for_dedup(para)
            if not norm:
                continue
            mh = MinHash(self.num_hashes)
            mh.update(_shingles(norm))
            entries.append((para, norm, mh.signature))
        self._sources[source_id] = entries
        return len(entries)

    def _build_lsh_buckets(self) -> dict[str, list[tuple[str, int]]]:
        """
        Build LSH band buckets across all sources.

        Returns {bucket_key: [(source_id, para_index), ...]}
        """
        buckets: dict[str, list[tuple[str, int]]] = defaultdict(list)
        for source_id, entries in self._sources.items():
            for i, (_, _, sig) in enumerate(entries):
                for band in range(self.num_bands):
                    start = band * self.band_size
                    end   = start + self.band_size
                    band_sig = tuple(sig[start:end])
                    key = f"{band}:{hash(band_sig)}"
                    buckets[key].append((source_id, i))
        return dict(buckets)

    def run(self, verbose: bool = False) -> dict[str, list[tuple[str, bool]]]:
        """
        Run cross-source deduplication.

        For each candidate pair in shared LSH buckets:
          - Compute exact Jaccard from signatures
          - If >= threshold: mark the LATER-added source paragraph as duplicate

        Priority rule: sources added first take precedence (first-wins).

        Returns:
          {source_id: [(paragraph, kept)]} for every source.
          kept=True means the paragraph should be included in the release.
          kept=False means it was found to be a near-duplicate of an earlier source.
        """
        buckets     = self._build_lsh_buckets()
        source_order = list(self._sources.keys())

        # duplicate_set: {(source_id, para_idx)} → marked for removal
        duplicate_set: set[tuple[str, int]] = set()

        checked_pairs: set[frozenset] = set()
        n_compared = 0
        n_duplicates = 0

        for bucket, candidates in buckets.items():
            if len(candidates) < 2:
                continue
            for i in range(len(candidates)):
                for j in range(i + 1, len(candidates)):
                    sid1, idx1 = candidates[i]
                    sid2, idx2 = candidates[j]

                    if sid1 == sid2:
                        continue  # within-source dedup handled by pipeline.py

                    pair = frozenset({(sid1, idx1), (sid2, idx2)})
                    if pair in checked_pairs:
                        continue
                    checked_pairs.add(pair)

                    sig1 = self._sources[sid1][idx1][2]
                    sig2 = self._sources[sid2][idx2][2]
                    jac  = MinHash.jaccard(sig1, sig2)
                    n_compared += 1

                    if jac >= self.threshold:
                        n_duplicates += 1
                        # Keep the one from the earlier-added source
                        order1 = source_order.index(sid1)
                        order2 = source_order.index(sid2)
                        if order1 <= order2:
                            duplicate_set.add((sid2, idx2))
                        else:
                            duplicate_set.add((sid1, idx1))

        if verbose:
            total = sum(len(v) for v in self._sources.values())
            print(f"[dedup] Cross-source: {total} paragraphs, "
                  f"{n_compared} pairs checked, {n_duplicates} duplicates found")

        # Build results
        results: dict[str, list[tuple[str, bool]]] = {}
        for sid, entries in self._sources.items():
            results[sid] = [
                (original, (sid, i) not in duplicate_set)
                for i, (original, _, _) in enumerate(entries)
            ]

        return results

    def stats(self, results: dict[str, list[tuple[str, bool]]]) -> dict:
        """Return deduplication statistics from run() results."""
        total = 0
        kept  = 0
        per_source: dict[str, dict] = {}
        for sid, items in results.items():
            n   = len(items)
            k   = sum(1 for _, keep in items if keep)
            total += n
            kept  += k
            per_source[sid] = {"total": n, "kept": k, "removed": n - k,
                                "retention_pct": round(k / n * 100, 1) if n else 0}
        return {
            "total_paragraphs": total,
            "kept":             kept,
            "removed":          total - kept,
            "retention_pct":    round(kept / total * 100, 1) if total else 0,
            "per_source":       per_source,
        }


# ── Release-level exact dedup check ──────────────────────────────────

def release_exact_dedup_check(
    train_path: str,
    val_path:   str,
    test_path:  str,
) -> dict:
    """
    Verify there are no exact-match paragraphs between train and val/test splits.

    This is the leakage prevention check run at Stage 9 (Split) before
    building a release.

    Args:
        train_path: Path to the training split file.
        val_path:   Path to validation split file.
        test_path:  Path to test split file.

    Returns:
        {
          "leakage_detected": bool,
          "train_in_val": int,   # paragraphs in val that also appear in train
          "train_in_test": int,  # paragraphs in test that also appear in train
        }
    """
    def _load_hashes(path: str) -> set[str]:
        hashes = set()
        text   = Path(path).read_text(encoding="utf-8")
        for para in re.split(r"\n{2,}", text):
            para = para.strip()
            if para:
                h = hashlib.sha256(para.lower().encode()).hexdigest()
                hashes.add(h)
        return hashes

    train_hashes = _load_hashes(train_path)
    val_hashes   = _load_hashes(val_path)
    test_hashes  = _load_hashes(test_path)

    leak_val  = len(train_hashes & val_hashes)
    leak_test = len(train_hashes & test_hashes)

    return {
        "leakage_detected": (leak_val > 0 or leak_test > 0),
        "train_in_val":     leak_val,
        "train_in_test":    leak_test,
        "train_size":       len(train_hashes),
        "val_size":         len(val_hashes),
        "test_size":        len(test_hashes),
    }
