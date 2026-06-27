from __future__ import annotations

import hashlib
import re
from pathlib import Path

WORD_RE = re.compile(r"[A-Za-z0-9_]+", re.UNICODE)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tokenize(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def simhash(text: str) -> int:
    vec = [0] * 64
    tokens = tokenize(text)
    if not tokens:
        return 0
    for tok in tokens:
        h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
        for i in range(64):
            vec[i] += 1 if ((h >> i) & 1) else -1
    out = 0
    for i, v in enumerate(vec):
        if v >= 0:
            out |= (1 << i)
    return out


def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def word_shingles(text: str, k: int = 5) -> set[str]:
    toks = tokenize(text)
    if len(toks) < k:
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i:i+k]) for i in range(len(toks) - k + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))
