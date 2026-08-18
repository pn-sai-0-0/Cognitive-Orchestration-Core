"""
CognitiveOC v3 — Corpus Scorer
================================

Multi-dimensional scoring for every paragraph in the corpus pipeline.

Three score axes:
  1. quality_score  (0.0–1.0) — raw text quality: length, richness, cleanliness
  2. category_score (0.0–1.0) — how well the paragraph serves the target category
  3. risk_score     (0.0–1.0) — licence risk from the source registry

Decision thresholds (from config.CORPUS):
  quality >= 0.70 AND risk <= 0.20 → auto-approve
  quality >= 0.45 AND risk <= 0.50 → human review queue
  quality < 0.45  OR  risk > 0.50  → auto-reject

Scoring is intentionally fast (no neural models) — all heuristic.
This keeps the pipeline CPU-bound and reproducible.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable

from data.pipeline import has_pii

# ── Config ────────────────────────────────────────────────────────────
try:
    from config import CORPUS
    _MIN_QUALITY   = CORPUS["min_quality_score"]
    _AUTO_APPROVE  = CORPUS["auto_approve_threshold"]
    _RISK_QUEUE    = CORPUS["risk_human_review"]
    _RISK_REJECT   = CORPUS["risk_reject"]
    _SYNTH_MIN     = CORPUS["synthetic_min_quality"]
except (ImportError, KeyError):
    _MIN_QUALITY  = 0.45
    _AUTO_APPROVE = 0.70
    _RISK_QUEUE   = 0.20
    _RISK_REJECT  = 0.50
    _SYNTH_MIN    = 0.70


@dataclass
class ParagraphScore:
    """Score container for a single paragraph."""
    quality_score:  float = 0.0
    category_score: float = 0.0
    risk_score:     float = 0.0

    # Quality sub-scores (for debugging/reporting)
    word_count_score:    float = 0.0
    lexical_richness:    float = 0.0
    sentence_variety:    float = 0.0
    printable_ratio:     float = 0.0
    pii_clean:           float = 0.0
    non_repetitive:      float = 0.0

    # Decision
    decision: str = "pending"   # auto_approve | human_review | auto_reject

    def to_dict(self) -> dict:
        return {
            "quality_score":   round(self.quality_score, 4),
            "category_score":  round(self.category_score, 4),
            "risk_score":      round(self.risk_score, 4),
            "decision":        self.decision,
            "sub_scores": {
                "word_count":      round(self.word_count_score, 4),
                "lexical_richness":round(self.lexical_richness, 4),
                "sentence_variety":round(self.sentence_variety, 4),
                "printable_ratio": round(self.printable_ratio, 4),
                "pii_clean":       round(self.pii_clean, 4),
                "non_repetitive":  round(self.non_repetitive, 4),
            },
        }


# ── Quality scoring ───────────────────────────────────────────────────

def score_quality(text: str) -> tuple[float, dict]:
    """
    Compute quality score for a paragraph.

    Returns (quality_score: float, sub_scores: dict).
    All sub-components are in [0.0, 1.0].
    """
    words     = text.split()
    n_words   = len(words)
    n_chars   = len(text)

    # 1. Word count score — target 20–500 words; penalise extremes
    if n_words < 8:
        wc_score = 0.0
    elif n_words < 20:
        wc_score = 0.5
    elif n_words <= 500:
        wc_score = 1.0
    else:
        # Long paragraphs: gentle decay
        wc_score = max(0.5, 1.0 - (n_words - 500) / 2000)

    # 2. Lexical richness — unique / total word ratio
    if n_words == 0:
        lr_score = 0.0
    else:
        lc_words = [w.lower() for w in words]
        lr_score = min(1.0, len(set(lc_words)) / n_words / 0.5)  # 0.5 is ideal min

    # 3. Sentence variety — avg words per sentence; target 10–30
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    if not sentences:
        sv_score = 0.0
    else:
        avg_words_per_sent = n_words / len(sentences)
        if 10 <= avg_words_per_sent <= 30:
            sv_score = 1.0
        elif avg_words_per_sent < 5:
            sv_score = 0.3
        elif avg_words_per_sent > 60:
            sv_score = 0.4
        else:
            sv_score = 0.7

    # 4. Printable ratio
    if n_chars == 0:
        pr_score = 0.0
    else:
        printable = sum(1 for c in text if c.isprintable() or c in "\n\t")
        pr_score  = printable / n_chars

    # 5. PII-clean — 1.0 if no PII detected
    pii_score = 0.0 if has_pii(text) else 1.0

    # 6. Non-repetitive — top-3 word types should not dominate
    if n_words < 3:
        rep_score = 0.0
    else:
        lc_words  = [w.lower() for w in words]
        top3      = sum(c for _, c in Counter(lc_words).most_common(3))
        top3_frac = top3 / n_words
        rep_score = max(0.0, 1.0 - top3_frac / 0.4)  # 0.4 = threshold

    # Weighted composite
    quality = (
        wc_score  * 0.20 +
        lr_score  * 0.20 +
        sv_score  * 0.15 +
        pr_score  * 0.15 +
        pii_score * 0.15 +
        rep_score * 0.15
    )

    sub = {
        "word_count":       wc_score,
        "lexical_richness":  lr_score,
        "sentence_variety":  sv_score,
        "printable_ratio":   pr_score,
        "pii_clean":         pii_score,
        "non_repetitive":    rep_score,
    }
    return round(quality, 4), sub


# ── Category scoring ──────────────────────────────────────────────────

# Vocabulary hints for each category's signal scoring
_REASONING_VOCAB = re.compile(
    r"\b(?:therefore|because|since|thus|if|then|implies|conclude|"
    r"reason|proof|derive|follows that|given that|suppose|assume|"
    r"consequently|hence|although|however|nevertheless)\b",
    re.IGNORECASE,
)
_INSTRUCTION_VOCAB = re.compile(
    r"\b(?:how to|steps?|first|second|next|finally|procedure|method|"
    r"example|solution|answer|question|explain|describe)\b",
    re.IGNORECASE,
)
_TECHNICAL_VOCAB = re.compile(
    r"\b(?:algorithm|function|class|module|API|parameter|variable|"
    r"equation|theorem|proof|model|architecture|network|gradient|"
    r"vector|matrix|tensor|hypothesis|experiment|dataset|entropy)\b",
    re.IGNORECASE,
)
_EMOTION_VOCAB = re.compile(
    r"\b(?:emotion|feeling|mood|affect|anxiety|joy|grief|anger|fear|"
    r"empathy|cognition|motivation|behaviour|psychology|mental|"
    r"emotional|cognitive|perception|attention|memory|learning)\b",
    re.IGNORECASE,
)
_RETRIEVAL_VOCAB = re.compile(
    r"\b(?:according to|source|evidence|passage|document|reference|"
    r"states?|mentions?|cites?|found in|based on|retrieval|query)\b",
    re.IGNORECASE,
)
_KG_VOCAB = re.compile(
    r"\b(?:is a|has a|consists? of|part of|related to|defined as|"
    r"entity|concept|relation|triple|property|instance of|taxonomy)\b",
    re.IGNORECASE,
)
_TEACHING_VOCAB = re.compile(
    r"\b(?:learn|understand|concept|explains?|teaches?|lesson|"
    r"student|curriculum|knowledge|skill|comprehension|pedagogy)\b",
    re.IGNORECASE,
)


def _vocab_density(text: str, pattern: re.Pattern) -> float:
    """Return density of pattern matches relative to word count."""
    words = text.split()
    if not words:
        return 0.0
    matches = len(pattern.findall(text))
    return min(1.0, matches / max(1, len(words) / 10))


_CATEGORY_SCORERS: dict[str, Callable[[str], float]] = {
    "A": lambda t: 0.8,                                    # Books: prose quality from quality_score
    "B": lambda t: _vocab_density(t, _INSTRUCTION_VOCAB),
    "C": lambda t: _vocab_density(t, _REASONING_VOCAB) * 0.6
                   + _vocab_density(t, _TECHNICAL_VOCAB) * 0.4,
    "D": lambda t: _vocab_density(t, _INSTRUCTION_VOCAB) * 0.7
                   + (0.3 if "<user>" in t else 0.0),
    "E": lambda t: _vocab_density(t, _TECHNICAL_VOCAB),
    "F": lambda t: 0.7,                                    # Long-form: rely on quality score
    "G": lambda t: _vocab_density(t, _TECHNICAL_VOCAB) * 0.5
                   + _vocab_density(t, _REASONING_VOCAB) * 0.5,
    "H": lambda t: 0.8,                                    # COC synthetic: scored at generation
    "I": lambda t: _vocab_density(t, _EMOTION_VOCAB) * 0.6
                   + _vocab_density(t, _TEACHING_VOCAB) * 0.4,
    "J": lambda t: _vocab_density(t, _RETRIEVAL_VOCAB),
    "K": lambda t: _vocab_density(t, _KG_VOCAB),
}


def score_category(text: str, category: str) -> float:
    """Compute category relevance score (0.0–1.0)."""
    fn = _CATEGORY_SCORERS.get(category, lambda t: 0.5)
    return round(min(1.0, max(0.0, fn(text))), 4)


# ── Risk scoring ──────────────────────────────────────────────────────

def score_risk(source_id: str) -> float:
    """
    Look up the licence risk score for a source from the source registry.

    Returns 0.8 (high risk) if the source is not found (unknown licence).
    """
    try:
        from corpus.source_registry import get_source
        record = get_source(source_id)
        if record:
            return float(record.get("licence_risk", 0.8))
    except Exception:
        pass
    return 0.8  # Unknown source = high risk by default


# ── Combined scorer ───────────────────────────────────────────────────

def score_paragraph(
    text:      str,
    source_id: str,
    category:  str,
    is_synthetic: bool = False,
) -> ParagraphScore:
    """
    Compute the full three-axis score for a single paragraph.

    Args:
        text:         Cleaned paragraph text.
        source_id:    Source identifier (used for risk lookup).
        category:     One of A-K.
        is_synthetic: If True, applies higher quality threshold.

    Returns:
        ParagraphScore with decision set.
    """
    quality, sub = score_quality(text)
    cat_score    = score_category(text, category)
    risk         = score_risk(source_id)

    ps = ParagraphScore(
        quality_score      = quality,
        category_score     = cat_score,
        risk_score         = risk,
        word_count_score   = sub["word_count"],
        lexical_richness   = sub["lexical_richness"],
        sentence_variety   = sub["sentence_variety"],
        printable_ratio    = sub["printable_ratio"],
        pii_clean          = sub["pii_clean"],
        non_repetitive     = sub["non_repetitive"],
    )

    # Decision
    min_q = _SYNTH_MIN if is_synthetic else _MIN_QUALITY
    if quality < min_q or risk > _RISK_REJECT:
        ps.decision = "auto_reject"
    elif quality >= _AUTO_APPROVE and risk <= _RISK_QUEUE:
        ps.decision = "auto_approve"
    else:
        ps.decision = "human_review"

    return ps


def batch_score(
    paragraphs:   list[str],
    source_id:    str,
    category:     str,
    is_synthetic: bool = False,
    verbose:      bool = False,
) -> list[ParagraphScore]:
    """
    Score a list of paragraphs in sequence.

    Returns a list of ParagraphScore objects in the same order as input.
    """
    scores = []
    n = len(paragraphs)
    for i, para in enumerate(paragraphs):
        s = score_paragraph(para, source_id, category, is_synthetic)
        scores.append(s)
        if verbose and i % 1000 == 0:
            print(f"  scored {i}/{n} paragraphs...")
    return scores


def score_summary(scores: list[ParagraphScore]) -> dict:
    """Return aggregate statistics for a batch of scores."""
    if not scores:
        return {}
    n     = len(scores)
    decisions: dict[str, int] = {"auto_approve": 0, "human_review": 0, "auto_reject": 0}
    total_q = 0.0
    total_r = 0.0
    for s in scores:
        decisions[s.decision] = decisions.get(s.decision, 0) + 1
        total_q += s.quality_score
        total_r += s.risk_score
    return {
        "total":          n,
        "auto_approve":   decisions.get("auto_approve", 0),
        "human_review":   decisions.get("human_review", 0),
        "auto_reject":    decisions.get("auto_reject", 0),
        "avg_quality":    round(total_q / n, 4),
        "avg_risk":       round(total_r / n, 4),
        "approve_rate":   round(decisions.get("auto_approve", 0) / n, 3),
        "reject_rate":    round(decisions.get("auto_reject", 0) / n, 3),
    }
