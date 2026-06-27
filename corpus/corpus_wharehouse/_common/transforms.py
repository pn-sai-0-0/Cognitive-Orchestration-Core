from __future__ import annotations

import math
import re
from collections import Counter

TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
OCR_NOISE_RE = re.compile(r"[|/_\\]{3,}|[Il1]{5,}|[0O]{5,}")
NONPRINT_RE = re.compile(r"[^\x09\x0A\x0D\x20-\x7E\u00A0-\u024F]")
WORD_RE = re.compile(r"[A-Za-z']+")

EN_STOPWORDS = {
    "the", "and", "of", "to", "in", "a", "is", "for", "that", "on", "with", "as", "by", "this", "from"
}
CATEGORY_KEYWORDS = {
    "Books": ["chapter", "novel", "author", "preface"],
    "Educational": ["lesson", "exercise", "student", "curriculum", "course"],
    "Research": ["abstract", "method", "results", "conclusion", "references"],
    "Technical": ["specification", "system", "architecture", "protocol", "deployment"],
    "Programming": ["python", "java", "function", "class", "algorithm", "code"],
    "Reasoning": ["therefore", "proof", "solve", "equation", "reasoning"],
    "Conversations": ["assistant", "user", "instruction", "response", "dialogue"],
    "Retrieval": ["question", "answer", "context", "evidence", "passage"],
    "Evaluation": ["benchmark", "accuracy", "multiple choice", "correct answer"],
    "Legal/Government": ["section", "act", "regulation", "court", "government"],
    "Knowledge Graph": ["entity", "relation", "triple", "ontology"],
    "Language Resources": ["synset", "frame", "lexicon", "lemma"],
    "Synthetic": ["synthetic", "generated", "artificial textbook"],
}


def clean_text(text: str) -> str:
    text = TAG_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = text.replace("\u00a0", " ").replace("\ufeff", " ")
    text = text.replace("“", '"').replace("”", '"').replace("’", "'")
    text = text.replace("–", "-").replace("—", "-")
    text = NONPRINT_RE.sub(" ", text)
    text = WS_RE.sub(" ", text).strip()
    return text


def shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text)
    total = len(text)
    return -sum((n / total) * math.log2(n / total) for n in counts.values())


def quality_metrics(text: str) -> dict:
    chars = len(text)
    alpha = sum(c.isalpha() for c in text)
    digits = sum(c.isdigit() for c in text)
    spaces = sum(c.isspace() for c in text)
    nonprint = len(NONPRINT_RE.findall(text))
    lines = [ln for ln in text.splitlines() if ln.strip()]
    repeated = 0
    if lines:
        counts = Counter(lines)
        repeated = sum(v for v in counts.values() if v > 1)
    ocr_noise = len(OCR_NOISE_RE.findall(text))
    entropy = shannon_entropy(text)
    alpha_ratio = alpha / chars if chars else 0.0
    digit_ratio = digits / chars if chars else 0.0
    nonprint_ratio = nonprint / chars if chars else 0.0
    repeated_line_ratio = repeated / max(1, len(lines))
    ocr_noise_ratio = ocr_noise / max(1, len(text.split()))
    score = 0.0
    score += min(1.0, alpha_ratio * 1.4)
    score += min(1.0, max(0.0, entropy / 5.0))
    score += max(0.0, 1.0 - min(1.0, nonprint_ratio * 20))
    score += max(0.0, 1.0 - min(1.0, repeated_line_ratio * 2))
    score += max(0.0, 1.0 - min(1.0, ocr_noise_ratio * 4))
    score = round(score / 5.0, 4)
    return {
        "chars": chars,
        "alpha_ratio": round(alpha_ratio, 4),
        "digit_ratio": round(digit_ratio, 4),
        "nonprintable_ratio": round(nonprint_ratio, 4),
        "repeated_line_ratio": round(repeated_line_ratio, 4),
        "ocr_noise_ratio": round(ocr_noise_ratio, 4),
        "entropy": round(entropy, 4),
        "quality_score": score,
    }


def detect_language(text: str) -> dict:
    words = [w.lower() for w in WORD_RE.findall(text)]
    if not words:
        return {"language": "unknown", "confidence": 0.0, "mixed": False, "unknown": True}
    ascii_letters = sum(all(ord(c) < 128 for c in w) for w in words)
    stop_hits = sum(1 for w in words if w in EN_STOPWORDS)
    confidence = min(1.0, 0.5 * (ascii_letters / len(words)) + 0.5 * min(1.0, stop_hits / 10.0))
    language = "en" if confidence >= 0.55 else "unknown"
    mixed = 0.15 < confidence < 0.55
    return {"language": language, "confidence": round(confidence, 4), "mixed": mixed, "unknown": language == "unknown"}


def classify_text(text: str) -> dict:
    low = text.lower()
    labels = []
    for label, keywords in CATEGORY_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in low)
        if hits:
            labels.append((label, hits))
    labels.sort(key=lambda x: (-x[1], x[0]))
    picked = [label for label, _ in labels[:3]] or ["Technical"]
    difficulty = "advanced" if any(k in low for k in ["theorem", "proof", "architecture", "formal"]) else "intermediate"
    if len(text.split()) < 120:
        difficulty = "basic"
    return {"labels": picked, "primary": picked[0], "difficulty": difficulty}
