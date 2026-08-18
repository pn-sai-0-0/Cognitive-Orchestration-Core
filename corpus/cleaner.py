"""
CognitiveOC v3 — Corpus Cleaner
=================================

Category-specific text cleaners that extend data/pipeline.py::clean().

Each cleaner in this module:
  1. Calls the base clean() function from data/pipeline.py
  2. Applies category-specific boilerplate removal
  3. Returns cleaned paragraphs (list[str])

Supported categories and their cleaners:
  A — Books (Project Gutenberg, Standard Ebooks)
  B — Educational (OpenStax, CK-12, MIT OCW)
  C — Reasoning / STEM (NuminaMath, GSM8K, ARC, MATH, LogiQA)
  D — Conversations / Instruction (Dolly, OASST2, Alpaca, FLAN)
  E — Technical documentation (Wikipedia, PyTorch, HuggingFace docs)
  F — Long-form articles (ArXiv abstracts+intro, The Conversation)
  G — Research papers (ArXiv full CC-BY, ACL Anthology)
  H — COC synthetic data (produced by dataset/generator.py)
  I — Human cognition (OpenStax Psychology, CogSci papers)
  J — Retrieval material (MS MARCO, NQ, TriviaQA)
  K — Knowledge graph material (DBpedia, ConceptNet, Wikidata text)

All cleaners produce double-newline-separated UTF-8 text.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Callable

# Base pipeline cleaner
from data.pipeline import clean, quality_ok, split_paragraphs


# ── Gutenberg-specific boilerplate patterns ──────────────────────────
_GUTENBERG_HEADER = re.compile(
    r"^.*?(?:START OF (?:THIS|THE) PROJECT GUTENBERG|"
    r"PROJECT GUTENBERG EBOOK|"
    r"\*\*\* START)",
    re.DOTALL | re.IGNORECASE,
)
_GUTENBERG_FOOTER = re.compile(
    r"(?:END OF (?:THIS|THE) PROJECT GUTENBERG|"
    r"END OF PROJECT GUTENBERG|"
    r"\*\*\* END).*$",
    re.DOTALL | re.IGNORECASE,
)
_GUTENBERG_META = re.compile(
    r"(?:Title|Author|Release Date|Language|Character set encoding|"
    r"Produced by|Updated editions will replace).*?\n",
    re.IGNORECASE,
)


def clean_gutenberg(text: str) -> list[str]:
    """
    Clean Project Gutenberg plain-text files.

    Steps beyond base clean():
      - Strip Gutenberg header (everything before "START OF PROJECT GUTENBERG")
      - Strip Gutenberg footer (everything after "END OF PROJECT GUTENBERG")
      - Remove metadata lines (Title:, Author:, Release Date:, etc.)
      - Split into paragraphs and quality-filter
    """
    # Strip header
    m = _GUTENBERG_HEADER.search(text)
    if m:
        text = text[m.end():]

    # Strip footer
    m = _GUTENBERG_FOOTER.search(text)
    if m:
        text = text[:m.start()]

    # Remove metadata lines at top of remaining text
    text = _GUTENBERG_META.sub("", text)

    # Base clean
    text = clean(text)

    # Split and filter
    return [p for p in split_paragraphs(text) if quality_ok(p)]


def clean_standard_ebooks(text: str) -> list[str]:
    """
    Clean Standard Ebooks plain-text files.
    Standard Ebooks are cleaner than raw Gutenberg; needs less stripping.
    """
    # They don't have the Gutenberg header/footer, but may have SE-specific meta
    text = re.sub(r"^Produced by.*?\n", "", text, flags=re.MULTILINE | re.IGNORECASE)
    text = clean(text)
    return [p for p in split_paragraphs(text) if quality_ok(p)]


# ── Educational content cleaners ─────────────────────────────────────

_OPENSTAX_EXERCISE = re.compile(
    r"(?:Learning Objectives?|Key Terms?|Summary|Review Questions?|"
    r"Critical Thinking Questions?|Personal Application Questions?|"
    r"Try It|Check Your Understanding)\s*\n+",
    re.IGNORECASE,
)
_SECTION_NUMBER = re.compile(r"^\d+\.\d+\s+\w", re.MULTILINE)
_FIGURE_CAPTION = re.compile(r"^Figure\s+\d+\.\d+.*?$", re.MULTILINE | re.IGNORECASE)


def clean_openstax(text: str, extract_exercises: bool = False) -> list[str]:
    """
    Clean OpenStax textbook text.

    By default, strips exercise/review sections (they are better suited
    for Category D instruction data — call with extract_exercises=True
    to return them separately as instruction pairs).

    Returns cleaned prose paragraphs.
    """
    if not extract_exercises:
        # Find and remove exercise blocks
        text = _OPENSTAX_EXERCISE.sub("\n", text)

    # Remove figure captions
    text = _FIGURE_CAPTION.sub("", text)

    # Remove bare section numbers like "1.2 Introduction"
    text = _SECTION_NUMBER.sub("", text)

    text = clean(text)
    return [p for p in split_paragraphs(text) if quality_ok(p, min_words=12)]


def clean_ck12(text: str) -> list[str]:
    """Clean CK-12 Foundation plain text."""
    # Remove chapter/section headers that are standalone lines
    text = re.sub(r"^#+\s+.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\d+\.\s+.*$", "", text, flags=re.MULTILINE)
    text = clean(text)
    return [p for p in split_paragraphs(text) if quality_ok(p, min_words=10)]


# ── Reasoning / STEM cleaners ─────────────────────────────────────────

def clean_gsm8k(record: dict) -> str | None:
    """
    Convert a GSM8K record dict to a COC special-token reasoning string.

    Input: {"question": str, "answer": str}
    Output: "<reasoning>...</reasoning>" tagged string, or None if invalid.
    """
    q = record.get("question", "").strip()
    a = record.get("answer", "").strip()
    if not q or not a:
        return None
    # Strip numeric answer line if present (#### 42 style)
    a_clean = re.sub(r"####\s*\S+\s*$", "", a, flags=re.MULTILINE).strip()
    text = f"<reasoning>\nQuestion: {q}\n\nSolution:\n{a_clean}\n</reasoning>"
    return text if quality_ok(text, min_words=15) else None


def clean_math_dataset(record: dict) -> str | None:
    """
    Convert a MATH dataset record to a COC reasoning string.

    Input: {"problem": str, "solution": str, "level": str, "type": str}
    """
    prob = record.get("problem", "").strip()
    sol  = record.get("solution", "").strip()
    if not prob or not sol:
        return None
    # Strip LaTeX boxed answers at end
    sol_clean = re.sub(r"\\boxed\{[^}]+\}", "[answer]", sol)
    text = f"<reasoning>\nProblem: {prob}\n\nSolution:\n{sol_clean}\n</reasoning>"
    return text if quality_ok(text, min_words=15) else None


def clean_arc(record: dict) -> str | None:
    """
    Convert an ARC reasoning record to a QA string.

    Input: {"question": {"stem": str, "choices": [{"text": str, "label": str}]},
            "answerKey": str}
    """
    stem    = record.get("question", {}).get("stem", "").strip()
    choices = record.get("question", {}).get("choices", [])
    answer  = record.get("answerKey", "").strip()
    if not stem or not choices:
        return None

    choice_text = "\n".join(
        f"  {c.get('label', '?')}. {c.get('text', '')}"
        for c in choices
    )
    # Find answer text
    answer_text = next(
        (c.get("text", answer) for c in choices if c.get("label") == answer),
        answer,
    )
    text = (f"<reasoning>\nQuestion: {stem}\n\nChoices:\n{choice_text}\n\n"
            f"Answer: {answer_text}\n</reasoning>")
    return text if quality_ok(text, min_words=10) else None


# ── Conversation / instruction cleaners ──────────────────────────────

def clean_dolly(record: dict) -> str | None:
    """
    Convert a Dolly 15k record to COC user/assistant format.

    Input: {"instruction": str, "context": str, "response": str, "category": str}
    """
    instruction = record.get("instruction", "").strip()
    context     = record.get("context", "").strip()
    response    = record.get("response", "").strip()

    if not instruction or not response:
        return None
    if len(response.split()) < 10:
        return None

    if context:
        user_text = f"{instruction}\n\nContext: {context}"
    else:
        user_text = instruction

    return f"<user>{user_text}</user><assistant>{response}</assistant>"


def clean_oasst2(record: dict) -> str | None:
    """
    Convert an OASST2 message record to COC format.
    Only processes English, human-authored, high-quality messages.

    Input: {"text": str, "role": str, "lang": str, "quality_score": float,
            "parent_text": str, "parent_role": str}
    """
    if record.get("lang", "en") != "en":
        return None
    if record.get("quality_score", 0) < 0.5:
        return None
    if record.get("role") != "assistant":
        return None

    parent = record.get("parent_text", "").strip()
    text   = record.get("text", "").strip()

    if not parent or not text:
        return None
    if len(text.split()) < 15:
        return None

    return f"<user>{parent}</user><assistant>{text}</assistant>"


def clean_flan(record: dict) -> str | None:
    """
    Convert a FLAN collection record to COC format.
    Only uses the formatted text, not the raw template fields.

    Input: {"inputs": str, "targets": str}
    """
    inputs  = record.get("inputs", "").strip()
    targets = record.get("targets", "").strip()
    if not inputs or not targets:
        return None
    if len(targets.split()) < 5:
        return None
    return f"<user>{inputs}</user><assistant>{targets}</assistant>"


# ── Technical documentation cleaners ─────────────────────────────────

_WIKI_TEMPLATE      = re.compile(r"\{\{[^}]*\}\}", re.DOTALL)
_WIKI_REF           = re.compile(r"<ref[^>]*>.*?</ref>", re.DOTALL | re.IGNORECASE)
_WIKI_FILE_LINK     = re.compile(r"\[\[File:[^\]]+\]\]", re.IGNORECASE)
_WIKI_CATEGORY      = re.compile(r"\[\[Category:[^\]]+\]\]", re.IGNORECASE)
_WIKI_WIKILINK      = re.compile(r"\[\[(?:[^|\]]+\|)?([^\]]+)\]\]")
_WIKI_SECTION_HDR   = re.compile(r"^={2,6}\s+(.+?)\s+={2,6}\s*$", re.MULTILINE)
_STUB_LINE          = re.compile(r"^This (?:article|stub|section).{0,80}stub", re.IGNORECASE)


def clean_wikipedia(text: str, min_words: int = 20) -> list[str]:
    """
    Clean a Wikipedia article (wikitext or plain-text extract).

    - Remove templates {{...}}
    - Remove <ref> blocks
    - Remove [[File:]] links
    - Remove [[Category:]] tags
    - Resolve [[link|display]] → display text
    - Strip section headers (keep content)
    - Filter stub articles
    """
    if _STUB_LINE.search(text[:500]):
        return []

    text = _WIKI_TEMPLATE.sub(" ", text)
    text = _WIKI_REF.sub(" ", text)
    text = _WIKI_FILE_LINK.sub("", text)
    text = _WIKI_CATEGORY.sub("", text)
    text = _WIKI_WIKILINK.sub(r"\1", text)
    text = _WIKI_SECTION_HDR.sub("", text)
    text = clean(text)
    return [p for p in split_paragraphs(text) if quality_ok(p, min_words=min_words)]


def clean_pytorch_docs(text: str) -> list[str]:
    """
    Clean PyTorch / HuggingFace RST-based documentation.
    Keeps explanatory prose; strips API reference tables and code blocks.
    """
    # Remove RST directives
    text = re.sub(r"\.\. \w+::.*?(?=\n\n|\Z)", "", text, flags=re.DOTALL)
    # Remove code blocks (indented 4+ spaces after directive)
    text = re.sub(r"(?m)^( {4,}|\t).*$", "", text)
    # Remove :type:, :param:, :returns: fields
    text = re.sub(r"^:(?:type|param|returns?|rtype)\s+\w*:.*$", "",
                  text, flags=re.MULTILINE)
    text = clean(text)
    return [p for p in split_paragraphs(text) if quality_ok(p, min_words=12)]


# ── Research paper cleaners ──────────────────────────────────────────

_ARXIV_ABSTRACT_MARKER = re.compile(r"^Abstract\s*\n+", re.MULTILINE | re.IGNORECASE)
_ARXIV_REFERENCES      = re.compile(
    r"\n+References?\s*\n+.*$", re.DOTALL | re.IGNORECASE
)
_LATEX_EQUATION        = re.compile(r"\$[^$]+\$|\$\$[^$]+\$\$")
_CITE_TAG              = re.compile(r"\[[\d,\s;–-]+\]")
_SECTION_LABEL         = re.compile(r"^\d+(\.\d+)*\s+\w", re.MULTILINE)


def clean_arxiv_abstract(text: str) -> list[str]:
    """
    Extract and clean just the abstract from an ArXiv paper text.
    Use when full text is not available.
    """
    # Try to find abstract block
    m = _ARXIV_ABSTRACT_MARKER.search(text)
    if m:
        # Take next 2000 chars as abstract region
        abstract_region = text[m.end():m.end() + 2000]
        # Cut at next section header
        next_section = re.search(r"\n\n(?:\d+|[A-Z])\s*\.", abstract_region)
        if next_section:
            abstract_region = abstract_region[:next_section.start()]
        text = abstract_region

    text = _LATEX_EQUATION.sub("[equation]", text)
    text = _CITE_TAG.sub("", text)
    text = clean(text)
    return [p for p in split_paragraphs(text) if quality_ok(p, min_words=15)]


def clean_arxiv_full(text: str) -> list[str]:
    """
    Clean a full ArXiv paper (CC-BY).
    Removes references section, LaTeX equations, citation tags.
    Keeps all prose sections.
    """
    # Remove references section
    text = _ARXIV_REFERENCES.sub("", text)
    # Remove LaTeX equations (keep surrounding prose)
    text = _LATEX_EQUATION.sub("[equation]", text)
    text = _CITE_TAG.sub("", text)
    text = _SECTION_LABEL.sub("", text)
    text = clean(text)
    return [p for p in split_paragraphs(text) if quality_ok(p, min_words=12)]


def clean_acl(record: dict) -> list[str]:
    """
    Clean an ACL Anthology paper record.

    Input: {"title": str, "abstract": str, "text": str (optional)}
    """
    parts = []
    if record.get("title"):
        parts.append(record["title"].strip())
    if record.get("abstract"):
        parts.append(record["abstract"].strip())
    if record.get("text"):
        parts.append(record["text"].strip())

    combined = "\n\n".join(parts)
    return clean_arxiv_full(combined) if combined else []


# ── KG and retrieval cleaners ─────────────────────────────────────────

def clean_dbpedia(record: dict) -> str | None:
    """
    Convert a DBpedia abstract record to a KG-tagged string.

    Input: {"entity": str, "abstract": str}
    """
    entity   = record.get("entity", "").strip()
    abstract = record.get("abstract", "").strip()
    if not entity or not abstract:
        return None
    text = f"<kg>\n{entity}: {abstract}\n</kg>"
    return text if quality_ok(abstract, min_words=10) else None


def clean_conceptnet(record: dict) -> str | None:
    """
    Convert a ConceptNet edge to a readable sentence for KG training.

    Input: {"start": str, "rel": str, "end": str, "weight": float}
    """
    start  = record.get("start", "").strip().replace("_", " ")
    rel    = record.get("rel", "").strip().replace("/r/", "").replace("_", " ")
    end    = record.get("end", "").strip().replace("_", " ")
    weight = record.get("weight", 0.0)

    if not start or not rel or not end:
        return None
    if weight < 1.0:  # Only high-confidence edges
        return None

    text = f"<kg>\n{start} {rel.lower()} {end}.\n</kg>"
    return text


def clean_msmarco(record: dict) -> str | None:
    """
    Convert an MS MARCO passage record to a retrieval-tagged string.

    Input: {"query": str, "passages": [{"passage_text": str, "is_selected": int}]}
    """
    query    = record.get("query", "").strip()
    passages = record.get("passages", [])
    selected = [p["passage_text"] for p in passages if p.get("is_selected")]

    if not query or not selected:
        return None

    text = f"<retrieval>\nQuery: {query}\n\nPassage: {selected[0].strip()}\n</retrieval>"
    return text if quality_ok(selected[0], min_words=15) else None


def clean_nq(record: dict) -> str | None:
    """
    Convert a Natural Questions record to a retrieval-tagged string.

    Input: {"question": {"text": str}, "long_answer": str, "short_answers": [str]}
    """
    question     = record.get("question", {}).get("text", "").strip()
    long_answer  = record.get("long_answer", "").strip()
    short_answer = (record.get("short_answers") or [""])[0].strip()

    if not question or not long_answer:
        return None
    if len(long_answer.split()) < 20:
        return None

    text = (f"<retrieval>\nQuestion: {question}\n\n"
            f"Answer: {long_answer}\n\n"
            f"Short answer: {short_answer or 'See above.'}\n</retrieval>")
    return text


# ── Generic fallback cleaner ──────────────────────────────────────────

def clean_generic(text: str, min_words: int = 8) -> list[str]:
    """
    Apply base clean() + split + quality filter with no category-specific logic.
    Used for any source not covered by a specific cleaner.
    """
    text = clean(text)
    return [p for p in split_paragraphs(text) if quality_ok(p, min_words=min_words)]


# ── Dispatcher ────────────────────────────────────────────────────────

CATEGORY_CLEANERS: dict[str, Callable] = {
    "A": clean_gutenberg,
    "B": clean_openstax,
    "C": clean_generic,       # Overridden per-dataset (GSM8K, ARC, MATH each have specific cleaners)
    "D": clean_generic,       # Overridden per-dataset (Dolly, OASST2, FLAN)
    "E": clean_wikipedia,
    "F": clean_arxiv_abstract,
    "G": clean_arxiv_full,
    "H": clean_generic,       # COC synthetic — already in COC token format
    "I": clean_generic,       # Cognition material — generic prose cleaner
    "J": clean_generic,       # Retrieval — overridden per-dataset (MSMARCO, NQ)
    "K": clean_generic,       # KG — overridden per-dataset (DBpedia, ConceptNet)
}


def clean_for_category(text: str, category: str, **kwargs) -> list[str]:
    """
    Apply the appropriate cleaner for a given category.

    Args:
        text:     Raw text string.
        category: One of A-K.
        **kwargs: Passed to the category-specific cleaner.

    Returns:
        List of cleaned paragraphs.
    """
    fn = CATEGORY_CLEANERS.get(category, clean_generic)
    try:
        result = fn(text, **kwargs)
    except TypeError:
        result = fn(text)
    return result if isinstance(result, list) else [result] if result else []
