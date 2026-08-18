"""
CognitiveOC v3 — Memory Summarizer & KG Synchronisation
=========================================================

summarizer.py: Compress long memory entries into concise summaries.
               Uses extractive summarisation (sentence scoring) by default.
               Neural summarisation via summarization encoder when available.

sync.py equivalent: memory ↔ KG sync functions integrated here.
  memory_to_kg()  — extract KG triples from a memory entry
  enrich_recall() — annotate recalled memories with KG context

File: memory/summarizer.py
Used by: memory/memory.py (compress_long_memories), engine.py
"""

from __future__ import annotations

import re
from typing import Optional


# ── Extractive sentence scoring ───────────────────────────────────────

def _sentence_score(sentence: str, key_tokens: set[str]) -> float:
    """Score a sentence by keyword overlap and position bias."""
    words = re.findall(r"[a-z0-9]+", sentence.lower())
    if not words:
        return 0.0
    overlap = len(set(words) & key_tokens)
    return overlap / max(len(words), 1)


def summarize(texts: list[str],
              max_chars: int = 300,
              use_neural: bool = False) -> str:
    """Produce a concise summary of a list of text strings.

    Strategy:
      1. If use_neural=True and summarization encoder available → neural path.
      2. Extractive: score sentences by keyword overlap, pick top sentences
         that fit within max_chars.

    Args:
        texts:      List of strings to summarise (treated as one document).
        max_chars:  Target output length in characters.
        use_neural: Try neural summarisation via summarization encoder.

    Returns:
        Summary string (always shorter than the combined input).
    """
    if not texts:
        return ""

    combined = " ".join(t.strip() for t in texts if t.strip())
    if len(combined) <= max_chars:
        return combined

    # ── Neural path (optional) ────────────────────────────────────────
    if use_neural:
        try:
            result = _neural_summarize(combined, max_chars)
            if result:
                return result
        except Exception:
            pass

    # ── Extractive path ───────────────────────────────────────────────
    sentences = re.split(r"(?<=[.!?])\s+", combined)
    if not sentences:
        return combined[:max_chars]

    # Build key token set from all text
    all_words  = re.findall(r"[a-z0-9]+", combined.lower())
    from collections import Counter
    freq       = Counter(all_words)
    stopwords  = {
        "the","a","an","and","or","is","are","was","were","i","you","my",
        "me","do","of","to","in","it","that","be","have","has","had",
        "will","would","could","should","can","this","these","those",
    }
    key_tokens = {w for w, c in freq.most_common(40)
                  if w not in stopwords and len(w) > 2}

    scored = [
        (_sentence_score(s, key_tokens), i, s)
        for i, s in enumerate(sentences)
        if s.strip()
    ]
    scored.sort(key=lambda x: -x[0])

    # Pick top sentences that fit within max_chars, re-order by position
    selected, total = [], 0
    for score, idx, sent in scored:
        if total + len(sent) + 2 > max_chars:
            continue
        selected.append((idx, sent))
        total += len(sent) + 2
        if total >= max_chars * 0.9:
            break

    if not selected:
        return combined[:max_chars].rsplit(" ", 1)[0] + "…"

    selected.sort(key=lambda x: x[0])
    return " ".join(s for _, s in selected)


def summarize_session(turns: list[dict], max_chars: int = 400) -> str:
    """Summarise a conversation session from a list of turn dicts.

    Each turn dict: {"role": str, "content": str}
    """
    if not turns:
        return ""
    # Weight assistant turns slightly lower (user intent matters more)
    texts = []
    for turn in turns:
        role    = turn.get("role", "")
        content = turn.get("content", "").strip()
        if content:
            prefix = "User: " if role == "user" else "Assistant: "
            texts.append(prefix + content)
    return summarize(texts, max_chars=max_chars)


def _neural_summarize(text: str, max_chars: int) -> Optional[str]:
    """Neural summarisation via summarization encoder embedding + extraction.

    Uses sentence embeddings to find the most representative sentence(s)
    relative to the document centroid.
    """
    try:
        from encoder.hub import get_hub
        import numpy as np
        hub       = get_hub()
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        if len(sentences) < 3:
            return None
        vecs      = hub.encode("summarization", sentences)
        centroid  = vecs.mean(axis=0)
        centroid /= (np.linalg.norm(centroid) + 1e-9)
        sims      = vecs @ centroid
        ranked    = sorted(enumerate(sentences), key=lambda x: -sims[x[0]])
        result, total = [], 0
        for idx, sent in ranked:
            if total + len(sent) > max_chars:
                continue
            result.append((idx, sent))
            total += len(sent) + 1
        result.sort(key=lambda x: x[0])
        return " ".join(s for _, s in result) if result else None
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════
# Memory ↔ KG Synchronisation
# ═══════════════════════════════════════════════════════════════════

# Relation extraction patterns (extends baseline 12 to 20 patterns)
_TRIPLE_PATTERNS = [
    # is/are
    (re.compile(r"([A-Z][a-zA-Z0-9\s]{1,40})\s+is\s+(?:a|an|the)?\s*([a-zA-Z][^\.\,]{2,60})", re.I),
     "is"),
    # has/have
    (re.compile(r"([A-Z][a-zA-Z0-9\s]{1,40})\s+has\s+([a-zA-Z][^\.\,]{2,60})", re.I),
     "has"),
    # uses
    (re.compile(r"([A-Z][a-zA-Z0-9\s]{1,40})\s+uses?\s+([a-zA-Z][^\.\,]{2,60})", re.I),
     "uses"),
    # created / made / built
    (re.compile(r"([A-Z][a-zA-Z0-9\s]{1,40})\s+(?:created|made|built)\s+([a-zA-Z][^\.\,]{2,60})", re.I),
     "created"),
    # works at / works for
    (re.compile(r"([A-Z][a-zA-Z0-9\s]{1,30})\s+works?\s+(?:at|for)\s+([A-Z][a-zA-Z0-9\s]{1,40})", re.I),
     "works_at"),
    # located in / based in
    (re.compile(r"([A-Z][a-zA-Z0-9\s]{1,40})\s+(?:located|based)\s+in\s+([A-Z][a-zA-Z0-9\s]{1,40})", re.I),
     "located_in"),
    # belongs to / part of
    (re.compile(r"([A-Z][a-zA-Z0-9\s]{1,40})\s+(?:belongs\s+to|is\s+part\s+of)\s+([A-Z][a-zA-Z0-9\s]{1,40})", re.I),
     "part_of"),
    # contains
    (re.compile(r"([A-Z][a-zA-Z0-9\s]{1,40})\s+contains?\s+([a-zA-Z][^\.\,]{2,50})", re.I),
     "contains"),
    # means / defined as
    (re.compile(r"([A-Z][a-zA-Z0-9]{1,30})\s+means?\s+([a-zA-Z][^\.\,]{2,60})", re.I),
     "means"),
    # causes / leads to
    (re.compile(r"([a-zA-Z][a-zA-Z0-9\s]{1,40})\s+causes?\s+([a-zA-Z][^\.\,]{2,60})", re.I),
     "causes"),
    # depends on / requires
    (re.compile(r"([A-Z][a-zA-Z0-9\s]{1,40})\s+(?:depends\s+on|requires?)\s+([a-zA-Z][^\.\,]{2,50})", re.I),
     "depends_on"),
    # produces / generates
    (re.compile(r"([A-Z][a-zA-Z0-9\s]{1,40})\s+(?:produces?|generates?)\s+([a-zA-Z][^\.\,]{2,50})", re.I),
     "produces"),
    # contradicts
    (re.compile(r"([A-Z][a-zA-Z0-9\s]{1,40})\s+contradicts?\s+([a-zA-Z][^\.\,]{2,60})", re.I),
     "contradicts"),
    # precedes / comes before
    (re.compile(r"([A-Z][a-zA-Z0-9\s]{1,40})\s+(?:precedes?|comes?\s+before)\s+([A-Z][a-zA-Z0-9\s]{1,40})", re.I),
     "precedes"),
    # similar to
    (re.compile(r"([A-Z][a-zA-Z0-9\s]{1,40})\s+(?:is\s+)?similar\s+to\s+([A-Z][a-zA-Z0-9\s]{1,40})", re.I),
     "similar_to"),
    # instance of
    (re.compile(r"([A-Z][a-zA-Z0-9\s]{1,30})\s+is\s+an?\s+instance\s+of\s+([A-Z][a-zA-Z0-9\s]{1,40})", re.I),
     "instance_of"),
    # derived from
    (re.compile(r"([A-Z][a-zA-Z0-9\s]{1,40})\s+(?:is\s+)?derived\s+from\s+([A-Z][a-zA-Z0-9\s]{1,40})", re.I),
     "derived_from"),
    # enables
    (re.compile(r"([A-Z][a-zA-Z0-9\s]{1,40})\s+enables?\s+([a-zA-Z][^\.\,]{2,50})", re.I),
     "enables"),
    # inhibits / prevents
    (re.compile(r"([A-Z][a-zA-Z0-9\s]{1,40})\s+(?:inhibits?|prevents?)\s+([a-zA-Z][^\.\,]{2,50})", re.I),
     "inhibits"),
    # named
    (re.compile(r"([A-Z][a-zA-Z0-9\s]{1,30})\s+(?:is\s+)?named\s+([A-Z][a-zA-Z0-9\s]{1,30})", re.I),
     "named"),
]


def extract_triples(text: str) -> list[tuple[str, str, str]]:
    """Extract (subject, relation, object) triples from text.

    Uses 20 regex patterns covering most common relation types.
    Returns de-duplicated list of (subj, rel, obj) tuples.
    """
    triples: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()

    for pat, relation in _TRIPLE_PATTERNS:
        for match in pat.finditer(text):
            groups = match.groups()
            if len(groups) >= 2:
                subj = groups[0].strip()[:60]
                obj  = groups[1].strip()[:60]
                if len(subj) > 2 and len(obj) > 2:
                    triple = (subj.lower(), relation, obj.lower())
                    if triple not in seen:
                        seen.add(triple)
                        triples.append((subj, relation, obj))
    return triples


def memory_to_kg(memory_text: str,
                 source: str = "memory",
                 confidence: float = 0.7) -> list[dict]:
    """Extract KG triples from a memory entry and return as triple dicts.

    Returns:
        list of {"subject": str, "relation": str, "object": str,
                 "confidence": float, "source": str}
    """
    triples = extract_triples(memory_text)
    return [
        {
            "subject":    s,
            "relation":   r,
            "object":     o,
            "confidence": confidence,
            "source":     source,
        }
        for s, r, o in triples
    ]


def enrich_recall(memories: list[dict],
                  kg=None) -> list[dict]:
    """Annotate recalled memories with KG context.

    For each recalled memory, find KG facts about entities mentioned
    in the memory text and attach them as 'kg_context'.

    Args:
        memories: List of memory dicts (from CognitiveMemory.ranked_recall).
        kg:       KnowledgeGraph instance. If None, skips enrichment.

    Returns:
        Same list with 'kg_context' key added to each entry.
    """
    if kg is None:
        return [{**m, "kg_context": []} for m in memories]

    enriched = []
    for mem in memories:
        text      = mem.get("text", "")
        # Extract entities (capitalised words as simple heuristic)
        entities  = re.findall(r"\b[A-Z][a-zA-Z0-9]{2,}\b", text)
        kg_facts  = []
        for entity in set(entities[:5]):  # limit to 5 entities per memory
            try:
                facts = kg.query(entity=entity)
                for f in facts[:3]:
                    kg_facts.append(
                        f"{f.get('subject','')} {f.get('relation','')} {f.get('object','')}"
                    )
            except Exception:
                pass
        enriched.append({**mem, "kg_context": kg_facts[:5]})
    return enriched


def sync_memory_to_kg(memory, kg, limit: int = 500) -> int:
    """Batch-sync recent memory entries to the knowledge graph.

    Extracts triples from all non-archived memories and adds them to KG.
    Returns number of triples added.

    Args:
        memory: CognitiveMemory instance.
        kg:     KnowledgeGraph instance.
        limit:  Max memories to process per call.
    """
    rows = memory._conn.execute(
        "SELECT id, text FROM memories WHERE archived=0 ORDER BY id DESC LIMIT ?",
        (limit,)
    ).fetchall()
    added = 0
    for mid, text in rows:
        triples = memory_to_kg(text, source=f"memory:{mid}", confidence=0.65)
        for t in triples:
            try:
                kg.add_triple(
                    subject    = t["subject"],
                    relation   = t["relation"],
                    object_    = t["object"],
                    confidence = t["confidence"],
                    source     = t["source"],
                )
                added += 1
            except Exception:
                pass
    return added
