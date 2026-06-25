"""
CognitiveOC v3 — File Search Tool
Local file search across uploaded/corpus files by filename and content.
"""
from __future__ import annotations
from pathlib import Path

_TEXT_EXTS = {'.txt', '.md', '.csv', '.py', '.json', '.yaml', '.toml', '.log'}


def search(query: str, root=None, limit: int = 10) -> list[dict]:
    """Search files by name and content under root directory.

    Returns list of {"file": str, "match": "filename"|"content"}.
    """
    try:
        from config import UPLOAD_DIR
        default_root = UPLOAD_DIR
    except ImportError:
        default_root = Path('var/uploads')

    root = Path(root) if root else Path(str(default_root))
    hits: list[dict] = []
    if not root.exists():
        return hits

    q = query.lower()
    for p in root.rglob('*'):
        if len(hits) >= limit:
            break
        if not p.is_file():
            continue
        if q in p.name.lower():
            hits.append({'file': str(p), 'match': 'filename', 'name': p.name})
        elif p.suffix.lower() in _TEXT_EXTS:
            try:
                if q in p.read_text(encoding='utf-8', errors='ignore').lower():
                    hits.append({'file': str(p), 'match': 'content', 'name': p.name})
            except OSError:
                pass
    return hits


# Alias for engine.py dispatch
search_files = search
