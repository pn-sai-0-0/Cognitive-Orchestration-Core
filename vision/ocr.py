"""
CognitiveOC v3 — OCR Module
=============================
Returns (location, text) pairs matching the parse_file() contract.
"""
from __future__ import annotations
from pathlib import Path


def analyze_image(path) -> list[tuple[str, str]]:
    """OCR an image file. Returns [(location, text), ...]."""
    p = Path(path)
    try:
        from PIL import Image
    except ImportError:
        return [('error', 'Image support requires: pip install pillow')]
    try:
        img  = Image.open(p)
        meta = f'[file={p.name} size={img.size[0]}x{img.size[1]} mode={img.mode}]'
        results: list[tuple[str, str]] = [('image_meta', meta)]
        try:
            import pytesseract
            text = pytesseract.image_to_string(img).strip()
            results.append(('image', text if text else '(no text detected by OCR)'))
        except ImportError:
            results.append(('image', '(OCR unavailable — install pytesseract + tesseract)'))
        except Exception as e:
            results.append(('error', f'OCR failed: {e}'))
        return results
    except Exception as e:
        return [('error', f'Cannot open image {p.name}: {e}')]


def image_to_text(path) -> str:
    """Convenience: return OCR text as a single string."""
    return '\n'.join(t for loc, t in analyze_image(path)
                     if loc not in ('error', 'image_meta'))
