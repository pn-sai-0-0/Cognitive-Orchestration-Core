from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Iterable, Iterator

SUPPORTED_EXTENSIONS = {".txt", ".md", ".json", ".jsonl", ".csv", ".xml", ".parquet"}


def iter_files(root: Path, extensions: set[str] | None = None) -> Iterator[Path]:
    exts = extensions or SUPPORTED_EXTENSIONS
    if not root.exists():
        return iter(())
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in exts:
            yield path


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict) -> None:
    ensure_parent(path)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def link_or_copy(src: Path, dst: Path) -> str:
    ensure_parent(dst)
    if dst.exists():
        return "exists"
    try:
        dst.hardlink_to(src)
        return "hardlink"
    except Exception:
        shutil.copy2(src, dst)
        return "copy"


def flatten_strings(value) -> list[str]:
    out: list[str] = []
    if isinstance(value, str):
        out.append(value)
    elif isinstance(value, dict):
        for v in value.values():
            out.extend(flatten_strings(v))
    elif isinstance(value, list):
        for v in value:
            out.extend(flatten_strings(v))
    return out


def read_textual_content(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".json":
        return "\n".join(flatten_strings(read_json(path)))
    if suffix == ".jsonl":
        parts = []
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    parts.extend(flatten_strings(json.loads(line)))
                except Exception:
                    parts.append(line)
        return "\n".join(parts)
    if suffix == ".csv":
        rows = []
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            for row in csv.reader(f):
                rows.append(" ".join(cell.strip() for cell in row if cell and cell.strip()))
        return "\n".join(rows)
    if suffix == ".xml":
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except Exception as e:  # pragma: no cover
            raise RuntimeError("pyarrow is required to read parquet") from e
        table = pq.read_table(path)
        rows = []
        for batch in table.to_batches(max_chunksize=128):
            rows.extend(flatten_strings(batch.to_pydict()))
        return "\n".join(rows)
    raise ValueError(f"Unsupported suffix: {suffix}")
