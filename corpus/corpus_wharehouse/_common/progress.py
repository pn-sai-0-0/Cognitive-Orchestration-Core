from __future__ import annotations

import json
from pathlib import Path


class ProgressTracker:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.items = set(json.loads(self.path.read_text(encoding="utf-8")))
        else:
            self.items = set()

    def contains(self, item: str) -> bool:
        return item in self.items

    def add(self, item: str) -> None:
        self.items.add(item)

    def flush(self) -> None:
        self.path.write_text(json.dumps(sorted(self.items), indent=2), encoding="utf-8")
