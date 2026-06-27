from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


class StageLogger:
    def __init__(self, stage: str, log_path: Path):
        self.stage = stage
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: str, **fields) -> None:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "stage": self.stage,
            "event": event,
            **fields,
        }
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        print(f"[{self.stage}] {event}: {fields}")
