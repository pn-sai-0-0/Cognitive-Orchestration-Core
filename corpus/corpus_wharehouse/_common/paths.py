from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_WAREHOUSE = Path(os.environ.get(
    "COC_WAREHOUSE_DIR",
    r"D:\projects\CognitiveOC\Final_Versions\cognitiveoc_v3\corpus\corpus_wharehouse",
))


@dataclass(frozen=True)
class WarehousePaths:
    root: Path
    raw: Path
    validated: Path
    cleaned: Path
    normalized: Path
    deduplicated: Path
    filtered: Path
    classified: Path
    balanced: Path
    metadata: Path
    analytics: Path
    statistics: Path
    tokenizer_input: Path
    training: Path
    splits: Path
    pipeline_state: Path
    reports: Path
    quality: Path
    manifests: Path
    logs: Path

    @classmethod
    def from_root(cls, root: str | Path) -> "WarehousePaths":
        root = Path(root)
        return cls(
            root=root,
            raw=root / "raw",
            validated=root / "validated",
            cleaned=root / "cleaned",
            normalized=root / "normalized",
            deduplicated=root / "deduplicated",
            filtered=root / "filtered",
            classified=root / "classified",
            balanced=root / "balanced",
            metadata=root / "metadata",
            analytics=root / "analytics",
            statistics=root / "statistics",
            tokenizer_input=root / "tokenizer_input",
            training=root / "training",
            splits=root / "splits",
            pipeline_state=root / "pipeline_state",
            reports=root / "reports",
            quality=root / "quality",
            manifests=root / "manifests",
            logs=root / "logs",
        )

    def ensure(self) -> None:
        for p in (
            self.validated,
            self.cleaned,
            self.normalized,
            self.deduplicated,
            self.filtered,
            self.classified,
            self.balanced,
            self.metadata,
            self.analytics,
            self.statistics,
            self.tokenizer_input,
            self.training,
            self.splits,
            self.pipeline_state,
            self.reports,
            self.quality,
            self.manifests,
            self.logs,
        ):
            p.mkdir(parents=True, exist_ok=True)
