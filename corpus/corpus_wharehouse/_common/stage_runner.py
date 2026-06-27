from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .paths import DEFAULT_WAREHOUSE, WarehousePaths


def build_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--root", default=str(DEFAULT_WAREHOUSE))
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.add_argument("--incremental", action="store_true", default=True)
    p.add_argument("--no-incremental", dest="incremental", action="store_false")
    p.add_argument("--dry-run", action="store_true")
    return p


def prepare(root: str | Path):
    paths = WarehousePaths.from_root(root)
    paths.ensure()
    cfg = load_config(root)
    return paths, cfg
