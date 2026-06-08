"""Shared helpers for booklet figure scripts. Import-safe (no heavy work at import)."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "outputs" / "booklets"


def out_dir(area: str) -> Path:
    d = OUT_ROOT / area
    d.mkdir(parents=True, exist_ok=True)
    return d
