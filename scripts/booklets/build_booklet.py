"""Assemble standalone booklet figures into captioned booklet PDFs."""
from __future__ import annotations

import argparse  # noqa: F401
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CAPTIONS_PATH = Path(__file__).resolve().parent / "captions.yaml"


def load_captions(path: Path = CAPTIONS_PATH) -> dict[str, str]:
    with open(path) as fh:
        data = yaml.safe_load(fh)
    return dict(data)
