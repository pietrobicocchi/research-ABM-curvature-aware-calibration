"""Float64 switch and experiment configuration."""
from __future__ import annotations

from pathlib import Path

import jax
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"


def enable_x64() -> None:
    """Use float64 throughout; must run before any array is created."""
    jax.config.update("jax_enable_x64", True)


def load_config(name: str) -> dict:
    """Load configs/<name>.yaml; the model config it names is available under "model"."""
    cfg = yaml.safe_load((CONFIG_DIR / f"{name}.yaml").read_text())
    if "model" in cfg:
        cfg["model"] = yaml.safe_load((CONFIG_DIR / f"{cfg['model']}.yaml").read_text())
    return cfg
