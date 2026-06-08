import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts" / "booklets"
FIG_SCRIPTS = sorted(
    [p for p in (SCRIPT_DIR / "models").glob("b1_*.py")]
    + [p for p in (SCRIPT_DIR / "methodology").glob("b2_*.py")]
)


@pytest.mark.parametrize("path", FIG_SCRIPTS, ids=lambda p: p.stem)
def test_script_exposes_main_and_names(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert callable(mod.main)
    assert isinstance(mod.OUT_AREA, str) and mod.OUT_AREA in {"models", "methodology"}
    assert isinstance(mod.OUT_NAME, str) and mod.OUT_NAME.startswith("fig_")
