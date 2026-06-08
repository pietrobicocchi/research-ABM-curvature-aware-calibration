import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_booklet", ROOT / "scripts" / "booklets" / "build_booklet.py")
build_booklet = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build_booklet)


def test_load_captions_returns_dict():
    caps = build_booklet.load_captions()
    assert isinstance(caps, dict)
    assert all(isinstance(v, str) and v for v in caps.values())


def test_caption_for_known_key():
    caps = build_booklet.load_captions()
    assert "fig_01_bh_agents" in caps
