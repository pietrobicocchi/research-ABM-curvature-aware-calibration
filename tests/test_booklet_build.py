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


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _make_pdf(path, label):
    fig, ax = plt.subplots()
    ax.set_title(label)
    fig.savefig(path)
    plt.close(fig)


def test_assemble_booklet_creates_pdf(tmp_path):
    fig_dir = tmp_path / "models"
    fig_dir.mkdir()
    _make_pdf(fig_dir / "fig_01_bh_agents.pdf", "one")
    _make_pdf(fig_dir / "fig_02_bh_equations.pdf", "two")
    out = tmp_path / "models_booklet.pdf"
    n = build_booklet.assemble_booklet(fig_dir, out, build_booklet.load_captions())
    assert out.exists()
    assert n == 2
