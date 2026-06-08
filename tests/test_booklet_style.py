import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path

import pytest

from curvature_calib.viz import booklet_style


@pytest.fixture(autouse=True)
def _restore_rcparams():
    yield
    mpl.rcdefaults()


def test_apply_sets_serif_and_grid():
    booklet_style.apply_booklet_style()
    assert plt.rcParams["axes.grid"] is True
    assert "serif" in plt.rcParams["font.family"][0] or plt.rcParams["font.family"] == ["serif"]


def test_save_vector_writes_pdf_and_png(tmp_path):
    booklet_style.apply_booklet_style()
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    paths = booklet_style.save_vector(fig, "demo", out_dir=tmp_path)
    assert (tmp_path / "demo.pdf").exists()
    assert (tmp_path / "demo.png").exists()
    assert paths["pdf"] == tmp_path / "demo.pdf"
    assert paths["png"] == tmp_path / "demo.png"
    plt.close(fig)
