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


from curvature_calib.viz import booklet_annotate


def test_callout_adds_annotation():
    booklet_style.apply_booklet_style()
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    n_before = len(ax.texts)
    booklet_annotate.callout(ax, (0.5, 0.5), "stiff", (0.7, 0.2))
    assert len(ax.texts) == n_before + 1
    plt.close(fig)


def test_tag_stiff_sloppy_adds_two_texts():
    booklet_style.apply_booklet_style()
    fig, ax = plt.subplots()
    ax.bar(range(5), [5, 4, 3, 2, 1])
    n_before = len(ax.texts)
    booklet_annotate.tag_stiff_sloppy(ax, stiff_xy=(0, 5), sloppy_xy=(4, 1))
    assert len(ax.texts) == n_before + 2
    plt.close(fig)
