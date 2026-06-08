"""Shared style for the thesis visualization booklets.

Journal-clean base (serif labels, muted palette, faint dashed grid) layered on
top of viz/style.py, plus a vector-friendly save helper. Annotation helpers live
in booklet_annotate.py.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

from curvature_calib.viz.style import DIV, QUAL, REGIME, SEQ  # re-export palette

__all__ = ["apply_booklet_style", "save_vector", "QUAL", "SEQ", "DIV", "REGIME"]


def apply_booklet_style() -> None:
    plt.rcParams.update({
        # vector-friendly, higher-DPI, serif: booklet-specific overrides vs style.py
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#2c3e50",
        "axes.labelcolor": "#2c3e50",
        "xtick.color": "#2c3e50",
        "ytick.color": "#2c3e50",
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Georgia"],
        "mathtext.fontset": "dejavuserif",
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "lines.linewidth": 1.6,
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,      # editable text in vector PDF
        "ps.fonttype": 42,
        "axes.prop_cycle": mpl.cycler(color=QUAL),
    })


def save_vector(fig: mpl.figure.Figure, name: str,
                out_dir: str | Path = "outputs/booklets") -> dict[str, Path]:
    """Save a figure as both vector PDF and raster PNG. `name` is stem only."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{name}.pdf"
    png = out_dir / f"{name}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    return {"pdf": pdf, "png": png}
