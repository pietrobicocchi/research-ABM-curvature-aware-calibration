"""Shared plotting style for the project's figures.

A unified palette and rcParams. Cycles:
    - SEQ: viridis-derived sequential (use for ordered quantities, eg. eigenvals)
    - DIV: diverging cool->warm (use for signed quantities, eg. eigenvectors)
    - QUAL: qualitative for categorical labels (regimes, trader types)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# Palette: clean, journal-friendly. Hand-picked to be print-safe and
# distinguishable in CVD simulation.
QUAL = ["#1f3a93", "#c0392b", "#27ae60", "#e67e22", "#8e44ad", "#16a085", "#7f8c8d"]
SEQ = "viridis"
DIV = "RdBu_r"

# Regime colors used in the gallery / phase portrait scripts.
REGIME = {
    "fundamental": "#3498db",
    "periodic":    "#f39c12",
    "chaotic":     "#c0392b",
}


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor":   "white",
        "axes.edgecolor":   "#2c3e50",
        "axes.labelcolor":  "#2c3e50",
        "xtick.color":      "#2c3e50",
        "ytick.color":      "#2c3e50",
        "axes.titlesize":   12,
        "axes.titleweight": "bold",
        "axes.labelsize":   10,
        "legend.fontsize":  9,
        "font.size":        10,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":        True,
        "grid.alpha":       0.25,
        "grid.linestyle":   "--",
        "lines.linewidth":  1.6,
        "figure.dpi":       100,
        "savefig.dpi":      140,
        "savefig.bbox":     "tight",
        "axes.prop_cycle":  mpl.cycler(color=QUAL),
    })


def save(fig: mpl.figure.Figure, name: str, out_dir: str | Path = "outputs") -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / name
    fig.savefig(p)
    return p
