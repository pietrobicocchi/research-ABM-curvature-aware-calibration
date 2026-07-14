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

# --- Semantic colour map (manuscript house style) --------------------------
# Colour follows the ENTITY, never its position in a list. Every figure imports
# these by name so the same concept is the same colour throughout the paper.
# CVD-validated (adjacent ΔE ≥ 12); colour never carries meaning alone — pair with
# the linestyle convention below (solid = truth/method, dashed = GGN quadratic
# approximation, dotted = reference guide) and/or markers.
ROLE = {
    "truth":    "#2c3e50",  # reference truth: exact Hessian, sampled posterior, analytic AᵀWA, full horizon
    "ggn":      "#1f3a93",  # the GGN / our method
    "opg":      "#c0392b",  # gradient second moment (the foil) — RESERVED, never a generic series
    "residual": "#e67e22",  # R = H − G, "where it breaks"
    "alt":      "#8e44ad",  # second categorical member (e.g. straight-through surrogate)
    "ref":      "#7f8c8d",  # neutral reference lines / guides (dotted)
}
LS = {"truth": "-", "ggn": "-", "ggn_quad": "--", "ref": ":"}

# --- Manuscript widths (inches) -------------------------------------------
COL_SINGLE = 3.4          # single journal column
COL_DOUBLE = 7.16         # full text width


def figsize(width="single", aspect=0.72):
    """(w, h) in inches. width ∈ {'single','double'} or a number; h = aspect·w."""
    w = {"single": COL_SINGLE, "double": COL_DOUBLE}.get(width, width)
    return (w, w * aspect)


def seq_colors(n, cmap=SEQ, lo=0.12, hi=0.88):
    """n colours sampled from an ordered (sequential) colormap, avoiding the
    washed-out extremes. Use for ordered families (eigenvalue index, horizon,
    observation-design richness) — never for unordered categories."""
    import numpy as np
    return [plt.get_cmap(cmap)(t) for t in np.linspace(lo, hi, max(n, 1))]


def panel_label(ax, letter, dx=-30, dy=10):
    """Bold (a)/(b)… tag in the top-left margin — above the axes and left of the
    centered title, so it never overprints the title or the data."""
    ax.annotate(f"({letter})", xy=(0, 1), xycoords="axes fraction",
                xytext=(dx, dy), textcoords="offset points",
                fontweight="bold", fontsize=11, ha="left", va="bottom",
                annotation_clip=False)


def apply_style() -> None:
    plt.rcParams.update({
        # Serif / LaTeX-matched type. mathtext.fontset='cm' gives true Computer
        # Modern for symbols; the serif family falls back to DejaVu Serif (always
        # bundled) when CMU / Latin Modern are not installed — no LaTeX required.
        "font.family":      "serif",
        "font.serif":       ["CMU Serif", "Latin Modern Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "mathtext.rm":      "serif",
        "figure.facecolor": "white",
        "axes.facecolor":   "white",
        "axes.edgecolor":   "#2c3e50",
        "axes.labelcolor":  "#2c3e50",
        "text.color":       "#2c3e50",
        "xtick.color":      "#2c3e50",
        "ytick.color":      "#2c3e50",
        "axes.titlesize":   11,
        "axes.titleweight": "bold",
        "axes.labelsize":   10,
        "legend.fontsize":  8,
        "legend.frameon":   True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.8",
        "font.size":        10,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":        True,
        "grid.alpha":       0.22,
        "grid.linestyle":   "--",
        "grid.linewidth":   0.6,
        "lines.linewidth":  1.7,
        "figure.dpi":       100,
        "savefig.dpi":      200,
        "savefig.bbox":     "tight",
        "axes.prop_cycle":  mpl.cycler(color=QUAL),
    })


def save(fig: mpl.figure.Figure, name: str, out_dir: str | Path = "outputs") -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / name
    fig.savefig(p)
    return p
