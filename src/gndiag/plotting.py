"""Shared figure style."""
from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

QUAL = ["#1f3a93", "#c0392b", "#27ae60", "#e67e22", "#8e44ad", "#16a085", "#7f8c8d"]
SEQ = "viridis"
DIV = "RdBu_r"

# One colour per entity, used consistently across figures.
ROLE = {
    "truth":    "#2c3e50",  # reference: exact Hessian, sampled posterior, full horizon
    "ggn":      "#1f3a93",  # the Gauss-Newton matrix
    "opg":      "#c0392b",
    "residual": "#e67e22",
    "alt":      "#8e44ad",  # second categorical member (e.g. point B)
    "ref":      "#7f8c8d",  # reference lines and guides
}
LS = {"truth": "-", "ggn": "-", "ggn_quad": "--", "ref": ":"}

COL_SINGLE = 3.4
COL_DOUBLE = 7.16

PARAM_LABELS = {"beta": r"$\beta$", "gamma": r"$\gamma$", "I0": r"$I_0$",
                "t_lock": r"$t_{\mathrm{lock}}$", "f_lock": r"$f_{\mathrm{lock}}$"}


def figsize(width="single", aspect=0.72):
    """(w, h) in inches; width is 'single', 'double' or a number."""
    w = {"single": COL_SINGLE, "double": COL_DOUBLE}.get(width, width)
    return (w, w * aspect)


def seq_colors(n, cmap=SEQ, lo=0.12, hi=0.88):
    """n colours from a sequential colormap, for ordered families."""
    return [plt.get_cmap(cmap)(t) for t in np.linspace(lo, hi, max(n, 1))]


def panel_label(ax, letter, dx=-30, dy=10):
    ax.annotate(f"({letter})", xy=(0, 1), xycoords="axes fraction",
                xytext=(dx, dy), textcoords="offset points",
                fontweight="bold", fontsize=11, ha="left", va="bottom",
                annotation_clip=False)


def apply_style() -> None:
    plt.rcParams.update({
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
