"""Camera-ready figure idiom for the AI4ABM 2026 paper.

Tufte-adjacent restraint: Computer-Modern type (so figure type *is* body type),
hairline rules, no grid, dropped top/right spines, direct labelling, a single
rank-ordered sequential ramp for the P eigenvalue lines, and exactly one accent
colour reserved for the thing each figure is about.

No titles, no captions, no legends-in-boxes — these figures are dropped into the
LaTeX source and captioned there.

LaTeX note: a full `text.usetex` pipeline is unavailable on the build machine
(TeX Live "basic" lacks type1cm and is not writable). We instead render through
matplotlib's mathtext with the Computer Modern fontset (`mathtext.fontset='cm'`,
serif body in `cmr10`), which is the same Latin-Modern/CM type family as the
camera-ready and needs no LaTeX subprocess. Flip `USETEX = True` if a full TeX
install becomes available.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

USETEX = False
OUT_DIR = Path("outputs/viz")

# ---------------------------------------------------------------- colour system
# One muted accent, reserved for the single thing each figure is about
# (stiff direction / noise floor / v_1 row). Deep claret.
ACCENT = "#8c2f39"
# A quiet secondary ink for the "everything else" line in two-way comparisons.
INK = "#222428"
MUTED = "#9aa0a6"

# Single-hue rank ramp for the P eigenvalue lines: dark+saturated for lambda_1
# (stiff), pale for lambda_P (sloppy). A single-hue progression, never a wheel.
_RANK = LinearSegmentedColormap.from_list(
    "rank_seq", ["#10243a", "#1f4e79", "#4a7fb0", "#9dbdd8", "#d8e6f0"]
)
# Single-hue white -> deep sequential ramp for |V| heatmaps in [0, 1].
SEQ = LinearSegmentedColormap.from_list(
    "white_ink", ["#ffffff", "#dbe5ee", "#9dbdd8", "#4a7fb0", "#1f4e79", "#10243a"]
)


def rank_colors(P: int) -> list:
    """P colours dark->pale, encoding stiff (lambda_1) -> sloppy (lambda_P)."""
    return [_RANK(t) for t in np.linspace(0.04, 0.92, P)]


# ----------------------------------------------------------------- rc machinery
def setup() -> None:
    base = "cmr10" if not USETEX else "Latin Modern Roman"
    plt.rcParams.update({
        "text.usetex": USETEX,
        "font.family": "serif",
        "font.serif": [base, "Latin Modern Roman", "CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "mathtext.rm": "serif",
        "axes.unicode_minus": False,
        "axes.formatter.use_mathtext": True,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#33363b",
        "axes.labelcolor": "#1a1c1f",
        "text.color": "#1a1c1f",
        "xtick.color": "#33363b",
        "ytick.color": "#33363b",
        "xtick.labelcolor": "#1a1c1f",
        "ytick.labelcolor": "#1a1c1f",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "axes.linewidth": 0.6,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "font.size": 9,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "lines.linewidth": 1.3,
        "lines.solid_capstyle": "round",
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


# Text-column measure (inches). Identical width across every figure so the
# small-multiple parallelism lands. FULL spans the text block; HALF is a column.
FULL = 6.9
HALF = 3.35


def rule(ax, x=None, y=None, **kw):
    """Thin reference rule (regime boundary / reference value)."""
    style = dict(color=MUTED, lw=0.7, ls=(0, (4, 3)), zorder=0)
    style.update(kw)
    if x is not None:
        ax.axvline(x, **style)
    if y is not None:
        ax.axhline(y, **style)


def direct_label(ax, x, y, text, color, **kw):
    """Place a label at the right margin of a line instead of a legend box."""
    style = dict(fontsize=7.5, va="center", ha="left", color=color,
                 clip_on=False)
    style.update(kw)
    return ax.text(x, y, text, **style)


def smallcaps(s: str) -> str:
    """Approximate small-caps annotation in mathtext."""
    return r"$\mathrm{%s}$" % s.replace(" ", r"\ ")


def save(fig, name: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not name.endswith(".pdf"):
        name = name + ".pdf"
    p = OUT_DIR / name
    fig.savefig(p)
    # Also a 300-dpi PNG for quick screen inspection (LaTeX uses the PDF).
    fig.savefig(p.with_suffix(".png"))
    plt.close(fig)
    return p
