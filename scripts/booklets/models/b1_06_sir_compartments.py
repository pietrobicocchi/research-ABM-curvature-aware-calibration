"""Booklet 1, Figure 6: SIR compartment schematic with lockdown (concept)."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from curvature_calib.viz.booklet_annotate import callout  # noqa: E402
from curvature_calib.viz.booklet_style import apply_booklet_style, save_vector  # noqa: E402
from curvature_calib.viz.style import REGIME  # noqa: E402
from scripts.booklets._figbase import out_dir  # noqa: E402

OUT_AREA = "models"
OUT_NAME = "fig_06_sir_compartments"

# ── Layout constants ──────────────────────────────────────────────────────────
_CIRCLE_R    = 0.70   # compartment circle radius
_CX          = [2.0, 5.0, 8.0]  # x centres: S, I, R
_CY          = 2.2               # shared y centre for all three circles

_ARROW_Y     = _CY               # arrows travel at the circle midline
_LABEL_Y     = _CY - 1.10       # full-name labels below circles
_BIG_FS      = 26                # large letter inside circle
_FULL_FS     = 9                 # compartment full-name fontsize

_LOCKBOX_XC  = 3.50              # lockdown box x-centre
_LOCKBOX_YC  = 4.05              # lockdown box y-centre
_LOCKBOX_W   = 1.50
_LOCKBOX_H   = 0.50

_CALLOUT_TIP = (3.50, _CY + 0.28)   # tip of callout arrow (onto S→I arrow)
_CALLOUT_TXT = (5.00, 0.42)          # callout text box centre


def _circle(ax, cx, cy, r, facecolor, edgecolor):
    """Draw a filled circle patch."""
    patch = mpatches.Circle(
        (cx, cy), r,
        facecolor=facecolor, edgecolor=edgecolor,
        linewidth=2.2, alpha=0.28, zorder=2,
    )
    ax.add_patch(patch)
    # Crisp edge ring (alpha=1)
    edge = mpatches.Circle(
        (cx, cy), r,
        facecolor="none", edgecolor=edgecolor,
        linewidth=2.2, zorder=3,
    )
    ax.add_patch(edge)


def _compartment_arrow(ax, x_start, x_end, y, color="#2c3e50"):
    """Horizontal arrow between two x-positions at height y."""
    ax.add_patch(mpatches.FancyArrowPatch(
        (x_start, y), (x_end, y),
        arrowstyle="-|>",
        mutation_scale=20,
        color=color,
        linewidth=1.8,
        zorder=4,
    ))


def main() -> None:
    apply_booklet_style()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)

    # ── 1. Compartment circles ────────────────────────────────────────────────
    _colors = {
        "S": (REGIME["fundamental"], REGIME["fundamental"]),
        "I": (REGIME["chaotic"],     REGIME["chaotic"]),
        "R": (REGIME["periodic"],    REGIME["periodic"]),
    }
    _labels  = ["S", "I", "R"]
    _names   = ["Susceptible", "Infected", "Recovered"]

    for i, (label, name) in enumerate(zip(_labels, _names, strict=True)):
        fc, ec = _colors[label]
        _circle(ax, _CX[i], _CY, _CIRCLE_R, fc, ec)
        # Big bold letter
        ax.text(_CX[i], _CY, label,
                ha="center", va="center",
                fontsize=_BIG_FS, fontweight="bold", color=ec, zorder=5)
        # Full name beneath
        ax.text(_CX[i], _LABEL_Y, name,
                ha="center", va="center",
                fontsize=_FULL_FS, color="#444444", zorder=5)

    # ── 2. Transition arrows (gap between circle edge and arrow tip) ──────────
    _gap = _CIRCLE_R + 0.08

    # S → I
    _compartment_arrow(ax, _CX[0] + _gap, _CX[1] - _gap, _ARROW_Y,
                       color="#2c3e50")
    # I → R
    _compartment_arrow(ax, _CX[1] + _gap, _CX[2] - _gap, _ARROW_Y,
                       color="#2c3e50")

    # ── 3. Arrow rate labels ──────────────────────────────────────────────────
    _SI_MID = (_CX[0] + _CX[1]) / 2
    _IR_MID = (_CX[1] + _CX[2]) / 2

    ax.text(_SI_MID, _ARROW_Y + 0.28,
            r"$\beta\,S I / N$",
            ha="center", va="bottom", fontsize=10, color="#2c3e50", zorder=5,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))

    ax.text(_IR_MID, _ARROW_Y + 0.28,
            r"$\gamma\,I$",
            ha="center", va="bottom", fontsize=10, color="#2c3e50", zorder=5,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))

    # ── 4. Lockdown control box ───────────────────────────────────────────────
    x0 = _LOCKBOX_XC - _LOCKBOX_W / 2
    y0 = _LOCKBOX_YC - _LOCKBOX_H / 2
    lockbox = mpatches.FancyBboxPatch(
        (x0, y0), _LOCKBOX_W, _LOCKBOX_H,
        boxstyle="round,pad=0.1",
        facecolor="#f5f5f5", edgecolor="#8e44ad",
        linewidth=1.6, zorder=4,
    )
    ax.add_patch(lockbox)
    ax.text(_LOCKBOX_XC, _LOCKBOX_YC, "lockdown",
            ha="center", va="center", fontsize=9,
            fontweight="bold", color="#8e44ad", zorder=5)

    # Short downward arrow: lockdown box → S→I arrow midpoint
    _lock_arrow_tip_y = _ARROW_Y + 0.32
    ax.add_patch(mpatches.FancyArrowPatch(
        (_LOCKBOX_XC, _LOCKBOX_YC - _LOCKBOX_H / 2 - 0.04),
        (_LOCKBOX_XC, _lock_arrow_tip_y),
        arrowstyle="-|>",
        mutation_scale=14,
        color="#8e44ad",
        linewidth=1.4,
        linestyle="dashed",
        zorder=5,
    ))

    # ── 5. Callout: lockdown law ──────────────────────────────────────────────
    # Split formula over two lines so it fits in the callout box width.
    _formula = (
        r"$\beta_{\mathrm{eff}}(t)=\beta\,[1-(1-f_{\rm lock})\,\sigma(k(t-t_{\rm lock}))]$"
    )
    callout(
        ax,
        xy=_CALLOUT_TIP,
        text=_formula,
        xytext=_CALLOUT_TXT,
        color="#8e44ad",
        fontsize=7.5,
    )

    # ── 6. Title ──────────────────────────────────────────────────────────────
    ax.set_title(
        "Mean-field SIR with a lockdown control on transmission",
        fontweight="bold",
    )

    fig.tight_layout()
    save_vector(fig, OUT_NAME, out_dir=str(out_dir(OUT_AREA)))
    plt.close(fig)


if __name__ == "__main__":
    main()
