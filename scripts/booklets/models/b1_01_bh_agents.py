"""Booklet 1, Figure 1: Brock-Hommes agent schematic (concept diagram)."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from curvature_calib.viz.booklet_annotate import callout  # noqa: E402
from curvature_calib.viz.booklet_style import QUAL, apply_booklet_style, save_vector  # noqa: E402
from scripts.booklets._figbase import out_dir  # noqa: E402

OUT_AREA = "models"
OUT_NAME = "fig_01_bh_agents"

# ── Layout constants ──────────────────────────────────────────────────────────
_BOX_H = 0.80   # box height
_BOX_W = 2.20   # box width

# X centres of the three columns
_X_LEFT   = 1.60
_X_MID    = 5.00
_X_RIGHT  = 8.40

# Y centres (top / bottom type boxes stacked in the middle column)
_Y_TOP    = 3.60
_Y_BOT    = 2.00
_Y_LEFT   = (_Y_TOP + _Y_BOT) / 2   # vertically centred with the two type boxes
_Y_RIGHT  = _Y_LEFT

# Callout placement
_CALLOUT_TIP    = (_X_MID - _BOX_W / 2 - 0.15, (_Y_TOP + _Y_BOT) / 2)
_CALLOUT_LABEL  = (_X_MID - _BOX_W / 2 - 1.40, (_Y_TOP + _Y_BOT) / 2 + 0.80)


def _add_box(ax, x_center, y_center, width, height, text,
             facecolor="white", edgecolor="#2c3e50", alpha=1.0,
             fontsize=10, bold=False):
    """Draw a rounded rectangle centred at (x_center, y_center) with a label."""
    x0 = x_center - width / 2
    y0 = y_center - height / 2
    patch = mpatches.FancyBboxPatch(
        (x0, y0), width, height,
        boxstyle="round,pad=0.1",
        facecolor=facecolor, edgecolor=edgecolor,
        linewidth=1.5, alpha=alpha, zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        x_center, y_center, text,
        ha="center", va="center", fontsize=fontsize,
        fontweight="bold" if bold else "normal",
        color="#2c3e50", zorder=3, wrap=False,
        multialignment="center",
    )


def _arrow(ax, x_start, y_start, x_end, y_end,
           connectionstyle="arc3,rad=0.0", color="#2c3e50"):
    """Draw a directed arrow between two data-coordinate points."""
    ax.add_patch(mpatches.FancyArrowPatch(
        (x_start, y_start), (x_end, y_end),
        arrowstyle="-|>",
        mutation_scale=18,
        connectionstyle=connectionstyle,
        color=color, linewidth=1.4, zorder=4,
    ))


def main() -> None:
    apply_booklet_style()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)

    # ── 1. LEFT box — past performance ───────────────────────────────────────
    _add_box(
        ax, _X_LEFT, _Y_LEFT, _BOX_W + 0.3, _BOX_H * 2.0,
        "Past performance\n" + r"realised profits $U_{h,t}$",
        facecolor="#ecf0f1", edgecolor="#2c3e50", fontsize=9.5,
    )

    # ── 2. MIDDLE boxes — two trader types ───────────────────────────────────
    # Top: trend follower (QUAL[0] = blue)
    _add_box(
        ax, _X_MID, _Y_TOP, _BOX_W, _BOX_H,
        r"Type 1: trend follower" + "\n" + r"$(g_1,\, b_1)$",
        facecolor=QUAL[0], edgecolor=QUAL[0], alpha=0.18, fontsize=9.5,
    )
    # Re-draw edge only (alpha=1) so the border is crisp despite fill alpha
    ax.add_patch(mpatches.FancyBboxPatch(
        (_X_MID - _BOX_W / 2, _Y_TOP - _BOX_H / 2), _BOX_W, _BOX_H,
        boxstyle="round,pad=0.1",
        facecolor="none", edgecolor=QUAL[0], linewidth=1.8, zorder=3,
    ))

    # Bottom: contrarian (QUAL[1] = red)
    _add_box(
        ax, _X_MID, _Y_BOT, _BOX_W, _BOX_H,
        r"Type 2: contrarian" + "\n" + r"$(g_2,\, b_2)$",
        facecolor=QUAL[1], edgecolor=QUAL[1], alpha=0.18, fontsize=9.5,
    )
    ax.add_patch(mpatches.FancyBboxPatch(
        (_X_MID - _BOX_W / 2, _Y_BOT - _BOX_H / 2), _BOX_W, _BOX_H,
        boxstyle="round,pad=0.1",
        facecolor="none", edgecolor=QUAL[1], linewidth=1.8, zorder=3,
    ))

    # ── 3. RIGHT box — market price ───────────────────────────────────────────
    _add_box(
        ax, _X_RIGHT, _Y_RIGHT, _BOX_W + 0.3, _BOX_H * 2.0,
        r"Market price $x_t$",
        facecolor="#ecf0f1", edgecolor="#2c3e50", fontsize=9.5,
    )

    # ── 4a. Arrows: past performance → type boxes (switching) ─────────────────
    _arrow(ax,
           _X_LEFT + (_BOX_W + 0.3) / 2, _Y_LEFT + 0.25,
           _X_MID - _BOX_W / 2,           _Y_TOP)
    _arrow(ax,
           _X_LEFT + (_BOX_W + 0.3) / 2, _Y_LEFT - 0.25,
           _X_MID - _BOX_W / 2,           _Y_BOT)

    # ── 4b. Arrows: type boxes → market price (price formation) ───────────────
    _arrow(ax,
           _X_MID + _BOX_W / 2, _Y_TOP,
           _X_RIGHT - (_BOX_W + 0.3) / 2, _Y_RIGHT + 0.25)
    _arrow(ax,
           _X_MID + _BOX_W / 2, _Y_BOT,
           _X_RIGHT - (_BOX_W + 0.3) / 2, _Y_RIGHT - 0.25)

    # ── 4c. Curved feedback arrow: market price → past performance ────────────
    _arrow(ax,
           _X_RIGHT - (_BOX_W + 0.3) / 2, _Y_RIGHT - 0.55,
           _X_LEFT  + (_BOX_W + 0.3) / 2, _Y_LEFT  - 0.55,
           connectionstyle="arc3,rad=-0.40")

    # Label the feedback arc
    ax.text(
        5.00, 0.62, "fitness feedback",
        ha="center", va="center", fontsize=8.5, fontstyle="italic",
        color="#555555", zorder=5,
    )

    # ── 5. Callout: switching rule ─────────────────────────────────────────────
    callout(
        ax,
        xy=_CALLOUT_TIP,
        text=r"switch $\propto e^{\,\beta U}$",
        xytext=_CALLOUT_LABEL,
        color=QUAL[4],
        fontsize=9,
    )

    # ── 6. Title ───────────────────────────────────────────────────────────────
    ax.set_title(
        "Brock–Hommes: heterogeneous agents switch by past fitness",
        pad=10,
    )

    save_vector(fig, OUT_NAME, out_dir=str(out_dir(OUT_AREA)))
    plt.close(fig)


if __name__ == "__main__":
    main()
