"""Booklet 1, Figure 2: Brock-Hommes annotated equations (concept)."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from curvature_calib.viz.booklet_style import QUAL, apply_booklet_style, save_vector  # noqa: E402
from scripts.booklets._figbase import out_dir  # noqa: E402

OUT_AREA = "models"
OUT_NAME = "fig_02_bh_equations"

# ── Vertical layout constants (0..1 axis coords) ─────────────────────────────
_TITLE_Y      = 0.945
_BLOCK1_LABEL = 0.855
_BLOCK1_EQ    = 0.800
_BLOCK2_LABEL = 0.690
_BLOCK2_EQ    = 0.635
_BLOCK3_LABEL = 0.510
_BLOCK3_EQ    = 0.430
_DIVIDER_Y    = 0.350
_LEGEND_TITLE = 0.295
_LEGEND_ROWS  = [0.235, 0.188, 0.141, 0.094, 0.047]

_LABEL_X  = 0.06   # left edge for section labels
_EQ_X     = 0.50   # centre for equations
_SYM_X    = 0.08   # left edge for parameter symbol
_DESC_X   = 0.18   # left edge for parameter description


def main() -> None:
    apply_booklet_style()
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # ── Title ─────────────────────────────────────────────────────────────────
    ax.text(
        0.50, _TITLE_Y,
        "Brock–Hommes heterogeneous-agent asset pricing",
        ha="center", va="center",
        fontsize=13, fontweight="bold", color="#2c3e50",
    )

    # ── Block 1: price formation ──────────────────────────────────────────────
    ax.text(
        _LABEL_X, _BLOCK1_LABEL,
        "price formation",
        ha="left", va="center",
        fontsize=9, fontstyle="italic", color="#888888",
    )
    ax.text(
        _EQ_X, _BLOCK1_EQ,
        r"$R\,x_{t+1} = \sum_h n_{h,t}\,(g_h x_t + b_h) + \epsilon_t$",
        ha="center", va="center",
        fontsize=15, color="#2c3e50",
    )

    # ── Block 2: fitness ──────────────────────────────────────────────────────
    ax.text(
        _LABEL_X, _BLOCK2_LABEL,
        "fitness (realised profit)",
        ha="left", va="center",
        fontsize=9, fontstyle="italic", color="#888888",
    )
    ax.text(
        _EQ_X, _BLOCK2_EQ,
        r"$U_{h,t} = (x_t - R\,x_{t-1})\,(g_h x_{t-1} + b_h - R\,x_{t-1})$",
        ha="center", va="center",
        fontsize=15, color="#2c3e50",
    )

    # ── Block 3: type switching ───────────────────────────────────────────────
    ax.text(
        _LABEL_X, _BLOCK3_LABEL,
        "type switching",
        ha="left", va="center",
        fontsize=9, fontstyle="italic", color="#888888",
    )
    ax.text(
        _EQ_X, _BLOCK3_EQ,
        r"$n_{h,t} = \dfrac{e^{\beta U_{h,t-1}}}{\sum_k e^{\beta U_{k,t-1}}}$",
        ha="center", va="center",
        fontsize=15, color="#2c3e50",
    )

    # ── Divider ───────────────────────────────────────────────────────────────
    ax.plot([0.05, 0.95], [_DIVIDER_Y, _DIVIDER_Y], color="#cccccc", lw=0.8)

    # ── Parameter legend title ─────────────────────────────────────────────────
    ax.text(
        _LABEL_X, _LEGEND_TITLE,
        "calibrated parameters  (P = 5)",
        ha="left", va="center",
        fontsize=9.5, fontweight="bold", color="#2c3e50",
    )

    # ── Parameter rows ────────────────────────────────────────────────────────
    _PARAMS = [
        (r"$\beta$",   "intensity of choice (how sharply agents switch)"),
        (r"$g_1$",     "trend coefficient, type 1"),
        (r"$b_1$",     "bias, type 1"),
        (r"$g_2$",     "trend coefficient, type 2"),
        (r"$b_2$",     "bias, type 2"),
    ]

    for y_row, (sym, desc) in zip(_LEGEND_ROWS, _PARAMS, strict=False):
        # Highlighted symbol
        ax.text(
            _SYM_X, y_row, sym,
            ha="center", va="center",
            fontsize=12, fontweight="bold", color=QUAL[0],
            bbox=dict(
                boxstyle="round,pad=0.2",
                fc="#eaf0fb", ec=QUAL[0], lw=0.6,
            ),
        )
        # Description
        ax.text(
            _DESC_X, y_row, r"$-$\;" + desc,
            ha="left", va="center",
            fontsize=10, color="#444444",
        )

    save_vector(fig, OUT_NAME, out_dir=str(out_dir(OUT_AREA)))
    plt.close(fig)


if __name__ == "__main__":
    main()
