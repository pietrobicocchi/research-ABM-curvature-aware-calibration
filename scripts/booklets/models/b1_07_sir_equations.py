"""Booklet 1, Figure 7: mean-field SIR equations (concept)."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from curvature_calib.viz.booklet_style import QUAL, apply_booklet_style, save_vector  # noqa: E402
from scripts.booklets._figbase import out_dir  # noqa: E402

OUT_AREA = "models"
OUT_NAME = "fig_07_sir_equations"

# ── Vertical layout constants (0..1 axis coords) ─────────────────────────────
_TITLE_Y       = 0.945

_BLOCK1_LABEL  = 0.870
_BLOCK1_EQ1    = 0.808
_BLOCK1_EQ2    = 0.738
_BLOCK1_EQ3    = 0.678

_BLOCK2_LABEL  = 0.590
_BLOCK2_EQ     = 0.520
_BLOCK2_NOTE   = 0.462

_DIVIDER_Y     = 0.390

_LEGEND_TITLE  = 0.330
_LEGEND_ROWS   = [0.265, 0.212, 0.159, 0.106, 0.053]

_LABEL_X  = 0.06   # left edge for section labels
_EQ_X     = 0.50   # centre for equations
_SYM_X    = 0.08   # centre of the symbol badge
_DESC_X   = 0.15   # left edge of the description text


def main() -> None:
    apply_booklet_style()
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # ── Title ─────────────────────────────────────────────────────────────────
    ax.text(
        0.50, _TITLE_Y,
        "Mean-field SIR with a smoothed lockdown control",
        ha="center", va="center",
        fontsize=13, fontweight="bold", color="#2c3e50",
    )

    # ── Block 1: compartment dynamics ─────────────────────────────────────────
    ax.text(
        _LABEL_X, _BLOCK1_LABEL,
        "compartment dynamics",
        ha="left", va="center",
        fontsize=9, fontstyle="italic", color="#888888",
    )
    ax.text(
        _EQ_X, _BLOCK1_EQ1,
        r"$S_{t+1} = S_t - \beta_{\mathrm{eff}}(t)\,S_t I_t / N$",
        ha="center", va="center",
        fontsize=14, color="#2c3e50",
    )
    ax.text(
        _EQ_X, _BLOCK1_EQ2,
        r"$I_{t+1} = I_t + \beta_{\mathrm{eff}}(t)\,S_t I_t / N - \gamma\,I_t$",
        ha="center", va="center",
        fontsize=14, color="#2c3e50",
    )
    ax.text(
        _EQ_X, _BLOCK1_EQ3,
        r"$R_{t+1} = R_t + \gamma\,I_t$",
        ha="center", va="center",
        fontsize=14, color="#2c3e50",
    )

    # ── Block 2: lockdown surrogate ───────────────────────────────────────────
    ax.text(
        _LABEL_X, _BLOCK2_LABEL,
        "lockdown surrogate (differentiable)",
        ha="left", va="center",
        fontsize=9, fontstyle="italic", color="#888888",
    )
    ax.text(
        _EQ_X, _BLOCK2_EQ,
        r"$\beta_{\mathrm{eff}}(t) = \beta\,[\,1 - (1-f_{\mathrm{lock}})"
        r"\,\sigma(k\,(t - t_{\mathrm{lock}}))\,]$",
        ha="center", va="center",
        fontsize=14, color="#2c3e50",
    )
    ax.text(
        _EQ_X, _BLOCK2_NOTE,
        "sigmoid relaxes the discrete lockdown switch → gradients flow",
        ha="center", va="center",
        fontsize=9, fontstyle="italic", color="#888888",
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
        (r"$\beta$",                 "transmission rate"),
        (r"$\gamma$",                "recovery rate"),
        (r"$I_0$",                   "initial infected fraction"),
        (r"$t_{\mathrm{lock}}$",     "lockdown timing (fraction of horizon)"),
        (r"$f_{\mathrm{lock}}$",     "post-lockdown transmission multiplier in (0,1]"),
    ]

    for y_row, (sym, desc) in zip(_LEGEND_ROWS, _PARAMS, strict=False):
        # Highlighted symbol badge
        ax.text(
            _SYM_X, y_row, sym,
            ha="center", va="center",
            fontsize=12, fontweight="bold", color=QUAL[0],
            bbox=dict(
                boxstyle="round,pad=0.2",
                fc="#eaf0fb", ec=QUAL[0], lw=0.6,
            ),
        )
        # Plain-text description (real em-dash, no mathtext)
        ax.text(
            _DESC_X, y_row, "— " + desc,
            ha="left", va="center",
            fontsize=10, color="#444444",
        )

    save_vector(fig, OUT_NAME, out_dir=str(out_dir(OUT_AREA)))
    plt.close(fig)


if __name__ == "__main__":
    main()
