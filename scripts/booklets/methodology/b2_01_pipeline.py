"""Booklet 2, Figure 1: the differentiable calibration loop and the OPG matrix (concept)."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from curvature_calib.viz.booklet_annotate import callout  # noqa: E402
from curvature_calib.viz.booklet_style import QUAL, apply_booklet_style, save_vector  # noqa: E402
from scripts.booklets._figbase import out_dir  # noqa: E402

OUT_AREA = "methodology"
OUT_NAME = "fig_01_pipeline"

# ── Layout constants ──────────────────────────────────────────────────────────
_DARK        = "#2c3e50"
_GREY        = "#7f8c8d"

# Box dimensions
_BOX_H_STD   = 0.80   # standard box height
_BOX_W_STD   = 2.00   # standard box width
_BOX_W_WIDE  = 2.20   # wider box for multi-line content

# Main spine Y (centre)
_Y_SPINE     = 3.50

# Branch Y centres (branching from the hub box)
_Y_TOP       = 5.40   # calibration branch (top)
_Y_BOT       = 1.60   # OPG branch (bottom)

# X positions (left to right) for the main spine
_X_PARAMS    = 1.20   # "parameters θ_t"
_X_SIM       = 3.60   # "simulate M seeds"
_X_HUB       = 6.10   # "{g_m} hub" — branching point

# X positions for branch boxes (top and bottom share same x grid)
_X_MEAN_G    = 8.30   # "mean gradient" / "OPG matrix"
_X_STEP      = 10.50  # "gradient step" / "eigendecompose"
_X_DIAG      = 12.40  # "identifiability diagnostic" (bottom only)


def _add_box(
    ax,
    x_center,
    y_center,
    width,
    height,
    text,
    facecolor="white",
    edgecolor=None,
    alpha=1.0,
    fontsize=9,
    bold=False,
):
    """Draw a rounded rectangle centred at (x_center, y_center) with a label."""
    if edgecolor is None:
        edgecolor = _DARK
    x0 = x_center - width / 2
    y0 = y_center - height / 2
    patch = mpatches.FancyBboxPatch(
        (x0, y0), width, height,
        boxstyle="round,pad=0.1",
        facecolor=facecolor, edgecolor=edgecolor,
        linewidth=1.6, alpha=alpha, zorder=2,
    )
    ax.add_patch(patch)
    # Crisp border at alpha=1 when fill is transparent
    if alpha < 1.0:
        ax.add_patch(mpatches.FancyBboxPatch(
            (x0, y0), width, height,
            boxstyle="round,pad=0.1",
            facecolor="none", edgecolor=edgecolor,
            linewidth=1.6, zorder=3,
        ))
    ax.text(
        x_center, y_center, text,
        ha="center", va="center", fontsize=fontsize,
        fontweight="bold" if bold else "normal",
        color=_DARK, zorder=4, multialignment="center",
    )


def _arrow(ax, x_start, y_start, x_end, y_end,
           connectionstyle="arc3,rad=0.0", color=_DARK, lw=1.6):
    """Directed arrow between two data-coordinate points."""
    ax.add_patch(mpatches.FancyArrowPatch(
        (x_start, y_start), (x_end, y_end),
        arrowstyle="-|>",
        mutation_scale=18,
        connectionstyle=connectionstyle,
        color=color, linewidth=lw, zorder=5,
    ))


def main() -> None:
    apply_booklet_style()
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.axis("off")
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)

    # ── 1. Main spine boxes ───────────────────────────────────────────────────

    # Box 1: parameters θ_t
    _add_box(ax, _X_PARAMS, _Y_SPINE, _BOX_W_STD, _BOX_H_STD,
             r"parameters $\theta_t$",
             facecolor="#ecf0f1", edgecolor=_DARK)

    # Box 2: simulate M seeds
    _add_box(ax, _X_SIM, _Y_SPINE, _BOX_W_WIDE, _BOX_H_STD,
             "simulate $M$ seeds\n(differentiable ABM)",
             facecolor="#ecf0f1", edgecolor=_DARK)

    # Box 3: hub — per-seed gradients {g_m}
    _add_box(ax, _X_HUB, _Y_SPINE, _BOX_W_WIDE, _BOX_H_STD,
             r"per-seed gradients" + "\n" + r"$\{g_m\}_{m=1}^{M}$",
             facecolor=QUAL[0], edgecolor=QUAL[0], alpha=0.15)

    # ── 2. TOP branch — calibration path ─────────────────────────────────────

    # "calibration (what everyone uses)" path label
    ax.text((_X_MEAN_G + _X_STEP) / 2, _Y_TOP + 0.68,
            "calibration (what everyone uses)",
            ha="center", va="bottom", fontsize=8, fontstyle="italic",
            color=_GREY, zorder=5)

    # Box: mean gradient
    _add_box(ax, _X_MEAN_G, _Y_TOP, _BOX_W_STD, _BOX_H_STD,
             r"mean gradient $\bar{g}$",
             facecolor="#ecf0f1", edgecolor=_GREY)

    # Box: gradient step
    _add_box(ax, _X_STEP, _Y_TOP, _BOX_W_WIDE + 0.2, _BOX_H_STD,
             r"gradient step" + "\n" + r"$\theta_{t+1} = \theta_t - \eta\,\bar{g}$",
             facecolor="#ecf0f1", edgecolor=_GREY)

    # ── 3. BOTTOM branch — OPG path ───────────────────────────────────────────

    # "our contribution" path label
    ax.text((_X_MEAN_G + _X_DIAG) / 2, _Y_BOT - 0.68,
            "our contribution",
            ha="center", va="top", fontsize=8, fontweight="bold",
            color=QUAL[1], zorder=5)

    # Box: OPG matrix
    _add_box(ax, _X_MEAN_G, _Y_BOT, _BOX_W_WIDE + 0.2, _BOX_H_STD + 0.20,
             r"OPG matrix" + "\n"
             + r"$\hat{F} = \frac{1}{M}\sum_m g_m g_m^\top$",
             facecolor=QUAL[1], edgecolor=QUAL[1], alpha=0.15)

    # Box: eigendecompose
    _add_box(ax, _X_STEP, _Y_BOT, _BOX_W_WIDE, _BOX_H_STD,
             r"eigendecompose" + "\n" + r"$\hat{F} = V\Lambda V^\top$",
             facecolor="#ecf0f1", edgecolor=_DARK)

    # Box: identifiability diagnostic
    _add_box(ax, _X_DIAG, _Y_BOT, _BOX_W_WIDE, _BOX_H_STD,
             "identifiability\ndiagnostic",
             facecolor=QUAL[1], edgecolor=QUAL[1], alpha=0.15)

    # ── 4. Arrows — main spine ────────────────────────────────────────────────

    # θ_t → simulate
    _arrow(ax,
           _X_PARAMS + _BOX_W_STD / 2, _Y_SPINE,
           _X_SIM - _BOX_W_WIDE / 2, _Y_SPINE)

    # simulate → hub
    _arrow(ax,
           _X_SIM + _BOX_W_WIDE / 2, _Y_SPINE,
           _X_HUB - _BOX_W_WIDE / 2, _Y_SPINE)

    # ── 5. Arrows — hub branches upward/downward ──────────────────────────────

    # hub → mean gradient (top branch)
    _arrow(ax,
           _X_HUB + _BOX_W_WIDE / 2, _Y_SPINE + _BOX_H_STD / 4,
           _X_MEAN_G - _BOX_W_STD / 2, _Y_TOP,
           color=_GREY)

    # hub → OPG matrix (bottom branch)
    _arrow(ax,
           _X_HUB + _BOX_W_WIDE / 2, _Y_SPINE - _BOX_H_STD / 4,
           _X_MEAN_G - (_BOX_W_WIDE + 0.2) / 2, _Y_BOT,
           color=QUAL[1])

    # ── 6. Arrows — top branch ────────────────────────────────────────────────

    # mean gradient → gradient step
    _arrow(ax,
           _X_MEAN_G + _BOX_W_STD / 2, _Y_TOP,
           _X_STEP - (_BOX_W_WIDE + 0.2) / 2, _Y_TOP,
           color=_GREY)

    # ── 7. Arrows — bottom branch ─────────────────────────────────────────────
    # Boxes nearly touch (OPG right edge 9.50, eigen left edge 9.40), so
    # using exact edge-to-edge coords gives posA.x > posB.x → right-to-left.
    # Fix: start at the right edge of the source box, end at the destination
    # centre — this guarantees posA.x < posB.x for both arrows.

    # OPG right edge → eigendecompose centre  (arrowhead on eigendecompose)
    _arrow(ax,
           _X_MEAN_G + (_BOX_W_WIDE + 0.2) / 2, _Y_BOT,
           _X_STEP, _Y_BOT,
           color=QUAL[1])

    # eigendecompose right edge → diagnostic centre  (arrowhead on diagnostic)
    _arrow(ax,
           _X_STEP + _BOX_W_WIDE / 2, _Y_BOT,
           _X_DIAG, _Y_BOT,
           color=QUAL[1])

    # ── 8. Feedback loop: gradient step → θ_t ────────────────────────────────
    # Arc below the canvas midline to avoid the bottom branch
    ax.add_patch(mpatches.FancyArrowPatch(
        (_X_STEP + (_BOX_W_WIDE + 0.2) / 2, _Y_TOP),
        (_X_PARAMS, _Y_SPINE + _BOX_H_STD / 2),
        arrowstyle="-|>",
        mutation_scale=18,
        connectionstyle="arc3,rad=-0.38",
        color=_GREY, linewidth=1.5, zorder=1,
    ))
    # "iterate" label — sits on the feedback arc near its visible apex.
    ax.text(6.5, 5.95,
            "iterate",
            ha="center", va="center", fontsize=8, fontstyle="italic",
            color=_GREY, zorder=6,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9))

    # ── 9. Callout: OPG already in memory ────────────────────────────────────
    # Arrow tip touches the top edge of the OPG box; label sits to the right,
    # in the clear space between the hub and the OPG box.
    callout(
        ax,
        xy=(_X_MEAN_G, _Y_BOT + _BOX_H_STD / 2 + 0.06),
        text="already in memory —\nnormally discarded",
        xytext=(_X_MEAN_G + 0.20, _Y_BOT + 2.10),
        color=QUAL[1],
        fontsize=8,
    )

    # ── 10. Title ─────────────────────────────────────────────────────────────
    ax.set_title(
        "Differentiable calibration already computes the OPG matrix — we keep it",
        fontweight="bold",
        pad=8,
    )

    save_vector(fig, OUT_NAME, out_dir=str(out_dir(OUT_AREA)))
    plt.close(fig)


if __name__ == "__main__":
    main()
