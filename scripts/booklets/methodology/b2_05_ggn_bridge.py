"""Booklet 2, Figure 5: the GGN bridge -- F̂ approximates the loss curvature (concept)."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from curvature_calib.viz.booklet_style import QUAL, apply_booklet_style, save_vector  # noqa: E402
from scripts.booklets._figbase import out_dir  # noqa: E402

OUT_AREA = "methodology"
OUT_NAME = "fig_05_ggn_bridge"

_DARK = "#2c3e50"
_GREY = "#7f8c8d"
_LABEL_X = 0.06
_EQ_X = 0.5


def _label(ax: plt.Axes, y: float, text: str) -> None:
    """Small grey step label left-aligned above an equation."""
    ax.text(
        _LABEL_X, y, text,
        ha="left", va="bottom",
        fontsize=8.5, color=_GREY, fontstyle="italic",
        transform=ax.transAxes,
    )


def _hline(ax: plt.Axes, y: float) -> None:
    """Draw a thin grey horizontal rule in axes-fraction coordinates."""
    ax.plot(
        [0.06, 0.94], [y, y],
        transform=ax.transAxes,
        lw=0.6, color=_GREY, alpha=0.5,
    )


def _eq(ax: plt.Axes, y: float, text: str, **kwargs) -> None:
    """Centered equation."""
    ax.text(
        _EQ_X, y, text,
        ha="center", va="center",
        fontsize=14, color=_DARK,
        transform=ax.transAxes,
        **kwargs,
    )


def main() -> None:
    apply_booklet_style()
    fig, ax = plt.subplots(figsize=(9.5, 6))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # ── Title ─────────────────────────────────────────────────────────────────
    ax.text(
        0.5, 0.95,
        "Why the OPG matrix is a curvature object",
        ha="center", va="top",
        fontsize=13, fontweight="bold", color=_DARK,
        transform=ax.transAxes,
    )

    # ── Step 1 ────────────────────────────────────────────────────────────────
    _label(ax, 0.815, "Step 1 — MMD loss = squared residual")
    _eq(
        ax, 0.78,
        r"$\mathcal{L}(\theta) = \| r(\theta) \|^2,"
        r"\quad r(\theta) = \mu_{\mathbb{P}_\theta} - \mu_{\rm ref}$",
    )

    # thin divider
    _hline(ax, 0.735)

    # ── Step 2 ────────────────────────────────────────────────────────────────
    _label(ax, 0.695, "Step 2 — exact Hessian = Gauss-Newton term + residual-curvature term")
    _eq(
        ax, 0.655,
        r"$\nabla^2 \mathcal{L} = 2\, J^\top J + 2 \sum_i r_i \nabla^2 r_i$",
    )
    # annotation under second term — J definition and vanishing note
    ax.text(
        0.5, 0.612,
        r"$J = \partial r / \partial \theta$" + "    |    "
        + "second term vanishes as  r -> 0  (near a good fit)",
        ha="center", va="center",
        fontsize=8.5, color=_GREY, fontstyle="italic",
        transform=ax.transAxes,
    )

    _hline(ax, 0.568)

    # ── Step 3 ────────────────────────────────────────────────────────────────
    _label(ax, 0.528, "Step 3 — per-seed gradient is the residual-contracted Jacobian")
    _eq(
        ax, 0.488,
        r"$g_m = J_m^\top r_m \Rightarrow"
        r" \hat F = \frac{1}{M}\sum_m g_m g_m^\top \approx J^\top J$",
    )

    _hline(ax, 0.443)

    # ── Conclusion box ────────────────────────────────────────────────────────
    conclusion_text = (
        r"$\hat F \approx \frac{1}{2} \nabla^2 \mathcal{L}$"
        "   —   a curvature object"
    )
    ax.text(
        0.5, 0.360,
        conclusion_text,
        ha="center", va="center",
        fontsize=14, fontweight="bold", color=_DARK,
        transform=ax.transAxes,
        bbox=dict(
            boxstyle="round,pad=0.45",
            fc="#e8ecf7",   # light blue tint (solid, so dark text stays legible)
            ec=QUAL[0],
            lw=1.8,
        ),
    )

    # ── Kunstner note ─────────────────────────────────────────────────────────
    kunstner_text = (
        "This is a generalised Gauss-Newton statement about the RESIDUAL structure of MMD\n"
        "— it does NOT assume a likelihood, so the Kunstner et al. (2019) 'OPG != Fisher'\n"
        "critique does not apply.  We call  F^  the OPG matrix throughout."
    )
    ax.text(
        0.5, 0.135,
        kunstner_text,
        ha="center", va="center",
        fontsize=8.0, color=_DARK,
        transform=ax.transAxes,
        linespacing=1.55,
        bbox=dict(
            boxstyle="round,pad=0.5",
            fc="white",
            ec=QUAL[1],
            lw=1.2,
            alpha=0.95,
        ),
    )

    save_vector(fig, OUT_NAME, out_dir=str(out_dir(OUT_AREA)))
    plt.close(fig)


if __name__ == "__main__":
    main()
