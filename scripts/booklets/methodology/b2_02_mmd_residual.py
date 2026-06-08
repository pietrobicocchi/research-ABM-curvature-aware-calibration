"""Booklet 2, Figure 2: MMD as the RKHS-norm of a kernel-mean residual (concept)."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from curvature_calib.viz.booklet_annotate import callout  # noqa: E402
from curvature_calib.viz.booklet_style import QUAL, apply_booklet_style, save_vector  # noqa: E402
from scripts.booklets._figbase import out_dir  # noqa: E402

OUT_AREA = "methodology"
OUT_NAME = "fig_02_mmd_residual"

_DARK = "#2c3e50"
_GREY = "#bdc3c7"


def main() -> None:
    apply_booklet_style()

    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    fig.suptitle(
        "MMD compares distributions through their kernel mean embeddings",
        fontweight="bold",
    )

    # ── Stage A: samples ──────────────────────────────────────────────────────
    ax_a = axes[0]
    rng = np.random.default_rng(0)

    sim_pts = rng.normal(loc=[0.3, 0.6], scale=0.12, size=(40, 2))
    ref_pts = rng.normal(loc=[0.7, 0.4], scale=0.12, size=(40, 2))

    ax_a.scatter(
        sim_pts[:, 0], sim_pts[:, 1],
        s=20, color=QUAL[0], alpha=0.75, label=r"simulated $\mathbb{P}_\theta$",
        zorder=3,
    )
    ax_a.scatter(
        ref_pts[:, 0], ref_pts[:, 1],
        s=20, color=QUAL[1], alpha=0.75, label=r"reference $\mathbb{P}_{\rm ref}$",
        zorder=3,
    )
    ax_a.legend(loc="upper right", fontsize=7.5, framealpha=0.85)
    ax_a.set_title("(a) samples")
    ax_a.set_xticks([])
    ax_a.set_yticks([])
    ax_a.set_xlim(-0.05, 1.05)
    ax_a.set_ylim(-0.05, 1.05)
    # Turn off grid for schematic panels
    ax_a.grid(False)
    for spine in ax_a.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(_DARK)
        spine.set_linewidth(0.8)

    # ── Stage B: kernel mean embeddings ──────────────────────────────────────
    ax_b = axes[1]
    ax_b.set_xlim(0, 10)
    ax_b.set_ylim(0, 10)
    ax_b.axis("off")
    ax_b.set_title("(b) mean embeddings")

    # RKHS ellipse
    rkhs_ellipse = mpatches.Ellipse(
        (5.0, 5.0), width=8.5, height=7.5,
        linewidth=1.6, edgecolor=_DARK,
        facecolor="#f0f0f0", alpha=0.55, zorder=1,
    )
    ax_b.add_patch(rkhs_ellipse)
    ax_b.text(
        5.0, 9.0, r"$\mathcal{H}$",
        ha="center", va="center", fontsize=14,
        color=_DARK, fontstyle="italic", zorder=4,
    )

    # Embedding positions
    mu_P_x, mu_P_y = 3.3, 5.8
    mu_R_x, mu_R_y = 6.8, 4.2

    # Connecting line
    ax_b.plot(
        [mu_P_x, mu_R_x], [mu_P_y, mu_R_y],
        color=_DARK, linewidth=1.4, linestyle="--", zorder=2, alpha=0.7,
    )

    # Midpoint distance label — use callout arrow pointing at midpoint of line
    mid_x = (mu_P_x + mu_R_x) / 2
    mid_y = (mu_P_y + mu_R_y) / 2
    callout(
        ax_b,
        xy=(mid_x, mid_y),
        text=r"$\|\mu_{\mathbb{P}_\theta} - \mu_{\rm ref}\|_{\mathcal{H}}$",
        xytext=(mid_x - 0.5, mid_y + 2.0),
        color=_DARK,
        fontsize=8.5,
    )

    # Embedding dots
    ax_b.scatter([mu_P_x], [mu_P_y], s=90, color=QUAL[0], zorder=5, linewidths=0)
    ax_b.scatter([mu_R_x], [mu_R_y], s=90, color=QUAL[1], zorder=5, linewidths=0)

    # Labels for each embedding
    ax_b.text(
        mu_P_x - 0.35, mu_P_y + 0.55,
        r"$\mu_{\mathbb{P}_\theta}$",
        ha="center", va="bottom", fontsize=10, color=QUAL[0], fontweight="bold", zorder=6,
    )
    ax_b.text(
        mu_R_x + 0.35, mu_R_y - 0.55,
        r"$\mu_{\rm ref}$",
        ha="center", va="top", fontsize=10, color=QUAL[1], fontweight="bold", zorder=6,
    )

    # Formula at bottom
    ax_b.text(
        5.0, 1.05,
        r"$\mu_\mathbb{P} = \mathbb{E}[k(\cdot,x)]$",
        ha="center", va="center", fontsize=8.5, color=_DARK,
        fontstyle="italic", zorder=4,
    )

    # ── Stage C: the loss ─────────────────────────────────────────────────────
    ax_c = axes[2]
    ax_c.axis("off")
    ax_c.set_title("(c) MMD loss")

    # Main equation
    _eq = (
        r"$\mathcal{L}(\theta) = "
        r"\|\mu_{\mathbb{P}_\theta} - \mu_{\mathbb{P}_{\rm ref}}\|_{\mathcal{H}}^2$"
    )
    ax_c.text(
        0.5, 0.62, _eq,
        ha="center", va="center", fontsize=15,
        transform=ax_c.transAxes, color=_DARK,
    )

    # Sub-label
    ax_c.text(
        0.5, 0.44,
        "a squared residual norm",
        ha="center", va="center", fontsize=9,
        transform=ax_c.transAxes, color=_DARK, fontstyle="italic",
    )

    # Callout box in axes coordinates using ax.text with bbox
    ax_c.text(
        0.5, 0.22,
        "residual structure\n"
        r"$\Rightarrow$ Gauss–Newton reading of $\hat{F}$"
        "\n(Sec. methodology)",
        ha="center", va="center", fontsize=8,
        transform=ax_c.transAxes,
        color=QUAL[1],
        bbox=dict(
            boxstyle="round,pad=0.4",
            fc="white", ec=QUAL[1], lw=1.2, alpha=0.95,
        ),
    )

    # ── Inter-panel arrows ────────────────────────────────────────────────────
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    # Arrows sit in the white space between subplots (figure-fraction coords).
    # With 3 equal subplots + default spacing the gaps are near x ≈ 0.345 and
    # x ≈ 0.665. Empirically tested below; adjust if panels shift.
    for x_frac in (0.345, 0.665):
        fig.text(
            x_frac, 0.48, r"$\rightarrow$",
            fontsize=26, ha="center", va="center",
            color=_DARK, transform=fig.transFigure,
        )

    save_vector(fig, OUT_NAME, out_dir=str(out_dir(OUT_AREA)))
    plt.close(fig)


if __name__ == "__main__":
    main()
