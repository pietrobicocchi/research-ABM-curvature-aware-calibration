"""Booklet 2, Figure 4: constructing the OPG matrix from per-seed gradients (concept)."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from curvature_calib.viz.booklet_style import QUAL, apply_booklet_style, save_vector  # noqa: E402
from scripts.booklets._figbase import out_dir  # noqa: E402

OUT_AREA = "methodology"
OUT_NAME = "fig_04_opg_construction"

_DARK = "#2c3e50"


def _annotate_matrix(ax: plt.Axes, M: np.ndarray, fmt: str = "{:.2f}") -> None:
    """Overlay cell-value text on a 2x2 imshow."""
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, fmt.format(M[i, j]),
                ha="center", va="center",
                fontsize=10, fontweight="bold",
                color="white" if abs(M[i, j]) > 0.5 * np.abs(M).max() else _DARK,
            )


def main() -> None:
    apply_booklet_style()

    # ── Synthetic data ────────────────────────────────────────────────────────
    rng = np.random.default_rng(1)
    A = np.array([[1.0, 0.0], [0.9, 0.4]])
    G = rng.standard_normal((200, 2)) @ A.T
    G -= G.mean(axis=0)                          # centre
    F = (G.T @ G) / len(G)                       # 2x2 OPG matrix

    g1, g2 = G[0], G[1]
    op1 = np.outer(g1, g1)                       # g_1 g_1^T (example outer product)

    # ── Layout: 3-panel fallback ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    ax_a, ax_b, ax_c = axes

    # ── Panel (a): gradient cloud ─────────────────────────────────────────────
    ax_a.scatter(G[:, 0], G[:, 1], color=QUAL[0], alpha=0.45, s=12, linewidths=0,
                 label=r"$g_m$, $m=1,\ldots,M$")
    ax_a.plot(0, 0, "k+", ms=10, mew=1.8, zorder=5)

    # Two highlighted example gradients as arrows from origin
    for g, color, label in zip(
        [g1, g2],
        [QUAL[1], QUAL[3]],
        [r"$g_1$", r"$g_2$"],
        strict=True,
    ):
        ax_a.annotate(
            "", xy=(g[0], g[1]), xytext=(0.0, 0.0),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=2.0, mutation_scale=14),
        )
        ax_a.text(
            g[0] * 1.15, g[1] * 1.15, label,
            color=color, fontsize=10, fontweight="bold",
            ha="center", va="center",
        )

    ax_a.set_aspect("equal")
    ax_a.set_xlabel(r"$\theta_1$ direction")
    ax_a.set_ylabel(r"$\theta_2$ direction")
    ax_a.set_title("(a) per-seed gradients $\\{g_m\\}$")
    ax_a.legend(loc="upper left", fontsize=8, framealpha=0.85)

    # ── Panel (b): example outer product g_1 g_1^T ───────────────────────────
    vmax_b = np.abs(op1).max()
    im_b = ax_b.imshow(op1, cmap="RdBu_r", vmin=-vmax_b, vmax=vmax_b,
                       aspect="equal", interpolation="nearest")
    _annotate_matrix(ax_b, op1)
    ax_b.set_xticks([0, 1])
    ax_b.set_yticks([0, 1])
    ax_b.set_xticklabels(["1", "2"], fontsize=9)
    ax_b.set_yticklabels(["1", "2"], fontsize=9)
    ax_b.tick_params(length=0)
    ax_b.set_title(r"(b) outer product $g_1 g_1^\top$")
    ax_b.set_xlabel(r"parameter index $j$")
    ax_b.set_ylabel(r"parameter index $i$")
    fig.colorbar(im_b, ax=ax_b, fraction=0.046, pad=0.08)
    ax_b.text(
        0.5, -0.22,
        r"average over $M$ seeds $\;\longrightarrow\;$ $\hat F$",
        transform=ax_b.transAxes, ha="center", va="top",
        fontsize=9, style="italic", color=_DARK,
    )

    # ── Panel (c): averaged OPG matrix F̂ ────────────────────────────────────
    vmax_c = F.max()
    im_c = ax_c.imshow(F, cmap="magma", vmin=0, vmax=vmax_c,
                       aspect="equal", interpolation="nearest")
    _annotate_matrix(ax_c, F, fmt="{:.3f}")
    ax_c.set_xticks([0, 1])
    ax_c.set_yticks([0, 1])
    ax_c.set_xticklabels(["1", "2"], fontsize=9)
    ax_c.set_yticklabels(["1", "2"], fontsize=9)
    ax_c.tick_params(length=0)
    ax_c.set_title(r"(c) OPG matrix $\hat F = \frac{1}{M}\sum_m g_m g_m^\top$")
    ax_c.set_xlabel(r"parameter index $j$")
    ax_c.set_ylabel(r"parameter index $i$")
    fig.colorbar(im_c, ax=ax_c, fraction=0.046, pad=0.08, label="magnitude")
    ax_c.text(
        0.5, -0.22,
        "eigen-axes = stiff / sloppy directions",
        transform=ax_c.transAxes, ha="center", va="top",
        fontsize=9, style="italic", color=_DARK,
    )

    # ── Arrow annotations between panels ─────────────────────────────────────
    fig.text(0.345, 0.52, r"$\longrightarrow$", ha="center", va="center",
             fontsize=18, color=_DARK)
    fig.text(0.655, 0.52, r"$\longrightarrow$", ha="center", va="center",
             fontsize=18, color=_DARK)

    fig.suptitle(
        r"The OPG matrix is the second moment of the per-seed gradient cloud",
        fontweight="bold",
    )
    fig.tight_layout()

    save_vector(fig, OUT_NAME, out_dir=str(out_dir(OUT_AREA)))
    plt.close(fig)


if __name__ == "__main__":
    main()
