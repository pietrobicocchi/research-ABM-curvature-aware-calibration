"""Booklet 2, Figure 12: Jacobian sensitivity vs OPG eigendecomposition contrast.

Three panels:
  (a) Per-parameter Jacobian column norms — ranks b₁, b₂ as most sensitive
      individually; β appears least important
  (b) OPG correlation matrix ρ — shows b₁ and b₂ are nearly co-linear (ρ≈0.99),
      so individual sensitivity mis-states the picture
  (c) OPG eigenspectrum — the symmetric b₁+b₂ combination is the one stiff
      (identifiable) direction; β is the sloppy direction

Key message: per-parameter Jacobian analysis (Quera-Bofarull §5.4) cannot
distinguish b₁+b₂ (identifiable) from b₁−b₂ (not identifiable) because it
ignores off-diagonal coupling.  The OPG eigendecomposition makes this precise.

Data: outputs/brock_hommes/13_jacobian_comparison.npz (pre-computed)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from curvature_calib.viz.booklet_annotate import SLOPPY_COLOR, STIFF_COLOR  # noqa: E402
from curvature_calib.viz.booklet_style import QUAL, apply_booklet_style, save_vector  # noqa: E402
from scripts.booklets._figbase import out_dir  # noqa: E402

OUT_AREA = "methodology"
OUT_NAME = "fig_12_jacobian_vs_opg"

_DARK = "#2c3e50"
_GREY = "#7f8c8d"


def main() -> None:
    apply_booklet_style()

    root = Path(__file__).resolve().parents[3]
    d = np.load(root / "outputs" / "brock_hommes" / "13_jacobian_comparison.npz")

    param_names: list[str] = list(d["param_names"])
    S_jac = np.array(d["S_jac"], dtype=float)
    eigvals = np.array(d["eigvals"], dtype=float)
    V = np.array(d["V"], dtype=float)          # shape (P, P); columns = eigvecs
    rho = np.array(d["rho"], dtype=float)

    P = len(param_names)
    xs = np.arange(P)

    print("S_jac:", S_jac)
    print("eigvals:", eigvals)
    print("rho(b1,b2):", rho[2, 4])   # b₁=index 2, b₂=index 4

    # ── Figure ─────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 4.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.0, 1.2], wspace=0.42)
    ax_jac = fig.add_subplot(gs[0])
    ax_rho = fig.add_subplot(gs[1])
    ax_eig = fig.add_subplot(gs[2])

    # ── Panel (a): Jacobian column norms ──────────────────────────────────────
    bar_colors = [
        STIFF_COLOR if s == S_jac.max()
        else (SLOPPY_COLOR if s == S_jac.min() else QUAL[0])
        for s in S_jac
    ]
    ax_jac.bar(xs, S_jac, color=bar_colors, alpha=0.82, width=0.6,
               edgecolor="white", linewidth=0.8)
    ax_jac.set_xticks(xs)
    ax_jac.set_xticklabels(param_names, fontsize=10)
    ax_jac.set_ylabel("Jacobian column norm", fontsize=9)
    ax_jac.set_title(
        "(a)  Per-parameter sensitivity\n(Jacobian)",
        fontweight="bold", fontsize=9.5)

    # Annotation: b₁ ≈ b₂ ranked equal → ambiguous
    b1_idx, b2_idx = 2, 4
    ymax = S_jac.max()
    ax_jac.annotate(
        r"$b_1 \approx b_2$" + "\n(ambiguous—\ncannot tell apart)",
        xy=(b1_idx, S_jac[b1_idx]),
        xytext=(2.0, ymax * 1.25),
        ha="center", va="bottom", fontsize=7.5, color=_GREY, fontstyle="italic",
        arrowprops=dict(arrowstyle="->", color=_GREY, lw=0.8,
                        connectionstyle="arc3,rad=0.2"),
    )
    ax_jac.annotate(
        "",
        xy=(b2_idx, S_jac[b2_idx]),
        xytext=(2.0, ymax * 1.25),
        arrowprops=dict(arrowstyle="->", color=_GREY, lw=0.8,
                        connectionstyle="arc3,rad=-0.2"),
    )
    ax_jac.set_ylim(0, ymax * 1.55)

    # β is sloppy annotation
    ax_jac.text(0, S_jac[0] + ymax * 0.04, "sloppy",
                ha="center", va="bottom", fontsize=8,
                color=SLOPPY_COLOR, fontweight="bold")

    # ── Panel (b): correlation heatmap ────────────────────────────────────────
    im = ax_rho.imshow(np.abs(rho), cmap="Blues", vmin=0, vmax=1.0,
                       aspect="auto")
    for i in range(P):
        for j in range(P):
            val = abs(rho[i, j])
            ax_rho.text(j, i, f"{val:.2f}",
                        ha="center", va="center", fontsize=7.5,
                        color="white" if val > 0.65 else _DARK)
    ax_rho.set_xticks(range(P))
    ax_rho.set_xticklabels(param_names, fontsize=9)
    ax_rho.set_yticks(range(P))
    ax_rho.set_yticklabels(param_names, fontsize=9)
    ax_rho.set_title(
        "(b)  OPG correlation matrix  |ρ|\n",
        fontweight="bold", fontsize=9.5)

    # Highlight the b₁–b₂ pair (high ρ)
    import matplotlib.patches as mpatches
    rect = mpatches.FancyBboxPatch(
        (b2_idx - 0.48, b1_idx - 0.48), 0.96, 0.96,
        boxstyle="square,pad=0",
        facecolor="none", edgecolor=STIFF_COLOR, lw=2.2, zorder=5,
    )
    ax_rho.add_patch(rect)
    rect2 = mpatches.FancyBboxPatch(
        (b1_idx - 0.48, b2_idx - 0.48), 0.96, 0.96,
        boxstyle="square,pad=0",
        facecolor="none", edgecolor=STIFF_COLOR, lw=2.2, zorder=5,
    )
    ax_rho.add_patch(rect2)
    ax_rho.text(
        P - 0.5, -0.75,
        rf"$\rho(b_1,b_2) = {abs(rho[b1_idx, b2_idx]):.3f}$",
        ha="right", va="bottom", fontsize=8.5,
        color=STIFF_COLOR, fontweight="bold",
    )

    fig.colorbar(im, ax=ax_rho, fraction=0.046, pad=0.04)

    # ── Panel (c): OPG eigenspectrum ──────────────────────────────────────────
    eig_xs = np.arange(P)
    bar_eig = ax_eig.bar(eig_xs, eigvals, width=0.55, alpha=0.85,
                         edgecolor="white", linewidth=0.8,
                         color=[STIFF_COLOR if k == 0 else
                                (SLOPPY_COLOR if k == P-1 else QUAL[0])
                                for k in range(P)])
    ax_eig.set_yscale("log")
    ax_eig.set_xticks(eig_xs)
    ax_eig.set_xticklabels([rf"$v_{k+1}$" for k in range(P)], fontsize=10)
    ax_eig.set_ylabel(r"eigenvalue $\lambda_k$", fontsize=9)
    ax_eig.set_title(
        "(c)  OPG eigenspectrum\n(identifiable combinations)",
        fontweight="bold", fontsize=9.5)

    # Label stiff and sloppy with content
    def _top2_params(k: int) -> str:
        vec = np.abs(V[:, k])
        top = np.argsort(-vec)
        names_clean = [n.replace("$", "").replace("\\", "") for n in param_names]
        return f"{names_clean[top[0]]}+{names_clean[top[1]]}"

    ax_eig.text(0, eigvals[0] * 2.5,
                r"$b_1+b_2$" + "\n(stiff)",
                ha="center", va="bottom", fontsize=8,
                color=STIFF_COLOR, fontweight="bold")
    ax_eig.text(P - 1, eigvals[-1] * 0.3,
                r"$\beta$" + "\n(sloppy)",
                ha="center", va="top", fontsize=8,
                color=SLOPPY_COLOR, fontweight="bold")

    span = np.log10(eigvals[0] / max(eigvals[-1], 1e-30))
    ax_eig.text(0.97, 0.97,
                f"span: {span:.1f} OOM",
                transform=ax_eig.transAxes, ha="right", va="top",
                fontsize=8.5, color="#555",
                bbox=dict(fc="white", ec="grey", alpha=0.8,
                          boxstyle="round,pad=0.3"))

    fig.suptitle(
        r"Jacobian sees $b_1 \approx b_2$ — OPG reveals only $b_1+b_2$ is identifiable",
        fontweight="bold", fontsize=11,
    )

    save_vector(fig, OUT_NAME, out_dir=str(out_dir(OUT_AREA)))
    plt.close(fig)


if __name__ == "__main__":
    main()
