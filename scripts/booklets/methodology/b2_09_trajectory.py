"""Booklet 2, Figure 9: OPG eigenstructure along the calibration trajectory.

Three stacked panels:
  (a) log λ_k(t) – eigenvalue hierarchy is preserved throughout the run
  (b) d_eff(t)   – effective dimension (near 1 → one direction dominates)
  (c) principal angles θ_k(t) – bootstrap CIs confirming the leading
      eigenvectors lock on within a few iterations

Data loaded from pre-computed outputs (gitignored, regenerable):
  outputs/brock_hommes/25_eigenvalue_trajectory.npz   — chaotic regime, (60,5)
  outputs/brock_hommes/19_trajectory_bootstrap.npz    — snapshot bootstrap
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import matplotlib.cm as cm  # noqa: E402
from curvature_calib.viz.booklet_style import QUAL, apply_booklet_style, save_vector  # noqa: E402
from scripts.booklets._figbase import out_dir  # noqa: E402

OUT_AREA = "methodology"
OUT_NAME = "fig_09_trajectory"

_DARK = "#2c3e50"
_GREY = "#7f8c8d"
_STIFF_COLOR = QUAL[1]   # red – stiff eigenvector v_1
_SLOPPY_COLOR = _GREY    # grey – sloppy eigenvector v_P

# Number of gradient-horizon iterations in the trajectory
_T = 60

# Colour ramp for the 5 eigenvalue curves (dark → light, from viridis)
_RAMP: list[tuple[float, float, float, float]] = [cm.viridis(v) for v in [0.85, 0.65, 0.45, 0.25, 0.05]]


def _effective_dim(eigvals: np.ndarray) -> np.ndarray:
    """Participation-ratio effective dimension for each row of eigvals."""
    lam = np.clip(eigvals, 1e-30, None)
    return lam.sum(axis=1) ** 2 / (lam ** 2).sum(axis=1)


def main() -> None:
    apply_booklet_style()

    # ── Load data ──────────────────────────────────────────────────────────────
    root = Path(__file__).resolve().parents[3]
    d25 = np.load(root / "outputs" / "brock_hommes" / "25_eigenvalue_trajectory.npz")
    d19 = np.load(root / "outputs" / "brock_hommes" / "19_trajectory_bootstrap.npz")

    eigvals = d25["chaotic"]           # (60, 5)  – chaotic regime trajectory
    snap_iters = d19["snapshot_iters"] # (8,)     – iteration indices
    angles = d19["angles"]             # (8, 500, 5) – principal angles in °

    t_axis = np.arange(_T)
    d_eff = _effective_dim(eigvals)

    # ── Debug print (verify callout annotations) ──────────────────────────────
    print("Chaotic eigenvalue OOM spans:")
    for t in [0, 10, 30, 59]:
        span = np.log10(eigvals[t].max() / max(eigvals[t].min(), 1e-30))
        print(f"  t={t}: {span:.1f} OOM")

    print("Effective dimension range:", d_eff.min(), "–", d_eff.max())

    # angles array is already in degrees
    ang_v1 = np.median(angles[:, :, 0], axis=1)
    ang_vP = np.median(angles[:, :, 4], axis=1)
    ang_v1_p25 = np.percentile(angles[:, :, 0], 25, axis=1)
    ang_v1_p75 = np.percentile(angles[:, :, 0], 75, axis=1)
    ang_vP_p25 = np.percentile(angles[:, :, 4], 25, axis=1)
    ang_vP_p75 = np.percentile(angles[:, :, 4], 75, axis=1)

    print("Principal angles v_1 (stiff) – median per snapshot:", ang_v1)
    print("Principal angles v_P (sloppy) – median per snapshot:", ang_vP)

    # ── Figure ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(9, 10),
                             gridspec_kw={"hspace": 0.40,
                                          "height_ratios": [2.5, 1.0, 1.8]})
    ax_eig, ax_deff, ax_ang = axes

    # ── Panel (a): log eigenvalue trajectory ──────────────────────────────────
    for k in range(5):
        lam_k = eigvals[:, k]
        ax_eig.plot(t_axis, lam_k,
                    color=_RAMP[k], lw=1.8,
                    label=rf"$\lambda_{k+1}$")

    ax_eig.set_yscale("log")
    ax_eig.set_xlim(0, _T - 1)
    ax_eig.set_ylabel(r"eigenvalue $\lambda_k$", fontsize=9)
    ax_eig.set_title(
        "OPG eigenstructure is stable along the calibration trajectory\n"
        "(chaotic regime, BH)",
        fontweight="bold", fontsize=10)
    ax_eig.legend(loc="lower left", ncol=5, fontsize=7.5, frameon=True,
                  columnspacing=0.8, handlelength=1.2)

    # Annotate hierarchy preservation
    ax_eig.text(
        0.98, 0.97,
        "hierarchy preserved\nthroughout",
        transform=ax_eig.transAxes,
        ha="right", va="top", fontsize=8, color=_GREY, fontstyle="italic",
    )

    # ── Panel (b): effective dimension ────────────────────────────────────────
    ax_deff.plot(t_axis, d_eff, color=QUAL[0], lw=2.0)
    ax_deff.axhline(1.0, color=_GREY, ls=":", lw=1.0)
    ax_deff.set_xlim(0, _T - 1)
    ax_deff.set_ylim(0.9, max(d_eff) * 1.15)
    ax_deff.set_ylabel(r"$d_{\rm eff}$", fontsize=9)
    ax_deff.set_xlabel("calibration iteration")

    # Annotate near-1 value
    ax_deff.annotate(
        rf"$d_{{\rm eff}}\approx{d_eff.mean():.2f}$  (one direction dominates)",
        xy=(t_axis[d_eff.argmax()], d_eff.max()),
        xytext=(38, max(d_eff) * 1.08),
        fontsize=8, color=QUAL[0],
        arrowprops=dict(arrowstyle="->", color=QUAL[0], lw=0.9),
    )

    # ── Panel (c): bootstrap principal angles ─────────────────────────────────
    # v_1 (stiff, red)
    ax_ang.fill_between(snap_iters, ang_v1_p25, ang_v1_p75,
                        color=_STIFF_COLOR, alpha=0.18, lw=0)
    ax_ang.plot(snap_iters, ang_v1, color=_STIFF_COLOR, lw=2.0,
                marker="o", ms=4, label=r"$v_1$ (stiff)")

    # v_P (sloppy, grey)
    ax_ang.fill_between(snap_iters, ang_vP_p25, ang_vP_p75,
                        color=_SLOPPY_COLOR, alpha=0.18, lw=0)
    ax_ang.plot(snap_iters, ang_vP, color=_SLOPPY_COLOR, lw=2.0,
                marker="s", ms=4, label=r"$v_P$ (sloppy)")

    # 5° reference line
    ax_ang.axhline(5.0, color="#bbbbbb", ls="--", lw=0.9)
    ax_ang.text(snap_iters[-1] + 0.5, 5.0, "5°",
                va="center", fontsize=8, color="#999999")

    ax_ang.set_xlim(snap_iters[0] - 1, snap_iters[-1] + 2)
    ax_ang.set_ylim(0, max(ang_v1.max(), ang_vP.max()) * 1.25)
    ax_ang.set_xlabel("calibration iteration")
    ax_ang.set_ylabel("principal angle (°)", fontsize=9)
    ax_ang.legend(loc="upper right", fontsize=8.5, frameon=True)

    # Annotate early convergence of v_1
    idx_lt5 = int(np.where(ang_v1 < 5.0)[0][0]) if (ang_v1 < 5.0).any() else None
    if idx_lt5 is not None:
        ax_ang.annotate(
            rf"$v_1$ < 5° by iter {snap_iters[idx_lt5]}",
            xy=(snap_iters[idx_lt5], ang_v1[idx_lt5]),
            xytext=(snap_iters[idx_lt5] + 3, ang_v1[idx_lt5] + 5.5),
            fontsize=8, color=_STIFF_COLOR,
            arrowprops=dict(arrowstyle="->", color=_STIFF_COLOR, lw=0.9),
        )

    fig.align_ylabels(axes)
    save_vector(fig, OUT_NAME, out_dir=str(out_dir(OUT_AREA)))
    plt.close(fig)


if __name__ == "__main__":
    main()
