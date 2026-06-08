"""Booklet 2, Figure 10: Bootstrap CIs and noise floor for the OPG spectrum.

Two panels:
  (a) OPG spectrum with bootstrap whiskers (2.5–97.5 %) per eigenvalue
  (b) Signal-to-noise ratio λ_k / std_boot(λ_k) — stiff directions are
      well above the noise floor; sloppy eigenvalues vanish into it

Re-uses the same BH computation as b2_07 (bootstrap_eigvals at θ*).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from curvature_calib.calibration.opg import bootstrap_eigvals, eigendecompose  # noqa: E402
from curvature_calib.calibration.per_seed_grads import (  # noqa: E402
    per_seed_loss_and_grads,
    vmap_simulate,
)
from curvature_calib.models.brock_hommes import simulate  # noqa: E402
from curvature_calib.viz.booklet_annotate import SLOPPY_COLOR, STIFF_COLOR  # noqa: E402
from curvature_calib.viz.booklet_style import QUAL, apply_booklet_style, save_vector  # noqa: E402
from scripts.booklets._figbase import out_dir  # noqa: E402

OUT_AREA = "methodology"
OUT_NAME = "fig_10_bootstrap"

T = 200
SIGMA = 0.05
R = 1.1
THETA_STAR = jnp.array([3.0, 1.2, 0.2, 1.2, -0.2], dtype=jnp.float64)
N_BOOT = 500
M = 200


def _sim(theta, key):
    return simulate(theta, key, T=T, sigma=SIGMA, R=R, x_init=0.0)


def main() -> None:
    apply_booklet_style()

    # ── Compute OPG and bootstrap at θ* ───────────────────────────────────────
    ref_keys = jax.random.split(jax.random.PRNGKey(0), M)
    Y_ref = vmap_simulate(_sim, THETA_STAR, ref_keys)
    keys = jax.random.split(jax.random.PRNGKey(7), M)
    stats = per_seed_loss_and_grads(_sim, THETA_STAR, keys, Y_ref)

    F = np.asarray(stats.opg)
    eig = eigendecompose(jnp.asarray(F))
    eigvals = np.asarray(eig.eigvals)          # (P,)

    boot = np.asarray(                         # (N_BOOT, P)
        bootstrap_eigvals(stats.per_seed_grads, n_boot=N_BOOT,
                          key=jax.random.PRNGKey(42))
    )

    P = len(eigvals)
    xs = np.arange(P)
    labels = [rf"$v_{k+1}$" for k in xs]

    lo_25 = np.percentile(boot, 25, axis=0)
    hi_75 = np.percentile(boot, 75, axis=0)
    lo_25_2 = np.percentile(boot, 2.5, axis=0)
    hi_975 = np.percentile(boot, 97.5, axis=0)
    med = np.median(boot, axis=0)
    snr = eigvals / np.std(boot, axis=0)

    # Noise floor: bootstrap std of the smallest eigenvalue
    noise_floor = np.std(boot[:, -1])

    print("eigenvalues:", eigvals)
    print("bootstrap IQR (25–75%):", lo_25, hi_75)
    print("SNR:", snr)
    print("noise_floor (std of λ_P boot):", noise_floor)

    # ── Figure ─────────────────────────────────────────────────────────────────
    fig, (ax, ax_snr) = plt.subplots(1, 2, figsize=(11, 4.8),
                                     gridspec_kw={"wspace": 0.38})

    # ── Panel (a): spectrum + bootstrap CIs ──────────────────────────────────
    # 2.5–97.5% whiskers
    for k in range(P):
        ax.plot([xs[k], xs[k]], [lo_25_2[k], hi_975[k]],
                color=QUAL[0], lw=1.2, alpha=0.5)
    # IQR box
    for k in range(P):
        ax.plot([xs[k] - 0.2, xs[k] + 0.2], [lo_25[k], lo_25[k]],
                color=QUAL[0], lw=1.5)
        ax.plot([xs[k] - 0.2, xs[k] + 0.2], [hi_75[k], hi_75[k]],
                color=QUAL[0], lw=1.5)
        ax.plot([xs[k] - 0.2, xs[k] + 0.2], [xs[k]*0 + med[k], xs[k]*0 + med[k]],
                color="white", lw=2.5, zorder=5)
    # Point estimate
    ax.scatter(xs, eigvals, s=70, color=QUAL[0], zorder=6,
               edgecolor="white", linewidth=1.2)

    # Noise floor line (bootstrap std of λ_P)
    ax.axhline(noise_floor, ls="--", color="#e74c3c", lw=1.3, alpha=0.8,
               label="noise floor\n(std of boot. $\\lambda_P$)")

    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"eigenvalue $\lambda_k$")
    ax.set_title("(a)  OPG spectrum — bootstrap CIs", fontweight="bold")
    ax.legend(fontsize=8.5, loc="lower left", frameon=True)

    span = np.log10(eigvals[0] / max(eigvals[-1], 1e-30))
    ax.text(0.97, 0.97,
            f"span: {span:.1f} OOM",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8.5, color="#555",
            bbox=dict(fc="white", ec="grey", alpha=0.8,
                      boxstyle="round,pad=0.3"))

    ax.text(xs[0], eigvals[0] * 2.5, "stiff",
            ha="center", va="bottom", fontsize=9,
            color=STIFF_COLOR, fontweight="bold")
    ax.text(xs[-1], eigvals[-1] * 0.32, "sloppy",
            ha="center", va="top", fontsize=9,
            color=SLOPPY_COLOR, fontweight="bold")

    # ── Panel (b): SNR ────────────────────────────────────────────────────────
    bar_colors = [STIFF_COLOR if s > 10 else
                  ("#e67e22" if s > 2 else SLOPPY_COLOR)
                  for s in snr]
    ax_snr.bar(xs, snr, color=bar_colors, alpha=0.82, width=0.6,
               edgecolor="white", linewidth=0.8)
    ax_snr.axhline(2.0, color="#bbbbbb", ls="--", lw=1.0)
    ax_snr.text(P - 0.5, 2.3, "SNR = 2", ha="right",
                fontsize=8, color="#999")
    ax_snr.set_xticks(xs)
    ax_snr.set_xticklabels(labels)
    ax_snr.set_ylabel(r"SNR  $= \lambda_k\,/\,\hat\sigma_k^{\,\rm boot}$")
    ax_snr.set_title("(b)  Signal-to-noise by direction", fontweight="bold")
    ax_snr.set_yscale("log")

    ax_snr.text(0.97, 0.97,
                "red  = stiff (high SNR)\ngrey = sloppy (noise floor)",
                transform=ax_snr.transAxes, ha="right", va="top",
                fontsize=8, color="#555",
                bbox=dict(fc="white", ec="grey", alpha=0.8,
                          boxstyle="round,pad=0.3"))

    fig.suptitle(
        "Stiff eigenvalues are robustly identified; sloppy ones merge with the noise floor",
        fontweight="bold",
    )

    save_vector(fig, OUT_NAME, out_dir=str(out_dir(OUT_AREA)))
    plt.close(fig)


if __name__ == "__main__":
    main()
