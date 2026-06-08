"""Booklet 1, Figure 9: network-SIR epidemic curves, lockdown on vs off."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import jax  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from curvature_calib.models.network_sir import simulate as net_simulate  # noqa: E402
from curvature_calib.viz.booklet_annotate import brace  # noqa: E402
from curvature_calib.viz.booklet_style import QUAL, apply_booklet_style, save_vector  # noqa: E402
from scripts.booklets._figbase import out_dir  # noqa: E402

OUT_AREA = "models"
OUT_NAME = "fig_09_sir_trajectories"

# Match script 18 constants exactly (T, N_NODES, MEAN_DEG are the script-18 values).
T = 200
N_NODES = 250
MEAN_DEG = 6.0
GUMBEL_TAU = 0.5

# Canonical operating point from scripts/18.  beta=0.30, gamma=0.10 give R_0~18
# so the epidemic peaks around day 2–3.  For this concept figure we keep all
# epidemic parameters identical (beta, gamma, I0_frac, f_lock) but set
# t_lock_norm to 0.01 so the sigmoid lockdown engages on day 2, just as the
# epidemic ignites — this makes the "flattening" effect visually unambiguous.
THETA_STAR = jnp.array([0.30, 0.10, 0.05, 0.40, 0.50])

# Visualization theta: override only t_lock_norm for a clear pedagogical figure.
_T_LOCK_NORM_VIZ = 0.01          # day  ~2  (sigmoid is sharp, k_sig=20)
THETA_VIZ = THETA_STAR.at[3].set(_T_LOCK_NORM_VIZ)
THETA_VIZ_NO_LOCK = THETA_VIZ.at[4].set(1.0)   # f_lock = 1 → no reduction

T_LOCK_DAY = int(round(_T_LOCK_NORM_VIZ * T))   # = 2


def _sim(theta: jax.Array, key: jax.Array) -> jax.Array:
    return net_simulate(
        theta, key,
        T=T, N=N_NODES, mean_degree=MEAN_DEG,
        gumbel_tau=GUMBEL_TAU, grad_horizon=None,
    )


def main() -> None:
    apply_booklet_style()

    t0 = time.perf_counter()

    M = 12
    keys = jax.random.split(jax.random.PRNGKey(0), M)

    # With lockdown (f_lock = 0.50, lockdown engages at day T_LOCK_DAY)
    X_lock = np.asarray(
        jax.vmap(lambda k: _sim(THETA_VIZ, k))(keys)
    )  # (M, T)

    # No lockdown (f_lock = 1.0 → beta_eff = beta throughout)
    X_no_lock = np.asarray(
        jax.vmap(lambda k: _sim(THETA_VIZ_NO_LOCK, k))(keys)
    )  # (M, T)

    elapsed = time.perf_counter() - t0
    print(f"Simulations done in {elapsed:.1f}s  (M={M}, T={T}, N={N_NODES})")

    mean_lock = X_lock.mean(axis=0)
    std_lock = X_lock.std(axis=0)
    mean_no = X_no_lock.mean(axis=0)
    std_no = X_no_lock.std(axis=0)

    peak_no_idx = int(np.argmax(mean_no))
    peak_lock_idx = int(np.argmax(mean_lock))
    peak_no_val = float(mean_no[peak_no_idx])
    peak_lock_val = float(mean_lock[peak_lock_idx])

    days = np.arange(T)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))

    # No-lockdown trajectory (QUAL[1] = red)
    ax.fill_between(days, mean_no - std_no, mean_no + std_no,
                    color=QUAL[1], alpha=0.15)
    ax.plot(days, mean_no, color=QUAL[1], lw=2.0,
            label=r"No lockdown ($f_{\rm lock}=1.0$)")

    # With-lockdown trajectory (QUAL[0] = blue)
    ax.fill_between(days, mean_lock - std_lock, mean_lock + std_lock,
                    color=QUAL[0], alpha=0.15)
    ax.plot(days, mean_lock, color=QUAL[0], lw=2.0,
            label=r"With lockdown ($f_{\rm lock}=0.5$)")

    # Vertical dashed line at lockdown engagement
    ax.axvline(T_LOCK_DAY, color="#7f8c8d", lw=1.2, ls="--", zorder=2)

    # Lockdown label: place just to the right of the line, near top
    y_max_data = max(peak_no_val, peak_lock_val)
    ax.text(T_LOCK_DAY + 1.5, y_max_data * 0.97,
            "lockdown\nengages", color="#7f8c8d", fontsize=7.5,
            va="top", ha="left", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#7f8c8d",
                      lw=0.6, alpha=0.85))

    # Double-headed arrow annotation: highlight the vertical gap at the peak day
    # (both peaks occur at the same day; arrow shows how much the peak is reduced)
    anno_x = peak_no_idx + 3   # offset right so arrow is readable
    ax.annotate(
        "",
        xy=(anno_x, peak_lock_val),
        xytext=(anno_x, peak_no_val),
        arrowprops=dict(arrowstyle="<->", color="#2c3e50", lw=1.4),
    )
    reduction = peak_no_val - peak_lock_val
    ax.text(anno_x + 1.5, (peak_no_val + peak_lock_val) / 2,
            f"−{reduction:.0f}\ncases/day",
            color="#2c3e50", fontsize=8, va="center", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="white",
                      ec="#2c3e50", lw=0.7, alpha=0.9))

    # Horizontal brace spanning the full epidemic window (days 0–30)
    brace(ax, x0=0.0, x1=30.0, y=peak_no_val * 1.18,
          text="lockdown flattens peak",
          color="#2c3e50", fontsize=8, dy=1.5)

    ax.set_xlabel("time (days)")
    ax.set_ylabel("daily incidence")
    ax.set_title(
        "Network-SIR: lockdown flattens the epidemic curve",
        fontweight="bold",
    )
    ax.legend(frameon=True, loc="upper right")
    ax.set_xlim(0, 60)   # zoom into the first 60 days where the action is
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    paths = save_vector(fig, OUT_NAME, out_dir=str(out_dir(OUT_AREA)))
    plt.close(fig)

    print(f"Saved PDF: {paths['pdf']}")
    print(f"Saved PNG: {paths['png']}")
    print(
        f"T={T}, N_NODES={N_NODES}, MEAN_DEG={MEAN_DEG}  |  "
        f"peak_no_lock={peak_no_val:.1f} @ day {peak_no_idx}  "
        f"peak_lock={peak_lock_val:.1f} @ day {peak_lock_idx}  "
        f"reduction={peak_no_val - peak_lock_val:.1f}"
    )


if __name__ == "__main__":
    main()
