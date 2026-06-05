"""Script 25: Eigenvalue trajectory log λ_k(t) vs iteration for BH calibration.

Paper §4.1: tracks how identifiability geometry evolves along the calibration
path. Three regimes in one figure:
    Stable   — low beta (mean-reverting)
    Periodic — medium beta (oscillating)
    Chaotic  — high beta (the canonical setting used throughout the paper)

Each regime is a separate subplot. Within each, log10(λ_k) is plotted for
k=1..P across calibration iterations.

Run: uv run python scripts/25_eigenvalue_trajectory.py
"""
from __future__ import annotations

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from curvature_calib.calibration.calibrate import calibrate
from curvature_calib.calibration.per_seed_grads import vmap_simulate
from curvature_calib.models.brock_hommes import simulate
from curvature_calib.viz.style import apply_style, save

SIGMA = 0.05
R = 1.1
T = 200
M = 64
N_ITER = 60
M_REF = 128

REGIMES = {
    "stable":   jnp.array([1.0,  0.5,  0.0,  0.5, 0.0], dtype=jnp.float64),
    "periodic": jnp.array([5.0,  1.2,  0.0, -0.5, 0.0], dtype=jnp.float64),
    "chaotic":  jnp.array([10.0, 1.2,  0.2,  1.2, -0.2], dtype=jnp.float64),
}

# Perturbation from theta* to get a non-trivial starting gradient
DELTA = jnp.array([0.0, 0.0, 0.1, 0.0, 0.1], dtype=jnp.float64)

# Fixed seeds per regime (avoid hash() which is non-reproducible across processes)
REGIME_SEEDS = {"stable": 42, "periodic": 43, "chaotic": 44}


def sim(theta, key):
    return simulate(theta, key, T=T, sigma=SIGMA, R=R, x_init=0.0)


def run_regime(theta_star: jax.Array, regime_name: str) -> dict:
    print(f"  {regime_name}...")
    theta0 = theta_star + DELTA
    ref_keys = jax.random.split(jax.random.PRNGKey(0), M_REF)
    Y_ref = vmap_simulate(sim, theta_star, ref_keys)
    log = calibrate(sim, theta0, Y_ref, M=M, n_iter=N_ITER,
                    init_damping=100.0, verbose=False,
                    seed_base=REGIME_SEEDS[regime_name])
    return {
        "eigvals":   np.asarray(log.eigvals),    # (n_iter, P)
        "val_losses": np.asarray(log.val_losses),  # (n_iter+1,)
    }


def main():
    apply_style()

    results = {}
    for name, theta_star in REGIMES.items():
        results[name] = run_regime(theta_star, name)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)
    try:
        palette = cm.get_cmap("viridis", 5)
    except Exception:
        palette = plt.colormaps["viridis"].resampled(5)

    for ax, (regime, data) in zip(axes, results.items()):
        ev = data["eigvals"]        # (n_iter, P)
        iters = np.arange(len(ev))
        P = ev.shape[1]
        for k in range(P):
            vals = ev[:, k]
            vals = np.where(vals > 0, vals, np.nan)
            ax.plot(iters, np.log10(vals + 1e-30),
                    color=palette(k / max(P - 1, 1)),
                    lw=1.5,
                    label=f"$\\lambda_{k+1}$" if k < 2 or k == P - 1 else "_nolegend_")
        ax.set_xlabel("iteration")
        ax.set_ylabel(r"$\log_{10} \lambda_k$")
        ax.set_title(f"({chr(97 + list(results.keys()).index(regime))}) {regime}",
                     fontweight="bold")
        ax.legend(fontsize=8)

    fig.suptitle("OPG eigenvalue trajectories — Brock-Hommes calibration",
                 fontweight="bold", fontsize=13)
    fig.tight_layout()

    out_dir = "outputs/brock_hommes"
    p = save(fig, "25_eigenvalue_trajectory.png", out_dir=out_dir)
    np.savez_compressed(f"{out_dir}/25_eigenvalue_trajectory.npz",
                        **{name: d["eigvals"] for name, d in results.items()})
    print(f"Saved {p}")


if __name__ == "__main__":
    main()
