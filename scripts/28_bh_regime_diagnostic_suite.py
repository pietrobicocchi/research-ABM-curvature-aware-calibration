"""Script 28: BH diagnostic suite across the three dynamical regimes.

Tier-1.1 gap fill. For each regime (stable, periodic, chaotic) — same
calibration runs as script 25 — produces a 3-panel figure:
    (a) eigenvalue trajectory log10 lambda_k(t) across calibration
    (b) eigenvalue spectrum at convergence + 95% bootstrap CI + d_eff
    (c) eigenvector content heatmap |V| at convergence

Script 25 only produced panel (a) for all three regimes; panels (b)/(c)
previously existed only for the canonical (chaotic-style) theta* via
script 21 (fig1_spectrum). This script completes the suite.

Run: uv run python scripts/28_bh_regime_diagnostic_suite.py
"""
from __future__ import annotations

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from curvature_calib.calibration.bootstrap import bootstrap_eigvals, eigenvalue_cis
from curvature_calib.calibration.calibrate import calibrate
from curvature_calib.calibration.diagnostic import d_eff_from_bootstrap
from curvature_calib.calibration.per_seed_grads import vmap_simulate
from curvature_calib.models.brock_hommes import simulate
from curvature_calib.viz.style import apply_style, save

SIGMA = 0.05
R = 1.1
T = 200
M = 64
N_ITER = 60
M_REF = 128
N_BOOT = 500

PARAM_NAMES = [r"$\beta$", r"$g_1$", r"$b_1$", r"$g_2$", r"$b_2$"]

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
    seed_base = REGIME_SEEDS[regime_name]
    log = calibrate(sim, theta0, Y_ref, M=M, n_iter=N_ITER,
                    init_damping=100.0, verbose=False,
                    seed_base=seed_base)

    eigvals_traj = np.asarray(log.eigvals)        # (n_iter, P)
    eigvals_T = eigvals_traj[-1]                  # (P,)
    V_T = np.asarray(log.eigvecs[-1])             # (P, P)
    grads_T = log.per_seed_grads[-1]              # (M, P)

    boot = np.asarray(bootstrap_eigvals(grads_T, n_boot=N_BOOT,
                                        key=jax.random.PRNGKey(seed_base + 1000)))
    cis = np.asarray(eigenvalue_cis(jnp.asarray(boot)))
    d_eff = d_eff_from_bootstrap(jnp.asarray(cis))

    return {
        "eigvals_traj": eigvals_traj,
        "eigvals_T": eigvals_T,
        "V_T": V_T,
        "boot": boot,
        "cis": cis,
        "d_eff": d_eff,
        "theta_T": np.asarray(log.thetas[-1]),
    }


def plot_regime(regime: str, data: dict, out_dir: str) -> None:
    P = data["eigvals_T"].shape[0]
    xs = np.arange(P)

    fig = plt.figure(figsize=(15, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.0, 1.2], wspace=0.32)

    # (a) eigenvalue trajectory
    ax = fig.add_subplot(gs[0, 0])
    try:
        palette = cm.get_cmap("viridis", P)
    except Exception:
        palette = plt.colormaps["viridis"].resampled(P)
    ev = data["eigvals_traj"]
    iters = np.arange(len(ev))
    for k in range(P):
        vals = np.where(ev[:, k] > 0, ev[:, k], np.nan)
        ax.plot(iters, np.log10(vals + 1e-30), color=palette(k / max(P - 1, 1)),
                lw=1.5, label=f"$\\lambda_{k+1}$" if k < 2 or k == P - 1 else "_nolegend_")
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"$\log_{10}\lambda_k$")
    ax.set_title("(a) eigenvalue trajectory", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)

    # (b) spectrum at convergence + bootstrap CI + d_eff
    ax2 = fig.add_subplot(gs[0, 1])
    eigvals_T = data["eigvals_T"]
    boot = data["boot"]
    lo = np.percentile(boot, 2.5, axis=0)
    hi = np.percentile(boot, 97.5, axis=0)
    lower_err = np.clip(eigvals_T - lo, a_min=0.0, a_max=None)
    upper_err = np.clip(hi - eigvals_T, a_min=0.0, a_max=None)
    plot_vals = np.clip(eigvals_T, 1e-30, None)
    ax2.errorbar(xs, plot_vals, yerr=[lower_err, upper_err],
                 fmt="o", color="#1f3a93", markersize=10, capsize=4,
                 markerfacecolor="#1f3a93", markeredgecolor="white",
                 markeredgewidth=1.3)
    ax2.set_yscale("log")
    ax2.set_xticks(xs)
    ax2.set_xticklabels([f"$v_{k+1}$" for k in xs])
    ax2.set_xlabel("eigenvector (descending)")
    ax2.set_ylabel(r"$\lambda_k$")
    ax2.set_title("(b) spectrum at convergence + 95% CI", fontsize=11, fontweight="bold")
    span_oom = np.log10(max(eigvals_T[0], 1e-30) / max(eigvals_T[-1], 1e-30))
    ax2.text(0.04, 0.04,
             f"span: {span_oom:.1f} OOM\n"
             rf"$d_\mathrm{{eff}}$: {data['d_eff']}",
             transform=ax2.transAxes, fontsize=9, va="bottom",
             bbox=dict(facecolor="white", edgecolor="grey", alpha=0.85,
                       boxstyle="round,pad=0.3"))

    # (c) eigenvector content heatmap |V|
    ax3 = fig.add_subplot(gs[0, 2])
    V = data["V_T"]
    im = ax3.imshow(np.abs(V), cmap="magma", aspect="auto", vmin=0, vmax=1)
    ax3.set_xticks(xs)
    ax3.set_xticklabels([f"$v_{k+1}$" for k in xs])
    ax3.set_yticks(np.arange(P))
    ax3.set_yticklabels(PARAM_NAMES)
    ax3.set_title("(c) parameter content $|V|$", fontsize=11, fontweight="bold")
    plt.colorbar(im, ax=ax3, label=r"$|v_{k,j}|$")
    for i in range(P):
        for j in range(P):
            val = np.abs(V[i, j])
            ax3.text(j, i, f"{val:.2f}", ha="center", va="center",
                     color="white" if val < 0.55 else "black", fontsize=8)

    fig.suptitle(f"BH OPG diagnostic — {regime} regime",
                 fontsize=13, fontweight="bold", y=1.04)

    p = save(fig, f"28_bh_diagnostic_{regime}.png", out_dir=out_dir)
    print(f"  saved {p}  (d_eff={data['d_eff']}, span={span_oom:.1f} OOM)")


def main():
    apply_style()
    out_dir = "outputs/brock_hommes"

    results = {}
    for name, theta_star in REGIMES.items():
        results[name] = run_regime(theta_star, name)
        plot_regime(name, results[name], out_dir)

    np.savez_compressed(
        f"{out_dir}/28_bh_diagnostic_suite.npz",
        **{f"{name}_{key}": val for name, data in results.items()
           for key, val in data.items()},
    )
    print(f"saved {out_dir}/28_bh_diagnostic_suite.npz")


if __name__ == "__main__":
    main()
