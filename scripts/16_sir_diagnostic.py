"""SIR diagnostic — the project plan's Phase 3 generalisation, Tier A.

Runs the OPG diagnostic + falsification protocol on the mean-field SIR
model (`curvature_calib.models.sir`), exactly the same pipeline used for
Brock-Hommes but with a different simulator. The point is to show that
the diagnostic mechanism is model-agnostic.

Produces one multi-panel figure with:
    A. Sample SIR trajectories at theta* (and lockdown counterfactual).
    B. OPG eigenspectrum at theta* with 95% bootstrap CI.
    C. |V| heatmap of the eigendecomposition (parameter content).
    D. §5.4 falsification on SIR: stiff/sloppy perturbations under three
       non-MMD discrepancies (moments, ACF, tail quantiles).
    E. Stiff/sloppy aggregate discrepancy heatmap.

Run: uv run python scripts/16_sir_diagnostic.py
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sps

from curvature_calib.calibration.opg import (
    bootstrap_eigvals, eigendecompose,
)
from curvature_calib.calibration.per_seed_grads import (
    per_seed_loss_and_grads, vmap_simulate,
)
from curvature_calib.losses.mmd import mmd_sq_with_median_bandwidth
from curvature_calib.models.sir import simulate
from curvature_calib.viz.style import QUAL, apply_style


T = 200
N_POP = 1e5
SIGMA_OBS = 10.0
# theta* = (beta=0.4, gamma=0.1, I0_frac=1e-3, t_lock_norm=0.4, f_lock=0.5)
# R0 = beta/gamma = 4 -> nontrivial epidemic; mid-trajectory 50% lockdown.
THETA_STAR = jnp.array([0.40, 0.10, 1e-3, 0.40, 0.50])
PARAM_NAMES = [r"$\beta$", r"$\gamma$", r"$I_0$", r"$t_{\mathrm{lock}}$",
               r"$f_{\mathrm{lock}}$"]


def _sim(theta, key):
    return simulate(theta, key, T=T, N=N_POP, sigma_obs=SIGMA_OBS,
                    grad_horizon=None)


def autocorr_mean(X, max_lag=20):
    out = np.zeros(max_lag + 1)
    for m in range(X.shape[0]):
        x = X[m] - X[m].mean()
        var = x.var() + 1e-12
        out += np.array([np.mean(x[: x.size - k] * x[k:]) / var
                         for k in range(max_lag + 1)])
    return out / X.shape[0]


def four_moments(X):
    x = X.reshape(-1)
    return np.array([x.mean(), x.std(),
                     float(sps.skew(x)), float(sps.kurtosis(x))])


def tail_quantiles(X, qs=(0.01, 0.05, 0.95, 0.99)):
    x = X.reshape(-1)
    return np.array([np.quantile(x, q) for q in qs])


def main() -> None:
    apply_style()
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    # Reference distribution at theta*.
    print("Building reference at theta*...")
    M_ref = 128
    ref_keys = jax.random.split(jax.random.PRNGKey(0), M_ref)
    Y_ref = vmap_simulate(_sim, THETA_STAR, ref_keys)

    # OPG eigendecomposition at theta*.
    print("Computing F_hat(theta*) eigenbasis...")
    M_eig = 96
    eig_keys = jax.random.split(jax.random.PRNGKey(1), M_eig)
    # Evaluate at theta_eval slightly off theta* so the gradient signal is
    # non-vanishing (at theta* exactly mean grad is ~0 by optimality).
    theta_eval = THETA_STAR + jnp.array([0.01, -0.005, 1e-4, 0.02, -0.02])
    stats = per_seed_loss_and_grads(_sim, theta_eval, eig_keys, Y_ref)
    eig = eigendecompose(stats.opg)
    eigvals = np.asarray(eig.eigvals)
    V = np.asarray(eig.eigvecs)
    print(f"  eigenvalues: {eigvals}")
    print(f"  condition  : {eigvals[0] / max(eigvals[-1], 1e-30):.2e}")

    boot = np.asarray(bootstrap_eigvals(
        stats.per_seed_grads, n_boot=300,
        key=jax.random.PRNGKey(7)))
    boot_lo = np.percentile(boot, 2.5, axis=0)
    boot_hi = np.percentile(boot, 97.5, axis=0)

    # Sample trajectories: theta*, and a "no lockdown" counterfactual.
    print("Sample trajectories...")
    M_show = 64
    show_keys = jax.random.split(jax.random.PRNGKey(11), M_show)
    X_star = np.asarray(vmap_simulate(_sim, THETA_STAR, show_keys))
    theta_no_lock = THETA_STAR.at[4].set(1.0)
    X_no_lock = np.asarray(vmap_simulate(_sim, theta_no_lock, show_keys))

    # §5.4 falsification on SIR.
    print("§5.4 falsification at theta*...")
    v_stiff = V[:, 0]
    v_sloppy = V[:, -1]
    alpha = 1e-2
    falsify_keys = jax.random.split(jax.random.PRNGKey(404), 128)

    def _sim_at(theta):
        return np.asarray(vmap_simulate(_sim, theta, falsify_keys))

    X_T = _sim_at(THETA_STAR)
    X_ps = _sim_at(THETA_STAR + alpha * jnp.asarray(v_stiff))
    X_ms = _sim_at(THETA_STAR - alpha * jnp.asarray(v_stiff))
    X_pl = _sim_at(THETA_STAR + alpha * jnp.asarray(v_sloppy))
    X_ml = _sim_at(THETA_STAR - alpha * jnp.asarray(v_sloppy))

    def disc(X_a):
        return {
            "moments": float(np.sum(np.abs(four_moments(X_a) - four_moments(X_T)))),
            "ACF":     float(np.sum(np.abs(autocorr_mean(X_a) - autocorr_mean(X_T)))),
            "quant":   float(np.sum(np.abs(tail_quantiles(X_a) - tail_quantiles(X_T)))),
        }

    results = {
        r"$+\alpha v_1$ (stiff)":  disc(X_ps),
        r"$-\alpha v_1$ (stiff)":  disc(X_ms),
        r"$+\alpha v_P$ (sloppy)": disc(X_pl),
        r"$-\alpha v_P$ (sloppy)": disc(X_ml),
    }

    # ============================================================ FIGURE
    fig = plt.figure(figsize=(17, 11))
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.36)

    # A. Sample trajectories (with-lockdown vs without)
    ax = fig.add_subplot(gs[0, 0])
    for m in range(min(M_show, 12)):
        ax.plot(X_star[m], color=QUAL[0], lw=0.6, alpha=0.4)
        ax.plot(X_no_lock[m], color=QUAL[1], lw=0.6, alpha=0.4)
    ax.plot(X_star.mean(0), color=QUAL[0], lw=2.4, label=r"$\theta^*$ (with lockdown)")
    ax.plot(X_no_lock.mean(0), color=QUAL[1], lw=2.4,
            label=r"no lockdown ($f_{\mathrm{lock}}=1$)")
    ax.set_xlabel("day")
    ax.set_ylabel("daily incidence (cases)")
    ax.set_title("A. Sample SIR trajectories at $\\theta^*$ + counterfactual")
    ax.legend(fontsize=9)

    # B. OPG spectrum + bootstrap CI.
    ax = fig.add_subplot(gs[0, 1])
    P = eigvals.size
    xs = np.arange(P)
    le = np.clip(eigvals - boot_lo, a_min=0.0, a_max=None)
    ue = np.clip(boot_hi - eigvals, a_min=0.0, a_max=None)
    ax.errorbar(xs, eigvals, yerr=[le, ue], fmt="o",
                color=QUAL[0], capsize=4, markersize=11,
                markerfacecolor=QUAL[0], markeredgecolor="white",
                markeredgewidth=1.5, lw=1.8)
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"$v_{k+1}$" for k in xs])
    ax.set_xlabel("eigendirection")
    ax.set_ylabel(r"$\lambda_k$ (log)")
    span = eigvals[0] / max(eigvals[-1], 1e-30)
    ax.set_title(f"B. OPG spectrum at $\\theta^*$ + 95% boot CI (span {span:.0e})")

    # C. |V| heatmap.
    ax = fig.add_subplot(gs[0, 2])
    V_abs = np.abs(V)
    im = ax.imshow(V_abs, cmap="magma", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"$v_{k+1}$" for k in xs])
    ax.set_yticks(np.arange(P))
    ax.set_yticklabels(PARAM_NAMES)
    for i in range(P):
        for j in range(P):
            ax.text(j, i, f"{V_abs[i, j]:.2f}", ha="center", va="center",
                    color="white" if V_abs[i, j] < 0.55 else "black", fontsize=9)
    ax.set_title(r"C. $|V|$ at $\theta^*$ — parameter content of each direction")
    plt.colorbar(im, ax=ax, label=r"$|v_{k,j}|$")

    # D. Falsification bar chart (moments, ACF, quantiles).
    ax = fig.add_subplot(gs[1, 0:2])
    channels = ["moments", "ACF", "quant"]
    xs_b = np.arange(len(channels))
    width = 0.2
    bar_specs = [
        (r"$+\alpha v_1$ (stiff)",  QUAL[1], 1.0),
        (r"$-\alpha v_1$ (stiff)",  QUAL[1], 0.55),
        (r"$+\alpha v_P$ (sloppy)", QUAL[2], 1.0),
        (r"$-\alpha v_P$ (sloppy)", QUAL[2], 0.55),
    ]
    for i, (name, color, alpha_b) in enumerate(bar_specs):
        vals = [results[name][c] for c in channels]
        ax.bar(xs_b + (i - 1.5) * width, vals, width,
               color=color, alpha=alpha_b, edgecolor="white", label=name)
    ax.set_xticks(xs_b)
    ax.set_xticklabels([r"moments ($\sum|\Delta|$)",
                        r"ACF ($\sum|\Delta|$)",
                        r"tail quantiles ($\sum|\Delta|$)"])
    ax.set_ylabel("aggregate discrepancy (log)")
    ax.set_yscale("log")
    ax.set_title(r"D. §5.4 Falsification on SIR (same $\alpha$, three non-MMD discrepancies)")
    ax.legend(fontsize=9, ncol=2)

    # E. Aggregate stiff/sloppy heatmap.
    ax = fig.add_subplot(gs[1, 2])
    summary = np.zeros((4, 3))
    for j, name in enumerate(results):
        summary[j, 0] = results[name]["moments"]
        summary[j, 1] = results[name]["ACF"]
        summary[j, 2] = results[name]["quant"]
    im = ax.imshow(np.log10(summary + 1e-12), cmap="magma", aspect="auto")
    ax.set_yticks(range(4))
    ax.set_yticklabels(list(results.keys()))
    ax.set_xticks(range(3))
    ax.set_xticklabels(["moments", "ACF", "quantiles"])
    for i in range(4):
        for j in range(3):
            ax.text(j, i, f"{summary[i, j]:.2e}", ha="center", va="center",
                    color="white" if np.log10(summary[i, j] + 1e-12) <
                                     np.log10(summary).mean() else "black",
                    fontsize=8)
    plt.colorbar(im, ax=ax, label=r"$\log_{10}$ aggregate")
    ax.set_title("E. Aggregate discrepancy heatmap")

    fig.suptitle(
        "Phase 3 generalisation (Tier A): OPG diagnostic on mean-field SIR",
        fontsize=14, fontweight="bold", y=0.995,
    )
    p = out_dir / "16_sir_diagnostic.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {p}")

    # ============================================================ verdict
    print("\n" + "=" * 70)
    print("SIR DIAGNOSTIC VERDICT")
    print("=" * 70)
    print(f"\nEigenvalues at theta*: {eigvals}")
    print(f"Condition number (span): {span:.2e}")
    print()
    print("Falsification ratios (stiff / sloppy):")
    for ch in channels:
        r_stiff = results[r"$+\alpha v_1$ (stiff)"][ch]
        r_sloppy = results[r"$+\alpha v_P$ (sloppy)"][ch]
        print(f"  {ch:<10s}: {r_stiff / max(r_sloppy, 1e-30):.0f}x")

    print("\nDominant eigenvector v_1 (stiffest direction):")
    v1 = V[:, 0]
    for k in range(P):
        print(f"  {PARAM_NAMES[k]:<10s}: {v1[k]:+.3f}")
    print("\nSloppiest eigenvector v_P:")
    vp = V[:, -1]
    for k in range(P):
        print(f"  {PARAM_NAMES[k]:<10s}: {vp[k]:+.3f}")


if __name__ == "__main__":
    main()
