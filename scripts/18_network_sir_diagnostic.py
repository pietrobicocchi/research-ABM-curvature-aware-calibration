"""Network-SIR diagnostic — Phase 3 Tier B.

Same OPG eigendecomposition + §5.4 falsification protocol as scripts 06, 08,
13, 16, applied to the *discrete-state network-SIR* with Gumbel-Sigmoid
surrogate gradients. The question this script answers:

    Does the diagnostic survive the surrogate-gradient bias regime
    the project was originally designed for?

If eigenvectors qualitatively match the mean-field SIR (I_0 stiff,
lockdown direction sloppy) and falsification ratios remain large under
the three non-MMD discrepancies, then YES -- diagnostic is robust to
surrogate-gradient bias and the paper's "generalises" claim survives
the hard regime.

If eigenvectors drift dramatically or falsification ratios collapse,
that is itself a paper-worthy result: the diagnostic has a regime of
validity, and the user should be told. Either outcome strengthens the
work.

Produces a 6-panel figure mirroring `scripts/16_sir_diagnostic.py`:
    A. Sample network-SIR trajectories at theta* (with vs without lockdown).
    B. OPG eigenspectrum at theta* with 95% bootstrap CI.
    C. |V| heatmap of the eigendecomposition.
    D. §5.4 falsification on network-SIR: stiff/sloppy under three non-MMD
       discrepancies.
    E. Stiff/sloppy aggregate discrepancy heatmap.
    F. Side-by-side comparison of mean-field SIR vs network-SIR spectra,
       so the surrogate-bias effect is visible at a glance.

Run: uv run python scripts/18_network_sir_diagnostic.py
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sps

from curvature_calib.calibration.opg import bootstrap_eigvals, eigendecompose
from curvature_calib.calibration.per_seed_grads import (
    per_seed_loss_and_grads, vmap_simulate,
)
from curvature_calib.models.network_sir import simulate as net_simulate
from curvature_calib.models.sir import simulate as mf_simulate
from curvature_calib.viz.style import QUAL, apply_style


# Network-SIR specifics. beta is per-contact rate; with mean degree 6 and
# gamma=0.1 the basic reproduction number R_0 ~ beta * mean_degree / gamma.
# Choosing beta=0.30 -> R_0 ~ 18, a vigorous epidemic.
T = 200
N_NODES = 250
MEAN_DEG = 6.0
GUMBEL_TAU = 0.5
SIGMA_OBS_FALSIFY = 0.0   # observable is integer count, no extra noise
THETA_STAR = jnp.array([0.30, 0.10, 0.05, 0.40, 0.50])
PARAM_NAMES = [r"$\beta$", r"$\gamma$", r"$I_0$",
               r"$t_{\mathrm{lock}}$", r"$f_{\mathrm{lock}}$"]


def _sim(theta, key):
    return net_simulate(theta, key, T=T, N=N_NODES, mean_degree=MEAN_DEG,
                        gumbel_tau=GUMBEL_TAU, grad_horizon=None)


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
    out_dir = Path("outputs/sir")
    out_dir.mkdir(exist_ok=True, parents=True)

    print(f"Network-SIR: N={N_NODES}, mean_degree={MEAN_DEG}, "
          f"gumbel_tau={GUMBEL_TAU}, T={T}")
    print(f"theta_star = {np.asarray(THETA_STAR)}")

    # Reference at theta*.
    print("Building reference at theta*...")
    M_ref = 96
    ref_keys = jax.random.split(jax.random.PRNGKey(0), M_ref)
    Y_ref = vmap_simulate(_sim, THETA_STAR, ref_keys)

    # OPG eigendecomposition at theta_eval slightly off theta*.
    print("Computing F_hat eigenbasis...")
    M_eig = 96
    eig_keys = jax.random.split(jax.random.PRNGKey(1), M_eig)
    theta_eval = THETA_STAR + jnp.array([0.005, -0.003, 0.005, 0.02, -0.02])
    stats = per_seed_loss_and_grads(_sim, theta_eval, eig_keys, Y_ref)
    eig = eigendecompose(stats.opg)
    eigvals = np.asarray(eig.eigvals)
    V = np.asarray(eig.eigvecs)
    print(f"  eigenvalues: {eigvals}")
    print(f"  condition number: {eigvals[0] / max(eigvals[-1], 1e-30):.2e}")

    boot = np.asarray(bootstrap_eigvals(
        stats.per_seed_grads, n_boot=300, key=jax.random.PRNGKey(7)))
    boot_lo = np.percentile(boot, 2.5, axis=0)
    boot_hi = np.percentile(boot, 97.5, axis=0)

    # Sample trajectories (with vs without lockdown counterfactual).
    print("Sample trajectories...")
    M_show = 32
    show_keys = jax.random.split(jax.random.PRNGKey(11), M_show)
    X_star = np.asarray(vmap_simulate(_sim, THETA_STAR, show_keys))
    theta_no_lock = THETA_STAR.at[4].set(1.0)
    X_no_lock = np.asarray(vmap_simulate(_sim, theta_no_lock, show_keys))

    # §5.4 falsification.
    print("§5.4 falsification...")
    v_stiff = V[:, 0]; v_sloppy = V[:, -1]
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

    # Compute mean-field SIR spectrum for comparison.
    print("Computing mean-field SIR spectrum (for comparison)...")
    MF_THETA = jnp.array([0.40, 0.10, 1e-3, 0.40, 0.50])

    def _mf_sim(theta, key):
        return mf_simulate(theta, key, T=T, N=1e5, sigma_obs=10.0,
                           grad_horizon=None)

    mf_ref_keys = jax.random.split(jax.random.PRNGKey(100), 96)
    Y_mf = vmap_simulate(_mf_sim, MF_THETA, mf_ref_keys)
    mf_eig_keys = jax.random.split(jax.random.PRNGKey(101), 96)
    mf_theta_eval = MF_THETA + jnp.array([0.01, -0.005, 1e-4, 0.02, -0.02])
    mf_stats = per_seed_loss_and_grads(_mf_sim, mf_theta_eval, mf_eig_keys, Y_mf)
    mf_eig = eigendecompose(mf_stats.opg)
    mf_eigvals = np.asarray(mf_eig.eigvals)
    mf_V = np.asarray(mf_eig.eigvecs)
    print(f"  mean-field eigvals: {mf_eigvals}")
    print(f"  mean-field condition: {mf_eigvals[0]/max(mf_eigvals[-1], 1e-30):.2e}")

    # ============================================================ FIGURE
    fig = plt.figure(figsize=(17, 11))
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.36)

    # A. Sample trajectories
    ax = fig.add_subplot(gs[0, 0])
    for m in range(min(M_show, 12)):
        ax.plot(X_star[m], color=QUAL[0], lw=0.6, alpha=0.4)
        ax.plot(X_no_lock[m], color=QUAL[1], lw=0.6, alpha=0.4)
    ax.plot(X_star.mean(0), color=QUAL[0], lw=2.4,
            label=r"$\theta^*$ (with lockdown)")
    ax.plot(X_no_lock.mean(0), color=QUAL[1], lw=2.4,
            label=r"no lockdown ($f_{\mathrm{lock}}=1$)")
    ax.set_xlabel("day")
    ax.set_ylabel("daily new infections")
    ax.set_title(r"A. Network-SIR trajectories ($N=$" + f"{N_NODES}" + ", "
                 + rf"$\tau_{{\mathrm{{Gumbel}}}}={GUMBEL_TAU}$)")
    ax.legend(fontsize=9)

    # B. OPG spectrum + bootstrap CI
    ax = fig.add_subplot(gs[0, 1])
    P = eigvals.size
    xs = np.arange(P)
    le = np.clip(eigvals - boot_lo, a_min=0.0, a_max=None)
    ue = np.clip(boot_hi - eigvals, a_min=0.0, a_max=None)
    ax.errorbar(xs, eigvals, yerr=[le, ue], fmt="o",
                color=QUAL[0], capsize=4, markersize=12,
                markerfacecolor=QUAL[0], markeredgecolor="white",
                markeredgewidth=1.6, lw=1.8)
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"$v_{k+1}$" for k in xs])
    ax.set_xlabel("eigendirection")
    ax.set_ylabel(r"$\lambda_k$ (log)")
    span = eigvals[0] / max(eigvals[-1], 1e-30)
    ax.set_title(f"B. OPG spectrum + 95% boot CI (span {span:.0e})")

    # C. |V| heatmap
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
    ax.set_title(r"C. $|V|$: parameter content of each eigendirection")
    plt.colorbar(im, ax=ax, label=r"$|v_{k,j}|$")

    # D. Falsification bars
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
    ax.set_title("D. §5.4 falsification on network-SIR (three non-MMD discrepancies)")
    ax.legend(fontsize=9, ncol=2)

    # E. Mean-field vs network-SIR spectrum side-by-side
    ax = fig.add_subplot(gs[1, 2])
    ax.semilogy(xs, mf_eigvals, "o-", color=QUAL[3], markersize=10, lw=1.8,
                markerfacecolor=QUAL[3], markeredgecolor="white",
                markeredgewidth=1.2, label="mean-field SIR")
    ax.semilogy(xs, eigvals, "s-", color=QUAL[0], markersize=10, lw=1.8,
                markerfacecolor=QUAL[0], markeredgecolor="white",
                markeredgewidth=1.2, label="network-SIR (Gumbel)")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"$v_{k+1}$" for k in xs])
    ax.set_ylabel(r"$\lambda_k$")
    ax.set_title("E. Mean-field vs network-SIR spectra")
    ax.legend(fontsize=10)

    fig.suptitle(
        rf"Phase 3 Tier B: network-SIR diagnostic with Gumbel-Sigmoid surrogate",
        fontsize=14, fontweight="bold", y=0.995,
    )
    p = out_dir / "18_network_sir_diagnostic.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {p}")

    # ============================================================ verdict
    print("\n" + "=" * 70)
    print("NETWORK-SIR DIAGNOSTIC VERDICT")
    print("=" * 70)
    print(f"\nNetwork-SIR eigenvalues: {eigvals}")
    print(f"Mean-field   eigenvalues: {mf_eigvals}")
    print(f"\nFalsification ratios (stiff / sloppy):")
    for ch in channels:
        s = results[r"$+\alpha v_1$ (stiff)"][ch]
        l = results[r"$+\alpha v_P$ (sloppy)"][ch]
        print(f"  {ch:<10s}: {s/max(l, 1e-30):.0f}x")

    print(f"\nNetwork-SIR dominant eigenvector v_1:")
    for k in range(P):
        print(f"  {PARAM_NAMES[k]:<10s}: {V[k, 0]:+.3f}")
    print(f"\nNetwork-SIR sloppiest eigenvector v_P:")
    for k in range(P):
        print(f"  {PARAM_NAMES[k]:<10s}: {V[k, -1]:+.3f}")

    print(f"\nFor comparison, mean-field SIR v_1:")
    for k in range(P):
        print(f"  {PARAM_NAMES[k]:<10s}: {mf_V[k, 0]:+.3f}")
    print(f"\nMean-field v_P:")
    for k in range(P):
        print(f"  {PARAM_NAMES[k]:<10s}: {mf_V[k, -1]:+.3f}")

    np.savez(out_dir / "18_network_sir_diagnostic.npz",
             eigvals=eigvals, V=V,
             mf_eigvals=mf_eigvals, mf_V=mf_V,
             falsification=np.array([[results[k][c] for c in channels]
                                     for k in results]))
    print(f"saved {out_dir / '18_network_sir_diagnostic.npz'}")


if __name__ == "__main__":
    main()
