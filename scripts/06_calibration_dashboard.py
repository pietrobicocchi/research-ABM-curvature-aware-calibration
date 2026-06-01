"""The headline diagnostic dashboard.

Splits into two figures for legibility:
    06a — OPG diagnostic: loss, params, eigenvalue waterfall, |V_T|, bootstrap CI,
          subspace drift, damping.
    06b — distributional sanity at theta_0, theta_T, theta*: return distributions,
          ACF, tail quantiles, sample trajectories.

Run: uv run python scripts/06_calibration_dashboard.py
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from curvature_calib.calibration.calibrate import calibrate
from curvature_calib.calibration.opg import (
    bootstrap_eigvals,
    eigendecompose,
    principal_angles,
)
from curvature_calib.calibration.per_seed_grads import vmap_simulate
from curvature_calib.losses.mmd import mmd_sq_with_median_bandwidth
from curvature_calib.models.brock_hommes import simulate
from curvature_calib.viz.style import QUAL, apply_style


T = 200
SIGMA = 0.05
R = 1.1

THETA_STAR = jnp.array([3.0, 1.2, 0.2, 1.2, -0.2])
PARAM_NAMES = [r"$\beta$", r"$g_1$", r"$b_1$", r"$g_2$", r"$b_2$"]


def _sim(theta, key):
    return simulate(theta, key, T=T, sigma=SIGMA, R=R, x_init=0.0)


def autocorr(x, max_lag=20):
    x = np.asarray(x) - np.mean(x)
    var = np.var(x) + 1e-12
    return np.array([np.mean(x[: x.size - k] * x[k:]) / var
                     for k in range(max_lag + 1)])


def noise_floor(theta, n_pairs=15, M=64, seed_off=10_000):
    vals = []
    for i in range(n_pairs):
        k1 = jax.random.split(jax.random.PRNGKey(seed_off + 2 * i), M)
        k2 = jax.random.split(jax.random.PRNGKey(seed_off + 2 * i + 1), M)
        A = vmap_simulate(_sim, theta, k1)
        B = vmap_simulate(_sim, theta, k2)
        vals.append(float(mmd_sq_with_median_bandwidth(A, B)))
    vals = np.array(vals)
    return float(np.mean(vals)), float(np.std(vals))


def main() -> None:
    apply_style()
    out = Path("outputs/brock_hommes")
    out.mkdir(exist_ok=True)

    # ---------------------------------------------------- run
    M_ref = 128
    ref_keys = jax.random.split(jax.random.PRNGKey(0), M_ref)
    Y_ref = vmap_simulate(_sim, THETA_STAR, ref_keys)

    # Init far from theta* IN THE STIFF DIRECTION. Naive large-distance
    # perturbations often land along sloppy directions (esp. beta), where
    # MMD^2 is already at the noise floor and there's nothing to descend.
    # The stiff direction at theta* is dominated by the symmetric bias
    # combination b_1 + b_2 (see scripts/13_jacobian_comparison.py),
    # so perturbing both biases together pushes loss off the noise floor.
    theta0 = THETA_STAR + jnp.array([0.0, 0.0, 0.1, 0.0, 0.1])
    # Distance ||delta|| = 0.14; both biases shift +0.1, giving an initial
    # MMD^2 around 1e-2 -- well above noise floor (~5e-4), but moderate
    # enough that LM-Newton steps don't immediately overshoot into the
    # unstable BH regime (g/R > 1.5).
    print("Calibrating...")
    # init_damping = 100: bound first-step magnitude (with gradient norm ~3.5
    # at this theta_0 the default damping=1 overshoots into the unstable
    # regime; LM will adapt downward as iterations proceed).
    log = calibrate(_sim, theta0, Y_ref, M=64, n_iter=60, init_damping=100.0,
                    verbose=True)
    arrs = log.as_arrays()

    # Plot the validation loss (fixed held-out seed set) -- monotone tracking.
    # `losses` (noisy training-time, fresh seeds each iter) also available.
    losses = arrs["val_losses"]
    thetas = arrs["thetas"]
    eigvals = arrs["eigvals"]
    eigvecs = arrs["eigvecs"]
    per_seed_grads = arrs["per_seed_grads"]

    print("Estimating noise floor...")
    nf_mean, nf_std = noise_floor(THETA_STAR, n_pairs=12, M=64)

    print("Bootstrapping final eigenvalues...")
    boot = np.asarray(bootstrap_eigvals(
        jnp.asarray(per_seed_grads[-1]), n_boot=400,
        key=jax.random.PRNGKey(42)))
    boot_lo = np.percentile(boot, 2.5, axis=0)
    boot_hi = np.percentile(boot, 97.5, axis=0)

    V_T = jnp.asarray(eigvecs[-1])[:, :2]
    angles_top2 = []
    for t in range(eigvecs.shape[0]):
        Vt = jnp.asarray(eigvecs[t])[:, :2]
        angles_top2.append(float(jnp.asarray(principal_angles(Vt, V_T)).max()))
    angles_top2 = np.array(angles_top2)

    # ===================================================== FIGURE A: diagnostic
    figA, ax = plt.subplots(2, 3, figsize=(18, 10))

    # A1: loss -- plot best-so-far (running minimum) as the primary signal
    # (standard convention for stochastic optimisation; guaranteed monotone)
    # and overlay the raw val curve faintly.
    a = ax[0, 0]
    its = np.arange(losses.size)
    best = np.minimum.accumulate(losses)
    a.semilogy(its, np.clip(best, 1e-12, None), "-", color=QUAL[0], lw=2.2,
               label="best-so-far val MMD²")
    a.semilogy(its, np.clip(losses, 1e-12, None), color=QUAL[0], lw=0.7,
               alpha=0.4, label="raw val MMD² (per iter)")
    a.axhspan(max(nf_mean - 2 * nf_std, 1e-12),
              max(nf_mean + 2 * nf_std, 1e-12),
              color="grey", alpha=0.25, label=r"MMD$^2$ noise floor")
    a.axhline(max(nf_mean, 1e-12), color="grey", lw=1, ls="--")
    a.set_xlabel("iteration")
    a.set_ylabel(r"MMD$^2$")
    a.set_title("A1. Loss trajectory (val, fixed seeds)")
    a.legend(fontsize=8)

    # A2: parameter trajectories
    a = ax[0, 1]
    for k, name in enumerate(PARAM_NAMES):
        a.plot(thetas[:, k], color=QUAL[k % len(QUAL)], lw=1.6, label=name)
        a.axhline(float(THETA_STAR[k]), color=QUAL[k % len(QUAL)],
                  ls=":", lw=1, alpha=0.7)
    a.set_xlabel("iteration")
    a.set_ylabel("parameter value")
    a.set_title(r"A2. Iterates (dotted = $\theta^*$)")
    a.legend(fontsize=8, ncol=2, framealpha=0.95)

    # A3: eigenvalue waterfall
    a = ax[0, 2]
    P = eigvals.shape[1]
    for k in range(P):
        a.semilogy(np.clip(eigvals[:, k], 1e-30, None),
                   color=plt.cm.viridis(k / max(P - 1, 1)), lw=2,
                   label=rf"$\lambda_{{{k+1}}}$")
    a.set_xlabel("iteration")
    a.set_ylabel(r"$\lambda_k$ (log)")
    a.set_title("A3. OPG spectrum over time")
    a.legend(fontsize=8, ncol=2, loc="center right", framealpha=0.95)

    # A4: |V_T| heatmap
    a = ax[1, 0]
    V_final = np.abs(eigvecs[-1])
    im = a.imshow(V_final, cmap="magma", aspect="auto", vmin=0, vmax=1)
    a.set_xticks(range(P))
    a.set_xticklabels([f"$v_{k+1}$" for k in range(P)])
    a.set_yticks(range(P))
    a.set_yticklabels(PARAM_NAMES)
    a.set_title(r"A4. $|V_T|$: identifiable combinations at convergence")
    for i in range(P):
        for j in range(P):
            a.text(j, i, f"{V_final[i, j]:.2f}", ha="center", va="center",
                   color="white" if V_final[i, j] < 0.55 else "black",
                   fontsize=9)
    plt.colorbar(im, ax=a, label="magnitude")

    # A5: bootstrap CI on final eigenvalues
    a = ax[1, 1]
    final_eigs = np.clip(eigvals[-1], 1e-30, None)
    xs_e = np.arange(P)
    le = np.clip(final_eigs - boot_lo, a_min=0.0, a_max=None)
    ue = np.clip(boot_hi - final_eigs, a_min=0.0, a_max=None)
    a.errorbar(xs_e, final_eigs, yerr=[le, ue], fmt="o",
               color=QUAL[0], capsize=4, markersize=10,
               markerfacecolor=QUAL[0], markeredgecolor="white",
               markeredgewidth=1.3, lw=1.5)
    a.set_yscale("log")
    a.set_xticks(xs_e)
    a.set_xticklabels([f"$v_{k+1}$" for k in xs_e])
    a.set_xlabel("eigendirection")
    a.set_ylabel(r"$\lambda_k$ at $\theta_T$")
    a.set_title("A5. Final eigenvalues + 95% bootstrap CI")

    # A6: subspace drift + damping (twin axis)
    a = ax[1, 2]
    a.plot(np.degrees(angles_top2), color=QUAL[2], lw=1.8,
           label="subspace drift (deg)")
    a.set_xlabel("iteration")
    a.set_ylabel("subspace drift (deg)", color=QUAL[2])
    a.tick_params(axis="y", labelcolor=QUAL[2])
    a.set_title("A6. Subspace drift + LM damping")
    a2 = a.twinx()
    a2.semilogy(arrs["dampings"], color=QUAL[3], lw=1.6, ls="--",
                label="LM damping (log)")
    a2.set_ylabel(r"$\lambda_{\mathrm{damp}}$", color=QUAL[3])
    a2.tick_params(axis="y", labelcolor=QUAL[3])
    a2.grid(False)

    figA.suptitle(
        r"OPG diagnostic — Brock-Hommes calibration ($P=5$, $M=64$)",
        fontsize=15, fontweight="bold", y=0.995,
    )
    figA.tight_layout(rect=(0, 0, 1, 0.97))
    pa = out / "06a_calibration_diagnostic.png"
    figA.savefig(pa, dpi=130, bbox_inches="tight")
    plt.close(figA)
    print(f"saved {pa}")

    # ===================================================== FIGURE B: distributions
    M_show = 96
    keys = jax.random.split(jax.random.PRNGKey(101), M_show)
    X0 = np.asarray(vmap_simulate(_sim, theta0, keys))
    XT = np.asarray(vmap_simulate(_sim, jnp.asarray(thetas[-1]), keys))
    X_star = np.asarray(vmap_simulate(_sim, THETA_STAR, keys))

    def acf_mean(X, max_lag=20):
        return np.mean(np.stack([autocorr(X[m], max_lag) for m in range(X.shape[0])]),
                       axis=0)

    figB, axB = plt.subplots(2, 2, figsize=(14, 9))

    # B1: returns
    a = axB[0, 0]
    for name, X, color in [("init $\\theta_0$", X0, QUAL[3]),
                            ("calibrated $\\theta_T$", XT, QUAL[1]),
                            ("truth $\\theta^*$",      X_star, QUAL[0])]:
        rets = np.diff(X, axis=1).reshape(-1)
        a.hist(rets, bins=80, alpha=0.6, density=True,
               label=name, color=color, edgecolor="none")
    a.set_xlabel(r"$\Delta x_t$")
    a.set_ylabel("density (log)")
    a.set_yscale("log")
    a.set_title(r"B1. Return distribution")
    a.legend(fontsize=10)

    # B2: ACF
    a = axB[0, 1]
    for name, X, color in [("init", X0, QUAL[3]),
                            ("calibrated", XT, QUAL[1]),
                            ("truth", X_star, QUAL[0])]:
        a.plot(acf_mean(X, max_lag=20), "o-",
               color=color, label=name, markersize=5, lw=1.4)
    a.set_xlabel("lag")
    a.set_ylabel(r"ACF($x_t$)")
    a.set_title(r"B2. Autocorrelation function")
    a.legend(fontsize=10)

    # B3: quantiles
    a = axB[1, 0]
    qs = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    width = 0.27
    xs_q = np.arange(len(qs))
    for offset, (name, X, color) in enumerate([("init", X0, QUAL[3]),
                                                ("calibrated", XT, QUAL[1]),
                                                ("truth", X_star, QUAL[0])]):
        q_vals = np.quantile(X.reshape(-1), qs)
        a.bar(xs_q + (offset - 1) * width, q_vals, width,
              color=color, edgecolor="white", label=name)
    a.set_xticks(xs_q)
    a.set_xticklabels([f"{q:.2f}" for q in qs])
    a.set_xlabel("quantile")
    a.set_ylabel(r"value of $x_t$")
    a.set_title(r"B3. Empirical quantiles")
    a.legend(fontsize=10)

    # B4: sample trajectories
    a = axB[1, 1]
    a.plot(X_star[0], color=QUAL[0], lw=1.2, alpha=0.95, label=r"truth $\theta^*$")
    a.plot(XT[0], color=QUAL[1], lw=1.2, alpha=0.85, label=r"calibrated $\theta_T$")
    a.plot(X0[0], color=QUAL[3], lw=1.2, alpha=0.7, label=r"init $\theta_0$")
    a.set_xlabel("t")
    a.set_ylabel(r"$x_t$ (one seed each, same key)")
    a.set_title(r"B4. Sample trajectories")
    a.legend(fontsize=10)

    figB.suptitle(
        r"Distributional comparison at $\theta_0$, $\theta_T$, $\theta^*$",
        fontsize=15, fontweight="bold", y=0.995,
    )
    figB.tight_layout(rect=(0, 0, 1, 0.96))
    pb = out / "06b_calibration_distributions.png"
    figB.savefig(pb, dpi=130, bbox_inches="tight")
    plt.close(figB)
    print(f"saved {pb}")

    # Persist log for downstream scripts (08_falsification).
    np.savez(out / "calibration_log.npz", **arrs)
    print(f"saved {out / 'calibration_log.npz'}")


if __name__ == "__main__":
    main()
