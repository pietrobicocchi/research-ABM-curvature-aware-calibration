"""Phase 1 §5.4 horizon-bias killswitch experiment.

Question:
    Is the OPG eigenstructure dominated by gradient-horizon truncation bias?

Protocol (matches project plan):
    At a fixed reference theta, compute F_hat with the per-seed gradient
    truncated to horizons H in {5, 10, 20, 40, T_full}. Compare:
        - Eigenvalue spectra across H.
        - Eigenvector subspaces via principal angles vs the full-horizon basis
          (the gold standard; the truncated estimators converge to this as
          H grows).
        - Bootstrap CIs on eigenvalues at each H, to separate truncation bias
          from sampling noise.

Decision rule:
    - If eigenstructure is stable across H (small principal angles, monotone
      convergence of eigenvalues), the diagnostic is robust to truncation
      and Phase 2 can proceed.
    - If unstable, the diagnostic is restricted to short-horizon-stable
      models, OR requires the unbiased StochasticAD estimator
      (Arya et al. 2022).

Run: uv run python scripts/09_horizon_bias.py

Note: forward (primal) pass is identical regardless of H; only the gradient
computation differs. So MMD itself does not change — only the per-seed
gradients g_m and their outer-product matrix F_hat do.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from curvature_calib.calibration.opg import (
    bootstrap_eigvals,
    eigendecompose,
    principal_angles,
)
from curvature_calib.calibration.per_seed_grads import (
    per_seed_loss_and_grads,
    vmap_simulate,
)
from curvature_calib.models.brock_hommes import simulate
from curvature_calib.viz.style import QUAL, apply_style


T = 200
SIGMA = 0.05
R = 1.1
THETA_STAR = jnp.array([3.0, 1.2, 0.2, 1.2, -0.2])
PARAM_NAMES = [r"$\beta$", r"$g_1$", r"$b_1$", r"$g_2$", r"$b_2$"]

HORIZONS = [5, 10, 20, 40, 80, T]  # T = T_full = 200
M_EVAL = 96
M_REF = 128


def make_sim(grad_horizon):
    """Closure: simulate function with a specific gradient horizon baked in.

    Forward pass identical across horizons (same primal output); only the
    backward (gradient) pass differs.
    """
    def _sim(theta, key):
        return simulate(theta, key, T=T, sigma=SIGMA, R=R,
                        x_init=0.0, grad_horizon=grad_horizon)
    return _sim


def main() -> None:
    apply_style()
    out = Path("outputs/brock_hommes")
    out.mkdir(exist_ok=True)

    # Build a reference distribution at theta*, using full-horizon simulator
    # (the reference does not need gradients).
    print("Building reference distribution...")
    ref_keys = jax.random.split(jax.random.PRNGKey(0), M_REF)
    Y_ref = vmap_simulate(make_sim(None), THETA_STAR, ref_keys)

    # Evaluate at a *perturbed* theta so the per-seed gradients carry signal.
    theta_eval = THETA_STAR + jnp.array([0.0, 0.05, 0.03, 0.05, -0.03])
    eval_keys = jax.random.split(jax.random.PRNGKey(11), M_EVAL)

    # Sweep horizons; collect spectra, eigenvectors, per-seed grads.
    print("Sweeping horizons...")
    results = {}
    for H in HORIZONS:
        sim_fn = make_sim(H if H < T else None)
        stats = per_seed_loss_and_grads(sim_fn, theta_eval, eval_keys, Y_ref)
        eig = eigendecompose(stats.opg)
        results[H] = {
            "eigvals": np.asarray(eig.eigvals),
            "eigvecs": np.asarray(eig.eigvecs),
            "grads": np.asarray(stats.per_seed_grads),
            "mean_grad_norm": float(jnp.linalg.norm(stats.mean_grad)),
            "loss": float(stats.loss),
        }
        print(f"  H={H:>3d}  ||g||={results[H]['mean_grad_norm']:.3e}  "
              f"lambda_1={results[H]['eigvals'][0]:.3e}  "
              f"lambda_P={results[H]['eigvals'][-1]:.3e}  "
              f"cond={results[H]['eigvals'][0] / max(results[H]['eigvals'][-1], 1e-30):.2e}")

    # Full-horizon (T) is the reference.
    V_full = results[T]["eigvecs"]  # (P, P)

    # Principal-angle stability: for each top-k subspace, compare H to T.
    print("\nComputing principal angles vs full horizon...")
    P = V_full.shape[0]
    subspace_dims = [1, 2, 3]
    angles_by_dim = {k: {} for k in subspace_dims}
    for H in HORIZONS:
        V_H = results[H]["eigvecs"]
        for k in subspace_dims:
            ang = np.asarray(principal_angles(
                jnp.asarray(V_H[:, :k]),
                jnp.asarray(V_full[:, :k]),
            ))
            angles_by_dim[k][H] = float(np.max(np.degrees(ang)))

    # Bootstrap eigenvalue CIs at the full horizon (for reference).
    print("Bootstrap CIs at full horizon...")
    boot_full = np.asarray(bootstrap_eigvals(
        jnp.asarray(results[T]["grads"]), n_boot=300,
        key=jax.random.PRNGKey(7)))
    boot_lo = np.percentile(boot_full, 2.5, axis=0)
    boot_hi = np.percentile(boot_full, 97.5, axis=0)

    # ===================================================== figure
    fig = plt.figure(figsize=(17, 11))
    gs = fig.add_gridspec(2, 3, hspace=0.40, wspace=0.35,
                          height_ratios=[1.0, 1.0])

    # A. Eigenvalues vs H (one line per eigenvalue).
    ax = fig.add_subplot(gs[0, 0])
    eig_array = np.array([results[H]["eigvals"] for H in HORIZONS])  # (n_H, P)
    for k in range(P):
        ax.semilogy(HORIZONS,
                    np.clip(eig_array[:, k], 1e-30, None),
                    "o-", color=plt.cm.viridis(k / max(P - 1, 1)),
                    lw=2, markersize=7, label=rf"$\lambda_{{{k+1}}}$")
    ax.axvline(T, color="grey", ls="--", lw=1, alpha=0.7,
               label=fr"full horizon $T={T}$")
    ax.set_xlabel("gradient horizon $H$ (steps)")
    ax.set_ylabel(r"$\lambda_k(H)$")
    ax.set_title("A. Eigenvalue convergence as $H \\to T$")
    ax.legend(fontsize=8, ncol=2)
    ax.set_xscale("log")

    # B. Principal angles vs full horizon, per subspace dim.
    ax = fig.add_subplot(gs[0, 1])
    for k in subspace_dims:
        vals = [angles_by_dim[k][H] for H in HORIZONS]
        ax.semilogy(HORIZONS, np.clip(vals, 1e-3, None),
                    "o-", color=QUAL[k - 1], lw=2, markersize=7,
                    label=f"top-{k} subspace")
    ax.axhline(1.0, color="grey", ls="--", lw=1, alpha=0.6,
               label=r"$1^\circ$ threshold")
    ax.axvline(T, color="grey", ls=":", lw=1, alpha=0.7)
    ax.set_xlabel("gradient horizon $H$")
    ax.set_ylabel("max principal angle (deg, log)")
    ax.set_title(r"B. Subspace drift: $\angle(\mathrm{span}(V_H^{:k}), \mathrm{span}(V_T^{:k}))$")
    ax.set_xscale("log")
    ax.legend(fontsize=9)

    # C. Eigenvalue ratios (relative bias) per eigenvalue.
    ax = fig.add_subplot(gs[0, 2])
    eig_ratios = eig_array / np.maximum(eig_array[-1:, :], 1e-30)  # divide by full-horizon value
    for k in range(P):
        ax.plot(HORIZONS, eig_ratios[:, k], "o-",
                color=plt.cm.viridis(k / max(P - 1, 1)),
                lw=2, markersize=7, label=rf"$\lambda_{{{k+1}}}(H) / \lambda_{{{k+1}}}(T)$")
    ax.axhline(1.0, color="grey", ls="--", lw=1, alpha=0.7)
    ax.set_xlabel("gradient horizon $H$")
    ax.set_ylabel("ratio vs full horizon")
    ax.set_title("C. Relative bias per eigenvalue")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize=7, ncol=2)

    # D. Full-spectrum heatmap: eigenvalues x horizons.
    ax = fig.add_subplot(gs[1, 0])
    log_eig = np.log10(np.clip(eig_array, 1e-30, None))
    im = ax.imshow(log_eig.T, cmap="viridis", aspect="auto",
                   extent=(0, len(HORIZONS), P, 0))
    ax.set_yticks(np.arange(P) + 0.5)
    ax.set_yticklabels([f"$\\lambda_{k+1}$" for k in range(P)])
    ax.set_xticks(np.arange(len(HORIZONS)) + 0.5)
    ax.set_xticklabels([str(H) for H in HORIZONS])
    ax.set_xlabel("gradient horizon $H$")
    ax.set_title("D. $\\log_{10}\\lambda_k$ across horizons")
    plt.colorbar(im, ax=ax, label=r"$\log_{10}\lambda$")

    # E. Eigenvector overlap matrix at the smallest horizon vs full horizon.
    ax = fig.add_subplot(gs[1, 1])
    V_5 = results[HORIZONS[0]]["eigvecs"]
    overlap = np.abs(V_5.T @ V_full)  # |<v_k(5), v_l(T)>|
    im = ax.imshow(overlap, cmap="magma", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(P))
    ax.set_xticklabels([f"$v_{k+1}(T)$" for k in range(P)])
    ax.set_yticks(range(P))
    ax.set_yticklabels([f"$v_{k+1}({HORIZONS[0]})$" for k in range(P)])
    for i in range(P):
        for j in range(P):
            ax.text(j, i, f"{overlap[i, j]:.2f}", ha="center", va="center",
                    color="white" if overlap[i, j] < 0.55 else "black",
                    fontsize=9)
    ax.set_title(f"E. Eigenvector overlap: H={HORIZONS[0]} vs T={T}")
    plt.colorbar(im, ax=ax, label=r"$|\langle v_k(H), v_l(T) \rangle|$")

    # F. Final-spectrum with bootstrap CI (full horizon) + each H's spectrum overlaid.
    ax = fig.add_subplot(gs[1, 2])
    xs = np.arange(P)
    full_eigs = results[T]["eigvals"]
    le = np.clip(full_eigs - boot_lo, a_min=0.0, a_max=None)
    ue = np.clip(boot_hi - full_eigs, a_min=0.0, a_max=None)
    ax.errorbar(xs, full_eigs, yerr=[le, ue], fmt="o",
                color="black", capsize=4, markersize=10,
                markerfacecolor="black", markeredgecolor="white",
                markeredgewidth=1.2, lw=1.5,
                label=f"$T={T}$ (full) + 95% boot CI")
    for H in HORIZONS[:-1]:
        ax.scatter(xs + 0.05 * (HORIZONS.index(H) - 2),
                   results[H]["eigvals"],
                   color=plt.cm.plasma(HORIZONS.index(H) / len(HORIZONS)),
                   s=55, alpha=0.85,
                   edgecolor="white", linewidth=0.8,
                   label=f"H={H}")
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"$v_{k+1}$" for k in xs])
    ax.set_xlabel("eigendirection")
    ax.set_ylabel(r"$\lambda_k$")
    ax.set_title("F. Spectrum at each $H$ vs $T$ + bootstrap CI")
    ax.legend(fontsize=7, ncol=2)

    fig.suptitle(
        "Phase 1 horizon-bias killswitch: OPG eigenstructure across "
        "gradient-truncation horizons",
        fontsize=14, fontweight="bold", y=0.995,
    )
    p = out / "09_horizon_bias.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {p}")

    # ===================================================== console verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    full_eigs = results[T]["eigvals"]
    print(f"\nFull-horizon spectrum (T={T}):")
    for k, lam in enumerate(full_eigs):
        bw = boot_hi[k] - boot_lo[k]
        print(f"  lambda_{k+1} = {lam:.4e}  (95% CI width {bw:.2e})")

    print(f"\nMax principal angle of top-1 subspace vs T:")
    for H in HORIZONS:
        a = angles_by_dim[1][H]
        print(f"  H={H:>3d}: {a:7.3f} deg  "
              f"{'STABLE' if a < 1.0 else ('drift' if a < 10 else 'UNSTABLE')}")

    print(f"\nMax principal angle of top-2 subspace vs T:")
    for H in HORIZONS:
        a = angles_by_dim[2][H]
        print(f"  H={H:>3d}: {a:7.3f} deg  "
              f"{'STABLE' if a < 1.0 else ('drift' if a < 10 else 'UNSTABLE')}")

    # Persist results for downstream / paper.
    np.savez(out / "09_horizon_bias.npz",
             horizons=np.array(HORIZONS),
             eigvals=eig_array,
             angles_top1=np.array([angles_by_dim[1][H] for H in HORIZONS]),
             angles_top2=np.array([angles_by_dim[2][H] for H in HORIZONS]),
             angles_top3=np.array([angles_by_dim[3][H] for H in HORIZONS]),
             V_full=V_full,
             boot_lo=boot_lo, boot_hi=boot_hi)
    print(f"saved {out / '09_horizon_bias.npz'}")


if __name__ == "__main__":
    main()
