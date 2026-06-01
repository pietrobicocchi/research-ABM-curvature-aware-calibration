"""Script 19: trajectory bootstrap of the OPG eigenvectors.

Tests whether the OPG eigenstructure is stable along the calibration
trajectory -- specifically whether the stiff (v_1) and sloppy (v_P)
eigenvectors computed at intermediate iterates theta_t match those at
theta*. If yes, the diagnostic is "live": you can compute F_hat at the
START of calibration and it already names the directions Adam will fail
along. If not, the diagnostic is a post-hoc oracle and the paper must
scope claims to "at convergence".

This closes attack #2 of docs/memory/honest_appraisal.md and is required
to support the live-diagnostic claim in paper_story_arc.md.

Methodology:
  * Reference V* computed from a fresh F_hat at theta* with M=128.
  * For each snapshot iterate t in the saved BH calibration log
    (M=64 per-seed grads), bootstrap-resample the seeds B times, refit
    the OPG, and compute the principal angle between v_k(theta_t) and
    v_k(theta*) for k = 1 (stiffest) and k = P (sloppiest).
  * Plot principal angle vs iterate, separate panel per direction,
    shaded bootstrap IQR.
  * Verdict rule: if the sloppy-direction angle median stays below 20
    degrees throughout, claim "live"; else scope to "at convergence".

Run: uv run python scripts/19_trajectory_bootstrap.py
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from curvature_calib.calibration.opg import eigendecompose
from curvature_calib.calibration.per_seed_grads import (
    per_seed_loss_and_grads,
    vmap_simulate,
)
from curvature_calib.models.brock_hommes import simulate
from curvature_calib.viz.style import QUAL, apply_style, save


T = 200
SIGMA = 0.05
R = 1.1
THETA_STAR = jnp.array([3.0, 1.2, 0.2, 1.2, -0.2])
PARAM_NAMES = [r"$\beta$", r"$g_1$", r"$b_1$", r"$g_2$", r"$b_2$"]

SNAPSHOT_ITERS = [0, 1, 2, 4, 8, 16, 30, 59]
N_BOOT = 500
LIVE_THRESHOLD_DEG = 20.0


def _sim(theta, key):
    return simulate(theta, key, T=T, sigma=SIGMA, R=R, x_init=0.0,
                    grad_horizon=None)


def reference_eigenbasis():
    """Compute V*(theta*) with M=128. Seed 0 generates Y_ref, seed 1 the keys
    used to estimate F_hat at theta*."""
    M_ref = 128
    ref_keys = jax.random.split(jax.random.PRNGKey(0), M_ref)
    Y_ref = vmap_simulate(_sim, THETA_STAR, ref_keys)
    eig_keys = jax.random.split(jax.random.PRNGKey(1), M_ref)
    stats = per_seed_loss_and_grads(_sim, THETA_STAR, eig_keys, Y_ref)
    eig = eigendecompose(stats.opg)
    return np.asarray(eig.eigvals), np.asarray(eig.eigvecs)


def bootstrap_angles_to_ref(grads_t, V_ref, n_boot, rng):
    """For one iterate t, return (n_boot, P) array of principal angles
    in degrees between v_k(t) and v_k(theta*) for each eigenvector k."""
    M, P = grads_t.shape
    angles = np.empty((n_boot, P))
    for b in range(n_boot):
        idx = rng.integers(0, M, M)
        Gb = grads_t[idx]
        Fb = (Gb.T @ Gb) / M
        Fb = 0.5 * (Fb + Fb.T)
        w, V = np.linalg.eigh(Fb)
        order = np.argsort(-w)
        V = V[:, order]
        # principal angle between 1-d subspaces spanned by V[:,k] and V_ref[:,k]
        # = arccos(|<V[:,k], V_ref[:,k]>|)
        cos = np.clip(np.abs(np.einsum("ij,ij->j", V, V_ref)), 0.0, 1.0)
        angles[b] = np.degrees(np.arccos(cos))
    return angles


def main() -> None:
    apply_style()
    out = Path("outputs/brock_hommes")

    log_path = out / "calibration_log.npz"
    if not log_path.exists():
        raise FileNotFoundError(
            f"Need {log_path}. Run scripts/06_calibration_dashboard.py first.")

    print("Computing reference eigenbasis V*(theta*) ...")
    eigvals_star, V_star = reference_eigenbasis()
    P = V_star.shape[0]
    print(f"  eigenvalues:   {eigvals_star}")
    print(f"  condition:     {eigvals_star[0] / max(eigvals_star[-1], 1e-30):.2e}")

    log = np.load(log_path)
    thetas = log["thetas"]                 # (n_iter+1, P)
    per_seed_grads = log["per_seed_grads"] # (n_iter, M, P)
    n_iter = per_seed_grads.shape[0]
    print(f"Loaded {n_iter}-iter log with M={per_seed_grads.shape[1]}.")

    valid_iters = [t for t in SNAPSHOT_ITERS if t < n_iter]
    if 0 not in valid_iters:
        valid_iters = [0] + valid_iters

    rng = np.random.default_rng(2026)
    angles_by_iter = {}
    for t in valid_iters:
        ang = bootstrap_angles_to_ref(per_seed_grads[t], V_star, N_BOOT, rng)
        angles_by_iter[t] = ang
        med = np.median(ang, axis=0)
        print(f"  iter t={t:3d}: stiff(v1)={med[0]:5.1f}deg  "
              f"sloppy(vP)={med[-1]:5.1f}deg")

    # distance from theta*  along the trajectory (for context on x-axis)
    dist_to_star = np.linalg.norm(thetas - np.asarray(THETA_STAR), axis=1)

    # ============================================================ FIGURE
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))

    # --- Panel A: stiff (v_1) ------------------------------------------------
    ax = axes[0]
    ts = np.array(valid_iters)
    med_stiff = np.array([np.median(angles_by_iter[t][:, 0]) for t in ts])
    lo_stiff = np.array([np.percentile(angles_by_iter[t][:, 0], 25) for t in ts])
    hi_stiff = np.array([np.percentile(angles_by_iter[t][:, 0], 75) for t in ts])
    ax.fill_between(ts, lo_stiff, hi_stiff, color=QUAL[0], alpha=0.25,
                    label="bootstrap IQR")
    ax.plot(ts, med_stiff, "o-", color=QUAL[0], label="median")
    ax.axhline(LIVE_THRESHOLD_DEG, color="grey", linestyle=":",
               label=f"{LIVE_THRESHOLD_DEG:.0f} deg threshold")
    ax.set_xlabel("iterate $t$")
    ax.set_ylabel(r"principal angle $\angle(v_1(\theta_t), v_1(\theta^*))$ (deg)")
    ax.set_title(r"(a) stiffest direction $v_1$")
    ax.set_ylim(0, max(90.0, hi_stiff.max() * 1.1))
    ax.legend(loc="best", frameon=False)

    # --- Panel B: sloppy (v_P) -----------------------------------------------
    ax = axes[1]
    med_sloppy = np.array([np.median(angles_by_iter[t][:, -1]) for t in ts])
    lo_sloppy = np.array([np.percentile(angles_by_iter[t][:, -1], 25) for t in ts])
    hi_sloppy = np.array([np.percentile(angles_by_iter[t][:, -1], 75) for t in ts])
    ax.fill_between(ts, lo_sloppy, hi_sloppy, color=QUAL[2], alpha=0.25,
                    label="bootstrap IQR")
    ax.plot(ts, med_sloppy, "o-", color=QUAL[2], label="median")
    ax.axhline(LIVE_THRESHOLD_DEG, color="grey", linestyle=":",
               label=f"{LIVE_THRESHOLD_DEG:.0f} deg threshold")
    ax.set_xlabel("iterate $t$")
    ax.set_ylabel(r"principal angle $\angle(v_P(\theta_t), v_P(\theta^*))$ (deg)")
    ax.set_title(r"(b) sloppiest direction $v_P$")
    ax.set_ylim(0, max(90.0, hi_sloppy.max() * 1.1))
    ax.legend(loc="best", frameon=False)

    # --- Panel C: trajectory context ----------------------------------------
    ax = axes[2]
    ax.plot(np.arange(thetas.shape[0]), dist_to_star, "-", color=QUAL[1])
    for t in valid_iters:
        ax.plot(t, dist_to_star[t], "o", color=QUAL[1])
    ax.set_xlabel("iterate $t$")
    ax.set_ylabel(r"$\|\theta_t - \theta^*\|$")
    ax.set_title("(c) trajectory")
    ax.set_yscale("log")

    fig.suptitle(
        "OPG eigenvectors along the calibration trajectory vs $V(\\theta^*)$",
        y=1.02,
    )
    fig.tight_layout()
    save(fig, "19_trajectory_bootstrap.png", out_dir=out)

    # ============================================================ VERDICT
    sloppy_max_median = float(np.max(med_sloppy))
    sloppy_at_init = float(med_sloppy[0])
    stiff_max_median = float(np.max(med_stiff))
    verdict = ("LIVE"
               if sloppy_max_median < LIVE_THRESHOLD_DEG
                  and stiff_max_median < LIVE_THRESHOLD_DEG
               else "AT-CONVERGENCE-ONLY")
    print()
    print("=" * 60)
    print(f"  sloppy v_P median angle at t=0:     {sloppy_at_init:5.1f} deg")
    print(f"  sloppy v_P median angle max:        {sloppy_max_median:5.1f} deg")
    print(f"  stiff  v_1 median angle max:        {stiff_max_median:5.1f} deg")
    print(f"  threshold:                          {LIVE_THRESHOLD_DEG:5.1f} deg")
    print(f"  VERDICT: {verdict}")
    print("=" * 60)

    np.savez_compressed(
        out / "19_trajectory_bootstrap.npz",
        snapshot_iters=np.array(valid_iters),
        angles=np.stack([angles_by_iter[t] for t in valid_iters], axis=0),
        eigvals_star=eigvals_star,
        V_star=V_star,
        dist_to_star=dist_to_star,
        thetas=thetas,
        verdict=verdict,
    )


if __name__ == "__main__":
    main()
