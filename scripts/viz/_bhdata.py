"""Shared Brock-Hommes calibration data for the in-motion (Fig 2) and
eigenvector-structure (Fig 3) figures.

Runs the same three-regime calibration as scripts/28, but additionally records
the *live* diagnostic at every iterate: a bootstrap noise floor tau_t and the
effective dimension d_eff(t) = #{lambda_k(t) > tau_t}. Cached to
outputs/viz/_cache so Fig 2 and Fig 3 share one set of runs.
"""
from __future__ import annotations

from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from curvature_calib.calibration.bootstrap import bootstrap_eigvals, eigenvalue_cis
from curvature_calib.calibration.calibrate import calibrate
from curvature_calib.calibration.per_seed_grads import vmap_simulate
from curvature_calib.models.brock_hommes import simulate

SIGMA, R, T, M, N_ITER, M_REF, N_BOOT = 0.05, 1.1, 200, 64, 60, 128, 500

# Precision-aware spectral floor. A direction counts as identified only if its
# bootstrap CI lower bound clears a relative cutoff tied to conditioning:
#   lambda_k / lambda_1 > REL_FLOOR  (condition-number ceiling 1/REL_FLOOR = 1e8,
#   ~ sqrt(eps) for float64). This prevents counting eigenvalues that sit at the
#   floating-point floor (lambda_k/lambda_1 ~ 1e-10..1e-30) as "constrained" just
#   because their bootstrap CI technically excludes zero.
REL_FLOOR = 1e-8

# Stiff -> sloppy ordering preserved across every figure.
PARAM_NAMES = [r"$\beta$", r"$g_1$", r"$b_1$", r"$g_2$", r"$b_2$"]
REGIME_ORDER = ["stable", "periodic", "chaotic"]
REGIMES = {
    "stable":   jnp.array([1.0, 0.5, 0.0, 0.5, 0.0], dtype=jnp.float64),
    "periodic": jnp.array([5.0, 1.2, 0.0, -0.5, 0.0], dtype=jnp.float64),
    "chaotic":  jnp.array([10.0, 1.2, 0.2, 1.2, -0.2], dtype=jnp.float64),
}
DELTA = jnp.array([0.0, 0.0, 0.1, 0.0, 0.1], dtype=jnp.float64)
REGIME_SEEDS = {"stable": 42, "periodic": 43, "chaotic": 44}

CACHE = Path("outputs/viz/_cache/bh_motion.npz")


def _sim(theta, key):
    return simulate(theta, key, T=T, sigma=SIGMA, R=R, x_init=0.0)


def _run_regime(theta_star, name):
    theta0 = theta_star + DELTA
    ref_keys = jax.random.split(jax.random.PRNGKey(0), M_REF)
    Y_ref = vmap_simulate(_sim, theta_star, ref_keys)
    log = calibrate(_sim, theta0, Y_ref, M=M, n_iter=N_ITER,
                    init_damping=100.0, verbose=False, seed_base=REGIME_SEEDS[name])

    eigvals_traj = np.asarray(log.eigvals)              # (n_iter, P)
    V_T = np.asarray(log.eigvecs[-1])                   # (P, P)
    psg = np.asarray(log.per_seed_grads)               # (n_iter, M, P)

    n_iter = eigvals_traj.shape[0]
    pfloor = np.empty(n_iter)               # precision floor tau_t = REL_FLOOR * lambda_1
    d_eff = np.empty(n_iter, dtype=int)
    boot_lo = np.empty_like(eigvals_traj)
    boot_hi = np.empty_like(eigvals_traj)
    for t in range(n_iter):
        # final iterate uses the canonical convergence key (matches script 28)
        bkey = (REGIME_SEEDS[name] + 1000 if t == n_iter - 1
                else REGIME_SEEDS[name] * 100 + t)
        boot = bootstrap_eigvals(jnp.asarray(psg[t]), n_boot=N_BOOT,
                                 key=jax.random.PRNGKey(bkey))
        cis = np.asarray(eigenvalue_cis(boot))
        boot_lo[t], boot_hi[t] = cis[:, 0], cis[:, 1]
        pfloor[t] = REL_FLOOR * max(eigvals_traj[t, 0], 1e-300)
        # identified := bootstrap CI lower bound clears the precision floor
        d_eff[t] = int(np.sum(boot_lo[t] > pfloor[t]))

    return dict(eigvals_traj=eigvals_traj, V_T=V_T, tau=pfloor, d_eff=d_eff,
                boot_lo=boot_lo, boot_hi=boot_hi, eigvals_T=eigvals_traj[-1])


def load(force: bool = False) -> dict:
    """Return {regime: data-dict}. Cached; pass force=True to recompute."""
    if CACHE.exists() and not force:
        raw = np.load(CACHE)
        out = {}
        for name in REGIME_ORDER:
            out[name] = {k[len(name) + 1:]: raw[k] for k in raw.files
                         if k.startswith(name + "_")}
        return out

    results = {}
    for name in REGIME_ORDER:
        print(f"  calibrating {name} regime ...", flush=True)
        results[name] = _run_regime(REGIMES[name], name)

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, **{f"{name}_{k}": v
                                  for name, d in results.items()
                                  for k, v in d.items()})
    return results


if __name__ == "__main__":
    load(force=True)
    print(f"cached -> {CACHE}")
