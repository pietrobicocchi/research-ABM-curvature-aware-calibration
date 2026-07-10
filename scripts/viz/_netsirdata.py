"""Shared network-SIR calibration data for the in-motion (fig7) and
subspace-rotation (fig7b) figures.

Runs three R0-regime calibrations of the network-SIR model at the
interventionist theta* (lockdown fires early, during the epidemic), recording
the live diagnostic at every iterate: bootstrap noise floor tau_t and effective
dimension d_eff(t). Cached to outputs/viz/_cache/sir_motion.npz.
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
from curvature_calib.models.network_sir import simulate

N, MEAN_DEG, T, M, M_REF, N_ITER, N_BOOT = 250, 6.0, 200, 64, 96, 40, 300
GAMMA, I0, TLOCK, FLOCK = 0.10, 0.01, 0.025, 0.50
REL_FLOOR = 1e-8

PARAM_NAMES = [r"$\beta$", r"$\gamma$", r"$I_0$", r"$t_{\rm lock}$", r"$f_{\rm lock}$"]
REGIME_ORDER = ["slow", "moderate", "fast"]
P = 5

def _theta_star(R0: float) -> jnp.ndarray:
    beta = R0 * GAMMA / MEAN_DEG
    return jnp.array([beta, GAMMA, I0, TLOCK, FLOCK], dtype=jnp.float64)

REGIMES = {"slow": _theta_star(1.3), "moderate": _theta_star(2.5), "fast": _theta_star(5.0)}
# Stiff-direction init: perturb along v1 ~ (I0, beta) so there is descent signal
# (a sloppy/random init lands at the MMD noise floor; see state.md BH note).
DELTA = jnp.array([0.004, 0.0, 0.004, 0.0, 0.0], dtype=jnp.float64)
REGIME_SEEDS = {"slow": 52, "moderate": 53, "fast": 54}

CACHE = Path("outputs/viz/_cache/sir_motion.npz")


def _sim(theta, key):
    return simulate(theta, key, T=T, N=N, mean_degree=MEAN_DEG, surrogate="gumbel")


def _run_regime(theta_star, name):
    theta0 = theta_star + DELTA
    ref_keys = jax.random.split(jax.random.PRNGKey(0), M_REF)
    Y_ref = vmap_simulate(_sim, theta_star, ref_keys)
    log = calibrate(_sim, theta0, Y_ref, M=M, n_iter=N_ITER,
                    init_damping=100.0, verbose=False, seed_base=REGIME_SEEDS[name])

    eigvals_traj = np.asarray(log.eigvals)
    eigvecs_traj = np.asarray(log.eigvecs)
    V_T = eigvecs_traj[-1]
    psg = np.asarray(log.per_seed_grads)

    n_iter = eigvals_traj.shape[0]
    pfloor = np.empty(n_iter)
    d_eff = np.empty(n_iter, dtype=int)
    boot_lo = np.empty_like(eigvals_traj)
    boot_hi = np.empty_like(eigvals_traj)
    for t in range(n_iter):
        bkey = (REGIME_SEEDS[name] + 1000 if t == n_iter - 1
                else REGIME_SEEDS[name] * 100 + t)
        boot = bootstrap_eigvals(jnp.asarray(psg[t]), n_boot=N_BOOT,
                                 key=jax.random.PRNGKey(bkey))
        cis = np.asarray(eigenvalue_cis(boot))
        boot_lo[t], boot_hi[t] = cis[:, 0], cis[:, 1]
        pfloor[t] = REL_FLOOR * max(eigvals_traj[t, 0], 1e-300)
        d_eff[t] = int(np.sum(boot_lo[t] > pfloor[t]))

    return dict(eigvals_traj=eigvals_traj, eigvecs_traj=eigvecs_traj, V_T=V_T,
                tau=pfloor, d_eff=d_eff, boot_lo=boot_lo, boot_hi=boot_hi,
                eigvals_T=eigvals_traj[-1])


def load(force: bool = False) -> dict:
    if CACHE.exists() and not force:
        raw = np.load(CACHE)
        out = {}
        for name in REGIME_ORDER:
            out[name] = {k[len(name) + 1:]: raw[k] for k in raw.files
                         if k.startswith(name + "_")}
        if all("eigvecs_traj" in out[name] for name in REGIME_ORDER):
            return out
        print("  cache missing eigvecs_traj -> recomputing ...", flush=True)

    results = {}
    for name in REGIME_ORDER:
        print(f"  calibrating {name} regime (R0-scaled) ...", flush=True)
        results[name] = _run_regime(REGIMES[name], name)

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, **{f"{name}_{k}": v
                                  for name, d in results.items()
                                  for k, v in d.items()})
    return results


if __name__ == "__main__":
    load(force=True)
    print(f"cached -> {CACHE}")
