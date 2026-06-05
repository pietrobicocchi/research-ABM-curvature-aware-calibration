# src/curvature_calib/calibration/falsification.py
"""Falsification protocol: perturbation along stiff/sloppy OPG directions.

Tests whether OPG-flagged non-identifiabilities are simulator-intrinsic
rather than MMD artefacts. Protocol (§5.4 of paper):
  - Perturb theta_T by +alpha * v_1 (stiff) and +alpha * v_P (sloppy).
  - Simulate M seeds at each perturbed parameter.
  - Measure three non-MMD discrepancies vs baseline at theta_T.
  - If OPG is genuine, sloppy perturbations are near-invisible; stiff are not.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from scipy import stats as sps

from curvature_calib.calibration.diagnostic import EigDecomp


@dataclass
class FalsificationResult:
    alpha_grid: np.ndarray        # (n_alpha,)
    stiff_moments: np.ndarray     # (n_alpha, 4)
    stiff_acf: np.ndarray         # (n_alpha,)
    stiff_quantiles: np.ndarray   # (n_alpha, 4)
    sloppy_moments: np.ndarray    # (n_alpha, 4)
    sloppy_acf: np.ndarray        # (n_alpha,)
    sloppy_quantiles: np.ndarray  # (n_alpha, 4)


def perturbed_parameters(
    theta: jax.Array,
    direction: jax.Array,
    alpha_grid: jax.Array,
) -> jax.Array:
    """theta +/- alpha * direction for each alpha.

    Returns (n_alpha, 2, P) where axis-1 index 0 = plus, 1 = minus.
    """
    alphas = alpha_grid[:, None]
    return jnp.stack([theta[None] + alphas * direction,
                      theta[None] - alphas * direction], axis=1)


def moments_difference(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Absolute differences in mean, std, skewness, kurtosis.

    X, Y: (n_seeds, T). Returns (4,).
    """
    x, y = X.ravel(), Y.ravel()
    return np.abs(np.array([
        x.mean() - y.mean(),
        x.std()  - y.std(),
        float(sps.skew(x))     - float(sps.skew(y)),
        float(sps.kurtosis(x)) - float(sps.kurtosis(y)),
    ]))


def acf_difference(X: np.ndarray, Y: np.ndarray, max_lag: int = 20) -> float:
    """Sup-norm of difference between mean empirical ACFs of X and Y.

    X, Y: (n_seeds, T). Returns scalar.
    """
    def _mean_acf(Z: np.ndarray) -> np.ndarray:
        out = np.zeros(max_lag + 1)
        for row in Z:
            row = row - row.mean()
            var = row.var() + 1e-12
            out += np.array([np.mean(row[:len(row)-k] * row[k:]) / var
                             for k in range(max_lag + 1)])
        return out / Z.shape[0]
    return float(np.max(np.abs(_mean_acf(X) - _mean_acf(Y))))


def quantile_difference(
    X: np.ndarray,
    Y: np.ndarray,
    quantiles: tuple[float, ...] = (0.01, 0.05, 0.95, 0.99),
) -> np.ndarray:
    """Absolute differences in empirical quantiles.

    X, Y: (n_seeds, T). Returns (n_q,).
    """
    x, y = X.ravel(), Y.ravel()
    qs = np.array(quantiles)
    return np.abs(np.percentile(x, 100 * qs) - np.percentile(y, 100 * qs))


def run_falsification(
    simulate_fn: Callable,
    theta_T: jax.Array,
    eig: EigDecomp,
    alpha_grid: jax.Array,
    M: int,
    key: jax.Array,
) -> FalsificationResult:
    """Run the full §5.4 falsification protocol.

    For v_1 (stiff) and v_P (sloppy): perturb theta_T by +alpha*v for each
    alpha in alpha_grid, simulate M seeds, compute three discrepancies vs
    baseline trajectories at theta_T.
    """
    from curvature_calib.calibration.per_seed_grads import vmap_simulate

    key, k_base = jax.random.split(key)
    base_keys = jax.random.split(k_base, M)
    X_base = np.asarray(vmap_simulate(simulate_fn, theta_T, base_keys))

    n_alpha = len(alpha_grid)
    stiff_mom    = np.zeros((n_alpha, 4))
    stiff_acf_   = np.zeros(n_alpha)
    stiff_quant  = np.zeros((n_alpha, 4))
    sloppy_mom   = np.zeros((n_alpha, 4))
    sloppy_acf_  = np.zeros(n_alpha)
    sloppy_quant = np.zeros((n_alpha, 4))

    v_stiff  = eig.eigvecs[:, 0]
    v_sloppy = eig.eigvecs[:, -1]

    for i, alpha in enumerate(np.asarray(alpha_grid)):
        key, k_s, k_sl = jax.random.split(key, 3)
        X_s  = np.asarray(vmap_simulate(simulate_fn, theta_T + alpha * v_stiff,
                                        jax.random.split(k_s,  M)))
        X_sl = np.asarray(vmap_simulate(simulate_fn, theta_T + alpha * v_sloppy,
                                        jax.random.split(k_sl, M)))
        stiff_mom[i]    = moments_difference(X_s,  X_base)
        stiff_acf_[i]   = acf_difference(X_s,  X_base)
        stiff_quant[i]  = quantile_difference(X_s,  X_base)
        sloppy_mom[i]   = moments_difference(X_sl, X_base)
        sloppy_acf_[i]  = acf_difference(X_sl, X_base)
        sloppy_quant[i] = quantile_difference(X_sl, X_base)

    return FalsificationResult(
        alpha_grid=np.asarray(alpha_grid),
        stiff_moments=stiff_mom,   stiff_acf=stiff_acf_,   stiff_quantiles=stiff_quant,
        sloppy_moments=sloppy_mom, sloppy_acf=sloppy_acf_, sloppy_quantiles=sloppy_quant,
    )
