"""EXP-003 estimator correctness: plug-in, cross-seed, bias model, OPG comparators."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from curvature_calib import metrics as M
from curvature_calib.benchmarks import gaussian_location_scale as gls
from curvature_calib.geometry import mmd_estimators as E
from curvature_calib.geometry import rff as R

CFG = gls.GLSConfig(sigma_m=1.0, rho0=0.0, sigma_s=0.3, sigma_l=0.3)
Z = jnp.zeros(5)


def _setup(D=48, gamma=2.0, seed=0):
    r = R.frozen_rff(jax.random.PRNGKey(seed), D, 2, gamma)
    fz = lambda zz, e: R.feature_of_z(CFG, r, zz, e)
    Gref = R.ggn_reference(CFG, r, Z)
    Jeta = R.feature_mean_jacobian(CFG, r, Z)
    return r, fz, Gref, Jeta


def _A(fz, M_seeds, key):
    eps = jax.random.normal(key, (M_seeds, 2))
    return E.feature_jacobians(fz, Z, eps)


def test_plugin_is_psd():
    _, fz, _, _ = _setup()
    A = _A(fz, 32, jax.random.PRNGKey(1))
    w = jnp.linalg.eigvalsh(E.ggn_plugin(A))
    assert float(w[0]) > -1e-10


def test_matrix_free_equals_dense():
    _, fz, _, _ = _setup()
    A = _A(fz, 24, jax.random.PRNGKey(2))
    G = E.ggn_plugin(A)
    for v in (jnp.array([1.0, 0, 0, 0, 0]), jnp.ones(5),
              jax.random.normal(jax.random.PRNGKey(9), (5,))):
        assert M.rel_l2_error(E.ggn_plugin_matvec(A, v), G @ v) <= 1e-10


def test_cross_seed_SQ_equals_explicit_pairs():
    _, fz, _, _ = _setup()
    A = np.asarray(_A(fz, 6, jax.random.PRNGKey(3)))
    m = A.shape[0]
    brute = sum(A[i].T @ A[j] for i in range(m) for j in range(m) if i != j) / (m * (m - 1))
    assert M.rel_frobenius_error(E.ggn_cross_seed(jnp.asarray(A)), jnp.asarray(brute)) <= 1e-10


def test_plugin_bias_decreases_with_M():
    _, fz, Gref, _ = _setup()
    err = {}
    for Ms in (8, 128):
        # average plug-in over batches to expose the (systematic) bias
        accum = jnp.zeros((5, 5))
        B = 60
        for b in range(B):
            accum += E.ggn_plugin(_A(fz, Ms, jax.random.PRNGKey(1000 + b)))
        err[Ms] = M.rel_frobenius_error(accum / B, Gref)
    assert err[128] < err[8]


def test_cross_seed_unbiased_over_batches():
    _, fz, Gref, _ = _setup()
    B, Ms = 200, 16
    accum = jnp.zeros((5, 5))
    for b in range(B):
        accum += E.ggn_cross_seed(_A(fz, Ms, jax.random.PRNGKey(5000 + b)))
    assert M.rel_frobenius_error(accum / B, Gref) < 0.05


def test_predicted_plugin_bias_matches_observed():
    _, fz, Gref, Jeta = _setup(D=40)
    Ms, B = 16, 400
    # observed bias = mean_b Ĝ_V − G_ref
    accum = jnp.zeros((5, 5))
    for b in range(B):
        accum += E.ggn_plugin(_A(fz, Ms, jax.random.PRNGKey(7000 + b)))
    observed = accum / B - Gref
    # predicted = Ĉ / M from a large pooled A
    A_big = _A(fz, 4000, jax.random.PRNGKey(123))
    predicted = E.plugin_bias_covariance(A_big, Jeta) / Ms
    assert M.rel_frobenius_error(observed, predicted) < 0.15


def test_population_opg_zero_at_exact_match_but_gref_nonzero():
    r, fz, Gref, _ = _setup()
    A = _A(fz, 32, jax.random.PRNGKey(11))
    r_zero = R.feature_mean(CFG, r, Z) - R.feature_mean(CFG, r, Z)   # δ=0 exact match
    F = E.population_residual_opg(A, r_zero)
    assert float(jnp.linalg.norm(F)) <= 1e-12
    assert float(jnp.linalg.norm(Gref)) > 1e-3


def test_population_opg_nonzero_and_distinct_under_mismatch():
    r, fz, Gref, _ = _setup()
    A = _A(fz, 64, jax.random.PRNGKey(12))
    z_y = Z + jnp.array([0.6, 0.0, 0.0, 0.0, 0.0])
    res = R.feature_mean(CFG, r, Z) - R.feature_mean(CFG, r, z_y)
    F = E.population_residual_opg(A, res)
    assert float(jnp.linalg.norm(F)) > 0.0
    # trace-normalized shapes differ from the GGN
    assert M.rel_frobenius_error(E.trace_normalized(F), E.trace_normalized(Gref)) > 1e-2
