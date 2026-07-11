"""EXP-003 reference: closed-form feature mean, J_eta, G_ref, exact RBF GGN."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from curvature_calib import metrics as M
from curvature_calib.benchmarks import gaussian_location_scale as gls
from curvature_calib.geometry import rff as R

CFG = gls.GLSConfig(sigma_m=1.0, rho0=0.0, sigma_s=0.3, sigma_l=0.3)
Z = jnp.array([0.3, -0.2, 0.1, -0.15, 0.25])


def test_closed_form_feature_mean_matches_mc():
    r = R.frozen_rff(jax.random.PRNGKey(0), D=128, n=2, gamma=2.0)
    eta = R.feature_mean(CFG, r, Z)
    eps = jax.random.normal(jax.random.PRNGKey(1), (400_000, 2))
    X = gls.gaussian_mean(CFG, Z) + eps @ gls.cholesky(CFG, Z).T
    eta_mc = R.feature_map(r, X).mean(0)
    assert M.rel_frobenius_error(eta, eta_mc) < 5e-3  # MC floor ~1/sqrt(4e5)


def test_feature_mean_jacobian_matches_finite_difference():
    r = R.frozen_rff(jax.random.PRNGKey(0), D=128, n=2, gamma=2.0)
    J = np.asarray(R.feature_mean_jacobian(CFG, r, Z))
    h = 1e-5
    Jfd = np.zeros_like(J)
    for i in range(5):
        e = np.zeros(5); e[i] = h
        Jfd[:, i] = np.asarray(
            (R.feature_mean(CFG, r, Z + jnp.array(e)) - R.feature_mean(CFG, r, Z - jnp.array(e)))
            / (2 * h)
        )
    assert M.rel_frobenius_error(jnp.asarray(J), jnp.asarray(Jfd)) < 1e-7


def test_ggn_reference_symmetric_psd():
    r = R.frozen_rff(jax.random.PRNGKey(0), D=256, n=2, gamma=2.0)
    G = R.ggn_reference(CFG, r, Z)
    assert jnp.allclose(G, G.T, atol=1e-12)
    assert float(jnp.linalg.eigvalsh(G)[0]) > -1e-10


def test_frozen_features_reproducible():
    r1 = R.frozen_rff(jax.random.PRNGKey(7), D=64, n=2, gamma=2.0)
    r2 = R.frozen_rff(jax.random.PRNGKey(7), D=64, n=2, gamma=2.0)
    assert jnp.allclose(r1.w, r2.w) and jnp.allclose(r1.b, r2.b)


def test_main_regime_has_clear_eigengap_after_k2():
    r = R.frozen_rff(jax.random.PRNGKey(0), D=1024, n=2, gamma=2.0)
    w = np.sort(np.asarray(jnp.linalg.eigvalsh(R.ggn_reference(CFG, r, jnp.zeros(5)))))[::-1]
    assert w[1] / w[2] > 5.0  # clear gap after the top-2 mean subspace


def test_rbf_ggn_symmetric_psd_and_finite_D_approaches_it():
    G_rbf = R.ggn_rbf(CFG, gamma=2.0, z=Z)
    assert jnp.allclose(G_rbf, G_rbf.T, atol=1e-10)
    assert float(jnp.linalg.eigvalsh(G_rbf)[0]) > -1e-8
    # average finite-D G_ref over feature seeds approaches G_RBF as D grows
    def avg_gref(D, seeds=6):
        Gs = [R.ggn_reference(CFG, R.frozen_rff(jax.random.PRNGKey(s), D, 2, 2.0), Z)
              for s in range(seeds)]
        return sum(Gs) / seeds
    e_small = M.rel_frobenius_error(avg_gref(128), G_rbf)
    e_large = M.rel_frobenius_error(avg_gref(2048), G_rbf)
    assert e_large < e_small
    assert e_large < 0.10
