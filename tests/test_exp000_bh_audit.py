"""EXP-000 audit primitives on Brock–Hommes (built from library modules)."""
import jax
import jax.numpy as jnp
import numpy as np

from curvature_calib import metrics as M
from curvature_calib.calibration import diagnostic as D
from curvature_calib.calibration.per_seed_grads import per_seed_loss_and_grads, vmap_simulate
from curvature_calib.geometry import mmd_estimators as E
from curvature_calib.geometry import rff as R
from curvature_calib.losses.mmd import median_heuristic
from curvature_calib.models.brock_hommes import simulate

T, RG, SIG = 80, 1.1, 0.03
S = jnp.array([0.30, 0.10, 0.05, 0.10, 0.05])
THETA0 = jnp.array([3.0, 1.35, 0.15, 1.25, -0.10])


def _sim(z, key):
    return simulate(THETA0 + S * z, key, T=T, R=RG, sigma=SIG)


def test_common_random_numbers_reproducible():
    """Same key ⇒ identical trajectory (keys ARE the reparameterized ε)."""
    k = jax.random.PRNGKey(5)
    assert jnp.allclose(_sim(jnp.zeros(5), k), _sim(jnp.zeros(5), k))


def _setup(M_seeds=32, D=128, seed=0):
    keys = jax.random.split(jax.random.PRNGKey(seed), M_seeds)
    Xp = vmap_simulate(_sim, jnp.zeros(5), keys)
    gamma = float(median_heuristic(Xp))
    rff = R.frozen_rff(jax.random.PRNGKey(7), D, T, gamma)
    fz = lambda z, key: R.feature_map(rff, _sim(z, key))
    eta_hat = lambda z: jnp.mean(jax.vmap(lambda kk: fz(z, kk))(keys), axis=0)
    return keys, rff, fz, eta_hat


def test_plugin_G_equals_jacobian_of_eta_hat():
    keys, rff, fz, eta_hat = _setup()
    A = E.feature_jacobians(fz, jnp.zeros(5), keys)
    Jeta = jax.jacfwd(eta_hat)(jnp.zeros(5))
    assert M.rel_frobenius_error(jnp.mean(A, axis=0), Jeta) < 1e-8
    G_plugin = E.ggn_plugin(A)
    assert M.rel_frobenius_error(G_plugin, Jeta.T @ Jeta) < 1e-8


def test_hessian_equals_G_at_zero_residual():
    keys, rff, fz, eta_hat = _setup()
    z0 = jnp.zeros(5)
    eta_y = eta_hat(z0)                       # δ=0 self-match ⇒ residual 0
    loss = lambda z: 0.5 * jnp.sum((eta_hat(z) - eta_y) ** 2)
    H = jax.hessian(loss)(z0)
    Jeta = jax.jacfwd(eta_hat)(z0)
    G = Jeta.T @ Jeta
    assert M.rel_frobenius_error(0.5 * (H + H.T), G) < 1e-6   # R = H - G ≈ 0


def test_historical_opg_is_tiny_relative_to_true_ggn():
    """The historical F_OPG is orders of magnitude smaller than the true GGN in
    Frobenius norm — a different object, not a curvature estimate (DEC-001)."""
    keys, rff, fz, eta_hat = _setup()
    z0 = jnp.zeros(5)
    G_fro = float(jnp.linalg.norm(E.ggn_plugin(E.feature_jacobians(fz, z0, keys))))
    Y = vmap_simulate(_sim, z0 + jnp.array([0.5, 0, 0, 0, 0]), keys)
    F_fro = float(jnp.linalg.norm(per_seed_loss_and_grads(_sim, z0, keys, Y).opg))
    assert F_fro < 0.05 * G_fro


def test_opg_covariance_moment_relation_on_bh():
    keys, rff, fz, eta_hat = _setup()
    z0 = jnp.zeros(5)
    Y = vmap_simulate(_sim, z0 + jnp.array([0.5, 0, 0, 0, 0]), keys)
    stats = per_seed_loss_and_grads(_sim, z0, keys, Y)
    g = stats.per_seed_grads
    gbar = jnp.mean(g, axis=0)
    F = D.scalar_gradient_opg(g)
    C = D.scalar_gradient_covariance(g)
    assert jnp.allclose(F, C + jnp.outer(gbar, gbar), atol=1e-5)
