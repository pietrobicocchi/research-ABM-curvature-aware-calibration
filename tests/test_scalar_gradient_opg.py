"""Rename compatibility + matrix definitions for the historical scalar-gradient
OPG (DEC-001). Ensures renaming did NOT change the numerical object."""
import jax
import jax.numpy as jnp

from curvature_calib.calibration import diagnostic as D
from curvature_calib.calibration.opg import opg_from_grads, scalar_gradient_opg


def _grads(seed=0, M=32, P=5):
    return jax.random.normal(jax.random.PRNGKey(seed), (M, P))


def test_scalar_gradient_opg_definition_unchanged():
    G = _grads()
    M = G.shape[0]
    assert jnp.allclose(D.scalar_gradient_opg(G), (G.T @ G) / M)


def test_deprecated_alias_is_identical():
    G = _grads(1)
    assert opg_from_grads is scalar_gradient_opg
    assert jnp.allclose(opg_from_grads(G), scalar_gradient_opg(G))


def test_covariance_definition_and_moment_relation():
    """C_g = (1/M) Σ (g-ḡ)(g-ḡ)ᵀ  and  F_OPG = C_g + ḡ ḡᵀ."""
    G = _grads(2)
    M = G.shape[0]
    gbar = jnp.mean(G, axis=0)
    C = D.scalar_gradient_covariance(G)
    centered = G - gbar[None]
    assert jnp.allclose(C, (centered.T @ centered) / M)
    assert jnp.allclose(D.scalar_gradient_opg(G), C + jnp.outer(gbar, gbar), atol=1e-5)


def test_calibstats_opg_field_matches_scalar_gradient_opg():
    from curvature_calib.calibration.per_seed_grads import per_seed_loss_and_grads, vmap_simulate
    from curvature_calib.models.brock_hommes import simulate

    theta = jnp.array([3.0, 1.2, 0.2, 1.2, -0.2])
    keys = jax.random.split(jax.random.PRNGKey(0), 16)
    Y = vmap_simulate(lambda t, k: simulate(t, k, T=120, sigma=0.05, R=1.1), theta, keys)
    stats = per_seed_loss_and_grads(
        lambda t, k: simulate(t, k, T=120, sigma=0.05, R=1.1), theta, keys, Y)
    assert jnp.allclose(stats.opg, D.scalar_gradient_opg(stats.per_seed_grads))
