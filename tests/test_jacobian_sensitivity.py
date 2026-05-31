import jax
import jax.numpy as jnp

from curvature_calib.calibration.jacobian_sensitivity import (
    opg_correlation_matrix,
    opg_diagonal_sensitivity,
    per_param_jacobian_sensitivity,
)
from curvature_calib.models.brock_hommes import simulate

THETA = jnp.array([3.0, 1.2, 0.2, 1.2, -0.2])


def _sim(theta, key):
    return simulate(theta, key, T=80, sigma=0.05, R=1.1, x_init=0.0)


def test_per_param_sensitivity_shape_and_positive():
    keys = jax.random.split(jax.random.PRNGKey(0), 16)
    s = per_param_jacobian_sensitivity(_sim, THETA, keys)
    assert s.shape == (5,)
    assert bool(jnp.all(s > 0))
    assert bool(jnp.all(jnp.isfinite(s)))


def test_opg_diagonal_matches_per_seed_grad_diagonal():
    """sqrt(diag(F_hat)) should equal the std of per-seed gradient components."""
    from curvature_calib.calibration.per_seed_grads import (
        per_seed_loss_and_grads, vmap_simulate,
    )
    keys = jax.random.split(jax.random.PRNGKey(2), 32)
    Y_ref = vmap_simulate(_sim, THETA, jax.random.split(jax.random.PRNGKey(99), 32))
    stats = per_seed_loss_and_grads(_sim, THETA + 0.02, keys, Y_ref)
    diag_sens = opg_diagonal_sensitivity(stats.opg)
    # Compute std of per-seed grads manually.
    M = stats.per_seed_grads.shape[0]
    std_manual = jnp.sqrt(
        (stats.per_seed_grads ** 2).sum(axis=0) / M  # E[g^2], not Var (centered or not)
    )
    assert jnp.allclose(diag_sens, std_manual, atol=1e-5, rtol=1e-4)


def test_correlation_matrix_diagonal_is_one():
    from curvature_calib.calibration.per_seed_grads import (
        per_seed_loss_and_grads, vmap_simulate,
    )
    keys = jax.random.split(jax.random.PRNGKey(3), 32)
    Y_ref = vmap_simulate(_sim, THETA, jax.random.split(jax.random.PRNGKey(99), 32))
    stats = per_seed_loss_and_grads(_sim, THETA + 0.02, keys, Y_ref)
    corr = opg_correlation_matrix(stats.opg)
    assert jnp.allclose(jnp.diag(corr), 1.0, atol=1e-5)
    assert jnp.allclose(corr, corr.T, atol=1e-5)
