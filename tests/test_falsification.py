import jax
import jax.numpy as jnp
import numpy as np

from curvature_calib.calibration.diagnostic import eigendecompose, opg_from_grads
from curvature_calib.calibration.falsification import (
    FalsificationResult,
    acf_difference,
    moments_difference,
    perturbed_parameters,
    quantile_difference,
    run_falsification,
)


def test_perturbed_parameters_shape():
    theta = jnp.zeros(5)
    direction = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0])
    alphas = jnp.array([0.1, 0.5, 1.0])
    out = perturbed_parameters(theta, direction, alphas)
    assert out.shape == (3, 2, 5)


def test_perturbed_parameters_plus_minus_values():
    theta = jnp.array([1.0, 2.0])
    direction = jnp.array([1.0, 0.0])
    alphas = jnp.array([0.5])
    out = perturbed_parameters(theta, direction, alphas)
    assert jnp.allclose(out[0, 0], jnp.array([1.5, 2.0]))  # +
    assert jnp.allclose(out[0, 1], jnp.array([0.5, 2.0]))  # -


def test_moments_difference_identical_is_zero():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (20, 50))
    assert np.max(moments_difference(X, X)) == 0.0


def test_acf_difference_identical_is_zero():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (20, 50))
    assert acf_difference(X, X) == 0.0


def test_quantile_difference_identical_is_zero():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (20, 50))
    assert np.max(quantile_difference(X, X)) == 0.0


def test_run_falsification_sloppy_smaller_than_stiff():
    """For f(theta) = theta[0] + theta[1], the sloppy direction is (1,-1)/sqrt(2).
    Perturbing along it should produce smaller discrepancies than the stiff (1,1)."""
    T = 30

    def toy_simulate(theta, key):
        noise = jax.random.normal(key, (T,), dtype=theta.dtype)
        return (theta[0] + theta[1]) * jnp.ones(T) + 0.1 * noise

    # OPG: all grads ~ (1, 1), so F ~ [[1,1],[1,1]] / norm
    # stiff eigvec ~ (1,1)/sqrt(2), sloppy ~ (1,-1)/sqrt(2)
    G = jnp.ones((60, 2))
    F = opg_from_grads(G)
    eig = eigendecompose(F)

    result = run_falsification(
        toy_simulate,
        theta_T=jnp.array([1.0, 1.0]),
        eig=eig,
        alpha_grid=jnp.array([0.2]),
        M=40,
        key=jax.random.PRNGKey(0),
    )
    assert isinstance(result, FalsificationResult)
    # Sloppy perturbation: theta[0]+theta[1] unchanged -> very small discrepancy
    # Stiff perturbation: theta[0]+theta[1] changes by 2*alpha -> larger discrepancy
    assert np.max(result.sloppy_moments) < np.max(result.stiff_moments) + 1e-6
