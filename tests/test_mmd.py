import jax
import jax.numpy as jnp
import pytest

from curvature_calib.losses.mmd import (
    median_heuristic,
    mmd_sq_unbiased,
    mmd_sq_with_median_bandwidth,
    rbf_kernel,
)


def _sample(key, n, mean=0.0, scale=1.0, d=10):
    return mean + scale * jax.random.normal(key, (n, d))


def test_kernel_is_psd_and_one_on_diagonal():
    X = _sample(jax.random.PRNGKey(0), 20, d=5)
    K = rbf_kernel(X, X, 1.0)
    # Diagonal must be 1.
    assert jnp.allclose(jnp.diag(K), 1.0, atol=1e-5)
    # Symmetric.
    assert jnp.allclose(K, K.T, atol=1e-6)
    # PSD up to numerical tolerance: smallest eigenvalue not too negative.
    eigs = jnp.linalg.eigvalsh(K)
    assert float(eigs.min()) > -1e-5


def test_mmd_same_distribution_small():
    """Unbiased MMD^2 between two draws from the same distribution should
    fluctuate around 0 (can be slightly negative because it's unbiased)."""
    k0, k1 = jax.random.split(jax.random.PRNGKey(0))
    X = _sample(k0, 200, d=4)
    Y = _sample(k1, 200, d=4)
    v = float(mmd_sq_unbiased(X, Y, sigma=1.0))
    assert abs(v) < 0.05


def test_mmd_different_distributions_positive():
    k0, k1 = jax.random.split(jax.random.PRNGKey(0))
    X = _sample(k0, 200, mean=0.0, d=4)
    Y = _sample(k1, 200, mean=2.0, d=4)  # shifted
    v = float(mmd_sq_unbiased(X, Y, sigma=1.0))
    assert v > 0.05  # clearly positive


def test_median_heuristic_scale_equivariant():
    X = _sample(jax.random.PRNGKey(0), 100, d=5)
    s1 = float(median_heuristic(X))
    s2 = float(median_heuristic(3.0 * X))
    assert s2 == pytest.approx(3.0 * s1, rel=1e-4)


def test_mmd_with_median_bandwidth_differentiable():
    """Gradient of MMD^2 w.r.t. X must be finite and non-zero when Y != X."""
    k0, k1 = jax.random.split(jax.random.PRNGKey(0))
    X = _sample(k0, 50, mean=0.0, d=3)
    Y = _sample(k1, 50, mean=1.5, d=3)

    def loss(x):
        return mmd_sq_with_median_bandwidth(x, Y)

    g = jax.grad(loss)(X)
    assert g.shape == X.shape
    assert bool(jnp.all(jnp.isfinite(g)))
    assert float(jnp.linalg.norm(g)) > 0.0
