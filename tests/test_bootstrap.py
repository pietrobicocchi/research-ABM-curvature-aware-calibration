import jax
import jax.numpy as jnp
import numpy as np

from curvature_calib.calibration.bootstrap import (
    bootstrap_eigvals,
    bootstrap_subspace_cis,
    eigenvalue_cis,
    noise_threshold,
)


def _make_grads(seed=0, M=50, P=3):
    return jax.random.normal(jax.random.PRNGKey(seed), (M, P))


def test_bootstrap_eigvals_shape():
    G = _make_grads()
    boot = bootstrap_eigvals(G, n_boot=20, key=jax.random.PRNGKey(0))
    assert boot.shape == (20, 3)


def test_bootstrap_eigvals_nonnegative():
    G = _make_grads()
    boot = bootstrap_eigvals(G, n_boot=50, key=jax.random.PRNGKey(1))
    assert float(jnp.min(boot)) >= -1e-8


def test_eigenvalue_cis_shape_and_ordering():
    G = _make_grads()
    boot = bootstrap_eigvals(G, n_boot=50, key=jax.random.PRNGKey(0))
    cis = eigenvalue_cis(boot)
    assert cis.shape == (3, 2)
    # Lower bound <= upper bound for every eigenvalue
    assert bool(jnp.all(cis[:, 0] <= cis[:, 1]))


def test_noise_threshold_is_upper_ci_of_smallest():
    cis = jnp.array([[1.0, 2.0], [0.5, 1.0], [0.01, 0.05]])
    assert abs(noise_threshold(cis) - 0.05) < 1e-6


def test_bootstrap_subspace_cis_large_M_small_angle():
    """With many seeds and a clear spectral gap, bootstrap variability is low — CI bound < 0.5 rad.

    Uses structured data where the first two parameter dimensions have 10x larger
    gradients, creating a well-defined top-2 subspace (spectral gap ~100x).
    Isotropic data has no spectral gap so top-k subspaces are arbitrary.
    """
    G_raw = jax.random.normal(jax.random.PRNGKey(0), (200, 4))
    G = G_raw.at[:, :2].multiply(10.0)  # clear spectral gap: top-2 eigenvalues ~100x larger
    angle = bootstrap_subspace_cis(G, k=2, n_boot=30, key=jax.random.PRNGKey(0))
    assert angle < 0.5
