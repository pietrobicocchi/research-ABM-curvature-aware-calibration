import jax
import jax.numpy as jnp
import numpy as np

from curvature_calib.calibration.diagnostic import (
    EigDecomp,
    d_eff_from_bootstrap,
    effective_dimension,
    eigendecompose,
    opg_from_grads,
    principal_angles,
)


def test_eigendecompose_descending_order():
    F = jnp.array([[4.0, 1.0], [1.0, 2.0]])
    eig = eigendecompose(F)
    assert eig.eigvals[0] >= eig.eigvals[1]


def test_eigendecompose_reconstruction():
    F = jnp.array([[3.0, 1.0], [1.0, 2.0]])
    eig = eigendecompose(F)
    F_rec = eig.eigvecs @ jnp.diag(eig.eigvals) @ eig.eigvecs.T
    assert jnp.allclose(F_rec, 0.5 * (F + F.T), atol=1e-5)


def test_opg_from_grads_shape_and_symmetry():
    G = jax.random.normal(jax.random.PRNGKey(0), (20, 5))
    F = opg_from_grads(G)
    assert F.shape == (5, 5)
    assert jnp.allclose(F, F.T, atol=1e-6)


def test_opg_from_grads_sum_only_model():
    """Simulator f(t1, t2) = t1 + t2 has all grads in (1,1) direction.
    OPG's smallest eigenvalue should be numerically zero."""
    G = jnp.ones((100, 2))  # every g_m = (1, 1)
    F = opg_from_grads(G)
    eig = eigendecompose(F)
    assert eig.eigvals[-1] / (eig.eigvals[0] + 1e-30) < 1e-5


def test_principal_angles_same_subspace():
    V = jax.random.normal(jax.random.PRNGKey(0), (5, 3))
    # rotate V by a random orthogonal matrix — same subspace
    Q, _ = jnp.linalg.qr(jax.random.normal(jax.random.PRNGKey(1), (3, 3)))
    ang = principal_angles(V, V @ Q)
    assert float(jnp.max(ang)) < 1e-3


def test_principal_angles_orthogonal():
    V1 = jnp.eye(4)[:, :2]
    V2 = jnp.eye(4)[:, 2:]
    ang = principal_angles(V1, V2)
    assert float(jnp.min(ang)) > jnp.pi / 2 - 1e-3


def test_effective_dimension_counts_above_floor():
    eigvals = jnp.array([10.0, 5.0, 1.0, 0.1, 0.001])
    assert effective_dimension(eigvals, noise_floor=0.5) == 3
    assert effective_dimension(eigvals, noise_floor=0.0) == 5
    assert effective_dimension(eigvals, noise_floor=20.0) == 0


def test_d_eff_from_bootstrap_counts_positive_lower_ci():
    # (P, 2): column 0 = lower, column 1 = upper
    cis = jnp.array([[0.1, 0.5], [0.05, 0.3], [-0.01, 0.1], [-0.1, 0.0]])
    assert d_eff_from_bootstrap(cis) == 2
