import jax.numpy as jnp

from curvature_calib import metrics as M


def test_rel_frobenius_zero_for_identical():
    A = jnp.arange(9.0).reshape(3, 3)
    assert M.rel_frobenius_error(A, A) == 0.0


def test_rel_frobenius_scale_invariant():
    A = jnp.eye(3)
    B = 1.01 * jnp.eye(3)
    assert abs(M.rel_frobenius_error(B, A) - 0.01) < 1e-12


def test_rel_l2_zero_for_identical():
    v = jnp.array([1.0, 2.0, 3.0])
    assert M.rel_l2_error(v, v) == 0.0


def test_numerical_rank_counts_above_threshold():
    w = jnp.array([1.0, 1e-2, 1e-14, 0.0])
    assert M.numerical_rank(w, rtol=1e-10) == 2


def test_numerical_rank_full():
    w = jnp.array([3.0, 2.0, 1.0])
    assert M.numerical_rank(w) == 3


def test_principal_angles_identical_subspace_zero():
    V = jnp.eye(5)[:, :2]
    assert M.max_principal_angle(V, V) < 1e-6


def test_principal_angles_orthogonal_subspace_pi_over_two():
    V1 = jnp.eye(4)[:, :2]
    V2 = jnp.eye(4)[:, 2:]
    angs = M.subspace_principal_angles(V1, V2)
    assert float(jnp.min(angs)) > jnp.pi / 2 - 1e-6


def test_eigenvalue_rel_error_shape_and_zero():
    w = jnp.array([2.0, 1.0, 0.5])
    e = M.eigenvalue_rel_error(w, w)
    assert e.shape == (3,)
    assert float(jnp.max(e)) == 0.0
