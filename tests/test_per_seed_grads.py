import jax
import jax.numpy as jnp

from curvature_calib.calibration.opg import (
    eigendecompose,
    opg_from_grads,
    principal_angles,
)
from curvature_calib.calibration.per_seed_grads import (
    per_seed_loss_and_grads,
    vmap_simulate,
)
from curvature_calib.models.brock_hommes import simulate

THETA = jnp.array([3.0, 1.2, 0.2, 1.2, -0.2])
T_REF = 200


def _make_ref(theta=THETA, M=64, seed=0, T=T_REF):
    keys = jax.random.split(jax.random.PRNGKey(seed), M)
    return vmap_simulate(lambda t, k: simulate(t, k, T=T, sigma=0.05, R=1.1, x_init=0.0),
                         theta, keys)


def _sim(theta, key):
    return simulate(theta, key, T=T_REF, sigma=0.05, R=1.1, x_init=0.0)


def test_mean_grad_matches_autodiff_loss_gradient():
    """Per-seed mean grad must equal nabla_theta MMD^2(theta) computed end-to-end."""
    Y = _make_ref()
    M = 32
    keys = jax.random.split(jax.random.PRNGKey(99), M)
    stats = per_seed_loss_and_grads(_sim, THETA, keys, Y)

    # Reference: end-to-end autodiff
    from curvature_calib.losses.mmd import mmd_sq_with_median_bandwidth

    def total_loss(theta):
        X = vmap_simulate(_sim, theta, keys)
        return mmd_sq_with_median_bandwidth(X, Y)

    g_ref = jax.grad(total_loss)(THETA)
    assert jnp.allclose(stats.mean_grad, g_ref, atol=1e-5, rtol=1e-4), (
        f"\nmean_grad: {stats.mean_grad}\ng_ref:    {g_ref}"
    )


def test_per_seed_shapes():
    Y = _make_ref()
    M = 16
    keys = jax.random.split(jax.random.PRNGKey(123), M)
    s = per_seed_loss_and_grads(_sim, THETA, keys, Y)
    P = THETA.shape[0]
    assert s.per_seed_grads.shape == (M, P)
    assert s.mean_grad.shape == (P,)
    assert s.opg.shape == (P, P)


def test_opg_symmetric_psd():
    Y = _make_ref()
    M = 20
    keys = jax.random.split(jax.random.PRNGKey(7), M)
    s = per_seed_loss_and_grads(_sim, THETA, keys, Y)
    # Symmetric.
    assert jnp.allclose(s.opg, s.opg.T, atol=1e-6)
    # PSD.
    eigs = jnp.linalg.eigvalsh(0.5 * (s.opg + s.opg.T))
    assert float(eigs.min()) > -1e-6


def test_eigendecomposition_descending():
    Y = _make_ref()
    keys = jax.random.split(jax.random.PRNGKey(7), 20)
    s = per_seed_loss_and_grads(_sim, THETA, keys, Y)
    d = eigendecompose(s.opg)
    diffs = jnp.diff(d.eigvals)
    assert bool(jnp.all(diffs <= 1e-8))


def test_principal_angles_zero_for_same_subspace():
    V = jax.random.normal(jax.random.PRNGKey(0), (5, 3))
    ang = principal_angles(V, V @ jax.random.normal(jax.random.PRNGKey(1), (3, 3)))
    # arccos near 1 is numerically tricky in float32 (sqrt(1 - s^2) catastrophic
    # cancellation); 1e-3 tolerance is the standard ceiling.
    assert float(jnp.max(ang)) < 1e-3


def test_principal_angles_orthogonal_subspaces():
    V1 = jnp.eye(4)[:, :2]
    V2 = jnp.eye(4)[:, 2:]
    ang = principal_angles(V1, V2)
    assert float(jnp.min(ang)) > jnp.pi / 2 - 1e-3


def test_per_seed_grads_at_truth_have_smaller_norm():
    """Sanity: at theta = truth, MMD^2 is at its noise floor and per-seed
    gradients are small relative to those at a perturbed theta."""
    Y = _make_ref()
    keys = jax.random.split(jax.random.PRNGKey(11), 32)
    s_truth = per_seed_loss_and_grads(_sim, THETA, keys, Y)
    # Use a small perturbation: bigger ones (e.g. +0.5) push g_h above the
    # explosion threshold (g/R > 1.5 with R=1.1) and trajectories diverge.
    s_off = per_seed_loss_and_grads(_sim, THETA + jnp.array([0.0, 0.1, 0.05, 0.1, 0.05]), keys, Y)
    assert float(jnp.linalg.norm(s_off.mean_grad)) > float(
        jnp.linalg.norm(s_truth.mean_grad)
    )
