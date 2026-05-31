import jax
import jax.numpy as jnp

from curvature_calib.models.brock_hommes import pack_canonical, simulate

THETA = jnp.array([1.0, 0.9, 0.2, 0.9, -0.2])


def test_pack_canonical_fundamentalist_is_zero():
    p = pack_canonical(THETA)
    assert jnp.allclose(p.g, jnp.array([0.0, 0.9, 0.9]))
    assert jnp.allclose(p.b, jnp.array([0.0, 0.2, -0.2]))


def test_shape_and_finite():
    xs = simulate(THETA, jax.random.PRNGKey(0), T=200)
    assert xs.shape == (200,)
    assert bool(jnp.all(jnp.isfinite(xs)))


def test_seed_reproducibility():
    k = jax.random.PRNGKey(42)
    assert jnp.allclose(simulate(THETA, k, T=100), simulate(THETA, k, T=100))


def test_different_seeds_diverge():
    a = simulate(THETA, jax.random.PRNGKey(0), T=100)
    b = simulate(THETA, jax.random.PRNGKey(1), T=100)
    assert not jnp.allclose(a, b)


def test_differentiable():
    def loss(t):
        return jnp.mean(simulate(t, jax.random.PRNGKey(0), T=100) ** 2)

    g = jax.grad(loss)(THETA)
    assert g.shape == (5,)
    assert bool(jnp.all(jnp.isfinite(g)))
    assert float(jnp.linalg.norm(g)) > 0.0


def test_zero_noise_deterministic():
    # With sigma=0, the same theta should give the same trajectory regardless of key.
    a = simulate(THETA, jax.random.PRNGKey(0), T=50, sigma=0.0)
    b = simulate(THETA, jax.random.PRNGKey(999), T=50, sigma=0.0)
    assert jnp.allclose(a, b)


def test_fundamentalist_only_stays_near_zero():
    # If beta is large and free types' biases dominate fundamentalist, dynamics
    # should still be bounded over a short horizon under small noise.
    theta = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0])  # all-zero -> trivial
    xs = simulate(theta, jax.random.PRNGKey(0), T=100, sigma=0.01)
    assert float(jnp.max(jnp.abs(xs))) < 1.0


def test_grad_horizon_does_not_change_primal_pass():
    """Forward (primal) pass must be identical regardless of grad_horizon."""
    k = jax.random.PRNGKey(0)
    full = simulate(THETA, k, T=100, sigma=0.05, R=1.1, grad_horizon=None)
    trunc = simulate(THETA, k, T=100, sigma=0.05, R=1.1, grad_horizon=10)
    assert jnp.allclose(full, trunc, atol=1e-5)


def test_grad_horizon_full_matches_default():
    """grad_horizon >= T should be identical to default."""
    k = jax.random.PRNGKey(0)
    full = simulate(THETA, k, T=50, sigma=0.05, R=1.1)
    same = simulate(THETA, k, T=50, sigma=0.05, R=1.1, grad_horizon=50)
    larger = simulate(THETA, k, T=50, sigma=0.05, R=1.1, grad_horizon=200)
    assert jnp.allclose(full, same)
    assert jnp.allclose(full, larger)


def test_grad_horizon_changes_gradient():
    """Truncated gradient should differ from full gradient."""
    k = jax.random.PRNGKey(0)

    def loss_full(t):
        return jnp.mean(simulate(t, k, T=100, sigma=0.05, R=1.1) ** 2)

    def loss_trunc(t):
        return jnp.mean(
            simulate(t, k, T=100, sigma=0.05, R=1.1, grad_horizon=10) ** 2
        )

    g_full = jax.grad(loss_full)(THETA)
    g_trunc = jax.grad(loss_trunc)(THETA)
    # Both must be finite, but they should differ.
    assert bool(jnp.all(jnp.isfinite(g_full)))
    assert bool(jnp.all(jnp.isfinite(g_trunc)))
    rel_diff = float(jnp.linalg.norm(g_full - g_trunc) /
                     jnp.linalg.norm(g_full + 1e-12))
    assert rel_diff > 0.01, f"Truncation made no difference: rel_diff={rel_diff}"


def test_grad_horizon_one_means_only_last_step():
    """With grad_horizon=1, only the very last step's contribution flows back."""
    k = jax.random.PRNGKey(0)

    def loss(t):
        return jnp.mean(
            simulate(t, k, T=50, sigma=0.05, R=1.1, grad_horizon=1) ** 2
        )

    g = jax.grad(loss)(THETA)
    assert bool(jnp.all(jnp.isfinite(g)))
    # With only 1 step of gradient, ||g|| should be much smaller than the full
    # gradient.
    def loss_full(t):
        return jnp.mean(simulate(t, k, T=50, sigma=0.05, R=1.1) ** 2)

    g_full = jax.grad(loss_full)(THETA)
    assert float(jnp.linalg.norm(g)) < float(jnp.linalg.norm(g_full))
