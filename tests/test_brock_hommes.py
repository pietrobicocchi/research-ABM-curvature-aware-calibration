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
