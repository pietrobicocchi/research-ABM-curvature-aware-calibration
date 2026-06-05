import jax
import jax.numpy as jnp
import numpy as np

from curvature_calib.models.surrogates import gumbel_sigmoid, straight_through_bernoulli


def test_gumbel_sigmoid_shape_and_bounds():
    key = jax.random.PRNGKey(0)
    p = jnp.full((10,), 0.5)
    out = gumbel_sigmoid(p, key, tau=0.5)
    assert out.shape == (10,)
    assert bool(jnp.all(out >= 0.0) and jnp.all(out <= 1.0))


def test_gumbel_sigmoid_mean_tracks_probability():
    """At large M and small tau, mean should be close to p."""
    key = jax.random.PRNGKey(42)
    p = jnp.full((2000,), 0.7)
    out = np.asarray(gumbel_sigmoid(p, key, tau=0.1))
    assert abs(out.mean() - 0.7) < 0.05


def test_straight_through_forward_is_hard():
    """Forward pass must return 0.0 or 1.0 only."""
    key = jax.random.PRNGKey(0)
    p = jnp.full((200,), 0.5)
    out = np.asarray(straight_through_bernoulli(p, key))
    assert set(out).issubset({0.0, 1.0})


def test_straight_through_gradient_passes_through():
    """Gradient of E[sample] wrt p should be ~1.0 (straight-through identity)."""
    def expected_value(p_scalar: jax.Array) -> jax.Array:
        keys = jax.random.split(jax.random.PRNGKey(0), 500)
        samples = jax.vmap(lambda k: straight_through_bernoulli(p_scalar, k))(keys)
        return jnp.mean(samples)

    g = jax.grad(expected_value)(jnp.array(0.5))
    assert abs(float(g) - 1.0) < 0.15
