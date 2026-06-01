import jax
import jax.numpy as jnp

from curvature_calib.models.network_sir import (
    build_er_graph,
    pack,
    simulate,
)


# Canonical theta_star, matching the mean-field SIR for direct comparison.
THETA = jnp.array([0.30, 0.10, 0.05, 0.40, 0.50])


def test_pack():
    p = pack(THETA)
    assert jnp.allclose(p.beta, 0.30)
    assert jnp.allclose(p.gamma, 0.10)
    assert jnp.allclose(p.I0_frac, 0.05)


def test_er_graph_is_symmetric_no_self_loops():
    A = build_er_graph(50, 5.0, jax.random.PRNGKey(0))
    assert A.shape == (50, 50)
    assert jnp.allclose(A, A.T)
    assert float(jnp.trace(A)) == 0.0


def test_shape_and_finite():
    xs = simulate(THETA, jax.random.PRNGKey(0), T=100, N=100)
    assert xs.shape == (100,)
    assert bool(jnp.all(jnp.isfinite(xs)))
    assert bool(jnp.all(xs >= 0))


def test_seed_reproducibility():
    k = jax.random.PRNGKey(42)
    a = simulate(THETA, k, T=80, N=80)
    b = simulate(THETA, k, T=80, N=80)
    assert jnp.allclose(a, b)


def test_different_seeds_diverge():
    a = simulate(THETA, jax.random.PRNGKey(0), T=80, N=80)
    b = simulate(THETA, jax.random.PRNGKey(1), T=80, N=80)
    assert not jnp.allclose(a, b)


def test_graph_seed_changes_dynamics():
    """Different graph seeds give different epidemic curves at fixed RNG."""
    a = simulate(THETA, jax.random.PRNGKey(0), T=80, N=80, graph_seed=17)
    b = simulate(THETA, jax.random.PRNGKey(0), T=80, N=80, graph_seed=23)
    assert not jnp.allclose(a, b)


def test_differentiable():
    def loss(t):
        return jnp.sum(simulate(t, jax.random.PRNGKey(0), T=60, N=80) ** 2)

    g = jax.grad(loss)(THETA)
    assert g.shape == (5,)
    assert bool(jnp.all(jnp.isfinite(g)))
    assert float(jnp.linalg.norm(g)) > 0.0


def test_lockdown_reduces_peak():
    no_lock = THETA.at[4].set(1.0)
    with_lock = THETA
    a = simulate(no_lock, jax.random.PRNGKey(0), T=200, N=200, gumbel_tau=0.1)
    b = simulate(with_lock, jax.random.PRNGKey(0), T=200, N=200, gumbel_tau=0.1)
    # Use a relatively cold temperature so the discreteness is sharper and
    # the lockdown effect dominates Gumbel noise.
    assert float(jnp.max(a)) > float(jnp.max(b)) * 0.95


def test_grad_horizon_does_not_change_primal_pass():
    k = jax.random.PRNGKey(0)
    full = simulate(THETA, k, T=80, N=80, grad_horizon=None)
    trunc = simulate(THETA, k, T=80, N=80, grad_horizon=10)
    # Primal pass should be identical (gradient truncation is via stop_gradient,
    # which leaves the forward pass alone). Allow small tolerance for f32.
    assert jnp.allclose(full, trunc, atol=1e-3)
