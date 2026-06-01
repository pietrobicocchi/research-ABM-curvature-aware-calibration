import jax
import jax.numpy as jnp

from curvature_calib.models.sir import pack, simulate

# Canonical "moderate epidemic with mid-trajectory lockdown".
#   beta=0.40 -> R_0 ~ 4 (highly transmissible)
#   gamma=0.10 -> mean infectious period 10 days
#   I0_frac=1e-3 -> 100 initial cases in N=1e5
#   t_lock_norm=0.4, f_lock=0.5 -> 50% reduction at 40% through trajectory.
THETA = jnp.array([0.40, 0.10, 1e-3, 0.40, 0.50])


def test_pack_unpacks_correctly():
    p = pack(THETA)
    assert jnp.allclose(p.beta, 0.40)
    assert jnp.allclose(p.gamma, 0.10)
    assert jnp.allclose(p.I0_frac, 1e-3)
    assert jnp.allclose(p.t_lock_norm, 0.40)
    assert jnp.allclose(p.f_lock, 0.50)


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


def test_zero_noise_deterministic():
    a = simulate(THETA, jax.random.PRNGKey(0), T=100, sigma_obs=0.0)
    b = simulate(THETA, jax.random.PRNGKey(999), T=100, sigma_obs=0.0)
    assert jnp.allclose(a, b)


def test_differentiable():
    def loss(t):
        return jnp.sum(simulate(t, jax.random.PRNGKey(0), T=100) ** 2)
    g = jax.grad(loss)(THETA)
    assert g.shape == (5,)
    assert bool(jnp.all(jnp.isfinite(g)))
    assert float(jnp.linalg.norm(g)) > 0.0


def test_epidemic_peaks_in_interior():
    # A reasonable epidemic should crest then decline -- so the maximum of the
    # incidence trajectory lies strictly inside the time window, not at the
    # boundary. Use noiseless to avoid spurious tail maxima.
    xs = simulate(THETA, jax.random.PRNGKey(0), T=300, sigma_obs=0.0)
    peak_idx = int(jnp.argmax(xs))
    assert 5 < peak_idx < 295


def test_lockdown_reduces_peak():
    # Same theta but no lockdown should produce a strictly higher peak.
    no_lock = THETA.at[4].set(1.0)  # f_lock=1 -> no lockdown effect
    with_lock = THETA  # default f_lock=0.5
    xs_no = simulate(no_lock, jax.random.PRNGKey(0), T=300, sigma_obs=0.0)
    xs_yes = simulate(with_lock, jax.random.PRNGKey(0), T=300, sigma_obs=0.0)
    assert float(jnp.max(xs_no)) > float(jnp.max(xs_yes))


def test_grad_horizon_does_not_change_primal_pass():
    k = jax.random.PRNGKey(0)
    full = simulate(THETA, k, T=100, grad_horizon=None)
    trunc = simulate(THETA, k, T=100, grad_horizon=10)
    assert jnp.allclose(full, trunc, atol=1e-3)


def test_grad_horizon_changes_gradient():
    k = jax.random.PRNGKey(0)
    g_full = jax.grad(
        lambda t: jnp.sum(simulate(t, k, T=100) ** 2)
    )(THETA)
    g_trunc = jax.grad(
        lambda t: jnp.sum(simulate(t, k, T=100, grad_horizon=5) ** 2)
    )(THETA)
    assert bool(jnp.all(jnp.isfinite(g_full)))
    assert bool(jnp.all(jnp.isfinite(g_trunc)))
    rel_diff = float(
        jnp.linalg.norm(g_full - g_trunc) /
        jnp.linalg.norm(g_full + 1e-12)
    )
    assert rel_diff > 0.01
