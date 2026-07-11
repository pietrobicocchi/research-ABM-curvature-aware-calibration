import jax
import jax.numpy as jnp
import pytest

from curvature_calib.benchmarks import linear_gaussian as lg


@pytest.mark.parametrize("cond", [1.0, 1e2, 1e6])
def test_random_A_hits_condition_number(cond):
    A = lg.random_A(jax.random.PRNGKey(0), K=40, P=20, cond=cond)
    s = jnp.linalg.svd(A, compute_uv=False)
    cond_AtA = float((s[0] / s[-1]) ** 2)
    assert cond_AtA == pytest.approx(cond, rel=1e-6)


def test_random_A_rank_deficient():
    A = lg.random_A(jax.random.PRNGKey(1), K=40, P=20, cond=1e2, rank=12)
    s = jnp.linalg.svd(A, compute_uv=False)
    assert int(jnp.sum(s > 1e-8 * s[0])) == 12


def test_full_rank_requires_K_ge_P():
    with pytest.raises(ValueError):
        lg.random_A(jax.random.PRNGKey(2), K=5, P=10)


def test_analytic_ggn_equals_AtWA_none():
    A = lg.random_A(jax.random.PRNGKey(3), K=12, P=6, cond=10.0)
    model = lg.LinearGaussian(A=A, y=jnp.zeros(12), W=None)
    assert jnp.allclose(lg.analytic_ggn(model), A.T @ A, atol=1e-10)


def test_analytic_gradient_matches_autodiff():
    A = lg.random_A(jax.random.PRNGKey(4), K=12, P=6, cond=10.0)
    y = jax.random.normal(jax.random.PRNGKey(5), (12,))
    W = jnp.linspace(0.5, 2.0, 12)
    model = lg.LinearGaussian(A=A, y=y, W=W)
    z = jax.random.normal(jax.random.PRNGKey(6), (6,))
    g_analytic = lg.analytic_gradient(model, z)
    g_ad = jax.grad(lg.make_loss(model))(z)
    assert jnp.allclose(g_analytic, g_ad, atol=1e-10)
