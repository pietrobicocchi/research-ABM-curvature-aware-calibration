"""EXP-002 benchmark checks: analytic H=G+R vs AD, minimizer, R behavior."""
import jax
import jax.numpy as jnp
import pytest

from curvature_calib import metrics as M
from curvature_calib.benchmarks import nonlinear_residual as nr
from curvature_calib.geometry import ggn as G

AD_TOL = 1e-10
R_TOL = 1e-8


# --------------------------- Case A: Rosenbrock ---------------------------- #
@pytest.mark.parametrize("a", [1.0, 5.0, 10.0])
def test_rosen_ad_ggn_matches_analytic(a):
    model = nr.RosenbrockResidual(a=a, b=1.0)
    r = nr.rosen_residual_fn(model)
    z = jnp.array([0.7, 1.3])
    assert M.rel_frobenius_error(G.ggn_dense(r, z), nr.rosen_ggn(model, z)) <= AD_TOL


@pytest.mark.parametrize("a", [1.0, 5.0, 10.0])
def test_rosen_ad_hessian_matches_analytic(a):
    model = nr.RosenbrockResidual(a=a, b=1.0)
    loss = nr.rosen_loss_fn(model)
    z = jnp.array([0.7, 1.3])
    assert M.rel_frobenius_error(G.exact_hessian(loss, z), nr.rosen_hessian(model, z)) <= AD_TOL


def test_rosen_zero_residual_curvature_at_optimum():
    model = nr.RosenbrockResidual(a=5.0, b=1.0)
    loss = nr.rosen_loss_fn(model)
    r = nr.rosen_residual_fn(model)
    z_opt = nr.rosen_optimum(model)
    assert float(jnp.linalg.norm(jax.grad(loss)(z_opt))) <= 1e-12
    H = G.exact_hessian(loss, z_opt)
    Gm = G.ggn_dense(r, z_opt)
    assert M.rel_frobenius_error(H, Gm) <= AD_TOL  # R = 0 at exact fit


def test_rosen_residual_curvature_grows_away_from_optimum():
    model = nr.RosenbrockResidual(a=5.0, b=1.0)
    z_opt = nr.rosen_optimum(model)
    d = jnp.array([0.3, -0.4])
    norms = [float(jnp.linalg.norm(nr.rosen_residual_curvature(model, z_opt + t * d)))
             for t in (0.0, 0.2, 0.5, 1.0)]
    assert norms[0] <= 1e-12
    assert all(norms[i] < norms[i + 1] for i in range(len(norms) - 1))


# --------------------------- Case B: irreducible --------------------------- #
@pytest.mark.parametrize("lam,c", [(0.3, 0.0), (1.0, 4.0)])
def test_irr_ad_ggn_and_hessian_match_analytic(lam, c):
    model = nr.IrreducibleResidual(lam=lam, c=c)
    r = nr.irr_residual_fn(model)
    loss = nr.irr_loss_fn(model)
    z = jnp.array([0.8, 1.1])
    assert M.rel_frobenius_error(G.ggn_dense(r, z), nr.irr_ggn(model, z)) <= AD_TOL
    assert M.rel_frobenius_error(G.exact_hessian(loss, z), nr.irr_hessian(model, z)) <= AD_TOL


@pytest.mark.parametrize("lam,c", [(0.3, 0.0), (1.0, 4.0)])
def test_irr_residual_curvature_matches_closed_form(lam, c):
    model = nr.IrreducibleResidual(lam=lam, c=c)
    r = nr.irr_residual_fn(model)
    loss = nr.irr_loss_fn(model)
    z = jnp.array([0.8, 1.1])
    R_ad = G.exact_hessian(loss, z) - G.ggn_dense(r, z)
    assert M.rel_frobenius_error(R_ad, nr.irr_residual_curvature(model, z)) <= R_TOL


@pytest.mark.parametrize("lam,c", [(0.3, 0.0), (1.0, 4.0)])
def test_irr_minimizer_has_nonzero_residual_curvature(lam, c):
    model = nr.IrreducibleResidual(lam=lam, c=c)
    loss = nr.irr_loss_fn(model)
    r = nr.irr_residual_fn(model)
    z_hat = nr.newton_minimize(loss, jnp.array([1.0, 1.0]))
    assert float(jnp.linalg.norm(jax.grad(loss)(z_hat))) <= 1e-10  # minimizer check
    H = G.exact_hessian(loss, z_hat)
    R = H - G.ggn_dense(r, z_hat)
    assert float(jnp.linalg.norm(R)) / float(jnp.linalg.norm(H)) > 1e-3  # R != 0


def test_irr_ad_fd_hessian_cross_check():
    """FD Hessian agrees with AD Hessian under its own looser tolerance."""
    model = nr.IrreducibleResidual(lam=1.0, c=4.0)
    loss = nr.irr_loss_fn(model)
    z = jnp.array([0.8, 1.1])
    H_ad = G.exact_hessian(loss, z)
    best = min(M.rel_frobenius_error(G.finite_difference_hessian(loss, z, h), H_ad)
               for h in (1e-3, 1e-4, 1e-5))
    assert best <= 1e-5
