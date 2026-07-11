"""EXP-002 local quadratic-prediction / validity-radius machinery."""
import jax
import jax.numpy as jnp
import numpy as np

from curvature_calib.benchmarks import linear_gaussian as lg
from curvature_calib.benchmarks import nonlinear_residual as nr
from curvature_calib.calibration.diagnostic import eigendecompose
from curvature_calib.geometry import ggn as G
from curvature_calib.geometry import validity as V

# numpy (float64) so the grid is not created float32 at import time (x64 is
# enabled per-test by the conftest fixture, after module import).
ALPHAS = np.array([-2.0, -1.5, -1.0, -0.75, -0.5, -0.3, -0.2, -0.1, -0.05, -0.02, -0.01,
                   0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0])
_ALPHA_MAX = float(np.max(np.abs(ALPHAS)))
TAU = 0.10


def test_quadratic_prediction_exact_on_quadratic_loss():
    """For a quadratic loss the GGN prediction is exact ⇒ error ~0, radius = max."""
    A = lg.random_A(jax.random.PRNGKey(0), K=6, P=3, cond=10.0)
    z_true = jax.random.normal(jax.random.PRNGKey(1), (3,))
    model = lg.LinearGaussian(A=A, y=A @ z_true, W=None)
    loss = lg.make_loss(model)
    Gm = lg.analytic_ggn(model)
    eig = eigendecompose(Gm)
    for k in range(3):
        v, lam = eig.eigvecs[:, k], float(eig.eigvals[k])
        err = np.asarray(V.quadratic_model_error(loss, z_true, v, lam, ALPHAS))
        assert np.max(err) <= 1e-9
        rad = V.validity_radius(loss, z_true, v, lam, ALPHAS, TAU)
        assert rad["rho"] == _ALPHA_MAX


def test_rosen_prediction_accurate_near_optimum_and_bounded_radius():
    model = nr.RosenbrockResidual(a=5.0, b=1.0)
    loss = nr.rosen_loss_fn(model)
    r = nr.rosen_residual_fn(model)
    z_opt = nr.rosen_optimum(model)
    eig = eigendecompose(G.ggn_dense(r, z_opt))
    v, lam = eig.eigvecs[:, 0], float(eig.eigvals[0])
    # error -> 0 as alpha -> 0
    small = float(V.quadratic_model_error(loss, z_opt, v, lam, jnp.array([0.01]))[0])
    assert small <= TAU
    rad = V.validity_radius(loss, z_opt, v, lam, ALPHAS, TAU)
    assert 0.0 < rad["rho"] <= _ALPHA_MAX


def test_rosen_validity_radius_is_local_and_direction_dependent():
    """Strong curvature (a=10) ⇒ the quadratic model is valid only locally, and
    the radius depends on the eigen-direction (stiff vs sloppy differ)."""
    model = nr.RosenbrockResidual(a=10.0, b=1.0)
    loss = nr.rosen_loss_fn(model)
    r = nr.rosen_residual_fn(model)
    z_opt = nr.rosen_optimum(model)
    eig = eigendecompose(G.ggn_dense(r, z_opt))
    radii = []
    for k in range(2):
        v, lam = eig.eigvecs[:, k], float(eig.eigvals[k])
        rad = V.validity_radius(loss, z_opt, v, lam, ALPHAS, TAU)
        assert 0.0 < rad["rho"] < _ALPHA_MAX      # finite and local
        radii.append(rad["rho"])
    assert radii[0] != radii[1]                    # direction-dependent


def test_irr_prediction_biased_by_residual_curvature():
    """At Case-B minimizer the missing R term biases the GGN quadratic prediction."""
    model = nr.IrreducibleResidual(lam=1.0, c=4.0)
    loss = nr.irr_loss_fn(model)
    r = nr.irr_residual_fn(model)
    z_hat = nr.newton_minimize(loss, jnp.array([1.0, 1.0]))
    eig = eigendecompose(G.ggn_dense(r, z_hat))
    v, lam = eig.eigvecs[:, 0], float(eig.eigvals[0])
    rad = V.validity_radius(loss, z_hat, v, lam, ALPHAS, TAU)
    # a finite radius exists but is not the full grid (bias limits it)
    assert rad["rho"] < _ALPHA_MAX


def test_curvature_bias_zero_at_exact_fit_and_matches_alpha0_error():
    """e_curv = 0 for Case A (R=0 at optimum); for Case B it is nonzero and
    approximates the α→0 quadratic-prediction relative error (residual bias)."""
    # Case A: exact fit ⇒ e_curv ≈ 0 along every eigenvector.
    ma = nr.RosenbrockResidual(a=5.0, b=1.0)
    la, ra = nr.rosen_loss_fn(ma), nr.rosen_residual_fn(ma)
    zo = nr.rosen_optimum(ma)
    Ha, Ga = G.exact_hessian(la, zo), G.ggn_dense(ra, zo)
    ea = eigendecompose(Ga)
    for k in range(2):
        assert V.curvature_bias(Ha, Ga, ea.eigvecs[:, k]) <= 1e-8

    # Case B: nonzero bias ~ constant α→0 prediction error.
    mb = nr.IrreducibleResidual(lam=0.3, c=0.0)
    lb, rb = nr.irr_loss_fn(mb), nr.irr_residual_fn(mb)
    zb = nr.newton_minimize(lb, jnp.array([1.0, 1.0]))
    Hb, Gb = G.exact_hessian(lb, zb), G.ggn_dense(rb, zb)
    eb = eigendecompose(Gb)
    v, lam = eb.eigvecs[:, 0], float(eb.eigvals[0])
    e_curv = V.curvature_bias(Hb, Gb, v)
    err_small = float(V.quadratic_model_error(lb, zb, v, lam, jnp.array([1e-3]))[0])
    assert e_curv > 1e-2
    assert abs(e_curv - err_small) <= 1e-2  # bias persists as alpha -> 0


def test_validity_radius_inputs_not_mutated():
    model = nr.RosenbrockResidual(a=5.0, b=1.0)
    loss = nr.rosen_loss_fn(model)
    r = nr.rosen_residual_fn(model)
    z_opt = nr.rosen_optimum(model)
    eig = eigendecompose(G.ggn_dense(r, z_opt))
    alphas_before = np.asarray(ALPHAS).copy()
    V.validity_radius(loss, z_opt, eig.eigvecs[:, 0], float(eig.eigvals[0]), ALPHAS, TAU)
    assert np.array_equal(np.asarray(ALPHAS), alphas_before)
