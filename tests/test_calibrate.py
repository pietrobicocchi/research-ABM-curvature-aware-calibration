"""End-to-end calibration sanity checks on Brock-Hommes."""

import jax
import jax.numpy as jnp
import numpy as np

from curvature_calib.calibration.calibrate import calibrate
from curvature_calib.calibration.per_seed_grads import vmap_simulate
from curvature_calib.calibration.preconditioner import damped_step, update_damping
from curvature_calib.models.brock_hommes import simulate


T = 150
SIGMA = 0.05
R = 1.1

THETA_STAR = jnp.array([3.0, 1.2, 0.2, 1.2, -0.2])


def _sim(theta, key):
    return simulate(theta, key, T=T, sigma=SIGMA, R=R, x_init=0.0)


def _make_ref(M_ref=128, seed=0):
    keys = jax.random.split(jax.random.PRNGKey(seed), M_ref)
    return vmap_simulate(_sim, THETA_STAR, keys)


def test_damped_step_pure_gradient_when_F_is_zero():
    g = jnp.array([1.0, -2.0, 3.0])
    F = jnp.zeros((3, 3))
    step = damped_step(F, g, damping=1.0)
    assert jnp.allclose(step, -g)


def test_damped_step_handles_psd_F():
    F = jnp.diag(jnp.array([10.0, 1.0, 0.1]))
    g = jnp.array([1.0, 1.0, 1.0])
    step = damped_step(F, g, damping=0.0)
    # Should rescale: step ~ -F^{-1} g = (-0.1, -1, -10)
    assert jnp.allclose(step, jnp.array([-0.1, -1.0, -10.0]), atol=1e-4)


def test_update_damping_monotonic_in_ratio():
    d0 = 1.0
    d_good = update_damping(d0, realised_reduction=1.0, predicted_reduction=1.0)
    d_bad = update_damping(d0, realised_reduction=0.05, predicted_reduction=1.0)
    assert d_good < d0      # accurate model -> decrease
    assert d_bad > d0       # bad model -> increase


def test_calibration_reduces_loss():
    """The optimizer must reduce MMD^2 from a perturbed initialization."""
    Y_ref = _make_ref()
    theta0 = THETA_STAR + jnp.array([0.0, 0.05, 0.03, 0.05, -0.03])
    log = calibrate(_sim, theta0, Y_ref, M=32, n_iter=30, verbose=False)
    L0 = log.losses[0]
    L_end = log.losses[-1]
    assert L_end < L0, f"loss did not decrease: L0={L0:.4e} L_end={L_end:.4e}"
    # And the reduction should be meaningful.
    assert L_end < 0.5 * L0 or L_end < 1e-3


def test_calibration_recovers_well_constrained_params():
    """After calibration, theta should be closer to truth than at init."""
    Y_ref = _make_ref()
    theta0 = THETA_STAR + jnp.array([0.0, 0.05, 0.03, 0.05, -0.03])
    log = calibrate(_sim, theta0, Y_ref, M=32, n_iter=60, verbose=False)
    err0 = float(jnp.linalg.norm(theta0 - THETA_STAR))
    err_end = float(jnp.linalg.norm(log.thetas[-1] - THETA_STAR))
    assert err_end < err0, f"err did not improve: err0={err0:.4f} err_end={err_end:.4f}"


def test_log_shapes():
    Y_ref = _make_ref()
    theta0 = THETA_STAR + 0.02
    log = calibrate(_sim, theta0, Y_ref, M=16, n_iter=5, verbose=False)
    arrs = log.as_arrays()
    P = THETA_STAR.shape[0]
    assert arrs["thetas"].shape == (6, P)
    assert arrs["losses"].shape == (6,)
    assert arrs["opgs"].shape == (5, P, P)
    assert arrs["eigvals"].shape == (5, P)
    assert arrs["eigvecs"].shape == (5, P, P)
    assert arrs["per_seed_grads"].shape == (5, 16, P)
