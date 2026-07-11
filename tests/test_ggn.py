"""EXP-001 core acceptance harness: AD GGN vs analytic on the linear benchmark.

float64 is enabled per-test by the autouse fixture in tests/conftest.py (scoped
to this module) and restored afterwards, so it never leaks into the float32
historical tests.
"""
import jax
import jax.numpy as jnp
import pytest

from curvature_calib import metrics as M
from curvature_calib.benchmarks import linear_gaussian as lg
from curvature_calib.calibration.diagnostic import eigendecompose
from curvature_calib.geometry import ggn as G

AD_TOL = 1e-10
ANGLE_TOL = 1e-6


def _weight(kind, K, dtype, key):
    if kind == "none":
        return None
    if kind == "diag":
        return jnp.linspace(0.5, 2.0, K, dtype=dtype)
    B = jax.random.normal(key, (K, K), dtype=dtype)
    return (B @ B.T) / K + jnp.eye(K, dtype=dtype)


def _build(P, cond, weight_kind, dtype=jnp.float64, rank=None, seed=0):
    K = 2 * P
    key = jax.random.PRNGKey(seed)
    kA, kz, kw = jax.random.split(key, 3)
    A = lg.random_A(kA, K, P, cond=cond, rank=rank, dtype=dtype)
    W = _weight(weight_kind, K, dtype, kw)
    z_true = jax.random.normal(kz, (P,), dtype=dtype)
    model = lg.LinearGaussian(A=A, y=A @ z_true, W=W)
    return model, z_true, kA


@pytest.mark.parametrize("P", [5, 20, 100])
@pytest.mark.parametrize("cond", [1.0, 1e2, 1e6])
@pytest.mark.parametrize("weight_kind", ["none", "diag", "dense"])
def test_ad_ggn_equals_analytic(P, cond, weight_kind):
    model, z, _ = _build(P, cond, weight_kind)
    rep = lg.make_representation(model)
    G_ad = G.ggn_dense(rep, z, model.W)
    err = M.rel_frobenius_error(G_ad, lg.analytic_ggn(model))
    assert err <= AD_TOL, f"P={P} cond={cond:.0e} W={weight_kind}: rel_fro={err:.2e}"


@pytest.mark.parametrize("P", [5, 20, 100])
@pytest.mark.parametrize("weight_kind", ["none", "diag", "dense"])
def test_matvec_equals_dense(P, weight_kind):
    model, z, _ = _build(P, 1e2, weight_kind)
    rep = lg.make_representation(model)
    G_ad = G.ggn_dense(rep, z, model.W)
    vecs = [jnp.eye(P)[i] for i in (0, P // 2, P - 1)]
    vecs += [jnp.ones((P,)), jax.random.normal(jax.random.PRNGKey(1), (P,))]
    for v in vecs:
        err = M.rel_l2_error(G.ggn_matvec(rep, z, v, model.W), G_ad @ v)
        assert err <= AD_TOL, f"P={P} W={weight_kind}: matvec rel_l2={err:.2e}"


@pytest.mark.parametrize("P", [5, 20])
@pytest.mark.parametrize("weight_kind", ["none", "diag", "dense"])
def test_affine_hessian_equals_ggn(P, weight_kind):
    """R = H - G = 0 for affine m."""
    model, z, _ = _build(P, 1e2, weight_kind)
    rep = lg.make_representation(model)
    loss = lg.make_loss(model)
    G_ad = G.ggn_dense(rep, z, model.W)
    H = G.exact_hessian(loss, z)
    # R = H - G; rel_frobenius_error(H, G) = ||H - G|| / ||G|| = ||R|| / ||G||.
    r_err = M.rel_frobenius_error(H, G_ad)
    assert r_err <= AD_TOL, f"P={P} W={weight_kind}: ||R||/||G||={r_err:.2e}"


def test_opg_zero_at_fit_and_rank1_off_fit():
    """At r=0: OPG=0 while GGN!=0. Off fit: OPG is rank 1 (DEC-001)."""
    model, z_hat, _ = _build(8, 1e2, "none")
    rep = lg.make_representation(model)
    loss = lg.make_loss(model)
    G_ad = G.ggn_dense(rep, z_hat, model.W)
    opg_fit = G.scalar_gradient_outer_product(loss, z_hat)
    assert float(jnp.linalg.norm(opg_fit)) <= 1e-8 * float(jnp.linalg.norm(G_ad))
    assert float(jnp.linalg.norm(G_ad)) > 1.0  # GGN nonzero

    z_off = z_hat + 0.3 * jax.random.normal(jax.random.PRNGKey(3), z_hat.shape)
    opg_off = G.scalar_gradient_outer_product(loss, z_off)
    w = jnp.linalg.eigvalsh(G.symmetrize(opg_off))
    assert M.numerical_rank(w) == 1
    assert M.numerical_rank(eigendecompose(G_ad).eigvals) == z_hat.shape[0]  # full rank


def test_rank_and_nullspace_recovery():
    P, rank = 10, 5
    model, z, kA = _build(P, 1e2, "none", rank=rank)
    rep = lg.make_representation(model)
    G_ad = G.ggn_dense(rep, z, model.W)
    eig = eigendecompose(G_ad)
    assert M.numerical_rank(eig.eigvals) == rank
    V = lg.right_singular_vectors(kA, 2 * P, P)
    null_true = V[:, rank:]
    null_rec = eig.eigvecs[:, rank:]
    assert M.max_principal_angle(null_rec, null_true) <= ANGLE_TOL


@pytest.mark.parametrize("P", [5, 20])
def test_eigvec_agreement(P):
    model, z, _ = _build(P, 1e2, "none")
    rep = lg.make_representation(model)
    G_ad = G.ggn_dense(rep, z, model.W)
    ang = M.max_principal_angle(
        eigendecompose(G_ad).eigvecs, eigendecompose(lg.analytic_ggn(model)).eigvecs
    )
    assert ang <= ANGLE_TOL


def _fd_errors(P, cond, steps, seed=0):
    model, z, _ = _build(P, cond, "none", seed=seed)
    loss = lg.make_loss(model)
    G_an = lg.analytic_ggn(model)
    return [M.rel_frobenius_error(G.finite_difference_hessian(loss, z, h), G_an) for h in steps]


def _fd_min_eig_relerr(P, cond, h, seed=0):
    """Relative error of the SMALLEST eigenvalue of the FD Hessian — the tail
    that ill-conditioning corrupts (round-off floor ~ eps * lambda_max)."""
    model, z, _ = _build(P, cond, "none", seed=seed)
    loss = lg.make_loss(model)
    w_an = jnp.linalg.eigvalsh(lg.analytic_ggn(model))
    w_fd = jnp.linalg.eigvalsh(G.symmetrize(G.finite_difference_hessian(loss, z, h)))
    return abs(float(w_fd[0] - w_an[0])) / abs(float(w_an[0]))


def test_fd_step_and_conditioning_sensitivity():
    """FD has its OWN looser tolerance and varies with step size AND conditioning.

    For a quadratic loss the central difference has no truncation error, so the
    Frobenius error is round-off-limited: it GROWS as the step shrinks (not
    U-shaped). Conditioning corrupts the small-eigenvalue tail, not ||.||_F.
    """
    steps = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    errs = _fd_errors(5, 1.0, steps)
    assert min(errs) <= 1e-5                       # separate, looser regime than AD
    assert errs[-1] > errs[0]                       # round-off grows as step -> 0
    # conditioning sensitivity: smallest-eigenvalue error explodes with cond(G).
    lo_well = _fd_min_eig_relerr(20, 1.0, 1e-3)
    lo_ill = _fd_min_eig_relerr(20, 1e12, 1e-3)
    assert lo_ill > 1e3 * lo_well


def test_float32_recorded_not_gated():
    """float32 degrades; assert only a loose sanity bound, never the AD gate."""
    model, z, _ = _build(20, 1e2, "none", dtype=jnp.float32)
    rep = lg.make_representation(model)
    err = M.rel_frobenius_error(G.ggn_dense(rep, z, model.W), lg.analytic_ggn(model))
    assert err <= 1e-3  # loose; documents float32, does not define acceptance
