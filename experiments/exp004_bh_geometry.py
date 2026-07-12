"""EXP-004 — Smooth Brock–Hommes geometry: GGN vs Hessian, horizon dependence.

Does the MMD GGN predict the local loss geometry of the actual Brock–Hommes model
(correct large-β regime), and how does it depend on the differentiation horizon?
Supports C03, C09; characterizes RQ4. See docs/experiments/EXP-004.md.

Run:  uv run python -m experiments.exp004_bh_geometry
Writes: outputs/EXP-004/<UTC-timestamp>_<short-commit>/...
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from curvature_calib.config import enable_x64, require_x64

enable_x64()
require_x64()

import numpy as np  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from curvature_calib import metrics as MET  # noqa: E402
from curvature_calib import provenance as prov  # noqa: E402
from curvature_calib.calibration import diagnostic as DIAG  # noqa: E402
from curvature_calib.geometry import ggn as GGN  # noqa: E402
from curvature_calib.geometry import rff as R  # noqa: E402
from curvature_calib.geometry import validity as V  # noqa: E402
from curvature_calib.calibration.per_seed_grads import vmap_simulate  # noqa: E402
from curvature_calib.losses.mmd import median_heuristic  # noqa: E402
from curvature_calib.models.brock_hommes import simulate  # noqa: E402

_COMMAND = "uv run python -m experiments.exp004_bh_geometry"

T, R_GR, SIGMA = 100, 1.01, 0.04
S_PRIOR = jnp.array([0.30, 0.10, 0.05, 0.10, 0.05])
POINTS = {"P1_complex": (50.0, 0.9, 0.2, 0.9, -0.2),
          "P2_chaotic": (80.0, 0.9, 0.2, 0.9, -0.2)}
HORIZONS = [1, 5, 10, 20, 50, None]      # None = full horizon
WORKING_HORIZON = 20
D_FEAT, FEATURE_SEED, M = 256, 7, 128
DELTAS = (0.0, 0.3, 0.8)
# fine, log-spaced (both signs) — the chaotic loss is extremely non-quadratic, so
# the predictive radius can be very small and needs sub-0.01 resolution.
_A = np.array([1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0])
ALPHAS = np.concatenate([-_A[::-1], _A])
TAU = 0.10
Z0 = jnp.zeros(5)
_RTOL = 1e-8


def _sim(theta0, grad_horizon):
    return lambda z, key: simulate(jnp.asarray(theta0) + S_PRIOR * z, key,
                                   T=T, R=R_GR, sigma=SIGMA, grad_horizon=grad_horizon)


def _eta_hat(theta0, rff, keys, grad_horizon):
    sim = _sim(theta0, grad_horizon)
    return lambda z: jnp.mean(R.feature_map(rff, jax.vmap(lambda k: sim(z, k))(keys)), axis=0)


def _G(eta_hat_fn, z):
    J = jax.jacfwd(eta_hat_fn)(z)                 # (D, P)
    return jnp.asarray(GGN.symmetrize(J.T @ J)), J


def _spectrum(Gm):
    ev = np.sort(np.asarray(jnp.linalg.eigvalsh(Gm)))[::-1]
    return {"eigvals": ev.tolist(),
            "effective_rank": int((ev > _RTOL * ev[0]).sum()),
            "cond": float(ev[0] / max(ev[-1], 1e-300))}


def _point(name, theta0):
    keys = jax.random.split(jax.random.PRNGKey(0), M)
    keys_ref = jax.random.split(jax.random.PRNGKey(1), M)
    Xp = vmap_simulate(_sim(theta0, WORKING_HORIZON), Z0, keys)
    gamma = float(median_heuristic(Xp))
    rff = R.frozen_rff(jax.random.PRNGKey(FEATURE_SEED), D_FEAT, T, gamma)
    ac1 = float(np.mean([np.corrcoef(np.asarray(x[:-1]), np.asarray(x[1:]))[0, 1] for x in Xp]))
    traj_std = float(np.std(np.asarray(Xp)))

    # self-consistent reference (z* is the optimum, residual 0). Forward η̂ is
    # horizon-independent, so use the working horizon to define the loss/target.
    eta_ref = _eta_hat(theta0, rff, keys, WORKING_HORIZON)
    eta_star = eta_ref(Z0)
    loss = lambda z: 0.5 * jnp.sum((eta_ref(z) - eta_star) ** 2)

    # Part A+B: horizon sweep. Full horizon (residual 0) = the TRUE forward Hessian;
    # truncated horizons are biased versions of it. Report the distortion + the
    # (fine-grid) predictive validity radius of each horizon's leading quadratic.
    Gs, eigs, Jn = {}, {}, {}
    for h in HORIZONS:
        Gh, Jh = _G(_eta_hat(theta0, rff, keys, h), Z0)
        Gs[str(h)] = Gh; eigs[str(h)] = DIAG.eigendecompose(Gh); Jn[str(h)] = float(jnp.linalg.norm(Jh))
    G_full, eig_full = Gs["None"], eigs["None"]
    lam1_full, v1_full = float(eig_full.eigvals[0]), eig_full.eigvecs[:, :1]
    horizon_rows = {}
    for h in HORIZONS:
        Gh, eig = Gs[str(h)], eigs[str(h)]
        v1, lam1 = eig.eigvecs[:, 0], float(eig.eigvals[0])
        rad = V.validity_radius(loss, Z0, v1, lam1, jnp.asarray(ALPHAS), TAU)
        horizon_rows[str(h)] = {
            "Jeta_norm": Jn[str(h)],
            "spectrum": _spectrum(Gh),
            "lead_eigval": lam1,
            "lead_eigval_ratio_vs_full": lam1 / max(lam1_full, 1e-300),
            "lead_angle_vs_full_deg": float(np.degrees(
                jnp.max(DIAG.principal_angles(eig.eigvecs[:, :1], v1_full)))),
            "rel_fro_vs_full": MET.rel_frobenius_error(Gh, G_full),
            "validity_radius": rad,
        }

    # Part C: H vs G at the working horizon, residual sweep along stiff dir.
    eh = _eta_hat(theta0, rff, keys, WORKING_HORIZON)
    G0, _ = _G(eh, Z0)
    v_stiff = DIAG.eigendecompose(G0).eigvecs[:, 0]
    hvg = {}
    for delta in DELTAS:
        z_y = Z0 + delta * v_stiff
        eta_y = eh(z_y)
        L = lambda z: 0.5 * jnp.sum((eh(z) - eta_y) ** 2)
        H = jnp.asarray(GGN.symmetrize(jax.hessian(L)(Z0)))
        Rres = H - G0
        eg, eh_ = DIAG.eigendecompose(G0), DIAG.eigendecompose(H)
        hvg[f"delta_{delta}"] = {
            "residual_norm": float(jnp.linalg.norm(eta_star - eta_y)),
            "R_over_H": float(jnp.linalg.norm(Rres)) / max(float(jnp.linalg.norm(H)), 1e-300),
            "top1_angle_H_G": float(jnp.max(DIAG.principal_angles(eg.eigvecs[:, :1], eh_.eigvecs[:, :1]))),
            "top2_angle_H_G": float(jnp.max(DIAG.principal_angles(eg.eigvecs[:, :2], eh_.eigvecs[:, :2]))),
            "e_curv_leadG": V.curvature_bias(H, G0, eg.eigvecs[:, 0]),
            "e_curv_weakG": V.curvature_bias(H, G0, eg.eigvecs[:, -1]),
        }

    return {
        "theta_physical": list(theta0), "gamma": gamma, "ac1": ac1, "traj_std": traj_std,
        "horizon_sweep": horizon_rows, "H_vs_G": hvg,
        "working_horizon": WORKING_HORIZON,
    }, {f"{name}_G_working": np.asarray(G0)}


def _new_run_dir(out_root):
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    d = Path(out_root) / f"{ts}_{prov.short_commit()}"
    (d / "figures").mkdir(parents=True, exist_ok=True)
    return d


def _figure(metrics, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.3))
    for nm, p in metrics["points"].items():
        hs = [h for h in ["1", "5", "10", "20", "50", "None"]]
        jn = [p["horizon_sweep"][h]["Jeta_norm"] for h in hs]
        rr = [p["horizon_sweep"][h]["validity_radius"]["rho"] for h in hs]
        xs = list(range(len(hs)))
        ax[0].semilogy(xs, jn, "o-", label=nm)
        ax[1].plot(xs, rr, "s-", label=nm)
    for a, t, yl in [(ax[0], "Jacobian norm vs horizon (explosion)", "‖J_η‖"),
                     (ax[1], "GGN validity radius vs horizon", "ρ")]:
        a.set_xticks(range(6)); a.set_xticklabels(["1", "5", "10", "20", "50", "full"])
        a.set_xlabel("grad_horizon"); a.set_ylabel(yl); a.set_title(t, fontsize=10); a.legend(fontsize=8)
    fig.suptitle("EXP-004 — BH geometry vs differentiation horizon", fontsize=11)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=140, bbox_inches="tight")
    plt.close(fig)


def run(out_root="outputs/EXP-004") -> str:
    require_x64()
    run_dir = _new_run_dir(out_root)
    points, arrays = {}, {}
    for name, th in POINTS.items():
        p, arr = _point(name, th)
        points[name] = p
        arrays.update(arr)
    metrics = {"points": points}
    config = {
        "T": T, "R": R_GR, "sigma": SIGMA, "points_physical": {k: list(v) for k, v in POINTS.items()},
        "horizons": [str(h) for h in HORIZONS], "working_horizon": WORKING_HORIZON,
        "D_feat": D_FEAT, "feature_seed": FEATURE_SEED, "M": M,
        "deltas": list(DELTAS), "alphas": ALPHAS.tolist(), "tau": TAU,
    }
    prov.write_json(run_dir / "config.json", config)
    prov.write_json(run_dir / "provenance.json",
                    prov.run_metadata("EXP-004", config, {"sim": 0, "ref": 1}, command=_COMMAND))
    prov.write_json(run_dir / "metrics.json", metrics)
    np.savez_compressed(run_dir / "arrays.npz", **arrays)
    _figure(metrics, run_dir / "figures" / "fig04_bh_geometry")
    return str(run_dir)


if __name__ == "__main__":
    print("run:", run())
