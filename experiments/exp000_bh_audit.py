"""EXP-000 — Brock–Hommes historical-OPG audit.

Compares, at matched points in prior-scaled z and with common random numbers:
    H_exact = ∇²_z L_feat,  G_MMD = J_η^T J_η (plug-in),
    F_OPG  = (1/M)Σ g_m g_mᵀ,  C_g = (1/M)Σ (g_m−ḡ)(g_m−ḡ)ᵀ,
to determine what the historical OPG measured. See EXP000_IMPLEMENTATION_PLAN.md.

Run:  uv run python -m experiments.exp000_bh_audit
Writes: outputs/EXP-000/<UTC-timestamp>_<short-commit>/...
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
from curvature_calib.calibration.per_seed_grads import per_seed_loss_and_grads, vmap_simulate  # noqa: E402
from curvature_calib.geometry import mmd_estimators as E  # noqa: E402
from curvature_calib.geometry import rff as R  # noqa: E402
from curvature_calib.geometry import validity as V  # noqa: E402
from curvature_calib.losses.mmd import median_heuristic  # noqa: E402
from curvature_calib.models.brock_hommes import simulate  # noqa: E402

_COMMAND = "uv run python -m experiments.exp000_bh_audit"

# Canonical BH regime (R=1.01, sigma=0.04). LARGE beta => rich/chaotic dynamics
# (verified: beta drives the route to chaos; see docs/papers/reference_brock_hommes_1998.md).
T, R_GR, SIGMA = 100, 1.01, 0.04
# Gradient-horizon truncation is REQUIRED in the chaotic regime: full-horizon
# pathwise gradients explode (sensitive dependence). We use ONE fixed horizon for
# all four matrices so they are comparable; the full-horizon explosion is reported
# as a finding. (Quera-Bofarull 2023 use H=0 for calibration; we use a small
# horizon that keeps the geometry finite and multi-directional.)
GRAD_HORIZON = 20
S_PRIOR = jnp.array([0.30, 0.10, 0.05, 0.10, 0.05])   # documented prior scales
POINTS = {
    "P1_complex": (50.0, 0.9, 0.2, 0.9, -0.2),   # strong endogenous dynamics
    "P2_chaotic": (80.0, 0.9, 0.2, 0.9, -0.2),   # chaotic (std ~12x noise)
}
D_FEAT, FEATURE_SEED = 256, 7
M_GRID = (32, 128)
B = 15
DELTAS = (0.0, 0.5, 1.5)   # target offset ALONG the leading-G (stiff) direction
ALPHAS = np.array([-1.0, -0.5, -0.2, -0.1, -0.05, 0.05, 0.1, 0.2, 0.5, 1.0])
Z0 = jnp.zeros(5)
_RTOL_RANK = 1e-8


def _theta(theta0, z):
    return jnp.asarray(theta0) + S_PRIOR * z


def _sim_z(theta0, grad_horizon=GRAD_HORIZON):
    return lambda z, key: simulate(_theta(theta0, z), key, T=T, R=R_GR, sigma=SIGMA,
                                   grad_horizon=grad_horizon)


def _frozen_features(theta0, keys):
    Xp = vmap_simulate(_sim_z(theta0), Z0, keys)
    gamma = float(median_heuristic(Xp))
    rff = R.frozen_rff(jax.random.PRNGKey(FEATURE_SEED), D_FEAT, T, gamma)
    return rff, gamma


def _eta_hat_fn(theta0, rff, keys, grad_horizon=GRAD_HORIZON):
    sim = _sim_z(theta0, grad_horizon)
    def eta_hat(z):
        X = jax.vmap(lambda k: sim(z, k))(keys)          # (M, T)
        return jnp.mean(R.feature_map(rff, X), axis=0)   # (D,)
    return eta_hat


def _matrix_summary(name, Gm):
    ev = np.sort(np.asarray(jnp.linalg.eigvalsh(0.5 * (Gm + Gm.T))))[::-1]
    return {
        "eigvals": ev.tolist(),
        "trace": float(jnp.trace(Gm)),
        "spectral_norm": float(np.abs(ev).max()),
        "numerical_rank": int((np.abs(ev) > _RTOL_RANK * np.abs(ev).max()).sum()),
    }


def _pair(Gi, Gj):
    ei, ej = DIAG.eigendecompose(Gi), DIAG.eigendecompose(Gj)
    def ang(k):
        return float(jnp.max(DIAG.principal_angles(ei.eigvecs[:, :k], ej.eigvecs[:, :k])))
    return {
        "rel_fro": MET.rel_frobenius_error(Gi, Gj),
        "rel_fro_tracenorm": MET.rel_frobenius_error(E.trace_normalized(Gi), E.trace_normalized(Gj)),
        "top1_angle": ang(1),
        "top2_angle": ang(2),
    }


def _point_audit(name, theta0):
    keys = jax.random.split(jax.random.PRNGKey(0), max(M_GRID))
    keys_ref = jax.random.split(jax.random.PRNGKey(1), max(M_GRID))
    rff, gamma = _frozen_features(theta0, keys)
    eta_hat = _eta_hat_fn(theta0, rff, keys)

    # true representation geometry (residual-independent G; H depends on target)
    Jeta = jax.jacfwd(eta_hat)(Z0)                       # (D, P), horizon-truncated
    G = jnp.asarray(0.5 * (Jeta.T @ Jeta + (Jeta.T @ Jeta).T))
    eta0 = eta_hat(Z0)
    v_stiff = DIAG.eigendecompose(G).eigvecs[:, 0]       # move target along stiff dir

    # dynamics descriptor
    Xp = vmap_simulate(_sim_z(theta0), Z0, keys)
    ac1 = float(np.mean([np.corrcoef(np.asarray(x[:-1]), np.asarray(x[1:]))[0, 1] for x in Xp]))
    traj_std = float(np.std(np.asarray(Xp)))

    # full-horizon explosion diagnostic: the "true" pathwise GGN is numerically
    # inaccessible in the chaotic regime (sensitive dependence). Compare the
    # feature-Jacobian norm and G condition number at full horizon vs truncated.
    eta_full = _eta_hat_fn(theta0, rff, keys, grad_horizon=None)
    J_full = jax.jacfwd(eta_full)(Z0)
    ev_t = np.sort(np.asarray(jnp.linalg.eigvalsh(G)))[::-1]
    horizon_explosion = {
        "grad_horizon": GRAD_HORIZON,
        "Jeta_norm_truncated": float(jnp.linalg.norm(Jeta)),
        "Jeta_norm_full_horizon": float(jnp.linalg.norm(J_full)),
        "explosion_ratio": float(jnp.linalg.norm(J_full)) / max(float(jnp.linalg.norm(Jeta)), 1e-300),
        "G_cond_truncated": float(ev_t[0] / max(ev_t[-1], 1e-300)),
        "G_effective_rank_truncated": int((ev_t > 1e-6 * ev_t[0]).sum()),
    }

    per_delta = {}
    for delta in DELTAS:
        z_y = Z0 + delta * v_stiff                        # residual grows along stiff dir
        eta_y = eta_hat(z_y)
        L_feat = lambda z: 0.5 * jnp.sum((eta_hat(z) - eta_y) ** 2)
        H = jax.hessian(L_feat)(Z0)
        H = jnp.asarray(0.5 * (H + H.T))
        Rres = H - G
        Y_ref = vmap_simulate(_sim_z(theta0), z_y, keys_ref)   # historical MMD reference sample

        # historical F_OPG, C_g across M and B batches (U-statistic MMD² per-seed grads)
        opg_by_M = {}
        for Ms in M_GRID:
            Fs, Cs, angFG, angFC = [], [], [], []
            for b in range(B):
                kk = jax.random.split(jax.random.PRNGKey(100 * b + 3), Ms)
                stats = per_seed_loss_and_grads(_sim_z(theta0), Z0, kk, Y_ref[:Ms])
                F = stats.opg
                Cg = DIAG.scalar_gradient_covariance(stats.per_seed_grads)
                Fs.append(np.asarray(F)); Cs.append(np.asarray(Cg))
                if float(jnp.linalg.norm(F)) > 1e-12:
                    angFG.append(_pair(F, G)["top1_angle"])
                    angFC.append(_pair(F, Cg)["top1_angle"])
            meanF = jnp.asarray(np.mean(Fs, 0)); meanC = jnp.asarray(np.mean(Cs, 0))
            opg_by_M[str(Ms)] = {
                "meanF_fro": float(jnp.linalg.norm(meanF)),
                "meanC_fro": float(jnp.linalg.norm(meanC)),
                "F_scale_over_G_scale": float(jnp.linalg.norm(meanF)) / max(float(jnp.linalg.norm(G)), 1e-300),
                "F_vs_G": _pair(meanF, G) if float(jnp.linalg.norm(meanF)) > 1e-12 else None,
                "F_vs_Cg": _pair(meanF, meanC) if float(jnp.linalg.norm(meanF)) > 1e-12 else None,
                "angle_F_G_mean_std": [float(np.mean(angFG)), float(np.std(angFG))] if angFG else None,
                "angle_F_Cg_mean_std": [float(np.mean(angFC)), float(np.std(angFC))] if angFC else None,
            }

        entry = {
            "residual_norm": float(jnp.linalg.norm(eta0 - eta_y)),
            "loss_feat": float(0.5 * jnp.sum((eta0 - eta_y) ** 2)),
            "H_summary": _matrix_summary("H", H),
            "R_over_H": float(jnp.linalg.norm(Rres)) / max(float(jnp.linalg.norm(H)), 1e-300),
            "H_vs_G": _pair(H, G),
            "e_curv_G_dirs": [V.curvature_bias(H, G, DIAG.eigendecompose(G).eigvecs[:, k])
                              for k in range(5)],
            "opg_by_M": opg_by_M,
        }
        per_delta[f"delta_{delta}"] = entry

    # local predictive test at moderate residual (delta=0.5 along stiff dir), M=128
    z_y = Z0 + 0.5 * v_stiff; eta_y = eta_hat(z_y)
    L_feat = lambda z: 0.5 * jnp.sum((eta_hat(z) - eta_y) ** 2)
    eigG = DIAG.eigendecompose(G)
    kk = jax.random.split(jax.random.PRNGKey(303), 128)
    st = per_seed_loss_and_grads(_sim_z(theta0), Z0, kk,
                                 vmap_simulate(_sim_z(theta0), z_y, keys_ref[:128]))
    eigF = DIAG.eigendecompose(st.opg)
    eigC = DIAG.eigendecompose(DIAG.scalar_gradient_covariance(st.per_seed_grads))
    dirs = {"G_leading": eigG.eigvecs[:, 0], "G_weakest": eigG.eigvecs[:, -1],
            "OPG_leading": eigF.eigvecs[:, 0], "Cg_leading": eigC.eigvecs[:, 0]}
    predictive = {}
    for dname, v in dirs.items():
        lam = float(v @ G @ v)
        dL = np.asarray(V.directional_loss_change(L_feat, Z0, v, jnp.asarray(ALPHAS)))
        Q = 0.5 * (ALPHAS ** 2) * lam
        predictive[dname] = {"lambda_G": lam, "alphas": ALPHAS.tolist(),
                             "dL": dL.tolist(), "quad_pred": Q.tolist()}

    # outcome classification from the largest-residual cell, M=128.
    # G is now multi-directional (rank ~4 in the rich regime), so we compare the
    # top-2 SUBSPACE of the mean F_OPG against G (curvature) vs C_g (gradient
    # covariance). Nearest wins; neither within threshold => C.
    ref = per_delta[f"delta_{DELTAS[-1]}"]["opg_by_M"]["128"]
    aFG = ref["F_vs_G"]["top2_angle"] if ref["F_vs_G"] else None
    aFC = ref["F_vs_Cg"]["top2_angle"] if ref["F_vs_Cg"] else None
    rF_Cg = ref["F_vs_Cg"]["rel_fro"] if ref["F_vs_Cg"] else None
    scale = ref["F_scale_over_G_scale"]
    thr = np.radians(20)
    if aFG is not None and aFC is not None:
        if aFG < thr and aFG <= aFC:
            outcome = "A_OPG~GGN (top-2 subspace)"
        elif aFC < thr and aFC < aFG:
            outcome = "B_OPG~grad_cov (top-2 subspace)"
        else:
            outcome = "C_neither"
    else:
        outcome = "undetermined"
    outcome_detail = {"top2_angle_F_G_deg": float(np.degrees(aFG)) if aFG is not None else None,
                      "top2_angle_F_Cg_deg": float(np.degrees(aFC)) if aFC is not None else None,
                      "rel_fro_F_vs_Cg": rF_Cg, "F_scale_over_G_scale": scale}

    return {
        "theta_physical": list(theta0), "z_star": [0.0] * 5, "gamma": gamma,
        "ac1": ac1, "traj_std": traj_std, "horizon_explosion": horizon_explosion,
        "G_summary": _matrix_summary("G", G),
        "per_delta": per_delta, "predictive": predictive,
        "outcome": outcome, "outcome_detail": outcome_detail,
    }, {f"{name}_G": np.asarray(G)}


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
    names = list(metrics["points"].keys())
    for nm in names:
        p = metrics["points"][nm]
        deltas = [float(k.split("_")[1]) for k in p["per_delta"]]
        aFG = [p["per_delta"][k]["opg_by_M"]["128"]["angle_F_G_mean_std"][0]
               if p["per_delta"][k]["opg_by_M"]["128"]["angle_F_G_mean_std"] else np.nan
               for k in p["per_delta"]]
        aFC = [p["per_delta"][k]["opg_by_M"]["128"]["angle_F_Cg_mean_std"][0]
               if p["per_delta"][k]["opg_by_M"]["128"]["angle_F_Cg_mean_std"] else np.nan
               for k in p["per_delta"]]
        ax[0].plot(deltas, np.degrees(aFG), "o-", label=f"{nm} F–G")
        ax[1].plot(deltas, np.degrees(aFC), "s--", label=f"{nm} F–Cg")
    for a, t in zip(ax, ["angle(F_OPG, G) vs residual", "angle(F_OPG, C_g) vs residual"]):
        a.set_xlabel("δ (residual)"); a.set_ylabel("top-1 angle (deg)"); a.set_title(t, fontsize=10)
        a.legend(fontsize=7)
    fig.suptitle("EXP-000 — historical BH OPG audit", fontsize=11)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=140, bbox_inches="tight")
    plt.close(fig)


def run(out_root="outputs/EXP-000") -> str:
    require_x64()
    run_dir = _new_run_dir(out_root)
    points, arrays = {}, {}
    for name, th in POINTS.items():
        p, arr = _point_audit(name, th)
        points[name] = p
        arrays.update(arr)
    metrics = {"points": points,
               "outcomes": {n: points[n]["outcome"] for n in points}}
    config = {
        "T": T, "R": R_GR, "sigma": SIGMA, "grad_horizon": GRAD_HORIZON,
        "prior_scales": S_PRIOR.tolist(),
        "points_physical": {k: list(v) for k, v in POINTS.items()},
        "D_feat": D_FEAT, "feature_seed": FEATURE_SEED, "M_grid": list(M_GRID),
        "B": B, "deltas": list(DELTAS), "alphas": ALPHAS.tolist(),
        "regime_note": "LARGE-beta rich/chaotic BH regime (beta=50,80; R=1.01, "
                       "sigma=0.04). Full-horizon gradients explode; a fixed "
                       "grad_horizon is used for all four matrices (see "
                       "horizon_explosion per point).",
    }
    prov.write_json(run_dir / "config.json", config)
    prov.write_json(run_dir / "provenance.json",
                    prov.run_metadata("EXP-000", config, {"sim": 0, "ref": 1}, command=_COMMAND))
    prov.write_json(run_dir / "metrics.json", metrics)
    np.savez_compressed(run_dir / "arrays.npz", **arrays)
    _figure(metrics, run_dir / "figures" / "fig00_bh_opg_audit")
    return str(run_dir)


if __name__ == "__main__":
    print("run:", run())
