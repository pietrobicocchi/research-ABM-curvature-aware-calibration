"""EXP-003 — Stochastic MMD GGN estimation.

Validates estimators of G_ref = J_η^T J_η for a controlled Gaussian
location-scale simulator with frozen RFF features, against the analytic
closed-form reference. Compares PSD plug-in Ĝ_V, cross-seed Ĝ_U, and two labeled
scalar-OPG comparators. Supports C04, C05, C06.

Run:  uv run python -m experiments.exp003_mmd_ggn
Writes: outputs/EXP-003/<UTC-timestamp>_<short-commit>/{config,provenance,metrics}.json,
        arrays.npz, figures/fig03_mmd_ggn.{png,pdf}
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
from curvature_calib.benchmarks import gaussian_location_scale as gls  # noqa: E402
from curvature_calib.calibration.diagnostic import eigendecompose  # noqa: E402
from curvature_calib.geometry import mmd_estimators as E  # noqa: E402
from curvature_calib.geometry import rff as R  # noqa: E402

_COMMAND = "uv run python -m experiments.exp003_mmd_ggn"

MAIN_CFG = gls.GLSConfig(m0=0.0, sigma_m=1.0, rho0=0.0, sigma_s=0.30, sigma_l=0.30)
NEARDEG_CFG = gls.GLSConfig(m0=0.0, sigma_m=0.5, rho0=0.0, sigma_s=1.0, sigma_l=1.0)
GAMMA = 2.0
D_MAIN = 512
M_GRID = (8, 16, 32, 64, 128, 256)
B = 200
FEATURE_SEED = 0
Z_STAR = jnp.zeros(5)
DELTAS = (0.0, 0.2, 0.6)
D_SWEEP = (64, 256, 1024)
_TOL_NEG = 1e-10


def _regime(cfg, feature_seed=FEATURE_SEED, D=D_MAIN):
    r = R.frozen_rff(jax.random.PRNGKey(feature_seed), D, 2, GAMMA)
    fz = lambda zz, e: R.feature_of_z(cfg, r, zz, e)
    cA = jax.jit(lambda eps: E.feature_jacobians(fz, Z_STAR, eps))
    Gref = R.ggn_reference(cfg, r, Z_STAR)
    Jeta = R.feature_mean_jacobian(cfg, r, Z_STAR)
    return r, cA, Gref, Jeta


def _bootstrap_bias_fro_ci(mats, Gref, n_boot=1000, key=0):
    """Bootstrap 95% CI of ‖mean(mats) − Gref‖_F over batch resamples."""
    Bn = mats.shape[0]
    rng = np.random.default_rng(key)
    gref = np.asarray(Gref)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, Bn, Bn)
        vals[i] = np.linalg.norm(mats[idx].mean(0) - gref)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def _top2_angle(Gest, Gref):
    return MET.max_principal_angle(eigendecompose(Gest).eigvecs[:, :2],
                                   eigendecompose(Gref).eigvecs[:, :2])


def _finite_M_study(cfg, label):
    r, cA, Gref, Jeta = _regime(cfg)
    gref_fro = float(jnp.linalg.norm(Gref))
    # pooled covariance correction Ĉ (predicted plug-in bias = Ĉ/M)
    A_big = cA(jax.random.normal(jax.random.PRNGKey(999), (4000, 2)))
    Chat = E.plugin_bias_covariance(A_big, Jeta)

    out = {}
    per_M_arrays = {}
    for Ms in M_GRID:
        GV = np.empty((B, 5, 5)); GU = np.empty((B, 5, 5)); negmin = np.empty(B)
        for b in range(B):
            A = cA(jax.random.normal(jax.random.PRNGKey(10_000 * (Ms + 1) + b), (Ms, 2)))
            gv = E.ggn_plugin(A); gu = E.ggn_cross_seed(A)
            GV[b] = np.asarray(gv); GU[b] = np.asarray(gu)
            negmin[b] = float(jnp.linalg.eigvalsh(gu)[0])
        meanGV = jnp.asarray(GV.mean(0)); meanGU = jnp.asarray(GU.mean(0))
        plugin_bias = meanGV - Gref
        cross_bias = meanGU - Gref
        predicted = Chat / Ms
        # cross-seed MC noise floor (SE of the mean, Frobenius)
        se_GU = float(np.sqrt((np.linalg.norm(GU - GU.mean(0), axis=(1, 2)) ** 2).mean() / B))
        out[str(Ms)] = {
            "rel_bias_plugin": float(jnp.linalg.norm(plugin_bias)) / gref_fro,
            "plugin_bias_fro": float(jnp.linalg.norm(plugin_bias)),
            "plugin_bias_fro_ci": _bootstrap_bias_fro_ci(GV, Gref, key=Ms),
            "predicted_plugin_bias_fro": float(jnp.linalg.norm(predicted)),
            "rel_fro_obs_vs_pred_plugin_bias": MET.rel_frobenius_error(plugin_bias, predicted),
            "rel_bias_crossseed": float(jnp.linalg.norm(cross_bias)) / gref_fro,
            "crossseed_bias_fro": float(jnp.linalg.norm(cross_bias)),
            "crossseed_bias_fro_ci": _bootstrap_bias_fro_ci(GU, Gref, key=Ms + 1),
            "crossseed_noise_floor_fro": se_GU,
            "crossseed_bias_compatible_zero": bool(float(jnp.linalg.norm(cross_bias)) <= 2 * se_GU),
            "var_GV_fro": float((np.linalg.norm(GV - GV.mean(0), axis=(1, 2)) ** 2).mean()),
            "top2_angle_meanGV_vs_ref": _top2_angle(meanGV, Gref),
            "top2_angle_meanGU_vs_ref": _top2_angle(meanGU, Gref),
            "crossseed_negeig_freq": float((negmin < -_TOL_NEG).mean()),
            "crossseed_negeig_mean_min": float(negmin.mean()),
            "rel_fro_meanGV_vs_ref": MET.rel_frobenius_error(meanGV, Gref),
            "rel_fro_meanGU_vs_ref": MET.rel_frobenius_error(meanGU, Gref),
        }
        per_M_arrays[f"{label}_M{Ms}_meanGV"] = np.asarray(meanGV)
        per_M_arrays[f"{label}_M{Ms}_meanGU"] = np.asarray(meanGU)

    ref_eigs = np.sort(np.asarray(jnp.linalg.eigvalsh(Gref)))[::-1]
    summary = {
        "config": cfg._asdict(), "gamma": GAMMA, "D": D_MAIN, "B": B,
        "reference_eigenvalues": ref_eigs.tolist(),
        "eigengap_l2_over_l3": float(ref_eigs[1] / ref_eigs[2]),
        "per_M": out,
    }
    per_M_arrays[f"{label}_Gref"] = np.asarray(Gref)
    return summary, per_M_arrays


def _residual_sweep_opg(cfg, Ms=128):
    r, cA, Gref, _ = _regime(cfg)
    rows = {}
    for delta in DELTAS:
        z_y = Z_STAR + jnp.array([delta, 0.0, 0.0, 0.0, 0.0])
        r_pop = R.feature_mean(cfg, r, Z_STAR) - R.feature_mean(cfg, r, z_y)
        eta_y = R.feature_mean(cfg, r, z_y)
        Fpop = jnp.zeros((5, 5)); Femp = jnp.zeros((5, 5))
        Bo = 100
        for b in range(Bo):
            eps = jax.random.normal(jax.random.PRNGKey(333_000 + b), (Ms, 2))
            A = cA(eps)
            feats = jax.vmap(lambda e: R.feature_of_z(cfg, r, Z_STAR, e))(eps)
            Fpop += E.population_residual_opg(A, r_pop)
            Femp += E.empirical_loss_opg(A, feats, eta_y)
        Fpop /= Bo; Femp /= Bo
        rows[f"delta_{delta}"] = {
            "residual_norm": float(jnp.linalg.norm(r_pop)),
            "Fpop_fro": float(jnp.linalg.norm(Fpop)),
            "Fpop_vs_ref_raw_relfro": MET.rel_frobenius_error(Fpop, Gref),
            "Fpop_vs_ref_tracenorm_relfro":
                (MET.rel_frobenius_error(E.trace_normalized(Fpop), E.trace_normalized(Gref))
                 if float(jnp.trace(Fpop)) > 1e-12 else None),
            "Fpop_top2_angle_vs_ref":
                (_top2_angle(Fpop, Gref) if float(jnp.linalg.norm(Fpop)) > 1e-10 else None),
            "Femp_fro": float(jnp.linalg.norm(Femp)),
            "Femp_vs_ref_tracenorm_relfro":
                MET.rel_frobenius_error(E.trace_normalized(Femp), E.trace_normalized(Gref)),
        }
    return rows


def _rff_approximation_sweep(cfg, seeds=6):
    G_rbf = R.ggn_rbf(cfg, GAMMA, Z_STAR)
    rows = {}
    for D in D_SWEEP:
        Gs = [R.ggn_reference(cfg, R.frozen_rff(jax.random.PRNGKey(s), D, 2, GAMMA), Z_STAR)
              for s in range(seeds)]
        meanG = sum(Gs) / seeds
        rel = [MET.rel_frobenius_error(g, G_rbf) for g in Gs]
        rows[str(D)] = {
            "mean_rel_fro_vs_RBF": MET.rel_frobenius_error(meanG, G_rbf),
            "per_seed_rel_fro_mean": float(np.mean(rel)),
            "per_seed_rel_fro_std": float(np.std(rel)),
        }
    return {"G_rbf_eigs": np.sort(np.asarray(jnp.linalg.eigvalsh(G_rbf)))[::-1].tolist(),
            "per_D": rows}


def _acceptance(main_summary):
    hits = []
    for Ms, m in main_summary["per_M"].items():
        if m["rel_bias_plugin"] <= 0.10 and m["top2_angle_meanGV_vs_ref"] <= 0.087:
            hits.append(int(Ms))
    return {"A3_recovery_M_values": sorted(hits), "A3_passed": len(hits) > 0}


def _figure(main_summary, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    Ms = [int(k) for k in main_summary["per_M"].keys()]
    pv = [main_summary["per_M"][str(m)]["rel_bias_plugin"] for m in Ms]
    cu = [main_summary["per_M"][str(m)]["rel_bias_crossseed"] for m in Ms]
    ang = [np.degrees(main_summary["per_M"][str(m)]["top2_angle_meanGV_vs_ref"]) for m in Ms]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.3))
    ax[0].loglog(Ms, pv, "o-", label="plug-in bias")
    ax[0].loglog(Ms, cu, "s-", label="cross-seed bias")
    ax[0].set_xlabel("M"); ax[0].set_ylabel("relative bias"); ax[0].legend(fontsize=8)
    ax[0].set_title("estimator bias vs M (main regime)", fontsize=10)
    ax[1].semilogx(Ms, ang, "o-"); ax[1].axhline(5.0, ls=":", c="k")
    ax[1].set_xlabel("M"); ax[1].set_ylabel("top-2 angle (deg)")
    ax[1].set_title("leading-subspace recovery", fontsize=10)
    fig.suptitle("EXP-003 FIG-03 — MMD GGN estimation", fontsize=11)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=140, bbox_inches="tight")
    plt.close(fig)


def _new_run_dir(out_root):
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    d = Path(out_root) / f"{ts}_{prov.short_commit()}"
    (d / "figures").mkdir(parents=True, exist_ok=True)
    return d


def run(out_root="outputs/EXP-003") -> str:
    require_x64()
    run_dir = _new_run_dir(out_root)

    main_summary, main_arr = _finite_M_study(MAIN_CFG, "main")
    near_summary, near_arr = _finite_M_study(NEARDEG_CFG, "neardeg")
    residual = _residual_sweep_opg(MAIN_CFG)
    rff_sweep = _rff_approximation_sweep(MAIN_CFG)
    acceptance = _acceptance(main_summary)

    metrics = {"main_regime": main_summary, "near_degenerate_regime": near_summary,
               "residual_sweep_opg": residual, "rff_approximation_sweep": rff_sweep,
               "acceptance": acceptance}
    config = {
        "main_cfg": MAIN_CFG._asdict(), "neardeg_cfg": NEARDEG_CFG._asdict(),
        "gamma": GAMMA, "D_main": D_MAIN, "M_grid": list(M_GRID), "B": B,
        "feature_seed": FEATURE_SEED, "z_star": [0.0] * 5, "deltas": list(DELTAS),
        "D_sweep": list(D_SWEEP), "k_leading": 2,
    }
    prov.write_json(run_dir / "config.json", config)
    prov.write_json(run_dir / "provenance.json",
                    prov.run_metadata("EXP-003", config, {"feature_seed": FEATURE_SEED},
                                      command=_COMMAND))
    prov.write_json(run_dir / "metrics.json", metrics)
    np.savez_compressed(run_dir / "arrays.npz", **main_arr, **near_arr)
    _figure(main_summary, run_dir / "figures" / "fig03_mmd_ggn")
    return str(run_dir)


if __name__ == "__main__":
    print("run:", run())
