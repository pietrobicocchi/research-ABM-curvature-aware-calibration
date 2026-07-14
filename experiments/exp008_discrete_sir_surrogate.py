"""EXP-008 — Stochastic/discrete-MMD keystone.

On the *stochastic, discrete* network-SIR (models/network_sir.py) with an MMD
mean-embedding representation, ask two questions of the local GGN geometry:

  C14 — is the important eigenspace ROBUST to the chosen surrogate gradient?
        Compare the GGN under Gumbel-Sigmoid vs straight-through (the two
        estimators our repo shares with Quera-Bofarull 2025 §3.4.1/§3.4.2;
        SPA/StochasticAD §3.4.3 deferred, DEC-011 scope note).
  C15 — does TRUNCATED differentiation (grad_horizon) destroy that geometry?
        Sweep the differentiation horizon and measure how the leading GGN
        eigenspace rotates away from the full-horizon reference.

Both questions are answered at TWO operating points (a near-threshold partial
epidemic on a sparse graph, and a more supercritical epidemic on a denser,
different graph) to check the result is a property of the geometry, not of one
regime.

Consistency with Quera-Bofarull 2025 (arXiv:2509.03303): each surrogate's
mean-embedding Jacobian is validated against a finite-difference reference on the
seed-AVERAGED embedding with common random numbers (their §5.3.1 FD baseline).
For straight-through the per-path forward is piecewise-constant in z, so FD only
tracks the surrogate on the expectation E_ξ[ψ]; the FD-vs-surrogate gap is the
estimator bias that §5.3.1 reports.

Representation: m(z) = E_ξ[ φ_RFF( incidence(z, ξ) ) ] with frozen random Fourier
features (median-heuristic bandwidth, DEC-006), z ~ N(0, I) prior-whitened
coordinates (DEC-003). GGN G = J_mᵀ J_m estimated by the plug-in mean-Jacobian
(EXP-003 Ĝ_V), float64 (CLAUDE.md invariant).

Run:  uv run python -m experiments.exp008_discrete_sir_surrogate
Writes: outputs/EXP-008/<UTC-timestamp>_<short-commit>/{config,provenance,metrics}.json,
        arrays.npz, figures/fig08_discrete_sir_surrogate.{png,pdf}
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

from curvature_calib.config import enable_x64, require_x64

enable_x64()
require_x64()

import numpy as np  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from curvature_calib import provenance as prov  # noqa: E402
from curvature_calib.calibration import diagnostic as DIAG  # noqa: E402
from curvature_calib.geometry import mmd_estimators as E  # noqa: E402
from curvature_calib.geometry import rff as R  # noqa: E402
from curvature_calib.losses import mmd as MMD  # noqa: E402
from curvature_calib.models import network_sir as NSIR  # noqa: E402

_COMMAND = "uv run python -m experiments.exp008_discrete_sir_surrogate"

PARAM_NAMES = ("beta", "gamma", "I0_frac", "t_lock_norm", "f_lock")


class OP(NamedTuple):
    """A discrete network-SIR operating point (physical theta + graph)."""
    name: str
    beta0: float
    gamma0: float
    i0f0: float
    tl0: float
    fl0: float
    mean_degree: float
    graph_seed: int
    T: int


# OP_A is the recorded primary point (EXP-008 first run); OP_B is a genuinely
# different regime + graph, added as the cross-regime robustness check.
OP_A = OP("A_near_threshold", 0.03, 0.06, 0.03, 0.20, 0.30, 6.0, 17, 40)
OP_B = OP("B_supercritical_denser", 0.045, 0.10, 0.03, 0.15, 0.25, 8.0, 29, 40)
OPERATING_POINTS = (OP_A, OP_B)

# ---- shared settings (fixed across operating points) ----------------------
PRIOR_SCALE = jnp.array([0.30, 0.30, 0.30, 0.50, 0.60])  # z ~ N(0, I) prior widths
N_POP, DT = 150, 1.0
K_SIG, K_INIT, GUMBEL_TAU = 20.0, 50.0, 0.5
D_RFF, M_SEEDS = 128, 128
SURROGATES = ("gumbel", "straight_through")
HORIZON_FRACS = (1.0, 0.5, 0.25, 0.125)   # grad_horizon = frac * T
FD_STEP = 1e-2
K_LEAD = 2                                 # leading subspace for principal angles
BASE_SEED = 20250714


def _logit(p: float) -> float:
    return float(np.log(p / (1.0 - p)))


def transform(z: jax.Array, op: OP) -> jax.Array:
    """z (prior-whitened, N(0,I)) -> physical network-SIR theta (P=5)."""
    beta = op.beta0 * jnp.exp(PRIOR_SCALE[0] * z[0])
    gamma = op.gamma0 * jnp.exp(PRIOR_SCALE[1] * z[1])
    i0 = op.i0f0 * jnp.exp(PRIOR_SCALE[2] * z[2])
    tl = jax.nn.sigmoid(_logit(op.tl0) + PRIOR_SCALE[3] * z[3])
    fl = jax.nn.sigmoid(_logit(op.fl0) + PRIOR_SCALE[4] * z[4])
    return jnp.stack([beta, gamma, i0, tl, fl])


def incidence(z, key, surrogate, grad_horizon, op: OP) -> jax.Array:
    """One incidence trajectory (T,) for whitened coords z under one seed."""
    return NSIR.simulate(
        transform(z, op), key, T=op.T, N=N_POP, mean_degree=op.mean_degree, dt=DT,
        k_sig=K_SIG, k_init=K_INIT, gumbel_tau=GUMBEL_TAU, surrogate=surrogate,
        graph_seed=op.graph_seed, grad_horizon=grad_horizon,
    )


def _seed_keys(n: int, seed: int) -> jax.Array:
    return jax.random.split(jax.random.PRNGKey(seed), n)


def _feature_fn(rff, surrogate, grad_horizon, op):
    return lambda z, key: R.feature_map(rff, incidence(z, key, surrogate, grad_horizon, op))


def _mean_embedding(z, keys, rff, surrogate, grad_horizon, op):
    f = _feature_fn(rff, surrogate, grad_horizon, op)
    return jnp.mean(jax.vmap(lambda k: f(z, k))(keys), axis=0)


def _feature_jacobians(z, keys, rff, surrogate, grad_horizon, op):
    return E.feature_jacobians(_feature_fn(rff, surrogate, grad_horizon, op), z, keys)


def _fd_jacobian(z, keys, rff, surrogate, grad_horizon, op, step):
    """Central-difference D_z m(z) on the seed-averaged embedding (CRN). (D, P)."""
    cols = []
    for i in range(z.shape[0]):
        e = jnp.zeros_like(z).at[i].set(step)
        m_plus = _mean_embedding(z + e, keys, rff, surrogate, grad_horizon, op)
        m_minus = _mean_embedding(z - e, keys, rff, surrogate, grad_horizon, op)
        cols.append((m_plus - m_minus) / (2.0 * step))
    return jnp.stack(cols, axis=1)


def _relspec(eigvals: np.ndarray) -> list[float]:
    top = float(eigvals[0])
    return [float(x / top) for x in eigvals]


def analyze(op: OP) -> tuple[dict, dict]:
    """Full C14 + C15 analysis at one operating point. Returns (metrics, arrays)."""
    z_star = jnp.zeros(5)
    keys = _seed_keys(M_SEEDS, BASE_SEED)

    ref_trajs = jax.vmap(lambda k: incidence(z_star, k, "gumbel", None, op))(keys)  # (M,T)
    gamma = float(MMD.median_heuristic(ref_trajs))
    rff = R.frozen_rff(jax.random.PRNGKey(BASE_SEED + 1), D_RFF, op.T, gamma)

    jac_fn = jax.jit(lambda z, ky, surr: _feature_jacobians(z, ky, rff, surr, None, op),
                     static_argnums=(2,))

    per_surrogate, G_full, eig_full = {}, {}, {}
    for surr in SURROGATES:
        A = jac_fn(z_star, keys, surr)
        G = E.ggn_plugin(A)
        eig = DIAG.eigendecompose(G)
        G_full[surr], eig_full[surr] = G, eig
        J_ad = jnp.mean(A, axis=0)
        J_fd = _fd_jacobian(z_star, keys, rff, surr, None, op, FD_STEP)
        col_err = [float(jnp.linalg.norm(J_ad[:, i] - J_fd[:, i]) /
                         (jnp.linalg.norm(J_fd[:, i]) + 1e-30)) for i in range(5)]
        per_surrogate[surr] = {
            "eigvals": [float(x) for x in np.asarray(eig.eigvals)],
            "relative_spectrum": _relspec(np.asarray(eig.eigvals)),
            "numerical_rank_rtol1e6": int(jnp.sum(eig.eigvals > 1e-6 * eig.eigvals[0])),
            "fd_reljac_error_per_param": {PARAM_NAMES[i]: col_err[i] for i in range(5)},
            "fd_reljac_error_median": float(np.median(col_err)),
        }

    Vg, Vs = eig_full["gumbel"].eigvecs, eig_full["straight_through"].eigvecs
    ang_lead = np.degrees(np.asarray(DIAG.principal_angles(Vg[:, :K_LEAD], Vs[:, :K_LEAD])))
    subspace_max_angle = {
        f"k{k}": float(np.max(np.degrees(np.asarray(
            DIAG.principal_angles(Vg[:, :k], Vs[:, :k]))))) for k in (1, 2, 3)}
    eg = np.asarray(eig_full["gumbel"].eigvals)
    es = np.asarray(eig_full["straight_through"].eigvals)
    c14 = {
        "leading_k": K_LEAD,
        "leading_subspace_principal_angles_deg": [float(a) for a in ang_lead],
        "max_leading_angle_deg": float(np.max(ang_lead)),
        "subspace_max_angle_deg_by_k": subspace_max_angle,
        "eigval_ratio_st_over_gumbel": [float(es[i] / eg[i]) for i in range(5)],
    }

    horizons = [int(round(f * op.T)) for f in HORIZON_FRACS]
    V_ref = eig_full["gumbel"].eigvecs
    lam_ref = np.asarray(eig_full["gumbel"].eigvals)
    horizon_rows = {}
    for gh, frac in sorted(zip(horizons, HORIZON_FRACS), reverse=True):
        A = jac_fn(z_star, keys, "gumbel") if gh >= op.T else \
            _feature_jacobians(z_star, keys, rff, "gumbel", gh, op)
        eig = DIAG.eigendecompose(E.ggn_plugin(A))
        ang = np.degrees(np.asarray(
            DIAG.principal_angles(V_ref[:, :K_LEAD], eig.eigvecs[:, :K_LEAD])))
        lam = np.asarray(eig.eigvals)
        horizon_rows[f"h{gh:02d}"] = {
            "grad_horizon": gh, "horizon_frac": frac,
            "max_leading_angle_vs_full_deg": float(np.max(ang)),
            "top_eigval_ratio_to_full": float(lam[0] / lam_ref[0]),
            "relative_spectrum": _relspec(lam),
        }

    fd_ok = {s: per_surrogate[s]["fd_reljac_error_median"] < 0.05 for s in SURROGATES}
    horizon_max_angle = max(r["max_leading_angle_vs_full_deg"] for r in horizon_rows.values())
    metrics = {
        "operating_point": {
            "spec": op._asdict(),
            "theta_star": {PARAM_NAMES[i]: float(transform(z_star, op)[i]) for i in range(5)},
            "peak_mean_incidence": float(jnp.max(jnp.mean(ref_trajs, axis=0))),
            "total_mean_incidence": float(jnp.mean(jnp.sum(ref_trajs, axis=1))),
            "attack_rate": float(jnp.mean(jnp.sum(ref_trajs, axis=1)) / N_POP),
            "rff_bandwidth_median_heuristic": gamma,
        },
        "C14_cross_surrogate": c14,
        "C15_horizon_sweep": horizon_rows,
        "per_surrogate": per_surrogate,
        "acceptance": {
            "fd_gate_pass_per_surrogate": fd_ok,
            "C14_leading_eigenspace_robust_lt15deg": bool(c14["max_leading_angle_deg"] < 15.0),
            "C14_max_leading_angle_deg": c14["max_leading_angle_deg"],
            "C15_horizon_destroys_geometry_gt15deg": bool(horizon_max_angle > 15.0),
            "C15_max_leading_angle_vs_full_deg": horizon_max_angle,
        },
    }
    arrays = {
        f"{op.name}_G_gumbel": np.asarray(G_full["gumbel"]),
        f"{op.name}_G_straight_through": np.asarray(G_full["straight_through"]),
        f"{op.name}_ref_incidence_mean": np.asarray(jnp.mean(ref_trajs, axis=0)),
    }
    return metrics, arrays


def run(out_root: str = "outputs/EXP-008") -> str:
    require_x64()
    ops_metrics, all_arrays = {}, {}
    for op in OPERATING_POINTS:
        m, a = analyze(op)
        ops_metrics[op.name] = m
        all_arrays.update(a)

    # cross-point robustness verdict
    cross = {
        "C14_max_leading_angle_deg_by_point": {
            n: ops_metrics[n]["C14_cross_surrogate"]["max_leading_angle_deg"]
            for n in ops_metrics},
        "C15_max_horizon_angle_deg_by_point": {
            n: ops_metrics[n]["acceptance"]["C15_max_leading_angle_vs_full_deg"]
            for n in ops_metrics},
        "gumbel_fd_median_by_point": {
            n: ops_metrics[n]["per_surrogate"]["gumbel"]["fd_reljac_error_median"]
            for n in ops_metrics},
        "C14_robust_all_points_lt15deg": bool(all(
            ops_metrics[n]["C14_cross_surrogate"]["max_leading_angle_deg"] < 15.0
            for n in ops_metrics)),
        "C15_destroys_all_points_gt15deg": bool(all(
            ops_metrics[n]["acceptance"]["C15_max_leading_angle_vs_full_deg"] > 15.0
            for n in ops_metrics)),
    }
    metrics = {"operating_points": ops_metrics, "cross_point_summary": cross}

    config = {
        "operating_points": {op.name: op._asdict() for op in OPERATING_POINTS},
        "prior_scale": [float(x) for x in PRIOR_SCALE], "N_pop": N_POP, "dt": DT,
        "k_sig": K_SIG, "k_init": K_INIT, "gumbel_tau": GUMBEL_TAU,
        "D_rff": D_RFF, "M_seeds": M_SEEDS, "surrogates": list(SURROGATES),
        "horizon_fracs": list(HORIZON_FRACS), "fd_step": FD_STEP, "k_lead": K_LEAD,
        "base_seed": BASE_SEED, "spa_estimator": "deferred (DEC-011)",
    }
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(out_root) / f"{ts}_{prov.short_commit()}"
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    prov.write_json(run_dir / "config.json", config)
    prov.write_json(run_dir / "provenance.json",
                    prov.run_metadata("EXP-008", config, {"base_seed": BASE_SEED},
                                      command=_COMMAND))
    prov.write_json(run_dir / "metrics.json", metrics)
    np.savez_compressed(run_dir / "arrays.npz", **all_arrays)
    _figure(metrics, run_dir / "figures" / "fig08_discrete_sir_surrogate")
    return str(run_dir)


def _figure(metrics, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    ops = metrics["operating_points"]
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.4))
    ks = ("k1", "k2", "k3")
    x = np.arange(len(ks))
    width = 0.8 / max(len(ops), 1)
    for j, (name, m) in enumerate(ops.items()):
        vals = [m["C14_cross_surrogate"]["subspace_max_angle_deg_by_k"][k] for k in ks]
        ax[0].bar(x + j * width, vals, width, label=name)
    ax[0].axhline(15.0, ls=":", c="k", lw=1)
    ax[0].set_xticks(x + width * (len(ops) - 1) / 2)
    ax[0].set_xticklabels(["k=1", "k=2", "k=3"])
    ax[0].set_ylabel("Gumbel↔ST leading angle (deg)")
    ax[0].set_title("C14: cross-surrogate robustness", fontsize=10)
    ax[0].legend(fontsize=7)
    for name, m in ops.items():
        rows = m["C15_horizon_sweep"]
        fracs = [rows[k]["horizon_frac"] for k in rows]
        angs = [rows[k]["max_leading_angle_vs_full_deg"] for k in rows]
        order = np.argsort(fracs)
        ax[1].plot(np.array(fracs)[order], np.array(angs)[order], "s-", label=name)
    ax[1].set_xlabel("differentiation horizon / T")
    ax[1].set_ylabel("leading-eigenspace angle vs full (deg)")
    ax[1].set_title("C15: horizon truncation", fontsize=10)
    ax[1].legend(fontsize=7)
    fig.suptitle("EXP-008 — discrete-SIR surrogate/horizon GGN geometry (2 operating points)",
                 fontsize=11)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=140, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    print("run:", run())
