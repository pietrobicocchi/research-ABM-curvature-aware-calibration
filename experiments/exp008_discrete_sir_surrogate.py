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

# ---- discrete network-SIR operating point (P=5) ---------------------------
PARAM_NAMES = ("beta", "gamma", "I0_frac", "t_lock_norm", "f_lock")
BETA0, GAMMA0, I0F0, TL0, FL0 = 0.03, 0.06, 0.03, 0.20, 0.30
PRIOR_SCALE = jnp.array([0.30, 0.30, 0.30, 0.50, 0.60])  # z ~ N(0, I) prior widths

# ---- simulation / estimator settings --------------------------------------
T_SIM, N_POP, MEAN_DEGREE, DT = 40, 150, 6.0, 1.0
K_SIG, K_INIT, GUMBEL_TAU, GRAPH_SEED = 20.0, 50.0, 0.5, 17
D_RFF, M_SEEDS = 128, 128
SURROGATES = ("gumbel", "straight_through")
HORIZON_FRACS = (1.0, 0.5, 0.25, 0.125)   # grad_horizon = frac * T_SIM
FD_STEP = 1e-2
K_LEAD = 2                                 # leading subspace for principal angles
BASE_SEED = 20250714


def _logit(p: float) -> float:
    return float(np.log(p / (1.0 - p)))


def transform(z: jax.Array) -> jax.Array:
    """z (prior-whitened, N(0,I)) -> physical network-SIR theta (P=5)."""
    beta = BETA0 * jnp.exp(PRIOR_SCALE[0] * z[0])
    gamma = GAMMA0 * jnp.exp(PRIOR_SCALE[1] * z[1])
    i0 = I0F0 * jnp.exp(PRIOR_SCALE[2] * z[2])
    tl = jax.nn.sigmoid(_logit(TL0) + PRIOR_SCALE[3] * z[3])
    fl = jax.nn.sigmoid(_logit(FL0) + PRIOR_SCALE[4] * z[4])
    return jnp.stack([beta, gamma, i0, tl, fl])


def incidence(z: jax.Array, key: jax.Array, surrogate: str,
              grad_horizon: int | None) -> jax.Array:
    """One incidence trajectory (T_SIM,) for whitened coords z under one seed."""
    return NSIR.simulate(
        transform(z), key, T=T_SIM, N=N_POP, mean_degree=MEAN_DEGREE, dt=DT,
        k_sig=K_SIG, k_init=K_INIT, gumbel_tau=GUMBEL_TAU, surrogate=surrogate,
        graph_seed=GRAPH_SEED, grad_horizon=grad_horizon,
    )


def _seed_keys(n: int, seed: int) -> jax.Array:
    return jax.random.split(jax.random.PRNGKey(seed), n)


def _feature_fn(rff: R.RFF, surrogate: str, grad_horizon: int | None):
    """feature_of_z(z, key) -> RFF embedding (D,) of the incidence trajectory."""
    return lambda z, key: R.feature_map(rff, incidence(z, key, surrogate, grad_horizon))


def _mean_embedding(z, keys, rff, surrogate, grad_horizon):
    """m(z) = mean_seed φ_RFF(incidence(z, key)) with common random numbers."""
    f = _feature_fn(rff, surrogate, grad_horizon)
    return jnp.mean(jax.vmap(lambda k: f(z, k))(keys), axis=0)


def _feature_jacobians(z, keys, rff, surrogate, grad_horizon):
    """A_m = D_z φ_RFF(incidence(z, key_m)), stacked (M, D, P)."""
    return E.feature_jacobians(_feature_fn(rff, surrogate, grad_horizon), z, keys)


def _fd_jacobian(z, keys, rff, surrogate, grad_horizon, step):
    """Central-difference D_z m(z) on the seed-averaged embedding (CRN). (D, P)."""
    cols = []
    for i in range(z.shape[0]):
        e = jnp.zeros_like(z).at[i].set(step)
        m_plus = _mean_embedding(z + e, keys, rff, surrogate, grad_horizon)
        m_minus = _mean_embedding(z - e, keys, rff, surrogate, grad_horizon)
        cols.append((m_plus - m_minus) / (2.0 * step))
    return jnp.stack(cols, axis=1)


def _relspec(eigvals: np.ndarray) -> list[float]:
    top = float(eigvals[0])
    return [float(x / top) for x in eigvals]


def run(out_root: str = "outputs/EXP-008") -> str:
    require_x64()
    z_star = jnp.zeros(5)
    keys = _seed_keys(M_SEEDS, BASE_SEED)

    # --- frozen RFF: median-heuristic bandwidth on reference incidence trajs ---
    ref_trajs = jax.vmap(lambda k: incidence(z_star, k, "gumbel", None))(keys)  # (M, T)
    gamma = float(MMD.median_heuristic(ref_trajs))
    rff = R.frozen_rff(jax.random.PRNGKey(BASE_SEED + 1), D_RFF, T_SIM, gamma)
    peak_incidence = float(jnp.max(jnp.mean(ref_trajs, axis=0)))
    total_incidence = float(jnp.mean(jnp.sum(ref_trajs, axis=1)))

    jac_fn = jax.jit(lambda z, ky, surr: _feature_jacobians(z, ky, rff, surr, None),
                     static_argnums=(2,))

    # --- per-surrogate GGN at full horizon + FD validation (C14) --------------
    per_surrogate = {}
    G_full = {}
    eig_full = {}
    for surr in SURROGATES:
        A = jac_fn(z_star, keys, surr)                      # (M, D, P)
        G = E.ggn_plugin(A)
        eig = DIAG.eigendecompose(G)
        G_full[surr] = G
        eig_full[surr] = eig
        J_ad = jnp.mean(A, axis=0)                          # (D, P) plug-in mean Jacobian
        J_fd = _fd_jacobian(z_star, keys, rff, surr, None, FD_STEP)
        col_err = [float(jnp.linalg.norm(J_ad[:, i] - J_fd[:, i]) /
                         (jnp.linalg.norm(J_fd[:, i]) + 1e-30)) for i in range(5)]
        per_surrogate[surr] = {
            "eigvals": [float(x) for x in np.asarray(eig.eigvals)],
            "relative_spectrum": _relspec(np.asarray(eig.eigvals)),
            "numerical_rank_rtol1e6": int(jnp.sum(eig.eigvals > 1e-6 * eig.eigvals[0])),
            "fd_reljac_error_per_param": {PARAM_NAMES[i]: col_err[i] for i in range(5)},
            "fd_reljac_error_median": float(np.median(col_err)),
        }

    # --- C14: cross-surrogate eigenspace agreement ----------------------------
    # NB: comparing two full-rank P×P bases in R^P is degenerate (always 0°);
    # the informative comparison is the leading-k stiff subspace for k < P.
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
        "relspec_gumbel": _relspec(eg),
        "relspec_straight_through": _relspec(es),
    }

    # --- C15: horizon sweep (gumbel, smooth path -> clean reference) -----------
    horizons = [int(round(f * T_SIM)) for f in HORIZON_FRACS]
    V_ref = eig_full["gumbel"].eigvecs
    lam_ref = np.asarray(eig_full["gumbel"].eigvals)
    horizon_rows = {}
    for gh, frac in sorted(zip(horizons, HORIZON_FRACS), reverse=True):
        A = jac_fn(z_star, keys, "gumbel") if gh >= T_SIM else \
            _feature_jacobians(z_star, keys, rff, "gumbel", gh)
        G = E.ggn_plugin(A)
        eig = DIAG.eigendecompose(G)
        ang = np.degrees(np.asarray(
            DIAG.principal_angles(V_ref[:, :K_LEAD], eig.eigvecs[:, :K_LEAD])))
        lam = np.asarray(eig.eigvals)
        horizon_rows[f"h{gh:02d}"] = {
            "grad_horizon": gh,
            "horizon_frac": frac,
            "leading_angle_vs_full_deg": [float(a) for a in ang],
            "max_leading_angle_vs_full_deg": float(np.max(ang)),
            "top_eigval": float(lam[0]),
            "top_eigval_ratio_to_full": float(lam[0] / lam_ref[0]),
            "relative_spectrum": _relspec(lam),
        }

    # --- acceptance (thresholds are PROPOSED — see EXP-008.md) -----------------
    fd_ok = {s: per_surrogate[s]["fd_reljac_error_median"] < 0.05 for s in SURROGATES}
    c14_robust = bool(c14["max_leading_angle_deg"] < 15.0)
    horizon_max_angle = max(r["max_leading_angle_vs_full_deg"]
                            for r in horizon_rows.values())
    c15_degrades = bool(horizon_max_angle > 15.0)

    metrics = {
        "operating_point": {
            "theta_star": {PARAM_NAMES[i]: float(transform(z_star)[i]) for i in range(5)},
            "peak_mean_incidence": peak_incidence,
            "total_mean_incidence": total_incidence,
            "rff_bandwidth_median_heuristic": gamma,
        },
        "C14_cross_surrogate": c14,
        "C15_horizon_sweep": horizon_rows,
        "per_surrogate": per_surrogate,
        "acceptance": {
            "fd_gate_pass_per_surrogate": fd_ok,
            "C14_leading_eigenspace_robust_lt15deg": c14_robust,
            "C14_max_leading_angle_deg": c14["max_leading_angle_deg"],
            "C15_horizon_destroys_geometry_gt15deg": c15_degrades,
            "C15_max_leading_angle_vs_full_deg": horizon_max_angle,
        },
    }
    config = {
        "theta0_physical": {"beta": BETA0, "gamma": GAMMA0, "I0_frac": I0F0,
                            "t_lock_norm": TL0, "f_lock": FL0},
        "prior_scale": [float(x) for x in PRIOR_SCALE],
        "T_sim": T_SIM, "N_pop": N_POP, "mean_degree": MEAN_DEGREE, "dt": DT,
        "k_sig": K_SIG, "k_init": K_INIT, "gumbel_tau": GUMBEL_TAU,
        "graph_seed": GRAPH_SEED, "D_rff": D_RFF, "M_seeds": M_SEEDS,
        "surrogates": list(SURROGATES), "horizon_fracs": list(HORIZON_FRACS),
        "fd_step": FD_STEP, "k_lead": K_LEAD, "base_seed": BASE_SEED,
        "spa_estimator": "deferred (DEC-011)",
    }

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(out_root) / f"{ts}_{prov.short_commit()}"
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    prov.write_json(run_dir / "config.json", config)
    prov.write_json(run_dir / "provenance.json",
                    prov.run_metadata("EXP-008", config, {"base_seed": BASE_SEED},
                                      command=_COMMAND))
    prov.write_json(run_dir / "metrics.json", metrics)
    np.savez_compressed(
        run_dir / "arrays.npz",
        G_gumbel=np.asarray(G_full["gumbel"]),
        G_straight_through=np.asarray(G_full["straight_through"]),
        eigvecs_gumbel=np.asarray(eig_full["gumbel"].eigvecs),
        eigvecs_straight_through=np.asarray(eig_full["straight_through"].eigvecs),
        ref_incidence_mean=np.asarray(jnp.mean(ref_trajs, axis=0)),
    )
    _figure(metrics, run_dir / "figures" / "fig08_discrete_sir_surrogate")
    return str(run_dir)


def _figure(metrics, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.3))
    for surr in SURROGATES:
        rs = metrics["per_surrogate"][surr]["relative_spectrum"]
        ax[0].semilogy(range(1, len(rs) + 1), rs, "o-", label=surr)
    ax[0].set_xlabel("GGN direction (stiff->sloppy)")
    ax[0].set_ylabel("relative eigenvalue")
    ax[0].set_title(
        f"C14: cross-surrogate spectrum\n(max leading angle "
        f"{metrics['C14_cross_surrogate']['max_leading_angle_deg']:.1f} deg)",
        fontsize=9)
    ax[0].legend(fontsize=8)
    rows = metrics["C15_horizon_sweep"]
    fracs = [rows[k]["horizon_frac"] for k in rows]
    angs = [rows[k]["max_leading_angle_vs_full_deg"] for k in rows]
    ax[1].plot(fracs, angs, "s-")
    ax[1].set_xlabel("differentiation horizon / T")
    ax[1].set_ylabel("leading-eigenspace angle vs full (deg)")
    ax[1].set_title("C15: horizon truncation", fontsize=9)
    fig.suptitle("EXP-008 - discrete-SIR surrogate-gradient GGN geometry", fontsize=11)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=140, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    print("run:", run())
