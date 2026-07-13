"""EXP-006 — SIR policy-functional analysis.

Does data non-identifiability leave a POLICY output undetermined? For each policy
functional Q, trace the profiled fit cost vs the policy value:

    P(q) = min_{z : Q(z)=q} L(z),

by sweeping a Lagrange multiplier (min_z L(z) − μ Q(z)). If Q can move materially
while the profiled fit L stays good (≤ a small budget), the data does not pin down
the policy answer — and we check that the movement lies along a SLOPPY (low
prior-relative-eigenvalue) GGN direction. Supports C12.

Run:  uv run python -m experiments.exp006_sir_policy
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
from scipy.optimize import minimize  # noqa: E402

from curvature_calib import provenance as prov  # noqa: E402
from curvature_calib.calibration import diagnostic as DIAG  # noqa: E402
from curvature_calib.geometry import ggn as GGN  # noqa: E402
from experiments.exp005_sir_posterior import T, m, SIGMA_DATA, PARAM_NAMES, Z0  # noqa: E402
from curvature_calib.models.sir import simulate  # noqa: E402

_COMMAND = "uv run python -m experiments.exp006_sir_policy"
_KEY0 = jax.random.PRNGKey(0)
# Lagrange multipliers μ (both signs), scaled by 1/|Q0| inside the objective.
_MUS = np.concatenate([-np.geomspace(0.05, 20.0, 8)[::-1], [0.0], np.geomspace(0.05, 20.0, 8)])
FIT_BUDGET = 2.0   # nats of profiled loss still counted as a "good fit"


def _incidence_flock(z, flock):
    theta = T(z).at[4].set(flock)
    return simulate(theta, _KEY0, T=200, N=1e5, sigma_obs=0.0)


def Q_peak(z):
    return jnp.max(m(z))


def Q_total(z):
    return jnp.sum(m(z))


def Q_intervention(z):
    return jnp.sum(_incidence_flock(z, 1.0)) - jnp.sum(m(z))


_FUNCTIONALS = {"total_cases": Q_total, "peak_incidence": Q_peak, "intervention_value": Q_intervention}


def _argmin(obj):
    vg = jax.jit(jax.value_and_grad(obj))
    def f(z):
        val, g = vg(jnp.asarray(z))
        return float(val), np.asarray(g, dtype=np.float64)
    res = minimize(f, np.zeros(5), jac=True, method="L-BFGS-B",
                   options={"maxiter": 300, "ftol": 1e-14, "gtol": 1e-11})
    return jnp.asarray(res.x)


def _profile_policy(loss, Qfn, Q0, eigvecs):
    """Sweep μ; for each, z* = argmin L − μ Q/|Q0|; record (Q frac change, fit, alignment)."""
    rows = []
    for mu in _MUS:
        z = _argmin(lambda zz: loss(zz) - mu * Qfn(zz) / abs(Q0))
        d = np.asarray(z)
        dn = d / (np.linalg.norm(d) + 1e-30)
        align = [float(abs(dn @ np.asarray(eigvecs[:, k]))) for k in range(5)]  # |cos| with each eigvec
        rows.append({"mu": float(mu), "Q_frac_change": float((Qfn(z) - Q0) / abs(Q0)),
                     "profiled_loss": float(loss(z)), "z_norm": float(np.linalg.norm(d)),
                     "alignment_abs_cos": align})
    return rows


def run(out_root="outputs/EXP-006") -> str:
    require_x64()
    y = m(Z0)
    loss = lambda z: 0.5 * jnp.sum((m(z) - y) ** 2) / SIGMA_DATA ** 2

    J = jax.jacfwd(m)(Z0)
    G = jnp.asarray(GGN.symmetrize((J.T @ J) / SIGMA_DATA ** 2))
    eig = DIAG.eigendecompose(G)                      # descending: col 0 stiffest, col 4 sloppiest
    prior_rel = np.asarray(eig.eigvals)
    Q0 = {name: float(f(Z0)) for name, f in _FUNCTIONALS.items()}

    functionals = {}
    for name, f in _FUNCTIONALS.items():
        rows = _profile_policy(loss, f, Q0[name], eig.eigvecs)
        # within the fit budget: how much can Q move, and along which direction?
        good = [r for r in rows if r["profiled_loss"] <= FIT_BUDGET]
        q_range = max(r["Q_frac_change"] for r in good) - min(r["Q_frac_change"] for r in good)
        # the largest-|Q change| good-fit point, and its dominant sloppy alignment
        extreme = max(good, key=lambda r: abs(r["Q_frac_change"]))
        dom_dir = int(np.argmax(extreme["alignment_abs_cos"]))      # 0=stiffest..4=sloppiest
        functionals[name] = {
            "Q_at_mode": Q0[name],
            "q_frac_range_within_budget": float(q_range),
            "extreme_q_frac_change": extreme["Q_frac_change"],
            "extreme_fit_cost": extreme["profiled_loss"],
            "extreme_dominant_eigdir": dom_dir,
            "extreme_dominant_eigval": float(prior_rel[dom_dir]),
            "curve": rows,
        }

    summary = {
        "prior_relative_eigs": [float(x) for x in prior_rel],
        "d_data_lambda_gt_1": int((prior_rel > 1).sum()),
        "fit_budget_nats": FIT_BUDGET,
        "functionals": functionals,
        "headline": {name: {
            "q_frac_range_within_budget": functionals[name]["q_frac_range_within_budget"],
            "extreme_change": functionals[name]["extreme_q_frac_change"],
            "at_fit_cost": functionals[name]["extreme_fit_cost"],
            "along_eigdir": functionals[name]["extreme_dominant_eigdir"],
            "eigval": functionals[name]["extreme_dominant_eigval"],
        } for name in _FUNCTIONALS},
    }

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(out_root) / f"{ts}_{prov.short_commit()}"
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    config = {"sigma_data": SIGMA_DATA, "mus": _MUS.tolist(), "fit_budget_nats": FIT_BUDGET,
              "functionals": list(_FUNCTIONALS), "reference": "EXP-005 SIR setup"}
    prov.write_json(run_dir / "config.json", config)
    prov.write_json(run_dir / "provenance.json",
                    prov.run_metadata("EXP-006", config, {"data": 0}, command=_COMMAND))
    prov.write_json(run_dir / "metrics.json", summary)
    _figure(summary, run_dir / "figures" / "fig06_sir_policy")
    return str(run_dir)


def _figure(summary, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.3))
    for name in ("total_cases", "peak_incidence"):
        rows = summary["functionals"][name]["curve"]
        q = [r["Q_frac_change"] for r in rows]
        L = [r["profiled_loss"] for r in rows]
        order = np.argsort(q)
        ax[0].plot(np.array(q)[order], np.array(L)[order], "o-", label=name)
    ax[0].axhline(summary["fit_budget_nats"], ls=":", c="k")
    ax[0].set_xlabel("policy Q fractional change"); ax[0].set_ylabel("profiled fit cost (nats)")
    ax[0].set_title("fit cost vs policy change", fontsize=10); ax[0].legend(fontsize=8)
    ev = np.array(summary["prior_relative_eigs"])
    ax[1].semilogy(range(1, 6), ev, "o-"); ax[1].axhline(1.0, ls=":", c="k")
    ax[1].set_xlabel("GGN direction (stiff→sloppy)"); ax[1].set_ylabel("prior-relative λ")
    ax[1].set_title(f"d_data={summary['d_data_lambda_gt_1']}", fontsize=10)
    fig.suptitle("EXP-006 — SIR policy-functional analysis", fontsize=11)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=140, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    print("run:", run())
