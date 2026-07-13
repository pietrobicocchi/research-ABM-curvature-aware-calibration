"""EXP-007 — SIR observation-design study.

EXP-006 showed the intervention value is unidentified because the lockdown strength
f_lock is barely constrained by incidence alone. Which ADDED observation recovers
information in that weak direction and collapses the policy uncertainty? Compare
observation designs by (a) the prior-relative GGN spectrum / data-dominant dimension,
(b) the smallest-eigenvalue direction and its f_lock loading, and (c) the in-fit
range of the counterfactual intervention value (EXP-006's Lagrangian profile).
Supports C13.

Run:  uv run python -m experiments.exp007_sir_obs_design
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
from experiments.exp005_sir_posterior import T, PARAM_NAMES, Z0  # noqa: E402

_COMMAND = "uv run python -m experiments.exp007_sir_obs_design"
T_SIR, N_POP, DT, K_SIG = 200, 1e5, 1.0, 20.0
SIGMA_INC = 150.0                       # incidence noise (EXP-005 baseline)
_MUS = np.concatenate([-np.geomspace(0.05, 20.0, 8)[::-1], [0.0], np.geomspace(0.05, 20.0, 8)])
FIT_BUDGET = 2.0


def sir_full(z):
    """Deterministic SIR replicating models/sir dynamics; returns observables."""
    beta, gamma, I0f, tl, fl = T(z)
    I0 = N_POP * I0f
    ts = jnp.arange(T_SIR, dtype=jnp.float64) / float(T_SIR)

    def step(state, t_norm):
        S, I = state
        beta_eff = beta * (1.0 - (1.0 - fl) * jax.nn.sigmoid(K_SIG * (t_norm - tl)))
        inc = beta_eff * jnp.clip(S, min=0.0) * jnp.clip(I, min=0.0) / N_POP
        S2 = S - DT * inc
        I2 = I + DT * (inc - gamma * I)
        return (S2, I2), (DT * inc, I, beta_eff)

    _, (incidence, prevalence, beta_eff) = jax.lax.scan(step, (N_POP - I0, I0), ts)
    return incidence, prevalence, beta_eff, ts


def incidence(z):
    return sir_full(z)[0]


def Q_intervention(z):
    """Cases averted by the lockdown = total(no lockdown) − total(actual)."""
    z_nolock = z  # override f_lock via a separate call
    beta, gamma, I0f, tl, fl = T(z)
    # no-lockdown counterfactual: set f_lock=1 by re-simulating with fl=1
    I0 = N_POP * I0f
    ts = jnp.arange(T_SIR, dtype=jnp.float64) / float(T_SIR)

    def step(state, t_norm):
        S, I = state
        inc = beta * jnp.clip(S, min=0.0) * jnp.clip(I, min=0.0) / N_POP
        return (S - DT * inc, I + DT * (inc - gamma * I)), DT * inc

    _, inc_nolock = jax.lax.scan(step, (N_POP - I0, I0), ts)
    return jnp.sum(inc_nolock) - jnp.sum(incidence(z))


# ---- observation designs: each returns a whitened representation m(z) ----
# window spanning the lockdown transition (t_lock=0.4) into the locked phase, so
# it carries information about BOTH the strength f_lock and the timing t_lock.
_LOCK_WINDOW = slice(int(0.3 * T_SIR), int(0.7 * T_SIR))


def _designs(z0):
    inc0, prev0, beta0, ts = sir_full(z0)
    sig_prev = 0.10 * float(jnp.max(prev0))     # 10% of peak prevalence
    sig_beta = 0.02                              # direct transmission measurement noise

    def m_incidence(z):
        return sir_full(z)[0] / SIGMA_INC

    def m_inc_prev(z):
        inc, prev, _, _ = sir_full(z)
        return jnp.concatenate([inc / SIGMA_INC, prev / sig_prev])

    def m_inc_compliance(z):
        inc, _, beta_eff, _ = sir_full(z)
        return jnp.concatenate([inc / SIGMA_INC, beta_eff[_LOCK_WINDOW] / sig_beta])

    return {"1_incidence_only": m_incidence,
            "2_incidence+prevalence": m_inc_prev,
            "3_incidence+compliance": m_inc_compliance}


def _argmin(obj):
    vg = jax.jit(jax.value_and_grad(obj))
    def f(z):
        val, g = vg(jnp.asarray(z))
        return float(val), np.asarray(g, dtype=np.float64)
    return jnp.asarray(minimize(f, np.zeros(5), jac=True, method="L-BFGS-B",
                                options={"maxiter": 300, "ftol": 1e-14, "gtol": 1e-11}).x)


def _intervention_range(loss, Q0):
    """In-fit fractional + absolute range of the intervention value under `loss`."""
    fr = []
    for mu in _MUS:
        z = _argmin(lambda zz: loss(zz) - mu * Q_intervention(zz) / abs(Q0))
        if float(loss(z)) <= FIT_BUDGET:
            fr.append(float(Q_intervention(z)))
    return (min(fr), max(fr)) if fr else (Q0, Q0)


def run(out_root="outputs/EXP-007") -> str:
    require_x64()
    designs = _designs(Z0)
    Q0 = float(Q_intervention(Z0))

    results = {}
    for name, m_fn in designs.items():
        y = m_fn(Z0)
        loss = lambda z, m_fn=m_fn, y=y: 0.5 * jnp.sum((m_fn(z) - y) ** 2)
        J = jax.jacfwd(m_fn)(Z0)
        G = jnp.asarray(GGN.symmetrize(J.T @ J))
        eig = DIAG.eigendecompose(G)
        pr = np.asarray(eig.eigvals)
        v_sloppy = eig.eigvecs[:, -1]
        lo, hi = _intervention_range(loss, Q0)
        results[name] = {
            "prior_relative_eigs": [float(x) for x in pr],
            "d_data_lambda_gt_1": int((pr > 1).sum()),
            "smallest_eigval": float(pr[-1]),
            "sloppiest_dir_loading": {PARAM_NAMES[i]: float(abs(v_sloppy[i])) for i in range(5)},
            "sloppiest_top_loading": PARAM_NAMES[int(np.argmax(np.abs(np.asarray(v_sloppy))))],
            "intervention_range_abs": [lo, hi],
            "intervention_range_frac": [(lo - Q0) / abs(Q0), (hi - Q0) / abs(Q0)],
            "intervention_range_width_cases": hi - lo,
        }

    summary = {"Q_intervention_at_mode": Q0, "fit_budget_nats": FIT_BUDGET,
               "sigma_incidence": SIGMA_INC, "designs": results,
               "headline": {name: {
                   "d_data": results[name]["d_data_lambda_gt_1"],
                   "smallest_eigval": results[name]["smallest_eigval"],
                   "sloppiest_loads_on": results[name]["sloppiest_top_loading"],
                   "intervention_range_cases": [round(results[name]["intervention_range_abs"][0]),
                                                round(results[name]["intervention_range_abs"][1])],
               } for name in results}}

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(out_root) / f"{ts}_{prov.short_commit()}"
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    config = {"designs": list(designs), "sigma_incidence": SIGMA_INC,
              "lock_window": [_LOCK_WINDOW.start, _LOCK_WINDOW.stop],
              "fit_budget_nats": FIT_BUDGET, "reference": "EXP-005/006 SIR setup"}
    prov.write_json(run_dir / "config.json", config)
    prov.write_json(run_dir / "provenance.json",
                    prov.run_metadata("EXP-007", config, {"data": 0}, command=_COMMAND))
    prov.write_json(run_dir / "metrics.json", summary)
    _figure(summary, run_dir / "figures" / "fig07_sir_obs_design")
    return str(run_dir)


def _figure(summary, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.3))
    names = list(summary["designs"])
    for name in names:
        ev = np.array(summary["designs"][name]["prior_relative_eigs"])
        ax[0].semilogy(range(1, 6), ev, "o-", label=name.split("_", 1)[1])
    ax[0].axhline(1.0, ls=":", c="k"); ax[0].set_xlabel("GGN direction (stiff→sloppy)")
    ax[0].set_ylabel("prior-relative λ"); ax[0].set_title("spectrum by design", fontsize=10)
    ax[0].legend(fontsize=7)
    widths = [summary["designs"][n]["intervention_range_width_cases"] for n in names]
    ax[1].bar(range(len(names)), widths)
    ax[1].set_xticks(range(len(names))); ax[1].set_xticklabels([n[0] for n in names])
    ax[1].set_ylabel("intervention-value in-fit range (cases)")
    ax[1].set_title("policy uncertainty by design", fontsize=10)
    fig.suptitle("EXP-007 — SIR observation design", fontsize=11)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=140, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    print("run:", run())
