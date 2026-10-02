"""Fig. 5, Sec. 5: the lockdown counterfactual Q(z) = C(z°) - C(z) (Eq. 17) on mean-field SIR.

For incidence alone and with each candidate observation added (Eq. 18), the profile
P(q) = min {U(z) : Q(z) = q} of the posterior energy U = w L + |z|^2 / 2 is traced,
together with the boundary of the in-budget set in the plane of the observed and
counterfactual totals. Also records the reference-trajectory facts quoted in Sec. 5.
"""
from __future__ import annotations

from gndiag.config import enable_x64, load_config

enable_x64()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

from gndiag import io, plotting  # noqa: E402
from gndiag.diagnostic.ggn import eigendecompose, symmetrize  # noqa: E402
from gndiag.models.mfsir import PARAM_NAMES, MeanFieldSIR, simulate  # noqa: E402

NAME = "mfsir_counterfactual"
DESIGNS = ("incidence", "incidence+prevalence", "incidence+compliance")
LABEL = {"incidence": "incidence only", "incidence+prevalence": "+ prevalence",
         "incidence+compliance": "+ compliance"}
PANEL_B = ("incidence", "incidence+compliance")


def designs(sir, cfg):
    """Whitened representations m(z) for each observation design."""
    z0 = jnp.zeros(5)
    _, prev0, _ = sir.trajectory(z0)
    sig_prev = cfg["prevalence_noise_fraction"] * float(jnp.max(prev0))
    c = cfg["compliance"]
    window = slice(c["first_day"], c["end_day"])
    sig_inc, sig_beta = sir.sigma_obs, c["noise_sd"]

    def m_incidence(z):
        return sir.trajectory(z)[0] / sig_inc

    def m_prevalence(z):
        inc, prev, _ = sir.trajectory(z)
        return jnp.concatenate([inc / sig_inc, prev / sig_prev])

    def m_compliance(z):
        inc, _, beta_eff = sir.trajectory(z)
        return jnp.concatenate([inc / sig_inc, beta_eff[window] / sig_beta])

    return dict(zip(DESIGNS, (m_incidence, m_prevalence, m_compliance)))


class Counterfactual:
    def __init__(self, sir, cfg):
        self.sir, self.cfg = sir, cfg
        self.scale = cfg["profile"]["scale_cases"]
        z0 = jnp.zeros(5)
        self.C0, self.A0 = float(self.C(z0)), float(self.A(z0))

    def C(self, z):
        return self.sir.total_without_lockdown(z)

    def A(self, z):
        return jnp.sum(self.sir.trajectory(z)[0])

    def Q(self, z):
        return self.C(z) - self.A(z)

    def solve(self, vg, x0, *params):
        def fun(x):
            v, g = vg(jnp.asarray(x), *params)
            return float(v), np.asarray(g, dtype=np.float64)
        o = self.cfg["optimizer"]
        r = minimize(fun, np.asarray(x0, float), jac=True, method="L-BFGS-B",
                     options={"maxiter": o["maxiter"], "ftol": o["ftol"], "gtol": o["gtol"]})
        return np.asarray(r.x)

    def record(self, z, U, L, U0):
        zj = jnp.asarray(z)
        C, A = float(self.C(zj)), float(self.A(zj))
        return {"z": [float(v) for v in z], "dU": float(U(zj)) - U0, "L": float(L(zj)),
                "Q": C - A, "C": C, "A": A, "dC": C - self.C0, "dA": A - self.A0}

    def profile(self, U, L, U0, q_grid, stop_dU):
        """P(q) by a quadratic penalty on Q(z) - q with increasing weight, warm-started
        by continuation in q outward from the fitted value; the upward walk continues
        past the grid until the energy exceeds stop_dU."""
        p = self.cfg["profile"]
        embed = lambda x: jnp.zeros(5).at[jnp.arange(5)].set(jnp.asarray(x))
        vg = jax.jit(jax.value_and_grad(
            lambda x, q, rho: U(embed(x)) + 0.5 * rho * ((self.Q(embed(x)) - q) / self.scale) ** 2))
        rows = []
        q0 = float(self.Q(jnp.zeros(5)))
        for direction in ("up", "down"):
            if direction == "up":
                grid = sorted([q for q in q_grid if q >= q0])
            else:
                grid = sorted([q for q in q_grid if 0.0 <= q <= q0], reverse=True)
            x = np.zeros(5)
            for q in grid:
                for rho in p["penalty_rhos"]:
                    x = self.solve(vg, x, float(q), float(rho))
                r = self.record(x, U, L, U0) | {"q_target": float(q)}
                rows.append(r)
                if r["dU"] > stop_dU:
                    break
            else:
                if direction == "up":
                    q = float(grid[-1])
                    while q < p["q_cap"]:
                        q += max(20.0, 0.03 * q)
                        for rho in p["penalty_rhos"]:
                            x = self.solve(vg, x, float(q), float(rho))
                        r = self.record(x, U, L, U0) | {"q_target": float(q)}
                        rows.append(r)
                        if r["dU"] > stop_dU:
                            break
        rows.sort(key=lambda r: r["Q"])
        return rows

    def region_boundary(self, U, L, U0, budget):
        """Extreme points of the in-budget set projected on (dA, dC): for each direction
        theta, minimise U - mu f_theta and bisect mu so that dU equals the budget."""
        rc = self.cfg["region"]

        def f(x, ct, st):
            return (ct * (self.A(x) - self.A0) + st * (self.C(x) - self.C0)) / self.scale

        vg = jax.jit(jax.value_and_grad(lambda x, ct, st, mu: U(x) - mu * f(x, ct, st)))
        dU = lambda z: float(U(jnp.asarray(z))) - U0
        pts = []
        for th in np.linspace(0.0, 2.0 * np.pi, rc["n_angles"], endpoint=False):
            ct, st = float(np.cos(th)), float(np.sin(th))
            lo, hi, z_lo = 0.0, 1.0, np.zeros(5)
            z_hi = self.solve(vg, z_lo, ct, st, hi)
            while dU(z_hi) < budget and hi < rc["mu_cap"]:
                lo, z_lo, hi = hi, z_hi, hi * 4.0
                z_hi = self.solve(vg, z_lo, ct, st, hi)
            for _ in range(rc["n_bisect"]):
                mid = 0.5 * (lo + hi)
                z_mid = self.solve(vg, z_lo, ct, st, mid)
                if dU(z_mid) <= budget:
                    lo, z_lo = mid, z_mid
                else:
                    hi = mid
            pts.append(self.record(z_lo, U, L, U0) | {"theta": float(th)})
        return pts


def profile_points(curve):
    """One point per target q (the lowest energy found), sorted by Q."""
    best = {}
    for r in curve:
        k = round(float(r["q_target"]), 6)
        if k not in best or r["dU"] < best[k]["dU"]:
            best[k] = r
    return sorted(best.values(), key=lambda r: r["Q"])


def profile_crossings(curve, budget):
    """(lo, hi) where the profile crosses the budget, linearly interpolated."""
    rows = profile_points(curve)
    q = np.array([r["Q"] for r in rows])
    u = np.array([r["dU"] for r in rows])
    idx = np.flatnonzero(u <= budget)
    i_lo, i_hi = int(idx.min()), int(idx.max())
    lo = (float(q[i_lo]) if i_lo == 0 else
          float(np.interp(budget, [u[i_lo], u[i_lo - 1]], [q[i_lo], q[i_lo - 1]])))
    hi = (float(q[i_hi]) if i_hi == len(q) - 1 else
          float(np.interp(budget, [u[i_hi], u[i_hi + 1]], [q[i_hi], q[i_hi + 1]])))
    return lo, hi


def reference_facts(sir, cfg):
    """Peak day, share infected when the lockdown starts, and the effect of f_lock on
    cumulative incidence, all at theta*."""
    theta = sir.theta(jnp.zeros(5))
    run = lambda th: np.asarray(simulate(th, sir.key, sir.T, sir.N, sir.dt, 0.0, sir.kappa))
    inc = run(theta)
    lock_day = int(round(float(theta[3]) * sir.T))
    infected_by_lock = float(theta[2]) * sir.N + float(inc[:lock_day].sum())
    totals = np.array([run(theta.at[4].set(f)).sum() for f in np.linspace(*cfg["f_lock_sweep"])])
    return {"peak_day": int(np.argmax(inc)), "lockdown_day": lock_day,
            "infected_by_lockdown_fraction": infected_by_lock / sir.N,
            "f_lock_cumulative_incidence_rel_range": float((totals.max() - totals.min()) / inc.sum())}


def compute():
    cfg = load_config(NAME)
    sir = MeanFieldSIR(cfg["model"])
    cf = Counterfactual(sir, cfg)
    z0 = jnp.zeros(5)
    budget = cfg["budget_nats"]
    q_grid = np.unique(np.concatenate([np.linspace(*s) for s in cfg["profile"]["q_grid"]]))
    stop_dU = cfg["profile"]["stop_factor"] * budget

    out = {}
    for name, m_fn in designs(sir, cfg).items():
        y = m_fn(z0)
        L = lambda z, m_fn=m_fn, y=y: 0.5 * jnp.sum((m_fn(z) - y) ** 2)
        U = lambda z, L=L: sir.w * L(z) + 0.5 * jnp.sum(jnp.asarray(z) ** 2)
        U0 = float(U(z0))
        J = jax.jacfwd(m_fn)(z0)
        eig = eigendecompose(jnp.asarray(symmetrize(J.T @ J)))
        lam, V = np.asarray(eig.eigvals), np.asarray(eig.eigvecs)
        curve = cf.profile(U, L, U0, q_grid, stop_dU)
        lo, hi = profile_crossings(curve, budget)
        out[name] = {
            "eigvals": lam.tolist(),
            "d_data": int((lam > 1).sum()),
            "sloppiest_loadings": {PARAM_NAMES[i]: float(V[i, -1]) for i in range(5)},
            "profile": curve,
            "Q_range": {"lo": lo, "hi": hi, "width": hi - lo},
        }
        if name in PANEL_B:
            out[name]["region_boundary"] = cf.region_boundary(U, L, U0, budget)

    width = {k: out[k]["Q_range"]["width"] for k in DESIGNS}
    results = {
        "budget_nats": budget, "C_fit": cf.C0, "A_fit": cf.A0, "Q_fit": cf.C0 - cf.A0,
        "designs": out,
        "narrowing_compliance": width["incidence"] / width["incidence+compliance"],
        "narrowing_prevalence": width["incidence"] / width["incidence+prevalence"],
        "reference": reference_facts(sir, cfg),
    }
    io.save_results(NAME, results, cfg)


def _profile_xy(curve):
    rows = profile_points(curve)
    return np.array([r["Q"] for r in rows]), np.array([r["dU"] for r in rows])


def _polygon(boundary):
    pts = sorted(boundary, key=lambda r: r["theta"])
    xy = np.array([[r["dA"], r["dC"]] for r in pts])
    return np.vstack([xy, xy[:1]])


def _extents(xy):
    """Extent along the diagonal (totals move together) and across it (Q changes)."""
    r2 = np.sqrt(2.0)
    return float(np.ptp((xy[:, 0] + xy[:, 1]) / r2)), float(np.ptp((xy[:, 1] - xy[:, 0]) / r2))


def plot():
    res = io.load_results(NAME)
    plotting.apply_style()
    R = plotting.ROLE
    D, B, Q0 = res["designs"], res["budget_nats"], res["Q_fit"]
    cmap = dict(zip(DESIGNS, plotting.seq_colors(len(DESIGNS))))
    ls = {"incidence": "-", "incidence+prevalence": "--", "incidence+compliance": "-"}

    fig, (a, b) = plt.subplots(1, 2, figsize=plotting.figsize("double", 0.50), layout="constrained")

    xmax = 1.12 * max(D[k]["Q_range"]["hi"] for k in DESIGNS)
    for k in DESIGNS:
        q, dU = _profile_xy(D[k]["profile"])
        sel = q <= xmax
        q, dU = q[sel], dU[sel]
        a.plot(q, dU, ls[k], color=cmap[k], lw=1.8, zorder=4,
               label=f"{LABEL[k]}  —  {D[k]['Q_range']['width']:,.0f} cases wide")
        a.fill_between(q, dU, B, where=(dU <= B), color=cmap[k], alpha=0.17, lw=0, zorder=2)
    a.axhline(B, ls=plotting.LS["ref"], c=R["ref"], lw=1.2, zorder=3)
    a.text(xmax * 0.30, B * 1.04, f"budget: {B:g} nats", va="bottom", ha="left",
           fontsize=7.5, color=R["ref"])
    a.axvline(Q0, ls=plotting.LS["ref"], c=R["truth"], lw=0.9, alpha=0.6, zorder=3)
    a.annotate(f"calibrated fit\n{Q0:.0f} cases averted", xy=(Q0, B * 0.30),
               xytext=(xmax * 0.22, B * 0.46), fontsize=7.5, color=R["truth"], va="center",
               arrowprops=dict(arrowstyle="->", lw=0.8, color=R["truth"]))
    a.set_xlim(0, xmax)
    a.set_ylim(0, B * 1.62)
    a.set_xlabel(r"averted infections $Q$")
    a.set_ylabel(r"profiled posterior energy  $\Delta U$  (nats)")
    a.legend(loc="upper left", fontsize=7.2)
    plotting.panel_label(a, "a")

    polys = {k: _polygon(D[k]["region_boundary"]) for k in PANEL_B}
    lim = 1.16 * max(float(np.abs(p).max()) for p in polys.values())
    for off in (-2000, -1000, 1000, 2000):
        b.plot([-lim, lim], [-lim + off, lim + off], ls=plotting.LS["ref"], c=R["ref"],
               lw=0.6, alpha=0.4, zorder=1)
    b.plot([-lim, lim], [-lim, lim], ls=plotting.LS["ref"], c=R["ref"], lw=1.1, zorder=2)
    b.text(lim * 0.63, lim * 0.63, r"$\Delta C(z^{\circ})=\Delta C(z)$   ($Q$ unchanged)",
           fontsize=7.0, color=R["ref"], ha="center", va="top", rotation=45,
           rotation_mode="anchor")
    for k in PANEL_B:
        xy = polys[k]
        b.fill(xy[:, 0], xy[:, 1], color=cmap[k], alpha=0.28, lw=0, zorder=3)
        b.plot(xy[:, 0], xy[:, 1], "-", color=cmap[k], lw=1.6, zorder=4, label=LABEL[k])
    d = lim * 0.34
    b.annotate("", xy=(-d * 0.70, d * 0.70), xytext=(0.0, 0.0),
               arrowprops=dict(arrowstyle="->", lw=1.3, color=R["truth"]), zorder=6)
    b.text(-d * 0.76, d * 0.78, "Q", fontsize=10, color=R["truth"], ha="right",
           va="bottom", zorder=6)
    par1, perp1 = _extents(polys["incidence"])
    par3, perp3 = _extents(polys["incidence+compliance"])
    b.text(0.035, 0.965,
           f"in-budget region ({B:g} nats)\nshrinkage with compliance\n"
           rf"$\parallel$ along the diagonal: {par1 / par3:.2f}$\times$" "\n"
           rf"$\perp$ across it ($Q$): {perp1 / perp3:.1f}$\times$",
           transform=b.transAxes, fontsize=7.2, va="top", ha="left", color=R["truth"],
           bbox=dict(boxstyle="round,pad=0.34", fc="white", ec="0.8", lw=0.6, alpha=0.92),
           zorder=7)
    b.set_aspect("equal")
    b.set_xlim(-lim, lim)
    b.set_ylim(-lim, lim)
    b.set_xlabel(r"$\Delta C(z)$  —  observed total cases")
    b.set_ylabel(r"$\Delta C(z^{\circ})$  —  counterfactual total")
    b.legend(loc="lower right", fontsize=7.2)
    plotting.panel_label(b, "b")

    io.save_figure(fig, NAME)
    plt.close(fig)


if __name__ == "__main__":
    io.main(compute, plot)
