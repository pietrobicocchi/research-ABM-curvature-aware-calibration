"""Appendix A - Preconditioning convergence (Brock-Hommes), multi-seed.

Best-so-far validation MMD^2 (top row) and parameter error ||theta_t - theta*||
(bottom row) vs iteration, OPG-preconditioned Levenberg-Marquardt against Adam,
SGD and L-BFGS, faceted by three difficulty levels (init distance along the
stiff direction). Median over N_SEED stochastic seeds with an IQR band; the
summary table reports medians [IQR]. The point the table makes: L-BFGS reaches
the loss floor fastest but lands at the wrong theta, while OPG's curvature
awareness recovers theta accurately - low loss != parameter recovery in a sloppy
landscape.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import scipy.optimize as so
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from curvature_calib.calibration.calibrate import calibrate
from curvature_calib.calibration.baselines import sgd, adam
from curvature_calib.calibration.diagnostic import eigendecompose
from curvature_calib.calibration.per_seed_grads import loss_only, vmap_simulate
from curvature_calib.models.brock_hommes import simulate
from curvature_calib.viz import paper_style as ps

ps.setup()

T, SIGMA, R, M, N_ITER = 200, 0.05, 1.1, 64, 40
N_SEED = 8
THETA_STAR = jnp.array([3.0, 1.2, 0.2, 1.2, -0.2], dtype=jnp.float64)
STIFF_DIR = jnp.array([0.0, 0.0, 1.0, 0.0, 1.0]) / jnp.sqrt(2.0)
DIFFS = {"easy": 0.07, "medium": 0.14, "hard": 0.28}
METHODS = ["OPG (LM)", "Adam", "SGD", "L-BFGS"]
COLORS = {"OPG (LM)": ps.ACCENT, "Adam": "#1f4e79", "SGD": "#4a7fb0",
          "L-BFGS": ps.MUTED}
CACHE = Path("outputs/viz/_cache/figA_multiseed.pkl")


def _sim(theta, key):
    return simulate(theta, key, T=T, sigma=SIGMA, R=R, x_init=0.0)


def best_so_far(v):
    return np.minimum.accumulate(np.asarray(v))


def pad(a, L):
    a = np.asarray(a)
    if len(a) >= L:
        return a[:L]
    return np.concatenate([a, np.full(L - len(a), a[-1])])


def run_lbfgs(theta0, Y_ref, val_keys):
    obj = jax.jit(lambda t: loss_only(_sim, t, val_keys, Y_ref))
    grad_fn = jax.jit(jax.grad(lambda t: loss_only(_sim, t, val_keys, Y_ref)))
    traj = [np.asarray(theta0, dtype=float)]
    so.minimize(
        lambda x: float(obj(jnp.asarray(x))),
        np.asarray(theta0, dtype=float),
        jac=lambda x: np.asarray(grad_fn(jnp.asarray(x)), dtype=float),
        method="L-BFGS-B",
        callback=lambda xk: traj.append(np.asarray(xk, dtype=float)),
        options=dict(maxiter=N_ITER),
    )
    thetas = np.array(traj)
    vloss = np.array([float(obj(jnp.asarray(t))) for t in thetas])
    return thetas, vloss


def iters_to_floor(val):
    floor = max(np.nanmin(val), 1e-12) * 3
    return int(np.argmax(val <= floor)) if np.any(val <= floor) else len(val) - 1


def _compute():
    L = N_ITER + 1
    # results[(diff, method)] = dict(val=(S,L), err=(S,L))
    results = {(diff, m): dict(val=[], err=[]) for diff in DIFFS for m in METHODS}
    conds = {diff: [] for diff in DIFFS}
    finals = {(diff, m): [] for diff in DIFFS for m in METHODS}
    iters = {(diff, m): [] for diff in DIFFS for m in METHODS}

    for diff, d0 in DIFFS.items():
        theta0 = THETA_STAR + d0 * STIFF_DIR
        for s in range(N_SEED):
            ref_keys = jax.random.split(jax.random.PRNGKey(1234 + s), 128)
            Y_ref = vmap_simulate(_sim, THETA_STAR, ref_keys)
            val_keys = jax.random.split(jax.random.PRNGKey(900_000 + s), M)

            logs = {
                "OPG (LM)": calibrate(_sim, theta0, Y_ref, M=M, n_iter=N_ITER,
                                      init_damping=100.0, verbose=False, seed_base=s),
                "Adam": adam(_sim, theta0, Y_ref, M=M, n_iter=N_ITER, lr=1e-2, seed_base=s),
                "SGD": sgd(_sim, theta0, Y_ref, M=M, n_iter=N_ITER, lr=1e-3, seed_base=s),
            }
            for m, log in logs.items():
                err = np.linalg.norm(np.asarray(log.thetas) - np.asarray(THETA_STAR), axis=1)
                val = best_so_far(log.val_losses)
                results[(diff, m)]["val"].append(pad(val, L))
                results[(diff, m)]["err"].append(pad(err, L))
                finals[(diff, m)].append(float(err[-1]))
                iters[(diff, m)].append(iters_to_floor(val))
            th_l, vl_l = run_lbfgs(theta0, Y_ref, val_keys)
            err_l = np.linalg.norm(th_l - np.asarray(THETA_STAR), axis=1)
            val_l = best_so_far(vl_l)
            results[(diff, "L-BFGS")]["val"].append(pad(val_l, L))
            results[(diff, "L-BFGS")]["err"].append(pad(err_l, L))
            finals[(diff, "L-BFGS")].append(float(err_l[-1]))
            iters[(diff, "L-BFGS")].append(iters_to_floor(val_l))

            ev = eigendecompose(logs["OPG (LM)"].opgs[-1]).eigvals
            conds[diff].append(float(ev[0] / max(ev[-1], 1e-30)))
        print(f"  {diff} done ({N_SEED} seeds)", flush=True)

    for k in results:
        results[k]["val"] = np.array(results[k]["val"])
        results[k]["err"] = np.array(results[k]["err"])
    return results, conds, finals, iters


if CACHE.exists():
    results, conds, finals, iters = pickle.loads(CACHE.read_bytes())
else:
    results, conds, finals, iters = _compute()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_bytes(pickle.dumps((results, conds, finals, iters)))


def med_iqr(x):
    x = np.asarray(x, float)
    return np.median(x), np.percentile(x, 25), np.percentile(x, 75)


# ---- figure: convergence curves (median + IQR band) -------------------------
fig, axes = plt.subplots(2, 3, figsize=(ps.FULL, 3.8), sharex=True, sharey="row")
it = np.arange(N_ITER + 1)
for ci, diff in enumerate(DIFFS):
    axv, axe = axes[0, ci], axes[1, ci]
    for m in METHODS:
        V = np.clip(results[(diff, m)]["val"], 1e-12, None)
        E = np.clip(results[(diff, m)]["err"], 1e-6, None)
        axv.fill_between(it, np.percentile(V, 25, 0), np.percentile(V, 75, 0),
                         color=COLORS[m], alpha=0.15, lw=0)
        axv.semilogy(it, np.median(V, 0), color=COLORS[m], lw=1.3)
        axe.fill_between(it, np.percentile(E, 25, 0), np.percentile(E, 75, 0),
                         color=COLORS[m], alpha=0.15, lw=0)
        axe.semilogy(it, np.median(E, 0), color=COLORS[m], lw=1.3)
    axv.text(0.5, 1.05, rf"{diff} ($d_0={DIFFS[diff]:.2f}$)",
             transform=axv.transAxes, ha="center", va="bottom", fontsize=8.5,
             color=ps.INK, style="italic")
    axe.set_xlabel("iteration")
    if ci == 0:
        axv.set_ylabel(r"best-so-far MMD$^2$")
        axe.set_ylabel(r"$\|\theta_t-\theta^*\|$")
    axv.set_xlim(0, N_ITER)

handles = [Line2D([0], [0], color=COLORS[m], lw=1.5, label=m) for m in METHODS]
axes[0, 2].legend(handles=handles, loc="center", fontsize=7.2,
                  handlelength=1.6, labelspacing=0.4)
fig.subplots_adjust(hspace=0.16, wspace=0.28)
ps.save(fig, "figA_preconditioning")
print("saved figA_preconditioning")

# ---- figure: summary table (medians [IQR]) ----------------------------------
figt, axt = plt.subplots(figsize=(ps.FULL, 1.9))
axt.axis("off")
col_labels = ["difficulty", "method", "iters to floor", r"final $\|\theta-\theta^*\|$",
              r"cond $\hat F$"]
rows = []
for diff in DIFFS:
    cmed, _, _ = med_iqr(conds[diff])
    for j, m in enumerate(METHODS):
        im, il, ih = med_iqr(iters[(diff, m)])
        fm, fl, fh = med_iqr(finals[(diff, m)])
        rows.append([
            diff if j == 0 else "",
            m,
            f"{im:.0f} [{il:.0f},{ih:.0f}]",
            f"{fm:.3f} [{fl:.3f},{fh:.3f}]",
            f"{cmed:.1e}" if m == "OPG (LM)" else "",
        ])
tbl = axt.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(7.2)
tbl.scale(1, 1.25)
for (r0, c0), cell in tbl.get_celld().items():
    cell.set_linewidth(0.4)
    cell.set_edgecolor("#cccccc")
    if r0 == 0:
        cell.set_text_props(fontweight="bold")
        cell.set_facecolor("#eef2f6")
ps.save(figt, "figA_table")
print(f"saved figA_table  (N_SEED={N_SEED})")
