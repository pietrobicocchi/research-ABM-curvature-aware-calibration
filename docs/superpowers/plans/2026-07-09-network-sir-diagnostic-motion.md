# Network-SIR Diagnostic-in-Motion Suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reproduce the Brock–Hommes viz "diagnostic in motion", "subspace rotation", and "falsification" experiments for the **network-SIR** model, plus a reference memory doc on ARNAU 2025, and a structured MD log.

**Architecture:** A parallel shared data loader `scripts/viz/_netsirdata.py` (analog of `_bhdata.py`) runs three R₀-regime calibrations recording per-iterate OPG diagnostics into `outputs/viz/_cache/sir_motion.npz`. Three figure scripts consume it (motion, subspace rotation) or re-derive the eigenbasis at θ\* (falsification), all in the `curvature_calib.viz.paper_style` idiom. A Sonnet subagent writes the reference memory doc in parallel. A final MD log ties it together.

**Tech Stack:** Python 3.12, JAX (float64), matplotlib, `uv run`, existing `curvature_calib` package (unchanged).

## Global Constraints

- JAX float64 must be enabled at the top of every script: `jax.config.update("jax_enable_x64", True)` **before** importing `jax.numpy`.
- Do **not** modify anything under `src/curvature_calib/`. These are viz/docs deliverables only.
- Model: `curvature_calib.models.network_sir.simulate` (Gumbel-Sigmoid surrogate, τ=0.5, fixed graph_seed). P=5, θ = (β, γ, I0_frac, t_lock_norm, f_lock).
- Interventionist θ\* base: γ=0.10, I0_frac=0.01, t_lock_norm=0.025, f_lock=0.50; β set per R₀ regime via R₀ = β·⟨k⟩/γ, ⟨k⟩=mean_degree=6.
- R₀ regimes: `slow` R₀=1.3 (β≈0.0217), `moderate` R₀=2.5 (β≈0.0417), `fast` R₀=5.0 (β≈0.0833). Regime order `["slow","moderate","fast"]` fixed across all figures.
- Trimmed runtime: N=250, T=200, M=64, M_REF=96, n_iter=40, N_BOOT=300, REL_FLOOR=1e-8.
- Framing rule ([[framing_kunstner_opg_not_fisher]]): F̂ is the "OPG matrix", never "empirical Fisher".
- Figure idiom: `from curvature_calib.viz import paper_style as ps`; call `ps.setup()`; save via `ps.save(fig, "name")` (writes `outputs/viz/name.{pdf,png}`).
- CalibLog fields available from `calibrate(...)`: `log.eigvals` (n_iter,P descending), `log.eigvecs` (n_iter,P,P), `log.per_seed_grads` (n_iter,M,P).

---

### Task 1: ARNAU 2025 reference memory doc (delegated, parallel)

**Files:**
- Create: `docs/memory/reference_quera_bofarull_2025_ad_abm.md`
- Modify: `docs/memory/MEMORY.md` (add one index line under a References heading)

**Interfaces:**
- Consumes: nothing from other tasks (fully independent; run in parallel with Tasks 2–6).
- Produces: a `[[reference_quera_bofarull_2025_ad_abm]]` memory the MD log (Task 6) links to.

- [ ] **Step 1: Dispatch a Sonnet subagent** with this task prompt:

> Read arXiv:2509.03303 ("Automatic Differentiation of Agent-Based Models", Quera-Bofarull, Bishop, Dyer, Jarne Ornia, Calinescu, Farmer, Wooldridge, 2025) in full — fetch the HTML at https://arxiv.org/abs/2509.03303 and, for detail, the full text at https://arxiv.org/pdf/2509.03303. Write a `reference`-type memory file at `docs/memory/reference_quera_bofarull_2025_ad_abm.md` with frontmatter (`name: reference-quera-bofarull-2025-ad-abm`, a one-line `description`, `metadata.type: reference`). Body must cover: (1) full title + authors + venue; (2) core contribution — AD through ABM simulators makes gradients available, enabling variational-inference calibration + first-order sensitivity analysis; (3) the three demo ABMs (Axtell firms, Sugarscape, SIR) and headline results; (4) how they estimate gradients through discrete agent decisions (surrogates / reparameterisation) — capture whatever the paper actually says; (5) a "Relation to our project" section: they use **first-order** gradient sensitivities and VI; we use the **second-moment / OPG eigenstructure** of the same gradients to expose parameter *combinations* and identifiability (sloppiness). CRITICAL framing rule: call F̂ the "OPG matrix", never "empirical Fisher". Note the distinction from other Quera-Bofarull works we cite (2023 AAMAS "Don't Simulate Twice"; the first-order Jacobian-sensitivity §5.4 reference). Then add one line to `docs/memory/MEMORY.md` under a "## Reference material" heading: `- [Quera-Bofarull 2025 — AD of ABMs](reference_quera_bofarull_2025_ad_abm.md) — canonical diff-ABM reference (AD→VI calibration + first-order sensitivity); we add the second-moment/OPG identifiability view.` Keep the file focused and factual.

Use `subagent_type: general-purpose`, `model: sonnet`, `run_in_background: true`.

- [ ] **Step 2: Verify on completion** the file exists and obeys the framing rule:

Run: `test -f docs/memory/reference_quera_bofarull_2025_ad_abm.md && grep -ci "empirical fisher" docs/memory/reference_quera_bofarull_2025_ad_abm.md`
Expected: file exists; count is `0` (no "empirical fisher"). If count > 0, edit those lines to "OPG matrix".

- [ ] **Step 3: Commit**

```bash
git add docs/memory/reference_quera_bofarull_2025_ad_abm.md docs/memory/MEMORY.md
git commit -m "docs(memory): reference doc for Quera-Bofarull 2025 (AD of ABMs)"
```

---

### Task 2: Shared network-SIR motion data loader

**Files:**
- Create: `scripts/viz/_netsirdata.py`
- Cache output: `outputs/viz/_cache/sir_motion.npz`

**Interfaces:**
- Consumes: `curvature_calib.models.network_sir.simulate`, `curvature_calib.calibration.calibrate.calibrate`, `curvature_calib.calibration.per_seed_grads.vmap_simulate`, `curvature_calib.calibration.bootstrap.{bootstrap_eigvals,eigenvalue_cis}`.
- Produces: `load(force=False) -> dict[str, dict]` keyed by regime (`slow`/`moderate`/`fast`); each inner dict has keys `eigvals_traj (n_iter,P)`, `eigvecs_traj (n_iter,P,P)`, `V_T (P,P)`, `tau (n_iter,)`, `d_eff (n_iter,) int`, `boot_lo (n_iter,P)`, `boot_hi (n_iter,P)`, `eigvals_T (P,)`. Also module constants `REGIME_ORDER`, `REGIMES` (dict regime→θ\* jnp array), `PARAM_NAMES`, `P=5`. Consumed by Tasks 3 and 4.

- [ ] **Step 1: Write the loader** (mirrors `scripts/viz/_bhdata.py`; note `_sim` wraps `network_sir.simulate` with trimmed N and the stiff δ init):

```python
"""Shared network-SIR calibration data for the in-motion (fig7) and
subspace-rotation (fig7b) figures.

Runs three R0-regime calibrations of the network-SIR model at the
interventionist theta* (lockdown fires early, during the epidemic), recording
the live diagnostic at every iterate: bootstrap noise floor tau_t and effective
dimension d_eff(t). Cached to outputs/viz/_cache/sir_motion.npz.
"""
from __future__ import annotations

from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from curvature_calib.calibration.bootstrap import bootstrap_eigvals, eigenvalue_cis
from curvature_calib.calibration.calibrate import calibrate
from curvature_calib.calibration.per_seed_grads import vmap_simulate
from curvature_calib.models.network_sir import simulate

N, MEAN_DEG, T, M, M_REF, N_ITER, N_BOOT = 250, 6.0, 200, 64, 96, 40, 300
GAMMA, I0, TLOCK, FLOCK = 0.10, 0.01, 0.025, 0.50
REL_FLOOR = 1e-8

PARAM_NAMES = [r"$\beta$", r"$\gamma$", r"$I_0$", r"$t_{\rm lock}$", r"$f_{\rm lock}$"]
REGIME_ORDER = ["slow", "moderate", "fast"]
P = 5

def _theta_star(R0: float) -> jnp.ndarray:
    beta = R0 * GAMMA / MEAN_DEG
    return jnp.array([beta, GAMMA, I0, TLOCK, FLOCK], dtype=jnp.float64)

REGIMES = {"slow": _theta_star(1.3), "moderate": _theta_star(2.5), "fast": _theta_star(5.0)}
# Stiff-direction init: perturb along v1 ~ (I0, beta) so there is descent signal
# (a sloppy/random init lands at the MMD noise floor; see state.md BH note).
DELTA = jnp.array([0.004, 0.0, 0.004, 0.0, 0.0], dtype=jnp.float64)
REGIME_SEEDS = {"slow": 52, "moderate": 53, "fast": 54}

CACHE = Path("outputs/viz/_cache/sir_motion.npz")


def _sim(theta, key):
    return simulate(theta, key, T=T, N=N, mean_degree=MEAN_DEG, surrogate="gumbel")


def _run_regime(theta_star, name):
    theta0 = theta_star + DELTA
    ref_keys = jax.random.split(jax.random.PRNGKey(0), M_REF)
    Y_ref = vmap_simulate(_sim, theta_star, ref_keys)
    log = calibrate(_sim, theta0, Y_ref, M=M, n_iter=N_ITER,
                    init_damping=100.0, verbose=False, seed_base=REGIME_SEEDS[name])

    eigvals_traj = np.asarray(log.eigvals)
    eigvecs_traj = np.asarray(log.eigvecs)
    V_T = eigvecs_traj[-1]
    psg = np.asarray(log.per_seed_grads)

    n_iter = eigvals_traj.shape[0]
    pfloor = np.empty(n_iter)
    d_eff = np.empty(n_iter, dtype=int)
    boot_lo = np.empty_like(eigvals_traj)
    boot_hi = np.empty_like(eigvals_traj)
    for t in range(n_iter):
        bkey = (REGIME_SEEDS[name] + 1000 if t == n_iter - 1
                else REGIME_SEEDS[name] * 100 + t)
        boot = bootstrap_eigvals(jnp.asarray(psg[t]), n_boot=N_BOOT,
                                 key=jax.random.PRNGKey(bkey))
        cis = np.asarray(eigenvalue_cis(boot))
        boot_lo[t], boot_hi[t] = cis[:, 0], cis[:, 1]
        pfloor[t] = REL_FLOOR * max(eigvals_traj[t, 0], 1e-300)
        d_eff[t] = int(np.sum(boot_lo[t] > pfloor[t]))

    return dict(eigvals_traj=eigvals_traj, eigvecs_traj=eigvecs_traj, V_T=V_T,
                tau=pfloor, d_eff=d_eff, boot_lo=boot_lo, boot_hi=boot_hi,
                eigvals_T=eigvals_traj[-1])


def load(force: bool = False) -> dict:
    if CACHE.exists() and not force:
        raw = np.load(CACHE)
        out = {}
        for name in REGIME_ORDER:
            out[name] = {k[len(name) + 1:]: raw[k] for k in raw.files
                         if k.startswith(name + "_")}
        if all("eigvecs_traj" in out[name] for name in REGIME_ORDER):
            return out
        print("  cache missing eigvecs_traj -> recomputing ...", flush=True)

    results = {}
    for name in REGIME_ORDER:
        print(f"  calibrating {name} regime (R0-scaled) ...", flush=True)
        results[name] = _run_regime(REGIMES[name], name)

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, **{f"{name}_{k}": v
                                  for name, d in results.items()
                                  for k, v in d.items()})
    return results


if __name__ == "__main__":
    load(force=True)
    print(f"cached -> {CACHE}")
```

- [ ] **Step 2: SMOKE run first** — temporarily shrink cost to validate the pipeline and that the stiff δ produces descent. Edit line `N, MEAN_DEG, T, M, M_REF, N_ITER, N_BOOT = 250, 6.0, 200, 64, 96, 40, 300` to `= 250, 6.0, 200, 32, 48, 5, 50`, then:

Run: `uv run python scripts/viz/_netsirdata.py`
Expected: prints "calibrating slow/moderate/fast", finishes without NaN error, writes the cache. Note whether `eigvals_traj[0,0]/eigvals_traj[-1,0]` moves (descent) — inspect by adding a temporary print or loading the npz. If the loss/eigenvalues are static AND identical across iterates, the init is too sloppy: increase `DELTA` I0/β components (e.g. `[0.008,0,0.006,0,0]`) and re-run. Do not proceed until at least the `moderate` regime shows movement in `eigvals_traj`.

- [ ] **Step 3: Restore full parameters** — revert the Step-2 edit back to `= 250, 6.0, 200, 64, 96, 40, 300`, delete the stale smoke cache, and do the real cached run:

```bash
rm -f outputs/viz/_cache/sir_motion.npz
uv run python scripts/viz/_netsirdata.py
```
Expected: `cached -> outputs/viz/_cache/sir_motion.npz`. This may take several minutes.

- [ ] **Step 4: Commit**

```bash
git add scripts/viz/_netsirdata.py
git commit -m "feat(viz): network-SIR motion data loader (3 R0 regimes, cached)"
```

---

### Task 3: Figure — network-SIR diagnostic in motion (fig7)

**Files:**
- Create: `scripts/viz/fig7_netsir_motion.py`
- Output: `outputs/viz/fig7_netsir_motion.{pdf,png}`

**Interfaces:**
- Consumes: `_netsirdata.load()` (Task 2), `curvature_calib.viz.paper_style`.
- Produces: the motion figure (no downstream code consumers).

- [ ] **Step 1: Write the figure script** (structurally identical to `scripts/viz/fig2_bh_motion.py`, retargeted to `_netsirdata` and R₀ regime labels):

```python
"""Figure 7 - network-SIR: the diagnostic in motion.

Three columns (slow / moderate / fast R0), shared axes. Top (tall): log10
lambda_k(t) for all P=5 eigenvalues across calibration, rank-ordered single-hue
ramp (stiff dark -> sloppy pale) with a faint bootstrap band per line and the
bootstrap noise floor tau_t drawn as the accent line. Bottom: d_eff(t) step.
"""
from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt

import _netsirdata
from curvature_calib.viz import paper_style as ps

ps.setup()
data = _netsirdata.load()
P = _netsirdata.P
colors = ps.rank_colors(P)
FLOOR = 1e-13

R0_LABEL = {"slow": "slow ($R_0\\!\\approx\\!1.3$)",
            "moderate": "moderate ($R_0\\!\\approx\\!2.5$)",
            "fast": "fast ($R_0\\!\\approx\\!5$)"}

ymax = max(np.nanmax(data[r]["eigvals_traj"]) for r in _netsirdata.REGIME_ORDER)

fig = plt.figure(figsize=(ps.FULL, 4.3))
gs = fig.add_gridspec(2, 3, height_ratios=[3.0, 1.0], hspace=0.10, wspace=0.16)

for c, regime in enumerate(_netsirdata.REGIME_ORDER):
    d = data[regime]
    ev = np.clip(d["eigvals_traj"], FLOOR, None)
    lo = np.clip(d["boot_lo"], FLOOR, None)
    hi = np.clip(d["boot_hi"], FLOOR, None)
    tau = np.clip(d["tau"], FLOOR, None)
    deff = d["d_eff"]
    it = np.arange(ev.shape[0])

    ax = fig.add_subplot(gs[0, c])
    ax.fill_between(it, FLOOR, tau, color="0.5", alpha=0.16, lw=0, zorder=0)
    for k in range(P):
        ax.fill_between(it, lo[:, k], hi[:, k], color=colors[k], alpha=0.13, lw=0, zorder=1)
        ax.plot(it, ev[:, k], color=colors[k], lw=1.2, zorder=3)
    ax.plot(it, tau, color=ps.ACCENT, lw=1.4, zorder=4)
    ax.set_yscale("log")
    ax.set_ylim(FLOOR, ymax * 3)
    ax.set_xlim(it[0], it[-1])
    ax.tick_params(labelbottom=False)
    if c == 0:
        ax.set_ylabel(r"$\lambda_k(t)$")
    else:
        ax.tick_params(labelleft=False)
    ax.text(0.04, 0.96, ps.smallcaps(R0_LABEL[regime]), transform=ax.transAxes,
            fontsize=8.5, va="top", ha="left", color=ps.INK)

    xr = it[-1]
    ys = ev[-1].copy()
    order = np.argsort(-ys)
    log_ys = np.log10(ys)
    min_gap = 0.9
    for rank in range(1, P):
        i_cur, i_prev = order[rank], order[rank - 1]
        if log_ys[i_prev] - log_ys[i_cur] < min_gap:
            log_ys[i_cur] = log_ys[i_prev] - min_gap
    for k in range(P):
        ax.text(xr + 0.8, 10 ** log_ys[k], rf"$\lambda_{{{k + 1}}}$",
                color=colors[k], fontsize=7.5, va="center", ha="left", clip_on=False)
    ax.text(xr + 0.8, tau[-1], r"$\tau_t$", color=ps.ACCENT, fontsize=8,
            va="center", ha="left", clip_on=False)

    axd = fig.add_subplot(gs[1, c], sharex=ax)
    axd.step(it, deff, where="post", color=ps.INK, lw=1.2)
    axd.set_ylim(-0.4, P + 0.4)
    axd.set_yticks(range(0, P + 1, 1))
    axd.set_xlim(it[0], it[-1])
    axd.set_xlabel("calibration iteration")
    if c == 0:
        axd.set_ylabel(r"$d_{\mathrm{eff}}(t)$")
    else:
        axd.tick_params(labelleft=False)

ps.save(fig, "fig7_netsir_motion")
print("saved fig7_netsir_motion")
```

- [ ] **Step 2: Run and verify output**

Run: `uv run python scripts/viz/fig7_netsir_motion.py`
Expected: prints `saved fig7_netsir_motion`; `outputs/viz/fig7_netsir_motion.png` exists (check `ls -la outputs/viz/fig7_netsir_motion.png`). Open the PNG and confirm: three columns, eigenvalue lines fanning across orders of magnitude, red τ_t line, d_eff step panel below.

- [ ] **Step 3: Commit**

```bash
git add scripts/viz/fig7_netsir_motion.py outputs/viz/fig7_netsir_motion.pdf outputs/viz/fig7_netsir_motion.png
git commit -m "feat(viz): fig7 network-SIR diagnostic in motion"
```

---

### Task 4: Figure — network-SIR subspace rotation (fig7b)

**Files:**
- Create: `scripts/viz/fig7b_netsir_subspace_rotation.py`
- Output: `outputs/viz/fig7b_netsir_subspace_rotation.{pdf,png}`

**Interfaces:**
- Consumes: `_netsirdata.load()` (Task 2), `curvature_calib.calibration.diagnostic.principal_angles`, `curvature_calib.viz.paper_style`.
- Produces: the subspace-rotation figure (no downstream consumers).

- [ ] **Step 1: Write the figure script** (structurally identical to `scripts/viz/fig2b_subspace_rotation.py`, retargeted to `_netsirdata` + R₀ labels):

```python
"""Figure 7b - network-SIR: rotation of the identifiable subspace.

Companion to fig7. Plots the cumulative reorientation of the top-k eigenvector
subspace away from its initial geometry,
    Theta_k(t) = largest principal angle between S_k(0) and S_k(t),
so Theta_k(0)=0 for every k. Three columns (slow/moderate/fast R0), linear axis
in degrees. Nested k=1..P-1; solid = identifiable (k<=d_eff), dashed-faint =
into the numerical noise floor (k>d_eff).
"""
from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt

import _netsirdata
from curvature_calib.calibration.diagnostic import principal_angles
from curvature_calib.viz import paper_style as ps

ps.setup()
data = _netsirdata.load()
P = _netsirdata.P
KS = list(range(1, P))
colors = ps.rank_colors(len(KS))

R0_LABEL = {"slow": "slow", "moderate": "moderate", "fast": "fast"}


def drift_series(eigvecs_traj: np.ndarray, k: int) -> np.ndarray:
    V0 = eigvecs_traj[0][:, :k]
    n = eigvecs_traj.shape[0]
    out = np.empty(n)
    for t in range(n):
        ang = np.asarray(principal_angles(V0, eigvecs_traj[t][:, :k]))
        out[t] = np.degrees(ang.max())
    return out


series = {r: {k: drift_series(data[r]["eigvecs_traj"], k) for k in KS}
          for r in _netsirdata.REGIME_ORDER}
ymax = max(s.max() for r in series for s in series[r].values())

fig = plt.figure(figsize=(ps.FULL, 2.7))
gs = fig.add_gridspec(1, 3, wspace=0.16)

for c, regime in enumerate(_netsirdata.REGIME_ORDER):
    ax = fig.add_subplot(gs[0, c])
    deff = int(data[regime]["d_eff"][-1])
    it = np.arange(data[regime]["eigvecs_traj"].shape[0])
    for i, k in enumerate(KS):
        identified = k <= deff
        ax.plot(it, series[regime][k], color=colors[i],
                lw=1.2 if identified else 1.0,
                alpha=1.0 if identified else 0.5,
                ls="-" if identified else (0, (3, 2)), zorder=3)
        ax.text(it[-1] + 0.8, series[regime][k][-1], rf"$k={k}$",
                color=colors[i], fontsize=7, va="center", ha="left",
                alpha=1.0 if identified else 0.55, clip_on=False)
    ax.set_ylim(-2, ymax * 1.05)
    ax.set_xlim(it[0], it[-1])
    ax.set_xlabel("calibration iteration")
    if c == 0:
        ax.set_ylabel(r"$\Theta_k(t)\ [^\circ]$")
    else:
        ax.tick_params(labelleft=False)
    ax.text(0.04, 0.96,
            ps.smallcaps(R0_LABEL[regime]) + rf"  ($d_{{\mathrm{{eff}}}}={deff}$)",
            transform=ax.transAxes, fontsize=8.5, va="top", ha="left", color=ps.INK)

fig.text(0.5, -0.02,
         r"solid: identifiable subspace ($k\leq d_{\mathrm{eff}}$)"
         r"      dashed: reaches the numerical noise floor ($k>d_{\mathrm{eff}}$)",
         ha="center", va="top", fontsize=7.5, color=ps.INK)

ps.save(fig, "fig7b_netsir_subspace_rotation")
print("saved fig7b_netsir_subspace_rotation")
```

- [ ] **Step 2: Run and verify output**

Run: `uv run python scripts/viz/fig7b_netsir_subspace_rotation.py`
Expected: prints `saved fig7b_netsir_subspace_rotation`; PNG exists. Confirm: three columns, all curves start at 0°, solid lines for k≤d_eff, dashed faint lines climbing for sloppy k.

- [ ] **Step 3: Commit**

```bash
git add scripts/viz/fig7b_netsir_subspace_rotation.py outputs/viz/fig7b_netsir_subspace_rotation.pdf outputs/viz/fig7b_netsir_subspace_rotation.png
git commit -m "feat(viz): fig7b network-SIR subspace rotation"
```

---

### Task 5: Figure — network-SIR falsification (fig7c)

**Files:**
- Create: `scripts/viz/fig7c_netsir_falsification.py`
- Output: `outputs/viz/fig7c_netsir_falsification.{pdf,png}`

**Interfaces:**
- Consumes: `curvature_calib.models.network_sir.simulate`, `curvature_calib.calibration.diagnostic.eigendecompose`, `curvature_calib.calibration.falsification.{moments_difference,acf_difference,quantile_difference}`, `curvature_calib.calibration.per_seed_grads.{per_seed_loss_and_grads,vmap_simulate}`, `curvature_calib.viz.paper_style`.
- Produces: the falsification figure; prints `v_stiff`/`v_sloppy` used by the MD log (Task 6).

- [ ] **Step 1: Write the figure script** (mirrors `scripts/viz/fig6_sir_falsification.py` but on network-SIR at the interventionist `moderate` θ\*, trimmed batch):

```python
"""Figure 7c - Falsification (network-SIR, interventionist lockdown).

Network analog of fig6. theta* = the interventionist moderate-R0 network point
(lockdown fires early, during the epidemic). 2x3 grid: rows = stiff v_1 / sloppy
v_P, columns = moments / ACF sup-norm / tail quantiles, discrepancy vs signed
step alpha. Trimmed batch for the network model's per-sim cost.
"""
from __future__ import annotations

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from curvature_calib.calibration.diagnostic import eigendecompose
from curvature_calib.calibration.falsification import (
    moments_difference, acf_difference, quantile_difference,
)
from curvature_calib.calibration.per_seed_grads import per_seed_loss_and_grads, vmap_simulate
from curvature_calib.models.network_sir import simulate
from curvature_calib.viz import paper_style as ps

ps.setup()

T, N = 200, 250
# interventionist moderate-R0 network theta* (R0 = beta*6/gamma ~ 2.5)
THETA_STAR = jnp.array([0.0417, 0.10, 0.01, 0.025, 0.50], dtype=jnp.float64)
M = 48
N_BATCH = 4
ALPHAS = np.linspace(-0.06, 0.06, 9)


def _sim(theta, key):
    return simulate(theta, key, T=T, N=N, mean_degree=6.0, surrogate="gumbel")


ref_keys = jax.random.split(jax.random.PRNGKey(0), 96)
Y_ref = vmap_simulate(_sim, THETA_STAR, ref_keys)
theta_eval = THETA_STAR + jnp.array([0.004, -0.005, 0.003, 0.005, -0.02], dtype=jnp.float64)
stats = per_seed_loss_and_grads(_sim, theta_eval, jax.random.split(jax.random.PRNGKey(1), 96), Y_ref)
eig = eigendecompose(stats.opg)
v_stiff = eig.eigvecs[:, 0]
v_sloppy = eig.eigvecs[:, -1]
print("v_stiff ", np.asarray(v_stiff).round(3))
print("v_sloppy", np.asarray(v_sloppy).round(3))

CHANNELS = ["moments", "ACF", "quantiles"]


def discrepancies(X_a, X_b):
    return (
        float(np.sum(moments_difference(X_a, X_b))),
        float(acf_difference(X_a, X_b)),
        float(np.sum(quantile_difference(X_a, X_b))),
    )


DIRS = {"stiff": v_stiff, "sloppy": v_sloppy}
curves = {d: np.zeros((len(CHANNELS), N_BATCH, len(ALPHAS))) for d in DIRS}

for b in range(N_BATCH):
    keys = jax.random.split(jax.random.PRNGKey(100 + b), M)
    X_base = np.asarray(vmap_simulate(_sim, THETA_STAR, keys))
    for dname, v in DIRS.items():
        for ai, a in enumerate(ALPHAS):
            X_a = np.asarray(vmap_simulate(_sim, THETA_STAR + float(a) * v, keys))
            for ci, val in enumerate(discrepancies(X_a, X_base)):
                curves[dname][ci, b, ai] = val
    print(f"  batch {b + 1}/{N_BATCH} done", flush=True)

fig, axes = plt.subplots(2, 3, figsize=(ps.FULL, 3.6), sharex=True)
row_for = {"stiff": 0, "sloppy": 1}
row_color = {"stiff": ps.ACCENT, "sloppy": ps.INK}
ylabels = {"moments": r"$\sum_j|\Delta m_j|$",
           "ACF": r"$\|\Delta\,\mathrm{ACF}\|_\infty$",
           "quantiles": r"$\sum_q|\Delta Q_q|$"}

for ci, ch in enumerate(CHANNELS):
    ymax = max(curves[d][ci].max() for d in DIRS)
    for dname in DIRS:
        r = row_for[dname]
        ax = axes[r, ci]
        arr = curves[dname][ci]
        ax.fill_between(ALPHAS, arr.min(0), arr.max(0), color=row_color[dname], alpha=0.15, lw=0)
        ax.plot(ALPHAS, arr.mean(0), color=row_color[dname], lw=1.4)
        ps.rule(ax, x=0.0)
        ax.set_ylim(-0.03 * ymax, 1.08 * ymax)
        if ci == 0:
            tag = r"$v_1$ (stiff)" if dname == "stiff" else r"$v_P$ (sloppy)"
            ax.set_ylabel(tag + "\n" + ylabels[ch], fontsize=8)
        else:
            ax.set_ylabel(ylabels[ch], fontsize=8)
        if r == 1:
            ax.set_xlabel(r"signed step $\alpha$")
        ax.set_xlim(ALPHAS[0], ALPHAS[-1])

heads = {"moments": "moments", "ACF": "autocorrelation", "quantiles": "tail quantiles"}
for ci, ch in enumerate(CHANNELS):
    axes[0, ci].text(0.5, 1.06, ps.smallcaps(heads[ch]), transform=axes[0, ci].transAxes,
                     ha="center", va="bottom", fontsize=8.5, color=ps.INK)

fig.subplots_adjust(hspace=0.16, wspace=0.34)
ps.save(fig, "fig7c_netsir_falsification")
print("saved fig7c_netsir_falsification")
```

- [ ] **Step 2: Run and verify output**

Run: `uv run python scripts/viz/fig7c_netsir_falsification.py`
Expected: prints `v_stiff`/`v_sloppy` vectors, four `batch k/4 done` lines, then `saved fig7c_netsir_falsification`; PNG exists. Confirm the top (stiff v₁) row shows larger discrepancies than the bottom (sloppy v_P) row. Record the printed v_stiff/v_sloppy for the log.

- [ ] **Step 3: Commit**

```bash
git add scripts/viz/fig7c_netsir_falsification.py outputs/viz/fig7c_netsir_falsification.pdf outputs/viz/fig7c_netsir_falsification.png
git commit -m "feat(viz): fig7c network-SIR falsification"
```

---

### Task 6: Structured MD log

**Files:**
- Create: `outputs/viz/SIR_DIAGNOSTIC_LOG.md`

**Interfaces:**
- Consumes: numbers/figures from Tasks 2–5 and the memory doc from Task 1.
- Produces: the human-readable experiment log (terminal deliverable).

- [ ] **Step 1: Extract the headline numbers** from the cache so the log quotes real values (not guesses):

```bash
uv run python -c "
import numpy as np, sys; sys.path.insert(0,'scripts/viz')
import _netsirdata as d
data=d.load()
for r in d.REGIME_ORDER:
    ev=data[r]['eigvals_T']; span=np.log10(ev[0]/max(ev[-1],1e-300))
    print(f'{r}: d_eff_final={int(data[r][\"d_eff\"][-1])}  span={span:.1f} OOM  lambda1={ev[0]:.2e} lambdaP={ev[-1]:.2e}')
"
```
Record the printed lines.

- [ ] **Step 2: Write the log** using the recorded numbers (fill the `<...>` placeholders from Steps 1 and Task 5's printout — do not leave them literal):

```markdown
# Network-SIR identifiability diagnostic — experiment log

Generated 2026-07-09. Companion to the Brock–Hommes viz suite (`fig2`, `fig2b`, `fig4`).
All three experiments below are on the **network-SIR** model at the interventionist θ\*.

## 1. Model and differentiation

Discrete-state SIR on a fixed Erdős–Rényi contact graph (`src/curvature_calib/models/network_sir.py`).
Per-node S→I and I→R transitions are Bernoulli events; force of infection is `foi = A @ I`
with `p_infect = 1 - exp(-beta_eff * foi * dt)`. Discreteness is relaxed by a **Gumbel-Sigmoid**
surrogate (τ=0.5) so the whole trajectory is differentiable in θ = (β, γ, I₀, t_lock, f_lock).
A sigmoid lockdown modulates `beta_eff` around `t_lock_norm`. Gradient-horizon truncation is
available via `stop_gradient` on the pre-window. The OPG matrix F̂ is built from per-seed MMD
gradients; this is **not** an empirical Fisher matrix.

## 2. Operating points (3 R₀ regimes)

R₀ = β·⟨k⟩/γ, ⟨k⟩=6, γ=0.10, I₀=0.01, t_lock_norm=0.025 (early), f_lock=0.50.

| regime | R₀ | β | final d_eff | spectrum span |
|---|---|---|---|---|
| slow | 1.3 | 0.0217 | <slow d_eff> | <slow span> OOM |
| moderate | 2.5 | 0.0417 | <mod d_eff> | <mod span> OOM |
| fast | 5.0 | 0.0833 | <fast d_eff> | <fast span> OOM |

## 3. Experiment A — the diagnostic in motion (`fig7_netsir_motion`)

λ_k(t) across N_ITER=40 calibration iterations, three R₀ columns, with bootstrap bands and the
precision floor τ_t = 1e-8·λ₁. Finding: <one line — e.g. spectrum fans to a sloppy tail; d_eff
settles at N per regime>.

## 4. Experiment B — subspace rotation (`fig7b_netsir_subspace_rotation`)

Θ_k(t) = largest principal angle between S_k(0) and S_k(t). Finding: <one line — e.g.
identifiable subspace k≤d_eff stays near X°, sloppy k>d_eff drifts to ~Y°>.

## 5. Experiment C — falsification (`fig7c_netsir_falsification`)

Signed-α perturbations along v₁ (stiff) vs v_P (sloppy) under three non-MMD channels.
v_stiff = <printed vector>, v_sloppy = <printed vector>. Finding: stiff-direction discrepancies
exceed sloppy-direction by <ratio>× — the sloppy direction is genuinely unfalsifiable by the data.

## 6. Relation to Quera-Bofarull 2025

See [[reference_quera_bofarull_2025_ad_abm]]. That paper uses AD for variational-inference
calibration and **first-order** sensitivity analysis on ABMs including SIR. This suite adds the
**second-moment / OPG eigenstructure** of the same gradients: it exposes the *combinations*
(t_lock×f_lock sloppy direction) and the effective dimension that first-order per-parameter
sensitivities cannot surface.
```

- [ ] **Step 3: Commit**

```bash
git add outputs/viz/SIR_DIAGNOSTIC_LOG.md
git commit -m "docs(viz): structured network-SIR diagnostic experiment log"
```

---

## Self-Review notes

- **Spec coverage:** memory doc → Task 1; `_netsirdata.py` loader (3 R₀ regimes, interventionist θ\*, trimmed) → Task 2; fig7 motion → Task 3; fig7b subspace → Task 4; fig7c network falsification → Task 5; MD log → Task 6. All spec deliverables mapped.
- **Type consistency:** `_netsirdata.load()` return keys (`eigvals_traj`, `eigvecs_traj`, `tau`, `d_eff`, `boot_lo`, `boot_hi`, `eigvals_T`) are produced in Task 2 and consumed with the same names in Tasks 3, 4, 6. `P`, `REGIME_ORDER`, `REGIMES` exported from Task 2 and imported in 3/4. `paper_style` API (`setup`, `rank_colors`, `FULL`, `ACCENT`, `INK`, `smallcaps`, `rule`, `save`) matches `src/curvature_calib/viz/paper_style.py`.
- **Runtime gate:** Task 2 Step 2 smoke run validates the stiff δ before the expensive full run; the δ fallback is spelled out.
- **Framing:** OPG-not-Fisher rule enforced in Task 1 Step 2 grep and stated in the log (Task 6).
```
