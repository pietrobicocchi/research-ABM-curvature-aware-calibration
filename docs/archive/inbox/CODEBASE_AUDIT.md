---
title: Codebase Audit
status: draft-for-review
date: 2026-07-11
author: Claude (onboarding audit, no code changes applied)
authority: subordinate to 02_MATHEMATICAL_SPECIFICATION.md, 07_DECISION_LOG.md, 01_PROJECT_CHARTER.md
---

# Codebase Audit

Audit of the working-tree code against the canonical vault (`docs/01`–`08`).
No canonical files were changed; no code was renamed, deleted, or repaired.
Only reads and static inspection were performed.

## Bottom line (read first)

- **The code does not compute the intended GGN** `G(z) = Dm(z)* W Dm(z)`.
  Its central curvature object is `F_OPG = (1/M) Σ_m g_m g_mᵀ` built from
  per-seed **scalar-loss** gradients — the exact object rejected in
  **DEC-001** and Math-Spec §14. Every downstream module (diagnostic,
  bootstrap, calibrate, preconditioner, falsification) consumes `F_OPG`.
- **No calibrated-representation Jacobian `J_m` is exposed.** The building
  blocks to compute it exist (`jax.jacfwd` is already used in
  `jacobian_sensitivity.py`), but nothing assembles `Ĵ_μ` or `Ĵ_μᵀ W Ĵ_μ`.
- **Coordinates are physical θ, not prior-whitened z.** There is no `T(z)`
  transform, no prior, and no weight matrix `W` anywhere in `src/`. This
  contradicts DEC-003 / DEC-004 and the Math-Spec §1 coordinate convention.
- **The library never enables float64.** The `jax_enable_x64` invariant lived
  only in the now-deleted `scripts/`; `src/` and `tests/` run in float32.
- **Terminology in docstrings is stale and wrong** relative to the vault:
  `F_OPG` is repeatedly called a "GGN approximation of the MMD² Hessian",
  "empirical curvature", and an "identifiability"/"effective-dimension" object.
- The `scripts/` experiment layer (25 scripts) and all figures **are staged
  for deletion in the working tree** (per CLAUDE.md repo-clean). They still
  exist at commit `e1b3a6f`. "Existing experiments" below are reconstructed
  from git history, not from live files.

---

## 1. Repository map

```
research-ABM-curvature-aware-calibration/
├── CLAUDE.md                     project standards (repo-clean note)
├── README.md                     (modified in working tree)
├── pyproject.toml                jax==jaxlib==0.4.30 pin; py3.12; ruff/pytest
├── uv.lock
├── docs/                         CANONICAL VAULT
│   ├── 01_PROJECT_CHARTER.md … 08_PAPER_ARCHITECTURE.md
│   ├── manifest.json
│   ├── memory/                   (gitignored session vault; MEMORY.md index)
│   └── papers/reference_quera_bofarull_2025_ad_abm.md
├── inbox/
│   └── CODEBASE_AUDIT.md         ← this file
├── src/curvature_calib/
│   ├── models/
│   │   ├── brock_hommes.py       BH deviation-form simulator (θ∈R^5)
│   │   ├── sir.py                mean-field smooth-lockdown SIR (θ∈R^5)
│   │   ├── network_sir.py        ER-graph discrete SIR + surrogate transitions
│   │   └── surrogates.py         gumbel_sigmoid, straight_through_bernoulli
│   ├── losses/
│   │   └── mmd.py                unbiased U-statistic MMD² + median heuristic
│   ├── calibration/
│   │   ├── per_seed_grads.py     ← CORE: per-seed scalar-loss grads + F_OPG
│   │   ├── diagnostic.py         eigendecompose, principal_angles, d_eff
│   │   ├── bootstrap.py          bootstrap CIs on F_OPG eigvals/subspace
│   │   ├── falsification.py      stiff/sloppy perturbation discrepancies
│   │   ├── calibrate.py          OPG-preconditioned LM loop
│   │   ├── preconditioner.py     damped (F_OPG+λI)⁻¹ step
│   │   ├── baselines.py          SGD / Adam
│   │   ├── jacobian_sensitivity.py  per-param jacfwd sensitivity (unused by OPG path)
│   │   └── opg.py                back-compat re-export shim
│   └── viz/style.py
├── tests/                        13 test_*.py (float32; no x64)
└── outputs/viz/                  6 fig7* network-SIR figures (staged for deletion)
```

**Main entry points (live, library-level):**
- `per_seed_grads.per_seed_loss_and_grads` — produces `CalibStats(loss, mean_grad, per_seed_grads, opg)`.
- `diagnostic.eigendecompose` — eigensystem of whatever matrix it is handed.
- `calibrate.calibrate` / `baselines.sgd|adam` — optimization loops.
- `falsification.run_falsification` — perturbation protocol.
- Model `simulate(theta, key, …)` functions.

There are **no live script entry points**; the `scripts/` CLI layer is deleted
in the working tree.

---

## 2. Execution paths

All 25 (deleted) experiment scripts share **one** canonical pipeline,
confirmed from `HEAD:scripts/16_sir_diagnostic.py` and the module wiring. It is
uniform across BH / SIR / network-SIR:

```text
THETA_STAR  (physical θ ∈ R^5, hard-coded per script)
  -> [NO parameter transform]         # differentiate directly wrt θ; no T(z), no prior scaling
  -> models.<model>.simulate(θ, key)  # vmap over M seeds -> X : (M, T)
  -> losses.mmd.mmd_sq_with_median_bandwidth(X, Y_ref)   # unbiased U-stat MMD², σ via median heuristic (stop_grad)
  -> [calibrated representation is IMPLICIT]  # no explicit ψ / m(z); MMD acts on raw trajectories
  -> per_seed_grads.per_seed_loss_and_grads   # VJP: g_m = M·(∂MMD²/∂x_m)·(∂x_m/∂θ)
  -> F_OPG = (1/M) Σ_m g_m g_mᵀ               # per-seed SCALAR-loss OPG   ← rejected object (DEC-001)
  -> diagnostic.eigendecompose(F_OPG)         # + bootstrap CIs, d_eff
  -> falsification.run_falsification          # perturb along v_1 (stiff) / v_P (sloppy)
  -> matplotlib figure (outputs/, gitignored except fig7*)
```

Two variant tails branch off the same `F_OPG`:

```text
… per_seed_grads -> F_OPG -> preconditioner.damped_step((F_OPG+λI)⁻¹ g) -> calibrate LM loop      (optimization)
… per_seed_grads -> per_seed_grads (M×P) -> bootstrap.bootstrap_eigvals / bootstrap_subspace_cis  (uncertainty)
```

**Per-experiment mapping (from git history — do not assume these run today):**

| Script (deleted) | Model | Tail of the pipeline |
|---|---|---|
| `16_sir_diagnostic`, `28_bh_regime_diagnostic_suite` | SIR / BH | spectrum + bootstrap + falsification |
| `06_calibration_dashboard`, `07_optimizer_comparison`, `10_phase2_convergence`, `12_adam_lr_sweep` | BH | calibrate/baselines optimization |
| `08_falsification`, `20_merged_falsification`, `06_sir_falsification` | BH/SIR | falsification protocol |
| `09_horizon_bias`, `26_horizon_sensitivity_sir` | BH/SIR | `grad_horizon` truncation sweep |
| `13_jacobian_comparison`, `27_jacobian_comparison_sir` | BH/SIR | `jacobian_sensitivity` vs OPG eigenvectors |
| `18_network_sir_diagnostic`, `23_fig4_network_sir` | net-SIR | surrogate-gradient OPG spectrum |
| `11_stiff_sloppy_decomposition`, `19_trajectory_bootstrap`, `25_eigenvalue_trajectory` | BH | subspace / bootstrap analysis |
| `24_surrogate_comparison` | net-SIR | gumbel vs straight-through |
| `scripts/booklets/*`, `scripts/viz/*` | — | figure/pedagogy generation |

**Coordinate note (applies to every path):** differentiation is wrt physical
θ. The "calibrated representation" `m(z)` of the Math-Spec is never
materialized — MMD is applied directly to raw simulator trajectories, and the
matrix is built from the scalar loss, not from a representation Jacobian.

---

## 3. Mathematical object mapping

Status legend: **correct** / **cond.** (conditionally correct) / **unclear** /
**inconsistent** (with vault) / **obsolete** (comparison object).

| Code symbol / function | File | Shape | Mathematical object | Averaged over | Derivative type | Status |
|---|---|---:|---|---|---|---|
| `per_seed_grads` (`CalibStats.per_seed_grads`) | per_seed_grads.py:78 | (M, P) | per-seed **scalar-loss** gradient `g_m = M·(∂L/∂x_m)(∂x_m/∂θ)` | none (stacked) | exact pathwise AD (VJP) | cond. (correct *as a gradient*; not a Jacobian `A_m`) |
| `mean_grad` | per_seed_grads.py:80 | (P,) | `∇_θ L`, gradient of MMD² | M seeds | exact pathwise AD | **correct** (verified vs `jax.grad`, test l.29) |
| `opg` / `F_hat` | per_seed_grads.py:81; diagnostic.py:27 | (P, P) | `F_OPG=(1/M)Σ g_m g_mᵀ` | M seeds | AD gradients | **inconsistent** — documented as "GGN"/curvature; is the DEC-001 rejected object |
| `loss` `L` | per_seed_grads.py:69 | scalar | unbiased MMD² `L(θ)` | M×N pairs | value | **correct** |
| `eigendecompose` | diagnostic.py:19 | (P,)+(P,P) | symmetric eigensystem | — | linalg | **correct** (generic; correctness depends on input matrix) |
| `opg_from_grads` | diagnostic.py:27 | (P,P) | `(1/M)GᵀG` | M | — | **inconsistent** (same as F_OPG) |
| `effective_dimension` / `d_eff_from_bootstrap` | diagnostic.py:49,54 | int | count of eigvals above floor | — | — | **inconsistent** — labeled identifiable dimension; withdrawn per DEC-001, C17 rejected |
| `principal_angles` | diagnostic.py:33 | (k,) | subspace angles (Björck–Golub) | — | linalg | **correct** |
| `per_param_jacobian_sensitivity` (`J = jacfwd`) | jacobian_sensitivity.py:50 | J:(T,P)→(P,) | trajectory Jacobian `∂x_m/∂θ`, then column norms | M seeds | exact pathwise AD (JVP/jacfwd) | cond. — this is the **only** true Jacobian in the tree; collapsed to norms, not assembled into `Ĵ_μ` |
| `opg_correlation_matrix`, `opg_diagonal_sensitivity` | jacobian_sensitivity.py:57,68 | (P,P)/(P,) | correlations/std from F_OPG | M | — | **inconsistent** (derived from rejected object) |
| `bootstrap_eigvals` / `bootstrap_subspace_cis` | bootstrap.py:13,57 | (n_boot,P)/scalar | resampled F_OPG eigvals/angles | M (resampled) | — | cond. — method sound; object is F_OPG |
| `damped_step` | preconditioner.py:24 | (P,) | `-(F_OPG+λI)⁻¹ g` LM step | — | — | **inconsistent** — treats F_OPG as curvature preconditioner (DEC-008 removes this) |
| `calibrate` (`opgs`, `eigvals`, `eigvecs`) | calibrate.py:99–101 | logs | trajectory of F_OPG spectra | M | AD | **inconsistent** / **obsolete** for curvature claims |
| `run_falsification` (`v_stiff=eigvecs[:,0]`, `v_sloppy=eigvecs[:,-1]`) | falsification.py:119 | — | perturbation along F_OPG eigvectors | M | AD for eig; sim forward | cond. — protocol is model-diagnostic; directions come from F_OPG |
| Exact Hessian `∇²L` | — | — | `H=G+R` (Math-Spec §7) | — | — | **absent** |
| GGN `G=J_mᵀ W J_m` | — | — | primary object (Math-Spec §8) | — | — | **absent** |
| `J_m = D_z m(z)` | — | — | representation Jacobian | — | — | **absent** |
| Weight `W` | — | — | loss metric `W⪰0` | — | — | **absent** (implicit `W=I`, and loss is not in `½‖m−m_y‖²_W` form) |
| Transform `T`, prior `π`, whitened `z` | — | — | Math-Spec §1 coordinates | — | — | **absent** |

---

## 4. MMD audit

`losses/mmd.py` implements the **unbiased U-statistic squared MMD**
(Gretton et al. 2012), diagonal excluded:

```
MMD²_u = [Σ_{i≠j} k(x_i,x_j)]/(M(M−1)) + [Σ_{i≠j} k(y_i,y_j)]/(N(N−1)) − 2[Σ k(x_i,y_j)]/(MN)
```
(`mmd.py:42–52`). Bandwidth σ is the **median heuristic** on pooled pairwise
squared distances (`mmd.py:28`), wrapped in `stop_gradient`
(`mmd.py:62`) so σ is treated as constant during differentiation.

**Correspondence to the spec:** Math-Spec §12–13 defines the MMD GGN via the
**mean-embedding residual** `μ_z−μ_y` and its Jacobian `J_μ=D_z μ_z`, with two
estimators — PSD plug-in `Ĝ_V=Ĵ_μᵀĴ_μ` and cross-seed `Ĝ_U`. **The implemented
loss is neither the population MMD nor the biased empirical embedding-norm; it
is the unbiased U-statistic scalar.** Its computational graph produces a scalar
whose gradient is fed into an outer product — it never forms `A_m=D_zψ(X_m)`,
never averages into `Ĵ_μ`, and never builds `Ĵ_μᵀĴ_μ`. So:

- The MMD **loss value** is a legitimate objective (a valid choice of `L`).
- The MMD **curvature** implemented (`F_OPG`) does **not** correspond to
  `G_MMD=J_μᵀJ_μ`. Per DEC-006 the code must state *which* MMD objective is
  used; currently only the unbiased U-statistic exists, and EXP-000/EXP-003
  additionally call for the biased/empirical variant and a finite-feature
  representation, both **absent**.

**Caveat on the residual argument:** the doc in `per_seed_grads.py:19–21`
claims F_OPG is "a stochastic GGN via the residual structure of MMD." This is
the invalid identification: for a scalar loss `g=Jᵀr` gives `ggᵀ=JᵀrrᵀJ ≠ JᵀJ`
(Math-Spec §14). The claim must be removed.

---

## 5. GGN readiness

**Already available:**
- Exact pathwise VJP/JVP through all three simulators (BH, SIR, net-SIR).
- `jax.jacfwd` of `simulate` wrt θ (`jacobian_sensitivity.py:50`) — this *is*
  the per-seed observable Jacobian `∂x_m/∂θ`, i.e. a valid `A_m` if the feature
  map is `ψ = identity` (finite summaries `m(z)=E[X]`).
- `vmap` seed batching; symmetric eigensolver; principal angles; bootstrap.

**Must be added to compute `J_m`, `G=J_mᵀWJ_m`, and matrix-free `Gv=J_mᵀW(J_mv)`:**
1. **A calibrated representation `ψ` / `m(z)`.** Decide and implement the
   finite representation (DEC-005: start with weighted summaries `S(X)∈R^K` or
   finite kernel/RFF features), so that `m(z)=E_ξ[ψ(X(z,ξ))]` is an explicit
   vector, not an implicit MMD scalar.
2. **Representation Jacobian** `A_m = D_z ψ(X(z,ξ_m))` via `jacfwd`/`jacrev`
   (extend the existing `jacobian_sensitivity` machinery from raw `x` to `ψ`),
   then `Ĵ_μ = (1/M)Σ A_m`.
3. **GGN assembly**: `Ĝ_V = Ĵ_μᵀ W Ĵ_μ` (PSD plug-in) and the cross-seed
   `Ĝ_U = 1/(M(M−1)) Σ_{m≠n} A_mᵀ W A_n` (Math-Spec §13).
4. **Matrix-free `Gv`**: `Ĵ_μᵀ W (Ĵ_μ v)` using JVP for `Ĵ_μ v` and VJP for the
   transpose — needed for EXP-009 and large `dim(m)`.
5. **Weight matrix `W`** plumbed through the loss and GGN (currently `W=I`
   implicit and not even represented).
6. **Coordinates**: a transform `T(z)` and prior so the GGN is built/whitened
   in `z` (DEC-003), or at minimum a documented physical→whitened mapping.
7. **float64** enabled at import in the library, not only in scripts.

**Assessment:** GGN readiness is **partial-to-low**. The differentiation
substrate is in place; the representation, the averaging into a Jacobian, the
GGN contraction, `W`, and the coordinate system are all missing.

---

## 6. Reusability assessment

**Reusable without modification**
- `losses/mmd.py` (valid loss; keep, but see EXP-000/003 needs for biased variant).
- `diagnostic.eigendecompose`, `principal_angles` (object-agnostic linalg).
- `models/*.simulate` forward passes (BH, SIR, net-SIR) and `surrogates.py`.
- `bootstrap.py` resampling machinery (object-agnostic once fed a valid matrix).

**Reusable after renaming / documentation**
- `per_seed_grads.py`: keep `mean_grad`/`per_seed_grads`; **rename** `opg`→
  something like `F_scalar_opg` and rewrite the "GGN"/curvature docstring to the
  Math-Spec §14 comparison-object framing.
- `diagnostic.opg_from_grads`, `jacobian_sensitivity.opg_*`: retain as explicit
  comparison objects; strip curvature/identifiability language.
- `effective_dimension` / `d_eff_from_bootstrap`: retain only as descriptive
  statistics of F_OPG; remove any identifiable-dimension interpretation (C17 rejected).

**Reusable after mathematical correction**
- The **GGN path itself** must be *built* (see §5); `jacobian_sensitivity.py`'s
  `jacfwd` is the seed to extend.
- `falsification.py`: sound as a model-intrinsic perturbation test, but the
  perturbation directions must come from the corrected GGN (or be explicitly
  labeled as F_OPG-derived comparisons).

**Should be archived (per decisions, not deleted yet)**
- `calibrate.py` + `preconditioner.py`: OPG-preconditioned optimization is
  removed from the MVP (DEC-008) and rests on the rejected curvature
  identification. Archive as historical/appendix material.
- `outputs/viz/fig7*`: superseded OPG-era figures (RES-000 superseded).

**Unclear pending tests**
- Whether `ψ=identity` (raw trajectory) is an acceptable finite representation
  for BH/SIR, or whether summaries/RFF are required for a well-conditioned GGN.
- Numerical behavior of `Ĝ_U` (indefinite at finite M) on these models.

---

## 7. Risks (likely sources of silent error)

1. **Coordinate mismatch (highest).** All derivatives are wrt physical θ, but
   the vault's geometry, prior-whitening, and eigenvalue interpretation
   (`λ≷1`) are defined in `z`. Any spectrum reported now is in unit-dependent
   physical coordinates and is not the prior-relative object (DEC-003/004).
2. **float32 corruption of the sloppy tail.** Library/tests never set
   `jax_enable_x64`; CLAUDE.md warns SIR condition numbers reach ~10¹³ and
   float32 destroys the small eigenvalues. Any GGN/OPG spectrum computed via
   the library defaults is suspect at the tail.
3. **Averaging-order error for the GGN.** The correct estimator averages the
   *Jacobian* first (`Ĵ_μ=E[A_m]`) then contracts; averaging `A_mᵀA_m`
   per-seed instead yields a different (biased-upward) matrix. The current
   `F_OPG` averages *outer products of scalar gradients*, an entirely different
   object. When the GGN is added, the average-then-contract vs
   contract-then-average distinction (`Ĝ_V` vs `Ĝ_U`) must be coded exactly.
4. **Rank-1-per-seed collapse.** `g_m` is a single (P,) vector, so each seed
   contributes rank 1 to F_OPG; the true `A_m` is (D×P) and contributes rank
   up to min(D,P). Interpreting F_OPG rank/`d_eff` as representation rank is
   invalid.
5. **`stop_gradient` on the bandwidth.** `mmd.py:62` freezes σ. Correct for the
   loss gradient, but if a GGN is later derived from an MMD whose σ depends on
   θ, the frozen-σ graph omits a term; must be re-examined per estimator.
6. **Straight-through / Gumbel surrogate bias (net-SIR).** `surrogates.py`
   `custom_jvp` passes `p_dot` straight through (`surrogates.py:42–45`); Gumbel
   is biased for τ>0. Any `G̃` from these is the *surrogate* GGN, not the
   discrete simulator's (Math-Spec §15) — must be labeled and validated, never
   reported as the true geometry.
7. **`grad_horizon` truncation.** BH/SIR/net-SIR `stop_gradient` the pre-horizon
   state; truncated differentiation changes the Jacobian and thus the geometry
   (RQ4/EXP-008). Must be an explicit variable, not a silent default.
8. **Seed coupling in bootstrap.** `bootstrap.py` resamples the M seeds of the
   scalar gradients; this propagates the wrong object's variance and does not
   estimate GGN-estimator uncertainty.
9. **Graph in float32 (net-SIR).** `build_er_graph` casts adjacency to
   `float32` (`network_sir.py:39`); mixed precision inside an otherwise-x64
   pipeline can silently downcast.
10. **NaN/explosion masking in `calibrate`.** Non-finite losses are caught and
    turned into damping increases (`calibrate.py:112`); acceptable for
    optimization but hides regime instability that would corrupt a curvature
    estimate at that θ.

---

## 8. Proposed repository changes (plan only — NOT applied)

### Critical for correctness
1. Add the **calibrated representation `ψ`/`m(z)`** and a **GGN module**
   computing `Ĵ_μ`, `Ĝ_V=Ĵ_μᵀWĴ_μ`, `Ĝ_U`, and matrix-free `Gv` (§5).
2. Add an **exact Hessian** (`jax.hessian` of end-to-end `L`) and the residual
   term `R=H−G` for validation (Math-Spec §7, EXP-002).
3. Introduce **prior-whitened coordinates**: transform `T(z)`, prior, and `W`;
   build/whiten the GGN in `z` (DEC-003/004).
4. Enable **float64** at library import for all diagnostic paths.
5. **Rewrite F_OPG documentation** to the comparison-object framing; delete the
   "GGN/curvature/Fisher/identifiability" claims (DEC-001). Rename `opg`→
   explicit `F_scalar_opg` (or similar) to prevent silent misuse.
   Remove the dangling reference to the deleted
   `docs/memory/framing_kunstner_opg_not_fisher.md` (`per_seed_grads.py:23`).

### Required for reproducibility
6. Restore a **results/metadata convention**: per-run git commit, seeds,
   config, environment, output dir (the Results-Ledger template exists but has
   no producing code).
7. Add **tests that set x64** and assert GGN-vs-analytic and GGN-vs-Hessian
   agreement (currently only F_OPG symmetry/PSD/shape is tested — trivially
   true by construction).
8. Register `EXP-000`/`EXP-001` **code entry points** and output directories.

### Useful cleanup
9. Archive `calibrate.py`, `preconditioner.py`, and `fig7*` outputs into an
   explicitly historical location (DEC-008; do not delete pending review).
10. Reframe `d_eff`/`effective_dimension` as descriptive-only; remove
    identifiability language (C17).

### Optional refactoring
11. Consolidate the `opg.py` shim once callers import from `diagnostic`/`bootstrap`.
12. Unify the `grad_horizon` truncation API and expose it as an experiment knob.

---

## 9. Questions that cannot be answered from the code

1. **Representation `ψ`:** for BH/SIR, is the finite calibrated representation
   the raw trajectory (`ψ=identity`), a set of summaries `S(X)`, or finite
   kernel/RFF features? The vault says "start with finite summaries" (DEC-005)
   but does not fix `ψ` for each model.
2. **Weight `W`:** what metric `W` should the summary-loss use (identity,
   inverse observation covariance, per-summary scaling)?
3. **Transform `T` and prior:** what are the physical→unconstrained maps and
   prior variances for BH `(β,g_1,b_1,g_2,b_2)` and SIR
   `(β,γ,I_0,t_lock,f_lock)`? None are specified in code or vault.
4. **Generalized-posterior weight `w`:** what value/selection rule for the
   learning rate `w` in `π_w` (needed for the `λ≷1` interpretation)?
5. **MMD variant scope:** EXP-000 asks for biased vs unbiased empirical MMD —
   should the biased empirical embedding-norm MMD be implemented, or is the
   finite-summary route the intended EXP-000 representation?
6. **BH reference regimes:** the exact `θ*` and `(R,σ,T)` defining "stable /
   periodic / chaotic" for EXP-000/EXP-004 are hard-coded per (deleted) script;
   which are canonical?
7. **net-SIR fidelity reference:** what is the "high-fidelity derivative
   reference" for EXP-008 against which the surrogate GGN is judged?

---

## Session summary (as requested)

**1. Does the current code compute the intended GGN?**
No. It computes `F_OPG=(1/M)Σ g_m g_mᵀ` from per-seed *scalar-loss* gradients —
the object explicitly rejected by DEC-001 / Math-Spec §14. No
calibrated-representation Jacobian `J_m` and no `G=J_mᵀWJ_m` exist. The
required differentiation primitives (VJP/JVP, and a `jacfwd` of the simulator)
are present, so the GGN is buildable, but it is not built.

**2. Smallest set of changes to run EXP-000** (BH four-matrix audit:
`H_exact`, `G_MMD`, `F_OPG`, `C_g`):
- keep `F_OPG` (have it);
- add `C_g = (1/M)Σ(g_m−ḡ)(g_m−ḡ)ᵀ` (trivial, from existing `per_seed_grads`);
- add `H_exact = jax.hessian` of the end-to-end MMD² wrt the chosen coordinates;
- add `G_MMD`: pick a finite representation `ψ`, form `A_m` via `jacfwd`, average
  to `Ĵ_μ`, contract `Ĵ_μᵀĴ_μ` (this is the real new work);
- enable float64; fix F_OPG terminology so the comparison is stated honestly.

**3. Smallest set of changes to implement EXP-001** (analytic linear
`m(z)=Az`): mostly greenfield but small and simulator-free —
- a generic GGN utility `G=J_mᵀWJ_m` from `jacfwd`/`jacrev` of `m`;
- matrix-free `Gv` via JVP+VJP;
- `H_exact` via `jax.hessian` and a finite-difference Hessian comparator;
- reuse the existing scalar-grad OPG as the comparison object;
- float64 + `W` + P∈{5,20,100}, full/rank-deficient `A`. No model code touched.

**4. Issues serious enough to block further work:**
- (a) The project's central object (the GGN / representation Jacobian) is
  **absent**; everything downstream currently rests on the rejected F_OPG.
- (b) **No coordinate system** (`T`, prior, `W`) — the prior-relative
  interpretation that the whole paper hinges on cannot be computed yet.
- (c) **float64 not enforced** in the library — spectra computed via defaults
  are numerically untrustworthy at the sloppy tail.
- (d) Several open **specification questions** (§9, esp. `ψ`, `T`, priors, `w`)
  must be answered before EXP-000/EXP-001 can be pinned down.

Awaiting review before any implementation.
