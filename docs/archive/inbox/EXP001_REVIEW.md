---
title: EXP-001 Review — Analytic linear GGN recovery
status: complete
date: 2026-07-11
authoritative_commit: 7c612df (clean working tree; dirty=false)
baseline_commit: e1a7dc9 (repo reset + vault); prior HEAD e1b3a6f
plan: EXP001_IMPLEMENTATION_PLAN.md (approved-with-revisions)
result: PASS
supports: C01 (linear/affine scope)
---

# EXP-001 Review

Implementation of EXP-001 (analytic linear GGN recovery) plus the minimum
reusable GGN infrastructure, executed exactly as revised. **EXP-001 passes.**

## 0. Authoritative clean-commit provenance

The reproducibility loop is closed on a **clean working tree** at commit
`7c612df` (`feat(geometry): EXP-001 …`, on top of baseline `e1a7dc9`). The
authoritative runs below were produced with `git status --porcelain` empty, so
their `provenance.json` records `git_dirty: false` and the exact commit.

| Run | Run ID (authoritative) | commit | dirty |
|---|---|---|---|
| float64 validation | `outputs/EXP-001/20260711T172324Z_7c612df/` | 7c612df983…050e5 | **false** |
| ~1e13 stress | `outputs/EXP-001/20260711T172342Z_7c612df/` | 7c612df983…050e5 | **false** |

Each `provenance.json` records: exact commit hash; `git_dirty: false`; Python
3.12.13; JAX/jaxlib 0.4.30; numpy 2.4.6; `x64_enabled: true`;
`default_float_dtype: float64`; `executed_command:
"uv run python -m experiments.exp001_linear_ggn"` (plus raw `argv`); and the
full run configuration.

**Historical (superseded, retained) — dirty-tree runs from the initial pass:**
`outputs/EXP-001/20260711T170640Z_e1b3a6f/` (main) and `…170707Z_e1b3a6f/`
(stress); those carry `git_dirty: true`. Kept as artifacts; not authoritative.

Clean-run acceptance metrics are **identical** to the historical runs
(deterministic, seed 0): float64 worst-case `rel_fro(AD GGN, AᵀWA)=0.00e+00`,
`matvec≤7.65e-16`, `‖R‖/‖G‖=0.00e+00`, eigvec angle `≤4.47e-08`, rank/null-space
exact (`null angle ≤3.33e-08`), OPG at fit `0.00e+00`, FD best `8.37e-15`.

## 1. Files changed

All additions; **no** production model / loss / OPG / calibrate code, **no**
canonical vault file, and **no** historical output was modified. `outputs/` is
gitignored.

**New library modules**
- `src/curvature_calib/config.py` — `enable_x64`, `x64_enabled`, `require_x64`.
- `src/curvature_calib/geometry/__init__.py`, `geometry/ggn.py` — GGN core.
- `src/curvature_calib/metrics.py` — scale-aware comparison metrics.
- `src/curvature_calib/provenance.py` — run metadata + JSON writer.
- `src/curvature_calib/benchmarks/__init__.py`, `benchmarks/linear_gaussian.py`.

**New experiment**
- `experiments/__init__.py`, `experiments/exp001_linear_ggn.py` (`run`, `run_stress`).

**New tests**
- `tests/conftest.py` (x64 scoping fixture — see Deviation D2),
  `tests/test_config.py`, `tests/test_metrics.py`,
  `tests/test_linear_gaussian.py`, `tests/test_provenance.py`,
  `tests/test_ggn.py`.

**Docs**
- `inbox/EXP001_IMPLEMENTATION_PLAN.md` updated with the seven decisions;
  `inbox/EXP001_REVIEW.md` (this file). No writes under `docs/` or the vault.

## 2. Test results

Full suite: **145 passed, 0 failed** (`uv run pytest -q`, exit 0, 2:27).
= 75 pre-existing skeleton tests + 70 new. The pre-existing float32 tests
(BH / SIR / calibrate / MMD) are unaffected (see D2).

New tests by module (all green):
- `test_ggn.py` (48) — the EXP-001 acceptance harness.
- `test_metrics.py` (8), `test_linear_gaussian.py` (6),
  `test_config.py` (3), `test_provenance.py` (4).

Isolation checks run during debugging:
- `test_calibrate.py` alone (native float32): 6/6 pass.
- The full-suite failure seen mid-implementation was **contamination I
  introduced** (global x64), now fixed (D2); re-run is clean.

## 3. Acceptance metrics

From the authoritative clean runs
`outputs/EXP-001/20260711T172324Z_7c612df/metrics.json` (main, seed 0) and
`…172342Z_7c612df/` (stress). Values are identical to the historical dirty-tree
runs (deterministic).

**float64 — ordinary validation grid** `P∈{5,20,100}`, `cond(AᵀA)∈{1,1e2,1e6}`,
full & rank-deficient, 18 cells; worst-case over cells:

| Metric | Gate | Worst (float64) | Verdict |
|---|---|---|---|
| A1 `rel_fro(AD GGN, AᵀWA)` | ≤1e-10 | **0.00e+00** | ✅ |
| A2 matrix-free `Gv` vs dense (rel ℓ²) | ≤1e-10 | **7.65e-16** | ✅ |
| A3 `rel_fro(H, G)` = ‖R‖/‖G‖ (affine⇒0) | ≤1e-10 | **0.00e+00** | ✅ |
| A5 numerical rank vs `rank(A)` | equal | equal (all cells) | ✅ |
| A5 null-space principal angle | ≤1e-6 | **3.33e-08** | ✅ |
| A6 OPG ‖·‖ at fit / ‖G‖ | ≈0 | **0.00e+00** | ✅ |
| A6 OPG rank off-fit | =1 | **1** | ✅ |
| eigenvector max principal angle | ≤1e-6 | **4.47e-08** | ✅ |

**Weight semantics** (Decision 5), P5/cond1e2/full/float64 — `rel_fro=0.00e+00`
and `matvec≤2.4e-16` for `weight ∈ {None, diagonal, dense SPD}`; all three are
also swept across the `test_ggn` grid.

**A4 — finite-difference (separate, looser regime, Decision 3).** Step-size
study, P5/cond1/float64:

| step h | 1e-2 | 1e-3 | 1e-4 | 1e-5 | 1e-6 |
|---|---|---|---|---|---|
| `rel_fro(H_fd, AᵀWA)` | 8.37e-15 | 7.24e-14 | 1.32e-12 | 4.21e-12 | 8.42e-11 |

Best = **8.37e-15 ≤ 1e-5** ✅. The AD `1e-10` bound is not applied to FD.

**A8 — float32 recorded, not gated.** Worst over the same 18 cells:
`rel_fro_ggn=0.00e+00`, `matvec=7.42e-07`, `eigvec angle=8.46e-04`. Recorded
only; never defines acceptance.

**Ill-conditioning stress (`~1e13`, non-gating, Decision 1).** P20:
- float64: `rel_fro_ggn=0.00e+00`, eigvec angle `2.11e-08`; the **numerical rank
  under the stated tolerance** (`rtol=1e-10`) is **15/20**, because the smallest
  σ²≈1e-13·λ_max fall below that threshold. This reflects the numerical
  tolerance, not a change in the mathematical rank of `AᵀWA`; it characterizes
  the sloppy-tail floor and is expected, not a failure.
- float32: eigvec angle `3.45e-04`; FD best `1.74e-06` vs float64 `6.83e-15` —
  the precision/conditioning degradation the CLAUDE.md invariant warns about.

## 4. Provenance path

**Authoritative (clean tree, commit `7c612df`, `dirty:false`):**
- Main run: `outputs/EXP-001/20260711T172324Z_7c612df/`
  (`config.json`, `provenance.json`, `metrics.json`, `arrays.npz`,
  `figures/fig01_linear_benchmark.{png,pdf}`).
- Stress run: `outputs/EXP-001/20260711T172342Z_7c612df/`.

**Historical (dirty tree, retained, not authoritative):**
- `outputs/EXP-001/20260711T170640Z_e1b3a6f/` (main),
  `…170707Z_e1b3a6f/` (stress), both `dirty:true`.

The authoritative `provenance.json` records git commit
`7c612df983feb7970bda7efc64456e59f01050e5`, `git_dirty:false`, jax/jaxlib 0.4.30,
numpy 2.4.6, python 3.12.13, `x64_enabled:true`, `default_float_dtype:float64`,
`executed_command:"uv run python -m experiments.exp001_linear_ggn"`, `argv`,
seeds, and the full config. `outputs/` is gitignored; nothing was written to
`docs/` or the Results Ledger.

## 5. Deviations from the plan

- **D1 — Two test-design corrections (no library change).**
  (a) An initial `test_affine_hessian_equals_ggn` compared `rel_fro(H−G, G)`
  (wrong arg order) instead of `rel_fro(H, G)`; corrected. (b) The FD test
  initially assumed a **U-shaped** error(h) curve; for a purely quadratic loss
  the central difference has **no truncation error**, so the curve is
  round-off-monotone (grows as h→0). The test now asserts round-off growth and
  measures conditioning sensitivity via the **smallest-eigenvalue** error
  (which is where ill-conditioning actually bites), consistent with required
  check #7. This is a factual correction to a plan assumption, documented here.
- **D2 — Added `tests/conftest.py` (not in the plan's file list).** x64 is a
  process-global JAX flag; enabling it at import of the geometry-test modules
  leaked float64 into the historical float32 tests and perturbed the stochastic
  `test_calibrate` threshold. The conftest fixture scopes x64 to the four
  geometry-test modules and restores the prior value after each test, so the
  historical suite runs untouched. This realizes Decision 6 (enable+verify at
  the entry point; no module-level significant arrays) in the test layer.
- **D3 — `residual_curvature` is inline, not a function.** Per revised Decision
  4, `R=H−G` is computed at the experiment/test layer, not exposed as API.
- **D4 — Provenance writer named `write_json`** with a `write_metadata` alias
  (plan referenced `write_metadata`). Cosmetic.
- No other deviations. Condition grid, run-record layout, minimal 5-function
  API + thin `ggn_linear_operator` wrapper, weight semantics, and the stress
  case all match the revised plan.

## 6. Did EXP-001 pass?

**Yes.** All gated acceptance criteria (A1–A8) are met in float64 on the
ordinary validation grid, with the FD and float32/stress cases handled under
their own non-gating regimes exactly as specified. All eight required EXP-001
checks are verified:

1. analytic GGN = `AᵀWA` — `test_analytic_ggn_equals_AtWA_none` ✅
2. AD GGN = analytic GGN — `rel_fro=0.00e+00` ✅
3. matrix-free `Gv` = dense `Gv` — `≤7.65e-16` ✅
4. exact Hessian = GGN for affine `m` — `‖R‖/‖G‖=0.00e+00` ✅
5. rank & null-space recovery — rank exact, null angle `≤3.33e-08` ✅
6. OPG zero at fit, GGN nonzero (and OPG rank-1 off-fit) ✅
7. FD accuracy varies with step size and conditioning ✅
8. float32 recorded, non-gating ✅

## 7. Exact evidence for C01

> **C01** — "AD reproduces analytic GGN matrices on controlled models."

EXP-001 establishes C01 **for the linear/affine controlled model**: the
explicit-Jacobian AD GGN `ggn_dense` equals the analytic `AᵀWA` to
`0.00e+00` relative Frobenius error (bit-identical, since the AD Jacobian of a
linear map is `A` exactly) across `P∈{5,20,100}`, `cond∈{1,1e2,1e6}`,
`weight∈{None,diagonal,dense SPD}`, and both full-rank and rank-deficient `A`;
the matrix-free product agrees with the dense matrix to `≤7.65e-16`; and the
exact Hessian equals the GGN (`R=0`) to `0.00e+00`, confirming the affine
special case of `H=G+R`.

**Scope / what this does NOT yet establish (C01 remains "proposed").** This is
the analytic-benchmark leg only. C01 in full also requires the nonlinear
(EXP-002) and MMD (EXP-003) evidence; those are not run here. The linear result
is necessary and now satisfied; it is not by itself sufficient for the unscoped
claim. No abstract/conclusion wording is licensed by this run beyond the
linear-benchmark statement.

## 8. Stopping point

Per instruction, work stops here: **no** EXP-000, EXP-002, or production-model
refactoring is begun. The tracked blocker (rename/redocument the historical
per-seed OPG before EXP-000 or any BH/SIR rerun) is recorded at the top of
`EXP001_IMPLEMENTATION_PLAN.md` and remains open.
