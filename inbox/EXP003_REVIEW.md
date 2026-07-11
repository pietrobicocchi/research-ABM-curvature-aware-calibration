---
title: EXP-003 Review — Stochastic MMD GGN estimation
status: complete
date: 2026-07-11
authoritative_commit: ce45c43 (clean tree, dirty=false)
run_id: outputs/EXP-003/20260711T215027Z_ce45c43/
plan: EXP003_IMPLEMENTATION_PLAN.md (approved-with-revisions)
result: PASS — C04, C05, C06 supported conditionally
supports: C04, C05, C06
---

# EXP-003 Review

Full suite **181 passed**. Authoritative run at `ce45c43` (`provenance.json`:
`dirty:false`, x64, float64, command recorded). Deterministic (seed 0).

## Benchmark parameters and reference spectra

Gaussian location-scale `X_z = a(z) + L(z)ε`, prior-scaled `z~N(0,I₅)`; frozen
RFF (γ=2.0, D=512, `feature_seed=0`); `z★=0`.

- **Main regime** `GLSConfig(σ_m=1.0, σ_s=σ_ℓ=0.30)`: `G_ref` eigenvalues
  `[0.1155, 0.1122, 0.0074, 0.0036, 0.0018]`, gap `λ2/λ3 ≈ 15.2` (top-2 = mean subspace).
- **Near-degenerate** `GLSConfig(σ_m=0.5, σ_s=σ_ℓ=1.0)`: `[0.0824, 0.0409, 0.0288,
  0.0279, 0.0196]`, `λ2/λ3 ≈ 1.4` (boundary near-degenerate by design).

## Analytic-reference validation

- closed-form `η(z)` vs 4×10⁵-sample MC: `rel_fro = 2.4e-3` (MC floor);
- `J_η` vs independent central finite differences: `rel_fro = 3.9e-11`;
- `G_ref` symmetric PSD; finite-D `G_ref → G_RBF` (exact Gaussian-RBF kernel via
  mixed 2nd derivative) as D grows: mean `rel_fro` `0.084 (D=64) → 0.031 (256) →
  0.028 (1024)`.

## Measured vs predicted plug-in bias (C05)

Relative plug-in bias decays with M: `0.187, 0.091, 0.053, 0.012, 0.011, 0.011`
for `M=8…256`. The predicted `Ĉ/M` (from `E[Ĝ_V]=G_ref+C/M`) lies **inside the
observed 95% bootstrap CI for M∈{8,16,32,64,128}**; only at `M=256` does the true
bias fall below the `B=200` sampling floor (`predInCI=F` there — a resolution
limit, not a model failure). The bias model is validated wherever the bias is
resolvable.

## Cross-seed bias and indefiniteness (C06)

Cross-seed relative bias is `≤ 0.04` at all M and **compatible with zero**
(‖bias‖ ≤ 2×MC noise floor) at every M — unbiasedness confirmed. Negative-
eigenvalue frequency: **main** `0.94 (M=8) → 0.28 (16) → 0.00 (M≥32)`;
**near-degenerate** identical pattern `0.94 → 0.28 → 0.00`. Reported separately;
no monotone requirement imposed on the near-degenerate regime (it also cleared).

## Leading-subspace recovery (C04, acceptance A3)

Top-2 principal angle of `mean Ĝ_V` vs `G_ref` (degrees): main `1.19, 0.65, 0.39,
0.40, 0.17, 0.20`; near-deg `4.94, 4.03, 1.12, 3.27, 1.06, 1.86`.
**A3 passes:** for `M∈{16,32,64,128,256}` the main regime has relative plug-in
bias ≤0.10 **and** top-2 angle ≤5°. The near-degenerate regime also recovers the
top-2 subspace within 5° at every M≥16 despite `λ2≈λ3` (reported, not gated).

## Residual-sweep OPG behavior (DEC-001)

`z★` fixed ⇒ `G_ref` fixed; only the target varies. `δ=0` (exact match): residual
`0`, `F_OPG_pop = 0` while `G_ref ≠ 0` — the direct counterexample to OPG=GGN.
`δ=0.2, 0.6`: `F_OPG_pop` nonzero and **residual-dependent**; trace-normalized
`rel_fro` vs `G_ref` ≈ `0.94` (strongly different shape). The OPG's top-1 can
align with `G_ref`'s dominant (mean) direction when the residual points along it,
but the OPG remains residual-dependent and shape-distinct — it is not the GGN.

## Fixed-D vs exact-RBF

The finite-D `G_ref` (the primary scoring reference) converges to the exact RBF
population GGN as D grows (0.084 → 0.028); the two roles are kept separate —
estimator bias/variance are always scored against the fixed finite-D `G_ref`.

## Runtime

~30 s for the full run (main + near-degenerate finite-M study at B=200, residual
sweep, D-sweep). B=200 was **not** reduced.

## Claim assessment

- **C04 — MMD GGN consistently estimated from feature Jacobians: supported
  conditionally.** Both estimators recover `G_ref` (top-2 ≤5°, rel bias →0) at
  feasible M on this controlled simulator.
- **C05 — PSD plug-in has acceptable, modeled finite-sample bias: supported
  conditionally.** Bias decays ~1/M and matches the analytic `Ĉ/M` within CIs
  where resolvable.
- **C06 — cross-seed has lower bias but can be indefinite at finite M: supported
  conditionally.** Cross-seed bias compatible with zero at all M; indefiniteness
  present at small M and vanishing by M≥32 in both regimes.

Scope: conditional on the controlled Gaussian location-scale simulator with
frozen RFF, float64. **Not** established for Brock–Hommes, SIR, surrogate/discrete
gradients, or prior-relative posterior agreement (EXP-004/005/008).

## Deviations / failures

- No deviations from the revised plan; no library bugs. The single `predInCI=F`
  at main `M=256` is the expected B=200 resolution floor (true bias below the
  sampling noise), documented above — not a model failure.

## Next

Phase 3 result recorded; EXP-003 accepted pending review. No EXP-000 or ABM work
performed. The OPG-rename blocker remains open.
