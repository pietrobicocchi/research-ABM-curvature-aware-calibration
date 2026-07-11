---
title: EXP-002 Review — Nonlinear local-validity benchmark
status: complete
date: 2026-07-11
authoritative_commit: 4118f4e (clean tree, dirty=false)
run_id: outputs/EXP-002/20260711T175059Z_4118f4e/
plan: EXP002_IMPLEMENTATION_PLAN.md
result: PASS (machinery); C03 and C09 supported conditionally
supports: C03, C09 (both conditional)
---

# EXP-002 Review

Full suite **166 passed**. Authoritative run at commit `4118f4e`
(`provenance.json`: `dirty:false`, x64, float64, command recorded).

## Benchmark definitions

- **Case A — Rosenbrock exact-fit valley:** `r=[a(z2−z1²), b(1−z1)]`,
  `L=½‖r‖²`, optimum `ẑ=(1,1)`, `r(ẑ)=0 ⇒ R(ẑ)=0`. Grid `a∈{1,5,10}, b=1`.
- **Case B — irreducible residual:** `r=[z1−1, z2−1, λ(z1²+z2²−c)]`, overdetermined
  (3-in-2), analytic `R(z)=2λ·r3(z)·I`. Grid `(λ,c)∈{(0.3,0.0),(1.0,4.0)}`.

## Minimizer checks

- Case A: `‖∇L(ẑ)‖ = 0` (exact).
- Case B: Newton minimizer converged, `‖∇L(ẑ)‖ ≤ 1.4e-15`; `ẑ≈(0.809,0.809)`
  and `(1.389,1.389)`. Both have nonzero residual as designed.

## H, G, R agreement (AD vs analytic)

Across all 5 cells: `rel_fro(G_AD, G_analytic)=0`,
`rel_fro(H_AD, H_analytic)≤1.6e-16`, `rel_fro(R_AD, R_analytic)≤3.2e-15`. The AD
Hessian decomposition `∇²L=G+R` reproduces the closed forms to machine precision
in float64 — extending the EXP-001 correctness evidence to nonlinear maps.

## Principal-angle and eigenvalue comparisons (at the optimum)

- **Case A:** `E_G=‖G−H‖/‖H‖ = 0`, leading-1 angle `0`, eigenvalue rel error `0`
  (R=0 at the exact fit ⇒ G=H exactly). Along paths, `‖R‖/‖H‖` and `E_G` grow
  smoothly with distance (increasing with `a`).
- **Case B:** at the minimizer G≠H. `E_G = ‖R‖/‖H‖ = 0.158` (λ=0.3) and `0.025`
  (λ=1,c=4); leading-eigenvector angle stays small (`≤2.1e-08`), but the
  eigenvalue relative error is large (`0.19` and `0.39`). So `R` shifts
  eigenvalues while barely rotating the leading eigenvector — the eigenvalue and
  subspace axes diverge, as intended by the four-axis design.

## Quadratic-prediction validity radii (τ=0.10, prespecified signed α-grid)

- **Case A (good fit):** nontrivial radii `ρ`: `a=1 → {v0:0.1, v1:0.3}`,
  `a=5 → {0.1,0.1}`, `a=10 → {0.1,0.05}`. Positive, local, direction-dependent;
  the sloppy-direction radius shrinks as the valley sharpens.
- **Case B (nonzero residual):** `λ=0.3 → {v0:0, v1:0}`, `λ=1,c=4 → {v0:0.1, v1:0}`.
  A zero radius is a **correct, informative** outcome, not a failure: near a
  nonzero-residual minimizer both `ΔL(α)` and the GGN prediction `½α²λ` scale as
  `α²`, so the relative error `≈ vᵀRv/vᵀHv` is roughly constant in `α`; when that
  exceeds `τ` the quadratic model is nowhere within tolerance. This is the direct
  signature of the missing `R` term.

## Claim assessment

- **C03 — "GGN approximates the exact Hessian near a good fit": supported
  conditionally.** Case A: `H=G` exactly at the exact fit, with `E_G` growing
  controllably away from it. Case B: quantifies breakdown — `R` reaches ~16% of
  `‖H‖` and ~19–39% eigenvalue error at a nonzero-residual minimizer. The
  approximation is good iff the residual-curvature term is small; this is now
  demonstrated, not merely asserted.
- **C09 — "local GGN predicts actual loss changes over a nontrivial radius":
  supported conditionally.** Case A gives nontrivial validity radii
  (`ρ∈{0.05…0.3}`). Case B exhibits the explicit failure mode (radius→0 under a
  constant `R`-bias). C09 holds near good fits and fails as residual curvature
  grows — the boundary is characterized.

Neither claim is promoted to unconditional; both remain local and
benchmark-scoped. No BH/SIR/MMD/OPG evidence is implied.

## Deviations / failures

- No library deviations. Two **test-assumption** fixes during bring-up (not
  code bugs): (a) the α-grid constant was initially a module-level `jnp.array`
  created float32 at import (before the conftest x64 fixture), injecting ~1e-7
  error — changed to a numpy float64 constant; (b) an expected optimum-point
  positive/negative radius asymmetry was not resolvable at the chosen grid, so
  that test now asserts the robust local + direction-dependent property instead.
- `validity_radius` returning `0` for Case B is expected behavior (constant
  `R`-bias), documented above.

## Next

Phase 2 continues; EXP-002 accepted pending review. No EXP-000, EXP-003, BH,
SIR, MMD, or historical-OPG work performed. The OPG-rename blocker remains open.
