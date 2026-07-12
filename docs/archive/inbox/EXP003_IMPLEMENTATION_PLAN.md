---
title: EXP-003 Implementation Plan — Stochastic MMD GGN estimation
status: approved-with-revisions
date: 2026-07-11
revised: 2026-07-11
base_commit: d2a01a5
authority: subordinate to 02_MATHEMATICAL_SPECIFICATION.md, 04_EXPERIMENT_REGISTRY.md, 07_DECISION_LOG.md
scope: EXP-003 only + reusable estimators. No production MMD/BH/SIR/OPG/calibration changes.
supports (intended): C04, C05, C06 (+ illustrates rejected C07)
---

# EXP-003 Implementation Plan

Validate estimators of the MMD generalized Gauss–Newton matrix

    G_ref(z) = J_η(z)ᵀ J_η(z),   J_η = D_z η,   η(z) = E_ε[ψ(X_z)]   (Math-Spec §12–13)

for a controlled stochastic simulator with a frozen random-Fourier-feature (RFF)
representation, against an **analytic** reference. Reuses EXP-001/002 infra
(`config`, `metrics`, `provenance`, `diagnostic.eigendecompose`, `geometry.ggn`).

> **Revision note (2026-07-11).** Approved subject to six revisions, folded in:
> (1) notation split `a(z)` (simulator mean) vs `η(z)` (feature mean embedding),
> `J_η`, `G_ref`; (2) exact plug-in bias formula + `S,Q` cross-seed; (3) fixed
> `k=2` with a clear-gap main regime and a near-degenerate stress regime,
> constants/spectra recorded below; (4) two labeled OPG comparators + trace
> normalization; (5) finite-difference reference validation + exact RBF kernel;
> (6) `B=200`, bias CIs, tightened acceptance.

## 1. Notation (no collision)

- `a(z)` — Gaussian simulator mean; `Σ(z)=L(z)L(z)ᵀ` — covariance.
- `η(z) = E_ε[ψ(X_z)]` — feature mean embedding; `J_η = D_z η`; `G_ref = J_ηᵀ J_η`.

Code names mirror this: `gaussian_mean` (a), `feature_mean` (η),
`feature_mean_jacobian` (J_η), `ggn_reference` (G_ref).

## 2. Controlled simulator (Gaussian location–scale)

    X_z = a(z) + L(z) ε,   ε ~ N(0, I₂).

Prior-scaled coordinates `z ~ N(0, I₅)`, transform `θ = T(z)`:

    a1 = m0 + σ_m·z1,   a2 = m0 + σ_m·z2,
    s1 = exp(ρ0 + σ_s·z3),  s2 = exp(ρ0 + σ_s·z4),  ℓ = σ_ℓ·z5,   L = [[s1,0],[ℓ,s2]].

Reparameterized (ε fixed per seed) ⇒ `X_z(ε)` is pathwise-differentiable.

## 3. Representation & analytic reference

Frozen RFF: `ψ(X)=√(2/D)·[cos(w_jᵀX+b_j)]`, `w_j~N(0,γ⁻²I)`, `b_j~U[0,2π]`,
drawn once from `feature_seed` and saved. Closed-form embedding for Gaussian `X_z`:

    η_j(z) = √(2/D)·exp(−½ w_jᵀ Σ(z) w_j)·cos(w_jᵀ a(z) + b_j).

`J_η = D_z η` by AD of this closed form; `G_ref = J_ηᵀ J_η` (exact for the frozen
finite-D features). **This finite-D `G_ref` is the primary scoring reference.**

### Reference validation (revision 5)

- closed-form `η` vs high-sample MC of `E[ψ(X_z)]` at several z (MC-limited);
- `J_η` vs **independent central finite differences** at several z.
  *Design-check result:* at `z=(0.3,−0.2,0.1,−0.15,0.25)`, D=256, γ=2:
  `‖η_cf−η_MC‖/‖η‖ = 2.4e-3` (MC floor) and `‖J_η−J_fd‖/‖J_η‖ = 3.9e-11`. ✓

### Exact Gaussian-RBF secondary reference (revision 5)

    K(z,z') = |I + (Σ(z)+Σ(z'))/γ²|^{−1/2}
              · exp[−½ (a(z)−a(z'))ᵀ (γ²I + Σ(z)+Σ(z'))⁻¹ (a(z)−a(z'))],
    [G_RBF]_ij = ∂²K/∂z_i∂z'_j |_{z'=z}   (AD mixed second derivative).

The `D`-sweep compares finite-D `G_ref(D) → G_RBF` (RFF approximation error),
**separately** from estimator bias. The primary estimator study is always scored
against the fixed finite-D `G_ref`.

## 4. Fixed leading subspace k=2, regimes and recorded spectra (revision 3)

`k = 2` is fixed (NOT eigenvalue-count-defined). `z★ = 0` in both regimes. Values
below are the analytic `G_ref` spectra at `D=2048`, `feature_seed=0`, `γ=2.0`
(the run recomputes `G_ref` for the actual `D`; the shape is `D`-stable).

**Main regime (clear gap after k=2).**
`GLSConfig(m0=0, σ_m=1.0, ρ0=0.0, σ_s=0.30, σ_ℓ=0.30)`, `γ=2.0`, `z★=0`.
Reference eigenvalues ≈ `[0.1176, 0.1137, 0.0065, 0.0032, 0.0017]`;
**eigengap λ2/λ3 ≈ 17.4** (top-2 = mean-direction subspace, well separated).

**Near-degenerate stress regime (λ2 ≈ λ3).**
`GLSConfig(σ_m=0.5, σ_s=1.0, σ_ℓ=1.0)`, `γ=2.0`, `z★=0`.
Reference eigenvalues ≈ `[0.0726, 0.0359, 0.0293, 0.0284, 0.0183]`;
**λ2/λ3 ≈ 1.23** (boundary near-degenerate; identification of the top-2 subspace
is deliberately hard).

**Residual-sweep targets (§7):** `z_y = z★ + δ·e₁`, `δ ∈ {0.0, 0.2, 0.6}`
(exact match / moderate / larger mismatch). `z★` fixed ⇒ `G_ref` fixed while the
residual varies.

*Design check:* the constants above were verified to produce the stated spectra
before the main run (this section records the check). If a run's actual-`D`
`G_ref` deviates from the intended gap structure, the constants are revised and
re-recorded here before the main study.

## 5. Estimators (revision 2)

Per-seed feature Jacobian `A_m = D_z ψ(X_z(ε_m)) ∈ R^{D×P}` (pathwise AD).
`E[A_m] = J_η` (interchange holds; smooth bounded features).

1. **PSD plug-in:** `Ĵ_η=(1/M)Σ_m A_m`, `Ĝ_V=Ĵ_ηᵀĴ_η` (PSD; biased at finite M).
   Exact bias:

       E[Ĝ_V] = G_ref + (1/M)·E[(A−J_η)ᵀ(A−J_η)].

   The `P×P` covariance correction `C = E[(A−J_η)ᵀ(A−J_η)]` is estimated as
   `Ĉ = (1/(N−1)) Σ (A_m−J_η)ᵀ(A_m−J_η)`; predicted plug-in bias `= Ĉ/M` is
   compared to the observed bias `mean_b Ĝ_V − G_ref` (revision 2 test).

2. **Cross-seed (no explicit M(M−1) pairs):**

       S = Σ_m A_m (D×P),   Q = Σ_m A_mᵀA_m (P×P),
       Ĝ_U = (SᵀS − Q) / (M(M−1)),   then symmetrize before eigendecomposition.

   Unbiased: `E[Ĝ_U] = G_ref`; possibly indefinite at finite M.

3. **Matrix-free** `Ĝ_V v = Ĵ_ηᵀ(Ĵ_η v)` (reuses the `geometry.ggn` JVP/VJP pattern).

## 6. Variables & sampling (revisions 4, 6)

- `M ∈ {8,16,32,64,128,256}`; main `D = 512`; bandwidth `γ = 2.0` (main);
  `γ ∈ {1,2,3}` recorded for sensitivity; simulator noise via `ρ0 ∈ {log 0.5, 0}`.
- **`B = 200` independent batches** for the bias analysis (reduced only if runtime
  measurements show excess; any reduction documented in the review).
- **Shared draws:** within a batch the same ε feed PSD, cross-seed, and both OPGs
  (common random numbers); batches independent.
- **Feature seeds:** ONE frozen `feature_seed` for the main finite-M study;
  MULTIPLE feature seeds for the secondary `D`-sweep (RFF-approximation variance).
- `D`-sweep: `D ∈ {64,256,1024}` scored `G_ref(D)` vs `G_RBF`.

## 7. Two OPG comparators (revision 4) — labeled, NOT curvature (DEC-001)

- **Population-residual OPG** `F_OPG_pop`: `r = η(z★) − η_y`, `g_m = A_mᵀ r`,
  `F_OPG_pop = (1/M) Σ g_m g_mᵀ`. Vary only the target (`δ∈{0,0.2,0.6}`) ⇒ `G_ref`
  fixed, residual changes. At `δ=0` (`r=0`) `F_OPG_pop=0` while `G_ref≠0`.
- **Empirical-loss OPG** `F_OPG_emp`: `r̂ = η̂ − η_y` with `η̂=(1/M)Σ_m ψ(X_m)`,
  `g_m = A_mᵀ r̂`, same outer-product form.

Both are reported as **raw** and **trace-normalized** matrices, and compared to
`G_ref` (Frobenius + top-2 principal angle). Residual scaling alone must not drive
the subspace conclusion — hence trace normalization.

## 8. Metrics (revision 6)

Per cell, over `B` batches, per estimator: matrix bias (with **bootstrap/normal
CIs**), matrix variance, relative Frobenius error vs `G_ref`, eigenvalue error,
top-2 principal angle vs `G_ref`, cross-seed negative-eigenvalue frequency &
magnitude, runtime/memory, and `G_ref`-vs-OPG (both comparators, raw &
trace-normalized). Predicted vs observed plug-in bias (§5.1).

## 9. Acceptance criteria (revision 6)

In the **main regime**:
- **A1 (plug-in bias model):** observed `Ĝ_V` bias is statistically compatible
  with the predicted `Ĉ/M` (CIs overlap) across M.
- **A2 (cross-seed unbiased):** `Ĝ_U` mean bias is statistically compatible with
  zero (bias CI contains 0) at each M.
- **A3 (recovery):** for at least one `M ≤ 256`, **relative plug-in bias ≤ 0.10**
  AND **top-2 principal angle ≤ 5° (0.087 rad)** vs `G_ref`.
- **A4 (OPG distinct):** `F_OPG_pop`/`F_OPG_emp` differ from `G_ref` in top-2
  subspace and/or Frobenius (raw and trace-normalized), and `F_OPG_pop → 0` as
  `δ → 0` while `G_ref` stays fixed.
- **A5:** reproducible run record (provenance + frozen features saved); green tests.

**Near-degenerate regime:** report cross-seed negative eigenvalues **separately**;
do **not** require monotone disappearance with M. Top-2 recovery is reported but
not gated (the boundary is intentionally ill-posed).

float64-only gate; no ABM, no production-code changes.

## 10. Files / tests

**New (no production edits):**
- `benchmarks/gaussian_location_scale.py` — `a(z)`, `L(z)`, `Σ(z)`, `simulate`. ✅ (created)
- `geometry/rff.py` — frozen RFF, `η` closed form, `J_η`, `G_ref`, exact `G_RBF`. ✅ (created)
- `geometry/mmd_estimators.py` — `feature_jacobians` (A_m), `ggn_plugin` (Ĝ_V),
  `ggn_cross_seed` (Ĝ_U via S,Q), matrix-free `Ĝ_V v`, `plugin_bias_covariance`
  (Ĉ), `population_residual_opg`, `empirical_loss_opg`.
- `experiments/exp003_mmd_ggn.py` — `run()` (main + near-degenerate + residual
  sweep + D-sweep).

**Tests:** closed-form `η` vs MC; `J_η` vs finite differences; `Ĝ_V,Ĝ_U → G_ref`
as M grows; `Ĝ_V` PSD; `Ĝ_U` unbiased over batches; **predicted bias `Ĉ/M`
matches observed**; matrix-free == dense; `F_OPG_pop→0` at `δ=0` while `G_ref≠0`;
frozen features reproducible; `G_RBF` symmetric PSD and finite-D `G_ref→G_RBF` as
D grows; provenance/config keys. Added to the conftest x64-scoped module set.

**Output:** `outputs/EXP-003/<UTC-timestamp>_<short-commit>/` with
`config.json` (constants, `z★`, targets, seeds, M/D/γ/noise/B), `provenance.json`,
`metrics.json`, `arrays.npz` (frozen features, `G_ref`, spectra, per-estimator
matrices), `figures/`. Nothing under `docs/`.

## 11. Claims / failure implications

**Can support:** C04 (consistent MMD-GGN estimation), C05 (plug-in bias
acceptable & modeled), C06 (cross-seed lower bias but indefinite). Strengthens
OPG≠GGN (A4).
**Failure:** if no estimator meets A3 at feasible M/D, the MMD-GGN leg
(Architecture §4) is demoted; finite summaries (DEC-005) become the primary
representation. Non-vanishing cross-seed indefiniteness in the **main** regime
would restrict C06 to a cautionary result.

## 12. Reuse and scope

Reused: `config`, `metrics`, `provenance`, `diagnostic.eigendecompose`,
`geometry.ggn`. **Untouched (scope):** `losses/mmd.py`, `per_seed_grads.py`/
historical OPG, Brock–Hommes, SIR, network-SIR, `surrogates.py`, calibration,
canonical vault. OPG-rename blocker remains open (gates ABM reruns only).

Implementation proceeds as revised (no further review cycle unless a blocker appears).
