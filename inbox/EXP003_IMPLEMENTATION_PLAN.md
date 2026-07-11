---
title: EXP-003 Implementation Plan — Stochastic MMD GGN estimation
status: proposed-for-review
date: 2026-07-11
base_commit: d2a01a5
authority: subordinate to 02_MATHEMATICAL_SPECIFICATION.md, 04_EXPERIMENT_REGISTRY.md, 07_DECISION_LOG.md
scope: EXP-003 design ONLY. Not implemented. No production MMD/BH/SIR/OPG/calibration changes.
supports (intended): C04, C05, C06 (+ illustrates rejected C07)
---

# EXP-003 Implementation Plan

Validate estimators of the MMD generalized Gauss–Newton matrix

    G_MMD(θ) = J_μ(θ)ᵀ J_μ(θ),   J_μ = D_z μ_θ,   μ_θ = E_ε[ψ(X_θ)]   (Math-Spec §12–13)

for a **controlled stochastic simulator** with a **finite random-Fourier-feature**
representation, against an **analytic reference**. Reuses EXP-001/002 infra
(`config`, `metrics`, `provenance`, `diagnostic.eigendecompose`,
`geometry.ggn.scalar_gradient_outer_product` as the labeled comparison object).

**Not to be implemented in this step.**

---

## 1. Controlled simulator (Gaussian location–scale)

    X_θ = μ_θ + L_θ ε,   ε ~ N(0, I_n),   n = 2.

Physical parameters (P=5): mean `μ=(m1,m2)`, and lower-triangular Cholesky
`L=[[s1,0],[ℓ,s2]]` with `s1,s2>0` ⇒ Σ_θ = L_θ L_θᵀ. Parameters affect **both**
mean and covariance.

**Prior-scaled coordinates (DEC-003).** Work in `z~N(0,I₅)` with a documented
transform `θ = T(z)`:

    m1 = m1₀ + σ_m·z1,        m2 = m2₀ + σ_m·z2,
    s1 = exp(ρ0 + σ_s·z3),    s2 = exp(ρ0 + σ_s·z4),   ℓ = σ_ℓ·z5.

`T` is smooth and invertible on the scale entries (log-parameterization keeps
`s>0`); constants `(m₀, ρ0, σ_m, σ_s, σ_ℓ)` are fixed in `config.json`. All
Jacobians `J_μ = D_z μ_θ` are taken in `z`.

Reparameterized (ε fixed per seed) so `X_θ(z,ε)` is pathwise-differentiable.

---

## 2. Representation — frozen random Fourier features (RBF)

    ψ(X) = sqrt(2/D) · [cos(w_jᵀ X + b_j)]_{j=1..D} ∈ R^D,
    w_j ~ N(0, γ⁻² I_n)  (frozen),  b_j ~ U[0, 2π]  (frozen).

`ψ` approximates an RBF kernel of bandwidth γ. **Frozen:** `{w_j, b_j}` are drawn
once from a fixed `feature_seed`, saved to the run record, and reused by every
estimator, batch, and the reference (design Q1).

**Analytic reference (design Q2).** For Gaussian `X_θ`,

    μ_θ,j = E[ψ_j] = sqrt(2/D) · exp(−½ w_jᵀ Σ_θ w_j) · cos(w_jᵀ μ_θ + b_j)

is **closed form**. Hence the finite-D reference is exact:

    J_μ^ref = D_z μ_θ  (AD of the closed form),   G_ref = J_μ^refᵀ J_μ^ref.

This is the ground truth against which estimators are scored — no sampling in the
reference. A second, coarser reference (`D→large`, or the closed-form RBF GGN)
characterizes the RFF approximation of the true kernel (see Q4).

---

## 3. Estimators to compare

Per-seed feature Jacobian `A_m = D_z ψ(X_θ(z,ε_m)) ∈ R^{D×P}` (pathwise AD,
reparameterized). Evaluated at a fixed `z★` with a fixed target `μ_y = μ_θ(z_y)`
for a small nonzero residual (so the OPG comparator is nonzero).

1. **PSD plug-in:** `Ĵ_μ = (1/M) Σ_m A_m`,  `Ĝ_V = Ĵ_μᵀ Ĵ_μ`  (PSD; biased at finite M).
2. **Cross-seed:** `Ĝ_U = 1/(M(M−1)) Σ_{m≠n} A_mᵀ A_n`  (unbiased; possibly indefinite).
3. **Raw scalar-loss OPG (comparison object, NOT curvature — DEC-001):**
   `F_OPG = (1/M) Σ_m g_m g_mᵀ`, `g_m = A_mᵀ W r`, `r = μ_θ(z★) − μ_y`. Built with
   a **new, clearly-labeled** helper (or `geometry.ggn.scalar_gradient_outer_product`
   on the feature-mean loss); the production `per_seed_grads.py` is not touched.
4. **Optional cross-check:** direct mixed kernel-derivative implementation of
   `G_MMD` (∂²k) as an independent reference.

Matrix-free `Ĝ_V v = Ĵ_μᵀ(Ĵ_μ v)` reuses the `geometry.ggn` JVP/VJP pattern.

---

## 4. Variables (prespecified)

- sample count `M ∈ {8, 16, 32, 64, 128, 256}`;
- feature dimension `D ∈ {64, 256, 1024}`;
- kernel bandwidth `γ ∈ {½, 1, 2} × γ₀` (γ₀ a fixed median-scale reference);
- simulator noise: covariance scale via `ρ0 ∈ {log 0.5, log 1.0}`;
- repeated independent batches `B = 50` (design Q5).

All fixed in `config.json` before running. float64 mandatory
(`enable_x64()`+`require_x64()` at entry, before arrays/trace).

**Shared draws (design Q3).** Within a batch, the same ε-draws feed PSD,
cross-seed, and OPG (common random numbers) for a fair comparison; batches are
independent (fresh ε). Frozen RFF are shared across everything.

**Bias vs RFF-approximation separation (design Q4).** Estimator bias/variance are
scored against the **finite-D analytic `G_ref(D)`** using the *same* frozen
features — this isolates Monte-Carlo (finite-M) error from RFF error. The
RFF-vs-true-RBF error is a *separate* study sweeping `D` against the large-D /
closed-form RBF reference. The two are never conflated.

---

## 5. Metrics

Over `B` batches, per `(M, D, γ, noise)` cell, for each estimator:
- **matrix bias:** `‖ (1/B)Σ_b Ĝ_b − G_ref ‖_F` (and relative);
- **matrix variance:** `(1/B)Σ_b ‖ Ĝ_b − mean_b ‖²_F`;
- **relative Frobenius error** vs `G_ref` (scale-aware, `metrics.rel_frobenius_error`);
- **eigenvalue error** (`metrics.eigenvalue_rel_error`);
- **leading-subspace principal angles** (top-k vs `G_ref`, `metrics.max_principal_angle`);
- **cross-seed indefiniteness:** frequency and magnitude of negative eigenvalues of `Ĝ_U`;
- **runtime and memory** per estimator;
- **GGN vs raw OPG:** `‖ G_ref − F_OPG ‖_F` and leading-subspace angle, to
  demonstrate that the OPG is a different object (DEC-001).

Expected scalings recorded (not gated): PSD-plug-in bias `~ O(1/M)`; cross-seed
bias `~ 0` within MC error; both variances `~ O(1/M)` (batch-wise).

---

## 6. Acceptance criteria (leading-subspace recovery, design Q6)

Prespecified:
- **A1 (consistency):** for PSD plug-in, `rel_fro(mean_b Ĝ_V, G_ref) → 0` as M
  grows, with the top-`k` subspace `max principal angle ≤ 0.087 rad (5°)` at some
  `M★ ≤ 256` for the leading data-informed subspace (`k` = #eigenvalues within 3
  decades of the top).
- **A2 (bias ordering):** cross-seed `Ĝ_U` bias ≤ PSD-plug-in bias at matched M
  (within batch MC error), confirming C05/C06.
- **A3 (indefiniteness characterized):** cross-seed negative-eigenvalue
  frequency/magnitude reported and shown to shrink with M.
- **A4 (OPG distinct):** `F_OPG` differs from `G_ref` by a non-negligible
  leading-subspace angle and/or Frobenius error (illustrates rejected C07).
- **A5:** reproducible run record (provenance, frozen features saved) + green tests.

float64-only gate; no ABM, no production-code changes.

---

## 7. Files / tests (design only)

**New (no production edits):**
- `src/curvature_calib/benchmarks/gaussian_location_scale.py` — simulator,
  transform `T`, closed-form `μ_θ`, analytic `J_μ^ref`, `G_ref`.
- `src/curvature_calib/geometry/rff.py` — frozen RFF feature map + freezing/serialization.
- `src/curvature_calib/geometry/mmd_estimators.py` — `feature_jacobians` (A_m),
  `ggn_plugin` (Ĝ_V), `ggn_cross_seed` (Ĝ_U), matrix-free `Ĝ_V v`, and a
  **labeled** `scalar_loss_opg` comparison helper.
- `experiments/exp003_mmd_ggn.py` — `run()`.

**Tests:**
- closed-form `μ_θ` matches high-sample `E[ψ(X_θ)]` (MC, within `~1/√(MC)`);
- `Ĝ_V, Ĝ_U → G_ref` as M grows (fixed frozen features), `Ĝ_V` PSD, `Ĝ_U`
  unbiased in expectation over batches;
- matrix-free `Ĝ_V v == Ĝ_V @ v`;
- OPG `≠ G_ref` at nonzero residual; OPG `→ 0` as residual `→ 0` while `G_ref`
  stays nonzero (DEC-001 illustration);
- frozen features are reproducible from `feature_seed`;
- provenance/config keys. All added to the conftest x64-scoped module set.

**Output/provenance:** `outputs/EXP-003/<UTC-timestamp>_<short-commit>/` with
`config.json` (incl. `feature_seed`, `M/D/γ/noise/B`), `provenance.json`,
`metrics.json`, `arrays.npz` (frozen features, `G_ref`, per-estimator matrices),
`figures/` (FIG-03: bias/variance/angle vs M and D). Nothing under `docs/`.

---

## 8. Claims and failure implications

**Can support (design Q7):**
- **C04** — the MMD GGN is consistently estimated from simulator feature
  Jacobians (A1).
- **C05** — PSD plug-in has acceptable finite-sample bias (bias decay + A1).
- **C06** — cross-seed has lower bias but can be indefinite at finite M (A2, A3).
- Strengthens the OPG≠GGN comparison (A4), groundwork for EXP-000.

**Cannot support:** anything about Brock–Hommes/SIR geometry, surrogate/discrete
gradients, prior-relative posterior agreement, or unconditional MMD validity;
those are EXP-004/005/008.

**Failure implication (design Q8):** if no estimator recovers the leading
subspace at feasible M/D (A1 fails), the MMD-GGN leg of the paper (Architecture
§4) is not viable as the primary route; the paper would fall back to
finite-summary representations (DEC-005) as the main calibrated representation
and treat MMD as secondary or future work. A cross-seed indefiniteness that does
not shrink with M would restrict C06 to a cautionary result.

---

## 9. Reuse and scope

Reused unchanged: `config`, `metrics`, `provenance`, `diagnostic.eigendecompose`,
`geometry.ggn` (JVP/VJP pattern + labeled scalar OPG), the `outputs/<EXP>/<run-id>/`
convention. **Untouched (scope):** `losses/mmd.py`, `per_seed_grads.py` and the
historical OPG, Brock–Hommes, SIR, network-SIR, `surrogates.py`, calibration
code, and the canonical vault. The OPG-rename blocker (EXP001 plan) remains open
and gates only the ABM reruns, not EXP-003.

Awaiting review before implementation. EXP-000 and ABM experiments are not begun.
