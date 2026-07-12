---
title: EXP-000 Implementation Plan — Brock–Hommes historical-OPG audit
status: approved-implementing
date: 2026-07-11
base_commit: ab85687
authority: subordinate to 02_MATHEMATICAL_SPECIFICATION.md, 04_EXPERIMENT_REGISTRY.md, 07_DECISION_LOG.md
scope: EXP-000 audit only. No production BH/SIR/MMD/calibration changes.
supports (intended): historical-audit claim only (see §Claims)
---

# EXP-000 Implementation Plan

Determine what the historical Brock–Hommes OPG experiments actually measured, by
comparing at matched points in prior-scaled coordinates `z`, with **common random
numbers**:

    H_exact = ∇²_z L,   G_MMD = J_η^T J_η,   F_OPG = (1/M)Σ g_m g_m^T,   C_g = (1/M)Σ (g_m−ḡ)(g_m−ḡ)^T.

## A. Brock–Hommes objective audit

1. **Physical parameters** `θ = (β, g₁, b₁, g₂, b₂)`, P=5 (`models/brock_hommes.pack_canonical`;
   type-0 fundamentalist g₀=b₀=0). Simulator constants `R`, `σ`, `T`, `H=3`.
2. **Transform from `z` (NEW, EXP-000-local, documented):** `θ = T(z) = θ₀ + s⊙z`,
   prior `z~N(0,I₅)`, `s = (0.30, 0.10, 0.05, 0.10, 0.05)`, `θ₀` = the regime
   reference (§C). `z★ = 0`. All four matrices computed in `z`. (Production BH
   differentiates wrt physical θ; this transform is added only in the experiment.)
3. **Simulator randomness / seeds:** `ε = σ·N(key,(T,))`, per-seed keys via
   `split`. Because `ε` depends only on `key`, reusing a seed's key across `z`
   gives identical `ε` — i.e. keys ARE the reparameterized common random numbers.
4. **Calibrated summaries/features:** historically NONE explicit — the loss is
   MMD² directly on the raw `(T,)` trajectories via an RBF kernel with
   median-heuristic bandwidth (`losses/mmd`).
5. **Exact empirical MMD objective:** unbiased **U-statistic** MMD²
   (`mmd_sq_unbiased`, diagonal excluded) with `stop_gradient` bandwidth.
6. → **unbiased U-statistic MMD** (not biased/V-statistic, not a squared residual).
7. **Averaging over draws:** MMD² averages over M×M kernel pairs; `mean_grad =
   (1/M)Σ g_m = ∇_z MMD²`.
8. **Historical per-seed gradients:** `g_m = M·(∂MMD²/∂x_m)·(∂x_m/∂z)` via VJP
   (`per_seed_loss_and_grads`); `F_OPG`, `C_g` from these (now
   `diagnostic.scalar_gradient_opg` / `scalar_gradient_covariance`).
9. **Is `J_η` exposed?** Not in production for BH. EXP-000 constructs it via a
   finite-feature representation (below), reusing `geometry/rff` + `mmd_estimators`.
10. **Simulator derivatives:** exact pathwise AD (smooth softmax + `lax.scan`,
    reparameterized noise). Confirmed.

## B. MMD representation for the true GGN

The historical scalar loss is the **unbiased U-statistic MMD²**, which has no
direct squared-residual representation — so its finite-batch decomposition is NOT
called a GGN. Instead EXP-000 uses a **finite-feature mean-embedding
representation consistent with EXP-003**:

- frozen RBF random Fourier features `ψ(x)∈R^D` on the `(T,)` trajectory,
  bandwidth `γ` = median heuristic on a pilot sample (frozen);
- `η̂(z) = (1/M)Σ_m ψ(X_z(ε_m))` (biased empirical mean embedding, fixed ε);
- `L_feat(z) = ½‖η̂(z) − η_y‖²`;  `G_MMD = Ĵ_η^T Ĵ_η` (plug-in, EXP-003);
  `H_exact = ∇²_z L_feat`;  `R = H − G` (exact, finite-dim).

**Documented difference:** `H`, `G`, `R` use the finite-feature mean-embedding
squared loss `L_feat`; `F_OPG`, `C_g` use the historical U-statistic MMD² scalar
loss. They are different objectives on the same simulator — the audit's purpose
is precisely to see whether the historical `F_OPG` (from the U-stat loss)
recovers the true representation GGN `G`.

## C. Parameter points and the regime blocker

> **BLOCKER (documented, not fatal to the audit).** This BH implementation has no
> bounded periodic/chaotic attractor: in the noiseless limit `x=0` is globally
> stable for net trend `< R·1.5` and trajectories diverge above it (no fold-back
> — the model lacks a fundamentalist cost / price-adjustment nonlinearity).
> Verified by a β∈{1…150}, g∈{0.6…2.8} sweep. Textbook stable/periodic/chaotic
> points are therefore **unavailable**. EXP-000 instead uses three points
> spanning the accessible **stochastic** dynamical range, classified by measured
> dynamics (lag-1 autocorrelation, variance), and reports this substitution.

Points (`R=1.1`, `σ=0.03`, `T=150`), `θ₀`:
- **P1 quiescent** `(β=2.0, g=0.6/0.5, b=0.05/−0.05)` — deep stable, ac₁≈0.32.
- **P2 near-critical** `(β=3.0, g=1.35/1.25, b=0.15/−0.10)` — ac₁≈0.76.
- **P3 edge** `(β=5.0, g=1.5/1.4, b=0.20/−0.15)` — near trend threshold, ac₁≈0.86.

The 4th "historical calibration final point" is **not reproducible** (historical
scripts were removed in the repo reset) — documented and omitted.

For each point record: `θ` and `z`; residual magnitude; loss; `G`, `H` spectra;
eigengaps. (Design check: `G` is dominated by 1–2 directions — top eigenvalue
0.37→1.73 across P1→P3 — so top-1 principal angle is the primary subspace metric;
top-2 reported with a low-effective-rank caveat.)

## D. Monte-Carlo design

- **Common random numbers:** the same M keys feed `A_m`, `L_feat` (via `η̂`), and
  the historical per-seed gradients at a point.
- **Sample sweep** `M ∈ {32, 128}`; **batches** `B = 15` for uncertainty.
- **Features** `D = 256`, RBF, frozen `γ` (median heuristic), `feature_seed=7`.
- **Residual sweep** target `z_y = z★ + δ·e₁`, `δ ∈ {0.0, 0.3, 0.8}` (`G` fixed;
  `F_OPG`, `C_g`, residual vary). float64 throughout.

## E. Metrics

For every matrix pair among {H, G, F_OPG, C_g}: relative Frobenius, trace-
normalized relative Frobenius, eigenvalue lists, top-1 and top-2 principal
angles, trace, spectral norm, numerical rank (rtol=1e-8). For (H,G): `R=H−G`
and `e_curv(v)=|vᵀ(H−G)v|/max(|vᵀHv|,ε)` along `G` eigenvectors. `F_OPG`/`C_g`
reported vs residual magnitude, `M`, batch, and regime.

## F. Local predictive test

`ΔL_feat(α)=L_feat(z★+αv)−L_feat(z★)` vs `½α²(vᵀGv)` on a prespecified symmetric
`α`-grid, for directions: leading-`G` eigenvector, weakest-`G` eigenvector,
leading-`F_OPG` eigenvector, leading-`C_g` eigenvector.

## G. Questions & outcome classification (per point)

Answer: (1) does `F_OPG` align with `G`? (2) with `C_g`? (3) residual/scale
dependence? (4) collapse near exact match? (5) `H` vs `G` per regime? (6)
stiff/sloppy recoverable? (7) which old figures/conclusions survive?

Classify each point:
- **A** OPG ≈ GGN (small `angle(F_OPG,G)`) — empirically useful, no theoretical equivalence.
- **B** OPG ≈ gradient covariance (small `angle(F_OPG,C_g)`) — measured gradient variability, not curvature.
- **C** OPG ≈ neither — historical interpretation discarded.

## H. Claims

Supports a **historical-audit claim** only:
> In Brock–Hommes, the historical per-seed scalar-gradient second moment does /
> does not recover the true local MMD GGN under the specified points and residual
> levels; it aligns primarily with [GGN / gradient covariance / neither].

Cannot support: posterior identifiability, prior-relative data information, global
sloppiness, policy implications, or any SIR/surrogate conclusion.

## I. Files / tests / output

- `experiments/exp000_bh_audit.py` — `run()` (audit at P1–P3, residual + M sweep,
  predictive test, classification). Reuses `geometry/rff`, `mmd_estimators`,
  `diagnostic` (`scalar_gradient_opg`, `scalar_gradient_covariance`,
  `eigendecompose`, `principal_angles`), `validity` (`curvature_bias`),
  `per_seed_grads`, `losses/mmd`, `config`, `provenance`, `metrics`.
- `tests/test_exp000_bh_audit.py` — CRN identity (keys reproduce ε), `G` plug-in ==
  `Ĵ_η^T Ĵ_η`, `H==G` at δ=0 (R≈0), `F_OPG→0` at δ=0, `F_OPG=C_g+ḡḡᵀ` relation,
  classification helper.
- Output `outputs/EXP-000/<run-id>/` (config, provenance, metrics, arrays, figures).

Proceed to implementation.
