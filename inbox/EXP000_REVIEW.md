---
title: EXP-000 Review — Brock–Hommes historical-OPG audit
status: complete
date: 2026-07-11
authoritative_commit: fa60bbc (clean tree, dirty=false)
run_id: outputs/EXP-000/20260711T223644Z_fa60bbc/
plan: EXP000_IMPLEMENTATION_PLAN.md
result: audit complete — historical OPG is NOT the true GGN (rank-1 coincidence only)
---

# EXP-000 Review

Full suite **190 passed**. Authoritative run at `fa60bbc` (`dirty:false`, x64, float64).

## Audited Brock–Hommes objective (see plan §A)

`θ=(β,g₁,b₁,g₂,b₂)`, P=5. Historical loss = **unbiased U-statistic MMD²**
(`losses/mmd`) directly on raw `(T,)` trajectories, RBF kernel, median-heuristic
(stop-gradient) bandwidth. Per-seed gradients `g_m=M·(∂MMD²/∂x_m)(∂x_m/∂z)` (exact
pathwise VJP; reparameterized noise). No feature-mean Jacobian `J_η` was exposed
in production; the true GGN is built here via a finite RFF mean-embedding
(consistent with EXP-003). A documented prior-scaled transform `θ=T(z)=θ₀+s⊙z`
(`s=(0.30,0.10,0.05,0.10,0.05)`) makes all matrices live in `z`.

## Exact matrix definitions

- `G_MMD = Ĵ_ηᵀ Ĵ_η`, `Ĵ_η = D_z η̂(z)`, `η̂(z)=(1/M)Σ_m ψ(X_z(ε_m))` (frozen RBF RFF, D=256).
- `H_exact = ∇²_z L_feat`, `L_feat=½‖η̂(z)−η_y‖²`; `R = H − G`.
- `F_OPG = scalar_gradient_opg = (1/M)Σ g_m g_mᵀ` (historical U-stat loss).
- `C_g = scalar_gradient_covariance = (1/M)Σ (g_m−ḡ)(g_m−ḡ)ᵀ`. (`F_OPG = C_g + ḡḡᵀ`.)

Common random numbers (shared keys) across all four objects at each point.

## Parameter points and the regime blocker

**Blocker (documented):** this BH implementation has **no bounded periodic/chaotic
attractor** — noiseless `x=0` is globally stable below a net-trend threshold and
diverges above it (verified over β∈{1…150}, g∈{0.6…2.8}). Textbook
stable/periodic/chaotic points are unavailable. Three points span the accessible
**stochastic** range, classified by lag-1 autocorrelation:
P1 quiescent (ac₁=0.32), P2 near-critical (0.77), P3 edge (0.86). The 4th
"historical calibration final point" is not reproducible (scripts were removed).

## Sample sizes / batches

M∈{32,128}; B=15 batches; residual sweep δ∈{0.0,0.5,1.5} **along the leading-G
(stiff) direction**; prespecified α-grid for the predictive test. float64.

## H, G, F_OPG, C_g comparisons

- **`G` is effectively rank-1** at all three points (eigenvalues e.g. P2
  `[1.43, 0.003, 0, 0, 0]`): the BH RFF-MMD representation informs essentially
  **one** parameter combination. (This may depend on kernel/features.)
- **`F_OPG` vs `G`:** trace-normalized `rel_fro` is small (P1 0.017, P2 0.065,
  P3 0.083) and the top-1 directions coincide — but this is a **rank-1
  coincidence**: `F_OPG = Jᵀ r rᵀ J` collapses onto the single informed direction
  that also spans `G = JᵀJ`. It is **not** curvature recovery.
- **`F_OPG` scale is residual-dependent and collapses at exact fit:**
  `‖F_OPG‖/‖G‖` runs `7e-4 (δ=0) → 0.31 → 0.66` (P1); `G` is residual-independent.
  At δ=0 `F_OPG → 0` while `G ≠ 0` — the DEC-001 counterexample, directly observed.
- **`F_OPG` vs `C_g`:** at large residual `F_OPG ≠ C_g` (`rel_fro` 4–24), because
  the mean-gradient term `ḡḡᵀ` dominates; at near-zero residual `F_OPG → C_g`.

## Residual-curvature diagnostics (H vs G)

`R = H − G` grows sharply with residual: `R/‖H‖` runs `0 → 0.6 → 3.1` (P1),
`0 → 2.8 → 22.6` (P2), `0 → 3.2 → 5.0` (P3); directional `e_curv` up to ~24. So
in BH the GGN approximates the Hessian **only near an exact/low-residual fit** —
the same residual-curvature-bias mechanism EXP-002 established, now in BH.

## Local predictive test

Along the (shared) leading direction at δ=0.5 residual the `G`-quadratic
mispredicts `ΔL_feat` by ~100% (relerr@α=0.1 ≈ 0.8–1.2), consistent with the
large `R` at that residual; near δ=0 the `G`-quadratic is accurate. The weakest
`G` direction has `λ_G≈0` (rank-1), so it carries no quadratic signal.

## Outcome classification (per point)

All three points: **Outcome A (directional) — but strictly a rank-1
coincidence**. `F_OPG`'s leading direction/shape coincides with `G` only because
the representation is effectively rank-1; the scale is residual-dependent and
vanishes at exact fit. **No theoretical equivalence** (DEC-001). This is not
Outcome B (F_OPG is not simply `C_g` at nonzero residual) nor a clean Outcome A
(no genuine multi-directional curvature match — there is only one direction).

## Historical conclusions retained / revised / rejected

- **Rejected:** any reading of the historical BH OPG spectrum as curvature, a
  Hessian/GGN, an identifiability matrix, or an effective dimension. Its scale is
  a residual artifact; it collapses at exact fit.
- **Revised:** the historical "stiff direction" is recoverable **only** as the
  single dominant sensitivity direction of the representation (which the true `G`
  also identifies); the historical "sloppy" directions are not resolved by this
  representation (rank-1) and carry no curvature claim.
- **Retained:** the qualitative observation that BH calibration is dominated by
  one parameter combination — now correctly attributed to `G`, not `F_OPG`.

## Production-code changes

Only the pre-approved OPG rename (commit `ab85687`): `opg_from_grads →
scalar_gradient_opg` (+ deprecated alias), new `scalar_gradient_covariance`,
misleading GGN/Fisher/curvature docstrings purged; numerical object unchanged
(compat tests). No BH/SIR/MMD/calibration numerics were altered.

## Claims supported / not supported

**Supported (historical-audit claim):**
> In Brock–Hommes, the historical per-seed scalar-gradient second moment does
> **not** recover the true local MMD GGN. Its leading direction coincides with
> the GGN only because the finite-feature representation is effectively rank-1;
> its scale is residual-dependent and collapses at an exact representation match,
> confirming it is not a curvature object (DEC-001).

**Not supported / out of scope:** posterior identifiability, prior-relative data
information, global sloppiness, policy implications, SIR, surrogate gradients.
Also **not** established: that BH has stable/periodic/chaotic regimes (the model
lacks a bounded non-trivial attractor); a richer (non-rank-1) BH representation
would need a different kernel/feature design.

## Next

Stop after EXP-000 (per instruction). No EXP-004/SIR/posterior/policy/observation/
surrogate work performed.
