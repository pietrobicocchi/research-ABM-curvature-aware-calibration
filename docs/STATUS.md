# Status

Where the project stands. One line per experiment; update on completion.

## Current phase

**Phase 4 — Brock–Hommes geometry (post-instrument-validation).**

The instrument is validated: we know how to compute the GGN (EXP-001), when it
approximates the Hessian (EXP-002), and how to estimate it under stochastic
simulation (EXP-003). The historical-OPG audit (EXP-000) is done (redone in the
correct large-β regime): the old OPG is the gradient covariance, not the GGN.

## What's proven (see CLAIMS.md for evidence)

- **C01** supported — AD reproduces the analytic GGN on affine benchmarks (float64).
- **C02** supported — GGN is PSD and measures first-order representation change.
- **C03, C09** supported conditionally — numerically exact `H=G+R`; GGN predicts
  local loss near low-residual fits; irreducible bias when residual-curvature is large.
- **C04, C05, C06** supported conditionally — MMD GGN estimable from feature
  Jacobians on a controlled Gaussian/RFF simulator; plug-in `O(1/M)` bias,
  cross-seed unbiased-but-indefinite.
- **C08** not supported — the historical BH OPG ≈ gradient covariance `C_g`,
  nearly orthogonal to the true GGN (EXP-000).
- **C07, C17, C18, C19** rejected (see CLAIMS.md).

## Experiments

| ID | Question | Status |
|---|---|---|
| EXP-001 | Does AD recover the analytic GGN? | ✅ done — C01 |
| EXP-002 | When does GGN ≈ Hessian (nonlinear)? | ✅ done — C03, C09 |
| EXP-003 | Can the MMD GGN be estimated under noise? | ✅ done — C04/05/06 |
| EXP-000 | What did the historical BH OPG measure? | ✅ done — gradient covariance, not GGN (C08 not supported) |

## Immediate next

**EXP-004 — smooth Brock–Hommes geometry** in the correct large-β regime: GGN vs
exact Hessian across dynamical intensities, validity radii, and comparison with
the corrected OPG (`scalar_gradient_opg`). Two open threads surfaced by EXP-000
to fold in:
- **Differentiation-horizon dependence (RQ4):** the full-horizon GGN explodes in
  the chaotic regime; the geometry depends on `grad_horizon`. Characterize it.
- Optionally add an H=4 canonical BH variant (β=120) to match the literature exactly.

## Deferred (revive when reached)

SIR posterior validation, policy-functional analysis, observation design,
discrete/surrogate-gradient fidelity, computational scaling. The future-paper
outline is archived at `archive/vault/08_PAPER_ARCHITECTURE.md`.
