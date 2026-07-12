# Status

Where the project stands. One line per experiment; update on completion.

## Current phase

**Phase 4 — Brock–Hommes geometry (post-instrument-validation).**

The instrument is validated: we know how to compute the GGN (EXP-001), when it
approximates the Hessian (EXP-002), and how to estimate it under stochastic
simulation (EXP-003). The historical-OPG audit (EXP-000) is done but flawed (ran
in the wrong dynamical regime — see below).

## What's proven (see CLAIMS.md for evidence)

- **C01** supported — AD reproduces the analytic GGN on affine benchmarks (float64).
- **C02** supported — GGN is PSD and measures first-order representation change.
- **C03, C09** supported conditionally — numerically exact `H=G+R`; GGN predicts
  local loss near low-residual fits; irreducible bias when residual-curvature is large.
- **C04, C05, C06** supported conditionally — MMD GGN estimable from feature
  Jacobians on a controlled Gaussian/RFF simulator; plug-in `O(1/M)` bias,
  cross-seed unbiased-but-indefinite.
- **C07, C17, C18, C19** rejected (see CLAIMS.md).

## Experiments

| ID | Question | Status |
|---|---|---|
| EXP-001 | Does AD recover the analytic GGN? | ✅ done — C01 |
| EXP-002 | When does GGN ≈ Hessian (nonlinear)? | ✅ done — C03, C09 |
| EXP-003 | Can the MMD GGN be estimated under noise? | ✅ done — C04/05/06 |
| EXP-000 | What did the historical BH OPG measure? | ⚠️ done but **flawed regime** — redo needed |

## Immediate next

1. **Redo EXP-000 in the correct BH regime.** The BH model is correct, but chaos
   needs large β (~50–120); EXP-000 used β=2–5 (trivial near-fundamental regime),
   which made the representation rank-1 and its conclusion an artifact. See
   `papers/reference_brock_hommes_1998.md` and `experiments/EXP-000.md`.
   Optionally add an H=4 canonical variant first.
2. Then the smooth Brock–Hommes geometry study (planned EXP-004): GGN vs exact
   Hessian across regimes, validity radii, comparison with the corrected OPG.

## Deferred (revive when reached)

SIR posterior validation, policy-functional analysis, observation design,
discrete/surrogate-gradient fidelity, computational scaling. The future-paper
outline is archived at `archive/vault/08_PAPER_ARCHITECTURE.md`.
