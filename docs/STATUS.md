# Status

Where the project stands. One line per experiment; update on completion.

**Bridge doc** — the single source of *factual* state, read by both agents. The
implementation specialist updates it (recording rule). For *interpretation* and
paper direction, see Layer 2 `paper/STRATEGIC_HANDOFF.md`.

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
| EXP-004 | GGN vs Hessian in BH; horizon dependence? | ✅ done — H=G at fit; GGN horizon-biased, tiny validity radius in chaos (C03/C09) |

## Key open finding (RQ4 — needs a decision)

**The GGN depends materially on the differentiation horizon** (EXP-004): truncation
under-estimates curvature magnitude by orders of magnitude at every β, preserves
the leading direction in the complex regime but distorts it in chaos, and the
full-horizon curvature explodes (λ₁ ~ 1.5e8 at β=80). This is a novel result
(second-order was future work in Quera-Bofarull 2025 §8.3). Not yet a registered
claim — flagged for the scientific lead to decide whether it becomes a headline
claim + a dedicated experiment (EXP-008 differentiation-fidelity).

## Immediate next

Candidates (for the scientific lead / gatekeeper to prioritize):
- **EXP-005 — smooth SIR posterior validation** (the inferential-validity leg, C10/C11).
- **EXP-008 — differentiation-horizon fidelity**, promoting the RQ4 finding above.
- Optionally an H=4 canonical BH variant (β=120) to match the literature exactly.

## Deferred (revive when reached)

SIR posterior validation, policy-functional analysis, observation design,
discrete/surrogate-gradient fidelity, computational scaling. The live paper outline
is Layer 2 `paper/PAPER_ARCHITECTURE.md`.
