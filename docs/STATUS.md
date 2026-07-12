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
- **C10, C11** supported conditionally — on smooth SIR the prior-relative GGN
  matches the local posterior contours (axes ~1°, d_data=3) and the profiled
  posterior energy (<0.5%). The positive inferential result (EXP-005).
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
| EXP-004b | Is the chaotic GGN explosion representation-driven? | ✅ done — **no, intrinsic**; robust summaries explode more (1.5e6× at β=80) |
| EXP-005 | Does the prior-relative GGN match the SIR posterior? | ✅ done — **yes** on smooth SIR: axes ~1°, profiles <0.5% (C10, C11) |

## Key open finding (RQ4 — needs a decision)

**The GGN depends materially on the differentiation horizon, and in chaos the
exact GGN is dominated by chaotic sensitivity** (EXP-004). EXP-004b shows this is
**intrinsic**, not a representation artifact: the explosion lives in `∂X/∂θ`
(Lyapunov), so *any* pathwise-differentiated observable inherits it (robust
summaries explode more, not less). Horizon truncation trades explosion for bias.
Novel (second-order was future work in Quera-Bofarull 2025 §8.3). Not yet a
registered claim — flagged for the scientific lead: is this a headline
limitation/claim + a dedicated experiment (EXP-008 differentiation-fidelity)? A
meaningful chaotic-regime geometry likely needs a fixed horizon or a
derivative-free/ensemble construction.

## Immediate next

The core instrument + inferential story is now in place: GGN computed (EXP-001),
Hessian-approx characterized (EXP-002), MMD-estimable (EXP-003), and validated
against a real posterior on smooth SIR (EXP-005). The BH chapter is an honest
scope boundary (EXP-004/004b: chaos breaks the exact geometry). Candidates (for
the scientific lead / gatekeeper to prioritize):
- **EXP-006 — SIR policy-functional analysis** (does a weak direction change a
  policy quantity? C12) — the "scientific consequence" leg.
- **EXP-008 — differentiation-horizon fidelity**, promoting the RQ4/chaos finding.
- **EXP-007 — observation-design** (C13); or discrete/stochastic SIR (C14/C15).

## Deferred (revive when reached)

SIR posterior validation, policy-functional analysis, observation design,
discrete/surrogate-gradient fidelity, computational scaling. The live paper outline
is Layer 2 `paper/PAPER_ARCHITECTURE.md`.
