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
- **C12** supported conditionally — a sloppy GGN direction leaves a policy quantity
  undetermined: the SIR intervention value spans ~0→2180 cases within a good fit
  (along f_lock), while observed cases stay ±0.3% (EXP-006). The scientific consequence.
- **C13** supported conditionally — a targeted added observation (compliance) lifts
  the weak f_lock direction λ 8.9e-6→10 (d_data 3→5) and collapses the policy
  uncertainty 57×; a redundant one (prevalence) does not. The diagnostic is
  actionable (EXP-007).
- **C14** supported conditionally — on discrete network-SIR the leading GGN
  eigenspace is robust to the surrogate gradient at **two** operating points
  (Gumbel↔straight-through leading-2 angle 8.9° & 8.4°; Gumbel FD-validated
  3.0e-5/3.4e-5); eigenvalue *scale* is surrogate-dependent (0.22–0.48×) (EXP-008).
- **C15** supported conditionally — truncated differentiation destroys the local
  geometry: halving the horizon rotates the leading eigenspace ~80° and collapses
  its top eigenvalue to 15% of full (→1% at T/8) (EXP-008). Empirical counterpart to
  the RQ4 limitation (DEC-011).
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
| EXP-006 | Does a sloppy direction change a policy output? | ✅ done — **yes**: intervention value ~0→2180 cases within a good fit (C12) |
| EXP-007 | Which added observation informs the weak direction? | ✅ done — compliance: λ 8.9e-6→10, policy uncertainty ↓57× (C13) |
| EXP-008 | Is discrete-SIR GGN geometry robust to the surrogate, and does horizon truncation destroy it? | ✅ done — leading eigenspace robust Gumbel↔ST ~8.4–8.9° at **two** operating points; horizon truncation rotates it ~80° + collapses λ (C14, C15) |

## Scope limitations

**RQ4 — differentiation-horizon dependence / chaotic GGN explosion (resolved as a
limitation, DEC-011).** The GGN depends materially on the differentiation horizon,
and in chaos the exact GGN is dominated by chaotic (Lyapunov) sensitivity
(EXP-004). EXP-004b shows this is **intrinsic**, not a representation artifact: the
explosion lives in `∂X/∂θ`, so *any* pathwise-differentiated observable inherits it
(robust summaries explode more, not less). This is recorded as an honest scope
boundary of the local diagnostic — Brock–Hommes chaos is the applicability
boundary, smooth SIR is the inferential core — **not** a headline claim and **not**
grounds for a dedicated experiment. A meaningful chaotic-regime geometry would need
a fixed horizon or a derivative-free/ensemble construction (future work). Was
novel relative to Quera-Bofarull 2025 §8.3 (second-order = future work there).

## Immediate next

The full arc is now demonstrated end-to-end on smooth SIR: compute the GGN
(EXP-001) → know when it ≈ Hessian (EXP-002) → estimate under noise (EXP-003) →
validate against a real posterior (EXP-005, C10/C11) → a sloppy direction breaks a
**policy** answer (EXP-006, C12) → the right added observation **fixes** it
(EXP-007, C13). BH is an honest scope boundary (EXP-004/004b). The
minimum-viable-paper claim arc **C01–C13 is covered**; remaining is depth/robustness.

EXP-008 (stochastic/discrete-MMD keystone) now extends the arc off smooth SIR:
the diagnosed geometry survives the transition to a genuinely discrete/stochastic
simulator (C14, surrogate-robust leading eigenspace) and the RQ4 horizon limitation
is now demonstrated, not just asserted (C15). The MVP claim arc is **C01–C15**.

Candidates (scientific lead / gatekeeper to prioritize):
- **First manuscript pass** — the evidence for the MVP arc is complete.
- **EXP-008 depth** (optional): C14/C15 now hold at two operating points (attack
  29% & 52%, different graphs). Remaining depth = add the SPA/StochasticAD estimator
  (deferred, DEC-011) to strengthen C14 beyond Gumbel-vs-ST with an unbiased third
  estimator, or a cross-seed-estimator variance check.
- **EXP-009 — computational scaling** (C16), if a scaling claim is wanted.

## Deferred (revive when reached)

SIR posterior validation, policy-functional analysis, observation design,
discrete/surrogate-gradient fidelity, computational scaling. The live paper outline
is Layer 2 `paper/PAPER_ARCHITECTURE.md`.
