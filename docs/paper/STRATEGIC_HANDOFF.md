---
title: Strategic Handoff
status: active
last_updated: 2026-07-14
project_phase: Phase 7 — arc frozen at C01–C15; outline locked; awaiting figure approval before drafting
---

# Strategic Handoff

**Layer 2 (paper-facing) entrypoint for a fresh scientific-lead session.** This
file carries interpretation and paper direction. For raw factual state (phase,
what's done, next experiment) the single source is Layer 1 `docs/STATUS.md`; this
file does not restate it, it interprets it.

## Project thesis

This project studies the **local, prior-relative generalized Gauss–Newton geometry**
of calibration objectives for differentiable stochastic agent-based models.

For prior-scaled coordinates \(z\), stochastic simulator \(X(z,\xi)\), calibrated
representation \(m(z)=\mathbb E_\xi[\psi(X(z,\xi))]\), and quadratic representation
loss \(\mathcal L(z)=\tfrac12\|m(z)-m_y\|_W^2\), the primary object is

\[
G(z)=D m(z)^\ast W D m(z).
\]

For MMD with feature mean embedding \(\eta(z)=\mathbb E[\psi(X_z)]\),
\(G_{\mathrm{MMD}}(z)=J_\eta(z)^\top J_\eta(z)\). The authoritative derivation lives
in Layer 1 `docs/MATH.md`; this file does not re-derive it.

The intended interpretation is **local and prior-relative**: near a fitted point,
the eigensystem identifies parameter combinations along which the calibration
targets supply strong or weak local information relative to the prior. This is not
a global or structural identifiability theorem.

## Current phase

**Phase 7 — arc frozen, outline locked.** The full claim arc **C01–C15** is now
covered end-to-end: compute the GGN (EXP-001) → know when it ≈ Hessian (EXP-002) →
estimate it under simulation noise (EXP-003) → validate against a real posterior on
smooth SIR (EXP-005, C10/C11) → a sloppy direction leaves a **policy** answer
undetermined (EXP-006, C12) → the right added observation **recovers** it
(EXP-007, C13) → the same loop survives a genuinely **stochastic, discrete,
surrogate-gradient** ABM at two independent operating points (EXP-008, C14/C15).
Brock–Hommes is the honest scope boundary (EXP-004/004b). The gradient
second-moment audit (EXP-000) is a demoted appendix clarification, not the
motivation.

**Spine decision (2026-07-14): methods-first.** The paper leads with the correct
local GGN geometry and its estimators, validated on BH and SIR, and closes with the
policy / observation-design consequence (EXP-006/007) as the scientific payoff, then
the stochastic/discrete robustness result (EXP-008). Not consequence-first.

**The structural gap is closed.** Every flagship SIR result (EXP-005/006/007) was on
smooth *deterministic* SIR under a Gaussian-summary loss, leaving "stochastic" and
"agent-based" carried only by the estimator toy (EXP-003) and the BH feature study.
EXP-008 (two operating points, different regime/graph/graph-seed) closes this: the
leading eigenspace is surrogate-robust and FD-validated, and the MMD estimators
(EXP-003) are now load-bearing in a genuinely stochastic result, not orphaned.

**Decision (2026-07-14, gatekeeper): write now.** No further experiments before
drafting. Two candidates were weighed and declined for this manuscript: SIR
policy-regime robustness (would-be EXP-006b — the payoff chapter states its θ*
scope honestly instead) and computational scaling (EXP-009 — an Appendix C
paragraph, not a claim). The only remaining gate is figure approval
(`FIGURE_LEDGER.md`: FIG-01–FIG-09 are all candidates, none approved yet). The
detailed section-by-section manuscript outline, with explicit asymmetric
weighting and jargon-free section titles, is now in `PAPER_ARCHITECTURE.md`.

## What has been established

### EXP-001 — affine implementation correctness

On affine finite-dimensional benchmarks \(m(z)=Az\), \(G=A^\top WA=\nabla^2\mathcal L\).
The implementation recovers the dense and matrix-free GGN to numerical precision in
float64; rank and null-space structure are recovered; at exact fit the scalar-gradient
OPG vanishes while \(G\neq0\). **Supports C01** (affine finite-dimensional scope).

### EXP-002 — nonlinear GGN validity

The implementation reproduces \(\nabla^2\mathcal L=G+R\) to machine precision on
controlled nonlinear residual problems, establishing two distinct failure modes:
(1) **residual-curvature bias** (\(R\neq0\) creates irreducible local bias even as
perturbation size \(\to 0\)); (2) **higher-order nonlinear error** (the quadratic
approximation fails beyond a finite radius even when \(R\) is negligible).
Directional bias: \(e_{\mathrm{curv}}(v)=|v^\top(H-G)v|/\max(|v^\top Hv|,\varepsilon)\).
**Supports C03, C09 conditionally** (controlled nonlinear benchmarks).

### EXP-003 — stochastic MMD estimator validation

On a controlled Gaussian location–scale simulator with frozen random Fourier
features (float64): the PSD plug-in MMD GGN estimator has the predicted positive
\(O(1/M)\) bias; the cross-seed estimator is unbiased within Monte-Carlo resolution
but may be indefinite at small \(M\); the leading two-dimensional subspace is
recoverable at feasible sample sizes; finite-RFF geometry approaches exact-RBF
geometry as feature dimension grows; the scalar-gradient OPG is residual-dependent
and collapses at exact match while the GGN remains nonzero. **Supports C04, C05,
C06 conditionally** (controlled Gaussian/RFF setting).

### EXP-000 — Brock–Hommes OPG clarification (appendix; C08 not supported)

A real-ABM instance of the gradient-second-moment distinction. At matched BH points
(commit `02d6bd6`), the empirical Fisher / OPG equals the gradient covariance,
\(F_{\mathrm{OPG}}=C_g+\bar g\bar g^\top\), not the GGN — in the chaotic regime it is
nearly orthogonal to the true GGN (top-2 angle 86°), while the true GGN is
multi-directional (effective rank ~4). **C08 → not supported.** Full detail lives in
Layer 1 (`docs/experiments/EXP-000.md`); in the paper this is a demoted appendix
clarification, not a motivating result.

### EXP-004 / EXP-004b — smooth Brock–Hommes geometry and the horizon boundary

In the actual Brock–Hommes model (correct large-β regime, finite-RFF mean
embedding, float64), the GGN equals the exact Hessian **at the fit** to machine
precision (\(\|H-G\|/\|H\|\approx10^{-16}\), residual 0), and off-fit \(R=H-G\)
grows by the same residual-curvature mechanism established on the toy benchmarks
(EXP-002) — now demonstrated in a real ABM. **Supports C03, C09** (conditionally,
extended to BH). The local quadratic is *correct* but predictive only over a very
small radius (\(\rho\approx0.003\) at β=50, →0 at β=80).

The chapter's substantive boundary is the **differentiation-horizon dependence**.
Truncating the horizon under-estimates curvature magnitude by orders of magnitude
at every β and, in chaos, distorts the leading direction (32–48° vs full); the
full-horizon curvature explodes (\(\lambda_1\approx1.5\times10^{8}\) at β=80).
EXP-004b shows this is **intrinsic**: the explosion lives upstream in
\(\partial X/\partial\theta\) (Lyapunov sensitivity), so any pathwise-differentiated
observable inherits it (robust summaries explode *more*, \(1.5\times10^{6}\times\)
at β=80). **This is recorded as a limitation of the exact geometry in chaotic
ABMs, not as a registered claim** (gatekeeper decision, 2026-07-13): it defines the
honest scope of the method rather than a headline result, and needs no dedicated
experiment.

### EXP-005 — smooth SIR posterior validation (the positive inferential result)

On smooth deterministic mean-field SIR with a near-Gaussian generalized posterior
(float64), the prior-relative GGN matches the true local posterior: at the mode
\(H=G\) (residual 0); the prior-relative spectrum gives a data-dominant dimension
\(d_{\mathrm{data}}=3\); the GGN-Laplace leading axes agree with the reference
posterior to **~1°** (top-1/top-2 principal angle 0.68°/1.0°, covariance relative
error 25%); and the profiled generalized-posterior energy matches the GGN quadratic
to **<0.5%** along both stiff and sloppy directions. **Supports C10, C11**
conditionally. Scope: smooth deterministic SIR, near-Gaussian posterior; mild
non-Gaussianity in one direction (t_lock, 20% marginal-width error).

### EXP-006 / EXP-007 — the scientific payoff (policy + observation design)

**EXP-006 (C12).** On smooth SIR, directly-observed functionals are tightly
identified (peak incidence ±0.2%, total cases ±0.3% within a 2-nat fit budget), but
the *counterfactual* intervention value (cases averted by the lockdown) spans
**~0 → 2180 cases** at an essentially unchanged fit — a policy-decisive range — and
that freedom lies along the **sloppiest** GGN direction (f_lock, λ≈8.9e-6). The
diagnostic separates data-determined from data-underdetermined policy conclusions.

**EXP-007 (C13).** The diagnostic is *actionable*: adding a **compliance**
observation (transmission across the lockdown transition) lifts the weak direction
by ~10⁶× (λ 8.9e-6→10), takes the data-dominant dimension from 3→5, and **collapses
the policy range 57×** (width 2181→38 cases). A *redundant* observation (prevalence)
does not help. The geometry tells you *which* observation to buy.

Together these are the strongest results in the project and the paper's scientific
consequence — established on smooth deterministic SIR; EXP-008 (below) closes the
gap to a genuinely stochastic, discrete, surrogate-gradient setting.

## What the manuscript states as scope, not as a gap to close

Two items were weighed and consciously left unestablished for this manuscript
rather than scheduled as further work: whether the SIR policy result is robust
across epidemic regimes / θ* beyond the one configuration studied, and whether the
method scales to large ABMs (compute cost is an Appendix C paragraph, not a claim).
Both are stated as explicit scope limits in the Discussion, not implied to be open
questions the project intends to close next.

## Current paper direction

The paper develops around this argument, presented positively and directly (not as
a chronology of corrections):

1. differentiable ABMs expose derivatives of calibrated representations and are used
   for counterfactual / policy questions;
2. a good fit does not determine a counterfactual — neither a point estimate nor a
   scalar gradient reveals which parameter combinations the data locally constrains;
3. the correct local object is the GGN of the calibrated representation;
4. for MMD, this is the Gram matrix of the feature-mean derivative;
5. finite-sample estimators have a precise bias–variance–definiteness trade-off;
6. GGN curvature is reliable only when residual curvature is controlled;
7. prior whitening separates data information from prior precision;
8. the geometry must be validated on actual ABMs against Hessians, profiles, and
   posterior references;
9. the payoff — a weak direction can leave a policy counterfactual undetermined, and
   the geometry prescribes the observation that constrains it.

The gradient-second-moment ("OPG") distinction is a **demoted technical clarification**
(appendix), anchored to the empirical-Fisher critique — not a motivating pillar.
EXP-000 is a real-ABM instance of it, not the engine of the argument.

## Intended experiment sequence

```text
FOUNDATION — complete
EXP-001: affine implementation correctness            [done]
EXP-002: nonlinear GGN validity                       [done]
EXP-003: stochastic MMD estimator validation          [done]

ABM GEOMETRY — complete
EXP-000: OPG vs true GGN in Brock–Hommes (appendix)    [done — C08 not supported]
EXP-004: full smooth Brock–Hommes geometry            [done — H=G at fit; horizon boundary]
EXP-004b: is the chaotic explosion representation-driven? [done — no, intrinsic]

INFERENCE — complete
EXP-005: smooth SIR posterior and profile validation   [done — C10, C11]

SCIENTIFIC CONSEQUENCE — complete
EXP-006: policy-functional analysis                    [done — C12]
EXP-007: observation design                            [done — C13]

ROBUSTNESS — complete
EXP-008: stochastic/discrete network-SIR, estimated MMD, two operating points
                                                        [done — C14, C15]

DECLINED FOR THIS MANUSCRIPT (stated as scope, not scheduled)
EXP-006b: regime/θ* robustness of the policy result    [declined 2026-07-14]
EXP-009: computational scaling                          [Appendix C paragraph only]
```

## Resolved thread — differentiation-horizon dependence (RQ4)

Settled by EXP-004 + EXP-004b: the full-horizon GGN explodes in the chaotic
Brock–Hommes regime, the geometry depends on `grad_horizon`, and the explosion is
**intrinsic** to \(\partial X/\partial\theta\) (representation-independent).
Gatekeeper decision (2026-07-13): this is recorded as a **limitation of the exact
geometry in chaotic ABMs** in the Discussion — not a registered claim, not a
headline, and it needs no dedicated experiment. EXP-008's own horizon sweep
(C15, on stochastic/discrete network-SIR) is a distinct, additional instantiation
of the same boundary in a non-chaotic transient model — not a re-opening of the
Brock–Hommes chaos question, which stays closed on the evidence above.

## Important terminology

Preferred: local data-informed direction; locally weakly informed parameter
combination; prior-relative local geometry; residual-curvature bias; nonlinear
validity radius; scalar-gradient second moment; centered stochastic-gradient
covariance.

Avoid unless separately established: identified parameter; structural
non-identifiability; global non-identifiability; exact ABM Fisher information;
practical identifiability proved by an eigenvalue; zero-cost curvature.

(Full terminology and writing principles: `WRITING_BRIEF.md`.)

## Fresh strategic-session reading order

1. `docs/paper/OPERATING_MODEL.md` — the two-agent operating model (roles, cycle)
2. `docs/STATUS.md` — current factual state (phase, done, next)
3. this file — interpretation + paper direction
4. `docs/MATH.md` — authoritative mathematical specification (Layer 1)
5. `docs/CLAIMS.md` — the claims ledger (Layer 1; all statuses)
6. `docs/paper/PAPER_ARCHITECTURE.md` — manuscript outline
7. `docs/paper/EVIDENCE_MAP.md` — claim → evidence → figure → section
8. `docs/paper/FIGURE_LEDGER.md` — approved figures + provenance
9. `docs/paper/WRITING_BRIEF.md` — accepted narrative, terminology, style
10. latest `docs/experiments/EXP-*.md` review

## Next action

The empirical arc (C01–C15) is accepted and frozen; no further experiments are
planned before drafting (EXP-006b and EXP-009 considered and declined, 2026-07-14).
The detailed section-by-section manuscript outline — 12 sections + 3 appendices,
explicit asymmetric weighting, manuscript-ready jargon-free section titles — is now
in `PAPER_ARCHITECTURE.md`. The one remaining gate before drafting: approve
FIG-01–FIG-09 in `FIGURE_LEDGER.md` (all are candidates; none approved yet).
