---
title: Paper Writing Brief
status: active-draft
last_updated: 2026-07-14
writing_stage: evidence arc frozen (C01–C15) — outline locked in PAPER_ARCHITECTURE.md; awaiting figure approval before prose (spine: methods-first)
---

# Paper Writing Brief

This document contains the positive, self-contained scientific story from which the eventual manuscript will be written.

It must not become a chronological record of corrections.

## Intended reader

Researchers working on:

- differentiable agent-based models;
- simulation-based calibration;
- generalized Bayesian inference;
- identifiability and sensitivity analysis;
- nonlinear inverse problems;
- information geometry;
- likelihood-free inference.

The reader should not be assumed to know any prior OPG-based formulation; the OPG
appears only as an appendix clarification.

## Central scientific question

> How can automatic differentiation be used to construct and validate a local, prior-relative information geometry for calibration objectives in stochastic differentiable agent-based models?

## One-sentence contribution

We formulate the generalized Gauss–Newton geometry of a differentiable ABM's calibrated representation, develop estimators for stochastic MMD objectives, and validate when this local geometry reflects loss curvature and inferential information.

## Main mathematical object

For stochastic simulator \(X(z,\xi)\) and calibrated representation

\[
m(z)=\mathbb E_\xi[\psi(X(z,\xi))],
\]

define

\[
\mathcal L(z)=\frac12\|m(z)-m_y\|_W^2
\]

and

\[
G(z)=D m(z)^\ast W D m(z).
\]

For MMD,

\[
G_{\mathrm{MMD}}(z)
=
J_\eta(z)^\top J_\eta(z),
\qquad
\eta(z)=\mathbb E[\psi(X_z)].
\]

The exact Hessian decomposes as

\[
\nabla^2\mathcal L(z)=G(z)+R(z).
\]

The prior-relative geometry compares \(wG\) with prior precision.

## Intended methodological contributions

1. A correct local GGN formulation for differentiable stochastic ABM calibration.
2. An MMD specialization through the derivative of the feature mean embedding.
3. PSD plug-in and cross-seed estimators with distinct finite-sample properties.
4. A validation framework separating residual-curvature bias from higher-order nonlinear failure.
5. A prior-relative local interpretation.
6. Actual ABM validation on Brock–Hommes and SIR.
7. A scientific consequence through policy analysis or observation design.

## Current empirical findings

Established:

- AD reproduces analytic GGN geometry on affine benchmarks.
- Residual curvature can create irreducible local GGN bias.
- The controlled MMD estimators show predictable bias, variance, and definiteness trade-offs.
- The leading MMD GGN subspace is recoverable at feasible sample sizes in the controlled simulator.
- *(Appendix clarification.)* The gradient second moment (empirical Fisher / OPG) can
  vanish at the exact fit while the GGN remains nonzero, and in a real ABM
  (Brock–Hommes) it equals the gradient covariance \(C_g\), not the GGN
  (\(F_{\mathrm{OPG}}=C_g+\bar g\bar g^\top\); ⟂ to the GGN at 86° in the chaotic
  regime). A real-ABM instance of the empirical-Fisher ≠ GGN distinction (EXP-000) —
  demoted to an appendix, not a motivating result.
- In smooth Brock–Hommes the GGN equals the exact Hessian at the fit
  (\(\|H-G\|/\|H\|\approx10^{-16}\)); off-fit residual curvature reintroduces \(R\),
  and the local quadratic is predictive only over a small radius (≈0.003 at β=50,
  →0 at β=80). In the chaotic regime the exact full-horizon geometry is dominated by
  intrinsic pathwise sensitivity (\(\partial X/\partial\theta\)) and depends
  materially on the differentiation horizon — representation-independent (EXP-004,
  EXP-004b). This is presented as a scope limitation of the exact geometry in
  chaotic ABMs.
- On smooth SIR the prior-relative GGN matches the true local generalized posterior:
  leading axes agree to ~1°, \(d_{\mathrm{data}}=3\) (C10), and the profiled
  posterior energy matches the GGN quadratic to <0.5% along stiff and sloppy
  directions (C11). (EXP-005.) The positive inferential result.
- On smooth SIR a locally weak (sloppy) direction leaves a **policy** quantity
  undetermined: observed functionals are pinned (±0.3%), but the counterfactual
  intervention value spans ~0→2180 cases within a good fit, along the sloppiest GGN
  direction (C12, EXP-006). A targeted added observation (compliance) recovers that
  direction (λ 8.9e-6→10, d_data 3→5) and collapses the policy range 57×, while a
  redundant observation does not (C13, EXP-007). The diagnostic is actionable — the
  paper's scientific consequence.
- On stochastic, discrete, surrogate-gradient network-SIR, at two independent
  operating points (different regime, graph density, and graph realization), the
  leading GGN eigenspace is stable across two surrogate-gradient constructions
  (8.4–8.9° leading-2 angle) and finite-difference-validated (C14); truncating the
  differentiation horizon rotates the leading eigenspace ~80° and collapses its top
  eigenvalue to 1–15% of full (C15, EXP-008). This is the result that earns
  "stochastic agent-based" and makes the controlled MMD estimators load-bearing
  outside the deterministic-SIR chapters — the arc is no longer confined to smooth
  deterministic SIR.

Declined for the first manuscript (not blocking; both stayed candidates in
`docs/STATUS.md` and were consciously not run before writing):

- robustness of the policy result across epidemic regimes/θ* on deterministic SIR
  (would have been EXP-006b) — the manuscript states this as a single-configuration
  scope rather than closing it empirically;
- computational scaling (EXP-009) — remains an Appendix C paragraph, not a claim or
  figure.

## Intended narrative

Section numbers below match `PAPER_ARCHITECTURE.md`. Its section headers are the
intended manuscript titles — jargon-free by construction. This list is the
discourse beat for each; keep it that way when it becomes prose.

### 1. Introduction

Differentiable ABMs are used to answer counterfactual and policy questions, but a
good calibration fit does not determine those answers: many parameter combinations
fit equally well, and neither a point estimate nor a scalar gradient reveals which
combinations — and therefore which counterfactuals — the data locally constrains.
(No villain framing: the alternative gradient-outer-product construction is a
later appendix clarification, not the motivating hook.)

### 2. Background

Four pillars, each anchored in real literature, not in the project's own history:
differentiable ABMs and distributional losses; practical identifiability as
curvature (sloppy models); the state of the art in ABM curvature (finite-difference
and surrogate-network Hessians); the GGN and generalized Bayesian calibration as
the paper's theoretical anchor.

### 3. Framework

Define the simulator, calibrated representation, generalized loss, generalized posterior, and prior-scaled coordinates.

### 4. Local Gauss–Newton geometry

Derive

\[
H=G+R
\]

interpret \(G\) as first-order change in the calibrated representation, and
validate immediately: exact affine recovery, then nonlinear residual-curvature and
validity-radius behavior. Validation sits inside the chapter that introduces the
object, not deferred to a separate later chapter.

### 5. MMD geometry and estimation

Show that

\[
G_{\mathrm{MMD}}=J_\eta^\top J_\eta,
\]

and characterize finite-sample estimator behavior (plug-in vs. cross-seed) on a
controlled simulator.

### 6. Prior-relative local information

Define the prior-relative eigenproblem and its interpretation (data-dominant,
prior-dominant, comparable local precision); sets up §7–9.

### 7. Brock–Hommes: exact validation and the boundary of local geometry

Use Brock–Hommes to test the GGN against the exact Hessian in a real, nonlinear,
chaotic-capable ABM, and to state — as a scope boundary, not an apology — where
pathwise-differentiated local geometry stops being meaningful. Kept deliberately
compact; not co-equal with §8–9.

### 8. SIR: inferential validation

Use smooth SIR to compare prior-relative GGN directions with an independently
computed local posterior reference, and with profiled objectives.

### 9. Policy counterfactuals under locally weak identifiability

The scientific consequence. A sloppy direction leaves a policy counterfactual
undetermined at an excellent fit; a targeted observation recovers it, a redundant
one does not.

### 10. Robustness under stochasticity, discreteness, and surrogate gradients

Run the same loop on a genuinely stochastic, discrete, surrogate-gradient network
model at two independent operating points; test surrogate-gradient and horizon
fidelity.

### 11. Discussion

State the local scope, loss dependence, prior dependence, residual-curvature
limitation, and surrogate-gradient/horizon limitations. One restrained closing
paragraph — no narration of how the project's own understanding evolved.

### 12. Conclusion

Short. Only supported or conditionally supported claims.

## Approved terminology

Use:

- calibrated representation;
- generalized Gauss–Newton matrix;
- local data-informed direction;
- locally weakly informed combination;
- prior-relative local geometry;
- residual-curvature bias;
- nonlinear validity radius;
- stochastic estimator;
- feature mean embedding;
- scalar-gradient second moment;
- centered gradient covariance.

Avoid:

- exact ABM Fisher information unless a likelihood justifies it;
- identified direction without qualification;
- structural non-identifiability from a local eigenvalue;
- practical identifiability proved by the GGN;
- curvature for the historical scalar-gradient OPG;
- zero-cost curvature;
- EXP-xxx / DEC-xxx / C-xx literals, or any internal-process label ("keystone",
  "the payoff", "closing the gap"), anywhere in manuscript prose — see "No
  internal scaffolding in prose" above.

## Claims to avoid

Do not state that:

- every small eigenvalue is a structural null direction;
- posterior concentration necessarily comes from the data;
- GGN always approximates the Hessian near a minimum;
- MMD uniquely permits Gauss–Newton geometry;
- surrogate gradients are exact;
- one threshold universally defines effective dimension;
- the method proves global identifiability;
- one observed trajectory identifies a full stochastic output law.

## Required limitations

The final paper must state that:

1. the geometry is local;
2. it depends on the calibrated representation and loss;
3. it depends on parameter scaling and prior geometry;
4. residual curvature can bias GGN curvature;
5. stochastic estimators have finite-sample error;
6. surrogate derivatives may distort eigenspaces;
7. local Gaussian approximations may fail for curved or multimodal posteriors;
8. policy interpretation requires explicit functional analysis;
9. in chaotic differentiable ABMs the exact GGN is intrinsically dominated by
   pathwise sensitivity (\(\partial X/\partial\theta\)) and depends on the
   differentiation horizon — the exact local geometry is not recoverable there
   without a fixed horizon (accepting bias) or a construction that does not inherit
   the sensitivity explosion.

## Style principles

### Present the final framework directly

Prefer:

> We define the local calibration geometry as...

Avoid:

> We previously used X, but this was incorrect, so instead we...

### Write from accepted evidence

Do not build prose from chat history, experiment chronology, or old drafts.

### No internal scaffolding in prose

The project tracks its own evidence with EXP-xxx labels, DEC-xxx decisions, and
C-xx claim IDs. That machinery is for us, across sessions — it must never surface
as a literal token in manuscript prose, and it must not shape section framing
either. Concretely:

- No "EXP-006 shows...", no "C12", no "commit 91fe277" in prose. Describe the
  experiment by what it is (model, design, conditions), never by internal label.
- No "we initially...", "this closes the gap...", "the keystone experiment...",
  "we previously used X but rejected it." A section exists because it is the next
  question a careful reader would ask of the method, in a logical validation
  order (simple case → nonlinear → stochastic → real ABM → inference →
  consequence) — not because of the order we happened to run it in.
- Every substantive claim traces to external literature or to a self-contained
  described experiment — never to our own prior drafts, rejected framings, or
  "what we learned." The paper reads as the first, direct, correct treatment.
- This applies to section titles too: `PAPER_ARCHITECTURE.md`'s section headers
  are the intended manuscript titles and are kept free of this jargon already
  (e.g. "Policy counterfactuals under locally weak identifiability", not "the
  payoff"; "Robustness under stochasticity, discreteness, and surrogate
  gradients", not "the keystone").

### Avoid correction-oriented transitions

Avoid repeated constructions such as:

- “We do not do X; instead...”
- “Unlike the earlier approach...”
- “To avoid this mistake...”

### Avoid meta-writing

Avoid:

- “This section aims to...”
- “The purpose of this experiment is...”
- “We now turn to...”
- “The key takeaway is...”

State the science directly.

### Avoid rhetorical repetition

Do not restate the contribution at the beginning and end of every section.

### Separate observation from interpretation

State measurements first, then their narrow scientific interpretation.

### Use precise scope language

Prefer:

> In the controlled Gaussian/RFF benchmark...

over:

> In stochastic models generally...

## Approved figures

Current candidates:

- FIG-01: analytic GGN recovery;
- FIG-02: residual curvature and nonlinear validity;
- FIG-03: stochastic MMD estimator behavior;
- FIG-04: Brock–Hommes historical-OPG audit;
- FIG-05: smooth Brock–Hommes GGN and the horizon boundary;
- FIG-06: smooth SIR local posterior geometry.

All are candidates; none is yet approved. Only approved figures from the Figure
Ledger may enter the manuscript.

## Literature positioning

Position the paper at the intersection of:

- differentiable ABM calibration;
- generalized Gauss–Newton methods;
- generalized Bayesian inference;
- MMD and kernel mean embeddings;
- sloppy-model and local identifiability analysis;
- prior-relative likelihood-informed geometry.

Literature notes must distinguish what is proved, what is empirical, and what remains novel.

## Manuscript-writing threshold

Do not begin the full rewrite until the project has:

- controlled mathematical validation; **[met — EXP-001/002]**
- controlled MMD estimator validation; **[met — EXP-003]**
- a genuine Brock–Hommes result; **[met — EXP-000/004/004b]**
- smooth SIR posterior/profile validation; **[met — EXP-005]**
- at least one policy-functional or observation-design result; **[met — EXP-006/007]**
- a result off deterministic smooth SIR that earns "stochastic agent-based"; **[met — EXP-008, two operating points]**
- stable approved figures. **[pending — figures are candidates, none approved]**

**Decision (2026-07-14, gatekeeper): write now.** The empirical arc is frozen at
C01–C15. Two further experiments were considered and declined for this
manuscript — SIR policy-regime robustness (would-be EXP-006b) and computational
scaling (EXP-009) — both left as stated scope limits rather than run. The only
remaining gate is figure approval.

Before that threshold, update only:

- Paper Architecture;
- Figure Ledger;
- Evidence to Paper Map;
- this Writing Brief.

## Fresh manuscript-session input

Provide:

1. Mathematical Specification;
2. Claims Ledger;
3. Results Ledger;
4. Paper Architecture;
5. Figure Ledger;
6. Evidence to Paper Map;
7. Paper Writing Brief;
8. verified literature notes;
9. approved figures and captions.

Do not initially provide:

- old manuscript drafts;
- weekly logs;
- rejected derivations;
- coding-agent conversations;
- archived OPG arguments.

## Current next action

The empirical arc (C01–C15) is complete and frozen; no further experiments are
planned before writing. The detailed section-by-section outline is in
`PAPER_ARCHITECTURE.md` (12 sections + 3 appendices, asymmetric weighting, jargon-
free manuscript titles). Remaining before prose begins: approve figures FIG-01–
FIG-09 in `FIGURE_LEDGER.md` (FIG-09 is now a candidate — EXP-008 landed with
cross-regime robustness). Once approved, begin drafting from the outline,
section by section, applying "No internal scaffolding in prose" throughout.
