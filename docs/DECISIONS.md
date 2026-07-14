---
title: Decision Log
status: active
last_verified: 2026-07-14
---

# Decision Log

Decisions should be concise, dated, and revisited only when new evidence satisfies the stated condition.

---

## DEC-001 — Reject raw OPG as a curvature estimator

**Date:** 2026-07-10

**Decision**

The matrix formed from per-seed scalar-loss gradient outer products will not be interpreted as a GGN, Hessian approximation, or Fisher information matrix.

**Reason**

For a residual loss,

\[
g=J^\top r,
\]

and therefore

\[
gg^\top=J^\top rr^\top J,
\]

which is not \(J^\top J\) in general.

At an exact fit, \(g\) may vanish while \(J^\top J\) remains nonzero.

**Consequences**

- existing spectra must be recomputed;
- old OPG results remain as comparison objects;
- the calibrated-representation Jacobian must be exposed in code;
- the old effective-dimension claims are withdrawn.

**Revisit only if**

A theorem establishes equivalence under explicit assumptions that are verified in the target models.

---

## DEC-002 — Adopt Route 1

**Date:** 2026-07-10

**Decision**

The main route will study local generalized Gauss--Newton geometry rather than posterior-integrated gradient second moments.

**Reason**

Route 1 is closer to the existing code and scientific motivation, while admitting a precise local curvature construction and direct validation against Hessians, profiles, and posterior contours.

**Consequences**

- claims are local;
- the GGN is the primary object;
- posterior-informed-subspace theory is background or future work, not the central method.

---

## DEC-003 — Work in prior-whitened coordinates

**Date:** 2026-07-11

**Decision**

Primary eigendecompositions will be performed in unconstrained, prior-scaled coordinates.

**Reason**

Raw eigenvectors depend on parameter units. Prior whitening gives a meaningful comparison between data curvature and prior precision.

**Consequences**

- parameter transforms must be documented;
- physical-coordinate eigenvectors may be reported only after mapping from whitened coordinates;
- any effective dimension must be defined relative to prior precision.

---

## DEC-004 — Separate data and prior geometry

**Date:** 2026-07-11

**Decision**

The paper will distinguish

\[
G_{\mathrm{data}}
\]

from prior precision and from local posterior precision.

**Reason**

Posterior concentration may be supplied by the prior rather than by the calibration targets.

**Consequences**

- posterior curvature will never be called data information without decomposition;
- the generalized eigenproblem or prior-whitened GGN is the principal inferential object.

---

## DEC-005 — Begin with transparent quadratic losses

**Date:** 2026-07-11

**Decision**

Controlled validation will begin with weighted squared summaries or finite representations before making MMD central.

**Reason**

This isolates GGN correctness from RKHS estimation issues.

**Consequences**

- analytic and nonlinear benchmarks use finite-dimensional representations;
- MMD receives a separate estimator-validation experiment.

---

## DEC-006 — Distinguish MMD objectives

**Date:** 2026-07-11

**Decision**

Population MMD, biased empirical MMD, and unbiased U-statistic MMD will be treated as distinct objectives.

**Reason**

Only population MMD and the biased empirical mean-embedding norm have immediate squared-residual representations.

**Consequences**

- code audit must identify the exact objective;
- no residual-based derivation may silently switch among these forms.

---

## DEC-007 — Smooth SIR before discrete SIR

**Date:** 2026-07-11

**Decision**

Inferential validation will be performed first on a smooth SIR model.

**Reason**

This provides a trustworthy derivative and posterior reference before introducing surrogate-gradient bias.

**Consequences**

- discrete stochastic SIR is a derivative-fidelity stress test;
- conclusions from the smooth model are not automatically transferred.

---

## DEC-008 — Remove preconditioning from the main contribution

**Date:** 2026-07-11

**Decision**

Optimization preconditioning is not part of the minimum viable paper.

**Reason**

The paper first needs to establish the validity of the geometry. Optimization would add a second contribution and require recomputation with the true GGN.

**Consequences**

- preconditioning may appear later as an appendix or separate project;
- old OPG preconditioning results are not retained as evidence.

---

## DEC-009 — Rewrite from a blank manuscript

**Date:** 2026-07-11

**Decision**

The new manuscript will not use the old draft as its structural skeleton.

**Reason**

The old paper was organized around an invalid OPG-curvature identification and contains overextended discussion and conclusions.

**Consequences**

- old material can supply model provenance and historical context;
- claims, equations, figures, and discussion must be rebuilt from the canonical vault.

---

## DEC-010 — Two-layer docs for a two-agent operating model

**Date:** 2026-07-12

**Decision**

Split the documentation into two layers matching a two-agent + gatekeeper workflow:
**Layer 1 — internal research memory** (`docs/` root: MATH, CLAIMS, DECISIONS,
experiments, papers, STATUS) owned by the implementation specialist; **Layer 2 —
paper-facing memory** (`docs/paper/`: OPERATING_MODEL, STRATEGIC_HANDOFF,
PAPER_ARCHITECTURE, WRITING_BRIEF, EVIDENCE_MAP, FIGURE_LEDGER) owned by the
scientific-lead conversation + Pietro. The manuscript is written from Layer 2.

**Reason**

The research history necessarily contains corrections (e.g. the OPG
re-interpretation, DEC-001). That history matters for integrity but is not the
paper's scientific argument. Separating the accepted state (Layer 2) from the full
history (Layer 1) prevents another correction-shaped manuscript and lets a fresh
agent with no experiment log write from the accepted state.

**Alternatives considered**

- A single narrative brief bolted onto the lean docs — rejected: no clean owner,
  drifts on every experiment.
- A separate results ledger (`RES-00x`) as in the archived vault — rejected: a
  second results record beside the EXP files; `CLAIMS.md` is the single ledger and
  Layer 2 cites experiment + commit.

**Consequences**

- Two definitions of done: implementation specialist updates Layer 1 (the three
  places) per experiment and never edits `docs/paper/`; the scientific lead updates
  Layer 2 per *accepted* result.
- Layer 2 is a **derived projection** — it references Layer-1 facts by ID and never
  invents them; on disagreement Layer 1 wins.
- `CLAUDE.md` and `docs/README.md` updated to encode the roles and the layer split.
- The uploaded `scientific-lead-agent/` drafts became the seed of `docs/paper/`,
  re-anchored onto the lean docs and de-staled to the EXP-000 result.
- `docs/archive/vault/` stays frozen; nothing resurrected wholesale.

**Revisit only if**

The two-agent workflow is abandoned, or Layer 2 duplication starts causing drift
despite the reference-by-ID rule.

---

## DEC-011 — RQ4 (differentiation-horizon dependence / chaotic GGN explosion) is a scope limitation, not a headline claim

**Date:** 2026-07-14

**Decision**

The finding that the GGN depends materially on the differentiation horizon — and
that in the chaotic Brock–Hommes regime the exact pathwise GGN is dominated by
chaotic (Lyapunov) sensitivity — is recorded as a **scope limitation** of the
local-geometry diagnostic. It will not be promoted into a headline claim, and no
dedicated differentiation-fidelity experiment is spun up to elevate it. Smooth SIR
(EXP-005/006/007) remains the inferential core; Brock–Hommes chaos is an honest
boundary of applicability.

**Reason**

EXP-004/004b established the effect is **intrinsic** — it lives in `∂X/∂θ`
(Lyapunov-driven), not in the choice of representation — so no change of observable
removes it and any pathwise-differentiated observable inherits it (robust summaries
explode more, not less). That makes it a property of pathwise differentiation of
chaotic simulators, i.e. a boundary on where the local GGN is meaningful, rather
than a result the paper must defend as a contribution. The MVP claim arc
(C01–C13) is complete without promoting it.

**Alternatives considered**

- Promote RQ4 to a headline limitation-claim plus a dedicated "EXP-008
  differentiation-fidelity" experiment — rejected: over-invests in a negative
  boundary result and conflates it with the genuinely open stochastic/discrete
  keystone, for which the EXP-008 label is now reserved.

**Consequences**

- `docs/STATUS.md` drops the "Key open finding (RQ4 — needs a decision)" block;
  RQ4 becomes a stated scope limitation (Discussion/scope), owned by Layer 2 at
  manuscript time.
- The label **EXP-008** is freed for its intended meaning: the stochastic/discrete
  MMD derivative-fidelity keystone (C14/C15, per [DEC-007](#dec-007--smooth-sir-before-discrete-sir)),
  not RQ4 promotion.
- A fixed-horizon or derivative-free/ensemble GGN for chaotic regimes remains
  future work, not a blocker.

**Evidence**

EXP-004 (H=G at the BH fit to 1e-16, GGN horizon-biased, validity radius →0 in
chaos); EXP-004b (explosion intrinsic; robust summaries explode more, 1.5e6× at
β=80).

**Revisit only if**

A fixed-horizon or ensemble/derivative-free GGN construction makes the
chaotic-regime geometry stable and informative, turning the limitation into a
tractable result.

---

# Decision template

## DEC-XXX — Title

**Date:**

**Decision**

**Reason**

**Alternatives considered**

**Consequences**

**Evidence**

**Revisit only if**
