---
title: Decision Log
status: active
last_verified: 2026-07-11
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

# Decision template

## DEC-XXX — Title

**Date:**

**Decision**

**Reason**

**Alternatives considered**

**Consequences**

**Evidence**

**Revisit only if**
