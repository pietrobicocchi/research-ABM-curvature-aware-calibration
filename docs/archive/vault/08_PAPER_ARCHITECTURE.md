---
title: Paper Architecture
status: draft
last_verified: 2026-07-11
---

# Paper Architecture

This is a claim--evidence outline, not manuscript prose.

## Provisional title

**Local Information Geometry of Differentiable Agent-Based Models**

Subtitle option:

**Gauss--Newton diagnostics for generalized Bayesian calibration**

## Core narrative

1. Differentiable ABMs make first derivatives available.
2. Scalar gradients do not by themselves reveal locally confounded parameter combinations.
3. Raw gradient outer products are not general curvature estimators.
4. The correct local object is the GGN of the calibrated representation.
5. Prior whitening is required for a data-information interpretation.
6. The geometry must be validated against exact Hessians, posterior references, and profiled objectives.
7. The result is a local diagnostic, not a global identifiability theorem.

---

## 1. Introduction

### Problem

Gradient-based calibration returns parameters and possibly an approximate posterior, but does not automatically reveal which parameter combinations are locally informed by the calibration targets.

### Gap

Existing ABM sloppiness analyses use finite differences or learned surrogates. Differentiable ABMs expose native derivative operations, but a correct second-order construction and inferential validation remain missing.

### Method

Construct

\[
G(z)=D m(z)^\ast W D m(z)
\]

using AD, and interpret it relative to the prior geometry.

### Claims allowed

- C02
- C20
- eventually C01, C04, C10, C16 if supported

### Claims forbidden

- C07
- C17
- C18
- C19

### Evidence

- mathematical derivation;
- EXP-001;
- EXP-003;
- smooth SIR validation.

---

## 2. Differentiable generalized calibration

### Definitions

\[
z,\quad T,\quad X,\quad \xi,\quad \psi,\quad m,\quad \mathcal L,\quad \pi_w.
\]

### Required content

- constrained versus unconstrained parameters;
- stochastic simulator;
- calibrated representation;
- generalized posterior;
- exact versus surrogate derivatives.

### Evidence

- Mathematical Specification;
- Quera-Bofarull literature entries.

---

## 3. Local Gauss--Newton geometry

### Main derivation

\[
\nabla^2\mathcal L=G+R,
\]

\[
G=Dm^\ast W Dm.
\]

### Proposition candidates

1. Hessian decomposition.
2. Positive semidefiniteness.
3. Directional interpretation.
4. Monte Carlo consistency of selected estimators.

### Limitations stated immediately

- locality;
- dependence on representation and loss;
- residual-curvature term;
- parameterization.

### Evidence

- algebra;
- EXP-001;
- EXP-002.

---

## 4. MMD geometry and estimation

### Definition

\[
\mathcal L_{\mathrm{MMD}}
=
\frac12\|\mu_z-\mu_y\|_{\mathcal H}^2.
\]

### Correct GGN

\[
G_{\mathrm{MMD}}
=
J_\mu^\ast J_\mu.
\]

### Estimators

- PSD plug-in;
- cross-seed estimator;
- direct kernel-derivative implementation.

### Essential distinction

Population MMD, biased empirical MMD, and unbiased U-statistic MMD are different objectives.

### Comparison object

Raw scalar-loss OPG is included only as a diagnostic comparator.

### Evidence

- EXP-000;
- EXP-003.

---

## 5. Prior-relative local information

### Main object

With a standard-normal prior,

\[
wGv_k=\lambda_kv_k.
\]

With general prior precision,

\[
wGv_k=\lambda_kP_\pi v_k.
\]

### Interpretation

- data-dominant;
- prior-dominant;
- comparable local precision.

### Terminology

Use local data-informed directions, not global identifiability.

### Evidence

- mathematical derivation;
- EXP-005;
- profile validation.

---

## 6. Estimation and computation

### Topics

- explicit Jacobians;
- JVP/VJP products;
- matrix-free \(Gv\);
- eigensolvers;
- Monte Carlo uncertainty;
- subspace bootstrap or repeated-batch analysis;
- computational scaling.

### Evidence

- EXP-003;
- EXP-009.

---

## 7. Controlled validation

### 7.1 Linear benchmark

Claim:

AD reproduces exact geometry.

Evidence:

EXP-001.

### 7.2 Nonlinear benchmark

Claim:

GGN validity is local and can be quantified through a validity radius.

Evidence:

EXP-002.

### 7.3 MMD estimator benchmark

Claim:

Selected estimator recovers the MMD GGN under tested conditions.

Evidence:

EXP-003.

---

## 8. Differentiable-ABM experiments

### 8.1 Smooth Brock--Hommes

Questions:

- GGN versus exact Hessian across regimes;
- local validity radius;
- behavior near nonlinear regime changes;
- comparison with historical OPG.

Evidence:

EXP-000 and EXP-004.

### 8.2 Smooth SIR

Questions:

- prior-relative local posterior geometry;
- timing--strength combinations;
- profiles and posterior contours.

Evidence:

EXP-005.

### 8.3 Policy-functional analysis

Question:

Does a locally weak direction materially alter a policy prediction?

Evidence:

EXP-006.

### 8.4 Observation design

Question:

Which additional observations add information in weak directions?

Evidence:

EXP-007.

### 8.5 Discrete stochastic SIR

Question:

How faithful is the surrogate derivative geometry?

Evidence:

EXP-008.

---

## 9. Discussion

### Required topics

1. Local versus global conclusions.
2. Sensitivity versus practical identifiability.
3. Data geometry versus prior geometry.
4. Dependence on observations, summaries, kernel, and loss.
5. Surrogate-gradient and truncation bias.
6. Single-trajectory limitations.
7. Model misspecification.
8. Policy implications stated at the correct level.

### Prohibited discussion style

- no broad claim that weak directions make all counterfactuals invalid;
- no claim that the method proves structural non-identifiability;
- no repeated self-summary;
- no speculative future-work catalogue.

### Ending

One restrained paragraph identifying the main unresolved limitation and the next scientific step.

---

# Figure plan

## FIG-01 — Analytic benchmark

Exact GGN, AD GGN, Hessian, and OPG.

## FIG-02 — Local validity

Hessian--GGN eigenspace agreement and quadratic-prediction error versus distance.

## FIG-03 — MMD estimator convergence

Bias, variance, and principal angles versus simulator sample count.

## FIG-04 — Brock--Hommes comparison

Exact Hessian, true MMD GGN, OPG, and gradient covariance across regimes.

## FIG-05 — SIR local posterior geometry

Reference posterior contours with Hessian and GGN approximations.

## FIG-06 — Directional profiles

Straight-line loss, profiled loss, Hessian quadratic, and GGN quadratic.

## FIG-07 — Policy functional

Profiled fit change and policy-output change along a weak direction.

## FIG-08 — Observation design

Prior-relative spectra and weak-direction uncertainty under richer observations.

## FIG-09 — Surrogate fidelity

Subspace error by surrogate, population size, sample count, and differentiation horizon.

## FIG-10 — Computational scaling

Runtime, memory, and subspace accuracy.

---

# Abstract gate

The abstract may be drafted only when:

- C01 is supported;
- at least one MMD estimator claim is supported;
- C10 or an equivalent inferential-validation claim is supported;
- C12 or C13 provides scientific consequence;
- the main limitations have been empirically characterized.

# Conclusion gate

The conclusion may contain only claims marked supported or supported conditionally in the Claims Ledger.
