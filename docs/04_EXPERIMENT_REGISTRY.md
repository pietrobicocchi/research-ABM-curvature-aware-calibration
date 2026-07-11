---
title: Experiment Registry
status: active
last_verified: 2026-07-11
---

# Experiment Registry

Experiments must be registered before their results are interpreted.

---

## EXP-000 — Brock--Hommes four-matrix audit

### Question

What did the old OPG experiments actually measure?

### Objects compared

\[
H_{\mathrm{exact}}
=
\nabla^2\mathcal L,
\]

\[
G_{\mathrm{MMD}}
=
J_\mu^\ast J_\mu,
\]

\[
F_{\mathrm{OPG}}
=
\frac1M\sum_mg_mg_m^\top,
\]

\[
C_g
=
\frac1M\sum_m(g_m-\bar g)(g_m-\bar g)^\top.
\]

### Variables

- stable, periodic, and chaotic Brock--Hommes regimes;
- simulator sample count \(M\);
- kernel bandwidth;
- distance to fitted parameter;
- biased versus unbiased empirical MMD, if both are implemented.

### Metrics

- relative matrix error;
- eigenvalue ratios;
- principal angles between leading and trailing subspaces;
- local quadratic-prediction error;
- behavior as the residual approaches zero.

### Pass criterion

No pass/fail claim is imposed. The purpose is diagnostic classification of the old OPG.

### Supports

- C08
- clarification of rejected C07

---

## EXP-001 — Analytic linear GGN recovery

### Question

Does the AD implementation recover the analytic matrix \(A^\top WA\)?

### Model

\[
m(z)=Az,
\qquad
\mathcal L(z)=\frac12\|Az-y\|_W^2.
\]

### Objects compared

- analytic GGN;
- explicit-Jacobian AD GGN;
- matrix-free \(Gv\);
- exact Hessian;
- raw scalar-gradient OPG;
- finite-difference Hessian.

### Variables

- \(P\in\{5,20,100\}\);
- full-rank and rank-deficient \(A\);
- controlled condition numbers;
- float32 versus float64.

### Metrics

- relative Frobenius error;
- eigenvalue error;
- principal-angle error;
- recovered rank;
- runtime and memory.

### Pass criterion

AD GGN agrees with the analytic matrix to numerical precision in float64.

### Supports

- C01

---

## EXP-002 — Nonlinear local-validity benchmark

### Question

When does the GGN approximate the Hessian and actual local loss variation?

### Model

Use a low-dimensional nonlinear residual with a curved valley and analytically or numerically accessible derivatives.

### Objects compared

\[
H(z),\qquad G(z),\qquad R(z)=H(z)-G(z).
\]

### Variables

- distance from optimum;
- residual magnitude;
- nonlinearity parameter;
- perturbation direction and radius.

### Metrics

- relative matrix error;
- leading-subspace principal angles;
- eigenvalue error;
- local quadratic-prediction error;
- empirical validity radius.

### Pass criterion

A nontrivial neighborhood exists in which leading GGN and Hessian subspaces agree and quadratic prediction satisfies the prespecified tolerance.

### Supports

- C03
- C09

---

## EXP-003 — MMD GGN estimator benchmark

### Question

Can the MMD GGN be estimated accurately at acceptable sample cost?

### Estimators

### PSD plug-in

\[
\widehat G_V
=
\widehat J_\mu^\top\widehat J_\mu.
\]

### Cross-seed

\[
\widehat G_U
=
\frac{1}{M(M-1)}
\sum_{m\ne n}A_m^\top A_n.
\]

### Comparators

- high-sample reference;
- direct kernel-derivative implementation;
- raw scalar-gradient OPG.

### Variables

- simulator sample count \(M\);
- feature dimension \(D\);
- kernel bandwidth;
- simulator noise;
- parameter dimension.

### Metrics

- estimator bias and variance;
- Frobenius error;
- principal-angle error;
- eigenvalue error;
- PSD violations;
- runtime and memory.

### Pass criterion

The leading subspace converges to the reference at a feasible sample size.

### Supports

- C04
- C05
- C06

---

## EXP-004 — Smooth Brock--Hommes geometry

### Question

How do exact Hessian and GGN geometry behave across stable, periodic, and chaotic regimes?

### Procedures

1. Calibrate to a reference representation.
2. Compute exact Hessian and GGN at the fitted point.
3. Compare eigenspaces and spectra.
4. Estimate validity radii.
5. Run directional and profiled objectives.
6. Track local geometry across selected parameter paths.

### Metrics

- Hessian--GGN principal angles;
- relative eigenvalue error;
- validity radius;
- profile curvature;
- regime-dependent rotation.

### Pass criterion

At least one regime exhibits a meaningful local region in which GGN geometry is predictive and interpretable.

### Supports

- C03
- C09

---

## EXP-005 — Smooth SIR posterior validation

### Question

Does the prior-relative GGN agree with the local generalized posterior?

### Model

Smooth SIR with parameters such as

\[
(\beta,\gamma,I_0,t_{\mathrm{lock}},f_{\mathrm{lock}}).
\]

### Required definitions

- parameter transforms;
- priors;
- observation design;
- calibration loss;
- generalized-posterior learning rate;
- intervention equations.

### Objects compared

- exact Hessian;
- GGN;
- prior-relative GGN;
- Hessian Laplace approximation;
- GGN Laplace approximation;
- high-quality posterior reference;
- profiled calibration loss;
- profiled posterior energy.

### Metrics

- contour orientation;
- covariance error;
- eigenspace principal angles;
- marginal interval error;
- profile agreement;
- non-Gaussianity diagnostics.

### Pass criterion

The GGN captures the principal local posterior orientation over a nontrivial neighborhood.

### Supports

- C10
- C11

---

## EXP-006 — SIR policy-functional analysis

### Question

Does a locally weakly informed direction materially change a policy-relevant output?

### Candidate functionals

\[
Q_{\mathrm{peak}}=\max_t I_t,
\]

\[
Q_{\mathrm{total}}=\sum_t I_t,
\]

\[
Q_{\mathrm{intervention}}
=
\text{cases without intervention}
-
\text{cases with intervention}.
\]

### Procedure

Move along a weak prior-relative direction while reoptimizing nuisance directions.

### Outputs

Plot:

\[
\mathcal P_v(a)-\mathcal P_v(0)
\]

against

\[
Q(z_a)-Q(\hat z).
\]

### Pass criterion

A direction exists along which the profiled fit changes little while a policy functional changes materially.

### Supports

- C12

---

## EXP-007 — Observation-design study

### Question

Which additional observations supply local information in weak directions?

### Observation designs

1. incidence only;
2. incidence and prevalence;
3. denser temporal observations;
4. group-stratified incidence;
5. direct intervention-compliance information.

### Metrics

- prior-relative eigenvalues;
- rotation of weak and strong subspaces;
- profiled-objective steepness;
- posterior interval changes;
- policy-functional uncertainty.

### Pass criterion

At least one added observation measurably increases information in a previously weak direction.

### Supports

- C13

---

## EXP-008 — Discrete stochastic SIR derivative fidelity

### Question

How do stochasticity, surrogate gradients, and differentiation truncation affect local geometry?

### Comparators

- smooth mean-field reference;
- finite differences of expected observables using common random numbers;
- alternative surrogate estimators;
- direct perturbation responses.

### Variables

- simulator sample count;
- population size;
- surrogate type;
- surrogate temperature;
- differentiation horizon;
- random-number coupling.

### Metrics

- eigenvalue error;
- leading- and trailing-subspace principal angles;
- quadratic-prediction error;
- variability across seed batches.

### Pass criterion

Important subspaces are either robust or their distortions can be characterized clearly.

### Supports

- C14
- C15

---

## EXP-009 — Computational scaling

### Question

At what scales is AD-based GGN estimation computationally useful?

### Methods

- explicit Jacobian;
- matrix-free \(Gv\);
- Lanczos eigensolver;
- exact Hessian-vector products;
- finite-difference Hessian.

### Variables

\[
P,\qquad \dim(m),\qquad M.
\]

### Metrics

- simulator executions;
- JVP/VJP calls;
- wall-clock time;
- memory;
- leading-subspace accuracy.

### Pass criterion

AD-based methods show a documented advantage at the tested scales.

### Supports

- C16

---

# New experiment template

## EXP-XXX — Title

### Question

### Hypothesis

### Mathematical objects

### Variables

### Controls

### Metrics

### Pass criterion

### Failure interpretation

### Code entry point

### Expected output directory

### Related claims
