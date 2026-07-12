---
title: Literature Map
status: active
last_verified: 2026-07-11
---

# Literature Map

For each source, record what it establishes, what assumptions it makes, what this project uses, and what it does not support.

## A. Differentiable agent-based models

### Quera-Bofarull et al. — Automatic Differentiation of Agent-Based Models

**Establishes**

- pathwise and surrogate gradient constructions for differentiable ABMs;
- gradient-based calibration and generalized variational inference;
- practical issues associated with discrete stochastic operations.

**Used for**

- computational foundation;
- notation for differentiable stochastic simulation;
- motivation for derivative-based calibration geometry.

**Does not establish**

- that raw per-seed scalar-loss OPG is curvature;
- that local gradient geometry proves practical identifiability;
- that surrogate gradients reproduce exact second-order geometry.

**Sections/pages to verify**

- [add exact references]

---

### Quera-Bofarull et al. — Bayesian Calibration of Differentiable ABMs

**Establishes**

- generalized Bayesian or variational calibration of differentiable ABMs;
- posterior dependence among parameters.

**Used for**

- generalized posterior formulation;
- connection between differentiable simulation and variational calibration.

**Does not establish**

- the proposed GGN diagnostic;
- global identifiability.

---

## B. Generalized Bayesian inference

### Bissiri, Holmes, and Walker — A General Framework for Updating Belief Distributions

**Establishes**

- coherent loss-based Bayesian updating.

**Used for**

- generalized posterior interpretation.

**Questions**

- how the learning rate \(w\) should be selected for the target experiments.

---

### Knoblauch, Jewson, and Damoulas — An Optimization-Centric View on Bayes' Rule

**Establishes**

- generalized variational inference framework;
- separation of loss, divergence, and variational family.

**Used for**

- formal relation between generalized posterior and variational approximation.

**Does not establish**

- that a chosen variational family preserves local posterior geometry.

---

## C. Generalized Gauss--Newton and curvature

### Martens — New Insights and Perspectives on the Natural Gradient Method

**Establishes**

- generalized Gauss--Newton constructions for composite losses;
- relationship among GGN, Fisher geometry, and optimization;
- dependence of GGN on the chosen model/output split.

**Used for**

- formal GGN definition;
- positive-semidefinite construction;
- warnings about interpretation.

**Exact sections to revisit**

- [add section references]

---

### Nocedal and Wright — Numerical Optimization

**Establishes**

- nonlinear least-squares and Gauss--Newton theory;
- local approximation and convergence conditions.

**Used for**

- classical mathematical grounding;
- local quadratic approximation.

---

### Kunstner, Balles, and Hennig — Limitations of the Empirical Fisher Approximation

**Establishes**

- empirical gradient outer products are not general Hessian or Fisher approximations;
- loss gradients may vanish near interpolation while useful curvature remains nonzero.

**Used for**

- rejecting the old OPG curvature argument;
- designing comparison experiments.

**Does not imply**

- that all gradient second moments are useless;
- that the true GGN is invalid.

---

## D. MMD and kernel embeddings

### Gretton et al. — A Kernel Two-Sample Test

**Establishes**

- MMD as the RKHS distance between kernel mean embeddings;
- population MMD;
- distinction between biased and unbiased empirical estimators.

**Used for**

- MMD definition;
- distinction between population, V-statistic, and U-statistic objectives.

**Important caution**

- the unbiased finite-sample MMD estimator is not itself an exact squared norm and can be negative.

---

### Chérief-Abdellatif and Alquier — MMD-Bayes

**Establishes**

- generalized Bayesian inference based on MMD-type losses;
- robustness properties under assumptions.

**Used for**

- MMD generalized-posterior context.

**Does not establish**

- the proposed GGN estimator for differentiable ABMs.

---

### Sriperumbudur et al. — Hilbert Space Embeddings and Metrics on Probability Measures

**Establishes**

- characteristic/universal kernel conditions;
- injectivity of kernel mean embeddings.

**Used for**

- population-level distributional interpretation of MMD.

---

## E. Identifiability and profile methods

### Rothenberg — Identification in Parametric Models

**Establishes**

- classical local identification conditions and information-based analysis.

**Used for**

- careful distinction between local rank and stronger identification claims.

---

### Raue et al. — Structural and Practical Identifiability via Profile Likelihood

**Establishes**

- profile-based practical-identifiability analysis;
- limitations of local Hessian intervals;
- use of profiles to motivate additional observations.

**Used for**

- directional profiled-objective validation;
- observation-design motivation.

---

### Wieland et al. — On Structural and Practical Identifiability

**Establishes**

- distinctions among structural, practical, and sensitivity-based notions.

**Used for**

- terminology and limitation statements.

---

### Chis et al. — On the Relationship Between Sloppiness and Identifiability

**Establishes**

- sloppiness and identifiability are related but not equivalent.

**Used for**

- preventing eigenvalue spread from being treated as a complete identifiability result.

---

## F. Sloppy models and information geometry

### Gutenkunst et al. — Universally Sloppy Parameter Sensitivities

**Establishes**

- broad eigenspectra and stiff/sloppy combinations in systems biology models.

**Used for**

- historical and geometric motivation.

**Does not establish**

- that every sloppy direction is practically non-identifiable;
- that the same matrix applies unchanged to likelihood-free ABMs.

---

### Transtrum, Machta, and Sethna — Geometry of Nonlinear Least Squares

**Establishes**

- model-manifold interpretation;
- local geometry of nonlinear least-squares problems;
- parameter combinations rather than coordinate sensitivities.

**Used for**

- geometric interpretation of \(J^\top J\);
- local valley and hyper-ribbon language.

---

## G. ABM-specific sloppiness

### Naumann-Woleske et al.

**Establishes**

- finite-difference local sensitivity/Hessian-like geometry for macroeconomic models;
- use of stiff directions for phase-space exploration.

**Used for**

- comparison with native AD;
- distinction between model exploration and data-informed calibration geometry.

---

### Scarrold — Computing Parameter Sloppiness in Nondeterministic Economic Models

**Establishes**

- surrogate-based recovery of Jacobian-derived sloppiness geometry in stochastic economic models;
- difficulty of estimating weak directions.

**Used for**

- benchmark comparison;
- caution about noisy weak eigenspaces;
- related-model references.

---

## H. Bayesian local information and inverse problems

### Cui et al. — Likelihood-Informed Dimension Reduction

**Establishes**

- prior-relative likelihood-informed subspaces.

**Used for**

- prior whitening and comparison of data geometry with prior geometry.

---

### Spantini et al.

**Establishes**

- optimal low-rank approximation in linear-Gaussian inverse problems.

**Used for**

- interpretation of prior-preconditioned eigenvalues.

---

# Literature entry template

### Citation

**Question addressed**

**Main mathematical object**

**Assumptions**

**What is proved**

**What is demonstrated empirically**

**What this project uses**

**What it does not support**

**Exact pages/sections**

**Open questions**
