---
title: Local Information Geometry of Differentiable ABMs
status: active
last_verified: 2026-07-11
---

# Local Information Geometry of Differentiable ABMs

## Current objective

Construct and validate the prior-relative generalized Gauss--Newton geometry of calibration objectives for stochastic differentiable agent-based models.

## Current phase

**Phase 1: mathematical specification, code audit, and controlled validation.**

## Canonical documents

1. [Project Charter](01_PROJECT_CHARTER.md)
2. [Mathematical Specification](02_MATHEMATICAL_SPECIFICATION.md)
3. [Claims Ledger](03_CLAIMS_LEDGER.md)
4. [Experiment Registry](04_EXPERIMENT_REGISTRY.md)
5. [Results Ledger](05_RESULTS_LEDGER.md)
6. [Literature Map](06_LITERATURE_MAP.md)
7. [Decision Log](07_DECISION_LOG.md)
8. [Paper Architecture](08_PAPER_ARCHITECTURE.md)

## Immediate next actions

1. Audit the current Brock--Hommes and SIR objectives.
2. Determine whether the current MMD objective is population, biased empirical, or unbiased U-statistic MMD.
3. Implement the analytic linear benchmark.
4. Implement a nonlinear benchmark with controlled curvature.
5. Compute the following four matrices at matched Brock--Hommes parameter values:
   - exact Hessian;
   - true MMD generalized Gauss--Newton matrix;
   - raw per-seed gradient outer-product matrix;
   - centered stochastic-gradient covariance.
6. Compare their eigenspaces, spectra, local quadratic predictions, and sample-size dependence.

## Project status vocabulary

- **proposed**: plausible but not yet tested;
- **in progress**: currently being investigated;
- **supported**: supported by completed evidence;
- **supported conditionally**: supported under explicitly stated conditions;
- **not supported**: evidence currently does not support the claim;
- **rejected**: contradicted mathematically or empirically.

## Working principle

The vault is the canonical source of project state.

```text
question
  -> mathematical claim
  -> registered experiment
  -> result
  -> reviewed interpretation
  -> paper statement
```

Chat histories, notebooks, preliminary figures, and manuscript drafts are not canonical sources of truth.
