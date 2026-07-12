---
name: reference-brock-hommes-1998
description: Brock & Hommes 1998 JEDC — "Heterogeneous beliefs and routes to chaos in a simple asset pricing model"; the original adaptive-belief-system model our BH simulator implements. Intensity of choice β drives a period-doubling / breaking-of-invariant-curve route to chaos.
metadata:
  type: reference
---

## Citation

William A. Brock, Cars H. Hommes. **"Heterogeneous beliefs and routes to chaos in
a simple asset pricing model."** Journal of Economic Dynamics and Control 22(8–9),
1235–1274 (1998).

> Note (2026-07-12): the canonical PDFs would not render via WebFetch (cert /
> binary). The equations below are the accepted BH98 adaptive-belief-system form,
> cross-checked against the differentiable implementation in Quera-Bofarull et al.
> 2023 (arXiv:2307.01085; see `reference_quera_bofarull_2023_differentiable_bh.md`),
> which our code matches. Exact BH98 example parameter tables were not re-verified
> from the primary PDF and should be confirmed there before being quoted in the paper.

## Model (adaptive belief system, ABS)

Mean-variance demand for a risky asset (gross risk-free return R), agents choose
among H predictor/belief types. In deviations from the fundamental `x_t = p_t − p_t*`:

- **Predictor (belief) of type h:** `f_{h,t} = g_h x_{t-1} + b_h`
  (`g_h` = trend/extrapolation coefficient, `b_h` = bias).
- **Temporary-equilibrium price:** `R x_t = Σ_h n_{h,t} f_{h,t}` (+ noise),
  i.e. `x_t = (1/R) Σ_h n_{h,t}(g_h x_{t-1} + b_h)`.
- **Fitness = realized profit:** `U_{h,t} = (x_t − R x_{t-1})(g_h x_{t-2} + b_h −
  R x_{t-1}) / (a σ²) − C_h`, where `a` = risk aversion, `σ²` = conditional
  variance of excess returns, `C_h` = per-type information cost (e.g. a positive
  cost for the fundamentalist in some examples).
- **Discrete choice (Gibbs / multinomial logit):**
  `n_{h,t} = exp(β U_{h,t-1}) / Σ_k exp(β U_{k,t-1})`, β = **intensity of choice**.

Type 0 is usually the **fundamentalist** `g_0 = b_0 = 0` (believes `x → 0`).

## Route to chaos — the qualitative story

- The **fundamental steady state x = 0 always exists.** Its local stability, and
  the existence of non-fundamental steady states / cycles, depend on the belief
  mix and on β.
- **β (intensity of choice) is the primary bifurcation parameter.** As β
  increases, agents switch strategies more sharply on small fitness differences;
  the system undergoes a sequence of bifurcations (period-doubling and/or
  breaking of an invariant curve — "rational routes to randomness") ending in
  **bounded chaotic** price fluctuations.
- **Boundedness mechanism:** near x=0, destabilizing trend-followers (g > R)
  dominate and push prices away; far from x=0 the trend extrapolation overshoots,
  those types incur losses, fitness shifts back to the fundamentalist (who bet on
  mean reversion and profits at the turning points), folding the trajectory back.
  This destabilize-near / stabilize-far interplay, sharpened by large β, is what
  bounds the dynamics. A **fundamentalist cost C** and/or **biased or contrarian
  types** are used in specific BH98 examples to place the system in the chaotic
  window.
- Chaos is reported for the fundamentalist-vs-trend and multi-type configurations
  at high β; diagnosed via bifurcation diagrams and (positive) largest Lyapunov
  exponents.

## Relation to our implementation (verified 2026-07-12)

- `models/brock_hommes.py` implements exactly this ABS with the Quera-Bofarull
  2023 normalization: **no explicit `a σ²` or cost `C`** (both folded into β), H=3
  by default (fundamentalist + 2 free types), `θ = (β, g₁, b₁, g₂, b₂)`.
- **Dynamics confirmed correct:** with R=1.01, σ=0.04, `θ=(β, 0.9,0.2, 0.9,−0.2)`,
  trajectory std grows from ≈ noise (0.05 at β=2) to 0.62 at β=100 (≈12× noise),
  and the noiseless map shows bounded complex/chaotic attractors for β≈50–100
  (hundreds of distinct tail values, positive sensitivity to initial conditions).
- **Because β is the knob, β must be large (≳50) to see the characteristic BH
  dynamics.** Studies run at small β (e.g. EXP-000's β=2–5) sit in the trivial
  near-fundamental regime and will not exhibit the model's rich behaviour.

## Open follow-ups

- Confirm the exact BH98 numeric example tables (β values, g_h, b_h, C, a, σ²)
  from the primary PDF before quoting them in the manuscript.
- Consider an H=4 option to reproduce the canonical 4-type β=120 chaos exactly.
