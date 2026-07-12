---
name: reference-quera-bofarull-2023-differentiable-bh
description: Quera-Bofarull, Dyer, Calinescu, Wooldridge 2023 (arXiv:2307.01085) — "Some challenges of calibrating differentiable ABMs"; implements a differentiable Brock–Hommes model. Source of our BH equations and the canonical chaotic parameters (4 types, R=1.01, β=120, σ=0.04).
metadata:
  type: reference
---

## Citation

Arnau Quera-Bofarull, Joel Dyer, Anisoara Calinescu, Michael Wooldridge.
**"Some challenges of calibrating differentiable agent-based models."**
arXiv:2307.01085 (2023). (ICML 2023 Differentiable Almost Everything workshop.)

## Why it matters for us

This is the paper our `models/brock_hommes.py` is based on (already cited in its
docstring). It gives the **exact differentiable BH equations** and the canonical
**chaotic** parameter regime. Verified 2026-07-12 by fetching the ar5iv HTML.

## Brock–Hommes equations they implement (verbatim structure)

Deviation-from-fundamental price update (`x_t = p_t − p_t*`):

    x_t = (1/R) [ Σ_j (g_j x_{t-1} + b_j) n_{j,t}  +  σ ε_t ],   ε_t ~ N(0,1)

Discrete-choice fractions (intensity of choice β):

    n_{j,t} = exp(β U_{j,t-1}) / Σ_{j'} exp(β U_{j',t-1})

Fitness / realized profit measure:

    U_{j,t-1} = (x_{t-1} − R x_{t-2}) (g_j x_{t-3} + b_j − R x_{t-2})

**No cost term C_h** appears in their formulation (unlike some BH98 examples).
The risk-aversion / variance normalization `a σ²` is folded into β.

> Our `_step` in `models/brock_hommes.py` matches this exactly: `forecasts =
> g x_{t-1} + b`, `x_t = Σ n·forecasts / R + ε`, `U = (x_t − R x_{t-1})(g x_{t-2}
> + b − R x_{t-1})`, `n = softmax(β U)`. Index-shift check confirms identical.

## Canonical parameters (the chaotic regime)

- **J = 4 trader types** (fundamentalist + 3).
- **R = 1.01**, **σ = 0.04**, **β = 120**.
- Type 1 (fundamentalist): g₁ = 0, b₁ = 0.
- Type 4: g₄ = 1.01, b₄ = 0.
- Calibrated types 2, 3: truth `(g₂, g₃, b₂, b₃) = (0.9, 0.9, 0.2, −0.2)`.

The **intensity of choice β = 120 is the essential ingredient**: it is what
activates the nonlinear strategy-switching that produces bounded complex/chaotic
dynamics. At small β the fractions stay near-uniform and the map is effectively a
weak linear system (near the fundamental steady state).

## Differentiability strategy

- The price path is a smooth (AD-differentiable) deterministic transform of the
  parameters given the noise draw (reparameterized ε) — softmax already smooths
  the discrete choice, so no straight-through/Gumbel needed for BH.
- **Gradient-horizon truncation:** apply `stop_gradient` to `x_{t'}` for
  `t' < t − H` (horizon H ≥ 0) to prune the exploding backprop graph through the
  chaotic recursion. They report **best performance at H = 0** (i.e. keep only
  the most local gradient term), because in the chaotic regime full-horizon
  gradients have enormous variance (sensitive dependence → exploding Jacobian).

## Consequences for our project (BH regime, verified 2026-07-12)

- Our `simulate` is structurally correct. Its default `H=3` (fundamentalist + 2
  free types) still produces bounded complex/chaotic dynamics — the chaos window
  for H=3 at R=1.01, σ=0.04, θ=(β, 0.9,0.2, 0.9,−0.2) is roughly **β ∈ [50, 100+]**
  (trajectory std grows 0.05 → 0.62 as β goes 2 → 100, i.e. up to ~12× the noise).
- **EXP-000 (RES-000 / EXP000_REVIEW) used β = 2–5**, which is the trivial
  near-fundamental regime (std ≈ noise level). That is why its BH representation
  looked effectively rank-1 and why no periodic/chaotic regime was found — a
  wrong-regime error, NOT a model bug. Any future BH geometry study must use the
  large-β regime (β ≳ 50, or the exact 4-type β=120 canonical).
- To reproduce the exact 4-type β=120 canonical example we would need an H=4
  option in `pack_canonical` (currently hardcoded H=3). Optional enhancement.
- The noiseless map has **multiple coexisting attractors** (fundamental x=0 plus
  non-fundamental complex attractors); the canonical σ=0.04 noise sustains the
  fluctuations off the fixed point. Basin depends on x_init and β.

See also `reference_brock_hommes_1998.md` (the original model) and
`reference_quera_bofarull_2025_ad_abm.md` (the later, broader AD-ABM paper).
