# EXP-004b — BH geometry with a summary-statistic representation (fork-resolver)

**Status:** done
**Authoritative code commit:** run `outputs/EXP-004b/*/`; deterministic
**Relates to:** EXP-004 (resolves its open fork)

## Question

Is EXP-004's chaotic GGN horizon-explosion **intrinsic** to differentiating
through chaotic BH dynamics, or an **artifact** of the trajectory-RFF
representation (which maximally amplifies sensitivity)? Test with a finite
**robust-summary** representation `S(X)` = {mean, std, skew, kurt, ACF lags
1/2/3/5/10, tail quantiles 5/25/50/75/95}, whitened by across-seed std (the
finite-summary route DEC-005 recommends).

## Result — explosion is INTRINSIC (worse, not better)

Full-horizon Jacobian-norm explosion vs horizon-1:

| representation | β=50 | β=80 |
|---|---|---|
| trajectory-RFF (EXP-004) | ~85× | ~1.7×10⁴× |
| **robust summaries (this)** | **292×** | **1.5×10⁶×** |

The summary representation does **not** tame the explosion — it is larger. Leading
`λ₁` still reaches ~1.3×10² (β=50) and **~1.9×10⁹** (β=80) at full horizon; the
leading eigenvector is still distorted at short horizons in chaos (angle-vs-full
38–66°); the predictive validity radius is still ~0.

## Interpretation

The explosion lives **upstream, in `∂X/∂θ`** (the Lyapunov/butterfly effect of the
chaotic recursion), not in the observable. Any smooth trajectory observable obeys
`∂S/∂θ = (∂S/∂X)(∂X/∂θ)`, so it inherits the exploding `∂X/∂θ` regardless of the
representation. Therefore, in a **chaotic** differentiable ABM the exact
(full-horizon) GGN is dominated by chaotic sensitivity for **any** pathwise-
differentiated representation; horizon truncation is the only thing that keeps it
finite, at the cost of biasing the geometry.

## Claim impact

None (interpretation / scope). **Strengthens the RQ4 limitation** from EXP-004:
the horizon-dependence and sensitivity-domination are representation-independent
and intrinsic to chaos. A meaningful identifiability geometry in the chaotic
regime needs either a fixed/principled differentiation horizon (accepting bias)
or a construction that does not inherit `∂X/∂θ` explosion (e.g. finite-differences
of expectations / ensemble geometry) — a scientific-lead decision, candidate for
EXP-008.
