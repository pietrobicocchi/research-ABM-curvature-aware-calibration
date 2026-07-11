# Brock–Hommes in JAX — implementation notes & supervisor-prep

This is a defence-prep document, not a tutorial (for the line-by-line tutorial
see `docs/brock_hommes_code_guide.md`). The goal here is: (1) a tight summary
of *how* the BH model is implemented and *why* each JAX choice was made, (2)
a list of peculiarities that look odd until explained, and (3) a Q&A section
rehearsing the questions Wooldridge / Calinescu / Farmer are most likely to
ask, with model answers.

Code lives in `src/curvature_calib/models/brock_hommes.py` (102 lines —
read it, it's short). Tests in `tests/test_brock_hommes.py`.

---

## 1. The model in one paragraph

Brock & Hommes (1998): a heterogeneous-agent asset-pricing model. State
`x_t` = price deviation from fundamental value. Three trader "types": type 0
is a fundamentalist (always forecasts `x=0`), types 1 and 2 are linear
trend-followers `f_h = g_h x_{t-1} + b_h`. At each step, the *fraction*
`n_h` of agents using rule `h` is a softmax over each rule's recent
*realised profit* `U_h` — agents switch toward whichever rule has been
paying off, with switching intensity controlled by `beta`. The price update
is the profit-weighted average forecast, discounted by the gross
risk-free return `R`, plus i.i.d. Gaussian noise. Five free parameters:

```
theta = (beta, g1, b1, g2, b2)
```

Canonical value used throughout the project: `theta* = (3.0, 1.2, 0.2, 1.2, -0.2)`,
`R = 1.1`, `sigma = 0.05`, `T = 200`.

**Physical reading of theta\*:** `beta=3` is moderate switching intensity
(not winner-take-all); `g1=g2=1.2 > 1` means *both* trend-following types are
individually destabilising (they amplify deviations); `b1=+0.2, b2=-0.2` are
equal-and-opposite biases — one type is structurally bullish, the other
bearish. The symmetric combination `(b1+b2)/√2` turns out to be the **stiffest**
(best-identified) direction in parameter space, and `beta` the **sloppiest**
— see §6.

---

## 2. The three update equations

```
n_t   = softmax(beta * U_{t-1})                                  # (3,) fractions
x_t   = (1/R) * sum_h n_{h,t} * (g_h * x_{t-1} + b_h) + eps_t     # price
U_h,t = (x_t - R*x_{t-1}) * (g_h * x_{t-2} + b_h - R*x_{t-1})     # profit
```

Memory structure: `x_t` needs `x_{t-1}` (1 lag); `U_t` needs `x_{t-2}` (because
the profit of rule `h` at time `t` is "what would I have earned if, at `t-1`,
I'd traded on the forecast I made looking at `x_{t-2}`"). So the carry state
is **3 lags worth of information**: `(x_{t-1}, x_{t-2}, U_{t-1})`.

---

## 3. The implementation, primitive by primitive

### 3.1 `BHParams` — a `NamedTuple` pytree

```python
class BHParams(NamedTuple):
    beta: jax.Array  # scalar
    g: jax.Array     # (3,)
    b: jax.Array     # (3,)
```

`pack_canonical(theta)` turns the flat 5-vector into this struct, hardcoding
`g[0]=b[0]=0` for the fundamentalist. **Why a NamedTuple and not a dict or a
dataclass?** JAX needs to know how to flatten/unflatten any object that flows
through `vmap`/`grad`/`scan`/`jit` — these are "pytrees". `NamedTuple` is a
pytree out of the box (no registration needed), unlike a plain `dataclass`.

### 3.2 `_step` — one time step, written as `(carry, input) -> (new_carry, output)`

This is the function `jax.lax.scan` calls `T` times. `params` and `R` are
**closed over** (captured from the enclosing scope) — they don't change
across the scan, only `state` and `eps` do.

### 3.3 `jax.lax.scan` instead of a Python `for` loop

```python
_, xs = jax.lax.scan(step_fn, init, eps)   # eps: (T,), xs: (T,)
```

**Why this matters:** JAX traces Python control flow at compile time. A
`for t in range(200): ...` loop would get *unrolled* into 200 copies of the
step's computation graph — 200x the compile time and memory, and `T` becomes
a baked-in constant rather than a runtime value. `scan` compiles the step
function **once** and applies it `T` times inside a single XLA loop
construct. It also makes the *backward pass* (reverse-mode autodiff through
the loop) tractable — JAX has a built-in VJP rule for `scan` that doesn't
require unrolling either.

### 3.4 Randomness: explicit PRNG keys (the reparameterisation trick)

```python
eps = sigma * jax.random.normal(key, (T,), dtype=theta.dtype)
```

JAX has **no global random state** — `np.random.seed(...)` has no JAX
equivalent. Every call that needs randomness takes an explicit `key`
(a `uint32[2]` array), and keys are never reused — they're `split()` into
fresh sub-keys. This is what makes `jax.vmap` over "independent random
seeds" trivial and reproducible: `jax.random.split(master_key, M)` gives `M`
keys that are each individually reproducible and collectively independent.

The differentiability angle: `eps` is drawn **once**, given `key`, and is
then a fixed array — `theta` never appears inside `jax.random.normal`. The
noise enters the dynamics *additively* (`x_t = ... + eps_t`), so
`d(x_t)/d(theta)` is well-defined and doesn't need to differentiate "through"
the sampling operation itself. This is exactly the location-scale
reparameterisation trick from VI/VAEs, applied for free because the noise is
additive and parameter-independent by construction.

### 3.5 `jax.vmap` — running `M` seeds in parallel

```python
X = jax.vmap(lambda k: simulate(theta, k))(keys)   # (M, T)
```

No Python loop over seeds, no batch dimension threaded manually through
`_step`. `vmap` adds a batch axis to every operation in `simulate`
automatically. This is used twice in the pipeline: once to produce `M`
trajectories forward, and again (see §3.7) to compute `M` *per-seed
gradients* in parallel.

### 3.6 `grad_horizon` — truncated BPTT

```python
n_pre = T - grad_horizon
state_pre, xs_pre = jax.lax.scan(step_fn, init, eps[:n_pre])
state_pre = jax.tree.map(jax.lax.stop_gradient, state_pre)
xs_pre = jax.lax.stop_gradient(xs_pre)
_, xs_post = jax.lax.scan(step_fn, state_pre, eps[n_pre:])
return jnp.concatenate([xs_pre, xs_post])
```

Two `scan`s instead of one. The **forward numbers are bit-identical** to the
untruncated version (same `eps`, same recursion) — only the *backward* graph
changes: `jax.lax.stop_gradient` severs the adjoint at the boundary, so
gradients only flow through the last `grad_horizon` steps. This is the
standard truncated-BPTT trick (Quera-Bofarull et al. 2023), used in the
project's Phase-1 "horizon-bias killswitch" experiment to test whether the
OPG eigenstructure depends on how much of the trajectory you backprop
through. **Result: the eigenvalue *hierarchy* (which directions are
stiff/sloppy) is stable across `H ∈ {5,...,200}`, but eigenvalue *magnitudes*
shift by up to 70x.** ([[phase1-horizon-bias-result]])

### 3.7 The hard part: per-seed gradients via two chained `jax.vjp` calls

This lives in `calibration/per_seed_grads.py`, not in `brock_hommes.py`
itself, but it's the reason the simulator is written the way it is.

The naive `jax.grad(loss)(theta)` gives you **one** gradient — the mean over
all `M` seeds, with the per-seed information already summed away inside the
MMD computation. To recover the `M` *individual* contributions, you manually
do the chain rule in two stages:

```python
X = vmap_simulate(simulate, theta, keys)                 # (M, T) forward
L, vjp_X = jax.vjp(lambda x: mmd_sq(x, Y_ref), X)        # stage 1
(dL_dX,) = vjp_X(1.0)                                    # (M, T): dL/dx_m

def one_seed_grad(key, cotangent):                       # stage 2
    _, vjp_t = jax.vjp(lambda t: simulate(t, key), theta)
    (g,) = vjp_t(cotangent)
    return M * g

per_seed = jax.vmap(one_seed_grad)(keys, dL_dX)          # (M, P)
mean_grad = jnp.mean(per_seed, axis=0)                   # == jax.grad(loss)(theta)
opg = (per_seed.T @ per_seed) / M                        # (P, P)
```

- **Stage 1** treats the `(M, T)` trajectory tensor `X` as the "input" to the
  loss and gets back `dL/dX`, an `(M, T)` array — row `m` says "how would the
  loss change if trajectory `m` shifted".
- **Stage 2** re-runs the backward pass through `simulate` *per seed*, using
  row `m` of `dL/dX` as the cotangent. By the chain rule this is exactly
  `(dL/dx_m) · (dx_m/dtheta) = dL/dtheta` *attributed to seed m*.
- **The factor `M`**: each `g_m` is scaled by `M` so that
  `mean_m(g_m) = mean_grad = jax.grad(loss)(theta)` exactly (sanity-checked
  in tests). Without the `M`, `mean(g_m)` would be `mean_grad / M`.
- **`jax.vmap(one_seed_grad)`** runs all `M` of these per-seed VJPs in
  parallel rather than in a Python loop.

The output you actually care about is the **OPG matrix**
`F̂ = (1/M) Σ_m g_m g_m^T`, a `(5,5)` matrix. Its eigendecomposition
(`jnp.linalg.eigh` on the symmetrised `0.5*(F+F.T)`) is the diagnostic: large
eigenvalue → stiff (well-identified) direction, small eigenvalue → sloppy
(unidentified) direction.

---

## 4. Why JAX rather than PyTorch / plain NumPy

If asked "why JAX":

1. **`vmap` composes with `vjp`/`grad`/`scan`.** The per-seed-gradient trick
   in §3.7 is a `vmap` of a `vjp` of a `scan`. In PyTorch this would mean
   either a Python loop over `M` separate `.backward()` calls (slow, and you
   have to manually zero/retain the graph each time) or the `functorch`/
   `torch.func` vmap-of-grad machinery, which is the same idea but bolted on
   later — JAX was built around this composability from the start.
2. **Functional purity removes a whole class of bugs.** No `.zero_grad()`,
   no in-place mutation of parameters, no accidental graph retention. Every
   transformation (`grad`, `vmap`, `jit`, `scan`) is a pure function of pure
   functions.
3. **Explicit PRNG keys** make "run `M` independent stochastic seeds and get
   reproducible per-seed gradients" a one-line `vmap`, with no global RNG
   state to worry about across `vmap`/`jit` boundaries.
4. **`jax.lax.scan`** gives an efficient, autodiff-aware fixed-length loop —
   essential for `T=200` sequential steps without 200x unrolling.
5. (Practical) `jit` + XLA gives free CPU/GPU portability for the same code.

---

## 5. Peculiarities / gotchas — things that look odd but are deliberate

| # | What you'll see | Why | If pressed |
|---|---|---|---|
| 1 | `jax.config.update("jax_enable_x64", True)` scattered in scripts, **inside `main()`**, not at module level | JAX defaults to **float32**. The OPG spectrum spans ~7-8 orders of magnitude (λ₁≈1e-2, λ₅≈1e-9) — float32 (~7 significant digits) puts the smallest eigenvalues *at* the precision floor, contaminating the "is this eigenvalue really ~0" claim. float64 is needed for the diagnostic to be trustworthy. | This is a **global, process-wide flag** in JAX (not a per-array dtype). Setting it at *import time* in a script leaked into the test suite when that script's module was imported anywhere — tests run after it would silently run in float64 too. Fixed in commit `159b712` by moving the call inside `main()`, which only runs when the script is executed directly, not on import. |
| 2 | Model file default is `R=1.01`; almost every experiment script overrides to `R=1.1` | `R=1.01` (~1%/period gross return) is closer to the original Brock-Hommes paper's calibration. `R=1.1` was the value used once exploration started and then kept for consistency across all later experiments (canonical θ* etc. were tuned around it). | **Be ready for "why 1.1 and not the textbook 1.01?"** — honest answer: it was the value settled on early during exploration (it widens/shifts the stable-fixed-point basin and the gradient scaling) and then frozen for cross-experiment consistency; it was not re-derived from the original paper's calibration. If asked whether results are sensitive to this, the honest answer is "not systematically checked" — flag as a limitation if pushed. |
| 3 | Regimes are labelled "stable / periodic / chaotic" at `beta = 1 / 5 / 10`, but **the deterministic skeleton (sigma=0) collapses to the fixed point `x=0` for all of these** — true deterministic chaos only onsets around `beta ≈ 36-40` | This is the single most likely "gotcha" question, especially from someone who knows the original BH bifurcation diagram. | The labels reflect **noise-driven persistence**, not deterministic attractors: at sigma=0.05, β=1 has fast-decaying ACF (ρ₁≈0.39), β=5 looks visually identical (ρ₁≈0.31), β=10 has slow-decaying ACF (ρ₁≈0.78) — that's the only one that's visibly different. **If asked "show me the chaotic attractor at β=10"**: say plainly that at R=1.1 the deterministic dynamics at β=10 sit in the stable zone (see `fig_05_bh_bifurcation`, the bifurcation diagram), and that the project's "chaotic" label refers to a noise-driven high-persistence *stochastic* regime, not the deterministic route-to-chaos (which is real, but needs β≈40+). This is documented honestly in the booklet and in `[[bh-regimes-are-stochastic]]`. |
| 4 | `softmax(beta * U)` needs **no Gumbel-softmax / straight-through surrogate** | Type-switching is a *continuous* mixture (fractions `n_h ∈ [0,1]`, not a hard discrete choice of one rule per agent at the population level), so `softmax` is already smooth and exactly differentiable. | Contrast with `models/network_sir.py` / `models/surrogates.py`, where *individual* agent infection events are discrete Bernoulli draws and **do** need `gumbel_sigmoid` / straight-through estimators. BH is the "easy", fully-smooth model in the project; network-SIR is the "hard", surrogate-gradient model. If asked "does BH need surrogate gradients?" — **no**, and that's a deliberate reason it's the primary/simplest benchmark. |
| 5 | `x_init` perturbs **both** `x_{t-1}` and `x_{t-2}` to the same value | The carry is `(x0, x0, zeros(H))` — both lags start equal, profits start at zero. | Used to probe the noiseless bifurcation diagram (sigma=0): with `x_init≠0` you can watch whether the deterministic dynamics return to 0 (stable) or not. Default `x_init=0.0` means "start exactly at the fundamental". |
| 6 | Per-seed gradients are scaled by `M` (`return M * g`) | So that `mean(per_seed_grads) == jax.grad(loss)(theta)` exactly — the OPG formula `(1/M) Σ g_m g_m^T` then has the right normalisation. | If asked to derive this: `dL/dtheta = sum_m (dL/dx_m)(dx_m/dtheta)`. Each `vjp` call with cotangent `dL_dX[m]` returns exactly the `m`-th term of that sum, i.e. `(1/1)` not `(1/M)` — but `mean_grad` is defined as the *average* contribution, hence `g_m := M × (term_m)` so that `(1/M) Σ g_m = Σ term_m = dL/dtheta`. |
| 7 | `F̂ = (1/M) Σ g_m g_m^T` is called the **OPG matrix** or **empirical curvature matrix**, *never* "the Fisher information" or "empirical Fisher" | Kunstner, Hennig & Balles (NeurIPS 2019) showed the OPG/empirical-Fisher matrix is **not** a good approximation to the true Fisher (or Hessian) in general — and any reviewer who knows that paper will reject the conflation immediately. | The legitimising argument used here is different and *does* hold: squared MMD is the squared RKHS norm of a residual (a difference of kernel mean embeddings), so the **Generalised Gauss-Newton (GGN)** interpretation applies via residual structure (Schraudolph 2002; Martens 2020 §8) — independent of any likelihood/Fisher claim. `F̂` is honestly described as a **stochastic GGN approximation**, and it is *also* finite-`M` (stochastic) and biased by `grad_horizon` truncation / surrogate gradients where those apply. See `[[framing-kunstner-opg-not-fisher]]`. |
| 8 | `eigendecompose` symmetrises `F` via `0.5*(F+F.T)` before `jnp.linalg.eigh` | `F̂ = (per_seed.T @ per_seed)/M` is symmetric *in exact arithmetic*, but floating-point matmul can introduce tiny asymmetries; `eigh` requires (or assumes) a symmetric/Hermitian input. | `eigh` (vs general `eig`) is used because it's faster, numerically stabler, and *guarantees real eigenvalues* for a symmetric matrix — appropriate since `F̂` is positive semi-definite by construction (sum of outer products `g g^T`). |
| 9 | Calibration needs `M ≥ 2P` seeds (P=5, so M≥10; in practice M=64) | `rank(F̂) ≤ min(M, P)`. With `M < P`, `F̂` is rank-deficient and some eigenvalues are *exactly* zero by construction (an artefact of too few samples, not a statement about identifiability). `M ≥ 2P` gives headroom for the bootstrap CIs to be informative. | If asked "why 64 and not 10": 64 was chosen for stable bootstrap confidence intervals (500 resamples) and to keep the smallest eigenvalues away from the `M`-induced rank-deficiency floor, not from a formal power calculation. |
| 10 | The calibration loop uses `jax.random.fold_in(master_key, t)` rather than storing/threading a PRNG state | `fold_in` deterministically derives a new key from `(master_key, integer)` without mutating `master_key`. Iteration `t`'s seeds are reproducible given `(master_key, t)` alone — useful for re-running/debugging a single iteration in isolation. | Standard JAX idiom for "a different but reproducible key per loop iteration" without a stateful RNG object. |

---

## 6. Anticipated supervisor questions (with model answers)

**Q: Walk me through what happens when I call `simulate(theta, key)`.**
A: `pack_canonical` turns the 5-vector into `(beta, g, b)` with the
fundamentalist's `g0=b0=0`. We draw `T` i.i.d. `N(0, sigma^2)` noise values
from `key` up front. Then `jax.lax.scan` runs `_step` `T` times: each step
computes the softmax mixture weights from last period's profits, the new
price as the profit-weighted forecast discounted by `R` plus noise, and the
new profit vector for next period. It returns the `(T,)` array of prices.

**Q: Why is this differentiable at all — isn't there a `softmax` (discrete
choice) and randomness in there?**
A: `softmax` is smooth everywhere — it's a continuous relaxation of "choose
the best", and here it literally *is* the population fraction, not an
approximation of a hard choice, so no surrogate is needed. The randomness is
additive Gaussian noise drawn once from an explicit key *before* the
recursion — `theta` doesn't influence the sampling, only the deterministic
recursion that the noise is added into. So `d(x_t)/d(theta)` is an ordinary
(if long) chain rule through `T` applications of `_step`.

**Q: How do you get a *gradient*, not just a loss, out of a stochastic
simulator with no closed-form likelihood?**
A: We never need a likelihood. The loss is MMD² between the *distribution* of
simulated trajectories (M independent seeds) and the reference data — a
two-sample distance computable from samples alone. JAX differentiates
*through the simulator* itself (reverse-mode AD over the `scan`), so
`d(MMD²)/d(theta)` is exact (to floating-point precision), not a
finite-difference or score-function (REINFORCE) estimate.

**Q: What's the OPG matrix and why do you compute it?**
A: While computing the mean gradient, the per-seed gradients `g_m`
(`m=1..M`) exist as intermediate quantities. `F̂ = (1/M) Σ g_m g_m^T` is their
second moment — a `(5,5)` curvature matrix, computed at essentially zero
extra cost. Its eigendecomposition tells us which combinations of parameters
the loss can/can't distinguish — large eigenvalue = stiff/identifiable
direction, near-zero eigenvalue = sloppy/non-identifiable direction. That's
the project's diagnostic.

**Q: Is `F̂` the Fisher information matrix?**
A: No — and that distinction matters. Kunstner et al. (2019) showed the
"empirical Fisher" (same formula, in a likelihood context) is generally a
poor Hessian approximation. We don't invoke a likelihood at all; the
licensing argument is that squared MMD has *residual structure* (it's a
squared RKHS norm of a kernel-mean-embedding difference), so `F̂` is a
**generalised Gauss-Newton** approximation to the loss Hessian — a different
and more defensible claim. We call it the OPG matrix or empirical curvature
matrix, never "the Fisher".

**Q: How do you compute per-seed gradients efficiently — don't you need M
separate backward passes?**
A: Conceptually yes, but `jax.vmap` parallelises them — there's one
`jax.vjp(simulate, theta)` per seed, but `vmap` traces it once and runs all
`M` as a single batched XLA computation, not a Python loop with `M` sequential
graph traversals.

**Q: What's `grad_horizon` for, and does truncating the gradient change the
simulation output?**
A: No — the forward trajectory is bit-identical regardless of
`grad_horizon` (verified by a test: `test_grad_horizon_does_not_change_primal_pass`).
Only the backward pass changes: `stop_gradient` is applied to the carry state
at the truncation boundary, so gradients only flow through the last `H`
steps. This is truncated BPTT, used to test how sensitive the OPG
eigenstructure is to how much history you backprop through (Phase 1
"horizon-bias killswitch": the eigenvalue *ranking* is stable across
`H ∈ {5,...,200}`, but magnitudes shift up to 70x — so the diagnostic's
*qualitative* claims are robust, *quantitative* ones need matched horizons).

**Q: Why `jax.lax.scan` instead of a Python loop, given `T=200` isn't huge?**
A: Two reasons: (1) compile time / graph size — 200 unrolled steps is 200x
the IR for both forward and backward passes; (2) it composes correctly with
`vmap` and `jit` and has a well-defined, efficient autodiff rule, so the
whole pipeline (`vmap(vjp(scan))`) stays a single compiled computation rather
than `M × T` separately-traced operations.

**Q: At β=10 you call this "chaotic" — show me the chaotic attractor.**
A: (See gotcha #3 above.) Be upfront: at `R=1.1`, the *deterministic*
skeleton at β=10 converges to the fixed point `x=0` — true deterministic
chaos in this model needs `β ≳ 36-40` (matches the original BH bifurcation
structure). The "chaotic" label in this project refers to a *stochastic*
regime distinguished by slow-decaying autocorrelation (persistence) under
noise, not a deterministic strange attractor. This is documented and the
labels are kept for consistency with earlier framing, but the distinction is
real and shouldn't be glossed over.

**Q: Why MMD instead of e.g. matching moments or a likelihood?**
A: No tractable likelihood (the model is a nonlinear stochastic recursion
with regime-switching). MMD is a kernel two-sample test statistic — it
vanishes iff the two distributions are equal (for a characteristic kernel
like Gaussian RBF), it's differentiable, and crucially it has *residual
structure* (squared RKHS-norm of a difference), which is what licenses the
GGN/OPG curvature interpretation in point 7 above. The bandwidth is set via
the median heuristic on pooled pairwise distances, computed under
`stop_gradient` so the optimiser can't "cheat" by shrinking the kernel
bandwidth to make MMD² artificially small.

**Q: Did Claude write this code? How much do you actually understand vs.
just accepted?**
A: Be honest and concrete rather than defensive: the implementation was
written with AI assistance, but the *design decisions* — which parameters to
calibrate, the choice of MMD/median-heuristic, the per-seed-gradient/OPG
construction, the LM preconditioner, the experiments and their
interpretation — are the project's contributions and you can derive/defend
each formula from first principles (the chain-rule argument in §3.7 above is
the one to be most fluent in, since it's the mathematical core). The value of
AI assistance here was in JAX *idiom* (scan signatures, vmap/vjp
composition, pytree registration) — mechanical, not conceptual. If a
supervisor asks you to derive the per-seed gradient formula on a whiteboard,
you should be able to (it's 4 lines: define `g_m`, write the chain rule,
note the `M` rescaling, define `F̂`).

---

## 7. One-line cheat sheet

- **Model**: 3 trader types, softmax switching on realised profit, additive
  Gaussian noise, 5 free params `(beta, g1, b1, g2, b2)`.
- **scan**: efficient, autodiff-friendly fixed-length loop over `T=200` steps.
- **vmap**: M=64 independent seeds run as one batched computation.
- **Differentiable because**: softmax is smooth, noise is additive &
  parameter-independent (reparameterisation trick) — no surrogates needed
  (unlike network-SIR).
- **Per-seed gradients**: two chained `jax.vjp` calls (loss→trajectories,
  then trajectories→theta per seed), rescaled by `M`.
- **OPG matrix** `F̂ = (1/M) Σ g_m g_m^T`: a *stochastic GGN* approximation
  (licensed by MMD's residual structure), **not** the Fisher information
  (Kunstner 2019).
- **eigh** on symmetrised `F̂`: eigenvalues = identifiability spectrum,
  eigenvectors = identifiable parameter combinations.
- **float64 matters** because the spectrum spans ~8 orders of magnitude;
  `jax_enable_x64` is global and must be set inside `main()`, not at import
  time (bug fixed in `159b712`).
- **"Chaotic" β=10 regime is stochastic, not deterministic** — the genuine
  deterministic route to chaos needs β≈36-40 at R=1.1. Don't get caught
  promising a strange attractor that isn't there.
