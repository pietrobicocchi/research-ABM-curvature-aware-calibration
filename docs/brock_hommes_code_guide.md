# Brock-Hommes Codebase Guide

A walkthrough of how the Brock-Hommes (BH) pipeline is built — from the economic model through the loss, gradient machinery, diagnostic, and calibration loop. JAX concepts are explained as they appear.

---

## Table of Contents

1. [What is the Brock-Hommes model?](#1-what-is-the-brock-hommes-model)
2. [JAX foundations you need first](#2-jax-foundations-you-need-first)
3. [The simulator — `models/brock_hommes.py`](#3-the-simulator)
4. [The loss — `losses/mmd.py`](#4-the-loss)
5. [Per-seed gradients and the OPG matrix — `calibration/per_seed_grads.py`](#5-per-seed-gradients-and-the-opg-matrix)
6. [The diagnostic — `calibration/diagnostic.py`](#6-the-diagnostic)
7. [Bootstrap confidence intervals — `calibration/bootstrap.py`](#7-bootstrap-confidence-intervals)
8. [The calibration loop — `calibration/calibrate.py` + `preconditioner.py`](#8-the-calibration-loop)
9. [How everything connects — end-to-end data flow](#9-end-to-end-data-flow)

---

## 1. What is the Brock-Hommes model?

Brock & Hommes (1998) is a **heterogeneous-agent asset pricing model**. Traders hold beliefs about where a stock price is going. Different "types" of traders use different simple forecasting rules. The key insight is that as traders observe who is making profit, they switch between these rules — and this switching produces rich, sometimes chaotic, dynamics even though each individual rule is trivial.

### The state variable

The model tracks `x_t` — the **deviation of the stock price from its fundamental value** (the price you'd pay if you discounted future dividends at rate `R`). So `x_t = 0` means the market is fairly valued; `x_t > 0` means it is overpriced.

### Trader types

There are `H = 3` types:
- **Type 0 (fundamentalist)**: believes price will return to fundamental — forecasts `f_0 = 0`.
- **Type 1**: linear trend-follower with slope `g_1` and bias `b_1` — forecasts `f_1 = g_1 * x_{t-1} + b_1`.
- **Type 2**: another trend-follower with slope `g_2` and bias `b_2`.

### Parameters to calibrate

```
theta = (beta, g_1, b_1, g_2, b_2)   # shape (5,)
```

- `beta` (β): **intensity of choice** — how aggressively traders switch to the most profitable rule. High β → winner-takes-all switching. Low β → uniform mixing.
- `g_1, g_2`: trend slopes. If `g_h * x_{t-1}` amplifies the deviation, the type is destabilising.
- `b_1, b_2`: biases (asymmetric bull/bear beliefs).

### The dynamics (three equations per step)

```
n_t   = softmax(β · U_{t-1})                             # fraction of each type
x_t   = (1/R) · Σ_h  n_h · (g_h · x_{t-1} + b_h) + ε_t  # price update
U_h,t = (x_t − R·x_{t-1}) · (g_h·x_{t-2} + b_h − R·x_{t-1})  # realised profit
```

- `n_t` tells us how popular each trader type is right now.
- `x_t` is a weighted average forecast, discounted by `R`, plus noise `ε_t ~ N(0, σ²)`.
- `U_h,t` is the profit from having followed rule `h` one step ago. It drives the next `n`.

The model has only **one step of memory** for the price state, but two steps of memory for the profit calculation (because profit depends on what you forecast two steps ago, comparing it against what happened one step ago).

---

## 2. JAX foundations you need first

Before reading the code, here are the four JAX ideas you will encounter repeatedly.

### 2.1 JAX arrays are like NumPy arrays, but immutable and traced

```python
import jax.numpy as jnp

x = jnp.array([1.0, 2.0, 3.0])  # like np.array
y = x + 1                         # returns a NEW array; x is unchanged
```

JAX arrays live on a device (CPU/GPU/TPU). When JAX transforms a function (e.g. for autodiff or JIT), it replaces concrete values with **abstract tracers** — symbolic placeholders that record operations without executing them. This is what makes all of JAX's magic possible.

### 2.2 `jax.lax.scan` — the loop primitive

Standard Python `for` loops are unrolled when JAX traces them, which blows up compile time for long sequences. `jax.lax.scan` is the solution: a **fixed-length loop that JAX compiles into a single efficient operation**.

```python
# Pattern:
# new_carry, stacked_outputs = jax.lax.scan(fn, init_carry, xs)
#
# fn(carry, x) -> (new_carry, output)
# init_carry: initial state
# xs: sequence of inputs, shape (T, ...)
# Output: stacked_outputs has shape (T, ...)

def fn(carry, x):
    new_carry = carry + x
    output = carry * x
    return new_carry, output

final_carry, outputs = jax.lax.scan(fn, 0.0, jnp.array([1.0, 2.0, 3.0]))
```

In our BH simulator, `scan` runs the 3-equation dynamics for `T` steps, carrying `(x_prev, x_prev2, U)` as the state.

### 2.3 `jax.vmap` — vectorise over a batch axis

`vmap` turns a function that operates on a single example into one that operates on a batch — without writing any for loops:

```python
def f(key):
    return jax.random.normal(key, (5,))

keys = jax.random.split(jax.random.PRNGKey(0), M)  # (M, 2)
xs = jax.vmap(f)(keys)  # (M, 5)  — M independent draws, run in parallel
```

We use this heavily: to run `M` simulator seeds in parallel, and to compute per-seed gradients in parallel.

### 2.4 `jax.vjp` — Vector-Jacobian Products (reverse-mode autodiff)

JAX's autodiff is built around **VJPs** (the "backward pass" of backpropagation).

```python
f = lambda x: x ** 2          # scalar output (imagine summed)
y, vjp_fn = jax.vjp(f, 3.0)   # y=9.0; vjp_fn is the backward function
(grad,) = vjp_fn(1.0)          # grad = df/dx at x=3 times cotangent 1.0 = 6.0
```

`jax.vjp(f, x)` returns:
1. The primal output `f(x)`.
2. A **closure `vjp_fn`** that, given a cotangent `v` (same shape as `f(x)`), computes `v^T · (∂f/∂x)` — the gradient pre-multiplied by `v`.

This is exactly the chain rule. We use it explicitly in `per_seed_grads.py` to chain two backward passes together.

### 2.5 `jax.lax.stop_gradient` — blocking gradient flow

```python
sigma = jax.lax.stop_gradient(median_heuristic(X, Y))
```

This tells JAX: "compute `sigma` normally in the forward pass, but pretend it is a constant during the backward pass." Gradients will not flow back through `sigma`. Used in two places:
- **Bandwidth in MMD**: the median-heuristic bandwidth is non-differentiable at ties; we treat it as a data-adaptive constant.
- **Gradient-horizon truncation**: when we only want gradients through the last `H` steps of a trajectory.

---

## 3. The simulator

**File**: `src/curvature_calib/models/brock_hommes.py`

### 3.1 Parameter packing

```python
class BHParams(NamedTuple):
    beta: jax.Array  # scalar
    g: jax.Array     # (H,) slopes for each type
    b: jax.Array     # (H,) biases for each type
```

`BHParams` is a **NamedTuple** — a plain Python data structure that JAX can treat as a "pytree" (a nested container it knows how to traverse for autodiff). JAX pytrees are important: `jax.tree.map` applies a function to every leaf array in a pytree.

```python
def pack_canonical(theta):
    zero = jnp.zeros((), dtype=theta.dtype)
    g = jnp.stack([zero, theta[1], theta[3]])  # type 0 has g=0 (fundamentalist)
    b = jnp.stack([zero, theta[2], theta[4]])
    return BHParams(beta=theta[0], g=g, b=b)
```

This converts the flat 5-vector `theta` into a structured `BHParams`. The fundamentalist (type 0) is hardcoded with `g=0, b=0` — it always forecasts the fundamental.

### 3.2 The step function

```python
def _step(state, eps, params, R):
    x_prev, x_prev2, U = state
    n = jax.nn.softmax(params.beta * U)          # (H,) fractions
    forecasts = params.g * x_prev + params.b      # (H,) linear forecasts
    x_t = jnp.sum(n * forecasts) / R + eps        # weighted mean, discounted + noise
    excess = x_t - R * x_prev                     # excess return
    U_new = excess * (params.g * x_prev2 + params.b - R * x_prev)  # profits
    return (x_t, x_prev, U_new), x_t
```

This is the per-step function consumed by `scan`. It has the signature `(carry, input) -> (new_carry, output)`:
- **carry** `(x_prev, x_prev2, U)`: the two most recent prices and current profit vector.
- **input** `eps`: the noise draw for this step.
- Returns the updated carry and the new price `x_t` (which gets stacked by `scan`).

Notice: `params` and `R` are **closed over** — they are fixed across all time steps. Only the carry and the per-step noise vary.

### 3.3 The simulate function

```python
def simulate(theta, key, T=500, R=1.01, sigma=0.05, H=3, x_init=0.0, grad_horizon=None):
    params = pack_canonical(theta)
    eps = sigma * jax.random.normal(key, (T,), dtype=theta.dtype)
    x0 = jnp.asarray(x_init, dtype=theta.dtype)
    init = (x0, x0, jnp.zeros((H,), dtype=theta.dtype))

    step_fn = lambda s, e: _step(s, e, params, R)
    _, xs = jax.lax.scan(step_fn, init, eps)
    return xs  # (T,)
```

Key points:
- **`key`** is a JAX random key (a 2-element uint32 array). JAX has **explicit, functional randomness** — there is no hidden global random state. You always pass a key in, and split it to generate sub-keys for different uses. This makes simulations reproducible and compatible with JIT.
- **`x_init=0.0`** for both lagged prices means we start at the fundamental. Setting it non-zero lets you explore bifurcations.
- The `scan` returns `(final_carry, stacked_xs)`. We discard the final carry and keep `xs` of shape `(T,)`.

### 3.4 Gradient-horizon truncation

The `grad_horizon` argument implements **truncated BPTT (backpropagation through time)**. When `grad_horizon < T`, the forward pass is run in two stages:

```python
n_pre = T - grad_horizon
state_pre, xs_pre = jax.lax.scan(step_fn, init, eps[:n_pre])
state_pre = jax.tree.map(jax.lax.stop_gradient, state_pre)  # cut gradient here
xs_pre = jax.lax.stop_gradient(xs_pre)
_, xs_post = jax.lax.scan(step_fn, state_pre, eps[n_pre:])
return jnp.concatenate([xs_pre, xs_post])
```

The simulation output is **identical** for any `grad_horizon` — same numbers. Only the backward pass changes: gradients only flow through the last `grad_horizon` steps. This is used in the Phase 1 horizon-bias experiment to measure how sensitive the OPG eigenstructure is to truncation.

---

## 4. The loss

**File**: `src/curvature_calib/losses/mmd.py`

We can't compare a single simulated trajectory to the data — there's noise. Instead we compare **distributions** of trajectories to the target distribution. The loss is **squared Maximum Mean Discrepancy (MMD²)**.

### 4.1 The idea

Given `M` simulated trajectories `X = [x¹, ..., x^M]` (each of length `T`) and reference data `Y = [y¹, ..., y^N]`, MMD² measures the distance between the distribution of `X` and the distribution of `Y` using a kernel:

```
MMD²(P, Q) = E[k(x, x')] + E[k(y, y')] − 2·E[k(x, y)]
```

where `k(x, y) = exp(−‖x−y‖² / 2σ²)` is the Gaussian RBF kernel. If `P = Q`, all three terms cancel and MMD² = 0.

We use the **unbiased U-statistic estimator** which excludes diagonal terms (comparing a sample with itself):

```python
def mmd_sq_unbiased(X, Y, sigma):
    Kxx = rbf_kernel(X, X, sigma)
    Kyy = rbf_kernel(Y, Y, sigma)
    Kxy = rbf_kernel(X, Y, sigma)
    sum_xx = (jnp.sum(Kxx) - jnp.sum(jnp.diag(Kxx))) / (M * (M - 1))
    sum_yy = (jnp.sum(Kyy) - jnp.sum(jnp.diag(Kyy))) / (N * (N - 1))
    sum_xy = jnp.sum(Kxy) / (M * N)
    return sum_xx + sum_yy - 2.0 * sum_xy
```

### 4.2 Bandwidth via the median heuristic

The kernel bandwidth `σ` is set automatically via the **median heuristic**: `σ = sqrt(median(pairwise_distances²) / 2)`. This is a standard data-adaptive choice.

```python
def mmd_sq_with_median_bandwidth(X, Y):
    sigma = jax.lax.stop_gradient(median_heuristic(X, Y))
    return mmd_sq_unbiased(X, Y, sigma)
```

The `stop_gradient` around `sigma` is critical: we don't want the optimiser to "cheat" by shrinking `σ` to make MMD² artificially small. Treating bandwidth as a constant from the gradient's perspective also avoids numerically nasty second-order terms from the median operator.

---

## 5. Per-seed gradients and the OPG matrix

**File**: `src/curvature_calib/calibration/per_seed_grads.py`

This is the mathematical heart of the project. We need not just the gradient of the loss, but a curvature matrix built from the individual per-seed gradient contributions.

### 5.1 Why per-seed gradients?

The total gradient `∇_θ MMD²` is the mean over all seeds. But the **Outer Product of Gradients (OPG) matrix** is:

```
F̂ = (1/M) Σ_m  g_m · g_m^T       where g_m is the per-seed gradient
```

This is a `(P × P)` matrix (here P=5) that approximates the curvature of the loss landscape. Its **eigenvalues** tell us which parameter directions are well-determined (large eigenvalue = stiff direction) and which are indeterminate (small eigenvalue = sloppy direction).

### 5.2 Computing per-seed gradients via chained VJPs

The challenge: MMD² is a single scalar that mixes all `M` seeds together. We need to attribute gradient contributions back to individual seeds.

The strategy uses two VJP calls chained together:

```python
def per_seed_loss_and_grads(simulate_fn, theta, keys, Y_ref):
    M = keys.shape[0]

    # Step 1: Forward — simulate M trajectories
    X = jax.vmap(lambda k: simulate_fn(theta, k))(keys)  # (M, T)

    # Step 2: VJP of MMD² wrt X — how does the loss change if any trajectory changes?
    L, vjp_X = jax.vjp(lambda x: mmd_sq_with_median_bandwidth(x, Y_ref), X)
    (dL_dX,) = vjp_X(jnp.ones(()))  # (M, T): dMMD²/dx_m for each seed m

    # Step 3: For each seed m, VJP through simulate with cotangent dL_dX[m]
    # This gives: dL/dx_m · dx_m/dtheta = dL/dtheta contributed by seed m
    def one_seed_grad(key, cotangent):
        _, vjp_t = jax.vjp(lambda t: simulate_fn(t, key), theta)
        (g,) = vjp_t(cotangent)
        return M * g  # scale so mean = total gradient

    per_seed = jax.vmap(one_seed_grad)(keys, dL_dX)  # (M, P)
    mean_grad = jnp.mean(per_seed, axis=0)            # (P,)
    opg = (per_seed.T @ per_seed) / M                 # (P, P)
```

**Step 2 unpacked**: `jax.vjp(f, X)` where `f: (M,T) -> scalar` returns a function `vjp_X` such that `vjp_X(1.0)` gives `∂L/∂X` — the gradient of MMD² with respect to the entire `(M, T)` trajectory matrix. The `[m]`-th row is "how much would the loss change if trajectory `m` shifted slightly".

**Step 3 unpacked**: For each seed `m`, we re-run the backward pass through the simulator itself. We call `jax.vjp(simulate, theta)` and pass in `dL_dX[m]` as the cotangent. By the chain rule, this gives `dL/dθ` attributed to seed `m`. The `jax.vmap` runs this for all `M` seeds in parallel.

**The factor M**: `g_m = M · (chain-rule result)`. This scaling ensures `(1/M) Σ g_m = ∇_θ MMD²`. Without it, the mean of per-seed gradients would be `(1/M²) Σ ...`.

### 5.3 The `CalibStats` namedtuple

```python
class CalibStats(NamedTuple):
    loss: jax.Array            # scalar
    mean_grad: jax.Array       # (P,) = nabla_theta MMD²
    per_seed_grads: jax.Array  # (M, P)
    opg: jax.Array             # (P, P)
```

This is the single object passed to the rest of the calibration pipeline.

---

## 6. The diagnostic

**File**: `src/curvature_calib/calibration/diagnostic.py`

The diagnostic layer is pure linear algebra — no JAX magic here, just eigendecomposition.

### 6.1 Eigendecomposition of F̂

```python
class EigDecomp(NamedTuple):
    eigvals: jax.Array  # (P,) descending — largest = stiffest direction
    eigvecs: jax.Array  # (P, P), column k is eigenvector k

def eigendecompose(F):
    F_sym = 0.5 * (F + F.T)         # symmetrise (guards against float rounding)
    w, V = jnp.linalg.eigh(F_sym)   # eigh: symmetric eigendecomposition
    order = jnp.argsort(-w)          # descending order
    return EigDecomp(eigvals=w[order], eigvecs=V[:, order])
```

`jnp.linalg.eigh` is the symmetric version of `eig` (faster, more numerically stable). It returns eigenvalues in ascending order; we reverse to get descending (stiffest first).

The **eigenvectors** `V[:, k]` are the key output:
- `V[:, 0]` (stiffest): the combination of the 5 parameters that the data constrains most tightly.
- `V[:, -1]` (sloppiest): the combination the data is almost completely blind to.

In the BH model, we consistently find `v_1 ≈ (b_1 + b_2) / √2` (the symmetric sum of biases is the stiffest) and `v_5 ≈ β` (the intensity of choice is the sloppiest).

### 6.2 Principal angles between subspaces

```python
def principal_angles(V1, V2):
    Q1, _ = jnp.linalg.qr(V1)
    Q2, _ = jnp.linalg.qr(V2)
    s = jnp.linalg.svd(Q1.T @ Q2, compute_uv=False)
    s = jnp.clip(s, -1.0, 1.0)
    return jnp.arccos(s)
```

This measures how similar two subspaces are (e.g. the stiff subspace at two different parameter values, or at two different gradient horizons). Angle near 0° = same subspace; angle near 90° = orthogonal (completely different).

The algorithm: orthonormalise each subspace (QR), compute the SVD of their inner product matrix. The singular values are the cosines of the principal angles.

### 6.3 Effective dimension

```python
def effective_dimension(eigvals, noise_floor):
    return int(jnp.sum(eigvals > noise_floor))
```

How many eigenvalues are meaningfully above zero? This tells us the number of **identifiable parameter combinations** — the dimensionality of what the data actually constrains.

---

## 7. Bootstrap confidence intervals

**File**: `src/curvature_calib/calibration/bootstrap.py`

We have M=64 seeds. How reliable is our OPG estimate? Bootstrap resampling answers this.

### 7.1 Eigenvalue bootstrap

```python
def bootstrap_eigvals(per_seed_grads, n_boot=500, key=None):
    G = np.asarray(per_seed_grads)  # (M, P)
    M, P = G.shape
    indices = np.asarray(jax.random.randint(key, (n_boot, M), 0, M))
    out = np.empty((n_boot, P))
    for b in range(n_boot):
        Gb = G[indices[b]]          # resample rows (seeds)
        Fb = (Gb.T @ Gb) / M        # recompute OPG from resample
        w = np.linalg.eigvalsh(Fb)
        out[b] = np.sort(w)[::-1]
    return jnp.asarray(out)  # (n_boot, P)
```

**Bootstrap principle**: treat the M seeds as a sample from the population of seeds. Resample with replacement `n_boot` times, recompute the OPG each time, and record the eigenvalue distribution. The spread of this distribution is the uncertainty in our eigenvalue estimates.

Note: this uses **NumPy** (not JAX) internally because the loop over bootstrap replicates doesn't benefit from JAX's compilation and would be slower if compiled.

### 7.2 Noise floor

```python
def noise_threshold(eigval_cis):
    return float(eigval_cis[-1, 1])   # upper CI bound of the smallest eigenvalue
```

If the smallest eigenvalue's upper confidence bound is near zero, that direction is statistically indistinguishable from zero — it's in the noise. Any eigenvalue below this threshold is not identifiable.

---

## 8. The calibration loop

**Files**: `calibration/calibrate.py`, `calibration/preconditioner.py`

### 8.1 The Levenberg-Marquardt preconditioned step

Instead of gradient descent (`θ ← θ − α·g`), we use:

```
step = −(F̂ + λ·I)⁻¹ · g
```

where:
- `F̂` is the OPG matrix (curvature approximation).
- `g` is the mean gradient.
- `λ` (damping) is a positive scalar.

This is **Levenberg-Marquardt (LM)**: it interpolates between Newton's method (λ→0, uses full curvature) and gradient descent (λ→∞, ignores curvature). The key benefit is that along **stiff directions** (large eigenvalues of F̂), the step is small and precise; along **sloppy directions** (small eigenvalues), the damping `λ·I` prevents wild steps into territory the loss doesn't constrain.

```python
def damped_step(F_hat, g, damping):
    P = F_hat.shape[0]
    A = F_hat + damping * jnp.eye(P, dtype=F_hat.dtype)
    # Solve A · delta = g  via Cholesky (A is positive definite for any damping > 0)
    L = jnp.linalg.cholesky(A)
    y = jax.scipy.linalg.solve_triangular(L, g, lower=True)
    delta = jax.scipy.linalg.solve_triangular(L.T, y, lower=False)
    return -delta
```

**Why Cholesky?** `A = F̂ + λI` is symmetric positive definite (all eigenvalues ≥ λ > 0). Cholesky factorisation is the numerically stable way to solve `A·x = b` for SPD matrices — faster and more stable than general `np.linalg.solve`.

### 8.2 Adaptive damping

The damping `λ` is updated each iteration using the **reduction ratio** ρ:

```
ρ = (actual loss decrease) / (predicted loss decrease from quadratic model)
```

- If ρ > 0.75: the quadratic approximation was good → **decrease λ** (trust curvature more).
- If ρ < 0.25: the quadratic approximation was bad → **increase λ** (fall back toward gradient descent).

```python
def update_damping(damping, realised_reduction, predicted_reduction, ...):
    rho = realised_reduction / predicted_reduction
    if rho > 0.75:
        return max(damping * decrease_factor, min_damping)
    if rho < 0.25:
        return min(damping * increase_factor, max_damping)
    return damping
```

This is the standard Marquardt update rule (Martens & Grosse 2015, §6.4).

### 8.3 The calibration loop

```python
def calibrate(simulate_fn, theta0, Y_ref, M=64, n_iter=80, ...):
    theta = theta0
    damping = init_damping
    master = jax.random.PRNGKey(seed_base)
    val_keys = jax.random.split(jax.random.PRNGKey(val_seed), val_M)  # FIXED

    for t in range(n_iter):
        keys = jax.random.split(jax.random.fold_in(master, t), M)
        stats = per_seed_loss_and_grads(simulate_fn, theta, keys, Y_ref)
        eig = eigendecompose(stats.opg)

        step = learning_rate * damped_step(stats.opg, stats.mean_grad, damping)
        theta_proposed = theta + step
        L_prop = loss_only(simulate_fn, theta_proposed, keys, Y_ref)

        accept = L_prop < stats.loss  # only accept if loss improved
        if accept:
            theta = theta_proposed

        damping = update_damping(damping, stats.loss - L_prop, predicted_reduction)
```

**`jax.random.fold_in(master, t)`**: JAX's way to get a fresh, deterministic sub-key from a master key and a step index `t`. This gives different seeds each iteration without storing state — `master` never changes.

**Two loss tracks**:
- `losses`: the MMD² on the **same** fresh seeds used for the step. This is the optimizer's view — noisy.
- `val_losses`: MMD² on a **fixed** set of held-out seeds (`val_keys`). This is the clean monitor — used for plotting progress.

**NaN guard**: if the proposed step pushes `θ` into an unstable regime where the simulator diverges, `L_prop = NaN`. The loop detects this with `math.isfinite(L_prop)`, forces a rejection, and raises damping by ×10.

---

## 9. End-to-end data flow

Here is the complete pipeline from `theta` to the calibration diagnostic:

```
theta (5,)
    │
    ▼
pack_canonical()          # theta -> BHParams(beta, g, b)
    │
    ▼
simulate(theta, key)  ─── jax.lax.scan ──► xs (T,)   [one trajectory]
    │
    ├── jax.vmap over M keys ──► X (M, T)
    │
    ▼
mmd_sq_with_median_bandwidth(X, Y_ref)  ──► loss (scalar)
    │
    ├── jax.vjp(MMD², X) ──► dL/dX (M, T)
    │
    ├── jax.vmap over seeds:
    │     jax.vjp(simulate, theta) with cotangent dL/dX[m]  ──► g_m (P,)
    │
    ├── stack ──► per_seed_grads (M, P)
    │
    ├── mean ──► mean_grad (P,)    = nabla_theta MMD²
    │
    └── outer product ──► OPG F̂ (P, P)
                              │
                              ▼
                    eigendecompose(F̂)
                              │
                    ┌─────────┴──────────┐
                    │                    │
                eigvals (P,)         eigvecs (P, P)
              descending             col k = k-th direction
                    │                    │
                    ▼                    ▼
              sloppy ↔ stiff     which param combos
              spectrum span      are identifiable
                    │
                    └──► bootstrap_eigvals()  ──► CIs on eigenvalues
```

The calibration loop wraps around this entire pipeline: at each iteration, it calls `per_seed_loss_and_grads`, takes an LM step using the OPG, and logs the full `EigDecomp` so we can track how the curvature landscape evolves as `θ` moves.

---

## Quick reference: module responsibilities

| Module | What it does |
|---|---|
| `models/brock_hommes.py` | Simulate one BH trajectory. `simulate(theta, key) -> (T,)` |
| `losses/mmd.py` | Compute MMD² between two sets of trajectories |
| `calibration/per_seed_grads.py` | Compute loss + gradient + OPG via chained VJPs |
| `calibration/diagnostic.py` | Eigendecompose OPG; principal angles; effective dimension |
| `calibration/bootstrap.py` | Bootstrap CIs for eigenvalues and subspaces |
| `calibration/preconditioner.py` | LM damped step + adaptive damping update rule |
| `calibration/calibrate.py` | Full calibration loop with logging |
| `calibration/baselines.py` | SGD and Adam baselines (share the same VJP backbone) |
| `viz/style.py` | Shared colour palette and matplotlib rcParams |
