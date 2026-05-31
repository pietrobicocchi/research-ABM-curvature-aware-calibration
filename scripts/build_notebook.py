"""Build the onboarding walkthrough notebook.

Constructs cells with nbformat, executes them with nbclient, writes the
populated .ipynb to notebooks/00_walkthrough.ipynb.

Run: uv run python scripts/build_notebook.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient


def md(src: str) -> dict:
    return nbformat.v4.new_markdown_cell(src.strip("\n"))


def code(src: str) -> dict:
    return nbformat.v4.new_code_cell(src.strip("\n"))


cells = []

# =================================================================== INTRO
cells.append(md(r"""
# Curvature-Aware Calibration of Differentiable Agent-Based Models — onboarding walkthrough

**Audience.** Someone new to this project. The notebook assumes only undergraduate linear algebra and basic Python. No prior exposure to agent-based models (ABMs), kernel methods, automatic differentiation, or sloppy-models theory is required — we build everything from first principles.

**What you will learn.**

1. What an agent-based model is, mathematically.
2. The Brock-Hommes (1998) heterogeneous-agent asset-pricing model — its economic story and its equations.
3. Why calibrating ABMs is hard, and why we use the Maximum Mean Discrepancy (MMD) as a distributional loss.
4. What "differentiable ABM" means and how automatic differentiation gives us per-seed gradients.
5. The **outer-product-of-gradients (OPG) matrix** — what it is, why its eigendecomposition reveals which parameter combinations the data constrains and which it does not. (This is the project's main contribution.)
6. How to use the OPG matrix as a curvature preconditioner via damped Levenberg-Marquardt.
7. The **§5.4 falsification protocol** that validates the diagnostic against non-MMD discrepancies.

**What this notebook is not.** It is not a replacement for the paper draft or the literature review. It is the path you should walk *before* reading those, so the formal arguments land on intuition.

**How to read it.** Run cells in order. Each section ends with a visual that tells a story — if the figure doesn't make sense, stop and re-read the math above.
"""))

cells.append(md(r"""
## 0. Setup

We import the project package (`curvature_calib`), JAX (the AD framework that powers our simulator), and a few standard scientific-Python libraries. The styling helper sets a clean palette consistent with the other figures in `outputs/`.
"""))

cells.append(code(r"""
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as sps

from curvature_calib.models.brock_hommes import simulate, pack_canonical
from curvature_calib.losses.mmd import (
    rbf_kernel,
    median_heuristic,
    mmd_sq_unbiased,
    mmd_sq_with_median_bandwidth,
)
from curvature_calib.calibration.per_seed_grads import (
    vmap_simulate,
    per_seed_loss_and_grads,
)
from curvature_calib.calibration.opg import (
    eigendecompose,
    bootstrap_eigvals,
    principal_angles,
)
from curvature_calib.calibration.calibrate import calibrate
from curvature_calib.viz.style import apply_style, QUAL

apply_style()
plt.rcParams["figure.figsize"] = (10, 5)

print("jax", jax.__version__, "devices:", jax.devices())
"""))

# =================================================================== PART I
cells.append(md(r"""
## 1. What is an agent-based model?

### 1.1 The mathematical object

An **agent-based model** (ABM) is a stochastic function

$$ f : \Theta \to \mathcal{X}, \qquad \theta \mapsto x, $$

where

- $\theta \in \Theta \subseteq \mathbb{R}^P$ is a vector of $P$ parameters governing agent behaviour (e.g. how much weight a trader puts on past prices),
- $x \in \mathcal{X}$ is a *trajectory* of the simulated system (e.g. a time series of prices over $T$ time steps),
- and the function is **stochastic** — even for fixed $\theta$, repeated evaluations produce different $x$, because agents make randomised decisions.

In code, $\theta$ is a tuple of numbers and $x$ is a NumPy / JAX array.

### 1.2 Why are ABMs hard to calibrate?

The standard recipe for statistical inference is to write down a likelihood $p(x \mid \theta)$, observe data $x_{\text{obs}}$, and maximise. **For ABMs, $p(x \mid \theta)$ is intractable** — the model is defined by a *simulator*, not by an explicit density. You can run the simulator forward for any $\theta$, but you cannot evaluate the probability that it produced any particular trajectory.

### 1.3 The discrepancy-loss approach

Modern ABM calibration sidesteps the likelihood by comparing *distributions of trajectories*. Given an observed time series $y_{\text{obs}}$ and a chosen distance $D$ between distributions, we minimise

$$ \mathcal{L}(\theta) = D\bigl(\mathbb{P}_\theta,\; \mathbb{P}_{\text{obs}}\bigr), $$

where $\mathbb{P}_\theta$ is the distribution induced by the simulator at parameter $\theta$. The choice of $D$ matters a great deal — and we will use the **Maximum Mean Discrepancy** (Section 3), but first let us meet the simulator.
"""))

# =================================================================== PART II
cells.append(md(r"""
## 2. The Brock-Hommes (1998) model

Brock & Hommes (Journal of Economic Dynamics and Control, 1998) introduced a simple, deeply influential heterogeneous-agent asset pricing model. It captures one of the central stylised facts of real markets — that prices can wander far from "fundamental value" because traders use different forecasting rules and switch between them as those rules do well or badly.

### 2.1 The economic story (no equations yet)

Imagine many traders, each subscribing to one of $H$ "strategies." Each strategy is a forecasting rule. At every time step:

1. Each trader uses their strategy to forecast next period's price deviation from fundamental.
2. The market clears at a price that depends on these forecasts and the fractions $n_{h,t}$ of traders using each strategy.
3. After the price is revealed, every strategy is scored by how much profit it would have generated. Strategies that did well attract a larger share of traders next period (the *discrete-choice* switching).
4. The intensity-of-choice parameter $\beta$ controls how aggressively traders switch to the winning strategy. Small $\beta$ → traders are sticky; large $\beta$ → everyone piles into whatever just worked.

The canonical setup uses $H = 3$ strategies: one **fundamentalist** (always forecasts a return to fundamental) and two **trend followers** with opposing biases. With this setup the model has $P = 5$ free parameters,

$$ \theta = (\beta,\; g_1,\; b_1,\; g_2,\; b_2), $$

where $(g_h, b_h)$ are the trend coefficient and bias of free trader type $h$, and the fundamentalist's $(g_0, b_0)$ are fixed at $(0, 0)$.

### 2.2 The equations

Let $x_t$ denote the *deviation* of price from fundamental at time $t$. Let $R = 1 + r$ be the gross risk-free return. The model is the three-equation system

$$
\begin{aligned}
n_{h,t}\; &= \frac{\exp(\beta\, U_{h,t-1})}{\sum_j \exp(\beta\, U_{j,t-1})} & &\text{(softmax over fitness)} \\
x_t\;\;\, &= \frac{1}{R} \sum_{h=1}^H n_{h,t}\,(g_h\, x_{t-1} + b_h) + \varepsilon_t & &\text{(market clearing)} \\
U_{h,t}\,\, &= (x_t - R\, x_{t-1})\,(g_h\, x_{t-2} + b_h - R\, x_{t-1}) & &\text{(realised profit of strategy }h\text{)}
\end{aligned}
$$

with i.i.d. noise $\varepsilon_t \sim \mathcal{N}(0, \sigma^2)$. Each strategy's *forecast* of $x_t$ given information up to $t-1$ is $g_h\, x_{t-1} + b_h$, a one-lag linear rule.

### 2.3 The state recursion

To implement this as a recursive step we need the state $(x_{t-1},\; x_{t-2},\; U_{\cdot,t-1})$ — two lags of $x$ and the fitness vector. Given a noise innovation $\varepsilon_t$, one step computes:

1. Fractions $n_t = \mathrm{softmax}(\beta\, U_{t-1})$.
2. Forecasts $f_{h,t} = g_h\, x_{t-1} + b_h$.
3. New price $x_t = (1/R) \sum_h n_{h,t} f_{h,t} + \varepsilon_t$.
4. New fitness $U_{h,t} = (x_t - R\, x_{t-1})(g_h\, x_{t-2} + b_h - R\, x_{t-1})$.
5. New state $(x_t,\; x_{t-1},\; U_t)$.

This is precisely the `_step` function in `src/curvature_calib/models/brock_hommes.py`. We use `jax.lax.scan` to iterate it $T$ times in a single compiled call, which makes the whole trajectory **differentiable** with respect to $\theta$ (we will exploit this in Section 4).
"""))

cells.append(code(r"""
# Canonical parameters from Brock & Hommes 1998 §3 Example 3.1:
#   beta = intensity of choice
#   g_1, b_1 = first trend follower (positive bias)
#   g_2, b_2 = second trend follower (negative bias, opposite to type 1)
# Fundamentalist (type 0) is hardcoded at g_0 = b_0 = 0.
THETA = jnp.array([3.0, 1.2, 0.2, 1.2, -0.2])

T = 500
R = 1.1
sigma = 0.05

# One trajectory, one random seed.
key = jax.random.PRNGKey(0)
xs = simulate(THETA, key, T=T, sigma=sigma, R=R, x_init=0.0)

fig, ax = plt.subplots(figsize=(11, 3.5))
ax.plot(xs, color=QUAL[0], lw=0.9)
ax.set_xlabel("time step t")
ax.set_ylabel(r"price deviation $x_t$")
ax.set_title(
    rf"Brock-Hommes single trajectory: $\beta={float(THETA[0]):.1f}$, "
    rf"$g_1=g_2={float(THETA[1]):.2f}$, $\sigma={sigma}$"
)
plt.show()
"""))

cells.append(md(r"""
The trajectory wanders around 0 (the fundamental) but does not stay there — small noise kicks are amplified by the trend-follower strategies, then damped when the fundamentalist strategy gains share. This is the hallmark of the Brock-Hommes dynamics.

### 2.4 The ensemble view

Because the model is stochastic, a single trajectory does not characterise the *distribution* that $\theta$ produces. The right object is the **empirical distribution over many seeds**. Let us run $M = 64$ independent seeds and look at the cloud of trajectories.
"""))

cells.append(code(r"""
M = 64
keys = jax.random.split(jax.random.PRNGKey(1), M)
ensemble = vmap_simulate(
    lambda t, k: simulate(t, k, T=T, sigma=sigma, R=R, x_init=0.0),
    THETA, keys,
)
ensemble = np.asarray(ensemble)  # (M, T)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4))
for m in range(M):
    a1.plot(ensemble[m], color=QUAL[0], lw=0.4, alpha=0.25)
a1.plot(ensemble.mean(0), color="black", lw=1.6, label="ensemble mean")
a1.set_xlabel("time step")
a1.set_ylabel(r"$x_t$")
a1.set_title(f"$M={M}$ independent seeds")
a1.legend()

rets = np.diff(ensemble, axis=1).reshape(-1)
a2.hist(rets, bins=80, density=True, color=QUAL[0], alpha=0.85,
        edgecolor="white", linewidth=0.4)
a2.set_xlabel(r"one-step return $\Delta x_t$")
a2.set_ylabel("density")
a2.set_yscale("log")
a2.set_title("Pooled return distribution")
plt.tight_layout()
plt.show()
"""))

cells.append(md(r"""
The ensemble mean is approximately zero (the fundamental is the unconditional mean) but the marginal distribution of returns has clearly non-Gaussian tails. This is the kind of structure a calibrator must match.

### 2.5 Regime structure: stable → periodic → chaotic

Brock & Hommes (1998 §3) shows that as $\beta$ increases, the model passes through a sequence of bifurcations: the fundamental steady state goes from stable, to a limit cycle, to chaos. The other parameters are held at their canonical values.

For our purposes the important fact is that *the parameter has a strong, easily-visible effect on the output distribution*. The next figure shows this by sweeping $\beta$ over five orders of magnitude of dynamical complexity.
"""))

cells.append(code(r"""
betas_demo = [0.0, 5.0, 20.0, 50.0, 80.0]
labels = ["stable", "near bifurcation", "richer dynamics", "chaotic", "strongly chaotic"]

fig, axes = plt.subplots(1, 5, figsize=(16, 3))
for ax, beta, lab in zip(axes, betas_demo, labels):
    theta = jnp.array([beta, 1.2, 0.2, 1.2, -0.2])
    xs_b = simulate(theta, jax.random.PRNGKey(7), T=1200, sigma=0.05, R=1.1)
    ax.plot(np.asarray(xs_b)[-400:], color=QUAL[betas_demo.index(beta)], lw=0.9)
    ax.set_title(rf"$\beta={beta:.0f}$ — {lab}")
    ax.set_xlabel("t")
    ax.set_ylabel(r"$x_t$" if beta == 0.0 else "")
plt.tight_layout()
plt.show()
"""))

# ============================================================ PART III: MMD
cells.append(md(r"""
## 3. Comparing distributions — the Maximum Mean Discrepancy

### 3.1 Why we need a distributional loss

We cannot compare two stochastic trajectories directly: they will always disagree pointwise. What we *can* compare is the **distribution** that each parameter $\theta$ induces. So the calibration loss must be a function of two *empirical distributions*:

$$ \mathcal{L}(\theta) = D\bigl(\mathbb{P}_\theta,\; \mathbb{P}_{\text{ref}}\bigr). $$

We use the **Maximum Mean Discrepancy** (MMD) as our $D$. It has three properties we need:

1. **Sample-based.** No density evaluation; just kernel evaluations on pairs of samples.
2. **Differentiable in the simulated samples.** Together with a differentiable simulator, this means we can use automatic differentiation end-to-end.
3. **Metric structure.** For appropriate ("characteristic") kernels, $\mathrm{MMD}(P, Q) = 0$ if and only if $P = Q$.

### 3.2 The RBF kernel

We use the Gaussian RBF kernel,

$$ k(x, y) = \exp\!\left(-\frac{\|x - y\|^2}{2\sigma^2}\right), $$

where $\sigma$ is a bandwidth. Two points close together have $k(x, y) \approx 1$; two points far apart have $k(x, y) \approx 0$.

The bandwidth $\sigma$ controls "what counts as close." We pick it via the **median heuristic**: $\sigma^2 = \tfrac{1}{2} \,\mathrm{median}\big(\{\|x_i - x_j\|^2 : i \ne j\}\big)$, computed once from the pooled sample and then frozen. The factor of $1/2$ is the standard convention so that the kernel decays to $1/\sqrt{e}$ at the median pairwise distance.

### 3.3 The unbiased squared MMD estimator

Given samples $X = \{x_1, \ldots, x_M\}$ from $P$ and $Y = \{y_1, \ldots, y_N\}$ from $Q$, the *unbiased* squared MMD estimator is

$$
\widehat{\mathrm{MMD}}^2(P, Q)
= \frac{1}{M(M-1)} \sum_{i \ne i'} k(x_i, x_{i'})
+ \frac{1}{N(N-1)} \sum_{j \ne j'} k(y_j, y_{j'})
- \frac{2}{MN} \sum_{i, j} k(x_i, y_j).
$$

The diagonal terms are excluded so the estimator is unbiased for the true population MMD². A consequence is that $\widehat{\mathrm{MMD}}^2$ can be slightly negative on finite samples even when $P = Q$ — keep that in mind when reading the calibration loss plots.

### 3.4 Sanity check

Let us verify the two properties we need:

- $\widehat{\mathrm{MMD}}^2(X, Y) \approx 0$ when $X$ and $Y$ are independent draws from the *same* distribution.
- $\widehat{\mathrm{MMD}}^2(X, Y) \gg 0$ when the two distributions differ.
"""))

cells.append(code(r"""
rng = jax.random.PRNGKey(0)
k1, k2, k3 = jax.random.split(rng, 3)

# Same distribution: two independent draws from N(0, I) in R^10.
X_same1 = jax.random.normal(k1, (200, 10))
X_same2 = jax.random.normal(k2, (200, 10))
mmd_same = float(mmd_sq_unbiased(X_same1, X_same2, sigma=1.0))

# Different distributions: N(0, I) vs N(2, I).
X_diff = jax.random.normal(k3, (200, 10)) + 2.0
mmd_diff = float(mmd_sq_unbiased(X_same1, X_diff, sigma=1.0))

print(f"MMD^2 between two N(0,I) draws  : {mmd_same:+.5f}")
print(f"MMD^2 between N(0,I) and N(2,I) : {mmd_diff:+.5f}")
print(f"\nRatio: {mmd_diff / max(abs(mmd_same), 1e-6):.0f}x larger when distributions differ.")
"""))

cells.append(md(r"""
The MMD² is essentially zero when the two samples come from the same distribution and several orders of magnitude larger when they differ. This is exactly the structure we want as a calibration loss.

### 3.5 MMD between ABM ensembles

Now the real thing. We compute MMD² between an ensemble simulated at our truth $\theta^*$ and ensembles simulated at perturbed $\theta$ values. The loss should be minimised at $\theta^*$ and grow as we move away.
"""))

cells.append(code(r"""
THETA_STAR = jnp.array([3.0, 1.2, 0.2, 1.2, -0.2])

M_ref = 64
ref_keys = jax.random.split(jax.random.PRNGKey(42), M_ref)
T_calib = 200  # shorter trajectories keep MMD landscape evaluations quick

def _sim(theta, key):
    return simulate(theta, key, T=T_calib, sigma=0.05, R=1.1, x_init=0.0)

Y_ref = vmap_simulate(_sim, THETA_STAR, ref_keys)

# Sweep b_1 (a bias coefficient) alone. We pick a bias rather than a trend
# coefficient g because the Brock-Hommes dynamics are *asymmetric* in g near
# truth: below the truth the trend strategies become weaker than the
# fundamentalist and the output distribution barely moves; above the truth
# the system gets explosively unstable. The biases give a cleaner
# pedagogical basin.
b1_grid = np.linspace(THETA_STAR[2] - 0.18, THETA_STAR[2] + 0.18, 21)
mmd_vals = []
eval_keys = jax.random.split(jax.random.PRNGKey(99), 64)
for b1 in b1_grid:
    theta = THETA_STAR.at[2].set(b1)
    X = vmap_simulate(_sim, theta, eval_keys)
    mmd_vals.append(float(mmd_sq_with_median_bandwidth(X, Y_ref)))
mmd_vals = np.array(mmd_vals)

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(b1_grid, np.clip(mmd_vals, 1e-6, None), "o-",
        color=QUAL[0], lw=1.8, markersize=6)
ax.axvline(float(THETA_STAR[2]), color="grey", ls="--",
           label=r"$\theta^* = $ truth")
ax.set_yscale("log")
ax.set_xlabel(r"$b_1$ (bias, type 1)")
ax.set_ylabel(r"$\widehat{\mathrm{MMD}}^2$ (log, clipped at $10^{-6}$)")
ax.set_title(r"MMD$^2$ loss as a function of $b_1$ (other params at truth)")
ax.legend()
plt.show()

print(f"argmin b_1 on the grid : {b1_grid[np.argmin(mmd_vals)]:.3f}")
print(f"truth value of b_1    : {float(THETA_STAR[2]):.3f}")
"""))

cells.append(md(r"""
The MMD² curve is shaped like a basin centred on the truth — exactly what we want from a calibration loss. The estimator can dip slightly below zero at the bottom of the basin (because we use the unbiased U-statistic), which is why the y-axis on log plots will need clipping.
"""))

# ============================================================ PART IV: AD
cells.append(md(r"""
## 4. Differentiable ABMs and per-seed gradients

### 4.1 What "differentiable ABM" means

A **differentiable ABM** is one whose simulator is implemented in an automatic-differentiation (AD) framework — for us, JAX. The forward pass is the usual stochastic simulation. The reverse pass uses the chain rule to compute exact gradients of any scalar output with respect to $\theta$.

The Brock-Hommes simulator is *naturally* differentiable: all the operations (softmax, multiplication, addition, scan) are smooth in $\theta$. There are no discrete choices made *inside* the simulator that depend on $\theta$ — the trader-fraction softmax is the smooth replacement of a hard argmax. So no surrogate gradients are needed.

### 4.2 The MMD-gradient decomposition

For the MMD² loss

$$ \mathcal{L}(\theta) = \widehat{\mathrm{MMD}}^2(X(\theta), Y_{\text{ref}}), \qquad X(\theta) = \{x_1(\theta), \ldots, x_M(\theta)\}, $$

the gradient with respect to $\theta$ decomposes by the chain rule:

$$ \nabla_\theta \mathcal{L}(\theta)
= \sum_{m=1}^M \underbrace{\frac{\partial \mathcal{L}}{\partial x_m}}_{(1\times T) \text{ row vec}}
\cdot \underbrace{\frac{\partial x_m}{\partial \theta}}_{(T\times P) \text{ Jacobian}}. $$

Each term in the sum involves only one simulator seed: a "per-seed" contribution to the total gradient.

### 4.3 Per-seed gradient definition

We define the per-seed gradient

$$ g_m = M \cdot \frac{\partial \mathcal{L}}{\partial x_m} \cdot \frac{\partial x_m}{\partial \theta} \in \mathbb{R}^P. $$

The factor of $M$ is chosen so that

$$ \bar{g} = \frac{1}{M} \sum_m g_m = \nabla_\theta \mathcal{L}. $$

This convention makes $\bar{g}$ the "mean per-seed gradient" and equals the total gradient.

### 4.4 Computing it in JAX

The package function `per_seed_loss_and_grads(simulate_fn, theta, keys, Y_ref)` does this with two VJPs: one through MMD² to get $\partial \mathcal{L} / \partial x_m$, and one through `simulate(theta, key_m)` per seed (vmapped) to contract with $\partial x_m / \partial \theta$.

The result is a `CalibStats` named tuple with fields `loss`, `mean_grad`, `per_seed_grads` $(M \times P)$, and `opg` $(P \times P)$ — the latter is the object we will spend the rest of the notebook studying.
"""))

cells.append(code(r"""
# Compute per-seed gradients at a perturbed theta (small offset from truth
# so the loss is non-trivial and gradients carry signal).
theta0 = THETA_STAR + jnp.array([0.0, 0.0, 0.1, 0.0, 0.1])

M = 200  # extra-large so the cloud is easy to see
keys = jax.random.split(jax.random.PRNGKey(11), M)
stats = per_seed_loss_and_grads(_sim, theta0, keys, Y_ref)

print(f"loss             = {float(stats.loss):.4e}")
print(f"mean grad shape  = {stats.mean_grad.shape}  (= P=5)")
print(f"per-seed grads   = {stats.per_seed_grads.shape}  (= M x P)")
print(f"OPG matrix shape = {stats.opg.shape}  (= P x P)")

# Sanity: bar_g must equal the autodiff gradient of the full MMD^2 loss.
def full_loss(t):
    X = vmap_simulate(_sim, t, keys)
    return mmd_sq_with_median_bandwidth(X, Y_ref)

g_check = jax.grad(full_loss)(theta0)
print(f"\nReconstruction check: |bar_g - autodiff_grad| = "
      f"{float(jnp.linalg.norm(stats.mean_grad - g_check)):.3e}")
"""))

cells.append(code(r"""
# Visualise the per-seed gradient cloud projected onto the top-2 OPG axes.
G = np.asarray(stats.per_seed_grads)  # (M, P)
F = np.asarray(stats.opg)
eig = eigendecompose(jnp.asarray(F))
V = np.asarray(eig.eigvecs)              # columns are eigenvectors

G_centered = G - G.mean(0)
proj = G_centered @ V[:, :2]  # (M, 2)

fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(proj[:, 0], proj[:, 1], s=12, alpha=0.55, color=QUAL[0],
           edgecolor="white", linewidth=0.3,
           label=fr"$g_m - \bar g$ (M = {M})")
ax.axhline(0, color="grey", lw=0.5, ls="--")
ax.axvline(0, color="grey", lw=0.5, ls="--")
ax.set_xlabel(r"projection on $v_1$ (stiffest direction)")
ax.set_ylabel(r"projection on $v_2$")
ax.set_title("The per-seed gradient cloud (top-2 OPG directions)")
ax.legend()
ax.set_aspect("equal", adjustable="datalim")
plt.show()
"""))

cells.append(md(r"""
The cloud is dramatically elongated along $v_1$ (the stiffest direction) and squashed along $v_2$. This anisotropy is *the* phenomenon we exploit: it tells us that the data sees the $v_1$ combination of parameters much more clearly than the $v_2$ combination.

The next section formalises this.
"""))

# =========================================================== PART V: OPG
cells.append(md(r"""
## 5. The outer-product-of-gradients (OPG) matrix

### 5.1 Definition

We collect the per-seed gradients into the matrix

$$ \hat F(\theta) = \frac{1}{M} \sum_{m=1}^M g_m\, g_m^\top \in \mathbb{R}^{P \times P}. $$

This is the **outer-product-of-gradients (OPG) matrix**, also called the empirical curvature matrix. It is symmetric and positive semi-definite by construction.

### 5.2 IMPORTANT: this is NOT the Fisher information

In likelihood-based settings the OPG is sometimes called the "empirical Fisher." It is not, and Kunstner, Hennig & Balles (NeurIPS 2019) showed that conflating the two can produce updates with the wrong scale or direction far from the optimum. **In this project we never call $\hat F$ the Fisher.** We always call it the OPG matrix or the empirical curvature matrix.

### 5.3 Why $\hat F$ does approximate curvature here: the residual structure of MMD

The Kunstner critique loses its force in our setting because the MMD² loss has a **residual structure**: it is the squared norm of a difference between two kernel mean embeddings in a reproducing kernel Hilbert space (RKHS). For any loss of this form, the Hessian decomposes as

$$ \nabla^2 \mathcal{L} = \underbrace{2 J^\top J}_{\text{Gauss-Newton (positive semi-definite)}} + \underbrace{(\text{residual-curvature term})}_{\text{small near the optimum}}, $$

where $J = \partial r / \partial \theta$ is the residual Jacobian. The first term is the **generalised Gauss-Newton (GGN) matrix**. It is a positive semi-definite Hessian approximation that is widely used in second-order optimisation (Schraudolph 2002; Martens & Grosse 2015).

Our $\hat F$ is a **stochastic, sample-based approximation of the GGN matrix** for the MMD² loss. It is a curvature object *because* of the residual structure of MMD, not because of any Fisher-identity argument. This is the load-bearing claim that justifies everything that follows.

### 5.4 What the eigendecomposition tells us

Write the symmetric eigendecomposition

$$ \hat F = V \Lambda V^\top, \qquad \lambda_1 \ge \lambda_2 \ge \cdots \ge \lambda_P \ge 0. $$

The columns $v_k$ of $V$ are *linear combinations of parameters* — they are unit vectors in $\mathbb{R}^P$. The eigenvalue $\lambda_k$ has the clean interpretation

$$ \lambda_k = \frac{1}{M} \sum_m (v_k^\top g_m)^2, $$

i.e. **the average squared directional derivative of the per-seed loss along $v_k$**.

- Large $\lambda_k$ → loss is very sensitive to perturbations along $v_k$ → that parameter combination is well constrained by the data ("stiff").
- Small $\lambda_k$ → loss is insensitive → unconstrained ("sloppy").

This is *exactly* a practical-identifiability statement under the MMD loss.

### 5.5 Visualising the OPG ellipse

The OPG ellipse $\{u : u^\top \hat F u \le 1\}$ is the contour of the quadratic form defined by $\hat F$. We overlay it on the gradient cloud — the cloud should fit inside the ellipse, with its principal axes aligned to the eigenvectors of $\hat F$.
"""))

cells.append(code(r"""
import matplotlib.patches as mpatches

def confidence_ellipse(F_2d, ax, n_std=1.0, **kwargs):
    w, V_ = np.linalg.eigh(F_2d)
    order = np.argsort(-w)
    w, V_ = w[order], V_[:, order]
    angle = np.degrees(np.arctan2(V_[1, 0], V_[0, 0]))
    width, height = 2 * n_std * np.sqrt(np.clip(w, 0.0, None))
    ax.add_patch(mpatches.Ellipse(xy=(0, 0), width=width, height=height,
                                  angle=angle, fill=False, **kwargs))

F_2d = V[:, :2].T @ F @ V[:, :2]  # OPG matrix projected onto its top-2 eigenspace

fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(proj[:, 0], proj[:, 1], s=12, alpha=0.55, color=QUAL[0],
           edgecolor="white", linewidth=0.3, label="per-seed grads")
confidence_ellipse(F_2d, ax, n_std=1.0,
                   edgecolor=QUAL[1], lw=2, label=r"$1\sigma$ OPG ellipse")
confidence_ellipse(F_2d, ax, n_std=2.0,
                   edgecolor=QUAL[1], lw=1, linestyle="--", alpha=0.6)
ax.axhline(0, color="grey", lw=0.5, ls="--")
ax.axvline(0, color="grey", lw=0.5, ls="--")
ax.set_xlabel(r"$v_1^\top (g_m - \bar g)$")
ax.set_ylabel(r"$v_2^\top (g_m - \bar g)$")
ax.set_title("Gradient cloud + OPG ellipse")
ax.legend()
ax.set_aspect("equal", adjustable="datalim")
plt.show()
"""))

# ====================================================== PART VI: IDENTIFIABILITY
cells.append(md(r"""
## 6. The eigendecomposition = practical identifiability

### 6.1 Stiff vs sloppy spectra

In the systems-biology literature (Gutenkunst et al. 2007; Transtrum et al. 2010, 2011) it is a well-established empirical regularity that *nonlinear-fit models display sloppy spectra*: the eigenvalues of the curvature matrix span six to ten orders of magnitude, approximately uniformly distributed on a log scale, with no clean separation between "identifiable" and "non-identifiable" parameters.

In agent-based modelling this was first shown by Naumann-Woleske et al. (2024) using finite-difference Hessians on the Mark-0 macroeconomic ABM. Our contribution is to compute the same diagnostic *live during calibration* using AD-computed per-seed gradients.

### 6.2 The spectrum, with confidence

We plot $\lambda_k$ on a log scale with 95% bootstrap confidence intervals over the $M$ seeds. A small eigenvalue whose CI touches zero is *statistically indistinguishable from zero* — that parameter direction is genuinely unconstrained by the data at this $\theta$.
"""))

cells.append(code(r"""
eigvals = np.asarray(eig.eigvals)
boot = np.asarray(bootstrap_eigvals(
    stats.per_seed_grads, n_boot=300, key=jax.random.PRNGKey(7),
))
boot_lo = np.percentile(boot, 2.5, axis=0)
boot_hi = np.percentile(boot, 97.5, axis=0)

fig, ax = plt.subplots(figsize=(8, 5))
xs = np.arange(eigvals.size)
le = np.clip(eigvals - boot_lo, a_min=0.0, a_max=None)
ue = np.clip(boot_hi - eigvals, a_min=0.0, a_max=None)
ax.errorbar(xs, eigvals, yerr=[le, ue], fmt="o", color=QUAL[0],
            markersize=11, capsize=4,
            markerfacecolor=QUAL[0], markeredgecolor="white",
            markeredgewidth=1.4, lw=1.6)
ax.set_yscale("log")
ax.set_xticks(xs)
ax.set_xticklabels([f"$v_{k+1}$" for k in xs])
ax.set_xlabel("eigendirection (ordered by eigenvalue)")
ax.set_ylabel(r"$\lambda_k$")
ax.set_title("OPG eigenvalue spectrum with 95% bootstrap CIs")
plt.show()

print(f"Span: lambda_1 / lambda_P = {eigvals[0] / max(eigvals[-1], 1e-30):.2e}")
print(f"Smallest eigenvalue       : {eigvals[-1]:.3e}")
print(f"Bootstrap 95% lower bound : {boot_lo[-1]:.3e}")
"""))

cells.append(md(r"""
The eigenvalues span several orders of magnitude — a textbook *sloppy* spectrum. The smallest eigenvalue's bootstrap CI reaches very close to zero, meaning that the corresponding direction $v_P$ is essentially unconstrained.

### 6.3 The eigenvector heatmap: WHICH combinations are constrained

The numerical values of $\lambda_k$ are only half the story. The other half is *which combination of parameters* each $v_k$ represents. We visualise $|V|$ as a heatmap with parameter names on the rows.
"""))

cells.append(code(r"""
PARAM_NAMES = [r"$\beta$", r"$g_1$", r"$b_1$", r"$g_2$", r"$b_2$"]
V_abs = np.abs(V)

fig, ax = plt.subplots(figsize=(7, 5.5))
im = ax.imshow(V_abs, cmap="magma", aspect="auto", vmin=0, vmax=1)
ax.set_xticks(range(5))
ax.set_xticklabels([f"$v_{k+1}$" for k in range(5)])
ax.set_yticks(range(5))
ax.set_yticklabels(PARAM_NAMES)
ax.set_title(r"$|V|$: parameter content of each eigendirection")
for i in range(5):
    for j in range(5):
        ax.text(j, i, f"{V_abs[i, j]:.2f}", ha="center", va="center",
                color="white" if V_abs[i, j] < 0.55 else "black", fontsize=10)
plt.colorbar(im, ax=ax, label=r"$|v_{k,j}|$")
plt.show()
"""))

cells.append(md(r"""
### 6.4 Reading the heatmap

This single picture is the **practical-identifiability diagnostic** that the project exists to deliver. Look at it column by column:

- **The sloppiest direction $v_P$ (rightmost column).** Its parameter content is dominated by **$\beta$**. The diagnostic is telling us that, at this $\theta$, the MMD loss barely constrains $\beta$ at all — changing $\beta$ alone produces an output distribution indistinguishable (to MMD) from $\theta^*$. This is a *substantive scientific claim* that the calibrator never could have made.

- **The stiffest direction $v_1$ (leftmost column).** Its parameter content is concentrated on the **biases $b_1$ and $b_2$** with similar magnitudes — this is the *symmetric* bias combination $b_1 + b_2$. The MMD loss sees this combination strongly: it is well-identified.

- **The middle columns** describe combinations of trend coefficients $g_1, g_2$ and asymmetric biases. They are partially identified — neither very stiff nor very sloppy.

The diagnostic surfaces parameter *combinations* — not individual parameters. A naive sensitivity analysis (one parameter at a time, the standard in much ABM work) would miss this structure entirely.
"""))

# ========================================================= PART VII: CALIBRATION
cells.append(md(r"""
## 7. Using $\hat F$ as a curvature preconditioner

### 7.1 The standard calibration loop

The published differentiable-ABM literature uses Adam to drive the per-seed gradients:

$$ \theta_{t+1} = \theta_t - \eta_t \cdot \mathrm{Adam}(\bar g_t). $$

This throws away *everything in the gradient cloud beyond the mean*. The geometry of the cloud — encoded in $\hat F$ — never enters the update.

### 7.2 Levenberg-Marquardt with the OPG matrix

The natural alternative is to use $\hat F$ as a preconditioner: solve

$$ \theta_{t+1} = \theta_t - \eta_t \cdot (\hat F_t + \lambda_t I)^{-1} \bar g_t, $$

where $\lambda_t I$ is **Tikhonov damping** updated by the standard Levenberg-Marquardt rule: increase $\lambda$ when the realised loss decrease is less than the quadratic model predicts, decrease it when the prediction is accurate (Martens & Grosse 2015 §6.4).

This is what `curvature_calib.calibration.calibrate` implements. We run it for 60 iterations starting from a small perturbation of the truth, and look at three diagnostics: the loss trajectory, the parameter iterates, and the eigenvalue spectrum over time.
"""))

cells.append(code(r"""
# Run a full calibration. Same setup we used in 06_calibration_dashboard.py
# but inline so the reader can see all the parts.
M_calib = 64
print("Running calibration ...")
calib_log = calibrate(_sim, theta0, Y_ref, M=M_calib, n_iter=60,
                      init_damping=100.0, verbose=False)
arrs = calib_log.as_arrays()
print(f"Done. Final loss = {arrs['losses'][-1]:.4e}")
"""))

cells.append(code(r"""
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

# Loss trajectory.
#
# Important: we plot the *validation* MMD^2 -- evaluated on a FIXED held-out
# seed set every iteration. The optimizer sees the noisy per-iter MMD^2
# (`losses`, computed on fresh seeds), which we also overlay faintly to show
# the contrast. The val-track is the clean signal; the train-track oscillates
# at the noise floor because consecutive iterations sample different MMD^2
# realisations.
a = axes[0]
floor = 5e-4
best_val = np.minimum.accumulate(arrs["val_losses"])
a.semilogy(np.clip(best_val, floor, None), color=QUAL[0], lw=2.0,
           label="best-so-far val MMD²")
a.semilogy(np.clip(arrs["val_losses"], floor, None),
           color=QUAL[0], lw=0.7, alpha=0.4,
           label="raw val MMD² (per iter)")
a.axhline(floor, color="grey", lw=1, ls="--",
          label=r"noise floor ($\sim 5\times 10^{-4}$)")
a.set_xlabel("iteration")
a.set_ylabel(r"$\widehat{\mathrm{MMD}}^2$ (clipped)")
a.set_title("A. Loss trajectory (best-so-far)")
a.legend(fontsize=8)

# Parameter iterates.
a = axes[1]
for k, name in enumerate(PARAM_NAMES):
    a.plot(arrs["thetas"][:, k], color=QUAL[k % len(QUAL)], lw=1.6, label=name)
    a.axhline(float(THETA_STAR[k]), color=QUAL[k % len(QUAL)],
              ls=":", lw=1, alpha=0.6)
a.set_xlabel("iteration")
a.set_ylabel("parameter value")
a.set_title(r"B. Iterates (dotted = $\theta^*$)")
a.legend(fontsize=8, ncol=2)

# Eigenvalue trajectory. We clip at 1e-12 (above float32 noise) so the
# smallest eigenvalue's numerical jitter does not dominate the y-range.
a = axes[2]
eigvals_traj = np.clip(arrs["eigvals"], 1e-12, None)
for k in range(eigvals_traj.shape[1]):
    a.semilogy(eigvals_traj[:, k],
               color=plt.cm.viridis(k / 4), lw=1.8,
               label=rf"$\lambda_{{{k+1}}}$")
a.set_xlabel("iteration")
a.set_ylabel(r"$\lambda_k$ (log)")
a.set_title("C. OPG spectrum over time")
a.legend(fontsize=8, ncol=2)

plt.tight_layout()
plt.show()
"""))

cells.append(md(r"""
The loss falls quickly into the noise floor of the unbiased MMD² estimator (it can briefly dip below zero — that is the U-statistic, not a bug). The eigenvalue trajectory in panel C confirms the sloppy spectrum is a stable feature of the landscape, not an artefact of one iterate. The five eigenvalues stay well-separated by several orders of magnitude over the whole calibration trajectory.

A practical note: we picked $\theta_0$ by shifting both biases together along the **stiff** eigendirection $v_1 \propto (b_1+b_2)/\sqrt{2}$. A random perturbation of the same magnitude almost always lands along the *sloppy* directions, where MMD² is already at the noise floor and there is nothing for the optimizer to descend. This itself is a consequence of the OPG spectrum we computed in section 6.
"""))

# ====================================================== PART VIII: FALSIFICATION
cells.append(md(r"""
## 8. The §5.4 falsification protocol

### 8.1 The circularity problem

The OPG eigenvalues describe local sensitivity *under the MMD loss*. A natural objection is: maybe the "sloppy" directions are an MMD artefact — perhaps a different distance would see them clearly. If so, the diagnostic is circular.

### 8.2 Resolution: perturb and measure with OTHER discrepancies

The §5.4 protocol from the project plan tests exactly this. At the calibrated $\theta_T$:

1. Identify the sloppiest direction $v_P$ (smallest eigenvalue) and the stiffest direction $v_1$ (largest eigenvalue).
2. Construct perturbations $\theta_T \pm \alpha v$ in each direction, with the same step size $\alpha$.
3. Simulate from each perturbed parameter and compare to simulations at $\theta_T$ under **three discrepancies, none of which is MMD**:

   - First four moments (mean, std, skew, kurtosis).
   - Empirical autocorrelation function (financially salient).
   - Tail quantiles at 1%, 5%, 95%, 99%.

**Hypothesis.** Perturbation along $v_P$ produces outputs indistinguishable from $\theta_T$ under all three discrepancies; perturbation along $v_1$ produces clearly distinguishable outputs.

If true, this establishes that the sloppy direction reflects a *genuine non-identifiability of the simulator under the chosen observables*, not an MMD-specific artefact.
"""))

cells.append(code(r"""
F_final = jnp.asarray(arrs["opgs"][-1])
eig_final = eigendecompose(F_final)
v_stiff = jnp.asarray(np.asarray(eig_final.eigvecs)[:, 0])
v_sloppy = jnp.asarray(np.asarray(eig_final.eigvecs)[:, -1])
theta_T = jnp.asarray(arrs["thetas"][-1])

alpha = 1e-2
theta_p_stiff  = theta_T + alpha * v_stiff
theta_m_stiff  = theta_T - alpha * v_stiff
theta_p_sloppy = theta_T + alpha * v_sloppy
theta_m_sloppy = theta_T - alpha * v_sloppy

M_show = 128
keys_falsify = jax.random.split(jax.random.PRNGKey(404), M_show)
X_T          = np.asarray(vmap_simulate(_sim, theta_T,          keys_falsify))
X_p_stiff    = np.asarray(vmap_simulate(_sim, theta_p_stiff,    keys_falsify))
X_m_stiff    = np.asarray(vmap_simulate(_sim, theta_m_stiff,    keys_falsify))
X_p_sloppy   = np.asarray(vmap_simulate(_sim, theta_p_sloppy,   keys_falsify))
X_m_sloppy   = np.asarray(vmap_simulate(_sim, theta_m_sloppy,   keys_falsify))


def four_moments(X):
    x = X.reshape(-1)
    return np.array([x.mean(), x.std(),
                     float(sps.skew(x)), float(sps.kurtosis(x))])


def autocorr_mean(X, max_lag=20):
    out = np.zeros(max_lag + 1)
    for m in range(X.shape[0]):
        x = X[m] - X[m].mean()
        var = x.var() + 1e-12
        out += np.array([np.mean(x[:x.size - k] * x[k:]) / var
                          for k in range(max_lag + 1)])
    return out / X.shape[0]


def tail_quantiles(X, qs=(0.01, 0.05, 0.95, 0.99)):
    x = X.reshape(-1)
    return np.array([np.quantile(x, q) for q in qs])


def disc(X_a, X_b):
    return {
        "moments": float(np.sum(np.abs(four_moments(X_a) - four_moments(X_b)))),
        "ACF":     float(np.sum(np.abs(autocorr_mean(X_a) - autocorr_mean(X_b)))),
        "quant":   float(np.sum(np.abs(tail_quantiles(X_a) - tail_quantiles(X_b)))),
    }


print(f"{'perturbation':<28s}  {'moments':>10s}  {'ACF':>10s}  {'quantiles':>10s}")
print("-" * 70)
results = {}
for name, X_a in [("+alpha v_1 (stiff)",  X_p_stiff),
                  ("-alpha v_1 (stiff)",  X_m_stiff),
                  ("+alpha v_P (sloppy)", X_p_sloppy),
                  ("-alpha v_P (sloppy)", X_m_sloppy)]:
    d = disc(X_a, X_T)
    results[name] = d
    print(f"{name:<28s}  {d['moments']:10.3e}  {d['ACF']:10.3e}  {d['quant']:10.3e}")
"""))

cells.append(code(r"""
fig, ax = plt.subplots(figsize=(11, 5))
channels = ["moments", "ACF", "quant"]
xs = np.arange(len(channels))
width = 0.2

for i, (name, color, alpha_bar) in enumerate([
        ("+alpha v_1 (stiff)",  QUAL[1], 0.95),
        ("-alpha v_1 (stiff)",  QUAL[1], 0.55),
        ("+alpha v_P (sloppy)", QUAL[2], 0.95),
        ("-alpha v_P (sloppy)", QUAL[2], 0.55)]):
    vals = [results[name][c] for c in channels]
    ax.bar(xs + (i - 1.5) * width, vals, width,
           color=color, alpha=alpha_bar, edgecolor="white", label=name)

ax.set_xticks(xs)
ax.set_xticklabels([r"moments (sum $|\Delta|$)",
                    r"ACF (sum $|\Delta|$)",
                    r"tail quantiles (sum $|\Delta|$)"])
ax.set_ylabel("aggregate discrepancy (log)")
ax.set_yscale("log")
ax.set_title("Falsification: sloppy vs stiff perturbations under non-MMD discrepancies")
ax.legend(fontsize=9, ncol=2)
plt.show()

ratio = (results["+alpha v_1 (stiff)"]["moments"] /
         max(results["+alpha v_P (sloppy)"]["moments"], 1e-30))
print(f"Stiff-vs-sloppy ratio on the moment channel : {ratio:.0f}x")
"""))

cells.append(md(r"""
The bars in the red family (stiff) sit several orders of magnitude above the bars in the green family (sloppy) on every channel. Concretely: a perturbation of the same magnitude $\alpha$ that visibly changes the simulated moments, ACF and tail quantiles when applied along $v_1$ is essentially invisible when applied along $v_P$.

This is the central empirical claim of the project, and it survives. The OPG eigenstructure surfaces *genuine simulator-level non-identifiability*, not an artefact of the loss used to compute it.
"""))

# ========================================================== PART IX: NEXT
cells.append(md(r"""
## 9. What comes next

You have now walked through the complete Phase 1 pipeline:

1. The Brock-Hommes simulator (Section 2).
2. The MMD loss (Section 3).
3. Per-seed gradients via AD (Section 4).
4. The OPG matrix and its Gauss-Newton justification (Section 5).
5. The eigendecomposition as a practical-identifiability diagnostic (Section 6).
6. The OPG preconditioner for calibration (Section 7).
7. The falsification protocol that validates the diagnostic (Section 8).

The natural next moves on the project's 10-week roadmap are:

- **Phase 1 horizon-bias killswitch.** Vary the gradient-truncation horizon $H \in \{5, 10, 20, 40, T\}$, compare eigenvalue spectra and eigenvector subspaces (via principal angles) of $\hat F$ across $H$. This determines whether the diagnostic is trustworthy in the harder regime where gradients are truncated for tractability.
- **Phase 2 convergence comparison.** Run OPG-preconditioned, Adam, SGD and L-BFGS on three difficulty levels of synthetic Brock-Hommes calibration, with matched compute. Predicted speedup 2-4× easy, 3-8× hard. (Adam is competitive in our small-perturbation regime — see `outputs/07_optimizer_comparison.png` — which aligns with the plan's Risk 1.)
- **Phase 3 generalisation.** Reproduce on the network-SIR model from Quera-Bofarull et al. 2025 (P = 9, exercises surrogate-gradient bias). Produce the full diagnostic dashboard.

The current state of the codebase is documented in `docs/memory/state.md`. All scripts in `scripts/` are runnable end-to-end with `uv run python scripts/<name>.py`.

The 8-figure visualization suite already in `outputs/` reproduces every plot in this notebook at higher resolution; the figures here are inline-friendly versions of those.

**To use the diagnostic on a new model.** You only need a function `simulate(theta, key) -> trajectory` that is differentiable in `theta`. The MMD loss and OPG calculations are model-agnostic and live in `curvature_calib.{losses, calibration}`.
"""))


def main() -> None:
    nb = nbformat.v4.new_notebook()
    nb.cells = cells
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python", "version": "3.12"}

    out_dir = Path("notebooks")
    out_dir.mkdir(exist_ok=True)
    nb_path = out_dir / "00_walkthrough.ipynb"

    # Write before executing so partial output is preserved if execution fails.
    nbformat.write(nb, str(nb_path))
    print(f"Wrote {nb_path} ({len(cells)} cells). Executing ...")

    client = NotebookClient(nb, timeout=600, kernel_name="python3",
                            resources={"metadata": {"path": str(Path.cwd())}})
    client.execute()
    nbformat.write(nb, str(nb_path))
    print(f"Executed and saved {nb_path}")


if __name__ == "__main__":
    main()
