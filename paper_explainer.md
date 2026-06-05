# Sloppy by Construction — a self-contained explainer

*A pedagogical companion to the paper "Sloppy by Construction: A Live OPG Identifiability Diagnostic for Differentiable Agent-Based Calibration." Written to be read cold by anyone with a working knowledge of calculus, probability, and one prior encounter with stochastic optimisation. Notation is established as it is used; nothing is assumed beyond linear algebra.*

---

## 1. The problem in one paragraph

Agent-based models (ABMs) are simulators. Each one is a chain of rules — heterogeneous agents interacting under stochastic dynamics — that produces synthetic data resembling, in distribution, the empirical record one is trying to explain. Their *calibration* is the inverse problem: given an observed trajectory $y$, find parameter values $\theta$ such that the simulator's output distribution at $\theta$ matches the data. The orthodox bottleneck is that ABMs have no closed-form likelihood, so the usual maximum-likelihood machinery does not apply. The recent bottleneck, less often named but more consequential, is that even when one *does* have a tractable surrogate loss and a working optimiser, **the calibration silently fails along directions in parameter space that the loss cannot distinguish from noise**. The optimiser converges to a low loss value while wandering off the true parameter by orders of magnitude in those directions. No published diagnostic flags which directions, before the compute is burned. That is the gap this paper fills.

---

## 2. What "differentiable" buys, and what it does not

A *differentiable* ABM is one whose simulator $f_\theta : \Xi \to \mathbb{R}^T$ — mapping a random seed $\xi \in \Xi$ and parameters $\theta \in \mathbb{R}^P$ to an observable trajectory of length $T$ — is written in an automatic-differentiation framework such as JAX. Each agent-level rule (an indicator, a sigmoid, a stochastic transition) is reparameterised so that gradients of the trajectory with respect to $\theta$ are obtained in a single backward pass through the simulator. The Quera-Bofarull line of work has established that this is now tractable for non-trivial ABMs in epidemiology and finance.

The first consequence is the obvious one: a differentiable simulator can be plugged into a gradient-based optimiser. The second consequence, less remarked, is what the paper exploits: **on the way to computing the scalar loss gradient, the calibrator computes an entire buffer of per-seed gradients.**

Concretely, suppose the loss is an empirical mean over $M$ simulator replicas under independent seeds $\xi_1, \dots, \xi_M$:
$$
L(\theta) \;=\; \widehat{\mathrm{MMD}}^2\bigl(\{f_\theta(\xi_m)\}_{m=1}^M,\; Y_{\mathrm{ref}}\bigr).
$$
The chain rule decomposes the total gradient into a sum over seed-level contributions:
$$
\nabla_\theta L \;=\; \frac{1}{M} \sum_{m=1}^M g_m, \qquad g_m \;=\; M \cdot \frac{\partial L}{\partial x_m}\,\frac{\partial x_m}{\partial \theta}, \quad x_m = f_\theta(\xi_m).
$$
The factor of $M$ in $g_m$ is bookkeeping so that $\mathrm{mean}(g_m) = \nabla L$ — i.e., each $g_m \in \mathbb{R}^P$ is the per-seed contribution rescaled so that averaging them returns the autodiff total gradient. **These $g_m$ exist inside the autodiff trace whether or not the calibrator chooses to look at them.** That fact is the paper's pivot.

---

## 3. The MMD loss: what it is, and why its structure matters

Maximum mean discrepancy (MMD; Gretton et al. 2012) is a kernel two-sample distance. Given two empirical samples $X = \{x_1, \dots, x_M\}$ and $Y = \{y_1, \dots, y_M\}$ and a positive-definite kernel $k$, the unbiased squared MMD is
$$
\widehat{\mathrm{MMD}}^2(X, Y) \;=\; \frac{1}{M(M-1)} \sum_{m \neq m'} k(x_m, x_{m'}) \;+\; \frac{1}{M(M-1)} \sum_{m \neq m'} k(y_m, y_{m'}) \;-\; \frac{2}{M^2} \sum_{m,m'} k(x_m, y_{m'}).
$$
With a Gaussian RBF kernel $k(a, b) = \exp(-\|a - b\|^2 / 2\sigma^2)$ and bandwidth $\sigma$ chosen by the *median heuristic* — the median pairwise distance in the combined sample — this is a workhorse loss for likelihood-free calibration. The MMD vanishes if and only if the two samples are drawn from the same distribution (up to the kernel's universality).

Two structural properties matter for what follows.

**It is a U-statistic.** The unbiased estimator above is an *average over pairs*, not a sum of squares. This is what licenses a residual-form reading of the loss near a minimum (\S6 below).

**It is computed on simulator output, not on individual seeds.** This is what makes the per-seed gradients $g_m$ interesting. Because $L$ depends on all $M$ seeds jointly through the kernel, the seed-level contribution $g_m$ is not a noisy copy of $\nabla L$ — it is a structured residual, and the *covariance* of those residuals carries information about which directions in parameter space the loss can or cannot distinguish.

---

## 4. The OPG matrix

The outer-product-of-gradients (OPG) matrix at $\theta$ is the empirical second moment of the per-seed gradients:
$$
\widehat F(\theta) \;=\; \frac{1}{M} \sum_{m=1}^{M} g_m\, g_m^{\top} \;\in\; \mathbb{R}^{P \times P}.
$$
$\widehat F$ is symmetric and positive semi-definite by construction. It admits an eigendecomposition
$$
\widehat F\, v_k \;=\; \lambda_k\, v_k, \qquad \lambda_1 \geq \lambda_2 \geq \dots \geq \lambda_P \geq 0,
$$
with orthonormal eigenvectors $v_k \in \mathbb{R}^P$ and non-negative eigenvalues $\lambda_k$. We will be reading both:

- The eigenvalues $\lambda_k$ tell us *how strongly* the loss varies, on average over seeds, when we move along direction $v_k$. A large $\lambda_k$ means the per-seed gradients all point sharply along $v_k$: the loss "cares" about this direction. A small $\lambda_k$ means the per-seed gradients are mutually cancelling along $v_k$: the loss is statistically blind to motion along it.
- The eigenvectors $v_k$ tell us *which combinations of parameters* are sharply or blindly seen. These are unit vectors in $\theta$-space; their components on the canonical basis read off the parameter combinations the calibrator is, or is not, identifying.

This is the diagnostic. The matrix is constructed from the gradient buffer the calibrator already has; the eigendecomposition is $O(P^3)$ in the small parameter count of agent-based models ($P \sim 5\text{--}10$), i.e., milliseconds. The pipeline is summarised in Figure 1 of the paper.

---

## 5. Sloppiness: the language we are stealing

The vocabulary of "stiff" and "sloppy" directions comes from systems biology, in particular Gutenkunst et al. (2007) and the Sethna group's subsequent work. They observed that for a wide class of nonlinear models — biochemical reaction networks, in their case — the Fisher information matrix at the maximum-likelihood parameter routinely has eigenvalues spanning many orders of magnitude. The stiffest eigenvectors point in directions the data constrains tightly; the sloppiest in directions along which the model can be perturbed by large amounts without measurable effect on its predictions. The term *sloppy* is technical and slightly unfortunate: it does not mean the model is bad, but that the data is structurally unable to pin down certain combinations of parameters, however many observations one collects.

The relevance to ABMs is that the same geometry shows up — not in the Fisher information (which would require a likelihood), but in the empirical OPG matrix of an MMD-calibration loss. *That* is the first methodological transfer of the paper: the sloppiness diagnosis, originally developed with the true Fisher in mind, ports over to a likelihood-free setting via the per-seed gradients of a differentiable simulator under a U-statistic loss.

---

## 6. The Kunstner caveat, and why the paper turns it into a contribution

A natural reading of $\widehat F$ is: it is an *empirical Fisher matrix*, and we are doing natural-gradient descent under it. **This reading is, strictly speaking, wrong.** Kunstner, Hennig and Balles (2019) showed that for likelihood maximisation the empirical Fisher (OPG of log-likelihood gradients) and the true Fisher (expected outer product) are genuinely different matrices: they differ by the gradient covariance, and using OPG as a preconditioner in place of the true Fisher can mislead an adaptive optimiser. This is the canonical objection to any work that uses $\widehat F$ for curvature-adaptive optimisation.

The paper makes two moves in response.

**The terminological move.** Call $\widehat F$ the OPG matrix throughout, never the empirical Fisher. The terminology is now standard in the optimisation literature; the paper enforces it consistently. (The phrase "empirical Fisher" appears exactly once in the method section, as a meta-disclaimer.)

**The substantive move.** Read $\widehat F$ as a stochastic *generalised Gauss-Newton* approximation of the loss Hessian, not as a surrogate for the Fisher. Near a minimum of the unbiased MMD U-statistic, the loss takes a residual form: the integrand has a small residual factor multiplying the Jacobian of the simulator. In that regime, the per-seed gradient $g_m$ behaves as a Jacobian row of a vanishing residual, and the outer-product matrix $\widehat F = \frac{1}{M}\sum g_m g_m^\top$ becomes a stochastic approximation of the GGN. This reading is independent of any likelihood claim, because the U-statistic structure substitutes for the log-likelihood structure that would otherwise be needed.

The deeper point — and this is where the Kunstner caveat becomes a contribution rather than an apology — is that **the failure mode Kunstner identified for likelihood optimisation appears in non-likelihood ABM calibration too**. A curvature-adaptive optimiser (Adam) whose preconditioning is misled by the gradient-noise covariance amplifies precisely the sloppy directions the OPG eigendecomposition flags. We do not claim $\widehat F$ is the right preconditioner. We claim that the *direction* of Adam's failure is predicted by $\widehat F$'s eigenbasis, and the paper verifies it empirically.

---

## 7. Non-circular validation

A natural reviewer worry: if we use the eigenstructure of $\widehat F$ — which is derived from the MMD loss — to claim that the MMD loss cannot constrain a parameter combination, we are arguing in a circle. Maybe $\widehat F$'s sloppy direction is an artefact of the kernel, not a property of the underlying model.

The paper defuses this with a protocol I will call **non-MMD falsification**. Pick a parameter $\theta^*$ and perturb it along the stiff eigenvector $v_1$ and along the sloppy eigenvector $v_P$, by the same step size $\alpha = 10^{-2}$. Simulate trajectories at $\theta^* + \alpha v_1$, $\theta^* - \alpha v_1$, $\theta^* + \alpha v_P$, $\theta^* - \alpha v_P$, and at $\theta^*$ itself. Now compute, *not* the MMD discrepancy, but three discrepancies *the calibrator never saw*:

- the first four moments of the trajectory,
- the autocorrelation function up to lag 20,
- four tail quantiles ($1\%, 5\%, 95\%, 99\%$).

These are summary statistics, not kernel embeddings; the MMD kernel mean does not appear anywhere in the test. For each channel $\phi$, the aggregate discrepancy is
$$
\Delta_\phi(v_k) \;=\; \sum_i \bigl|\phi_i\bigl(X_{\theta^* + \alpha v_k}\bigr) - \phi_i\bigl(X_{\theta^*}\bigr)\bigr|.
$$
If the OPG diagnostic identifies real identifiability structure of the *model*, the ratio $\Delta_\phi(v_1) / \Delta_\phi(v_P)$ should be large on every channel: moving along the stiff direction should measurably change the trajectory's statistics; moving along the sloppy direction should not. If instead the diagnostic identifies a kernel artefact, the ratio should be close to one — moves along $v_1$ and $v_P$ should both be visible (or invisible) to a non-MMD discrepancy.

This is the spirit of the protocol. The paper's headline empirical finding is that the ratio is enormous on every model and every channel: from $489\times$ at its smallest (Brock-Hommes moments) to $9.3 \times 10^5$ at its largest (mean-field SIR autocorrelation). Every one of the twelve panels in Figure 3 of the paper exceeds $10^2$. The diagnostic survives a test it could in principle have failed.

---

## 8. The three models, briefly

The paper stress-tests the diagnostic on two model families across three regimes. Briefly:

**Brock-Hommes (1997, 1998)** is a canonical heterogeneous-agent asset-pricing model. Agents switch between forecasting rules — extrapolators of past returns versus fundamentalists betting on mean reversion — with a switching intensity $\beta$ that governs how rationally they respond to past forecasting profit. The model has five parameters $(\beta, g_1, b_1, g_2, b_2)$ controlling the switching intensity and the two heuristics' coefficients. It is the classical testbed for ABM calibration because its phase portrait ranges from a stable fixed point through limit cycles to deterministic chaos as $\beta$ rises.

**Mean-field SIR (Kermack-McKendrick 1927)**, the simplest epidemic compartmental model. A population is partitioned into susceptible, infected and recovered fractions, evolving by deterministic ODEs (here, with a Euler discretisation under JAX for differentiability) under transmission rate $\beta$, recovery rate $\gamma$, and an exogenously imposed lockdown intensity $f_{\mathrm{lock}}$ that activates at time $t_{\mathrm{lock}}$. Five parameters total. Mean-field SIR has *no* network structure; transmission is mass-action.

**Network-SIR with Gumbel-Sigmoid surrogates** is the same SIR model run on an Erdős-Rényi graph of $N = 250$ nodes, with per-node infection and recovery happening as stochastic, individual events. Discrete-state transitions are not naturally differentiable: the gradient of an indicator function is a delta, which autodiff cannot consume. The standard workaround is the *Gumbel-Sigmoid* surrogate (Maddison-Mnih-Teh 2017; Jang-Gu-Poole 2017): replace each Bernoulli decision $\mathbb{1}[\mathrm{logit} > 0]$ with a soft sigmoid relaxation $\sigma((\mathrm{logit} + g) / \tau)$, where $g$ is a Gumbel-distributed noise sample and $\tau$ is a temperature. As $\tau \to 0$ the surrogate recovers the hard decision; for $\tau > 0$ the gradient is a biased but tractable estimator of the true response. The standing concern with this regime is that the resulting gradients carry surrogate bias, and a diagnostic that consumes them could simply be reading off the bias rather than identifiability of the model.

The paper's claim is that all three give qualitatively the same diagnostic picture, including the hardest case where the gradients are biased by construction.

---

## 9. The four results, with interpretation

**Result 1: the spectrum is sloppy.** The OPG eigenvalues at a near-truth Brock-Hommes evaluation point span 8.4 orders of magnitude, with condition number $2.4 \times 10^8$. The sloppiest direction $v_P$ is the parameter $\beta$ alone: the MMD loss cannot distinguish among values of the switching intensity at this $\theta^*$. The stiffest direction $v_1$ is the symmetric-bias combination $(b_1 + b_2)/\sqrt{2}$: the data sees clearly when both bias parameters move together, but neither $b_1$ nor $b_2$ on its own. **This is the qualitative content of the diagnostic in one example**: it surfaces combinations that no per-parameter sensitivity analysis can name.

The same section also reports that the eigenstructure is stable from the first iterate, not just at convergence. Bootstrapping the eigenvectors at intermediate calibration iterates of a far-from-equilibrium run, the stiff direction is within $9.7^\circ$ of its converged value at iteration zero, the sloppy within $20.3^\circ$, and both narrow rapidly. *That* is what "live diagnostic" means: a calibrator who computes $\widehat F$ at the first step already sees, before any optimisation, the directions in which a first-order method will struggle.

**Result 2: non-MMD falsification across three models.** Already explained in §7 above. All twelve panels of Figure 3 (three regimes × three discrepancy channels) exceed a $10^2$ stiff/sloppy ratio. The diagnostic is not a self-fulfilling consequence of the MMD kernel.

**Result 3: the diagnostic predicts where Adam fails.** This is the most empirically surprising result. Take three optimisers — Adam at default hyperparameters, SGD at a tuned learning rate, and OPG-preconditioned Levenberg-Marquardt — and run all three from random initialisations at fixed distance from $\theta^*$. Decompose the *recovery error* $\theta_T - \theta^*$ along the OPG eigenbasis at $\theta^*$ (which is known in this synthetic setting). The result is a per-direction breakdown of where the optimiser ended up. The pattern is unambiguous: every optimiser's error grows along small-$\lambda_k$ directions, as the theory predicts; but Adam's error along the sloppiest direction towers above the others by orders of magnitude. Across 15 Brock-Hommes seeds and 10 SIR seeds, Adam diverges from $\theta^*$ in 25 of 25 runs (95% lower confidence bound: 88%). On a five-seed Brock-Hommes subset at matched difficulty, the cross-optimiser ratio Adam's sloppy-direction error / OPG's stiff-direction error reaches $9.4 \times 10^4$.

This is the Kunstner failure mode, now visible in a non-likelihood setting. Adam's per-parameter variance adaptation amplifies the directions in which the gradient is most noise-like — which, for a sloppy loss, are precisely the directions in which the loss carries no real signal. The diagnostic flags this *before* the optimiser runs: a user who computed $\widehat F$ at $\theta_0$ would already know which direction the run is going to wander in.

**Result 4: the diagnostic survives Gumbel-Sigmoid surrogate bias.** The eigenvectors transfer to network-SIR: the stiff direction is dominated by the initial-infection fraction with a contact-rate correction, $v_1 \approx 0.89\, I_0 + 0.46\, \beta$; the sloppy direction is the lockdown intensity, $v_P \approx 0.995\, f_{\mathrm{lock}}$. The latter is the early-weak-versus-late-strong policy degeneracy: a public-health analyst familiar with the model would recognise it as the timing-intensity confound that bedevils any retrospective evaluation of lockdown effectiveness. The spectrum widens to 20.7 orders of magnitude under the surrogate (a numerical artefact of the surrogate's gradient magnitude, partly), and the non-MMD falsification ratios *grow* rather than shrink. Surrogate-gradient bias acts on the magnitudes of $\widehat F$, not on the identifiability structure it carries.

---

## 10. The reframe: diagnostic > preconditioner

The OPG matrix has been used as a curvature surrogate for two decades (Schraudolph 2002; Martens-Grosse 2015's K-FAC is a related construction in the deep-learning literature). The natural assumption is that any new use of $\widehat F$ proposes it as a *better optimiser*: precondition the gradient by $\widehat F^{-1}$, take a Newton-like step, beat Adam on convergence speed.

This is not what the paper claims, and the honest version of the story is that the paper *tried* this framing and was empirically rebuffed. OPG-preconditioned Levenberg-Marquardt does not beat Adam or SGD on the calibration speed measure. SGD at a well-tuned learning rate is statistically tied with OPG-LM on the Brock-Hommes recovery task.

The productive reading of $\widehat F$ for this domain is different: it is **a name for the parameter combinations the data cannot constrain**, available at the cost of one eigendecomposition before the next update is taken. It is a diagnostic. The eigenvectors recover combinations a domain expert would derive from the model equations — symmetric biases on Brock-Hommes, lockdown timing-intensity on SIR. Structurally correct combinations fall out of $\widehat F$ without being put in. That, not preconditioner-replacement, is the use case.

A useful slogan, which the paper takes as its closer: **identifying the parameter combinations the data cannot constrain is a safety property of a calibration pipeline.** A diagnostic that flags an overconfident calibration is more useful than one that silently passes. The same matrix the optimiser would use as a curvature surrogate delivers this flag on every iterate, at no marginal cost.

---

## 11. What the paper does *not* claim, and why

Pre-empting a critical reading, the scope rules of the work are deliberately narrow.

- It does not claim a new optimiser. The optimiser-replacement framing was tested and falsified.
- It does not claim Adam diverges 100% of the time. It claims 25/25 in a finite sample with a 95% lower confidence bound of $\approx 88\%$.
- It does not extend the divergence-tally claim to network-SIR. Only a diagnostic transfer was tested there; no calibration race was run.
- It does not claim that $\widehat F$ approximates the loss Hessian uniformly. The generalised Gauss-Newton reading is a motivation for the eigendecomposition, not a uniform-convergence theorem.
- It does not claim that the eigenvalue *magnitudes* are robust to the gradient horizon $H$ used when backpropagating through the simulator. The qualitative stiff/sloppy hierarchy is stable across $H \in \{5, \dots, 200\}$; absolute eigenvalues shift up to $70\times$ between truncated and full-horizon gradients. Reported spectra in the paper are at the full horizon $H = T = 200$.
- It does not claim a real-data calibration. All experiments use synthetic ground truth at a known $\theta^*$. The natural next test is to apply the diagnostic to actual financial returns or actual case-incidence series; this is named as the immediate extension.

These limits are stated as scope, not weakness. A reviewer who attacks the paper will land where the paper has already conceded.

---

## 12. Wider context, and where this fits

It is worth stepping back to locate this work in three adjacent literatures.

**Sloppiness in systems biology.** Gutenkunst, Transtrum, Sethna and collaborators developed the language of stiff and sloppy directions over twenty years, originally in the context of ODE models of biochemical networks fitted by maximum likelihood. Their tool was the Fisher information matrix at the MLE. The transfer in this paper is to a regime where there is no likelihood and no MLE — but where, thanks to differentiable simulators and a U-statistic surrogate loss, the per-seed-gradient covariance is still computable and still carries identifiability information.

**Stochastic curvature-adaptive optimisation.** Schraudolph's Stochastic Meta-Descent (2002), Martens and Grosse's K-FAC (2015), and the broader natural-gradient literature have used per-sample outer products as proxies for the Fisher information. Kunstner et al. (2019) is the canonical critique of that practice. This paper is unusual in *agreeing* with Kunstner's critique — yes, $\widehat F$ is not a good Fisher surrogate; yes, it can mislead Adam — and turning the critique into a positive content: *that* misleading is precisely what the eigendecomposition predicts, in a setting where MMD has replaced log-likelihood.

**Differentiable ABMs.** Quera-Bofarull and collaborators have established the technical infrastructure: simulators written in JAX, per-seed VJPs, calibration by gradient descent. Their 2025 paper proposes first-order Jacobian sensitivities — per-parameter $\partial L / \partial \theta_i$ — as the identifiability diagnostic for differentiable ABMs. The methodological gap this paper closes is that per-parameter sensitivities cannot surface *combinations*. The OPG eigendecomposition can. On Brock-Hommes the per-parameter Jacobian span is $5 \times 10^2$; the OPG eigenvalue span is $5 \times 10^6$. The four-orders-of-magnitude gap is the off-diagonal coupling — $|\rho(g_1, g_2)| = 0.999$, $|\rho(b_1, b_2)| = 0.96$ — that no per-parameter view can read.

---

## 13. Open questions

Three honest open questions for anyone picking this up.

**Does the diagnostic transfer to real data?** The synthetic-$\theta^*$ setting is what licences the falsification protocol — one can perturb $\theta^*$ along $v_k$ and measure $\Delta_\phi$ exactly because one knows $\theta^*$. On real data, $\theta^*$ does not exist; what one has is a fitted $\hat\theta$. The natural substitute is to validate the eigenvectors against expert-known degeneracies of the model. This is the immediate empirical extension.

**How does the diagnostic scale with $P$?** All experiments here have $P \leq 5$. ABMs of policy interest can have $P$ in the tens. The eigendecomposition itself is $O(P^3)$, trivial at that scale; the cost is in the simulator throughput and in the seed count $M$ needed for a stable $\widehat F$. Whether the eigenvectors remain interpretable at larger $P$ is an empirical question.

**Is the GGN reading provable?** The paper motivates the GGN approximation of $\widehat F$ near a minimum without proving it. A clean proof would identify the regularity conditions on the simulator and the kernel under which $\widehat F$ converges in some norm to the loss Hessian as $M \to \infty$. The U-statistic literature is the natural place to look for the analytical tools.

---

## 14. Reading order if you are picking up the project cold

If you are looking at this for the first time and want to understand what the paper does in minimum time:

1. **The four-figure tour.** Open `outputs/paper/figures/`. Figure 1: sloppy spectrum on Brock-Hommes. Figure 2: non-circular validation across three models. Figure 3: Adam's failure decomposed along the OPG eigenbasis. Figure 4: same diagnostic structure on network-SIR with surrogate gradients. Each figure is captioned with its headline number.

2. **The paper itself.** `paper/main.pdf`. Ten minutes from abstract through Discussion gives the argument; another ten through Limitations gives the scope.

3. **The story arc memory.** `docs/memory/paper_story_arc.md` — the locked claims, the figure inventory, and the scope rules in one place. Useful when revising prose.

4. **The reviewer-hat audit.** `docs/memory/honest_appraisal.md` — six reviewer attacks the project has self-pre-empted, with the experiment that closes each one.

5. **The state file.** `docs/memory/state.md` — the live snapshot of what is implemented, what is verified, and what is next. Always check this first when resuming work.

The codebase itself is in `src/curvature_calib/`. The diagnostic — per-seed gradients to OPG eigendecomposition — is in three files: `losses/mmd.py`, `calibration/per_seed_grads.py`, `calibration/opg.py`. Everything else is plumbing.

---

*This document is a reading aid, not a publication. It is meant to be self-contained for a future collaborator or for the author returning to the project after a gap. If anything in it conflicts with the paper or the memory files, the paper and the memory take precedence; this is a synthesis written at a single point in time and will date.*
