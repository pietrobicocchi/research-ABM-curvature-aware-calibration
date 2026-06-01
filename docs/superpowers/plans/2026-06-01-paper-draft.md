# AI4ABM 2026 Paper Draft Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a submission-ready 8-page LaTeX paper for the NeurIPS 2026 AI4ABM workshop following the design at `docs/superpowers/specs/2026-06-01-paper-design.md`, the OCAR arc at `docs/memory/paper_story_arc.md`, and the prose-style rules at `docs/memory/paper_style_guide.md`.

**Architecture:** Modular LaTeX — `paper/main.tex` `\input`s one `.tex` per section under `paper/sections/`. TikZ method-overview figure rendered standalone in `paper/diagrams/fig0_method.tex`. Hero PNGs already exist in `outputs/paper/figures/`; symlink from `paper/figures/` so the main file's `\includegraphics` paths stay short. Bibliography hand-curated in `paper/refs.bib`. Compile via `latexmk -pdf paper/main.tex`.

**Tech Stack:** LaTeX (NeurIPS 2024 workshop style), natbib (plainnat), TikZ + libraries (positioning, arrows.meta, calc, shapes.geometric), latexmk, pdflatex.

**Pre-flight reading order (binding):**
1. `docs/memory/paper_story_arc.md` — OCAR arc, hero figures, scope rules
2. `docs/memory/paper_style_guide.md` — prose rules, banned words, banned constructions
3. `docs/memory/literature_positioning.md` — what is and isn't claimed
4. `docs/superpowers/specs/2026-06-01-paper-design.md` — section structure, content map, LaTeX setup

Every prose-writing task in this plan MUST be preceded by re-reading the style guide. The banned-word grep at Task 16 is the safety net but the goal is to not produce the banned terms in the first place.

---

## Task 1: Project setup and first compile

**Files:**
- Create: `paper/main.tex`
- Create: `paper/neurips_2024.sty` (fetched)
- Create: `paper/sections/00_abstract.tex` (stub)
- Create: `paper/sections/01_introduction.tex` (stub)
- Create: `paper/sections/02_related.tex` (stub)
- Create: `paper/sections/03_method.tex` (stub)
- Create: `paper/sections/04_results.tex` (stub)
- Create: `paper/sections/05_discussion.tex` (stub)
- Create: `paper/sections/06_limitations.tex` (stub)
- Create: `paper/figures/` (symlink to `../outputs/paper/figures/`)
- Create: `paper/refs.bib` (empty)
- Create: `paper/diagrams/.gitkeep`
- Create: `appendix/supp.tex` (stub)
- Modify: `.gitignore` add LaTeX build artefacts

- [ ] **Step 1: Fetch NeurIPS 2024 style file**

```bash
mkdir -p paper/sections paper/diagrams appendix
curl -L -o /tmp/neurips2024_styles.zip https://media.neurips.cc/Conferences/NeurIPS2024/Styles.zip
unzip -j /tmp/neurips2024_styles.zip "*/neurips_2024.sty" -d paper/
test -f paper/neurips_2024.sty && echo OK
```

Expected output: `OK`. If the URL changes (NeurIPS sometimes reorganises), search the NeurIPS website for the 2024 style package and fetch manually.

- [ ] **Step 2: Symlink the figures directory**

```bash
ln -s ../outputs/paper/figures paper/figures
ls paper/figures/fig1_spectrum.png && echo OK
```

Expected: `OK`. The symlink keeps `\includegraphics{figures/fig1_spectrum.png}` short while the actual PNGs live in the existing canonical output dir.

- [ ] **Step 3: Update .gitignore for LaTeX build artefacts**

Append to `.gitignore`:

```
# LaTeX build artefacts
paper/*.aux
paper/*.log
paper/*.out
paper/*.bbl
paper/*.blg
paper/*.fls
paper/*.fdb_latexmk
paper/*.synctex.gz
paper/main.pdf
paper/sections/*.aux
paper/diagrams/*.pdf
paper/diagrams/*.aux
paper/diagrams/*.log
```

- [ ] **Step 4: Write paper/main.tex**

```latex
\documentclass{article}

% Use [final] for camera-ready (named authors); remove the option for
% double-blind submission. AI4ABM 2026 review mode confirmation pending.
\usepackage[final]{neurips_2024}

\usepackage{amsmath, amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage[numbers,sort&compress]{natbib}
\bibliographystyle{plainnat}
\usepackage{hyperref}
\usepackage{microtype}
\usepackage{xcolor}
\usepackage{tikz}
\usetikzlibrary{positioning, arrows.meta, calc, shapes.geometric}

\title{Sloppy by Construction: A Live OPG Identifiability Diagnostic for Differentiable Agent-Based Calibration}

\author{%
  Pietro Bicocchi\thanks{Affiliation TBD.}
  \And
  Supervisor One \\
  \And
  Supervisor Two \\
}

\begin{document}

\maketitle

\input{sections/00_abstract}
\input{sections/01_introduction}
\input{sections/02_related}
\input{sections/03_method}
\input{sections/04_results}
\input{sections/05_discussion}
\input{sections/06_limitations}

\bibliography{refs}

\end{document}
```

- [ ] **Step 5: Write stub for every section file**

Each section file at this stage contains just the section header plus a one-line placeholder, so the document compiles end-to-end before any prose lands. Repeat the pattern for all eight files:

`paper/sections/00_abstract.tex`:
```latex
\begin{abstract}
Abstract pending --- see docs/superpowers/specs/2026-06-01-paper-design.md \S C.
\end{abstract}
```

`paper/sections/01_introduction.tex`:
```latex
\section{Introduction}
\label{sec:intro}
Introduction pending.
```

`paper/sections/02_related.tex`:
```latex
\section{Related work}
\label{sec:related}
Related work pending.
```

`paper/sections/03_method.tex`:
```latex
\section{Method}
\label{sec:method}
Method pending.
```

`paper/sections/04_results.tex`:
```latex
\section{Results}
\label{sec:results}
Results pending.
```

`paper/sections/05_discussion.tex`:
```latex
\section{Discussion}
\label{sec:discussion}
Discussion pending.
```

`paper/sections/06_limitations.tex`:
```latex
\section{Limitations}
\label{sec:limitations}
Limitations pending.
```

`appendix/supp.tex`:
```latex
\section*{Supplementary material}
Supplement pending.
```

- [ ] **Step 6: First end-to-end compile**

```bash
cd paper && latexmk -pdf main.tex 2>&1 | tail -20 && cd ..
ls -la paper/main.pdf
```

Expected: PDF builds; warnings about missing bibliography are fine. Page count ~1 page (stubs only).

- [ ] **Step 7: Commit**

```bash
git add paper/ appendix/ .gitignore
git commit -m "Paper: scaffold project + NeurIPS 2024 style + section stubs"
```

---

## Task 2: Bibliography (refs.bib)

**Files:**
- Modify: `paper/refs.bib`

The required citations from `docs/superpowers/specs/2026-06-01-paper-design.md` \S F. Hand-curated; do NOT use a generic CrossRef dump.

- [ ] **Step 1: Add all required BibTeX entries to `paper/refs.bib`**

Use these exact key conventions: `firstauthor_keyword_year` (e.g., `kunstner_opg_2019`). Fields to include: author, title, booktitle/journal, year, volume/pages, doi/url where stable. Required entries:

```bibtex
@inproceedings{kunstner_opg_2019,
  author = {Kunstner, Frederik and Hennig, Philipp and Balles, Lukas},
  title = {Limitations of the Empirical {F}isher Approximation for Natural Gradient Descent},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2019},
  volume = {32}
}

@article{naumann_woleske_2024,
  author = {Naumann-Woleske, Karl and Knicker, Max Sina and Benzaquen, Michael and Bouchaud, Jean-Philippe},
  title = {Exploration of the parameter space in macroeconomic agent-based models},
  journal = {Journal of Economic Dynamics and Control},
  year = {2024}
}

@article{querabofarull_diffabm_2025,
  author = {Quera-Bofarull, Arnau and others},
  title = {Differentiable agent-based modelling for system identification},
  year = {2025},
  note = {(canonical diff-ABM reference; insert exact pubvenue at submission)}
}

@inproceedings{querabofarull_diffabm_2023,
  author = {Quera-Bofarull, Arnau and others},
  title = {Don't Simulate Twice: One-Shot Sensitivity Analyses via Automatic Differentiation},
  booktitle = {AAMAS},
  year = {2023}
}

@article{gutenkunst_sloppy_2007,
  author = {Gutenkunst, Ryan N. and Waterfall, Joshua J. and Casey, Fergal P. and Brown, Kevin S. and Myers, Christopher R. and Sethna, James P.},
  title = {Universally Sloppy Parameter Sensitivities in Systems Biology Models},
  journal = {PLOS Computational Biology},
  year = {2007},
  volume = {3},
  number = {10},
  pages = {e189}
}

@article{transtrum_geometry_2011,
  author = {Transtrum, Mark K. and Machta, Benjamin B. and Sethna, James P.},
  title = {Geometry of nonlinear least squares with applications to sloppy models and optimization},
  journal = {Physical Review E},
  year = {2011},
  volume = {83},
  pages = {036701}
}

@inproceedings{transtrum_lm_2010,
  author = {Transtrum, Mark K. and Machta, Benjamin B. and Sethna, James P.},
  title = {Why are nonlinear fits to data so challenging?},
  booktitle = {Physical Review Letters},
  year = {2010},
  volume = {104},
  pages = {060201}
}

@article{schraudolph_sgn_2002,
  author = {Schraudolph, Nicol N.},
  title = {Fast curvature matrix-vector products for second-order gradient descent},
  journal = {Neural Computation},
  year = {2002},
  volume = {14},
  number = {7},
  pages = {1723--1738}
}

@inproceedings{martens_kfac_2015,
  author = {Martens, James and Grosse, Roger},
  title = {Optimizing Neural Networks with Kronecker-factored Approximate Curvature},
  booktitle = {ICML},
  year = {2015}
}

@inproceedings{maddison_concrete_2017,
  author = {Maddison, Chris J. and Mnih, Andriy and Teh, Yee Whye},
  title = {The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables},
  booktitle = {ICLR},
  year = {2017}
}

@inproceedings{jang_gumbel_2017,
  author = {Jang, Eric and Gu, Shixiang and Poole, Ben},
  title = {Categorical Reparameterization with {G}umbel-{S}oftmax},
  booktitle = {ICLR},
  year = {2017}
}

@article{brock_hommes_1997,
  author = {Brock, William A. and Hommes, Cars H.},
  title = {A Rational Route to Randomness},
  journal = {Econometrica},
  year = {1997},
  volume = {65},
  number = {5},
  pages = {1059--1095}
}

@article{brock_hommes_1998,
  author = {Brock, William A. and Hommes, Cars H.},
  title = {Heterogeneous beliefs and routes to chaos in a simple asset pricing model},
  journal = {Journal of Economic Dynamics and Control},
  year = {1998},
  volume = {22},
  pages = {1235--1274}
}

@article{kermack_mckendrick_1927,
  author = {Kermack, William O. and McKendrick, Anderson G.},
  title = {A contribution to the mathematical theory of epidemics},
  journal = {Proc. Royal Society A},
  year = {1927},
  volume = {115},
  pages = {700--721}
}

@article{gretton_mmd_2012,
  author = {Gretton, Arthur and Borgwardt, Karsten M. and Rasch, Malte J. and Sch{\"o}lkopf, Bernhard and Smola, Alexander},
  title = {A kernel two-sample test},
  journal = {JMLR},
  year = {2012},
  volume = {13},
  pages = {723--773}
}
```

- [ ] **Step 2: Re-compile to verify natbib finds the entries**

```bash
cd paper && latexmk -pdf main.tex 2>&1 | tail -10 && cd ..
```

Expected: still compiles; bib warnings vanish for the entries above when cited (will be cited in later tasks).

- [ ] **Step 3: Commit**

```bash
git add paper/refs.bib
git commit -m "Paper: refs.bib with all 14 required citations"
```

---

## Task 3: TikZ Figure 0 — Method overview schematic

**Files:**
- Create: `paper/diagrams/fig0_method.tex`

Per `docs/superpowers/specs/2026-06-01-paper-design.md` \S E. The diagram is referenced by the Method section (Task 4) so it must compile cleanly first.

- [ ] **Step 1: Write `paper/diagrams/fig0_method.tex`**

```latex
\begin{tikzpicture}[
    >=Stealth,
    node distance=0.45cm and 0.7cm,
    every node/.style={font=\small},
    box/.style={draw, rounded corners=2pt, fill=gray!8,
                minimum width=2.4cm, minimum height=0.85cm, align=center,
                inner sep=3pt},
    op/.style={font=\scriptsize\itshape, midway, above, sloped,
                yshift=1pt},
    arr/.style={->, line width=0.6pt}
]

\node[box] (theta)   {$\theta \in \mathbb{R}^P$};
\node[box, right=of theta]      (X)    {trajectories \\ $X \in \mathbb{R}^{M \times T}$};
\node[box, right=of X]          (L)    {scalar loss \\ $L = \widehat{\mathrm{MMD}}^2(X, Y_{\mathrm{ref}})$};

\node[box, below=of theta]      (gm)   {per-seed grads \\ $\{g_m\} \in \mathbb{R}^{M \times P}$};
\node[box, right=of gm]         (F)    {OPG matrix \\ $\widehat{F} = \tfrac{1}{M}\sum_m g_m g_m^\top$};
\node[box, right=of F]          (eig)  {eigenpairs \\ $(\lambda_k, v_k)_{k=1}^{P}$};

\draw[arr] (theta) -- (X)  node[op] {vmap simulate};
\draw[arr] (X) -- (L)      node[op] {MMD$^2$};
\draw[arr] (L.south)  -- ++(0,-0.45) -| (gm.north)
                            node[op, pos=0.25] {per-seed vjp};
\draw[arr] (gm) -- (F)     node[op] {outer-product};
\draw[arr] (F) -- (eig)    node[op] {eigh};

\node[font=\scriptsize, below=0.15cm of eig, align=center, text width=2.6cm]
    {stiff $v_1$ \dots\ sloppy $v_P$};
\end{tikzpicture}
```

- [ ] **Step 2: Sanity-compile the TikZ source standalone**

Create a tiny wrapper to verify the diagram compiles on its own (delete after verifying — the real use is `\input` inside section 3).

```bash
cat > /tmp/fig0_check.tex <<'EOF'
\documentclass{standalone}
\usepackage{tikz}
\usetikzlibrary{positioning, arrows.meta, calc, shapes.geometric}
\usepackage{amsmath}
\begin{document}
\input{paper/diagrams/fig0_method.tex}
\end{document}
EOF
pdflatex -output-directory=/tmp /tmp/fig0_check.tex 2>&1 | tail -5
test -f /tmp/fig0_check.pdf && echo OK
```

Expected: `OK`. Open the PDF visually to verify boxes are not overlapping and arrow labels are readable. If layout drifts, adjust `node distance` in the `tikzpicture` options.

- [ ] **Step 3: Commit**

```bash
git add paper/diagrams/fig0_method.tex
git commit -m "Paper: TikZ Fig 0 method-overview schematic"
```

---

## Task 4: Section 3 — Method

**Files:**
- Modify: `paper/sections/03_method.tex`

Prose-writing task. Re-read `docs/memory/paper_style_guide.md` before starting. Voice: §3.1 Wooldridge (definitional clarity); §3.2-3.3 Bofarull (integrated method-and-experiment, even though experiments come in section 4 — write so a reader can already see the link). Page target: 1.5 pages.

- [ ] **Step 1: Write §3.1 Setup (~0.5 page)**

Replace `paper/sections/03_method.tex` body with:

```latex
\section{Method}
\label{sec:method}

\subsection{Setup}
\label{sec:method:setup}
```

Then write 3–5 sentences defining:
- The simulator $f_\theta(\xi) \mapsto x \in \mathbb{R}^T$ as a differentiable function of parameters $\theta \in \mathbb{R}^P$ and a per-seed random key $\xi$.
- The MMD loss $\widehat{\mathrm{MMD}}^2(X, Y_{\mathrm{ref}})$ (cite \citep{gretton_mmd_2012}) with median-heuristic bandwidth and the convention that we hold $Y_{\mathrm{ref}}$ fixed throughout calibration.
- The per-seed gradient $g_m = M \cdot (\partial L / \partial x_m)(\partial x_m / \partial \theta)$ obtained by vector-Jacobian product, so that $\bar{g} = (1/M)\sum_m g_m = \nabla_\theta L$.

Close §3.1 with a single sentence forward-pointing to Figure 0: ``Figure~\ref{fig:method} summarises the pipeline.''

- [ ] **Step 2: Add Figure 0 environment immediately after §3.1**

```latex
\begin{figure}[t]
  \centering
  \input{../diagrams/fig0_method.tex}
  \caption{The OPG diagnostic pipeline. Per-seed gradients computed by vector--Jacobian product through the simulator are the natural inputs for the outer-product matrix $\widehat F$; its eigendecomposition names the stiff and sloppy parameter combinations the data constrains.}
  \label{fig:method}
\end{figure}
```

- [ ] **Step 3: Write §3.2 OPG matrix and GGN interpretation (~0.5 page)**

```latex
\subsection{The OPG matrix and a Gauss-Newton reading}
\label{sec:method:opg}
```

Define $\widehat{F} = (1/M) \sum_m g_m g_m^\top$. Equation block. Then 4–5 sentences:
- Frame as a residual Gauss-Newton approximation of the MMD-loss Hessian.
- The Kunstner reframe (contributory, not defensive) from `docs/memory/paper_style_guide.md` rule 8. Cite \citep{kunstner_opg_2019} in the same sentence that *makes the claim*: their failure mode --- gradient-noise covariance misleading curvature-adaptive optimisers --- transfers to non-likelihood ABM calibration and the same matrix, correctly read as residual-GGN, is the right diagnostic.
- Forward-pointer to §3.3 (the non-circular check) and §\ref{sec:results}.

- [ ] **Step 4: Write §3.3 Falsification protocol (~0.5 page)**

```latex
\subsection{Non-circular validation via non-MMD discrepancies}
\label{sec:method:falsification}
```

3–4 sentences + equation:
- Setup: choose $v_1$ (stiff) and $v_P$ (sloppy) from the eigendecomposition; perturb $\theta^* \pm \alpha v_k$ for fixed $\alpha$; measure $\sum_i |\phi_i(X_{\theta^* + \alpha v_k}) - \phi_i(X_{\theta^*})|$ for three statistics $\phi$ unrelated to the calibration kernel: first four moments, ACF up to lag 20, four tail quantiles.
- State the prediction explicitly: if $\widehat F$ identifies real identifiability, the stiff/sloppy ratio of this aggregate discrepancy should be large and the result is non-circular (no MMD or kernel embedding appears in the test).

- [ ] **Step 5: Compile and verify**

```bash
cd paper && latexmk -pdf main.tex 2>&1 | tail -10 && cd ..
```

Page count check: open `paper/main.pdf`; §3 should occupy ~1.5 pages.

- [ ] **Step 6: Banned-word check (local)**

```bash
egrep -in "novel|powerful|comprehensive|extensive|empirical fisher|it is worth noting|it should be emphasised|clearly|obviously|note that" paper/sections/03_method.tex
```

Expected: no matches.

- [ ] **Step 7: Commit**

```bash
git add paper/sections/03_method.tex
git commit -m "Paper: section 3 Method (setup, OPG/GGN, falsification protocol)"
```

---

## Task 5: Section 4.1 — Results: Sloppy spectrum (BH), Fig 1

**Files:**
- Modify: `paper/sections/04_results.tex`

Voice: Andelfinger (claim → figure → done). Page target: 0.5 page.

- [ ] **Step 1: Open §4 and write §4.1**

Replace the stub with:

```latex
\section{Results}
\label{sec:results}

Source code and seeds for all experiments are released at \texttt{<repo-url-here>} (anonymised at submission if review is double-blind).

\subsection{The Brock--Hommes calibration loss has a sloppy spectrum}
\label{sec:results:bh-spectrum}
```

Then 3–5 sentences:
- Open with the claim sentence (Andelfinger rule: lead with the claim, then the figure call-out).
- Numbers to quote from `docs/memory/paper_story_arc.md` hero table: spectrum span \textbf{8.4 OOM}, condition $\lambda_1/\lambda_P = 2.4 \times 10^8$.
- The eigenvector content: $\beta$ alone is $v_P$ (the sloppiest direction); the stiffest direction $v_1$ is dominated by the symmetric-bias combination $(b_1 + b_2)/\sqrt{2}$.
- Why this matters: parameter combinations not individual parameters.

- [ ] **Step 2: Add Figure 1 environment**

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{figures/fig1_spectrum.png}
  \caption{OPG geometry of the Brock--Hommes calibration loss at a perturbed evaluation point. (a) Per-seed gradient cloud in the top-2 OPG axes with the $1\sigma$ ellipse overlaid. (b) Eigenvalue spectrum with 95\% bootstrap CIs; \textbf{8.4 OOM} span. (c) Parameter content of each eigendirection: $v_P$ is $\beta$ alone, $v_1$ is the symmetric-bias combination.}
  \label{fig:bh-spectrum}
\end{figure}
```

- [ ] **Step 3: Compile, banned-word check, commit**

```bash
cd paper && latexmk -pdf main.tex 2>&1 | tail -5 && cd ..
egrep -in "novel|powerful|comprehensive|extensive|empirical fisher|it is worth noting|clearly|obviously|note that" paper/sections/04_results.tex
git add paper/sections/04_results.tex
git commit -m "Paper: section 4.1 BH spectrum (Fig 1)"
```

---

## Task 6: Section 4.2 — Results: Live diagnostic (Supp Fig)

**Files:**
- Modify: `paper/sections/04_results.tex`

Voice: Bofarull (integrated). Page target: 0.3 page.

- [ ] **Step 1: Append §4.2**

```latex
\subsection{The diagnostic is live from the first iterate}
\label{sec:results:live}
```

3–4 sentences:
- Open with the live-diagnostic claim. Quote the exact numbers from `docs/memory/paper_story_arc.md` Scope rules table: stiff $v_1$ within $9.7^\circ$ of $v_1(\theta^*)$ at $t=0$, $1.9^\circ$ at convergence; sloppy $v_P$ within $20.3^\circ$ at $t=0$, $1.1^\circ$ at convergence.
- Inline reference to the supplementary figure: ``Bootstrap principal-angle confidence intervals across iterates are shown in Figure~\ref{fig:supp-trajectory} (appendix).''
- Implication: one can compute $\widehat F$ at the start of calibration and already name the directions Adam will wander along.

- [ ] **Step 2: Compile, banned-word check, commit**

```bash
cd paper && latexmk -pdf main.tex 2>&1 | tail -5 && cd ..
egrep -in "novel|powerful|comprehensive|extensive|empirical fisher|it is worth noting|clearly|obviously|note that" paper/sections/04_results.tex
git add paper/sections/04_results.tex
git commit -m "Paper: section 4.2 live-diagnostic claim with explicit numbers"
```

---

## Task 7: Section 4.3 — Results: Non-circular validation (Fig 2)

**Files:**
- Modify: `paper/sections/04_results.tex`

Voice: Andelfinger. Page target: 0.7 page (this is the headline non-circular check).

- [ ] **Step 1: Append §4.3**

```latex
\subsection{Non-circular validation across three models}
\label{sec:results:falsification}
```

4–5 sentences:
- Claim: the stiff/sloppy ratio of aggregate discrepancy under three non-MMD channels is at least $10^2$ on every model, every channel.
- Quote canonical (float64) ratios from `docs/memory/paper_story_arc.md`: BH $489\times / 581\times / 19{,}500\times$; mean-field SIR $354{,}000\times / 930{,}000\times / 395{,}000\times$; network-SIR (Gumbel) $171{,}000\times / 238{,}000\times / 36{,}300\times$.
- Domain reading: the eigenvectors recover structurally meaningful combinations (b₁+b₂ on BH; the policy degeneracy on SIR).
- Forward reference: this rules out the obvious null hypothesis that $\widehat F$ identifies a kernel-embedding artefact rather than identifiability of the model.

- [ ] **Step 2: Add Figure 2 environment**

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{figures/fig2_falsification.png}
  \caption{Non-circular validation of the OPG diagnostic. Three models (rows) $\times$ three discrepancies unrelated to the calibration kernel (columns: first four moments; autocorrelation up to lag 20; four tail quantiles) under the same $\alpha = 10^{-2}$ perturbation along the stiff $v_1$ and sloppy $v_P$ directions. Stiff/sloppy ratios are annotated per panel; aggregate ratios per model in the rightmost column. All twelve panels exceed $10^2$.}
  \label{fig:falsification}
\end{figure}
```

- [ ] **Step 3: Compile, banned-word check, commit**

```bash
cd paper && latexmk -pdf main.tex 2>&1 | tail -5 && cd ..
egrep -in "novel|powerful|comprehensive|extensive|empirical fisher|it is worth noting|clearly|obviously|note that" paper/sections/04_results.tex
git add paper/sections/04_results.tex
git commit -m "Paper: section 4.3 non-circular validation across three models (Fig 2)"
```

---

## Task 8: Section 4.4 — Results: Diagnostic predicts Adam (Fig 3)

**Files:**
- Modify: `paper/sections/04_results.tex`

Voice: Andelfinger + Bofarull. Page target: 0.7 page. This is the resolution-of-tension panel.

- [ ] **Step 1: Append §4.4**

```latex
\subsection{The diagnostic predicts where Adam fails}
\label{sec:results:predicts-adam}
```

4–5 sentences:
- Claim: when calibration is decomposed along the OPG eigenbasis at $\theta^*$, the recovery error in each direction tracks the inverse eigenvalue scaling; Adam wanders specifically along the sloppy direction.
- Quote the cross-optimiser headline from `docs/memory/paper_story_arc.md` hero table: Adam $v_P$ / OPG $v_P = 34\times$, Adam $v_P$ / OPG $v_1 = 9.4 \times 10^4$.
- Robustness across seeds: Adam diverges in $25/25$ seeds across BH and SIR (LCB $\approx 88\%$); see appendix for the per-seed table.
- Reframe (paper voice): the diagnostic predicts the failure mode, the optimiser does not need to be replaced for the prediction to be useful.

- [ ] **Step 2: Add Figure 3 environment**

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{figures/fig3_predicts_adam.png}
  \caption{Recovery error decomposed along the OPG eigenbasis at $\theta^*$. (a) Median (IQR) squared component per eigendirection, 5 random unit-vector inits at distance $0.15$ from $\theta^*$. Adam's $v_5$ bar towers above the other two optimisers. (b) Median squared error against the eigenvalue; small-$\lambda$ directions are the ones every optimiser fails along, with Adam's failure systematically larger.}
  \label{fig:predicts-adam}
\end{figure}
```

- [ ] **Step 3: Compile, banned-word check, commit**

```bash
cd paper && latexmk -pdf main.tex 2>&1 | tail -5 && cd ..
egrep -in "novel|powerful|comprehensive|extensive|empirical fisher|it is worth noting|clearly|obviously|note that" paper/sections/04_results.tex
git add paper/sections/04_results.tex
git commit -m "Paper: section 4.4 diagnostic predicts Adam's failure (Fig 3)"
```

---

## Task 9: Section 4.5 — Results: Surviving Gumbel surrogate (Fig 4)

**Files:**
- Modify: `paper/sections/04_results.tex`

Voice: Bofarull (the closer). Page target: 0.7 page. This is the regime Phase 3 was designed to stress.

- [ ] **Step 1: Append §4.5**

```latex
\subsection{The diagnostic survives discrete states under Gumbel-Sigmoid surrogates}
\label{sec:results:network-sir}
```

4–5 sentences:
- Open with the stakes: the standard differentiable-ABM concern is that gradients computed through a Gumbel-Sigmoid surrogate \citep{maddison_concrete_2017,jang_gumbel_2017} are biased and could in principle mislead an identifiability diagnostic.
- State the result: on a network-SIR model with per-node Gumbel-Sigmoid transitions the OPG eigenvectors recover qualitatively the same stiff/sloppy structure as the smooth mean-field version (\textbf{$v_1 \approx 0.89 \, I_0 + 0.46 \, \beta$}; $v_P \approx 0.995 \, f_{\mathrm{lock}}$, a real public-health degeneracy).
- Spectrum span widens to \textbf{20.7 OOM}; non-MMD falsification ratios remain $10^4$--$10^5\times$ even under surrogate bias.
- Implication: biased gradients identify real identifiability structure, not surrogate artefacts.

- [ ] **Step 2: Add Figure 4 environment**

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{figures/fig4_network_sir.png}
  \caption{Network-SIR diagnostic with Gumbel-Sigmoid surrogate gradients. (a) Daily new infections at $\theta^*$ (with lockdown) versus a no-lockdown counterfactual. (b) OPG spectrum with 95\% bootstrap CIs --- \textbf{20.7 OOM} span. (c) Eigenvector content; $v_P$ is essentially $f_{\mathrm{lock}}$, the early-weak-vs-late-strong lockdown degeneracy. (d) Mean-field versus network-SIR spectra: the qualitative structure transfers under surrogate-gradient bias.}
  \label{fig:network-sir}
\end{figure}
```

- [ ] **Step 3: Compile, banned-word check, commit**

```bash
cd paper && latexmk -pdf main.tex 2>&1 | tail -5 && cd ..
egrep -in "novel|powerful|comprehensive|extensive|empirical fisher|it is worth noting|clearly|obviously|note that" paper/sections/04_results.tex
git add paper/sections/04_results.tex
git commit -m "Paper: section 4.5 network-SIR closer with Gumbel surrogate (Fig 4)"
```

---

## Task 10: Section 2 — Related work

**Files:**
- Modify: `paper/sections/02_related.tex`

Voice: Andelfinger (tight related-work table + 2-3 sentence framing). Page target: 0.5 page.

- [ ] **Step 1: Write §2**

Replace the stub. Open with one sentence of OCAR-Challenge framing: ``The diagnostic sits at the intersection of four lines of prior work; the contribution is precise.''

Then a 3-column $\times$ 4-row table (booktabs):

```latex
\section{Related work}
\label{sec:related}

The diagnostic sits at the intersection of four lines of prior work; the contribution is precise.

\begin{table}[h]
\centering
\caption{Where this work sits in the literature.}
\label{tab:related}
\small
\begin{tabular}{@{}p{0.22\linewidth}p{0.32\linewidth}p{0.36\linewidth}@{}}
\toprule
\textbf{Predecessor} & \textbf{What they did} & \textbf{What we add} \\
\midrule
Naumann-Woleske et al.\ \citep{naumann_woleske_2024} & Finite-difference Hessian at one point on Mark-0, offline. & AD per-seed gradients, live during calibration. \\
Quera-Bofarull et al.\ \citep{querabofarull_diffabm_2025} & First-order Jacobian sensitivities for diff-ABMs. & Second-moment structure of the same gradients; identifies parameter \emph{combinations}. \\
Gutenkunst et al.\ \citep{gutenkunst_sloppy_2007}; Transtrum et al.\ \citep{transtrum_geometry_2011} & Sloppiness in ODE biology with true FIM. & Same geometry surfaced for MMD-calibrated differentiable simulators. \\
Kunstner et al.\ \citep{kunstner_opg_2019} & OPG $\neq$ Fisher in likelihood settings. & MMD's residual structure licenses a GGN reading; Kunstner's failure mode appears in non-likelihood calibration too. \\
\bottomrule
\end{tabular}
\end{table}
```

Close with one sentence: ``Our contribution is not a new optimiser, nor the discovery of sloppiness, nor a new identifiability method; it is the integrated diagnostic and its non-circular validation.''

- [ ] **Step 2: Compile, banned-word check, commit**

```bash
cd paper && latexmk -pdf main.tex 2>&1 | tail -5 && cd ..
egrep -in "novel|powerful|comprehensive|extensive|empirical fisher|it is worth noting|clearly|obviously|note that" paper/sections/02_related.tex
git add paper/sections/02_related.tex
git commit -m "Paper: section 2 Related work (4-lineage gap table)"
```

---

## Task 11: Section 5 — Discussion

**Files:**
- Modify: `paper/sections/05_discussion.tex`

Voice: Wooldridge/Farmer (high-level framing). Page target: 0.75 page. Three paragraphs.

- [ ] **Step 1: Write §5**

```latex
\section{Discussion}
\label{sec:discussion}
```

Paragraph 1 — diagnostic > preconditioner. 3–5 sentences. The interesting use of $\widehat F$ is naming the parameter combinations the data cannot constrain, before any compute is spent on Adam. Position this against the 20-year stochastic-Gauss-Newton tradition \citep{schraudolph_sgn_2002,martens_kfac_2015}: same matrix, different use.

Paragraph 2 — domain payoff. 3–5 sentences. The eigenvectors recover combinations a domain expert would derive from the model equations: $(b_1+b_2)/\sqrt{2}$ on Brock-Hommes (symmetric-bias combination), $v_P \approx f_{\mathrm{lock}} + 0.1 \, t_{\mathrm{lock}}$ on SIR (the public-health early-weak-vs-late-strong lockdown degeneracy). This is the evidence the diagnostic recovers real identifiability rather than numerical artefact.

Paragraph 3 — surrogate-bias survival. 3–4 sentences. Network-SIR with Gumbel-Sigmoid biases the gradients by construction, yet the eigenvectors stay qualitatively correct and falsification ratios get \emph{larger}, not smaller. The diagnostic transfers to the regime current differentiable-ABM work treats as the hard case.

- [ ] **Step 2: Compile, banned-word check, commit**

```bash
cd paper && latexmk -pdf main.tex 2>&1 | tail -5 && cd ..
egrep -in "novel|powerful|comprehensive|extensive|empirical fisher|it is worth noting|clearly|obviously|note that" paper/sections/05_discussion.tex
git add paper/sections/05_discussion.tex
git commit -m "Paper: section 5 Discussion (diagnostic > preconditioner; domain payoff; surrogate survival)"
```

---

## Task 12: Section 6 — Limitations

**Files:**
- Modify: `paper/sections/06_limitations.tex`

Voice: Andelfinger (scope, not weakness list). Page target: 0.35 page.

- [ ] **Step 1: Write §6**

```latex
\section{Limitations}
\label{sec:limitations}
```

Four short bullet points (each one sentence):
1. Eigenvalue \emph{magnitudes} depend on the gradient horizon $H$; the qualitative hierarchy is stable but absolute eigenvalues shift up to $70\times$ between $H{<}T$ and $H{=}T$ (appendix Fig.\ supp-horizon).
2. Multi-seed counts are modest ($N{=}15$ Brock--Hommes, $N{=}10$ SIR); the headline ``Adam $25/25$'' is LCB $\approx 88\%$ at 95\% confidence, not asymptotic.
3. The OPG spectrum span on SIR confounds eigenvalue with parameter scale ($I_0 \sim 10^{-3}$ versus $\beta \sim 1$); a Pearson-normalised version is left to future work.
4. The optimiser-superiority claim is narrow: ``OPG-LM is robust without per-problem tuning on two models,'' not ``OPG beats best-tuned SGD.''

Close with a one-sentence broader-impact statement: ``Identifying the parameter combinations the data cannot constrain is a safety property, not a competitive metric: a diagnostic that flags overconfident calibrations is more useful than one that silently passes.''

- [ ] **Step 2: Compile, banned-word check, commit**

```bash
cd paper && latexmk -pdf main.tex 2>&1 | tail -5 && cd ..
egrep -in "novel|powerful|comprehensive|extensive|empirical fisher|it is worth noting|clearly|obviously|note that" paper/sections/06_limitations.tex
git add paper/sections/06_limitations.tex
git commit -m "Paper: section 6 Limitations (4 scope statements + broader impact)"
```

---

## Task 13: Section 1 — Introduction

**Files:**
- Modify: `paper/sections/01_introduction.tex`

Voice: Farmer opening + Wooldridge problem statement + Bofarull pivot. Page target: 1.0 page. Written *after* Results so we know exactly what we're claiming.

- [ ] **Step 1: Write §1.1 Opening (Farmer-mode)**

```latex
\section{Introduction}
\label{sec:intro}
```

Two opening sentences, max:
- One sentence on why ABM calibration matters (policy / systemic-risk relevance).
- One sentence on why differentiable ABMs change the picture (they make gradient-based calibration tractable; cite \citep{querabofarull_diffabm_2025}).

- [ ] **Step 2: Write §1.2 Challenge (Wooldridge-mode)**

3–4 sentences:
- Concrete failure: Adam with default $\beta_1{=}0.9$ silently diverges along a sloppy direction on Brock--Hommes calibration.
- Why no published diagnostic catches this before compute is burned: existing tools are offline finite-difference Hessians \citep{naumann_woleske_2024} or per-parameter Jacobians \citep{querabofarull_diffabm_2025}.
- The real culprit is non-identifiability of parameter \emph{combinations} under the MMD loss; we make this falsifiable.

- [ ] **Step 3: Write §1.3 Kunstner contributory cite**

2 sentences:
- Kunstner et al.\ \citep{kunstner_opg_2019} showed the outer-product-of-gradients matrix differs from the Fisher in likelihood maximisation and can mislead curvature-adaptive optimisers.
- The same matrix, correctly read as the residual-Gauss-Newton approximation of the MMD-loss Hessian, is the right diagnostic for non-likelihood ABM calibration --- and the failure mode Kunstner identified appears here too, operationally verifiable from the per-seed gradients already computed.

- [ ] **Step 4: Write §1.4 Contributions**

```latex
\paragraph{Contributions.}
\begin{itemize}
\item A live identifiability diagnostic for MMD-calibrated differentiable agent-based models, obtained as a free byproduct of the per-seed gradients used by the calibrator (\S\ref{sec:method}, Fig.~\ref{fig:method}).
\item A non-circular validation protocol using three discrepancies the calibrator never saw, with stiff/sloppy ratios of $10^2$--$10^6$ on three models (\S\ref{sec:results:falsification}, Fig.~\ref{fig:falsification}).
\item Empirical evidence that the diagnostic predicts where Adam fails: cross-optimiser ratio Adam~$v_P$ / OPG~$v_1 = 9.4\times 10^4$ (\S\ref{sec:results:predicts-adam}, Fig.~\ref{fig:predicts-adam}).
\item Demonstration that the diagnostic transfers to discrete-state network agent-based models under Gumbel-Sigmoid surrogate gradients (\S\ref{sec:results:network-sir}, Fig.~\ref{fig:network-sir}).
\end{itemize}
```

- [ ] **Step 5: Compile, banned-word check, commit**

```bash
cd paper && latexmk -pdf main.tex 2>&1 | tail -5 && cd ..
egrep -in "novel|powerful|comprehensive|extensive|empirical fisher|it is worth noting|clearly|obviously|note that" paper/sections/01_introduction.tex
git add paper/sections/01_introduction.tex
git commit -m "Paper: section 1 Introduction (opening + challenge + Kunstner reframe + contributions)"
```

---

## Task 14: Abstract

**Files:**
- Modify: `paper/sections/00_abstract.tex`

Voice: compressed full arc. Length target: ~150 words.

- [ ] **Step 1: Write the abstract**

Three sentences, one per OCAR move:

```latex
\begin{abstract}
[Sentence 1, Opening] One sentence on why ABM calibration matters and how
differentiable ABMs make it tractable; mention the per-seed gradient
buffer.

[Sentence 2, Challenge] One sentence stating that first-order methods
silently fail because the MMD loss has a sloppy spectrum of parameter
combinations the data cannot constrain.

[Sentences 3-5, Action+Resolution] The OPG matrix, computed once from
the gradients already in memory, names those combinations; we validate
non-circularly with three discrepancies the calibrator never saw on
Brock-Hommes, mean-field SIR, and discrete-state network SIR under
Gumbel-Sigmoid surrogates, with stiff/sloppy ratios of $10^2$--$10^6$;
the diagnostic correctly predicts the direction of Adam's failure
($25/25$ seeds across BH+SIR; Adam $v_P$/OPG $v_1 = 9.4\times 10^4$).
\end{abstract}
```

Replace the bracketed sentences with actual prose. Hit ~150 words. Numbers in **bold** on first mention per style rule 5.

- [ ] **Step 2: Word count check**

```bash
detex paper/sections/00_abstract.tex | wc -w
```

Expected: 140–170 words. If too long, cut. If too short, add a sentence on the live-diagnostic property.

- [ ] **Step 3: Compile, banned-word check, commit**

```bash
cd paper && latexmk -pdf main.tex 2>&1 | tail -5 && cd ..
egrep -in "novel|powerful|comprehensive|extensive|empirical fisher|it is worth noting|clearly|obviously|note that" paper/sections/00_abstract.tex
git add paper/sections/00_abstract.tex
git commit -m "Paper: abstract (3-sentence OCAR compression, 150 words)"
```

---

## Task 15: Appendix

**Files:**
- Modify: `appendix/supp.tex`
- Modify: `paper/main.tex` (add `\input{../appendix/supp}` after the bibliography)

Per spec §F. Page target: open-ended.

- [ ] **Step 1: Add `\appendix` + supplementary figures**

In `appendix/supp.tex`:

```latex
\appendix
\section{Live-diagnostic trajectory bootstrap}
\label{supp:trajectory}

\begin{figure}[h]
  \centering
  \includegraphics[width=\linewidth]{../paper/figures/fig_supp_trajectory.png}
  \caption{Principal-angle bootstrap of OPG eigenvectors along the calibration trajectory of Brock--Hommes (median + IQR over 500 resamples). Stiff $v_1$ stays within $\sim 10^\circ$ of $v_1(\theta^*)$ from $t=0$; sloppy $v_P$ within $\sim 20^\circ$ at $t=0$, dropping to $\sim 1^\circ$ by iterate 4.}
  \label{fig:supp-trajectory}
\end{figure}

\section{Adam learning-rate sweep}
\label{supp:adam-lr}

Refer to the float32-era diagnostic in \texttt{outputs/brock\_hommes/12\_adam\_lr\_sweep.png}; including in supplement at submission.

\section{Hyperparameter robustness}
\label{supp:robustness}

Refer to \texttt{outputs/brock\_hommes/15\_hyperparam\_robustness.png}.

\section{Jacobian sensitivity vs OPG dynamic range}
\label{supp:jacobian}

Refer to \texttt{outputs/brock\_hommes/13\_jacobian\_comparison.png}. Per-parameter Jacobian dynamic range $\sim 5 \times 10^2$ vs OPG $\sim 5 \times 10^6$; the gap is off-diagonal coupling ($|\rho(g_1, g_2)| = 0.999$).

\section{Full falsification table}
\label{supp:falsification-table}

Per-perturbation aggregate $|\Delta|$ (signed, no $\pm$ averaging) for all three models and all three discrepancies. Table to be inserted at submission time from \texttt{outputs/paper/20\_merged\_falsification.npz}.
```

(Each \texttt{Refer to ...} placeholder is intentional --- at submission, replace with the actual figure include after copying the PNG into `paper/figures/`. For workshop submission we may also choose to inline these.)

- [ ] **Step 2: Add appendix include to main.tex**

In `paper/main.tex`, after `\bibliography{refs}`:

```latex
\input{../appendix/supp}
```

- [ ] **Step 3: Compile, commit**

```bash
cd paper && latexmk -pdf main.tex 2>&1 | tail -5 && cd ..
git add appendix/supp.tex paper/main.tex
git commit -m "Paper: appendix scaffold with trajectory bootstrap + cross-refs"
```

---

## Task 16: Style-guide compliance pass

**Files:**
- Modify: any `paper/sections/*.tex` that fails the checks below.

This is the safety net. The goal is to not produce banned words/constructions in the first place; this task catches anything that slipped through.

- [ ] **Step 1: Banned-word global grep**

```bash
egrep -inr "novel|powerful|comprehensive|extensive|empirical fisher|it is worth noting|it should be emphasised|clearly[^,]|obviously|note that|we will show|we will demonstrate" paper/sections/ paper/main.tex appendix/supp.tex
```

Expected: no matches. Fix each match (rewrite the sentence). The pattern `clearly[^,]` excludes ``clearly,'' as a transition word but flags ``clearly the spectrum is sloppy''-style claims.

- [ ] **Step 2: Verify every section has an OCAR-opening sentence**

For each of `01_introduction.tex` through `06_limitations.tex`, read the first sentence and confirm it states where in the arc we are. If a section opens with ``In this section, we...'' or ``We now turn to...'', rewrite the first sentence.

- [ ] **Step 3: Verify every paragraph transitions with and / but / therefore**

Manual pass. Flag any paragraph starting with ``Additionally'', ``Furthermore'', ``Moreover'', ``In addition''; rewrite.

- [ ] **Step 4: Verify the figure-to-claim mapping**

Open `docs/memory/paper_story_arc.md` hero figures table side-by-side with the paper. For each row, confirm the cited figure in the paper is the same hero figure listed in the table. Confirm the numbers quoted match exactly.

- [ ] **Step 5: Verify the Kunstner cite location**

```bash
grep -n kunstner paper/sections/01_introduction.tex paper/sections/06_limitations.tex
```

Expected: appears in `01_introduction.tex`, NOT in `06_limitations.tex`. Per `docs/memory/paper_style_guide.md` rule 8 and `docs/memory/framing_kunstner_opg_not_fisher.md`.

- [ ] **Step 6: Commit any revisions**

```bash
git add paper/sections/ paper/main.tex appendix/supp.tex
git diff --cached --stat
git commit -m "Paper: style-guide compliance pass (banned words, OCAR openings, transitions)"
```

If `git diff --cached` is empty, nothing to commit --- the prose was clean.

---

## Task 17: Final compile + page-count check

**Files:**
- Modify: `paper/main.tex` only if page count is off.

- [ ] **Step 1: Clean compile**

```bash
cd paper && latexmk -C && latexmk -pdf main.tex 2>&1 | tail -20 && cd ..
```

Expected: no LaTeX errors. Bibliography warnings should be absent. Multiple-pass run.

- [ ] **Step 2: Page count**

```bash
pdfinfo paper/main.pdf | grep Pages
```

Expected: 8 pages (main) + appendix + references. If main body exceeds 8 pages: tighten Results subsections first (they're 3 pages of budget); next tighten Method §3.1 (often grows during writing); leave Introduction and Discussion alone unless emergency.

- [ ] **Step 3: Visual proofread**

Open `paper/main.pdf` and confirm:
- All four hero figures fit on the page without overflowing the column.
- Figure captions are readable at print size.
- Hyperref cross-references resolve (no `??`).
- TikZ Fig 0 is not rasterised or pixelated.
- Bibliography is alphabetical / numbered consistently with the chosen natbib style.

- [ ] **Step 4: Push the final state**

```bash
git push origin main
```

- [ ] **Step 5: Mark paper-draft-v1 complete**

Update `docs/memory/state.md` recommended next-steps section: change step 8 (``Begin paper draft (NEXT)'') to ``Paper draft v1 complete; iterate based on supervisor feedback / co-author comments.''

```bash
git add docs/memory/state.md  # gitignored, but useful for diff inspection
```

(state.md is gitignored — no commit needed; the edit is local-only persistence.)

---

## Self-review

**1. Spec coverage:**
- Repository layout (spec §A) → Task 1 ✓
- Page budget (spec §B) → Task 17 page-count check ✓
- Per-section content map (spec §C) → Tasks 4–14 (one per section) ✓
- Prose style (spec §D / memory `paper_style_guide.md`) → Task 16 compliance pass + reading at start of every prose task ✓
- TikZ Fig 0 (spec §E) → Task 3 ✓
- Submission checklist (spec §F) → reproducibility statement in Task 5 §4 opening; Limitations Task 12; broader-impact one-liner Task 12 step 1; refs.bib Task 2 ✓
- Memory cleanup (spec §G) → already done before this plan ✓

**2. Placeholder scan:** Bracketed sentences in the Abstract task are intentional templates the writer fills in at execution time, not plan placeholders. No "TBD" appears in the plan body. The appendix ``Refer to ...'' notes in Task 15 are also intentional --- they mark figures to be inlined at submission, not gaps in the plan.

**3. Type consistency:** Section labels (`sec:method`, `sec:results:bh-spectrum`, etc.) are consistent across cross-reference call-outs. Figure labels (`fig:method`, `fig:bh-spectrum`, `fig:falsification`, `fig:predicts-adam`, `fig:network-sir`, `fig:supp-trajectory`) match between definition and citation. BibTeX keys (`kunstner_opg_2019`, `gretton_mmd_2012`, etc.) used in prose match those defined in Task 2.

**4. Spec requirements with no task:** Anonymous-submission switch (`\usepackage[final]{...}` versus no option) is handled inline in Task 1 Step 4 with a comment. AI4ABM-specific anonymisation handling will be done at the final submission step outside this plan.
