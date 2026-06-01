# Paper design — AI4ABM 2026 workshop submission

Date: 2026-06-01
Status: approved by user 2026-06-01

This is the binding design for the LaTeX write-up that follows. It is committed because subsequent drafting decisions will refer back to it.

Target venue: **NeurIPS 2026 Workshop on AI for Agent-Based Modelling (AI4ABM)**.
Template: NeurIPS 2024 workshop style (`neurips_2024.sty`), 8 pages main + unlimited references + appendix.

The story arc, hero figures, and scope rules are already locked in `docs/memory/paper_story_arc.md`. This spec adds: section structure, per-section content map, prose-style rules, LaTeX setup, and a TikZ method diagram.

## A. Repository layout

```
paper/
  main.tex                       — preamble + \input each section
  sections/
    00_abstract.tex
    01_introduction.tex
    02_related.tex
    03_method.tex
    04_results.tex
    05_discussion.tex
    06_limitations.tex
  diagrams/
    fig0_method.tex              — TikZ method-overview schematic
  figures/                       — symlinks to ../outputs/paper/figures/
  refs.bib
  neurips_2024.sty               — fetched at setup time
appendix/
  supp.tex                       — supplementary figures + tables
```

LaTeX preamble:
- `\documentclass{article}` + `\usepackage[final]{neurips_2024}` (the `final` option enables author block; switch to no option for double-blind if AI4ABM confirms anonymous review).
- Packages: `amsmath`, `amssymb`, `graphicx`, `booktabs`, `natbib` (style `plainnat`), `hyperref`, `microtype`, `tikz` (libraries `positioning`, `arrows.meta`, `calc`, `shapes.geometric`), `xcolor`.
- Bibliography: hand-curated `refs.bib`. Required cites listed in §F below.

## B. Page budget

8 pages main + refs + appendix. Allocation:

| Section | Pages | OCAR role |
|---|---|---|
| Abstract (~150 words) | 0.4 | Compressed full arc |
| 1. Introduction | 1.0 | **O** (Opening) + **C** (Challenge) |
| 2. Related work | 0.5 | Positioning vs four lineages |
| 3. Method | 1.5 | **A** setup (incl. TikZ Fig 0) |
| 4. Results | 3.0 | **A** evidence (four hero figures inline) |
| 5. Discussion | 0.75 | **R** (Resolution) |
| 6. Limitations | 0.35 | Scope statements |
| **Total** | **7.5** | + ~0.5 safety margin |

## C. Per-section content map

Each row names the claim and the figure / numbers that support it. Anything that doesn't fit a row gets cut or pushed to the appendix.

| Section | Claim | Figure / numbers |
|---|---|---|
| Abstract | Three sentences (Opening, Challenge, Action+Resolution). Numbers to include: BH 489× falsification, network-SIR 36k×, Adam v_P / OPG v_1 = 9.4×10⁴. | none |
| 1. Intro hook | Differentiable ABMs make calibration tractable; Adam silently fails | concrete one-paragraph Adam-fails example |
| 1. Intro pivot | The failure is *predictable* from the per-seed gradients already computed | forward-ref Fig 3 |
| 1. Intro contributions | Three numbered bullets: (i) live OPG identifiability diagnostic on differentiable ABMs, (ii) §5.4 non-MMD falsification protocol, (iii) two-model + surrogate-regime verification | — |
| 2. Related | 4-lineage gap table (Naumann-Woleske, Quera-Bofarull, Gutenkunst/Transtrum, Kunstner) | mini-table (3 cols × 4 rows) |
| 3.1 Method: setup | Differentiable ABM + MMD + per-seed VJP | **Fig 0 (TikZ schematic)** |
| 3.2 Method: OPG + GGN | F̂ definition, residual-GGN interpretation, Kunstner contributory cite | equation block |
| 3.3 Method: §5.4 falsification | Equal-α perturbation along v₁ vs v_P under three discrepancies the calibrator never saw | equation block, no figure |
| 4.1 Results: sloppy spectrum (BH) | 8.4-OOM spectrum; β alone is v_P; b₁+b₂ stiff combo | **Fig 1** |
| 4.2 Results: live diagnostic | Eigenstructure stable from t=0; ⟨v₁(θ_t), v₁(θ\*)⟩ ≤ 10° throughout, v_P ≤ 20° at init dropping to 3° | inline reference to **Fig 5 (supp)** |
| 4.3 Results: non-circular validation | Three models × three channels: ratios ≥ 10² everywhere | **Fig 2** |
| 4.4 Results: diagnostic predicts Adam | Recovery error decomposition along F̂ eigenbasis | **Fig 3** |
| 4.5 Results: surviving Gumbel surrogate | Network-SIR eigenvectors match MF qualitatively; falsification ratios 10⁴–10⁵× | **Fig 4** |
| 5. Discussion | 3 paragraphs: (i) diagnostic > preconditioner, (ii) domain payoff (b₁+b₂; v_P public-health degeneracy), (iii) surrogate-bias survival | — |
| 6. Limitations | (i) horizon-bias yellow light, (ii) N modest (15+10) with LCB caveat, (iii) Pearson normalisation deferred, (iv) optimiser claims narrow | — |

## D. Prose style guide

The paper is a hybrid of four influences. Each plays a specific role; none dominates. The full prose-style memory lives in `docs/memory/paper_style_guide.md` and is binding.

- **Doyne Farmer** (ABM-for-systemic-risk macroscopic framing): 2–3 sentences max in Intro opening and Discussion broadening. Forces the reader to care about the stakes without rhetorical excess.
- **Michael Wooldridge** (definitional clarity, MAS-textbook prose): drives §3.1 Method setup. Crisp definitions, no jargon-creep.
- **Arnau Quera-Bofarull** (differentiable-ABM 2025): integrated method-and-experiment style. Sets the §3↔§4 transitions: never separate "we implemented X" from "and here is what X reveals."
- **Philipp Andelfinger** (simulation-engineering): default Results voice. Each subsection states a claim, supports with one figure, moves on. Used in §2 Related work and §4 Results.

Mechanical rules carried throughout (after Schimel, *Writing Science*):
1. Every section opens with one OCAR sentence stating where in the arc we are.
2. **And/but/therefore** beats "and/and/and" — every paragraph transitions with one of those three moves.
3. No paragraph longer than 5 sentences in the main text.
4. Active voice: "We compute F̂" not "F̂ is computed".
5. Numbers are **bold on first mention** if they are headline (e.g., **20.7 OOM**, **94 000×**).
6. Banned adjectives: "novel", "powerful", "comprehensive", "extensive". They signal review-bait.
7. Kunstner is cited in the *introduction* (not Limitations), contributorily.
8. Lead each Results subsection with the claim sentence; figure call-out follows.

## E. TikZ Fig 0 — Method overview schematic

Half-column, 3.0 in × 2.0 in. Black-and-white with light-grey fills, NeurIPS-standard line weights.

Logical flow (boxes left-to-right, then wrap):
```
[ABM simulator θ→x] ──vmap, M seeds──▶ [trajectories X∈ℝ^{M×T}]
                                              │
                                              ▼ MMD²(X, Y_ref)
[scalar loss L]                       [unbiased squared MMD]
       │
       ▼ per-seed VJP
[per-seed grads g_m∈ℝ^P] ────────────▶ [F̂ = (1/M) Σ g_m g_mᵀ]
                                              │
                                              ▼ eigh
                                      [stiff v_1 … sloppy v_P]
```

Arrow labels: the JAX operation that realises each transition (`vmap`, `vjp`, `eigh`). The schematic exists to make the "F̂ is a free byproduct of the gradients you already compute" claim legible at a glance.

## F. AI4ABM/NeurIPS submission checklist

- Author block — anonymous if AI4ABM 2026 confirms double-blind. Default assumption: yes, anonymous; switch to non-`final` neurips_2024.sty.
- Reproducibility statement appearing near §4 opening with code-release URL. If anonymous, use a GitHub-anonymous proxy at submission time.
- NeurIPS-style **Limitations** subsection in main text (not appendix).
- Brief **Broader impact** paragraph at end of §6 or in appendix: identifying parameter combinations the data cannot constrain is a *safety property*, flagging overconfident calibrations rather than masking them.
- Required `refs.bib` entries: Brock & Hommes (1997, 1998), Kermack & McKendrick (1927), Kunstner, Hennig & Balles (NeurIPS 2019), Naumann-Woleske et al. (2024 Mark-0 paper), Quera-Bofarull et al. (2023, 2025), Gutenkunst et al. (PLOS Comp Bio 2007), Transtrum, Machta & Sethna (2010, 2011), Schraudolph (2002), Martens & Grosse (2015), Maddison et al. (Concrete distribution, ICLR 2017), Jang et al. (Gumbel-Softmax, ICLR 2017), Bouchaud-Mark-0 (Naumann-Woleske 2024), Schimel (*Writing Science*, optional in acknowledgments).
- Appendix: live-diagnostic trajectory figure with full IQR bands; Adam-lr sweep (Fig 12); hyperparameter robustness (Fig 15); Jacobian-vs-OPG comparison (Fig 13); full falsification table with all four signed perturbations (not just ±averaged).

## G. Memory cleanup actions taken with this spec

1. `MEMORY.md` index gets a new entry pointing to `paper_style_guide.md` under "How to write the paper".
2. `paper_story_arc.md` cross-references checked (`[[scripts-19]]` etc. — these don't resolve to slugs but mark intent, fine per convention).
3. `state.md` recommended next steps: "draft paper" promoted to step 5; scripts 19/20/21/22/23 marked done.
4. New `paper_style_guide.md` memory created with the §D rules above (this spec is the design *context*; the memory is the *binding rule* for drafting).

## H. Open questions (non-blocking)

- **Anonymous submission?** AI4ABM has been double-blind in past editions. Confirm before submission; controls `neurips_2024.sty` option.
- **Author / affiliations** — Pietro Bicocchi + supervisors (see `project_overview.md`). To be confirmed before final.

## I. Next step

After this spec is approved, invoke the `writing-plans` skill to produce a step-by-step implementation plan for the LaTeX write-up: which sections in what order, when to insert figures, when to compile, what counts as "done" for each.
