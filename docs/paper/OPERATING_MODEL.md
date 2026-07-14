---
title: Operating Model — two agents + gatekeeper
status: active
last_updated: 2026-07-12
---

# Operating Model

How this project is run. Two agents and a human, two memory layers, one working
cycle. Read this first in any strategic-lead session; the implementation
specialist's slice of it is mirrored in the repo-root `CLAUDE.md`.

## Roles

**Implementation specialist** (the coding agent — Claude Code, driven by
`CLAUDE.md`). Maintains and develops the code, runs experiments, manages logs and
provenance. Executes **bounded** tasks, reports results, preserves reproducibility.

It does **not** independently: broaden the research question; rewrite the central
mathematical object; promote a claim into the paper; determine the paper narrative;
or draft the manuscript from chronological logs. It updates **Layer 1 only** (see
below).

**Scientific lead & editor** (a strategic conversation — this is its charter).
Runs the arc: scientific question → experiment design → hand a bounded task to the
implementation specialist → interpret the returned result → decide the scientific
consequence → maintain the paper argument → eventually write the manuscript. Owns
**Layer 2**.

**You (Pietro)** — the gatekeeper between them. You design or approve each
experiment, carry results from the coding agent into the strategic conversation,
decide what the results mean and which claims survive, and admit accepted
conclusions into Layer 2.

## Two memory layers

The paper is written from **Layer 2**, never from the chronological history of
corrections. Layer 2 is a curated *projection* of Layer 1 — it selects, reorders,
and rewords Layer-1 facts by ID; it never invents a fact.

### Layer 1 — internal research memory (`docs/` root + subfolders)

Everything needed to understand how decisions were reached. May contain
corrections, disagreements, false starts, and negative results.

| File | Holds |
|---|---|
| `docs/MATH.md` | authoritative mathematical specification |
| `docs/CLAIMS.md` | the one claims ledger — every claim, all statuses |
| `docs/DECISIONS.md` | decisions + rejected arguments (DEC-xxx) |
| `docs/experiments/EXP-*.md` | experiment registry + results + concise reviews |
| `docs/papers/` | literature notes |
| `docs/STATUS.md` | **bridge**: current factual state (phase · done · next) |

### Layer 2 — paper-facing memory (`docs/paper/`)

Only the accepted scientific state. No chronological repair story.

| File | Holds |
|---|---|
| `STRATEGIC_HANDOFF.md` | fresh-session entrypoint: thesis, interpretation, paper direction |
| `PAPER_ARCHITECTURE.md` | manuscript outline (claim–evidence, not prose) |
| `WRITING_BRIEF.md` | accepted narrative · terminology · limitations · style |
| `EVIDENCE_MAP.md` | claim → evidence → figure → section (the supported-claims view) |
| `FIGURE_LEDGER.md` | approved figures + provenance |
| `OPERATING_MODEL.md` | this file |

## The two definitions of done

Each agent has its own DoD, so the layers stay in sync without extra bookkeeping.

**Implementation specialist — per experiment (Layer 1, "the three places"):**

1. `docs/experiments/EXP-xxx.md` — fill in the result;
2. `docs/CLAIMS.md` — set the affected claim's status + one-line evidence;
3. `docs/STATUS.md` — advance the phase / set the next experiment.

Nothing in `docs/paper/`. No narrative, no claim promotion, no figure approval.

**Scientific lead — per _accepted_ result (Layer 2):**

At interpretation time (cycle step 6), promote the claim's paper wording in
`EVIDENCE_MAP.md`, approve the figure in `FIGURE_LEDGER.md`, advance
`STRATEGIC_HANDOFF.md`; occasionally touch `PAPER_ARCHITECTURE.md` / `WRITING_BRIEF.md`.

## The normal working cycle

```text
1. We define the scientific question.            (scientific lead + you)
2. You produce the bounded implementation prompt: (you)
   math objects · controls · metrics · pass criteria · scope · required output.
3. Coding agent implements & runs it; returns:    (implementation specialist)
   commit · run path · principal metrics · deviations · concise review.
   → updates Layer 1 (the three places).
4. You bring the result into the strategic conversation.   (you)
5. We interpret: what was observed, what it supports,      (scientific lead + you)
   what it does not establish, whether paper direction changes, what comes next.
6. Accepted conclusions enter Layer 2.                     (scientific lead)
```

Most experiments should not need several rounds of process discussion.

## Manuscript threshold

Do not begin the full rewrite until the project has: controlled mathematical
validation; controlled MMD estimator validation; a genuine Brock–Hommes result;
smooth SIR posterior/profile validation; at least one policy-functional or
observation-design result; and stable approved figures. Until then, Layer 2
accumulates evidence — it is a living scaffold, not a draft. (Detailed gate:
`WRITING_BRIEF.md`.)

## Fresh manuscript-session input

When the threshold is met, a manuscript session receives: `docs/MATH.md`,
`docs/CLAIMS.md`, the `docs/paper/` set, verified literature notes, and approved
figures with captions. It does **not** receive old manuscript drafts, weekly logs,
rejected derivations, coding-agent conversations, or archived OPG arguments.
