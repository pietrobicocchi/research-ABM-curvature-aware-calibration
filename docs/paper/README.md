# docs/paper/ — Layer 2, paper-facing memory

The accepted scientific state, from which the manuscript is written. This layer is
a curated **projection** of Layer 1 (`docs/` root): it selects, reorders, and
rewords Layer-1 facts by ID (`C0x`, `EXP-00x`, commit) — it never invents a fact,
never carries the chronological story of corrections.

**Owned by** the scientific lead + Pietro. The implementation specialist (coding
agent) never edits this folder. See `OPERATING_MODEL.md` for roles, layers, and the
working cycle.

## Fresh strategic-session reading order

1. `OPERATING_MODEL.md` — two agents + gatekeeper, the two layers, the cycle
2. `../STATUS.md` — current factual state (phase · done · next) [Layer 1 bridge]
3. `STRATEGIC_HANDOFF.md` — interpretation + paper direction (start here each session)
4. `../MATH.md` — authoritative mathematical specification [Layer 1]
5. `../CLAIMS.md` — the claims ledger, all statuses [Layer 1]
6. `PAPER_ARCHITECTURE.md` — manuscript outline
7. `EVIDENCE_MAP.md` — claim → evidence → figure → section
8. `FIGURE_LEDGER.md` — approved figures + provenance
9. `WRITING_BRIEF.md` — accepted narrative, terminology, limitations, style
10. latest `../experiments/EXP-*.md` review

## Files

| File | Role | Updated |
|---|---|---|
| `OPERATING_MODEL.md` | roles, layers, working cycle | rarely |
| `STRATEGIC_HANDOFF.md` | fresh-session entrypoint: thesis, interpretation, next | per accepted result |
| `PAPER_ARCHITECTURE.md` | claim–evidence manuscript outline | on structural change |
| `WRITING_BRIEF.md` | narrative, terminology, limitations, style | per accepted result |
| `EVIDENCE_MAP.md` | claim → evidence → figure → section | per accepted result |
| `FIGURE_LEDGER.md` | approved figures + provenance | per approved figure |

## Discipline

- **Never invent facts here.** Every statement traces to a Layer-1 claim ID +
  evidence (experiment · commit). Promotion Layer 1 → Layer 2 is the gatekeeper's act.
- **Single claims ledger:** `../CLAIMS.md`. This folder holds no second claim list —
  `EVIDENCE_MAP.md` is the supported-claims *view*, by reference.
- **No manuscript yet:** the rewrite waits for the threshold in `WRITING_BRIEF.md`.
