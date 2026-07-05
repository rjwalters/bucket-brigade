# Line-level comments — anvil_pub.bb-workshop.11

Grouped by severity: **blocker** / **major** / **minor** / **nit**.
Line anchors refer to `main.tex`.

## Blockers

**None.**

## Majors

**None.** The v10 major (transposed Figure 1 caption clause) is fixed
at L565-L566 and verified against the rendered figure, the generating
script, and the caption's own internal clauses.

## Minors

### §5 footnote (L1105-L1111) — non-regenerable provenance, carried

"the $300/300$ RNG/resample-combination sweep and the $t$-interval
trace to the PR~\#460 judge recomputation" — the honest-disclosure
footnote is retained verbatim from v10 and remains the one statistic
that does not trace to a regenerable `experiments/` artifact.
Camera-ready operator task (regenerate as a committed script + JSON),
explicitly carried in the v11 changelog; re-recorded here so it stays
on the audit's radar.

### §7 Reproducibility (L1359-L1364) — HuggingFace baselines still in progress

"the publication pathway for the baselines is in progress and not yet
complete at the time of this draft" remains a true status statement,
but the NeurIPS checklist expects it resolved by submission. Operator
task, carried from v9/v10.

### Dims 1–2 (§4) — cross-class significance sweep remains the scoring gate

The class-ordering headline is still ordering-only ("does not reject
the null of equal class means under a standard test", L652-L654); the
cross-class $4\times$-budget sweep named in §4 and §7 is the
camera-ready experiment that would lift dims 1–2 off 5/6. Declined for
this polish revision per the operator instruction — correctly so — but
it is the highest-leverage remaining scientific item.

## Nits

- **Table 2 caption path breaks** (L949-L980): the `xurl` break of
  caption-resident artifact paths without hyphens stands, per the
  documented decline (acceptable cost of the 0-overfull build).
  Revisit only if a downstream reader misparses.
- **28 rendered pages**: the §1 footnote (L203-L215) now answers the
  page-budget policy in-paper; the remaining deliverable is the actual
  4-page compressed body at submission time (operator task, not a
  defect of this artifact).
- **§6 climbing/penalty-game attribution** (L1235-L1239)
  [related-work]: "the climbing/penalty-game
  tradition~\citep{christianos2022pareto}" cites the tradition's
  modern representative because the primary (Claus & Boutilier 1998)
  has no resolvable identifier per the litsearch write contract — a
  correct and documented decision, but a reader may misattribute the
  tradition's origin to the 2022 paper. If the author hand-enters a
  verified BibTeX entry for Claus & Boutilier into `<thread>/refs/`
  before camera-ready, an "e.g.," or a direct primary citation would
  be cleaner. Lead for a `pub-litsearch` re-run with author-supplied
  BibTeX; no `.bib` entry is added by this review.
- **Related-work lead (from reviewer recall, NOT a live search)**
  [related-work]: the statistical-mechanics-of-games line — Galla &
  Farmer, "Complex dynamics in learning complicated games," PNAS
  110(4):1232–1236 (2013), DOI 10.1073/pnas.1109672110 as recalled —
  studies parametric families of random two-player games with a
  phase diagram of *learning dynamics* over the parameter space. It
  is not a MARL benchmark and supplies no per-cell NE ground truth
  for trainability scoring, so the paper's scoped claim ("the only
  published parametric cooperative-competitive MARL family in
  which... a phase diagram supplies the ground truth that a
  trainability sweep can then be scored against") survives it; but a
  one-clause acknowledgment of the phase-diagram-over-game-families
  precedent could preempt a reviewer raising it. The BRIEF sets
  `web_search: true`, but network access was unavailable in this
  review environment, so this is a training-data-recall lead —
  recommend a `pub-litsearch` re-run to resolver-verify before any
  citation is added. No `.bib` entry is added by this review.
