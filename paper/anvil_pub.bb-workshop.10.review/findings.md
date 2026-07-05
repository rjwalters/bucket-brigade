# Findings — anvil_pub.bb-workshop.10

Cross-section observations from the v10 review pass. The per-dimension
scorecard lives in `scoring.md`; line-level items in `comments.md`.

## Artifact verification log (recompute-at-HEAD, not changelog trust)

The v10 changelog claims every §4 statistic was recomputed from the
artifacts at repository HEAD. This review re-derived them
independently rather than accepting the claim:

- **Class means** from
  `experiments/p3_specialization/phase_diagram_ppo_v2/recalibrated_verdict.json`
  (37 cells; 8 cells at n_seeds=20, matching the named buy-down set):
  `gap_closed_ne` symmetric 0.1416 (n=11), mixed 0.0964 (n=9),
  asymmetric 0.0602 (n=11); homogeneous fallback for no_convergence
  -0.0243 (n=6). Paper reports 0.142/0.096/0.060/-0.024. Match.
- **Homogeneous class means** 0.0406/0.0456/0.0298/-0.0243 — the
  symmetric/mixed inversion (0.041 < 0.046) behind the withdrawn
  robustness sentence is real at HEAD. The withdrawal paragraph in §4
  Caveats states the inversion and the superseded 0.0005 margin
  accurately.
- **Adjacent separations** 0.045/0.036; per-class mean within-cell std
  0.399/0.198/0.422 (paper: "0.20–0.42"); per-cell std max 0.993,
  median 0.248; beta-column example at kappa=0.1, c=1.0:
  -0.317/+0.010/-0.001; max within-column range 0.338 ("up to 0.34").
  All match.
- **Entropy artifact**
  (`experiments/nash/phase_diagram/entropy_vs_trainability.json`):
  n=31 joined cells, mean-vs-ne rho=0.1088 (p=0.560); the single
  nominally significant entry is min-vs-homogeneous p=0.0384; eight
  tests vs Bonferroni alpha=0.00625 as stated. The pinned provenance
  checks out: `git show 22b1fda6:.../entropy_vs_trainability.json`
  yields rho=0.00693 (p=0.970) — the paper's "rho=0.007 (p=0.97),
  preserved at artifact revision 22b1fda6" is exactly right.
- **k\* join**
  (`experiments/nash/phase_diagram/kstar_vs_trainability.json`):
  primary rank-biserial 0.0, exact one-sided p=0.5394, permutation
  delta-mean -0.00184 (two-sided p=0.9939, 165 assignments, floor
  0.00606); homogeneous split rank-biserial 0.40, p=0.185; post-hoc
  failure zone rank-biserial -1.0, exact p=0.0035, permutation
  two-sided p=0.0210; Spearman cell-level -0.556 (p=0.000354),
  column-level -0.635 (p=0.0196). All match the §4 text.
- **4x-budget sweep**: mean of `gap_closed_homogeneous_mean` over the
  six `phase_diagram_ppo_longbudget` cell summaries = -0.1079; the
  paper's "-0.108 (from -0.024)" is correct at HEAD (v9 had said
  "from -0.049").
- **rest_trap anchors** (`scripted_battery/rest_trap.md`, frozen NE
  JSON, tier-1 verdicts, 16x cell summary): 2984.04/ep -> <=248.67;
  288.55 [285.20, 291.65]; 302.87 point, 302.94 [301.46, 304.31] at
  n=10k; 306.26 [302.95, 309.33]; 307.83 [305.00, 310.71] with the
  16x `escaped_trap` verdict reason string matching the paper's
  reading; 386.60 [386.17, 387.03]; +83.67 [+82.36, +84.89]. All
  match.
- **Nash grid** (`experiments/nash/phase_diagram/results.json`):
  verdict counts 12/11/6/10; the six unsampled cells are exactly
  kappa in {0.3, 0.7} x beta in {0.1, 0.5, 0.9} at c=0.5; Table 1's
  per-kappa empirical distribution matches row for row; the lone
  splitting row is (kappa=0.9, c=0.5) with beta=0.1 mixed vs
  asymmetric_only at beta in {0.5, 0.9}; payoffs 80.915 vs 72.0095.
  CF-2's fix is correct at all three sites.
- **Figure 2**: re-rendered in a sandbox from the preserved
  `figures/src/recalibrated_heatmap.py` against the HEAD JSON; the
  output's extracted text layer is byte-identical to the shipped
  `figures/recalibrated_heatmap.pdf` (the superseded +0.37 value is
  absent; the kappa=0.3, c=1.0, beta=0.5 cell reads +0.04).
- **Build**: independent `pdflatex + bibtex + pdflatex x2` -> exit 0,
  27 pages, 0 overfull hboxes, 0 undefined references/citations.
- **Numeric pre-check** (issue #462): 0 findings over 1053 extracted
  numbers; sidecar written at `anvil_pub.bb-workshop.10.numeric/`.

## Cross-section observations

1. **The revision discipline is the story of this version.** v10 chose
   the harder CF-1 resolution (recompute at HEAD rather than pin the
   superseded revision) and executed it completely — including
   withdrawing a previously-claimed robustness result whose ordering
   inverted on the fresh data, and saying so in the body rather than
   silently dropping the sentence. That is exactly the honesty norm
   the BRIEF asks for.
2. **One new defect was introduced by a fix.** The Figure 1 caption's
   new panel-layout half-sentence is transposed (rows/columns swapped;
   see `comments.md` major). Figure 1 and Figure 2 use opposite
   layouts, which is a latent trap for future caption edits — consider
   normalizing the two figures to one orientation at camera-ready.
3. **The remaining score headroom is structural, not editorial.**
   Dims 1 and 2 are held at 5/6 by the ordering-only power of the §4
   headline — fixable only by the named camera-ready cross-class
   budget sweep, not by prose. Dims 4, 5, 9 are held by items already
   tracked (litsearch verification, HF hosting, conclusion length).
4. **Render-gate ordering note**: this review ran before any v10
   `pub-audit`, so the render-gate pre-flight was skipped fail-open
   (no `paper.pdf` / `compile-log.txt`); the independent build above
   substitutes for it evidentially but a `pub-audit` pass should still
   run next to produce the canonical compile log and the per-citation
   claim-support audit (openspiel2019/psro2017 are new and unverified
   on disk).

## Rubric version transition

Not applicable — the prior review sibling
(`anvil_pub.bb-workshop.9.review/`) was scored against the same
`anvil-pub-v2` rubric (/44, >=35) as this review, per its
`_meta.json.rubric_id`. (Transition subsection omitted per the
steady-state rule; this stub records only that the check was
performed.)
