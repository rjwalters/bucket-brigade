# Verdict — anvil_pub.bb-workshop.10

- **Total score**: **38 / 44**
- **Decision**: `advance: true` (38 >= 35 threshold; no critical flags)
- **Rubric**: `anvil-pub-v2` (9 dimensions, /44, advance >= 35;
  on-disk `.anvil/skills/pub/rubric.md`).
- **Critical flags**: **none.**

## Critical flags

None. Specifically adjudicated and NOT raised:

- **Numerical inconsistency / artifact drift (the v9-audit CF-1
  class)** — not raised. This review recomputed every headline
  statistic from the committed artifacts at repository HEAD rather than
  trusting the changelog: the §4 class means (0.142/0.096/0.060 on
  `gap_closed_ne`; -0.024 homogeneous fallback), the homogeneous class
  means and their symmetric/mixed inversion (0.041 < 0.046) behind the
  withdrawn robustness sentence, the adjacent separations 0.045/0.036,
  the beta-column example (-0.32/+0.01/-0.00) and 0.34 max range, the
  per-cell std 0.99/median 0.24, the entropy headline rho=0.109
  (p=0.56) at HEAD with rho=0.007 (p=0.97) verified at the pinned
  artifact revision `22b1fda6`, the min-vs-homogeneous p=0.038 entry,
  the full k* battery (rank-biserial 0.00 / p=0.539 / permutation
  -0.002, p=0.994; homogeneous 0.40 / p=0.19; failure zone -1.00 /
  p=0.0035 / p=0.021; Spearman -0.556, p=0.0004 and -0.635, p=0.020),
  the 4x-budget mean -0.108 recomputed from the six longbudget cell
  summaries, and the complete rest_trap anchor ladder
  (248.67 / 288.55 / 302.87 / 302.94 [301.46, 304.31] / 306.26
  [302.95, 309.33] / 307.83 [305.00, 310.71] / 386.60 [386.17, 387.03];
  +83.67 [+82.36, +84.89]) all reconcile with the on-disk artifacts.
  Figure 2 was independently re-rendered from
  `figures/src/recalibrated_heatmap.py` against the HEAD JSON and its
  text layer is identical to the shipped PDF (the superseded 0.37 value
  is gone; the spot-checked cell reads +0.04). The deterministic
  numeric-consistency pre-check (`anvil_pub.bb-workshop.10.numeric/`)
  found 0 findings over 1053 extracted numbers.
- **Build / compile failure** — not raised. This review's independent
  `pdflatex + bibtex + pdflatex x2` build against the shipped
  `anvil-paper.cls` exits 0 at 27 pages with zero overfull hboxes of
  any size (v9: 10, five severe) and zero unresolved
  references/citations. (The render-gate pre-flight proper was skipped
  fail-open: no `paper.pdf` / `compile-log.txt` from a v10 `pub-audit`
  exists yet, so no `_gate.json` is emitted.)
- **Caption/figure contradiction (the v9-audit CF-2 class)** —
  considered for the NEW Figure 1 defect and not raised at flag
  severity. The half-sentence added to Figure 1's caption ("within
  each panel rows are $\kappa$ and columns are $\beta$") is
  transposed: the rendered figure and its generating script have rows
  = beta and columns = kappa. This is a real factual error in the
  headline figure's caption, but the figure's own axis labels are
  correct, the caption's own later "$\kappa\in\{0.3,0.7\}$ columns"
  clause and the §3 body text are correct, and no number or verdict is
  affected — a sophisticated reader is annoyed, not misled. Scored as
  a major finding under dim 6 (see `comments.md`), not a flag.
- **Close prior work ignored** — not raised, and the v9 positioning
  gap is resolved: §6 now engages OpenSpiel and PSRO/NashConv
  explicitly and cedes the "first environment where equilibrium
  convergence is measurable" credit to that tradition, scoping the
  paper's claim to the parametric/phase-diagram property.

## Verification of the v9 review's top-3 priorities

1. **Reconcile with the regenerated `recalibrated_verdict.json`** —
   done, via the stronger option (recompute at HEAD, not pin). Every
   affected number verified by independent recomputation; the
   metric-robustness sentence is withdrawn with the inversion stated
   explicitly; the superseded rho=0.007 is retained only as pinned
   provenance.
2. **Fix the LaTeX margin overflows** — done and verified by
   independent build: 0 overfull hboxes (target was 0 > 10pt).
3. **Correct the unsampled-cell description** — done at all three
   sites and verified against `experiments/nash/phase_diagram/results.json`
   (the six unsampled cells are the kappa in {0.3, 0.7} columns of the
   c=0.5 panel; Table 1 caption now matches its body, row totals
   9/6/9/6/9).

Close-behind items from v9 also landed: OpenSpiel/PSRO engagement
(dim 4 3->4) and abstract compression ~700 -> ~325 words (dim 3 4->5,
dim 9 2->3).

## Dimension summary

| # | Dimension | Weight | Score | v9 -> v10 move |
|---|---|---:|---:|---|
| 1 | Rigor of method / argument | 6 | 5 | unchanged (structural power limit) |
| 2 | Evidence sufficiency | 6 | 5 | unchanged (staleness fixed; ordering-only headline remains) |
| 3 | Clarity of contribution | 5 | 5 | +1 (abstract compressed; extraction restored) |
| 4 | Related-work positioning | 5 | 4 | +1 (exploitability tradition engaged; thin) |
| 5 | Reproducibility | 5 | 4 | unchanged score, defect swapped (drift fixed; HF hosting + NC-5 provenance remain) |
| 6 | Figure & table quality | 4 | 3 | +1 (CF-2 sites fixed; NEW transposed-axes clause in Fig. 1 caption) |
| 7 | Prose & structural quality | 4 | 4 | +2 (0 overfull hboxes, verified by independent build) |
| 8 | Citation hygiene | 5 | 5 | unchanged |
| 9 | Rhetorical economy | 4 | 3 | +1 (abstract deduplicated; conclusion still restates) |
| | **Total** | **44** | **38** | **+6 vs v9's 32** |

## Venue overlay (advisory)

Advisory venue overlay scored **12/16** against `anvil-pub-neurips-v1`
(soundness 3/4, presentation 1/2, contribution 3/4, novelty 3/3,
reproducibility 2/3); see `_review.venue.json` for findings. The
overlay does NOT affect the /44 convergence gate.

## Top revision priorities (advisory — paper advances)

The paper advances to READY; `pub-audit` should run next. The
highest-leverage remaining items for the next revision or the
camera-ready:

1. **Fix the transposed Figure 1 caption clause** — "rows are
   $\kappa$ and columns are $\beta$" should read "rows are $\beta$ and
   columns are $\kappa$" (one-line fix; the only new defect this
   revision introduced).
2. **Re-run `pub-litsearch`** to resolver-verify `openspiel2019` and
   `psro2017` (and `ppo2017`/`mappo2022`), source the Appendix A
   named-template citations, and check for additional close work in
   the exploitability-evaluation line.
3. **Tighten the conclusion** — it still restates the abstract's arc
   at comparable coverage; and the submission-time page-budget answer
   (27 pages vs. the 4-page workshop body target) is still owed.
