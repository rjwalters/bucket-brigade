# Verdict — anvil_pub.bb-workshop.9

- **Total score**: **32 / 44**
- **Decision**: `advance: false` (32 < 35 threshold)
- **Rubric**: `anvil-pub-v2` (9 dimensions, /44, advance >= 35;
  on-disk `.anvil/skills/pub/rubric.md`).
- **Critical flags**: **none.**

## Critical flags

None. Specifically adjudicated and NOT raised:

- **Build / compile failure** — not raised. A full
  `pdflatex + bibtex + pdflatex x2` cycle against the shipped
  `anvil-paper.cls` completes with exit 0, a 26-page PDF matching the
  committed `main.pdf`, and zero unresolved references or citations in
  the final pass. (The render-gate pre-flight proper was skipped
  fail-open: no `paper.pdf` / `compile-log.txt` from a v9 `pub-audit`
  exists yet.)
- **Numerical inconsistency (text vs figure/table)** — not raised. The
  deterministic numeric-consistency pass (issue #462 sidecar at
  `anvil_pub.bb-workshop.9.numeric/`) found 0 findings over 1043 extracted
  numbers, and manual cross-checks of the section-4 separations
  (0.001/0.017/0.082 vs the 0.051/0.050/0.033/-0.049 class means) and
  the section-5 anchor ladder (248.67 < 302.87 < 307.83 < 386.60; the
  ~80/step and +4.96/step ~ 6% arithmetic) all reconcile internally.
  The **claim-vs-committed-artifact drift** on
  `recalibrated_verdict.json` is real but is a reproducibility/staleness
  defect (majors below), not an internal text-vs-table contradiction; a
  v9 `pub-audit` pass should re-adjudicate it with the artifact-side
  evidence.
- **Close prior work ignored** — not raised as a flag. The
  OpenSpiel/exploitability-evaluation omission (dim 4) is a positioning
  gap, but the paper's claim survives literally via its "parametric"
  and "cooperative-competitive" qualifiers, so it is not a
  misrepresentation of novelty at flag severity.

## Dimension summary

| # | Dimension | Weight | Score | v6 -> v9 move |
|---|---|---:|---:|---|
| 1 | Rigor of method / argument | 6 | 5 | unchanged |
| 2 | Evidence sufficiency | 6 | 5 | unchanged (new fragility noted) |
| 3 | Clarity of contribution | 5 | 4 | -1 (abstract bloat) |
| 4 | Related-work positioning | 5 | 3 | -1 (claim hardened; tradition still unengaged) |
| 5 | Reproducibility | 5 | 4 | -1 (committed-artifact drift) |
| 6 | Figure & table quality | 4 | 2 | -2 (caption factual error; Table 3 overflow) |
| 7 | Prose & structural quality | 4 | 2 | -2 (five severe overfull hboxes) |
| 8 | Citation hygiene | 5 | 5 | unchanged |
| 9 | Rhetorical economy | 4 | 2 | -1 (abstract/conclusion duplication grew) |
| | **Total** | **44** | **32** | **-8 vs v6's 40** |

Note the v6 -> v9 delta spans three unreviewed revisions (v7, v8, v9
were revised outside the anvil lifecycle; no v7/v8 review siblings
exist). The scientific content strengthened materially over those
revisions; the deductions are concentrated in presentation and
staleness defects those revisions introduced, all fixable in one pass.

## Venue overlay (advisory)

Advisory venue overlay scored **11/16** against `anvil-pub-neurips-v1`
(soundness 3/4, presentation 1/2, contribution 3/4, novelty 2/3,
reproducibility 2/3); see `_review.venue.json` for findings. The
overlay does NOT affect the /44 convergence gate.

## Top 3 revision priorities

1. **Reconcile the paper with the regenerated
   `recalibrated_verdict.json`** (commit a5b8ccdc, 2026-07-04). Either
   pin the paper's numbers to a frozen snapshot of the n=4 sweep
   (commit hash or a versioned copy committed alongside the paper) or
   regenerate Figure 2 and the section-4 class means from the merged
   n=20 data — and in either case re-derive the "robust to the metric
   choice" sentence, whose homogeneous-metric ordering (0.051 vs
   0.050) inverts (0.041 vs 0.046) on the current artifact.
2. **Fix the LaTeX margin overflows.** Five overfull hboxes of
   96.7-128.9pt (pages 4, 10, 11, 12 of the rendered PDF) push long
   `\texttt` artifact paths visibly into the right margin; break the
   paths, footnote them, or move them into table notes, then re-check
   the log.
3. **Correct the unsampled-cell description.** The six unsampled Nash
   cells are the kappa in {0.3, 0.7} columns of the c=0.5 panel (per
   `experiments/nash/phase_diagram/results.json`), not a
   "high-kappa x c=0.5 corner"; fix the phrase in the section-3 body,
   the Figure 1 caption, and the Table 1 caption ("row totals are 9
   except at kappa=0.7" — the kappa=0.3 row also has 6 cells).

Close behind: engage (or qualify against) the OpenSpiel /
exploitability-evaluation tradition for the "only published" claim,
and compress the abstract to ~250 words.
