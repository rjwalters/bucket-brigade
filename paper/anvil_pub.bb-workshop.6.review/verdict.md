# Verdict — anvil_pub.bb-workshop.6

- **Total score**: **40 / 44**
- **Decision**: `advance: true` (40 ≥ 35 threshold; no critical flags)
- **Rubric**: `anvil-pub-v2` (9 dimensions, /44, advance ≥35; on-disk
  `.anvil/skills/pub/rubric.md`).

## Critical flags

**None.**

All 5 v6 changes land cleanly. No issue rises to "a sophisticated PC
member would stop reading." Specifically scrutinized:

- **v5 R1 (Figure 1 raw payoffs)** is **resolved**. The new
  `figures/phase_diagram.pdf` is a 3-panel categorical heatmap (one
  panel per c ∈ {0.5, 1.0, 2.0}), colored by NE verdict, with no
  inline payoff numerics. The six unsampled cells in the
  high-κ × c=0.5 corner render as hatched grey blocks labelled
  "n/a". All four regimes (no_convergence, symmetric_only, mixed,
  asymmetric_only) are visible in the plotted body so the legend
  swatches all have a corresponding cell. Source script
  `figures/src/phase_diagram.py` ships alongside (matches Figure 2's
  pattern).
- **v5 R2 (Table 2 margin overflow)** is **resolved**. Page 7 of the
  rendered PDF shows the "Coop/comp" column fully on the page; the
  `\resizebox{\linewidth}{!}{...}` wrap plus tightened cell text
  ("vs.\ scr.", "per-scen.") fits within the body width without an
  overfull \hbox. The body §5 prose carries the load-bearing claim
  ("dominates on exactly one column — NE characterisability — and
  loses on every other") so the abbreviations do not move semantic
  load.
- **v5 R3 (§3 false promise of a per-cell predicted-vs-empirical
  table)** is **resolved**. L393–394 now reads "The full bias
  accounting and the 7-cell preview predicted-vs-empirical table
  appear in Appendix B." Appendix B §B.4 carries the 7-cell preview
  Table 5 (now correctly cross-referenced) and §B.5 the bias
  accounting. The body promise matches the appendix delivery.
- **v5 audit M1 (std-range estimator)** is **resolved**. §4 Results
  (L476) now reads "per-class mean within-cell std (≈ 0.21–0.44)",
  naming the estimator inline. Closes the read-ambiguity without
  changing the conclusion.
- **v5 audit M2 (cross-metric separation arithmetic)** is
  **resolved**. §4 Results (L477–481) switches the separation list to
  the metric-consistent homogeneous numbers (0.001, 0.017, 0.082)
  with an inline metric-consistency clause ("we use the homogeneous
  separations here for metric consistency since no-convergence cells
  lack gap_closed_ne"). Strengthens the ordering-not-significance
  conclusion: per-class std now dwarfs the largest separation by
  ~3× and the smallest by ~200×, where v5's cross-metric list
  had it at 2–6×.

## Dimension summary

| # | Dimension | Weight | Score | v5 → v6 move |
|---|---|---:|---:|---|
| 1 | Rigor of method / argument | 6 | 5 | unchanged |
| 2 | Evidence sufficiency       | 6 | 5 | unchanged |
| 3 | Clarity of contribution    | 5 | 5 | unchanged |
| 4 | Related-work positioning   | 5 | 4 | unchanged |
| 5 | Reproducibility            | 5 | 5 | unchanged |
| 6 | Figure & table quality     | 4 | 4 | **+2** (R1 + R2) |
| 7 | Prose & structural quality | 4 | 4 | **+1** (R3 fix removed structural promise mismatch) |
| 8 | Citation hygiene           | 5 | 5 | unchanged |
| 9 | Rhetorical economy         | 4 | 3 | unchanged |
| | **Total**                  | **44** | **40** | **+3** |

Full per-dimension justifications in `scoring.md`; line-level comments
in `comments.md`.

## Rationale (one paragraph)

v6 is the targeted punch-list pass v5 reviewer + auditor asked for.
The dim 6 finding that drove the v5 verdict ("Figure & table quality:
2/4 — Figure 1 has a real content defect ... Table 2 overflows the
right margin") is fully closed by the regenerated 3-panel categorical
heatmap (with a real source script) and the `\resizebox`-wrapped
Table 2. The dim 7 sub-finding ("§3 promises a per-cell table that
Appendix B does not deliver") is closed by softening the body
cross-reference. The auditor's two M-flags (M1 std-range estimator
under-specified, M2 cross-metric separation arithmetic) are closed by
inline edits to the §4 Results paragraph that strengthen rather than
weaken the ordering-not-significance conclusion. No new soft spots are
introduced. Net +3 on the /44 score (37 → 40) with all five v5 items
closed and no regression on any other dimension. Dim 9 (Rhetorical
economy) holds at 3/4 because the appendix word budget is unchanged
— that is a structural choice the user has opted into, not an
omission v6 promised to fix.

## Advance to audit

`advance: true`. The next phase is `pub-audit`, which will (a) re-run
the `pdflatex && bibtex && pdflatex && pdflatex` cycle on v6, (b)
confirm the Figure 1 regenerated PDF embeds cleanly and renders at
the documented page (verified at the reviewer level by inspecting
the rendered v6 PDF page 5), (c) confirm Table 2 has no overfull
\hbox warning, and (d) re-audit the §4 Results paragraph against the
homogeneous separation numbers in `recalibrated_verdict.json` (the
0.001, 0.017, 0.082 should match `0.051 − 0.050`, `0.050 − 0.033`,
`0.033 − (−0.049)`).

## Top revision priorities

**None — ready to ship.**

No actionable items remain at the workshop-submission bar. Optional
camera-ready polish items, not blocking advance:

1. **Dim 9 appendix bulk** is the one dimension that could move up
   without a hard re-architecture: promoting Appendix A §A.6 and
   parts of Appendix B §B.5 to supplementary materials would let the
   body alone read tighter. Not blocking; the user has opted out of
   workshop budget enforcement.
2. **Appendix A §A.6** still names Volunteer's Dilemma, N-player
   Public Goods, Stag Hunt, and free-rider problem as informal
   attributions without `\cite{}` links. For the camera-ready a
   workshop reviewer may expect a `\cite{}` per named template.
   Minor only.
3. **§B.4 / §B.5 boundary** still reads as a layout accident:
   §B.4 contains one lead-in sentence before Table 5 floats into
   §B.5. Not a content issue; restructuring to fold Table 5 into
   §B.4 proper would tighten the appendix.
