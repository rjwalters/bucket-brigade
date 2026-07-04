# changelog: anvil_pub.bb-workshop.5 → .6

## Trigger

v6 lands the v5-reviewer and v5-auditor punch list. Both critics
advanced v5 (reviewer 37/44; auditor pass) with 0 critical flags, but
together surfaced 5 substantive items worth closing before submission.
v6 addresses all 5.

## Headline changes

### 1. Figure 1 regenerated as a categorical regime heatmap (reviewer R1)

The v5 (and v4) `figures/phase_diagram.png` was a static carry-forward
PNG that displayed **raw payoff numbers** (−9693, −648, +72) in the
heatmap cells instead of the four NE regime classes the caption
described. Color-coding matched the legend but the inline numerics
contradicted the regime-class story §3 was telling. The `mixed`
legend swatch was unused (the plotted c=0.5 panel has no mixed cells),
and two unsampled cells rendered as bare em-dashes.

v6 ships a new source script at `figures/src/phase_diagram.py` that
reads `experiments/nash/phase_diagram/results.json` and renders a
**3-panel categorical heatmap** (one panel per c ∈ {0.5, 1.0, 2.0},
each 3×5 β×κ grid) colored by verdict, with no inline payoff
numerics, the unsampled cells hatched grey with "n/a" labels, and a
shared legend at the bottom. The new figure is the visual anchor §3
already claims it is.

### 2. Table 2 (benchmark comparison) margin fix (reviewer R2)

The v5 rendered PDF clipped the "Coop/comp" column on page 7. v6
reduces `\tabcolsep` from 4pt to 3pt, narrows the
SMAC/SMACv2 entry from "vs. scripted" to "vs.\\ scr.", and tightens
"per-scenario" to "per-scen.". The body §5 prose carries the
load-bearing claim, so the abbreviations are purely visual; the
table caption is unchanged. No overfull `\hbox` warnings post-edit.

### 3. §3 cross-reference softened (reviewer R3)

v5 §3 L389 promised:
> "The per-cell predicted-vs-empirical table and the full bias
> accounting appear in Appendix B."

Appendix B does not carry a full 39-cell per-cell predicted-vs-
empirical table; it has the 7-cell preview (Table 5 / §B.4) and the
per-κ summary that v5 promoted to the body as Table 1. v6 softens
the cross-reference to:
> "The full bias accounting appears in Appendix B."

This matches what is actually there.

### 4. §4 std-range estimator named (auditor M1)

v5 §4 said "per-class std (≈ 0.21-0.44)" without telling the reader
which estimator the range was. The bound is correct under the
**per-class mean of within-cell stds** reading but a reader who
interprets the words as "the per-cell std values within each class"
sees a wider range (0.16 to 0.99 across all cells). v6 names the
estimator inline:
> "per-class mean within-cell std (≈ 0.21-0.44)"

One clause. Closes the ambiguity without changing the conclusion.

### 5. §4 separations switched to homogeneous-only (auditor M2)

v5 §4 reported class-mean separations `(0.073, 0.048, 0.108)` where
the first two are within-metric (all on `gap_closed_ne`) and the
third **crossed metrics** (asym on `gap_closed_ne` minus collapse on
`gap_closed_homogeneous`). The arithmetic was correct; the metric
switch was silent in the load-bearing ordering-not-significance
sentence.

v6 switches the separations to the **metric-consistent
`gap_closed_homogeneous` numbers** (which are already reported in
the §4 Caveats paragraph for the homogeneous ordering):
- sym 0.051 − mix 0.050 = **0.001**
- mix 0.050 − asym 0.033 = **0.017**
- asym 0.033 − col −0.049 = **0.082**

All three separations are now on the same metric. The
ordering-not-significance conclusion strengthens (the per-class
mean within-cell std at 0.21-0.44 now dwarfs all three separations
by a factor of 3-200×, not 2-6×), which is exactly the
auditor's expected effect of removing the cross-metric arithmetic.

The Results paragraph itself continues to lead with the
`gap_closed_ne` class means (0.180 / 0.107 / 0.059 / -0.049) for
class ordering — that ordering is unaffected. Only the
**separation list inside the parenthetical** switches metric. A
half-sentence makes the metric source explicit.

## What is unchanged

- All §2 content, all §3 derivation, all §5 prose other than Table 2 cell text, all §6 Discussion paragraphs, all Appendices.
- All numerical claims about per-class means (0.180 / 0.107 / 0.059 / -0.049), the 4×-budget shift (-0.049 → -0.108), the Nash 39-cell verdict counts (12/11/10/6), Table 1's per-κ counts, the κ-thresholds (0.028, 0.65, 0.972), and the survival coefficient Ã=36.24.
- refs.bib is unchanged.
- Figure 2 (recalibrated_heatmap.pdf) and its source script are unchanged.

## Files changed in v6

```
anvil_pub.bb-workshop.6/main.tex            edits at §3 L389, §4 Results + Caveats, §5 Table 2, header comment
anvil_pub.bb-workshop.6/figures/src/phase_diagram.py    NEW source script
anvil_pub.bb-workshop.6/figures/phase_diagram.pdf       NEW output (was .png in v5)
anvil_pub.bb-workshop.6/main.pdf            re-compiled
anvil_pub.bb-workshop.6/changelog.md        this file
anvil_pub.bb-workshop.6/_progress.json      iteration=6, revised_from=5
```

## Page count

v5 PDF: 20 pages. v6 PDF: expected 20 pages (no body content added;
edits are within-paragraph or cell-text only). The Figure 1 file
extension changes from `.png` to `.pdf` to align with the
`\includegraphics{figures/phase_diagram}` extension-search order
(LaTeX prefers `.pdf` when available, which both keeps figure quality
high and avoids an `.png` re-rasterise).

## Soft spots carried forward into v6

Same as v5. No new soft spots introduced.
