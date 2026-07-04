# Verdict — anvil_pub.bb-workshop.5

- **Total score**: **37 / 44**
- **Decision**: `advance: true` (37 ≥ 35 threshold; no critical flags)
- **Rubric**: `anvil-pub-v2` (9 dimensions, /44, advance ≥35; on-disk
  `.anvil/skills/pub/rubric.md`).

## Critical flags

**None.**

No issue rises to "a sophisticated PC member would stop reading."
Specifically scrutinized:

- The v4 audit's M1 (`bbenvspec` / `bbnestructure` `@misc` placeholders)
  is **resolved**: both `@misc` entries are deleted from `refs.bib`; the
  memo content is inlined verbatim as Appendix A (formal env spec) and
  Appendix B (analytical NE characterisation). All five v4 citation
  sites now resolve to `\ref{app:envspec}` / `\ref{app:nestructure}`.
  All 9 remaining `\cite{}` keys resolve to bib entries (verified by
  PDF render: no `[??]` in pages 1–20).
- The v4 audit's M3 (β-independence stated "13/13" vs. correct "12/13"
  on the Nash 39-cell grid) is **resolved**: §3 L282–286 now reads
  "12 of the 13 (κ,c) rows" and explicitly names the single splitting
  row (κ=0.9, c=0.5) and what its β values verdict to. §4 L477–478
  correctly carries the PPO-side "all 13" claim with the PPO-skipped-
  the-splitting-row caveat in parentheses. Figure 1 caption (L408–411)
  is consistent.
- The §4 "exact agreement" → "consistent with" calibration and the
  explicit per-class-std-vs-class-separation disclosure (L471–476) are
  a real improvement on the v2 reviewer's overclaim concern and are
  worth keeping.

## Dimension summary

| # | Dimension | Weight | Score |
|---|---|---:|---:|
| 1 | Rigor of method / argument | 6 | 5 |
| 2 | Evidence sufficiency       | 6 | 5 |
| 3 | Clarity of contribution    | 5 | 5 |
| 4 | Related-work positioning   | 5 | 4 |
| 5 | Reproducibility            | 5 | 5 |
| 6 | Figure & table quality     | 4 | 2 |
| 7 | Prose & structural quality | 4 | 3 |
| 8 | Citation hygiene           | 5 | 5 |
| 9 | Rhetorical economy         | 4 | 3 |
| | **Total**                  | **44** | **37** |

Full per-dimension justifications in `scoring.md`; line-level comments
in `comments.md`.

## Rationale (one paragraph)

v5 is a real net positive over v4 on dim 2, dim 5, and dim 8: the
v2 reviewer's major flags about contribution clarity (per-cell-baseline
buried), overclaimed "confirming" β-independence language, and missing
inline truth table are all addressed in calibrated prose. The two new
appendices are well-organised, consistent with body §2 and §3 notation,
and remove the v4 M1 placeholder-citation flag at the source. The
v4 M3 numerical over-count is fixed cleanly. The price paid is dim 6
(Figure & table quality) drops from 3 to 2: **Figure 1 has a real
content defect** — it shows raw numerical payoff values (-9693, -648,
72) in the heatmap cells rather than the four regime classes the
caption describes, the `mixed` class swatch in the legend is unused
in the plotted panel, and two cells render as bare dashes. The figure
is the visual anchor for §3's headline claim about four regimes; a
reader scanning the PDF will land on the figure and see numerics that
do not match the caption's qualitative regime story. Figure 1 was
unchanged from v4 (where it was also flagged as a static PNG with no
source script), so this is a carry-forward defect rather than a v5
regression — but a workshop reviewer who skims the PDF will see it.
Additionally Table 2 overflows the right margin on the rendered PDF
(p.7 cuts the "Coop/comp" column header and contents). Net: the paper
clears the advance bar at 37/44 with comfort, but should not ship to
a workshop without regenerating Figure 1 as a categorical regime
heatmap and reflowing Table 2.

## Advance to audit

`advance: true`. The next phase is `pub-audit`, which will (a) re-run
the `pdflatex && bibtex && pdflatex && pdflatex` cycle on v5, (b)
re-audit per-claim citation support (the half of D8 that `pub-review`
defers) — particularly for the new appendix attributions to Volunteer's
Dilemma, Public Goods, and Stag Hunt that ship without `\cite{}`
links, (c) confirm Table 1's per-κ predicted-class-share numbers (0/9,
6/6, 6/9, 0/6, 7/9) reconcile against `results.json`, and (d) flag
the Figure 1 and Table 2 rendering defects for the revisor.

## Top revision priorities (advisory, not required to advance)

1. **Regenerate Figure 1 as a regime heatmap.** The current PNG shows
   raw payoff numbers and a legend entry (`mixed`) that does not
   appear in the plotted body. A categorical heatmap colored by
   `verdict` (one of {symmetric_only, mixed, asymmetric_only,
   no_convergence}) and showing all three c-panels (the caption
   promises "the c=0.5 panel of the full 39-cell grid" but deferring
   c=1.0 and c=2.0 to Appendix B is rhetorically weak when those are
   the panels where the `mixed` class actually appears). Source script
   should land at `figures/src/phase_diagram.py` to match Figure 2's
   pattern.
2. **Fix Table 2 right-margin overflow.** The "Coop/comp" column header
   and entries clip on rendered page 7. Either narrow `\tabcolsep`,
   wrap the "vs. scripted" / "per-scenario" cells, or move the table
   to a `\begin{table*}` 2-column span if the class supports it.
3. **Body §3's "per-cell predicted-vs-empirical table appears in
   Appendix B" promise is not fulfilled.** L389 says "The per-cell
   predicted-vs-empirical table and the full bias accounting appear in
   Appendix B." Appendix B carries the 7-cell preview (Table 5) and
   the per-κ-band summary (Table 4), but no full 39-cell per-cell
   predicted-vs-empirical table exists in Appendix B. Either populate
   §B.4 with the full per-cell breakdown the body promises, or soften
   the body cross-reference to "The 7-cell preview comparison and the
   full bias accounting appear in Appendix B."
