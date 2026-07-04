# Audit flags for `anvil_pub.bb-workshop.5`

## Critical flags (block advancement)

**None.**

The paper compiles cleanly (3/3 pdflatex passes + bibtex all exit 0; 0 warnings,
0 overfull/underfull boxes, 0 unresolved citations). The new Table 1 row counts
all reconcile to `experiments/nash/phase_diagram/results.json` exactly. The §4
class-mean numbers (0.180, 0.107, 0.059, -0.049), the separations
(0.073, 0.048, 0.108), and the 4×-budget mean (-0.108) all reconcile to source.
Two of v4's three majors are fully resolved (M1: appendix inlining; M3:
"12/13" prose now in place); v4's M2 (page count) is documented but the user
has opted out of workshop budget enforcement.

## Major flags (worth fixing before submission, do not block AUDITED)

### M1. "Per-class std (≈ 0.21–0.44)" range underspecifies the read

**Section / line:** §4 Results paragraph, the new v5 sentence.

The new sentence claims: *"per-class std (≈ 0.21–0.44) is larger than the
class-mean separations (0.073, 0.048, 0.108)."*

Under the **class-level mean of per-cell stds** reading, the bounds are
correct (mixed 0.21, no_conv 0.26, asym 0.42, sym 0.44 — fits 0.21–0.44 to
two d.p.). But the raw per-cell std list spans **0.16 to 0.99** (across
classes and individual cells, with sym and asym cells hitting cells with
0.85 and 0.99 within-cell std). A reader who interprets "per-class std"
as "the per-cell std values, viewed within each class" will see a wider
range than the parenthetical suggests, and the ordering-not-significance
framing would land harder. The current bound is also defensible (it is
the mean per-class within-cell std, which is the natural pooled estimator
for the variance the ordering-not-significance argument needs to clear),
but the paper does not tell the reader which estimator the range is. A
one-clause clarification ("per-class mean within-cell std, ≈ 0.21–0.44")
would close the ambiguity without disturbing the conclusion. **Major
because** the sentence is the v5's load-bearing new statistical-language
calibration; submitting at the current phrasing leaves a small but real
opening for "you compared the wrong variance to the wrong separation."

### M2. The asymmetric→collapse class-mean separation `0.108` crosses metrics

**Section / line:** §4 Results paragraph, separation list `(0.073, 0.048, 0.108)`.

The first two separations (0.073 = sym - mix, 0.048 = mix - asym) are
**within-metric** (all on gap_closed_ne). The third separation
0.108 = asym (gap_closed_ne, 0.059) - collapse (gap_closed_homogeneous, -0.049)
**straddles two metrics**. The previous sentence acknowledges the fallback
("the metric falls back to gap_closed_homogeneous because no NE policy
exists"), but the separation arithmetic itself does not asterisk the
cross-metric subtraction. A strict reader would object that 0.108 is not
a meaningful gap_closed_ne separation (one operand isn't gap_closed_ne)
and a different baseline could produce a different number. The §4 Caveats
paragraph independently reports the homogeneous-only ordering
(0.051, 0.050, 0.033, -0.049), which sidesteps this — but the prose path
through the Results paragraph relies on the 0.108 number to set up the
"separations are smaller than the std" comparison. **Major because**
load-bearing on the ordering-not-significance framing. Fix: add a
half-sentence noting the asy→collapse arithmetic mixes
gap_closed_ne and gap_closed_homogeneous because collapse cells lack the
former, or alternatively use the all-homogeneous separations
(sym 0.051 → mix 0.050 → asym 0.033 → col -0.049) which keep the metric
consistent.

### M3. v4 M2 (page count: 20 pages) carries forward

v4 PDF was 11 pages; v5 PDF is **20 pages** (the appendix inlining is the
expected ~7-8 page growth driver, plus modest growth in §3 from the new
Table 1). The user opted out of workshop page-budget enforcement at v4
audit time and the v5 changelog explicitly inherits that opt-out, so
this is reported for the render-gate's awareness, **not** as a blocking
issue. The body alone (through Conclusion, before References) is **9
pages**, so a strict-budget submission target could promote
Appendix A and Appendix B to a supplementary-materials file without
disturbing the body.

## Minor / informational notes

### N1. v4 M1 (internal-memo cites) — RESOLVED in v5

v4's M1 flagged `bbenvspec` and `bbnestructure` as `@misc` placeholders.
v5 deletes both entries from `refs.bib`, inlines the memo content into
Appendix A and B, and rewires the five v4 citation sites to
`\ref{app:envspec}` / `\ref{app:nestructure}`. Verified: zero remaining
`\cite{}` to either deleted key (only string mentions are in the file-
header comment and a bib-file comment block, both expected). The stray
`\citep{diekmann1985}` mentioned in the v5 changelog is also gone (zero
matches). Citation health is clean.

### N2. v4 M3 (β-independence count: "12/13" vs "13/13") — RESOLVED in v5

v4's M3 flagged the §3 / Figure 1 caption claim "13/13 rows identical
across β" as a one-token overstatement (the Nash data has 12/13). v5
correctly says "**12 of the 13** (κ, c) rows" at L282 and L410, and
names the splitter "(κ=0.9, c=0.5)" with the β=0.1 → mixed / β∈{0.5,0.9}
→ asymmetric_only breakdown. Cross-checked against `results.json`:
the splitter is identified correctly. Closed.

### N3. Compile produced zero warnings, zero overfull/underfull boxes

Three pdflatex passes plus bibtex all exit 0. `grep -i "warning|error|
overfull|underfull|undefined"` on `main.log` returns nothing. PDF is
**20 pages**, 454,661 bytes, letter (612×792 pt). Typography is clean.

### N4. Table 1 row totals match changelog and Nash JSON exactly

The changelog item 4 predicted 5 row counts that match the source JSON:
- κ=0.1: 6 no_convergence + 3 asymmetric (n=9) ✓
- κ=0.3: 6 symmetric (n=6) ✓
- κ=0.5: 6 symmetric + 3 mixed (n=9) ✓
- κ=0.7: 6 asymmetric (n=6) ✓
- κ=0.9: 7 mixed + 2 asymmetric (n=9) ✓

The κ=0.3 and κ=0.7 rows are 6-cell rows because the Nash sweep only
sampled those κ values at c ∈ {1.0, 2.0} (the high-κ × c=0.5 corner is
subsumed by c=1.0). The Table 1 caption discloses this. The reduction's
"modal class on 3/5 κ rows" claim verifies on these counts.

### N5. Figure 1 / Figure 2 unchanged from v4

The two figure files (`phase_diagram.png`, `recalibrated_heatmap.pdf`)
ship at v4 mtimes (2026-06-13 07:27 — recompile timestamp, but file
contents unchanged). Both figures were spot-checked in v4 audit against
their respective source JSONs and reconciled; no v5 change requires a
re-audit of the figures themselves. The Figure 1 / Figure 2 caption
polish (changelog item 7) does not alter any rendered number; only
prose tightening.

### N6. Appendix B's "single-cell baseline inverted the ordering" carries 7-cell preview numbers

The v4-verified 7-cell preview numbers (asym 0.262, sym 0.091,
no_conv -0.176 under the v1 single-cell baseline) are reused in §4's
methodological-observation paragraph and the Conclusion. The source for
those numbers is `experiments/p3_specialization/diagnostics/results/` —
verified at v4 audit, unchanged in v5.

## Build / compile summary

```
pdflatex pass 1 -> exit 0 (pre-bibtex; undefined cites expected)
bibtex          -> exit 0
pdflatex pass 2 -> exit 0
pdflatex pass 3 -> exit 0 (no warnings; no overfull/underfull boxes)
Final PDF: 20 pages, 454,661 bytes, letter (612×792 pt)
Unresolved citations: 0
```

## Render-gate documentation (for the next review pass)

- **Page count: 20** — user opted out of workshop page-budget enforcement;
  body is 9 pages, appendix A+B are ~11 pages.
- **PDF size: 454,661 bytes** — small, no oversized embedded figures.
- **Letter (612×792 pt)** — the `anvil-paper.cls` default.

## Auditor verdict

**Block: no.** Zero critical flags. The two v4 majors that were
substantive (M1 internal-memo cites, M3 12-vs-13 β-independence count)
are both fully resolved. v4's M2 (page count) is reported but carries
forward under the user's existing opt-out.

The two v5-specific majors (**M1 std-range ambiguity**, **M2 cross-metric
separation arithmetic**) both attach to the new "ordering-not-significance"
sentence introduced by changelog item 5. Neither is a load-bearing
correctness error: the underlying class-mean ordering and the
qualitative "separations are smaller than the noise" conclusion both
survive any reasonable disambiguation. Both are transparency / phrasing
gaps that would close cleanly under one or two clarifying sub-clauses.

## most_important_flag

**M2 (cross-metric separation 0.108).** The asymmetric→collapse separation
crosses gap_closed_ne and gap_closed_homogeneous, and the sentence using
it for the ordering-not-significance comparison does not flag the
metric switch. Both M1 (std range) and M2 attach to the same new v5
sentence; M2 is the load-bearing one because a strict reader can
challenge the arithmetic, whereas M1 is interpretation-shape only.
A camera-ready pass should land both fixes together: pin the std-range
estimator and asterisk the cross-metric arithmetic (or switch the whole
separation list to the metric-consistent homogeneous version).
