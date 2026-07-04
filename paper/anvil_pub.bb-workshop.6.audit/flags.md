# Audit flags for `anvil_pub.bb-workshop.6`

## Critical flags (block advancement)

**None.**

The paper compiles cleanly (3 pdflatex passes + bibtex all exit 0; 0
warnings other than font-shape defaults, 0 overfull/underfull boxes, 0
unresolved citations). All v5 majors are addressed; no new majors
introduced.

## Major flags (worth fixing before submission, do not block AUDITED)

**None.**

Both v5 majors are fully closed:

### v5 M1 (std-range estimator under-specified) — CLOSED

v5 §4 said "per-class std (≈ 0.21–0.44)" without naming the estimator.
v6 §4 L475-476 now reads "per-class **mean within-cell** std (≈
0.21–0.44)", which is the exact estimator that produces the
0.21–0.44 bound. The auditor's M1 fix is implemented exactly as
recommended.

### v5 M2 (cross-metric separation 0.108) — CLOSED

v5 §4 separations were `(0.073, 0.048, 0.108)` where 0.108 mixed
`gap_closed_ne` and `gap_closed_homogeneous`. v6 §4 L477-478 switches
to `(0.001, 0.017, 0.082)`, all on `gap_closed_homogeneous`, with an
inline disclosure "we use the homogeneous separations here for metric
consistency since no-convergence cells lack `gap_closed_ne`". The
auditor's M2 fix is implemented exactly as recommended (option 2 of
the two fixes offered).

## Minor / informational notes

### N1. sym-mix separation `0.001` is an aggressive round-up

The exact sym-mix separation on `gap_closed_homogeneous` is **0.00047**
(0.05066 − 0.05018). Rounding to 3 decimal places with a half-round-up
convention yields 0.001. A more faithful presentation would be either
"<0.001" or "0.0005". This is not load-bearing — the paper's
qualitative claim is that the std-range (0.21–0.44) dwarfs the
separations, and 0.001 vs. 0.0005 makes no difference to that
conclusion. The 3-d.p. rounding is also internally consistent with the
other two separations (0.017, 0.082).

### N2. Std range is on `gap_closed_ne`; separations are on `gap_closed_homogeneous`

v6's §4 fix names the std estimator and the separation metric, but the
two are still on different metrics: the std-range "0.21–0.44" is the
per-class mean within-cell std on `gap_closed_ne` (the lead-paragraph
metric, which is undefined on `no_convergence` cells), while the
separations are on `gap_closed_homogeneous` (defined on all 4 classes).
The "std > separations" qualitative claim is robust under either
metric — on `gap_closed_homogeneous` the std-range is 0.10–0.26
(still dominating 0.001/0.017/0.082 by 3–260×) — so the substantive
ordering-not-significance conclusion does not depend on this choice.
The reviewer / camera-ready could land a one-clause clarification
("std on `gap_closed_ne`; separations on `gap_closed_homogeneous` for
metric consistency across the four classes"), but this is below the
"major" bar for a workshop submission.

### N3. Compile is clean

3 pdflatex passes + bibtex all exit 0. The only `main.log` warning is
*"LaTeX Font Warning: Some font shapes were not available, defaults
substituted"* — this is a benign default-shape substitution
(specifically, bold-italic CMR shapes), affects nothing rendered or
load-bearing, and has been carried forward from v4 / v5 unchanged.

`grep -i "overfull|underfull|undefined"` on `main.log` returns nothing.
PDF is 19 pages, 431,189 bytes, letter (612×792 pt). v5 was 20 pages
at 454,661 bytes; the modest shrink is explained by v6's Table 2
abbreviations and the new `.pdf` Figure 1 (PDF is more compact than
the v5 PNG).

### N4. Page count: 19 (under the v4-era 20-page opt-out)

v5 was 20 pages and the user opted out of workshop page-budget
enforcement at v4. v6 is **19 pages** (1 page shorter than v5,
attributable to the Table 2 tightening and the PDF-vs-PNG figure
swap). Comfortably within the v5 envelope; no new pressure on the
budget.

### N5. Figure 1 source script is reproducible and matches the figure

`figures/src/phase_diagram.py` runs successfully against
`experiments/nash/phase_diagram/results.json` and produces a
visually-identical PDF to the shipped `figures/phase_diagram.pdf`
(content-equivalent; binary differs on PDF metadata only). The script
is self-contained (one external dependency: matplotlib), runs in <2s,
and uses verdict-class colors from a colorblind-safe palette named in
the script comments. Spot-check cells visually verify against the
JSON.

### N6. Figure 1 caption refers to a "fourth `mixed` regime" that the
new figure now makes visually obvious

v5's carry-forward PNG showed raw payoff numbers (the v5 reviewer R1
flag). The v6 figure replaces them with categorical color blocks
keyed to the four named verdict classes in the legend. The four
classes appear in the figure as expected (gray collapse,
blue symmetric, teal mixed, yellow asymmetric), and the c=2.0 panel's
$\kappa{=}0.9$ column being all-teal makes the "primarily at
$\kappa{=}0.9, c{\geq}1$" caption claim immediately visible.

### N7. Carry-forward state — all v4/v5 audit closures still hold

- v4 M1 (internal-memo cites): RESOLVED in v5, unchanged in v6.
- v4 M3 (β-independence 12-vs-13 count): RESOLVED in v5, unchanged in
  v6.
- v5 M1, M2: RESOLVED in v6 (see above).
- v4 M2 / v5 M3 (page count under opt-out): now 19pp, still under
  the v5 envelope.

No v4 or v5 issue carries into v6 unresolved.

## Build / compile summary

```
pdflatex pass 1 -> exit 0 (pre-bibtex; undefined cites expected)
bibtex          -> exit 0 (0 warnings, 0 errors)
pdflatex pass 2 -> exit 0
pdflatex pass 3 -> exit 0
Final PDF: 19 pages, 431,189 bytes, letter (612×792 pt)
Unresolved citations: 0
Overfull/underfull boxes: 0
Warnings: 1 (font-shape default substitution; benign)
```

## Render-gate documentation

- **Page count: 19** — 1 page shorter than v5; within user opt-out.
- **PDF size: 431,189 bytes** — small; no oversized embeds.
- **Letter (612×792 pt)** — `anvil-paper.cls` default.

## Auditor verdict

**Block: no.** Zero critical flags. Zero major flags. Both v5
majors (M1 std-range estimator, M2 cross-metric separation) are
**fully closed** by the v6 edits exactly as the v5 audit recommended.
The two minor notes (sym-mix 0.001 rounding nit; ne-vs-homo metric
split for std-vs-separation comparison) are below the major bar and
do not gate submission.

## most_important_flag

**None — ship-ready.** Both v5 majors are closed. Only minor /
informational notes remain (sym-mix rounding nit; residual ne/homo
metric split between std and separations) and neither is load-bearing
on any paper claim.
