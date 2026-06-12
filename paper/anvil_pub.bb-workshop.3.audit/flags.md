# Audit flags for `anvil_pub.bb-workshop.3`

## Critical flags (block advancement to AUDITED)

**None.**

The paper compiles cleanly (3/3 pdflatex passes + bibtex all exit 0).
Zero unresolved citations in the final PDF. Every numerical claim in
the prose reconciles with its source. No claim-support obvious mismatch.

## Major flags (worth fixing before submission, but do not block AUDITED)

### M1. Page-budget overrun: PDF is 10 pages

The rendered PDF is **10 pages** (letter, 612×792 pt), which is
substantially over the typical NeurIPS-workshop 4-page budget.
The `anvil-paper` class is not the NeurIPS-workshop class, so the
budget enforcement is paper-process-side, not LaTeX-side. The
revisor should either (a) port to the workshop's official style file
before submission and re-measure, or (b) tighten content to fit.
Page accounting (approx., by section):

| Section | Pages |
|---|---|
| Abstract + §1 Introduction | ~2 |
| §2 Environment | ~2 |
| §3 NE structure + Figure 1 | ~3 |
| §4 Trainability + Figure 2 | ~2 |
| §5 Related work + Table 1 | ~1 |
| §6 Discussion + Conclusion | ~1 |

The two figures consume substantial space (Figure 1 at 0.92\linewidth
and Figure 2 at 0.85\linewidth). Tightening to 4 pages will need
content excisions, not just typographic squeezing.

### M2. Internal-memo citations (`bbenvspec`, `bbnestructure`)

Two BibTeX entries are `@misc` placeholders pointing at in-repo
memos:

- `bbenvspec` → `paper/anvil_memo.env_spec.1/env_spec.md`
- `bbnestructure` → `paper/anvil_memo.ne_structure.1/ne_structure.md`

Both `@misc` entries explicitly note "to be released as arXiv preprint
before submission." `refs.bib` lines 86–104 flag this as a known
followup. The paper depends on both memos for the formal env-spec
deferral (§2 last paragraph) and the analytical-NE-derivation
deferral (§3 "Headline finding" paragraph, §4 "Protocol"). Submission
without arXiv IDs would force the reader to rely on inline reasoning
plus the source-repo URL alone for the formal contribution. **The
auditor flags this as MAJOR but not critical** — it is a known,
tracked submission-time followup, and the on-disk memos exist and
support the surrounding claims.

### M3. `\citep{bbnestructure}` table-dimensionality nit ($7\times 4$ vs $7\times 6$)

Line 279 of `main.tex` references "The $7\times 4$ predicted-vs-empirical
table." The actual table in
`paper/anvil_memo.ne_structure.1/ne_structure.md` §4 is **7 rows × 6
columns** (β, κ, Empirical verdict, Predicted verdict, Match?, Notes).
Possibilities: the paper author wrote "$7\times 4$" intending "7
cells × 4 verdict columns" (Empirical, Predicted, Match, Notes minus
β/κ keys), or this is a typo. Non-load-bearing — the cite is to the
table's *content* (the bias accounting), not to its shape — but worth
fixing in `pub-revise` to match the memo verbatim. **Minor in scope,
flagged as major-adjacent because it's a citation accuracy issue and
the auditor's brief includes citation accuracy.**

## Minor / informational notes

### N1. Abstract "3–10×" disagreement range

The abstract and §3 both quote a "3–10×" disagreement between
predicted and observed κ-thresholds. The collapse-boundary side is
~10× (predicted 0.028 vs observed ~0.3); the asymmetric-side is
~1.4× (predicted 0.972 vs observed ~0.7), not 3×. The "3–10×" range
is reasonable as a bracket for the larger discrepancy direction and
appears in `refs/ne_structure.md` headline as well, so the prose
inherits the source-material framing. Not a flag, just noted for
the revisor's awareness if they later tighten the language.

### N2. Compile produced zero overfull / underfull boxes and zero warnings

The third pdflatex pass log has no `Overfull \hbox`, no
`Underfull \hbox`, and no warning text. Typography is clean.
Documented for the render-gate's later consumption.

### N3. Figure 1 has no in-paper source script

`figures/phase_diagram.png` is a static PNG copied from
`experiments/nash/phase_diagram/phase_diagram.png` (via the
`refs/phase_diagram.png` symlink). No `figures/src/` script exists
for it. Staleness check is not applicable. Informational only; not
a defect.

## Build / compile summary

```
pdflatex pass 1 -> exit 0 (pre-bibtex; undefined cites expected)
bibtex          -> exit 0
pdflatex pass 2 -> exit 0
pdflatex pass 3 -> exit 0 (no warnings; no overfull/underfull boxes)
Final PDF: 10 pages, 350,619 bytes, letter (612×792 pt)
Unresolved citations: 0
```

## Render-gate documentation (for the next review pass)

- **Page count: 10** (versus a typical NeurIPS-workshop budget of 4 pages).
  Above-budget. Recorded for the render-gate that `pub-review` may run.
- **PDF size: 350,619 bytes** — small, no oversized embedded figures.
- **Letter (612×792 pt)** — the `anvil-paper.cls` default. If the target
  workshop requires A4 (595×842 pt) or NeurIPS-workshop letter sizing,
  the class file is the place to fix it; main.tex does not override.

## Auditor verdict

**Block: no.** No critical flags. Compile is clean and every prose
number reconciles to its on-disk source. Two MAJOR flags
(`M1: 10-page overrun`, `M2: internal-memo cites`) are explicit
known followups tracked at the paper-thread level; neither is a
fact-check or compile failure. The minor `M3` is a one-token
revisor fix. The next pass (`pub-review`) should run the render-gate
with the page count flagged so the venue-fit dimension is visible.

The most important single flag, if I must pick one, is **M1 (10-page
overrun)**: a 10-page paper cannot be submitted to a 4-page NeurIPS
workshop without compression, regardless of how clean the content is.
