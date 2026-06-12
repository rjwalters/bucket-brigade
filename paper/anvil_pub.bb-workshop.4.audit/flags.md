# Audit flags for `anvil_pub.bb-workshop.4`

## Critical flags (block advancement to AUDITED)

**None.**

The paper compiles cleanly (3/3 pdflatex passes + bibtex all exit 0).
Zero unresolved citations in the final PDF. All four §4 class-summary
numbers, the 4×-budget structural-failure claim, the Nash 39-cell
verdict_counts, and Figure 2 spot-checks reconcile to source. No
claim-support obvious mismatch. The single numerical issue surfaced by
this audit (M3 below) is a one-token over-count, not a conclusion shift.

## Major flags (worth fixing before submission, but do not block AUDITED)

### M1. Internal-memo citations (`bbenvspec`, `bbnestructure`)

Two BibTeX entries are `@misc` placeholders pointing at in-repo memos:

- `bbenvspec` → `paper/anvil_memo.env_spec.1/env_spec.md`
- `bbnestructure` → `paper/anvil_memo.ne_structure.1/ne_structure.md`

Both `@misc` entries explicitly note "to be released as arXiv preprint
before submission" and `refs.bib` flags this as a known followup. The
paper depends on both memos for the formal env-spec deferral (§2 last
paragraph) and the analytical-NE-derivation deferral (§3 "Headline
finding" paragraph). Submission without arXiv IDs would force the reader
to rely on inline reasoning plus the source-repo URL alone for the
formal contribution. Carried forward from v3 M2 unchanged. **Major but
not critical** — known, tracked followup.

### M2. Page-budget overrun: PDF is 11 pages

The rendered PDF is **11 pages** (letter, 612×792 pt), versus the typical
NeurIPS-workshop 4-page budget. v3 was 10 pages; v4 grew by ~1 page
because §3 added the `mixed`-class paragraph (~6 lines), §4 added the
4×-budget paragraph (~10 lines), and the Figure 2 caption grew. Per the
audit brief, the user has opted out of workshop page-count enforcement,
so this is **NOT critical**. Reported for the render-gate's awareness.

Rough page accounting:

| Section | Pages |
|---|---|
| Abstract + §1 Introduction | ~2 |
| §2 Environment | ~2 |
| §3 NE structure + Figure 1 | ~3 |
| §4 Trainability + Figure 2 + new 4× paragraph | ~2.5 |
| §5 Related work + Table 1 | ~1 |
| §6 Discussion + Conclusion | ~0.5 |

### M3. β-independence count overstated: "13/13" should be "12/13"

`main.tex` L262 (§3), L394 (§4), and the Figure 1 caption L332 all claim:

> "every $(\kappa,c)$ row shows an identical verdict across all sampled
> $\beta$ ($13/13$ rows with $\geq 2$ $\beta$ samples)"

Recomputing from `experiments/nash/phase_diagram/results.json`: 13
(κ, c) groups have ≥2 β samples, but **only 12 of them** have an
identical verdict across all sampled β. The single exception is
(κ=0.9, c=0.5):

- β=0.1 → `mixed`
- β=0.5 → `asymmetric_only`
- β=0.9 → `asymmetric_only`

The qualitative β-independence picture survives (12/13 = 92% identical),
and the κ=0.9, c=0.5 row sits on the `mixed`/`asymmetric_only` boundary
where some sensitivity is intuitive. But the unqualified "13/13"
overstates the test on this paper's own Nash-solver data.

**Loadedness**: load-bearing on a structural-claim count, not on a
conclusion. The §3 phase-order claim and the §4 PPO ordering are
unaffected. **Fix is one-token**: replace "13/13" with "12/13" and add
a half-sentence ("the one exception sits on the mixed/asymmetric
boundary at $\kappa{=}0.9, c{=}0.5$") or "all but one row at the
mixed/asymmetric boundary." Should be done in `pub-revise` for v5 if a
v5 is needed, or as a 1-line errata fix at submission time.

Note: the PPO sweep's 13/13 row claim (paper §4 L394) IS correct on its
own data, because the PPO sweep skipped the β=0.1 cell at (κ=0.9, c=0.5)
— one of the two cells called out in §4's protocol disclosure. The
13/13 PPO-side count is technically correct on the 12 (κ, c) rows that
have ≥2 β samples in PPO_v2. The flag is specifically about the Nash
solver grid claim, which §3 and the Figure 1 caption attribute the
13/13 to.

**Flagged major because it is a structural claim about a load-bearing
property of the analytical-vs-empirical comparison, even though the
gap-to-conclusion is small.**

## Minor / informational notes

### N1. v3 M3 (`$7\times 4$` vs `7×6` table dimensionality) is resolved

v3 M3 flagged main.tex L279 saying "the $7\times 4$ predicted-vs-empirical
table" while the actual memo table was 7×6. v4 rewrote the surrounding
sentence (L313–314) to "the per-cell predicted-vs-empirical table and
the full bias accounting appear in~\citep{bbnestructure}" without the
$7\times 4$ descriptor. Closed in v4.

### N2. Abstract "3–10×" disagreement range

Same as v3 N1. Collapse-boundary side is ~10× (predicted 0.028 vs
observed ~0.3 / ring-corrected 0.1); asymmetric-side is ~1.4×
(predicted 0.972 vs observed ~0.7). The "3–10×" range is a reasonable
bracket for the larger discrepancy direction; appears in the source
memo and the prose inherits it. Not a flag, just noted for revisor
awareness.

### N3. Compile produced zero overfull / underfull boxes and zero warnings

The third pdflatex pass log has no `Overfull \hbox`, no `Underfull \hbox`,
and no warning text beyond the expected `rerunfilecheck` outline-bookmark
note (which is harmless and disappears after pass 3). Typography is clean.

### N4. Figure 1 has no in-paper source script

Same as v3 N3. `figures/phase_diagram.png` is a static PNG carried over
from v3 with the caption updated; no `figures/src/` script exists for
it. Staleness check is not applicable.

### N5. Figure 2 source script and PDF are aligned

`figures/recalibrated_heatmap.pdf` mtime 2026-06-11 22:57 sits after the
PPO_v2 sweep's `recalibrated_verdict.json` mtime, and pdfinfo confirms
1-page valid PDF generated by Matplotlib v3.10.9. The figure data was
spot-checked against 5 cells from the source JSON and all match to
5 d.p. Re-runnable; no staleness concern.

## Build / compile summary

```
pdflatex pass 1 -> exit 0 (pre-bibtex; undefined cites expected)
bibtex          -> exit 0
pdflatex pass 2 -> exit 0
pdflatex pass 3 -> exit 0 (no warnings; no overfull/underfull boxes)
Final PDF: 11 pages, 362,005 bytes, letter (612×792 pt)
Unresolved citations: 0
```

## Render-gate documentation (for the next review pass)

- **Page count: 11** — user has opted out of workshop page-budget enforcement.
- **PDF size: 362,005 bytes** — small, no oversized embedded figures.
- **Letter (612×792 pt)** — the `anvil-paper.cls` default.

## Auditor verdict

**Block: no.** No critical flags. Compile is clean and (with the M3
caveat) every load-bearing prose number reconciles to its on-disk
source. Three majors: **M1** (internal-memo cites, a known submission-
time followup), **M2** (11-page count, but user has explicitly opted
out of page-budget enforcement), and **M3** (β-independence stated as
13/13 should be 12/13 on the Nash 39-cell grid). M3 is a one-token
revisor fix and does not change any of the paper's conclusions.

The single most important flag is **M3**: it is the only flag this
audit produced that actually affects a number in the paper's prose,
and it is the only one that wasn't already known going into v4. The
fix is trivial but the claim should not ship at "13/13" when the
source says "12/13."
