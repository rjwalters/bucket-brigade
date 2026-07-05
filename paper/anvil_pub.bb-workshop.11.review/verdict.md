# Verdict — anvil_pub.bb-workshop.11

- **Total score**: **41 / 44**
- **Decision**: `advance: true` (41 >= 35 threshold; no critical flags)
- **Rubric**: `anvil-pub-v2` (9 dimensions, /44, advance >= 35;
  on-disk `.anvil/skills/pub/rubric.md`).
- **Critical flags**: **none.**

## Critical flags

None. Specifically adjudicated and NOT raised:

- **Numerical inconsistency** — not raised. v11 is a prose-only polish
  revision of the advancing v10, whose every headline statistic the
  v10 review recomputed from the committed artifacts at repository
  HEAD. This review verified by full-file diff (v10 -> v11
  `main.tex`) that no experimental number, statistic, figure, or
  table value moved: the entire change set is the Figure 1 caption
  clause swap, the §4 NC-3 wording-precision fix, the §6
  citation-bearing extension, the §2/Appendix A template citations,
  the §1 page-budget footnote, the rewritten (shorter) conclusion,
  the `hypertexnames` preamble line, and header-comment updates.
  Both figures and their `figures/src/` scripts are byte-identical
  carry-overs. The fresh deterministic numeric-consistency pass
  (`anvil_pub.bb-workshop.11.numeric/`) found 0 findings over 1055
  extracted numbers. The rendered Figure 1 was additionally
  cross-checked cell-by-cell against Table 1's per-$\kappa$ verdict
  distributions (all five rows match: 6+3 / 6 / 6+3 / 6 / 7+2).
- **Build / compile failure** — not raised. This review's independent
  `pdflatex + bibtex + pdflatex x2` build against the shipped
  `anvil-paper.cls` converges at 28 pages with 0 overfull hboxes of
  any size, 0 undefined references/citations, 0 `??` in the
  `pdftotext` layer, 0 bibtex warnings, and 0 duplicate-destination
  warnings (the v10 nit class, verified eliminated by
  `hypertexnames=false`). (The render-gate pre-flight proper was
  skipped fail-open: no `paper.pdf` / `compile-log.txt` from a v11
  `pub-audit` exists yet, so no `_gate.json` is emitted; run
  `pub-audit` next.)
- **Caption/figure contradiction (the v10 major's class)** — not
  raised, and the v10 major itself is verified fixed: the Figure 1
  caption now reads "rows are $\beta$ and columns are $\kappa$",
  matching the rendered figure's axis labels, the generating script's
  `BETAS`-rows x `KAPPAS`-columns layout, and the caption's own later
  clauses. Figure 2's opposite-layout caption was checked and remains
  correct.
- **Citation error** — not raised. All 18 `refs.bib` entries resolve
  (0 undefined citations); the seven v11 additions are
  resolver-verified per the litsearch sibling with bibliographic
  fields consistent with this reviewer's domain knowledge (arXiv IDs,
  Shapley PNAS 39(10):1095–1100, Diekmann JCR 29(4):605–610, Ledyard
  Handbook pp. 111–194), and each is cited in a context its paper
  actually supports at review-level reading. Per-citation
  claim-support verification remains `pub-audit`'s job.
- **Close prior work ignored** — not raised. The litsearch sibling's
  live search found no closer parametric-family prior; the §6
  engagement now covers the exploitability line's own follow-ups and
  the equilibrium-selection tradition. Two low-severity
  `related-work` leads are recorded in `comments.md` (the deliberately
  uncited Claus & Boutilier primary; a Galla–Farmer-style
  learning-dynamics-phase-diagram recall lead), neither of which
  threatens the paper's precisely-scoped novelty claim.

## Verification of the v10 review's top revision priorities

1. **Fix the transposed Figure 1 caption clause** — done and verified
   against the rendered PDF, the generating script, and Table 1
   (dim 6: 3 -> 4).
2. **Re-run `pub-litsearch`** — done (`anvil_pub.bb-workshop.10.litsearch/`):
   all four flagged keys resolver-CONFIRMED with zero corrections;
   seven verified candidates merged and cited (§6 x3, §2/Appendix A
   x4); two candidates declined with documented reasons (dim 4:
   4 -> 5; dim 8 additions clean).
3. **Tighten the conclusion + answer the page-budget question** —
   done: conclusion cut ~340 -> ~165 words with no statistic
   restated outside its owning section, and the §1 footnote states
   the 4-page-body vs. supplementary-appendix split explicitly
   (dim 9: 3 -> 4).

All eight v10 line-level comments are accounted for in v11's
changelog: 5 fixed, 2 carried as explicit operator tasks (HF hosting;
§5 footnote provenance), 1 declined with a documented reason
(Table 2 path-break). The carries are re-recorded in `comments.md`.

## Dimension summary

| # | Dimension | Weight | Score | v10 -> v11 move |
|---|---|---:|---:|---|
| 1 | Rigor of method / argument | 6 | 5 | unchanged (structural power limit; camera-ready gate) |
| 2 | Evidence sufficiency | 6 | 5 | unchanged (ordering-only headline remains) |
| 3 | Clarity of contribution | 5 | 5 | unchanged |
| 4 | Related-work positioning | 5 | 5 | +1 (litsearch merged; substrate-backed engagement) |
| 5 | Reproducibility | 5 | 4 | unchanged (HF hosting + footnote provenance carried) |
| 6 | Figure & table quality | 4 | 4 | +1 (transposed caption fixed, triple-verified) |
| 7 | Prose & structural quality | 4 | 4 | unchanged (28pp clean build; log nit eliminated) |
| 8 | Citation hygiene | 5 | 5 | unchanged (11 -> 18 entries, all resolver-backed) |
| 9 | Rhetorical economy | 4 | 4 | +1 (conclusion cut; page budget answered in-paper) |
| | **Total** | **44** | **41** | **+3 vs v10's 38** |

## Venue overlay (advisory)

Advisory venue overlay scored **13/16** against `anvil-pub-neurips-v1`
(soundness 3/4, presentation 2/2, contribution 3/4, novelty 3/3,
reproducibility 2/3); see `_review.venue.json` for findings. The
presentation point recovered with the Figure 1 caption fix. The
overlay does NOT affect the /44 convergence gate.

## Top items for camera-ready (advisory — paper advances)

The paper advances; `pub-audit` should run on v11 next. The remaining
items are the standing operator tasks, unchanged in kind from v10:

1. **Cross-class 4x-budget significance sweep** — the named gate on
   the last rigor/evidence points (dims 1–2 held at 5/6).
2. **Resolve the HuggingFace baselines hosting promise** and
   regenerate the 300/300 bootstrap sweep as a committed
   `experiments/` artifact (dim 5 held at 4/5).
3. **Produce the actual 4-page workshop body** the §1 footnote now
   specifies — the footnote answers the policy; the compressed
   artifact itself is the submission-time deliverable.
