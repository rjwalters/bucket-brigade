# Findings — anvil_pub.bb-workshop.11

Cross-section observations from the v11 review pass. The
per-dimension scorecard lives in `scoring.md`; line-level items in
`comments.md`. No rubric version transition applies: the prior review
(`anvil_pub.bb-workshop.10.review/`) was scored against the same
`anvil-pub-v2` rubric (/44, >= 35), so v10's 38 and v11's 41 are
directly comparable.

## Verification log (fresh checks, not changelog trust)

v11's changelog claims a prose-only polish with no statistic moved.
This review verified each claim independently:

- **Full-file diff v10 -> v11 `main.tex`** (2098 -> 2136 lines): the
  complete change set is (1) header comments, (2) the
  `\PassOptionsToPackage{hypertexnames=false}{hyperref}` preamble
  line, (3) the S1 page-budget footnote, (4) four `\citep{}`
  insertions in S2's template paragraph, (5) the Figure 1 caption
  clause swap, (6) the S4 NC-3 skipped-cell wording fix, (7) an
  11-line citation-bearing extension of the S6 exploitability
  paragraph, (8) the rewritten ~165-word conclusion, and (9) four
  `\citep{}` insertions in Appendix A.6. **No numeric token changed
  anywhere in the diff**; the conclusion's remaining numbers (39-cell,
  37-cell, 3-10x, ~80/step, 16x) all match their owning sections.
- **Independent build** (scratchpad copy, shipped `anvil-paper.cls`
  via TEXINPUTS): `pdflatex -> bibtex -> pdflatex x2` (+1 convergence
  pass) all exit 0. Converged log: **28 pages, 0 overfull hboxes, 0
  undefined references/citations, 0 bibtex warnings, 0
  duplicate-destination warnings** (v10 had them; the fix works), 0
  `??` in `pdftotext` output. Residual: two cosmetic `h`->`ht` float
  promotions and the class's pre-existing `OT1/cmr/bx/sc`
  font-substitution notice — matching the changelog's build claims
  exactly. Shipped `main.pdf` is also 28 pages with the corrected
  caption text present in its text layer.
- **Figure 1 caption fix, triple-checked**: the rendered
  `figures/phase_diagram.pdf` has y-axis "$\beta$ (spread prob)"
  (3 rows: 0.1/0.5/0.9) and x-axis "$\kappa$ (extinguish prob)"
  (5 columns: 0.1-0.9); `figures/src/phase_diagram.py` builds
  `BETAS` rows x `KAPPAS` columns; the caption's later
  "$\kappa\in\{0.3,0.7\}$ columns" clause agrees. Bonus cross-check:
  every per-class cell count read off the rendered figure matches
  Table 1's per-kappa empirical distributions (kappa=0.1: 6
  collapse + 3 asymmetric; 0.3: 6 symmetric; 0.5: 6 symmetric + 3
  mixed; 0.7: 6 asymmetric; 0.9: 7 mixed + 2 asymmetric) and the S3
  class totals (n=12/11/10/6), including the lone splitting row
  (kappa=0.9, c=0.5: beta=0.1 mixed, beta in {0.5,0.9} asymmetric).
- **Numeric-consistency pre-check** (step 4c): re-ran
  `anvil.lib.numeric_consistency` with `--write-review`; **0 findings
  over 1055 extracted numbers** (v10: 0 over 1053). Sidecar written
  at `anvil_pub.bb-workshop.11.numeric/`.
- **refs.bib delta**: 11 -> 18 entries, additions exactly the seven
  claimed keys; the four benchmark-key header disclosures and the
  Skyrms 2003-vs-2004 year note are present. Spot-checked entries
  against reviewer domain knowledge: `lanctot2023population`
  (arXiv:2303.03196, Lanctot/Schultz/Burch/Smith/Hennes/Anthony/
  Perolat, 2023), `li2024meta` (arXiv:2405.00243, Li & Wellman,
  2024), `christianos2022pareto` (arXiv:2209.14344, Christianos/
  Papoudakis/Albrecht), `shapley1953stochastic` (PNAS 39(10):
  1095-1100), `diekmann1985volunteer` (JCR 29(4):605-610),
  `ledyard1995public` (Kagel & Roth Handbook, pp. 111-194) — all
  consistent. Live web verification was not possible (network
  sandboxed); the litsearch sibling's resolver provenance is the
  verification of record and `pub-audit` re-checks off-disk sources.
- **NC-3 wording fix**: the S4 clause now says the sweep skipped the
  splitting row's $\beta{=}0.1$ *cell* and the row "enters with two
  \textsf{asymmetric\_only} samples" — consistent with the Figure 1
  splitting-row layout and the 13/13-rows claim (unchanged).

## Cross-section observations

- **The revision discipline is exemplary.** The changelog's
  critic-note -> change map accounts for every v10 finding across all
  four sibling critics (review, audit, numeric, litsearch), with
  explicit fixed/carried/declined dispositions and reasons. The two
  declines (Table 2 path-break; cross-class significance sweep) are
  both defensible and documented.
- **The remaining score ceiling is scientific, not editorial.** Dims
  1/2/5 (5+5+4) are held by exactly three items — the cross-class
  significance sweep, the HF hosting, and the footnote provenance —
  all named in-paper as camera-ready/operator tasks. No further prose
  revision can move the /44 total materially; the next point purchase
  is an experiment.
- **Convergence note**: score trajectory 32 (v9) -> 38 (v10) -> 41
  (v11) on `anvil-pub-v2`, all >= 35 since v10 with zero critical
  flags on the last two passes. The thread is READY pending a v11
  `pub-audit` pass; iteration 11 of max 12.
