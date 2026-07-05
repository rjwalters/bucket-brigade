# Audit flags for anvil_pub.bb-workshop.10

## Critical flags (block advancement to AUDITED)

None. **0 critical flags.**

- **CF-1 (v9: artifact supersession) — CLEARED.** Every §4/abstract-adjacent
  statistic was independently recomputed from the artifacts at repository
  HEAD (`bf26f818`; both artifacts unchanged since `a5b8ccdc`): class means
  0.142/0.096/0.060/−0.024, separations 0.045/0.036, homogeneous means
  0.041/0.046/0.030/−0.024 with the sym/mixed inversion now stated (and the
  old robustness claim withdrawn in-text), entropy headline ρ=0.109 (p=0.56,
  31 cells), min-vs-homog p=0.038 vs Bonferroni α=0.0063, β-column example
  −0.32/+0.01/−0.00 with range up to 0.34, buy-down ratios 2.00–3.64×
  (median 2.65) / 1.84–3.59×, 0/8 flips, 0.372→0.043, 4×-budget −0.108 from
  −0.024, per-cell std 0.99 / median 0.24. Figure 2's rendered values were
  extracted from the shipped PDF and match the HEAD JSON cell-for-cell at
  2-dp (37/37, incl. the κ=0.3, c=1.0, β=0.5 = +0.04 spot-check). The pinned
  provenance ρ=0.007 (p=0.97) was verified to exist at git revision
  `22b1fda6` of `entropy_vs_trainability.json` exactly as cited. The §6
  "match the artifacts at repository HEAD" sentence is now literally true.
- **CF-2 (v9: Table 1 caption/descriptor contradiction) — CLEARED.** All
  three sites now read κ∈{0.3,0.7}; the Table 1 caption ("row totals are 9
  except at κ∈{0.3,0.7} (6 cells each…)") matches the table body and the
  recomputed `results.json` per-κ totals 9/6/9/6/9; the six unsampled cells
  recompute to exactly (β∈{0.1,0.5,0.9})×(κ∈{0.3,0.7})×c=0.5; no
  "high-κ corner"/"subsumed" language remains in main.tex.

## Non-critical notes

- **NC-1 — Unverified citations (4)**: `ppo2017`, `mappo2022`, and the two
  NEW v10 entries `openspiel2019`, `psro2017` have no source material in
  `<thread>/refs/`; claim support could not be verified on disk. The two new
  entries were specifically re-checked for internal consistency (the reviser
  flagged them for audit verification): titles, author lists, years, venues,
  and arXiv IDs (1908.09453; 1711.00832) all match the canonical papers per
  the auditor's knowledge, `openspiel2019` is correctly cited as an arXiv
  preprint, and `psro2017` is exactly the paper that introduced PSRO/NashConv
  — the §6 credit attribution is accurate. Neither looks fabricated.
  Off-disk (primary-PDF / web) verification remains open for all four —
  a `pub-litsearch` work item, as the changelog itself notes.
- **NC-2 — Secondary-source-only verification (6 benchmark keys, carried
  from v9)**: the Overcooked / Melting Pot / Hanabi / SMAC+SMACv2 / MAgent /
  PettingZoo claims and Table 3 rows verify point-for-point against the
  author-supplied `refs/benchmark_comparison.md` evidence index, but primary
  PDFs are not on disk — verdicts recorded as `partial`.
- **NC-3 — Wording imprecision, §4 Results (new, minor)**: "($\beta$-invariance
  holds …) the PPO sweep skipped the lone splitting Nash row,
  $(\kappa{=}0.9, c{=}0.5)$" — strictly, the sweep skipped only that row's
  $\beta{=}0.1$ *cell* (the one carrying the \textsf{mixed} verdict); the
  row itself remains in the PPO subgrid with 2 $\beta$ samples, both
  \textsf{asymmetric\_only}. The quantitative claim (13/13 rows
  verdict-identical) is recomputed TRUE, so this is a one-word fix
  ("skipped the … row's splitting cell"), not a numerical error.
- **NC-4 — Provenance-standard exception, now disclosed (carried from v9
  NC-5, downgraded)**: the 300/300 bootstrap-combination sweep and the
  t-interval [304.67, 310.99] still trace to the PR #460 judge recomputation
  via the committed v8 paper trail rather than a regenerable
  `experiments/` artifact, but v10 adds an explicit in-paper footnote saying
  exactly that, which is the disclosure this audit asked for. Both numbers
  re-verified against `paper/anvil_pub.bb-workshop.8/{changelog.md,
  stale_claims_audit.md}`.
- **NC-5 — Stale figures**: none. Both rendered figures are at least as new
  as their `figures/src/` scripts, and Figure 2's content was verified
  against its declared data source at HEAD (see numerical-audit.md).
- **Build**: clean. `pdflatex` + `bibtex` + 2×`pdflatex` all exit 0; 27
  pages; **0 unresolved citations/cross-references** (no `??` in the
  rendered text); **0 overfull hboxes** (v9 had 10, five severe — the v10
  fix is confirmed in the log); remaining warnings are two cosmetic
  `h`→`ht` float moves.

## Verdict

**0 critical flags.** Both v9 critical flags are cleared at HEAD, the two
new citations are internally consistent, and the full citation + numerical
+ build re-audit found no inconsistencies. Per the pub-audit state machine,
a zero-critical-flag audit alongside a `READY` version makes the thread
**AUDITED** (terminal). Note for the orchestrator: v10 carries no `.review/`
sibling of its own at audit time (the v9 review scored 32/44,
`advance: false`, 0 critical flags); this audit clears the audit gate —
whether v10 is `READY` for the AUDITED derivation is the reviewer/orchestrator's
call, not this critic's.
