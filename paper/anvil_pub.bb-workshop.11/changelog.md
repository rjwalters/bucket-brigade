# changelog: anvil_pub.bb-workshop.10 → .11

## Trigger

v11 is the `pub-revise` polish pass consuming **all four** v10 critic
siblings:

- `anvil_pub.bb-workshop.10.review/` — **38/44, `advance: true`**, no
  critical flags (generic `anvil-pub-v2` rubric; advisory
  `anvil-pub-neurips-v1` overlay 12/16). One major (Figure 1 caption),
  three minors, four nits.
- `anvil_pub.bb-workshop.10.audit/` — **0 critical flags** (both v9 CFs
  cleared at HEAD) + 5 non-critical notes (NC-1…NC-5).
- `anvil_pub.bb-workshop.10.numeric/` — deterministic pre-check, **0
  findings** over 1053 extracted numbers (nothing to consume).
- `anvil_pub.bb-workshop.10.litsearch/` — first litsearch sibling on
  the thread. All four previously-unverified `refs.bib` keys
  (`ppo2017`, `mappo2022`, `openspiel2019`, `psro2017`)
  **resolver-CONFIRMED** with zero corrections (audit NC-1 closes);
  nine resolver-verified candidates delivered in `candidates.bib`.

Note on the convergence gate: the v10 review is `advance: true` with no
critical flags anywhere, so the `pub-revise` step-4 pre-check would
ordinarily exit `READY` with no revision. This pass was explicitly
operator-requested as a polish revision (fix the one major, integrate
the litsearch, tighten dim 9) before submission; no verdict semantics
are overridden — v10 remains the advancing version of record for the
v10 review cycle.

## What did NOT change

No experimental number, statistic, verdict, figure, or table value was
touched in this revision. Both figures and their `figures/src/` scripts
are carried over byte-identical (mtimes preserved). The only
claim-adjacent edits are the two wording-precision fixes (Figure 1
caption axis clause; §4 skipped-cell clause), both of which align prose
with already-verified artifacts. Every dimension that scored ≥5/6 in
the v10 review (rigor 5, evidence 5, clarity 5, citation hygiene 5) had
its supporting text preserved verbatim.

## Critic-note → change map

| Source | Note | Resolution |
|---|---|---|
| bb-workshop.10.review (generic, **major**) + (venue:neurips, minor, presentation) | Figure 1 caption's panel-layout clause is transposed: "rows are $\kappa$ and columns are $\beta$" contradicts the rendered figure and `figures/src/phase_diagram.py` (rows = $\beta$, columns = $\kappa$), the caption's own later "$\kappa\in\{0.3,0.7\}$ columns" clause, and the §3 body text | Caption clause swapped to "rows are $\beta$ and columns are $\kappa$" (main.tex Figure 1 caption). Caption-prose-only fix; the figure, its axis labels, and its generating script were already correct and are untouched. Figure 2's caption (opposite layout) verified not copy-pasted and left alone |
| bb-workshop.10.review (generic, minor, `[related-work]`) | §6 exploitability paragraph rests on two memory-entered citations; litsearch should (i) verify `openspiel2019`/`psro2017`/`ppo2017`/`mappo2022`, (ii) survey the line's empirical follow-ups and other parametric-family constructions, (iii) source the Appendix A named templates | Litsearch ran ((i) all four CONFIRMED, zero corrections; (ii) five candidates + the finding that no closer parametric-family prior exists; (iii) four verified template sources). This revision merges the results: §6 paragraph extended with the line's own later evaluation work — `lanctot2023population` (population-based RPS benchmark: a single fixed game) and `li2024meta` (meta-game evaluation: equilibria estimated empirically, not read off a map) — plus the equilibrium-selection matrix-game tradition via `christianos2022pareto`; each is framed to sharpen the parametric-family differentiator, and the "to our knowledge" scoping sentence now stands on a live search |
| bb-workshop.10.review (generic, minor) | Conclusion restates every result of the abstract at comparable coverage — the largest remaining dim 9 duplication | Conclusion cut from ~340 to ~160 words / four sentences: what was established (scored against the regime map), the concrete open challenge, and the positioning sentence. All statistics remain in their owning sections; the per-cell-metric qualifier on the PPO-ordering claim is retained to avoid overclaim |
| bb-workshop.10.review (generic, minor; = audit NC-4) | §5 footnote: the 300/300 bootstrap sweep and t-interval [304.67, 310.99] trace to the PR #460 judge recomputation via committed paper-trail files, not a regenerable `experiments/` artifact | **Carried** — the honest disclosure footnote is retained verbatim; regenerating the sweep as a committed script + JSON is a camera-ready operator task (it requires re-running the recomputation, out of scope for a prose revision). Flagged for the camera-ready checklist |
| bb-workshop.10.review (generic, minor) + (venue:neurips, minor, reproducibility) | §7 HuggingFace baseline hosting "in progress"; NeurIPS checklist expects it resolved by submission | **Carried** — operator task (complete or remove the hosting promise before submission); the in-paper status statement remains true as written. Unchanged from v9/v10 |
| bb-workshop.10.review (generic, nit) | Header comment says abstract compressed "to ~260 words"; rendered abstract is ~325 | Fixed — the v11 header block records the v10 abstract at ~325 rendered words; the stale ~260 figure is gone |
| bb-workshop.10.review (generic, nit) | Table 2 caption `xurl` breaks the artifact path after "ex-" with no hyphen | **Declined** — per the reviewer's own note this is an acceptable cost of the 0-overfull fix; a shorter alias would trade a nit for provenance opacity. Revisit only if a downstream reader actually misparses |
| bb-workshop.10.review (generic, nit) | Duplicate hyperref destination warnings from the `\tag`-ed (S)/(A)/(C) equations | Fixed — `\PassOptionsToPackage{hypertexnames=false}{hyperref}` before `\documentclass`; v11 build log shows **0** "destination with the same identifier" warnings (v10 had them; all refs/links verified intact) |
| bb-workshop.10.review (generic, nit; verdict priority 3, second half) | 27 rendered pages vs. the BRIEF's 4-page workshop body target; the submission-time page-budget answer owed | Answered in-paper: new §1 footnote states the split explicitly — the 4-page body keeps §1, the §2 contract, §3 boundaries + Fig. 1, §4 ordering/retired-predictor results + Fig. 2, §5 anchor ladder (Table 2), and §6 positioning in compressed form; Appendices A/B, Tables 1 and 3, and the predictor post-mortem detail move to the supplementary appendix outside the body count |
| bb-workshop.10.review (verdict: dims 1–2 held at 5/6) | Cross-class 4×-budget significance sweep gates the remaining rigor/evidence points | **Declined for this revision** — camera-ready experiment, explicitly out of scope per the operator instruction; already named in §4 and §7 as the significance gate |
| bb-workshop.10.audit (NC-1) | Four unverified citations (`ppo2017`, `mappo2022`, `openspiel2019`, `psro2017`); off-disk verification open | **Closed by litsearch** — all four resolver-CONFIRMED against live arXiv metadata (titles, full author lists, years, venues, arXiv IDs 1707.06347 / 2103.01955 / 1908.09453 / 1711.00832). `refs.bib` header comment updated from "not yet auditor-verified" to record the live re-check |
| bb-workshop.10.audit (NC-2) | Six benchmark keys verified only against the secondary `refs/benchmark_comparison.md` evidence index; primary PDFs not on disk | **Carried** — primary-PDF acquisition remains an operator task (litsearch re-confirmed the status is unchanged); `refs.bib` header now discloses the `partial` verdicts explicitly |
| bb-workshop.10.audit (NC-3) | §4 wording: "the PPO sweep skipped the lone splitting Nash row" — strictly the sweep skipped only that row's $\beta{=}0.1$ *cell* (the mixed-verdict one); the row remains with 2 β samples, both asymmetric_only | Fixed in §4: "the PPO sweep skipped the lone splitting Nash row's $\beta{=}0.1$ \emph{cell}---the one carrying its \textsf{mixed} verdict---so the $(\kappa{=}0.9, c{=}0.5)$ row enters with two \textsf{asymmetric\_only} samples". The 13/13-rows quantitative claim (audit-recomputed TRUE) is unchanged |
| bb-workshop.10.audit (NC-4) | Provenance-standard exception disclosed in-paper (footnote); regenerable artifact still absent | **Carried** — same item as the review's §5-footnote minor above; see that row |
| bb-workshop.10.audit (NC-5) | Stale figures: none found | No action needed; v11 carries both figures and `figures/src/` over with mtimes preserved (`cp -p`), so the newer-than-source property is maintained |
| bb-workshop.10.audit (build note) | v10 build clean: 0 overfull, 0 unresolved refs, 27pp | Maintained at v11: `pdflatex + bibtex + pdflatex ×2` all exit 0, **0 overfull hboxes**, **0 undefined citations/references**, 0 `??` in the rendered text, 0 bibtex warnings; 28 pages (the §1 footnote + §6 extension + Appendix A citations add ~1 page net of the conclusion cut) |
| bb-workshop.10.numeric | 0 findings over 1053 extracted numbers | Nothing to consume; no number changed in v11 |
| bb-workshop.10.litsearch (cluster 1) | `lanctot2023population` / `li2024meta` fill the §6 empirical-follow-up gap and strengthen the scoping sentence | Both merged into `refs.bib` (arXiv-preprint house style; identifier-verified fields unchanged) and cited in the §6 paragraph as described above |
| bb-workshop.10.litsearch (cluster 2) | `christianos2022pareto` (equilibrium-selection tradition — should be acknowledged); `krever2025guard` and `papoudakis2020benchmarking` also verified | `christianos2022pareto` merged + cited in §6 (the tradition's modern representative). **`krever2025guard` declined** — litsearch's own assessment: solver-benchmarking with generated instances, "not close"; citing it would dilute §6. **`papoudakis2020benchmarking` declined** — litsearch marks it optional/weakest; §1's framing already carries six benchmark citations, and dim 9 (rhetorical economy, 3/4) argues against list-inflating the related-work section |
| bb-workshop.10.litsearch (cluster 3) | Four resolver-verified Appendix A template sources (`diekmann1985volunteer`, `ledyard19952`, `skyrms2003stag`, `shapley1953stochastic`) — the standing v6 carry-forward | All four merged and cited: Volunteer's Dilemma ¶ → Diekmann 1985; Public Goods ¶ → Ledyard 1995; Stag Hunt ¶ → Skyrms 2003; Stochastic Game ¶ ("every formal result…") → Shapley 1953. Also cited at the §2 body paragraph naming the same templates. The free-rider ¶ needs no new source per the litsearch (Diekmann + Ledyard jointly cover it) |
| bb-workshop.10.litsearch (cosmetics) | `ledyard19952` key inherits the chapter-number "2."; entry types emitted as `@article`; Skyrms year 2003 (Crossref) vs. 2004 (print convention) | Fixed on merge: rekeyed `ledyard1995public`, "2. " title prefix stripped, retyped `@incollection` (Kagel & Roth eds., Princeton UP, pp. 111–194); `skyrms2003stag` retyped `@book` (Cambridge UP); Skyrms year kept at the resolver-verified 2003 (choice documented in `refs.bib`). Identifier-verified fields (author, year, DOI, pages) unchanged, per the litsearch contract |
| bb-workshop.10.litsearch (gap) | Claus & Boutilier 1998 (climbing/penalty games) has no resolvable identifier — a web lead, not a citation; author promotion required | **Not cited** — the litsearch write contract forbids inventing the entry, and no author-supplied BibTeX exists in `<thread>/refs/`. §6 acknowledges the tradition through the resolver-verified `christianos2022pareto` without naming the unverifiable primary. If the author hand-enters a verified entry before camera-ready, the §6 clause can absorb it |
| bb-workshop.10.litsearch (gap) | Primary PDFs still not on disk for any cited paper; verification is metadata-level | **Carried** — same operator item as audit NC-2; see that row |

## Citations delta

- **Added (7)**: `lanctot2023population`, `li2024meta`,
  `christianos2022pareto` (§6); `diekmann1985volunteer`,
  `ledyard1995public`, `skyrms2003stag`, `shapley1953stochastic`
  (§2 + Appendix A.6). All resolver-verified by
  `anvil_pub.bb-workshop.10.litsearch` (provenance table in its
  `notes.md`).
- **Declined (2)**: `krever2025guard`, `papoudakis2020benchmarking`
  (reasons above).
- **Removed (0)**. Total `refs.bib` entries: 11 → 18; every `\cite`
  in `main.tex` resolves (verified: 0 undefined citations, 0 bibtex
  warnings).

## Build verification

`TEXINPUTS=../../.anvil/skills/pub/templates:` then
`pdflatex → bibtex → pdflatex → pdflatex`, all exit 0.
28 pages; **0 overfull hboxes** (of any size); **0 undefined
references/citations**; 0 `??` in `pdftotext` output; 0 bibtex
warnings; 0 duplicate-destination warnings (v10's nit class,
eliminated). Remaining log warnings: two cosmetic `h`→`ht` float moves
and the class's pre-existing `OT1/cmr/bx/sc` font-substitution notice
(present in v10's log as well).
