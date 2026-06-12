# Line-level comments — anvil_pub.bb-workshop.2

Comments are keyed to `main.tex` section headings and grouped by severity
(`blocker` / `major` / `minor` / `nit`). The reviewer found no `blocker`
issues. No critical flags raised.

Severity legend:
- **major** — actionable revision the reviewer would expect before the
  camera-ready (or before re-submission to a stricter venue).
- **minor** — defensible-as-is but worth addressing for polish.
- **nit** — typographic / wording / cosmetic.

---

## Abstract (lines 24–53)

- **minor — Number sourcing visible.** The abstract reports the row
  means (`0.326`, `0.106`, "near-zero") and attributes them to the
  metric `gap_closed_ne`. A reader who reaches Figure 2 will see
  cell-level values `0.334 / 0.317` (symmetric row) and
  `0.162 / 0.051` (asymmetric row) and may briefly wonder if the
  abstract's `0.326` is a cell or a row mean. Consider one extra word:
  "is largest on the symmetric-only row (mean $0.326$)". Five
  characters; clarifies the abstraction level.
- **minor — "3--10$\times$" without a unit.** Line 46 reads "disagree
  with the equilibrium solver by 3--10$\times$". The §3 prose at
  line 244 carries the same number but adds "$\kappa$-threshold"
  framing. The abstract version reads slightly ambiguously (3–10× of
  what?); a parenthetical like "in the $\kappa$-threshold location"
  would tighten it.
- **nit — Long opening sentence.** The first sentence (lines 25–30)
  packs six canonical benchmarks, a definitional clause, a verdict, and
  a hypothetical Overcooked example into one 60-word sentence. Splitting
  after "Pareto-different one." would improve scan-ability.

## §1 Introduction (lines 55–103)

- **major — Contributions paragraph could lead with the methodological
  inversion.** The Contributions list (lines 87–100) ordering is:
  (1) env spec, (2) closed-form boundaries, (3) protocol + metric +
  results, (4) honest positioning. The novel-to-v2 contribution — that
  per-cell baselines invert the cross-cell PPO ordering, making the
  per-cell calibration itself a finding — is folded into (3) as "per-cell
  NE-anchored metric." A reviewer skimming the contributions list will
  not see the methodological inversion as a standalone contribution. The
  Conclusion (line 547) does call it out as a separate contribution.
  Consider splitting (3) into 3a (protocol + first empirical results) and
  3b (per-cell baseline methodology — independently applicable to other
  cross-cell PPO sweeps). At the current weighting, the v2's distinctive
  contribution is buried in a sub-clause.
- **minor — "right" in scare quotes.** Line 30 and again line 69 use
  "the `right' joint policy" and "the `right' equilibrium" in
  scare-quote LaTeX. The abstract uses it once; §1 uses it twice. A
  workshop reviewer is fine with "the analytically-characterised NE" as
  a more direct phrasing.
- **minor — One-sentence transitional paragraph.** Lines 102–103
  ("We treat the analytical NE characterization as ground truth and
  the PPO trainability sweep as a falsification test, in that order.")
  is doing real work but is structurally an orphan after the
  Contributions list. Consider moving it as the closing sentence of
  Contributions or removing it. The 4-page budget rewards tight
  transitions.

## §2 Environment (lines 105–166)

- **minor — Phase-order load-bearing claim could be sharper.** Lines
  138–143 explain that the spread phase is inert under the canonical
  order because phase 4 ruins burning houses before phase 5 fires. This
  is correct per `env_spec.md` §3.4 and is the load-bearing reason for
  the §3 β-independence. The text is right but a sophisticated reader
  may still need to pause to confirm. Consider a parenthetical pointer:
  "(this is the source of the §3 β-independence prediction)".
- **nit — Notation `$h_h$`.** Line 110 introduces `h_h\in\{\textsc{Safe},\textsc{Burning},\textsc{Ruined}\}`,
  where the inner `h` is the house index and the outer `h` is the
  status. In `env_spec.md` §2.1 this is `h_t,h` with a time subscript.
  The paper's notation drops the time index for compactness, which is
  fine, but the doubled-`h` reads as a typo on first encounter.
  Consider `\sigma_h` or `s_h` for the status to disambiguate.

## §3 Equilibrium structure (lines 167–300)

- **major — The 3–10× threshold gap is now defended; consider a
  sentence on the implicit reader-test.** §3 closes (lines 270–283)
  with the "Headline finding" paragraph that calls the qualitative
  ordering, β-independence, and collapse-regime existence as the
  paper's claims. This is the right framing. But a skeptical reviewer
  may still wonder: "if the closed-form thresholds are wrong by 10×,
  what is the value of the closed-form derivation?" The implicit
  answer — that the closed form predicts *which* axis matters
  ($\kappa$, not $\beta$) — is in the prose but not made directly. A
  one-sentence defense like "The value of the closed-form derivation
  is not the quantitative thresholds but the structural prediction
  about which parameter axis controls the phase transition; the
  quantitative residual identifies where the ring-locality and
  per-agent-ownership corrections must enter" would close the loop
  for a doubting reader.
- **minor — Eq. (A) intersection issue.** Lines 200–214 derive the
  asymmetric NE conditions: lone-Worker prefers Work AND each free-rider
  prefers Rest. The two inequalities at $\tilde A=36.24, c_{gap}=1.0$
  give $\kappa \geq 0.0276$ AND $\kappa \notin (0.028, 0.972)$, an
  intersection that is empty for the lower bound (off by 0.0004 in the
  strict inequality) and gives the high-$\kappa$ regime
  $\kappa \gtrsim 0.972$. The paper at line 230 reports
  "$\kappa\gtrsim 0.972$" cleanly but does not mention the
  "lower interval is empty" detail that `ne_structure.md` §3.2 calls
  out. This is fine for the workshop pitch — the empty interval is a
  numerical artefact of the strict inequality — but a careful reader
  re-deriving will notice the omission. Not worth elevating, but flag
  for the camera-ready.
- **minor — Predicted-vs-observed paragraph: where is the
  $7\times 4$ table?** Line 282 says "the $7\times 4$ predicted-vs-empirical
  table" appears in `\citep{bbnestructure}`. The reader cannot see
  this table without reaching outside the paper. For a workshop pitch
  this is acceptable, but for a publication the reader will expect at
  least a 7-row truth-table inline. Consider promoting the 7-row table
  from `ne_structure.md` §4 into the paper at a page cost, or naming
  one or two of the disagreement cells inline (e.g., "the
  $\beta{=}0.5,\kappa{=}0.1$ cell is empirically `no_convergence` but
  the analytical reduction predicts `symmetric_only`").

## §4 Trainability (lines 302–399)

- **major — Within-class sample size is the chief evidence concern.**
  The Caveats paragraph (lines 369–381) is honest about $n{=}2$ cells
  per symmetric/asymmetric class and $n{=}3$ for collapse. But the
  ordering claim
  (`symmetric (0.326) > asymmetric (0.106) > collapse (-0.033)`) rests
  on the class means. The within-class std on the underlying
  `gap_closed_ne` values per cell is ~0.33 (Figure 2 shows
  $\pm 0.331$ and $\pm 0.344$ at the symmetric cells), which is larger
  than the symmetric-vs-asymmetric class-mean separation (0.220). A
  classical statistical test on these data would not reject the null
  of "all three classes have the same mean." The paper's claim is
  ordering rather than statistical significance, and this is honest
  given the disclosed sample size — but the workshop reviewer may
  still ask. Consider naming explicitly in §4: "We report the
  ordering, not significance; the 4-seed-per-cell variance is larger
  than the class separations, so the ordering is consistent with the
  prediction but does not reject the null. The 75-cell sweep is the
  significance test." This would head off the most-likely reviewer
  objection.
- **major — β-independence claim at the PPO layer is overclaimed by
  one degree.** Lines 346–350 say the 0.017 cell-to-cell difference
  at κ=0.5 is "well within seed variance, confirming the
  β-independence prediction." The data does not "confirm" — it is
  *consistent with*, given two data points and std ~0.33. The
  language "consistent with the β-independence prediction" (or
  "does not reject the β-independence prediction at the PPO layer")
  is the calibrated phrasing. The current "confirming" reads stronger
  than the evidence supports. A one-word edit ("consistent with" or
  "is not in tension with") fixes it.
- **major — Compute disclosure is workshop-acceptable but minimally
  so.** Line 311: "single 16-core host in approximately two hours."
  The compute platform host is named in `changelog.md` as `alc-2`
  (the alcubierre cluster). The paper omits the host model, the
  exact CPU model, and the memory footprint. NeurIPS overlay
  dim 5 (Reproducibility) calibration says "supplied or sufficiently
  described that an independent group could replicate." An
  independent replicator needs to know whether "two hours" extends to
  ten on a workstation. Consider naming the CPU model (or just saying
  "modern x86 server class with $\geq 32$ GiB RAM"). Not worth
  blocking for, but the reviewer flagged it as the operator
  requested.
- **minor — Methodological-observation paragraph framing.** Lines
  358–367 carry the "the original single-cell baseline inverted the
  ordering" paragraph. This is correctly visible as a paragraph (not a
  footnote), correctly framed as a methodological finding (per
  changelog.md line 22). The one tightening: the paragraph's first
  sentence reads "A methodological observation worth recording" — a
  workshop reviewer will respect a more direct framing like "A
  methodological observation we report as a separate contribution"
  (which is the framing §4's prior paragraph at lines 335–338 already
  uses, but the observation paragraph itself softens). Small
  consistency fix.
- **minor — `gap_closed_homogeneous` vs `gap_closed_ne` distinction
  is clear in prose but worth one more sentence on metric meaning.**
  Lines 377–381 say the `gap_closed_homogeneous` metric gives
  "quantitatively smaller cross-cell separation than the NE-anchored
  metric, but yields the same qualitative ordering on the 7-cell
  preview." A reader who has not opened
  `recalibrated_verdict.md` does not know what either metric *is* as
  a single number for the collapse row. The `gap_closed_homogeneous`
  is per-cell Random → SpecialistPolicy×4, which the paper does say at
  line 378, but the implicit "the Hero-only baseline is the right
  reference for cells where no NE asymmetric profile exists" thought is
  not stated. One sentence: "On no-pure-NE cells, the
  `gap_closed_homogeneous` is the natural fallback because the
  SpecialistPolicy×4 baseline is the strongest available stationary
  policy when no NE profile is computable" would close the loop.
- **nit — `\texttt{gap\_closed\_ne}` vs `\texttt{gap\_closed\_homogeneous}`
  capitalization.** The §4 prose and Figure 2 caption use the
  underscore form consistently. The recalibrated_verdict.md uses the
  same form. Consistency is fine; this is a non-finding.

## §5 Related work (lines 401–458)

- **minor — Three back-to-back "we do not claim" disclaims.** Lines
  453–456 read:
  > "We do not claim Bucket Brigade scales to large populations; we
  > do not claim it tests `emergent cooperation' in any general
  > sense; we do not claim it is a general-purpose MARL benchmark.
  > It is none of these things."
  The triple negation hits the forbidden-phrases compliance bar
  exactly as the BRIEF requires, and the "It is none of these things"
  punch line lands. But three "we do not claim"s in a row is
  rhetorically heavy. A reviewer with the BRIEF in hand will appreciate
  the precision; a reviewer without it may read this as overstated
  modesty. Consider:
  > "Bucket Brigade does not scale to large populations, does not
  > test emergent cooperation in any general sense, and is not a
  > general-purpose MARL benchmark."
  Same compliance, less repetition.
- **minor — Table 1 NE column.** The "NE characterisable?" column
  values are "No (any Pareto-opt.)" / "No (intractable)" /
  "Partial (2P only)" / "No" / "No (population)" / "No" /
  "**Yes (closed-form)**". The "**Yes (closed-form)**" sells well, but
  a strict reviewer will note that §3 itself reports the closed-form
  thresholds are off by 3–10× — so "closed-form" is technically true
  (the inequalities are closed-form) but the table's binary framing
  glosses the §3 honesty. A footnote on the table row or a slight
  hedge like "Yes (closed-form, qualitative)" would reconcile the two
  sections. Minor because the §3 discussion is already in the paper;
  Table 1 is a reasonable summary.

## §6 Discussion and limitations (lines 460–532)

- **major — "Intentionally small and artificial" paragraph (a)/(b)/(c)
  defense is in place and well-written.** This is a positive
  observation, not a request to change. The paragraph reads as the
  BRIEF requires (verified at lines 463–488).
- **minor — Threats-to-analytical and threats-to-empirical paragraphs
  have content overlap.** Both paragraphs (lines 490–503 and 505–515)
  end with "the full 75-cell sweep is the gate" (or similar). The
  redundancy is rhetorical-economy cost in a paper at 3705 words. One
  paragraph titled "Threats to the contribution" with sub-bullets for
  analytical and empirical would save ~80 words.
- **minor — HuggingFace baselines status.** Lines 528–531: "Frozen
  baseline checkpoints ... will be hosted on the HuggingFace Hub; the
  publication pathway for the baselines is in progress and not yet
  complete at the time of this draft." This is honest disclosure. The
  NeurIPS reproducibility overlay (`unverified_reproducibility_claim`
  flag) would care about this — but the language used is
  "in progress" rather than "available", so the disclosure is
  compliant. The auditor should re-check this language before
  submission to confirm the baselines have landed or the language
  still reads as honest.
- **nit — `\#360-sweep` styling.** Line 526 has `\#360-sweep checkpoints`
  which renders as "#360-sweep" — fine, but a workshop reviewer
  unfamiliar with the project's GitHub will see "#360" and not know it
  is a GitHub issue. A `\cite{}` to a project tracker is excessive;
  consider "the #360 PPO sweep checkpoints (see the project tracker
  on GitHub for issue numbers)" or simply drop the issue number.

## Conclusion (lines 534–551)

- **minor — Conclusion is tight but does not name the methodological
  contribution as a contribution.** The Conclusion at line 547 names
  "the per-cell baseline correction itself is a methodological
  contribution applicable to cross-cell PPO comparison more broadly."
  This is good. It echoes §4. But the Conclusion's "(1)/(2)/(3)/(4)"
  contributions enumeration from §1 is not mirrored here, so a reader
  who jumps directly to the Conclusion does not get the four-item
  enumeration. Workshop papers commonly do not re-list contributions
  in the Conclusion; this is OK as-is.

## Figures

### Figure 1 — `figures/phase_diagram.png` (NE phase diagram)

- **minor — Caption density is high but readable.** The caption
  (lines 288–299) packs the parameter cell, the heuristic-Double-Oracle
  solver, three regime names, the β-independence observation, and the
  κ-threshold disagreement into ~125 words. Self-contained per the
  rubric. Could be split into two sentences for readability.

### Figure 2 — `figures/recalibrated_heatmap.tex` (PPO heatmap)

- **major — Std reported as text labels rather than visual
  indicators.** Each cell shows the mean (e.g., $0.334$) and the std
  (e.g., $\pm 0.331$) as two text lines. The std is roughly equal to
  the mean for the symmetric cells, and a reader at-a-glance will see
  the colored boxes and the bold means but may miss that the
  uncertainty is comparable to the cell-to-cell difference. A bar-style
  error indicator (Tufte-style sparkline or a small CI bar inside the
  box) would communicate the signal-vs-noise ratio more directly. The
  figure header comment names a planned matplotlib replacement; the
  reviewer endorses doing that replacement for the camera-ready and
  using it to visualize the uncertainty.
- **minor — Hatch + dagger marker for the metric switch is
  appropriate.** The κ=0.1 row hatching (`pattern=north east lines`)
  plus the $\dagger$ marker on the std values communicates the metric
  switch (`gap_closed_ne` → `gap_closed_homogeneous`) cleanly. The
  caption (lines 386–397) explains both signals. This is well-handled.
- **minor — β=0.1 column "n/a" cells are correctly empty.** The
  caption says "The $\beta{=}0.1$ column was not sampled in the
  7-cell preview." The TikZ correctly renders these as gray-shaded
  "n/a" cells. This is the right disclosure shape; the alternative
  (omitting the column entirely) would lose the 3×3 grid context.

## Bibliography (`refs.bib`)

- **major — Two internal-memo cites still need arXiv replacement.**
  `bbenvspec` and `bbnestructure` are `@misc` entries pointing to
  in-repo Anvil memo directories. Both carry `note` fields flagging
  the planned arXiv-preprint replacement. The `pub-audit` phase should
  confirm whether the env-spec and ne-structure memos have been (or
  will be) preprinted before submission. If they have not, the
  workshop reviewer will see "internal memo" citations, which is a
  weak signal at NeurIPS. The substitution is straightforward (replace
  the `@misc` with an `@article` / `@misc` carrying the arXiv ID once
  the preprint is up); the reviewer flags the work item.
- **minor — All 11 `\cite{}` keys resolve.** Verified at the source
  level: every key used in `main.tex` matches an entry in `refs.bib`.
  No `[??]`-resolving cite errors at compile time.
- **nit — `refs.bib` carries a pruned-entries comment.** Lines 87–89
  note that `grf2020` and `hideseek2020` were pruned from v1. The
  comment is fine; the reviewer flags it only because a reviewer
  scanning the bib may briefly wonder why those entries appear in the
  comment but not the body.

---

## Tagged for `pub-litsearch` (related-work re-run)

None. The §5 + Table 1 positioning is solid against the six surveyed
benchmarks, no obvious close-prior-work omission was identified, and
the BRIEF's voice-norms wording is compliant. If a litsearch sibling
is ever run for this thread, it would target broader
"NE-transparent MARL benchmark" coverage — but the current set is
defensible for a workshop pitch.

## Tagged for `pub-audit`

The reviewer recommends the auditor focus on these in addition to the
standard pipeline:

1. **Render-gate**: compile `paper.pdf` and run
   `anvil/lib/render_gate.py::gate(...)` against the PDF + log + source.
   Specifically check page count vs. 4-page workshop cap and any
   overfull boxes in §3 (the equation-dense section).
2. **Number reconciliation**: confirm `0.326`, `0.106`, `-0.033` in §4
   prose match `recalibrated_verdict.md` row means (they do per
   reviewer spot-check, but audit should re-derive). Also confirm
   Figure 2 cell values reconcile against the recalibrated table.
3. **Citation claim support**: per-`\cite{}` audit on §4 (PPO/MAPPO
   references) and §5 (six benchmark references) to confirm each cite
   actually supports the surrounding claim.
4. **Internal-memo cites**: re-evaluate whether `bbenvspec` and
   `bbnestructure` should be replaced with arXiv preprints before the
   audit's `AUDITED` verdict.
