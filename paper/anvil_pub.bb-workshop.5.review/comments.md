# Line-level comments — anvil_pub.bb-workshop.5

Comments are keyed to `main.tex` section headings and grouped by severity
(`blocker` / `major` / `minor` / `nit`). No `blocker` issues found. No
critical flags raised.

Severity legend:

- **major** — actionable revision the reviewer would expect before the
  camera-ready (or before re-submission to a stricter venue).
- **minor** — defensible-as-is but worth addressing for polish.
- **nit** — typographic / wording / cosmetic.

---

## Carry-forward status from prior reviews

- **v2 Major: Contributions paragraph buries per-cell-baseline finding.**
  **Resolved in v5.** §1 L122–135 now splits (3) into (3a) protocol +
  results and (3b) per-cell metric as standalone methodological
  contribution, with the single-cell-baseline-inverts-the-ordering
  finding named in the (3b) sentence directly.
- **v2 Major: "confirming" overclaim on β-independence at PPO layer.**
  **Resolved in v5.** Abstract / §3 / §4 calibrated to "consistent
  with" / "is consistent with" throughout. §4 L471–476 adds explicit
  ordering-not-significance prose with per-class std (0.21–0.44) vs.
  class-mean separations (0.073, 0.048, 0.108).
- **v2 Major: Within-class sample size weak.** **Resolved-in-prose** by
  the §4 ordering-not-significance sentence and the named
  "cross-class 4×-budget sweep is the natural significance gate"
  follow-up. The sweep itself is not run at v5 cut, and v5 owns this.
- **v2 Major: Threats-to-analytical and Threats-to-empirical paragraphs
  overlap.** **Resolved in v5** via §6 merge to a single
  "Threats to the contributions" paragraph (L643–662).
- **v2 Major: At least a small inline truth table in §3.** **Resolved
  in v5** via the new Table 1 at the full 39-cell scale (an upgrade on
  the 7-cell preview the v2 reviewer originally asked for).
- **v2 Major: Compute disclosure thin.** Carried-forward unchanged.
  §4 L429–430 says "single 16-core host (`alc-2`) in approximately six
  hours" without CPU model or memory footprint. Not flagged as new
  here; minor.
- **v4 audit M1 (internal-memo cites).** **Resolved in v5** (`bbenvspec`
  and `bbnestructure` deleted; content inlined as Appendices A and B).
- **v4 audit M2 (11-page PDF).** Not applicable — v5 PDF is 20 pages
  (appendix inlining); user has opted out of workshop page-budget
  enforcement.
- **v4 audit M3 (β-independence 13/13 → 12/13).** **Resolved in v5**
  with the corrected count, named splitting row, and PPO-side
  reconciliation prose. §3 L282–286, §4 L477–478, and Fig 1 caption
  L408–411 all consistent.

---

## Abstract (L43–78)

- **minor — Two numerical formats for the same fact.** L64–68 reports
  the four class means (0.180 / 0.107 / 0.059 / -0.049) and the
  4×-budget worsening (-0.108) in compact in-line LaTeX. L66 mentions
  "$9/37$ PPO cells" but the body §4 L463 reports $n=9$ mixed cells
  with cardinality 9 not 9/37 — the abstract should either say
  "$9$ of $37$ PPO cells" or just $n=9$ to avoid the ambiguous slash
  notation that reads like a fraction.
- **nit — "is the only published parametric MARL game".** L77 ends with
  "the only published parametric MARL game where the question \ldots
  admits a ground-truth answer." This is a strong superlative for a
  workshop pitch; while not flagged as overclaim (the paper has
  earned the parametric / NE-characterisable axis at the §5 Table 2
  comparison), a soft "the first published" or "the first we know of"
  would absorb any reviewer challenge about unknown obscure prior
  benchmarks.

## §1 Introduction (L80–141)

- **minor — Long Contributions paragraph at 26 lines.** L115–141 packs
  five contribution items (1, 2, 3a, 3b, 4) plus the
  "ground-truth-and-falsification-test" framing sentence. The (3a)
  and (3b) split is the right v5 move per v2 reviewer feedback, but
  the resulting paragraph is dense. Consider breaking into a
  bulleted `\begin{enumerate}` list — the LaTeX class supports it
  and a workshop reviewer can scan the contributions in 15 seconds
  instead of 60.

## §2 Environment (L143–203)

- **minor — "Formal proofs and a complete notation table appear in
  Appendix A".** L201 promises "formal proofs" in Appendix A. Appendix
  A §A.6 contains informal-attribution descriptions of Volunteer's
  Dilemma / Public Goods / Stag Hunt / free-rider as named limits —
  these are not formal proofs in the strict mathematical sense (no
  `\begin{proof}` blocks, no formal lemma statements). The
  appendix is good content but the body's "formal proofs" promise
  oversells what the appendix delivers. Soften L201 to "Formal
  specification and a complete notation table appear in Appendix A."

## §3 Equilibrium structure (L205–415)

- **major — Body promise of "per-cell predicted-vs-empirical table" in
  Appendix B is unfulfilled.** L389 reads: "The per-cell
  predicted-vs-empirical table and the full bias accounting appear in
  Appendix B." Appendix B carries Table 4 (predicted-NE per-κ-band
  summary, 5 rows, no empirical comparison) and Table 5 (7-cell
  preview predicted-vs-empirical, 7 rows at c=0.5 only). Neither is a
  "per-cell" comparison across the full 39-cell grid — that view
  exists only at the per-κ aggregate in body Table 1 (5 rows). Either
  add a 39-row table to §B.4 (the §B.4 subsection is currently empty —
  see prose comment below) or soften L389 to "The 7-cell preview
  comparison and the full bias accounting appear in Appendix B."
- **minor — Table 1 column "Empirical distribution" entries use
  inconsistent abbreviation.** Table 1 (L308–348) Empirical
  distribution column reads:
  - $\kappa=0.1$: "6 collapse, 3 asymmetric"
  - $\kappa=0.3$: "6 symmetric"
  - $\kappa=0.5$: "6 symmetric, 3 mixed"
  - $\kappa=0.7$: "6 asymmetric"
  - $\kappa=0.9$: "7 mixed, 2 asymmetric"

  The four class names (`no_convergence`, `symmetric_only`, `mixed`,
  `asymmetric_only`) are abbreviated to bare singletons (collapse,
  symmetric, asymmetric, mixed). The body prose uses the full
  underscore form (`symmetric\_only`, `asymmetric\_only`,
  `no\_convergence`) and the `\textsf{}` family throughout. Table 1
  drops the suffix to fit the column width — defensible but worth
  a footnote like "abbreviated from `symmetric\_only`, etc., for
  table width."
- **minor — Eq (A) intersection note still missing.** Carry-forward
  from v2 reviewer minor. L246–252 derives the asymmetric NE with
  conditions $\tilde A\kappa\geq c_{\text{gap}}$ AND
  $\tilde A\kappa(1-\kappa)<c_{\text{gap}}$. Appendix B §B.3
  (asymmetric 1-Worker NE paragraph) does spell out the intersection
  reasoning — but the body §3 reader sees "The intersection of (a)
  and (b) is essentially empty below $\kappa=0.028$ and gives the
  high-$\kappa$ band $\kappa\gtrsim 0.972$" only in the appendix,
  not in body §3. Add the appendix reference inline: "(see
  Appendix B §B.3 for the intersection bookkeeping)."

## §4 Trainability (L417–525)

- **minor — `gap_closed_homogeneous` fallback rationale is now in Fig
  2 caption — also worth a body sentence.** L466–467 says
  "the metric falls back to $\texttt{gap\_closed\_homogeneous}$ because
  no NE policy exists". The Fig 2 caption (L540–545) explains the
  rationale ("the all-Hero SpecialistPolicy is the strongest
  stationary policy available on those cells, even though it is not
  a NE"). The body sentence is correct but lean; consider lifting one
  half-sentence from the figure caption into the body for readers
  who do not pause at figures.
- **minor — "approximately six hours" compute disclosure.** L430 (v5)
  carries the v2 reviewer's flagged "single 16-core host (`alc-2`) in
  approximately six hours" without CPU model. The v2 reviewer asked
  for "modern x86 server class with $\geq 32$ GiB RAM" or similar.
  Carried forward unchanged. Workshop-acceptable but worth fixing at
  camera-ready.

## §5 Related work (L554–611)

- **nit — Table 2 right-margin overflow (rendered defect).** See the
  Figures section below — flagged here for completeness because the
  defect manifests in §5.

## §6 Discussion (L613–679)

- **minor — Threats-to-the-contributions paragraph is now load-bearing
  on the per-cell-baseline finding.** L657–662 places the per-cell-
  baseline result as "load-bearing on cross-cell PPO comparison
  whenever the random-policy return varies across cells, regardless
  of the analytical reduction's fate." This is the right
  contribution-level claim. One small polish: the paragraph could
  name the v1-baseline inversion magnitude (the v1 ordering had
  asymmetric (0.262) > symmetric (0.091)) so a reader of §6 alone
  sees the methodological-finding stakes without flipping back to
  §4 L500–511. One sentence: "On this paper's data the v1 single-cell
  baseline inverted the symmetric-vs-asymmetric ordering (asymmetric
  0.262 vs. symmetric 0.091 on the 7-cell preview) — per-cell
  calibration recovers symmetric 0.180 > asymmetric 0.059."
- **nit — "Bucket Brigade is intentionally small and artificial"
  paragraph is unchanged from v4.** L616–641 is well-written and
  unchanged from v4 per the changelog. No flag.

## Conclusion (L681–704)

- **minor — Per-cell-baseline finding promotion is the right move.**
  L698–702 now promotes the methodological finding to sentence-and-a-half
  per the changelog. Good. One stylistic note: the Conclusion does not
  re-enumerate the four contributions from §1 (it has been a
  contributions-not-re-listed Conclusion since v2). Workshop papers
  vary on this; the choice here is defensible.

## Appendix A — Formal environment specification (L711–1063)

- **minor — §A.6 named-template attributions ship without `\cite{}`.**
  L962–1010 names Volunteer's Dilemma, N-player Public Goods, Stag Hunt,
  free-rider problem (rest-trap), and "Stochastic Game" as named limits
  / restrictions of Bucket Brigade without `\cite{}` keys to the
  game-theory literature (Diekmann 1985 for Volunteer's Dilemma; Olson
  1965 / Hardin 1968 for Public Goods; Rousseau 1755 / Skyrms 2004 for
  Stag Hunt). The changelog (line 53) flags this as a deliberate
  informal-attribution style choice; for the camera-ready a workshop
  reviewer may expect at least one `\cite{}` per named template.
- **minor — §A.5 "Optional dynamic variants" lists four variants out
  of scope.** L941–956. Useful context but rhetorically heavy in an
  appendix that the body promises is "the minimal contract a reader
  needs to follow §3" (body L202). Workshop readers will not act on
  the variants; a `\paragraph{Note on variants.}` instead of a full
  list would tighten.
- **nit — §A.7 Notation summary `\paragraph{Reproducibility note.}`
  closes Appendix A.** L1053–1063 ends Appendix A with a
  "Reproducibility note" that re-states the seven-phase ordering
  claim. Body §6 has a Reproducibility paragraph; the appendix's
  Reproducibility note repeats the contract spirit. Not redundant
  enough to flag but worth tightening.

## Appendix B — Analytical NE characterisation (L1065–1413)

- **major — §B.4 is structurally empty.** L1276–1281 contains:

  > "B.4 Predicted vs. empirical phase table (7-cell preview)
  >
  > The 7-cell preview at $c=0.5, \rho=0.02$ against the predicted
  > verdict:"

  followed immediately by §B.5 "Where the reduction breaks". The
  intended Table 5 has floated to mid-page 17 inside §B.5 due to
  `\begin{table}[h]` placement. The §B.4 / §B.5 layout boundary
  reads as a layout accident — a reader scanning the appendix will
  see §B.4 as an empty subsection header. Fix: either use
  `\begin{table}[!htbp]` with stronger placement to anchor Table 5
  inside §B.4, or absorb the §B.4 content (the one-sentence lead-in
  + Table 5) into §B.3 and renumber §B.4 → §B.4 (the gap-derivation
  was previously §B.5).
- **minor — Forward reference §B.1 → §B.5.** L1086 says "§B.5 reports
  where this restriction produces visibly wrong predictions". Common
  in appendices but worth a parenthetical on first encounter.
- **minor — §B.5 "Five sources of systematic bias" vs body §3
  "Three identifiable systematic biases".** Body §3 L358–375
  enumerates (i) ring locality, (ii) per-agent ownership, (iii)
  heuristic strategy space. Appendix §B.5 adds (iv) β-independence as
  leading-order, not exact, and (v) single-stage vs. multi-stage SPE.
  The expansion is a real value-add of the appendix — but a reader
  who reads body §3 first will be momentarily confused when §B.5
  promises "Five sources" of what §3 named "three." A one-sentence
  bridge at the top of §B.5 ("Body §3 lists the three biases most
  load-bearing for the κ-threshold gap; §B.5 below expands to five
  by adding two derivation-level biases that do not affect the
  qualitative phase order.") would help.
- **minor — §B.2 small-q Taylor expansion is undocumented as a bias
  source.** L1145–1148 uses the approximation
  $(1-q(k))^{T_{\min}}\approx 1-T_{\min}\cdot q(k)$ for small $q(k)$.
  At $\kappa=0.5,\rho=0.02$ this is $q(k=0)=0.02$ (fine) but the
  approximation does not name its own valid-regime. §B.5 enumerates
  five biases but does not include the linearisation. Either
  add a sixth bias bullet to §B.5 or add a single sentence to §B.2
  bounding the approximation error.
- **minor — §B.3 the "Borderline" / "Empty" cells in Table 4.** Table
  4 (L1255–1273) reports for $0.028\leq\kappa\leq 0.030$ the
  symmetric column as "Borderline" and the asymmetric column as
  "Empty". The body §3 derivation closes the discussion with
  "the symmetric NE exists on $\kappa\in[0.030,0.65]$" and "the
  asymmetric NE exists for $\kappa\gtrsim 0.972$." The "Empty" entry
  is the v2 reviewer's flagged "lower interval is empty" point now
  documented in the appendix — good. The "Borderline" entry is new
  v5 content and reads slightly oddly because the predicted verdict
  for the row is "borderline; may be \textsf{mixed}", which is the
  table's only verdict that names \textsf{mixed} pre-derivation. A
  reader will wonder why \textsf{mixed} appears at the
  $\kappa\in[0.028,0.030]$ boundary and not at the
  $\kappa\in[0.65,0.972]$ band — Table 4 names both as "mixed or
  transition" / "borderline; may be mixed" but does not connect them.
  Worth one sentence in the table caption.

## Figures

### Figure 1 — `figures/phase_diagram.png` (NE phase diagram)

- **major — Figure encoding does not match caption.** The PNG renders
  raw numerical payoff values inside heatmap cells (−9693, −648, +72)
  rather than the four NE regime categories the caption names. The
  legend below the plot shows four regime swatches (symmetric_only,
  mixed, asymmetric_only, no_convergence) but the `mixed` swatch is
  unused in the plotted c=0.5 panel (no cell is colored as mixed).
  Two unsampled cells render as bare em-dashes ("—"). A reader who
  reaches the figure cannot read off the four-regime story the §3
  prose makes. The figure is the visual anchor for §3's headline
  claim, and a workshop reviewer skimming the PDF will land on it
  first. The fix: re-render with cell color = verdict category and
  cell label = verdict name (or leave numerics small below the
  category label). The figure is also static PNG with no source script
  in `figures/src/` per the v4 audit's N4 note, so a re-render is not
  one-command — landing a `figures/src/phase_diagram.py` would also
  enable the c=1.0 and c=2.0 panels currently deferred to Appendix B
  per L398.
- **minor — Caption split into two sentences (per v5 changelog item
  7).** L408–414 caption is now two sentences (the parameter-cell /
  DO-solver framing as opening, the four-regime + β-independence
  description as second sentence). Reads cleaner than v4.

### Figure 2 — `figures/recalibrated_heatmap.pdf` (PPO heatmap)

- **minor — Figure quality unchanged from v4 audit.** Three-panel
  matplotlib figure with Tufte-style std bars; the gap_closed_ne
  vs. gap_closed_homogeneous fallback is visually encoded via
  hatching + dagger marker on the collapse row, and the caption now
  explains why the homogeneous fallback is the natural baseline on
  no-pure-NE cells (per v5 changelog item 7). No regression.
- **nit — Three-panel layout fits the page at print size.** Panel
  width is ~5cm; cell labels are readable at print size; legend
  placement (below the panels) is clear. No issue.

### Table 2 — Benchmark comparison (§5, L576–601)

- **major — Right-margin overflow on rendered PDF page 7.** The
  table's "Coop/comp" column is clipped at the page edge. Rendered
  entries are truncated to "coop", "mixe", "vs. s", "per-s", and the
  header itself clips. Fix: either narrow `\tabcolsep` (currently
  `4pt`), wrap the longer cells (`vs.\ scripted` and `per-scenario`
  can break), shorten the column header to "Coop", or move to
  `\begin{table*}` if the class supports two-column span. The PDF
  load-bearing claim ("dominates on exactly one column — NE
  characterisability") is in the §5 prose so the table truncation
  does not lose the headline; but a workshop reviewer will see a
  truncated table and dock points.

### Table 1 (§3, predicted-vs-empirical per-κ)

- **minor — Caption-prose mismatch on the "predicts the modal
  empirical class" count.** Table 1 caption (L334–347) says "The
  reduction predicts the modal empirical class on $3/5$ $\kappa$
  rows." The §3 lead-in prose (L302–306) says "The qualitative
  agreement is strongest at $\kappa\in\{0.3, 0.5, 0.9\}$ (predicted
  class is the modal empirical class)" — that's the same three rows.
  But "modal empirical class" needs a definition: for $\kappa=0.9$
  the predicted class is "mixed/transition" and the modal empirical
  is mixed (7 of 9). For $\kappa=0.5$ the predicted is
  symmetric_only and the modal empirical is symmetric (6 of 9, with
  3 mixed). For $\kappa=0.3$ the predicted is symmetric_only and
  empirical is 6/6 symmetric. These all check out — but
  the table caption's "$3/5$" figure should be named in the body too.

### Table 3 (Appendix A, Notation summary)

- No issues. Clean two-column notation table with consistent symbol /
  meaning structure. Caption explicitly says "every downstream
  section reuses this vocabulary without redefinition" — load-bearing
  for Appendix B's reuse of (κ, c_gap, A, $\tilde A$, etc.).

### Table 4 (Appendix B, predicted-NE per-κ-band summary)

- See §B.3 comment above on "Borderline" / "Empty" / "mixed" naming
  inconsistency in the κ ranges.

### Table 5 (Appendix B, 7-cell preview)

- **minor — Table floated past its §B.4 anchor.** Per the §B.4
  layout-empty flag above, Table 5 has floated to mid-page 17 inside
  §B.5 due to `[h]` placement permissiveness. Fix together with the
  §B.4 layout issue.

## Bibliography (`refs.bib`)

- **resolved — Two internal-memo `@misc` entries deleted.** v5
  resolves the v2/v3/v4 carry-forward flag on `bbenvspec` and
  `bbnestructure`. The pre-amble comment block explaining the
  `refs/<file>.md` placeholder convention is also deleted. Refs.bib
  is now 9 entries, all `@inproceedings` / `@article` venue cites.
- **minor — `ppo2017` entry is `arXiv preprint` only.** The PPO
  paper has appeared in many venues but the original is the 2017
  arXiv. Fine as-is; standard practice. No flag.

---

## Tagged for `pub-audit`

The reviewer recommends the auditor focus on these in addition to the
standard pipeline:

1. **Render-gate (re-run).** v4 audit's compile-passes were 3/3 + bibtex
   clean on the 11-page v4 PDF. v5 is 20 pages with two appendices;
   re-run `pdflatex && bibtex && pdflatex && pdflatex` and confirm zero
   `[??]` (no `\ref{app:envspec}` / `\ref{app:nestructure}` regression
   from the appendix-label inlining).
2. **Numerical reconciliation on Table 1.** Confirm the per-κ counts
   (0/9, 6/6, 6/9, 0/6, 7/9) match `results.json` for the 39-cell
   Nash grid. Specifically confirm the $\kappa=0.7$ row totaling 6 not
   9 (subsumed by $c=1.0$ per the table caption note).
3. **Figure 1 regeneration scope.** If the auditor's writ extends to
   the figure defect, scope the v6 work item: replace
   `figures/phase_diagram.png` with a regime-categorical heatmap and
   ship `figures/src/phase_diagram.py` to make the figure re-runnable.
4. **Table 2 overflow fix.** Confirm the rendered PDF's right-margin
   clip on the "Coop/comp" column and recommend a fix path
   (`\tabcolsep` / cell wrapping / `table*`).
5. **Appendix A §A.6 named-template citations.** Decide whether
   workshop-camera-ready should add `\cite{}` keys for Volunteer's
   Dilemma, Public Goods, Stag Hunt.
