# Line-level comments — anvil_pub.bb-workshop.10

Grouped by severity: **blocker** / **major** / **minor** / **nit**.
Line anchors refer to `main.tex`.

## Blockers

**None.**

## Majors

### Figure 1 caption — transposed axis description (L527-L529)

"Three panels, one per cost $c\in\{0.5,1.0,2.0\}$; within each panel
rows are $\kappa$ and columns are $\beta$." The half-sentence added in
v10 (closing the v6 carry-forward about panel structure) is transposed.
Verified against the rendered PDF and the generating script
(`figures/src/phase_diagram.py` builds a 3-row grid over `BETAS` and a
5-column grid over `KAPPAS`; the y-axis is labeled "$\beta$ (spread
prob)", the x-axis "$\kappa$ (extinguish prob)"): rows are $\beta$ and
columns are $\kappa$. The caption also contradicts its own later
clause ("the $\kappa\in\{0.3,0.7\}$ **columns** of the $c{=}0.5$
panel" — correct) and the §3 body text ("the two mid-$\kappa$ columns"
— correct). This is the same caption/figure-contradiction class as the
v9 CF-2, introduced by the fix itself; it stops short of flag severity
because the figure's own axis labels are correct and no number is
affected. Fix: swap the clause to "rows are $\beta$ and columns are
$\kappa$" (note Figure 2 has the opposite layout — rows $\kappa$,
columns $\beta$ — so the two captions should not be copy-pasted from
each other, which is presumably how this happened).

## Minors

### §6 Related work — engagement depth (L1174-L1195) [related-work]

The new equilibrium-computation/exploitability paragraph is honest and
correctly scoped, but rests on exactly two citations entered from
memory (`openspiel2019`, `psro2017`; both flagged unverified in
refs.bib's own header). A `pub-litsearch` re-run should
(i) resolver-verify both entries plus `ppo2017`/`mappo2022`,
(ii) survey the exploitability-evaluation line's empirical follow-ups
(NashConv-style measurement in MARL evaluation) and any other
parametric-family game constructions, and (iii) source the Appendix A
named-template citations (Volunteer's Dilemma, Public Goods, Stag
Hunt) — the standing declined item. No `.bib` entries are added by
this review; verification is litsearch's job.

### Conclusion (L1315-L1354) — still near-abstract-length restatement

The unnumbered conclusion restates every result of the abstract at
comparable coverage (reduction, thresholds, ordering, double negative,
rest_trap ladder, positioning). It is ~30% tighter than v9 but remains
the largest remaining dim 9 duplication. Consider cutting to the two
sentences a PC member needs after reading the paper: what was
established, and what the open challenge is.

### §5 footnote (L1065-L1071) — non-regenerable provenance, carried

The 300/300 bootstrap-combination sweep and the t-interval
[304.67, 310.99] trace to the PR #460 judge recomputation via the
committed v8 paper-trail files, not a regenerable `experiments/`
artifact. The footnote discloses this honestly; carrying it as an open
item — regenerate as a committed script + JSON before camera-ready so
the paper's every-number-traces-to-an-artifact standard is uniform.

### §7 Reproducibility (L1308-L1313) — HuggingFace baselines still in progress

"the publication pathway for the baselines is in progress and not yet
complete at the time of this draft" is a true status statement, but the
NeurIPS checklist expects it resolved by submission. Operator task,
carried from v9.

## Nits

- **Header comment stale** (L24): the v10 header comment says the
  abstract was compressed "to ~260 words"; the rendered abstract is
  ~325 words (the changelog's ~318 is close). Cosmetic — the comment
  never renders — but worth syncing since the file is the artifact of
  record.
- **Table 2 caption path breaks** (L937-L940): `xurl` breaks
  `experiments/p3_specialization/...` after "ex-" with no hyphen; a
  reader can misparse the linebreak as a space in the path. Acceptable
  cost of the 0-overfull fix; consider `\sloppy`-tolerant placement or
  a shorter alias if it bothers anyone downstream.
- **Duplicate hyperref destinations**: the `\tag`-ed display equations
  (S)/(A)/(C) appear in both §3 and Appendix B, producing repeated
  `destination with the same identifier (name{equation.1})` pdfTeX
  warnings. Harmless; silence with `hypertexnames=false` or distinct
  tags if the audit's log-cleanliness bar tightens.
- **27 rendered pages** against the BRIEF's 4-page workshop body
  target; standing operator decision (appendix policy), recorded again
  for the submission-time answer.
