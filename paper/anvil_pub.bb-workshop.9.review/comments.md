# Line-level comments — anvil_pub.bb-workshop.9

Grouped by severity: **blocker** / **major** / **minor** / **nit**.
Line anchors refer to `main.tex`.

## Blockers

**None.**

## Majors

### §4 Results + Figure 2 + §7 Reproducibility — committed-artifact drift (L589-L611, L807-L832, L1206-L1209)

"Figure~\ref{fig:ppo} reports the per-cell $\texttt{gap\_closed\_ne}$
(mean$\pm$std over 4 seeds)" / "reproduce byte-for-byte from the
committed per-seed summaries". The committed
`experiments/p3_specialization/phase_diagram_ppo_v2/recalibrated_verdict.json`
was regenerated on 2026-07-04 (commit a5b8ccdc) with n=20 means for the
8 buy-down cells. Consequences, all verified by recomputation: (a) the
§4 class means 0.180/0.107/0.059 recompute to 0.142/0.096/0.060 from
the current artifact; (b) the shipped Figure 2 shows the n=4 values
(e.g. +0.37 at kappa=0.3, c=1.0, beta=0.5 where the artifact now holds
0.043), so re-running `figures/src/recalibrated_heatmap.py` today
produces a different figure; (c) the homogeneous-metric ordering behind
"the load-bearing claim is robust to the metric choice" (L805) inverts
(symmetric 0.041 < mixed 0.046). The paper already discloses the
buy-down and the 0.372->0.043 shift, so this is staleness, not
concealment — but the reproduction pointer now contradicts the reported
numbers. Fix: pin a frozen snapshot or regenerate figure + numbers from
the merged data, and re-derive the robustness sentence either way.

### §3 + Figure 1 caption + Table 1 caption — unsampled cells mischaracterized (L381-L383, L529-L531, L466-L469)

"less six cells in the high-$\kappa\times c{=}0.5$ corner subsumed by
the $c{=}1.0$ row". Verified against
`experiments/nash/phase_diagram/results.json`: the sampled c=0.5 cells
are kappa in {0.1, 0.5, 0.9}; the six unsampled cells are the
kappa in {0.3, 0.7} columns (all three beta), which are neither
high-kappa nor a corner — and the rendered Figure 1 hatches exactly
those columns, contradicting its own caption. Table 1's caption
compounds it: "the row totals are 9 except at $\kappa=0.7$ (6 cells...)"
— the kappa=0.3 row in the same table also lists 6 cells. Fix at all
three sites.

### §3, §5 — severe overfull hboxes (L320-L329, L851-L869, L874-L905, L965-L978)

Build log: 10 overfull hboxes, five severe — 119.9pt (L320-329, p.4),
128.9pt (L851-869, p.10), 98.3pt + 105.3pt (Table 3 body + caption,
p.11), 96.7pt (L965-978, p.12). Visually confirmed on the rendered
pages: long `\texttt` artifact paths (e.g.
`bucket_brigade/baselines/release/local/nash/rest_trap-v1.json`,
`experiments/.../improvability_oracle.md`) run up to ~4.5cm past the
right margin. Fix: `\path`/`\seqsplit`-style breakable paths, or move
provenance paths to footnotes/table notes.

### §6 Related work — closest evaluative tradition unengaged (L1067-L1126) [related-work]

"it is the only published parametric MARL game where the question ...
admits a ground-truth answer" (Abstract L144-147, echoed §6 and
Conclusion). The related-work section engages only the six rich
benchmarks. The closest prior line for "did the algorithm converge to
the right equilibrium" is the small-game equilibrium-computation and
exploitability-evaluation tradition — OpenSpiel's small games with
exact solvers, and NashConv/PSRO-style convergence-to-equilibrium
measurement — which a MARL-literate PC member will raise immediately.
The claim survives literally on the "parametric" qualifier, but the
omission reads as a survey gap. **Recommendation: re-run
`pub-litsearch`** targeting (i) OpenSpiel, (ii) exploitability-based
MARL evaluation, (iii) any parametric-family game benchmarks; then
either engage the cluster in §6 or scope the claim explicitly to
parametric phase-diagram families. (No `.bib` entries are added by this
review; verification is litsearch's job.)

## Minors

### Abstract (L74-L148) — one-page abstract

~700 words carrying per-test statistics (rank-biserials, exact p's,
CI bounds). Move statistics to their owning sections; target ~250
words. Also serves dim 9.

### §4 (L589-L611) — ordering-not-significance framing

Honest as written, but with class separations of 0.001/0.017 the
class-mean ordering itself is within noise; percentile-bootstrap CIs on
class means (the machinery already exists from the buy-down) would make
the "consistent with the predicted ordering" sentence quantitative.

### Carry-forwards from the v6 review, unaddressed in v7-v9

- Appendix A §A.6 named templates (Volunteer's Dilemma, Public Goods,
  Stag Hunt, free-rider) still have no `\cite{}` links (L1547-1599).
- Appendix B §B.4/§B.5 boundary still reads as a layout accident
  (one lead-in sentence, then the table, then §B.5; L1865-1901).
- The small-$q(k)$ Taylor linearisation (L1735-1737) is still not named
  among §B.5's "Five sources of systematic bias" (L1904).
- HuggingFace baselines pathway still "in progress" (L1235-1240).
- Figure 1 caption still lacks the explicit "three panels, one per
  $c$" half-sentence that Figure 2's caption has.

## Nits

- Figure 2 caption "top- and middle-right" is imprecise for the
  asymmetric cells at c=1.0 (they occupy the kappa=0.1 bottom row
  there); consider "the asymmetric-only cells" without a position
  gloss.
- 26 rendered pages against the BRIEF's 4-page workshop body target;
  the operator has opted into the appendix policy (per the v6 cycle),
  so recorded as a nit, but the venue page budget still needs a
  submission-time answer.
