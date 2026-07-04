# Per-dimension scoring — anvil_pub.bb-workshop.6

Rubric: `.anvil/skills/pub/rubric.md` (9 dimensions, /44, advance ≥35).

| # | Dimension | Weight | Score | Justification |
|---|---|---:|---:|---|
| 1 | Rigor of method / argument | 6 | **5** | §3 derives closed-form boundary inequalities (S/A/C) via an explicit single-house mean-field reduction with $\tilde A=A\rho=36.24$; the 3–10× quantitative gap against the equilibrium solver is owned in the same section with three named systematic biases (ring locality, per-agent ownership, heuristic-strategy approximation); Appendix B §B.5 expands the bias accounting to five sources. v6's §4 metric-consistency fix (separations now all on `gap_closed_homogeneous`: 0.001, 0.017, 0.082) is a real rigor improvement: the ordering-not-significance argument is now arithmetically clean and the per-class mean within-cell std (≈0.21–0.44) dwarfs all three separations by 3–200× rather than the v5 cross-metric list's 2–6×. Same rigor weakness as v5 remaining: the small-$q(k)$ Taylor expansion in §B.2 is not named as a fifth systematic bias source. Not load-bearing for the qualitative claim. Held at 5/6. |
| 2 | Evidence sufficiency | 6 | **5** | The headline ordering rests on $n=11/9/11/6$ cells across the four NE classes on the 37-cell PPO sweep (carried unchanged from v5). The inline Table 1 (per-κ predicted-class share: 0/9, 6/6, 6/9, 0/6, 7/9) gives a reader an in-body view of where the analytical prediction does and does not hit the modal empirical class. The 4×-budget structural-failure disambiguation ($-0.049 → -0.108$ on no_convergence cells) is the right experiment for the right claim. The v6 metric-consistency fix (M2) actually strengthens the evidence story: the now-cleanly-stated 0.001/0.017/0.082 separations make the "consistent with the predicted ordering but does not reject the null of equal class means" framing sharper. Held at 5/6 because per-class std remains comparable to class-mean separation; the named cross-class 4×-budget sweep as the camera-ready significance gate is the right amount of honesty for a workshop pitch. |
| 3 | Clarity of contribution | 5 | **5** | The §1 Contributions split (3a) / (3b) is unchanged from v5 and directly addresses the v2 reviewer's flag about the per-cell baseline correction being buried in a sub-clause. Abstract (L48–83) gives a reader the four NE classes, the ordering, the 3–10× gap, and the methodological-complement framing in a single ~270-word paragraph. The contribution is unambiguous and one-sentence-extractable per item. Held at 5/5. |
| 4 | Related-work positioning | 5 | **4** | §5 + Table 2 cover the six canonical MARL benchmarks (Overcooked, Melting Pot, Hanabi, SMAC/SMACv2, MAgent, PettingZoo MPE) with year, agent count, per-agent action space, NE characterisability, and coop/competitive axis. v6 fixes the v5 R2 margin overflow without losing any column content. No close prior work obviously omitted; the six benchmarks are the right reference set for the workshop's audience. Held at 4/5 for the same legacy-baseline reason: no `pub-litsearch` sibling was run, so the perspective-substrate bonus the rubric describes does not apply. |
| 5 | Reproducibility | 5 | **5** | Appendix A is a self-contained mathematical contract (state cardinality formula, seven-phase ordering, independent-workers extinguish formula, per-agent reward decomposition, Table 3 notation summary). Appendix A §A.7 explicitly says "a reader who reimplements from this appendix alone — using any RNG and any programming language — should obtain matching equilibrium structure and policy-evaluation results." Appendix B §B.1–§B.3 spells out the derivation of (S)/(A)/(C) line-by-line. §6 Reproducibility paragraph names `pip install bucket-brigade`, `docs/PAPER_RESULTS.md`, `bucket_brigade/baselines/per_cell.py`, and the GitHub URL. v6 additionally ships the real `figures/src/phase_diagram.py` source script for Figure 1 — a quiet upgrade for the reproducibility contract (v5 had only the static PNG). Held at 5/5. |
| 6 | Figure & table quality | 4 | **4** | **The major v5 deduction is closed.** Figure 1 (the new `phase_diagram.pdf`) is a 3-panel categorical regime heatmap colored by NE verdict — one panel per $c\in\{0.5,1.0,2.0\}$, each a 3×5 $\beta\times\kappa$ grid, with all four regimes visible (no_convergence at $\kappa=0.1$, symmetric_only at $\kappa=0.3–0.5$, mixed and asymmetric_only at $\kappa=0.7–0.9$). The six unsampled cells in the high-$\kappa\times c=0.5$ corner are hatched light grey with "n/a" labels — no bare dashes. Legend swatches all correspond to cells in the plot. Source script `figures/src/phase_diagram.py` ships alongside, matching Figure 2's repro pattern. Table 2 (v5 R2) is wrapped in `\resizebox{\linewidth}{!}{...}` plus tightened cell text ("vs.\ scr.", "per-scen."); rendered page 7 shows the "Coop/comp" column entries complete on the page. Figure 2 (`recalibrated_heatmap.pdf`) is the v4-audited well-rendered three-panel matplotlib figure — unchanged and still fine. Tables 1, 3, 4, 5 are clean. **+2 vs. v5** — the figure now is the visual anchor the §3 prose claims it is. |
| 7 | Prose & structural quality | 4 | **4** | LaTeX compiles cleanly (verified at the v5 audit; v6 changes are within-paragraph or cell-text only, no expected regression). Section flow is canonical (Abstract → Intro → Env → NE structure → Trainability → Related work → Discussion → Conclusion → References → Appendices A,B). The v5 dim-7 deduction (§3 L389 promised a "per-cell predicted-vs-empirical table" that Appendix B did not deliver) is closed by v6's softening of the cross-reference to "The full bias accounting and the 7-cell preview predicted-vs-empirical table appear in Appendix B" — matches what Appendix B §B.4 / §B.5 actually carries. §4 Results paragraph reads cleanly post-edit: the "we use the homogeneous separations here for metric consistency since no-convergence cells lack `gap_closed_ne`" half-sentence is exactly the asterisk the auditor asked for. **+1 vs. v5**. The §B.4 / §B.5 layout-accident sub-issue persists (§B.4 contains one lead-in sentence and then immediately the §B.5 header) — not blocking, flagged in `comments.md` as a minor. |
| 8 | Citation hygiene | 5 | **5** | `refs.bib` is unchanged from v5 (per changelog). The 9 `\cite{}` keys (overcooked2019, meltingpot2021, hanabi2020, smac2019, smacv2_2023, magent2018, pettingzoo2021, ppo2017, mappo2022) all resolved at the v5 audit and v6 introduces no new cites. Every bibliography entry has the full author/title/venue/year set. Appendix A §A.6 still names Volunteer's Dilemma, Public Goods, Stag Hunt without `\cite{}` links — flagged as a minor for camera-ready but not enough to deduct the full point. Held at 5/5. |
| 9 | Rhetorical economy | 4 | **3** | The v6 changes preserve the §6 Threats merge and the §4 Caveats compression from v5. Net rhetorical economy of the body (§1–§6 + Conclusion) is high. Held at 3/4 because the appendix bulk is unchanged — the 247 + 294 source-markdown lines ported as appendix prose still re-cover some body ground (Appendix A §A.6 re-states §2 L195–203 with elaboration; Appendix B §B.5 re-frames §3 with two added biases). A camera-ready pass that promoted the appendices to supplementary materials would let the body alone read tighter. The user has opted out of workshop budget enforcement so this is not a structural blocker, just the lowest-relative-score dimension carrying forward. |
| | **Total** | **44** | **40** | |

## Critical flags

None set by the reviewer.

## Calibration notes

- v6 nets **+3 on the /44 score vs. v5** (37 → 40), driven entirely
  by dim 6 +2 (R1 + R2 both closed) and dim 7 +1 (R3 closed). Dims
  1, 2, 3, 4, 5, 8, 9 hold their v5 scores. No regression on any
  dimension.
- The +2 on dim 6 is the leverage the v5 reviewer named: "a
  one-evening fix to Figure 1 (re-encode as categorical regime
  heatmap, three c-panels) and Table 2 (column wrap or `table*`)
  would lift dim 6 to 3/4 and the total to 38/44." v6 went further
  than 3/4 by also shipping the real source script (matching
  Figure 2's pattern) and using all four regime cells in the
  plotted body — dim 6 lifts cleanly to 4/4.
- The +1 on dim 7 is the v5 reviewer's R3 finding ("Body §3's
  'per-cell predicted-vs-empirical table appears in Appendix B'
  promise is not fulfilled"). v6's softened cross-reference matches
  what Appendix B delivers and removes the structural-quality
  deduction.
- The two auditor M-flags (M1 std-range estimator, M2 cross-metric
  separations) close at the dim 1 + dim 2 level by strengthening
  the ordering-not-significance argument arithmetically. Neither
  M-flag was load-bearing enough on its own to move a /6 dimension,
  but the combined polish on dim 1 and dim 2 keeps both at 5/6
  without any soft regression.
- No dimension scored below ~75% of its weight. The lowest relative
  score is dim 9 (Rhetorical economy) at 3/4 = 75%.
