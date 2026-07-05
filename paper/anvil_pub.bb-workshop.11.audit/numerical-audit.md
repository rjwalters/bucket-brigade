# Numerical audit for anvil_pub.bb-workshop.11

Two-track audit. **Track A (no-numbers-moved verification)**: full diff of
v11 `main.tex` against v10 `main.tex` (218 diff lines, reviewed in
entirety) plus `diff -rq` of the two `figures/` trees. **Track B
(artifact re-verification)**: independent recomputation of the headline
statistics from the committed `experiments/` artifacts at repo HEAD
`73911a75`.

## Track A: no-numbers-moved verification — PASS

The changelog attests "No experimental number, statistic, verdict,
figure, or table value was touched." The diff confirms it. The complete
v10→v11 change set is:

1. Header comment rewrite (v11 provenance block; stale "~260 words"
   abstract figure removed per the v10 nit).
2. `\PassOptionsToPackage{hypertexnames=false}{hyperref}` (v10 nit fix).
3. §1 page-budget footnote (NEW prose; the only number is "4-page",
   which is the BRIEF's stated venue budget, not an experimental value).
4. §2 template sentence gains 3 citations (no numeric change).
5. Figure 1 caption: "rows are $\kappa$ and columns are $\beta$" →
   "rows are $\beta$ and columns are $\kappa$" (axis-word swap only).
6. §4 skipped-cell parenthetical reworded (NC-3 fix; the row/cell
   language changes, no value changes).
7. §6 gains one new paragraph with 3 citations (no numbers).
8. Conclusion rewritten ~340→~160 words. Numbers appearing in the new
   conclusion (39-cell, 37-cell, 3–10×, ≈80/step, 16×) are a strict
   subset of the old conclusion's numbers with identical values; the
   dropped statistics (class means, 4×, 20-seed, k* details) remain in
   their owning sections unchanged.
9. Appendix A gains 4 citations + the Stag Hunt clause reworded
   (no numbers).

`figures/` (both PDFs and both `src/` scripts) are **byte-identical** to
v10 (`diff -rq` clean). Every table (Tables 1–4, App. tables) is
untouched by the diff. **Verdict: the no-numbers-moved attestation is
TRUE. Zero changed quantitative values.**

## Track B: artifact re-verification

| Text claim | Source (artifact) | Source value | Match | Notes |
|---|---|---|---|---|
| 39-cell empirical grid | `experiments/nash/phase_diagram/results.json` | 39 cells | yes | |
| Class counts n=12 sym / 11 asym / 10 mixed / 6 no_conv (§3) | results.json | 12/11/10/6 | yes | |
| Six unsampled cells = c=0.5 panel, κ∈{0.3,0.7}, all β (§3, Tab.1 caption, Fig.1 caption) | results.json | exactly those 6 absent | yes | |
| 12 of 13 (κ,c) rows with ≥2 β samples verdict-identical; lone split (κ=0.9,c=0.5): β=0.1 mixed, β∈{0.5,0.9} asymmetric_only (§3, Fig.1 caption) | results.json | 13 rows, 1 split, exactly as stated | yes | |
| Solver payoff 80.9 vs 72.0 at (κ=0.9,c=0.5); −614.4 vs −648.0 at (κ=0.5,c=0.5) (§3, App.B) | results.json | 80.915 / 72.0095; −614.433 / −648.015 | yes | |
| Table 1 per-κ empirical distributions (6C+3A / 6S / 6S+3M / 6A / 7M+2A; row totals 9/6/9/6/9) | results.json | identical | yes | |
| **NC-3 fix**: PPO sweep skipped the splitting Nash row's β=0.1 *cell* (the mixed-verdict one); (κ=0.9,c=0.5) row enters PPO subgrid with two asymmetric_only samples (§4) | results.json + `recalibrated_verdict.json` | Nash (0.1,0.9,0.5)=mixed IS absent from the PPO artifact; PPO row {0.5: asym, 0.9: asym} | yes | The v10 NC-3 wording defect is verified FIXED and the new wording is exactly artifact-true |
| 37 cells with PPO data; skipped = b0.10_k0.50_c0.50, b0.10_k0.90_c0.50 (§4, Fig.2 caption) | recalibrated_verdict.json | 37 cells; missing set = 6 Nash-unsampled + exactly those two | yes | |
| Class means 0.142 (n=11) / 0.096 (n=9) / 0.060 (n=11) / −0.024 (n=6) (§4) | recalibrated_verdict.json | 0.1416 / 0.0964 / 0.0602 / −0.0243 | yes | no_convergence via gap_closed_homogeneous fallback, as the text states |
| Adjacent class-mean separations 0.045 and 0.036 (§4) | derived | 0.142−0.096=0.045; 0.096−0.060=0.036 | yes | |
| Homogeneous class means 0.041 / 0.046 / 0.030 / −0.024; symmetric/mixed inversion on a 0.005 separation (§4 Caveats) | recalibrated_verdict.json | 0.0406 / 0.0456 / 0.0298 / −0.0243 | yes | |
| β-invariance on all 13 PPO rows with ≥2 β samples (§4) | recalibrated_verdict.json | 13 rows, 0 splits | yes | |
| 20 seeds on 8 buy-down cells, 4 elsewhere (§4, Fig.2 caption) | recalibrated_verdict.json | n_seeds: {20: 8, 4: 29} | yes | |
| Entropy: Spearman ρ=0.109 (p=0.56), n=31; min-vs-homogeneous p=0.038; homogeneous mean ρ=0.342 (§4, §6) | `entropy_vs_trainability.json` | 0.1088 (p=0.5600), n=31; p=0.0384; 0.3422 | yes | |
| Within-column gap_closed_ne varies by up to 0.34; e.g. −0.32/+0.01/−0.00 at κ=0.1,c=1.0; per-cell std reaches 0.99, median 0.24 (§4) | recalibrated_verdict.json | max column range 0.3384 (at κ=0.3,c=2.0); −0.317/+0.010/−0.001; max std 0.993, median 0.241 | yes | |
| k* binary test: rank-biserial 0.00, exact one-sided p=0.539 (3 vs 8 columns); permutation Δmean=−0.002, two-sided p=0.994 (165 assignments); power floor 0.006 (§1, §4) | `kstar_vs_trainability.json` | 0.0 / 0.5394; −0.00184 / 0.9939 / 165; 0.00606 | yes | |
| Homogeneous split: rank-biserial 0.40, one-sided p=0.19 (13 columns) (§4) | kstar json | 0.3999 / 0.1853 | yes | |
| Post-hoc failure zone: rank-biserial −1.00, one-sided p=0.0035 (floor at 3 vs 10), permutation two-sided p=0.021 (§4) | kstar json | −1.0 / 0.003497 (= floor) / 0.02098 | yes | |
| Spearman k* vs homogeneous gap ρ=−0.556 (p=0.0004, n=37); −0.635 (p=0.020) at column level (§4) | kstar json | −0.5559 (p=0.000354, n=37); −0.6355 (p=0.0196) | yes | |
| Buy-down: CI half-widths shrink 2.00–3.64× (median 2.65); 1.84–3.59× homogeneous; √5≈2.24; largest shift 0.372→0.043; 0/8 verdict flips (§4) | `noise_buydown_precision.md` | all present verbatim | yes | |
| Improvability oracle: +13.4/step, paired 95% CI [+10.5, +16.6] (§4) | `improvability_oracle.md` | +13.374 [+10.490, +16.628] | yes | |
| rest_trap frozen NE payoff 2984.04/ep, ≤248.67/step (§5, Tab.2) | `bucket_brigade/baselines/release/local/nash/rest_trap-v1.json` | 2984.0437; /12 = 248.67 | yes | |
| always_rest 288.55 [285.20, 291.65] (§5, Tab.2) | `scripted_battery/rest_trap.md` | 288.548 [285.200, 291.649] | yes | |
| Random 302.87; upper bound 304.31 (302.94 at n=10k) (§5, Tab.2) | rest_trap.json + trap_escape cell_summary.json | 302.87 / 304.31 | yes | |
| specialist 386.60 [386.17, 387.03]; paired uplift +83.67 [+82.36, +84.89], n=10,000 (§5, Tab.2) | rest_trap.{json,md} | 386.60 [386.17, …]; +83.667 [+82.355, +84.886] | yes | |
| het_ppo 1×: 306.26 [302.95, 309.33]; uplift +3.39±7.34 (§5, Tab.2) | `tier1_runs/tier1_verdict.md` | identical | yes | |
| 16× escaped_trap: 307.83 [305.00, 310.71], clears 304.31 by +0.69; 4× het_ppo 304.03; ippo 301.77/303.46; mean ladder 306.26→307.66→307.83; +1.57±4.80; uplift std 8.36→6.59; mean uplift +4.96 (§5) | `tier1_runs_trap_escape/*/cell_summary.json` | all present; 16× verdict reason string states exactly the claimed CI and anchors | yes | |
| Seeded-DO cycling: min improvement 11.79 vs ε=0.01; payoff oscillation [1611.56, 2316.61]; mixture exploitable ≥407.50; specialist genome image exploitable +605 to +870; support scores −258 to +237 (§5) | `experiments/nash/rest_trap_seeded_do/RESULTS.md` | all present | yes | |
| A = 50·12+100·12+10·12/10 = 1812; Ã = 36.24; collapse κ≈0.028; symmetric band [0.030, 0.65] (f(1/4)=27/256≈0.105); asymmetric onset κ≈0.972; g(1/2)=9.06 (§3, App.B) | arithmetic re-derivation | 600+1200+12=1812; 36.24; 1/36.24=0.0276; roots verified | yes | |
| State cardinality 2304 = 3²·2⁴·2⁴ at H=2,N=4; ≈9.44×10⁹ at H=10; per-agent actions 8/40; joint 8⁴=4096 (abstract, §2, App.A, Tab.3) | arithmetic | 9·16·16=2304; 3¹⁰·10⁴·2⁴≈9.45×10⁹ | yes | |

Abstract ↔ body ↔ table cross-checks (2304/8, 39-cell, 3–10×, 37-cell,
20-seed, k* p=0.54, ≈80/step, 16×, "one marginal variance-driven
escape") are all internally consistent; the rewritten conclusion
introduces no number not already verified in its owning section.

**Numerical inconsistencies found: 0.**

## Figure source-of-truth check (informational)

- `figures/phase_diagram.pdf` ↔ `figures/src/phase_diagram.py`: script
  verified against the render contract — the script draws rows over
  `BETAS` (y-axis label `$\beta$`) and columns over `KAPPAS` (x-axis
  label `$\kappa$`), 3 panels by `c`, hatched "n/a" for unsampled
  cells. **The v11 Figure 1 caption fix ("rows are $\beta$ and columns
  are $\kappa$") is verified correct against the generating script**
  (the v10 review's single major is properly resolved). Figure 2's
  caption layout (rows/columns per its own script) was checked and is
  not the copy-paste transposition.
- mtime check: both `src/` scripts are 0.1–0.6 **milliseconds** newer
  than their rendered PDFs — a `cp -p` copy-ordering artifact of the
  v10→v11 carry-over, not a content-staleness signal. Both figure PDFs
  and both scripts are byte-identical to v10, whose audit verified the
  renders as fresh. Recorded as informational only (non-critical note
  6); no re-render is required.
