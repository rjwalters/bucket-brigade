# Numerical audit for anvil_pub.bb-workshop.10

This is the CF-1/CF-2 clearance re-audit after the v10 `pub-revise` pass. All
recomputations below were performed against the artifacts at repository HEAD
(`bf26f818`; both flagged artifacts unchanged since commit `a5b8ccdc`, i.e.
HEAD content == the revision v10 recomputed against). Sources:
`experiments/p3_specialization/phase_diagram_ppo_v2/recalibrated_verdict.json`
(rv), `experiments/nash/phase_diagram/entropy_vs_trainability.json` (ev),
`experiments/nash/phase_diagram/results.json` (rj),
`experiments/nash/phase_diagram/kstar_vs_trainability.json` (kv),
`experiments/nash/phase_diagram/noise_buydown_precision.md` (nb),
`experiments/p3_specialization/phase_diagram_ppo_longbudget/cell_*/cell_summary.json` (lb),
`experiments/p3_specialization/scripted_battery/rest_trap.md` (sb),
`experiments/p3_specialization/tier1_runs/tier1_verdict.md` (t1),
`experiments/p3_specialization/tier1_runs_trap_escape/{4x,16x}/tier1_verdict.json` (te),
`experiments/nash/rest_trap_seeded_do/RESULTS.md` (sd),
`experiments/nash/phase_diagram/improvability_oracle.md` (io),
`bucket_brigade/baselines/release/local/nash/rest_trap-v1.json` (nej).

## CF-1 clearance (v9 flag: artifact supersession) — §4 / §6 / Figure 2

| Text claim | Source | Recomputed value | Match |
|---|---|---|---|
| Class means `gap_closed_ne` sym/mixed/asym = 0.142 / 0.096 / 0.060 (n=11/9/11) | rv | 0.1416 / 0.0964 / 0.0602 (n=11/9/11) | YES |
| `no_convergence` homogeneous-fallback mean −0.024 (n=6) | rv | −0.0243 (n=6) | YES |
| Adjacent class-mean separations 0.045 and 0.036 | rv | 0.0452 / 0.0362 | YES |
| Per-class mean within-cell std ≈0.20–0.42 | rv | 0.1976 – 0.4223 | YES |
| Homogeneous class means sym/mixed/asym/collapse = 0.041 / 0.046 / 0.030 / −0.024; sym/mixed inverts on a 0.005 separation ("robust to metric choice" withdrawn) | rv | 0.0406 / 0.0456 / 0.0298 / −0.0243; inversion confirmed at HEAD, separation 0.0050 | YES |
| Entropy headline: Spearman ρ=0.109 (p=0.56), 31 cells, mean h_cond vs `gap_closed_ne` | ev | ρ=0.10884, p=0.56000, n_cells=31 | YES |
| Pinned provenance: "the original all-n=4 computation, preserved at artifact revision `22b1fda6`, gave ρ=0.007, p=0.97" | `git show 22b1fda6:…/entropy_vs_trainability.json` | ρ=0.0069300, p=0.97049 at that revision; revision verified to exist and to be the pre-buy-down (#431) commit of exactly this artifact | YES |
| Single nominally significant entry: min vs homogeneous, p=0.038; fails Bonferroni α=0.0063 (8 tests) | ev | min/homog p=0.03842; all other 7 entries p>0.05; 0.05/8=0.00625 | YES |
| β-column example −0.32 / +0.01 / −0.00 at (κ=0.1, c=1.0); within-column range up to 0.34 | rv | −0.3169 / +0.0100 / −0.0006; max within-column range 0.3384 (at κ=0.3, c=2.0) | YES |
| Per-cell std reaches 0.99, median 0.24 | rv | max 0.9925, median 0.2411 | YES |
| Buy-down: CI half-widths shrink 2.00–3.64× (median 2.65); 1.84–3.59× homogeneous; 0/8 verdict flips (all `insufficient`); largest shift 0.372→0.043 at (κ=0.3, c=1.0); √5≈2.24 | nb | 2.00–3.64 (median 2.65); 1.84–3.59 (median 2.65); 8/8 remain insufficient; 0.372→0.043 at b0.50_k0.30_c1.00 | YES |
| 4×-budget no-convergence sweep: mean `gap_closed_homogeneous` −0.108 (from −0.024) | lb (6 cells) | mean −0.1079 | YES |
| Figure 2 rendered values | pdftotext of `figures/recalibrated_heatmap.pdf` vs rv | All 37 rendered cell values match the HEAD JSON at 2-dp rounding, cell for cell (incl. the changelog spot-check κ=0.3, c=1.0, β=0.5 = +0.04); the 2 skipped + 6 unsampled cells render n/a | YES |
| §6 "every §4 statistic and the rendered Figure 2 match the artifacts at repository HEAD" | all above | Confirmed by this audit's independent recomputation | YES |
| Protocol disclosure: 20 seeds on the 8 buy-down cells, 4 elsewhere | rv | n_seeds ∈ {4, 20}; exactly 8 cells at 20 | YES |

**CF-1 is CLEARED.** Every number the v9 audit traced to superseded
revisions now matches the artifacts at HEAD; the withdrawn metric-robustness
sentence is replaced by an explicit statement of the inversion; the lone
retained superseded number (ρ=0.007) is explicitly revision-pinned and the
pin is correct.

## CF-2 clearance (v9 flag: Table 1 caption vs body vs §3 vs Fig 1)

| Site | v10 text | Recomputed from rj | Match |
|---|---|---|---|
| Table 1 caption | "row totals are 9 except at κ∈{0.3, 0.7} (6 cells each: the c=0.5 panel was not sampled at those two κ columns)" | Per-κ totals 9/6/9/6/9; the 6 missing cells are exactly (β∈{0.1,0.5,0.9}) × (κ∈{0.3,0.7}) × c=0.5 | YES |
| Table 1 body | κ=0.1: 6 collapse + 3 asym (0/9); κ=0.3: 6 sym (6/6); κ=0.5: 6 sym + 3 mixed (6/9); κ=0.7: 6 asym (0/6); κ=0.9: 7 mixed + 2 asym (7/9); modal on 3/5 rows | Identical distribution recomputed; modal rows κ∈{0.3,0.5,0.9} = 3/5 | YES |
| §3 prose | "the six unsampled c=0.5 cells at κ∈{0.3,0.7} — the two mid-κ columns of that panel, across all three β" | as above | YES |
| Figure 1 caption | "the six unsampled cells (the κ∈{0.3,0.7} columns of the c=0.5 panel) are hatched" | as above | YES |
| Splitting row | "(κ=0.9, c=0.5): β=0.1 mixed, β∈{0.5,0.9} asymmetric_only"; 12/13 rows identical | 13 rows with ≥2 β samples; exactly one splits, at (0.9, 0.5) with that pattern | YES |

No "high-κ corner" / "subsumed" language remains anywhere in main.tex.
**CF-2 is CLEARED.**

## Full re-audit of the remaining body numbers

| Text claim | Source | Recomputed / found | Match |
|---|---|---|---|
| §2/App.A: 2304 = 3²·2⁴·2⁴ states; 8=4H actions at H=2 (40 at H=10); 8⁴=4096 joint; ≈9.44×10⁹ at H=10 | arithmetic + refs/env_spec.md | 9·16·16=2304; 3¹⁰·10⁴·2⁴=9.4478×10⁹ | YES |
| §3/App.B: A=1812 (=50·12+100·12+10·12/10), Ã=36.24; S-band [0.030, 0.65]; A-onset κ≳0.972 (lower root 0.028); C-boundary κ≈0.028; f(1/4)=27/256≈0.105 | arithmetic | roots 0.0303/0.6514; 0.0284/0.9716; 1/Ã=0.0276; 27/256=0.10547 | YES |
| §3: verdict counts sym 12 / asym 11 / no_conv 6 / mixed 10; mixed primarily at κ=0.9, c≥1 and κ=0.5, c=2.0 | rj | verdict_counts identical; mixed = 3×(0.5,2.0) + 6×(0.9,c≥1) + (0.9,0.5,β=0.1) | YES |
| §3/App.B solver-noise probes: payoff 80.9 vs 72.0 at (0.9,0.5); −614.4 vs −648.0 at (0.5,0.5) | rj `best_team_payoff` | 80.915 vs 72.0095; −614.433 vs −648.015 | YES |
| §4 protocol: 37 of 39 cells, skipped b0.10_k0.50_c0.50 + b0.10_k0.90_c0.50; N=148 original runs; 31 cells with converged NE baseline | rv, rj | 37 cells; those two absent; 37×4=148; 37−6=31 | YES |
| §4 PPO-subgrid β-invariance: 13/13 rows verdict-identical | rv | 13 rows ≥2 β samples; all identical (the Nash-splitting cell β=0.1@(0.9,0.5) is one of the two skipped) | YES (see wording note in flags.md) |
| §4 k* primary test: rank-biserial 0.00, exact one-sided p=0.539, permutation Δmean −0.002 (two-sided p=0.994), 165 assignments, power floor p=0.006; 3 vs 8 columns | kv | 0.0 / 0.53939 / −0.001844 / 0.99394 / 165 / 0.00606; n1=3, n2=8 | YES |
| §4 homogeneous split: rank-biserial 0.40, one-sided p=0.19, 13 columns | kv | 0.4000 / 0.18531 / 3+10 | YES |
| §4 failure zone (post hoc): rank-biserial −1.00, exact one-sided p=0.0035 (= combinatorial floor at 3 vs 10), permutation two-sided p=0.021 | kv | −1.0 / 0.0034965 (= floor) / 0.020979 | YES |
| §4 Spearman k* vs homog: ρ=−0.556, p=0.0004, n=37; column-level ρ=−0.635, p=0.020; entropy n.s. ρ=0.109/0.342 | kv, ev | −0.55592 / 0.000354 / 37; −0.63549 / 0.019587; 0.10884 (p=0.56) / 0.34222 (p=0.0595) | YES |
| §4 k* = pure function of κ: 0.1→4, 0.3–0.7→2, 0.9→1; artifact joins 13 of 15 columns, 11 on the NE gap | kv | mapping identical; n_columns_joined=13, two κ∈{0.3,0.7}×c=0.5 columns lack PPO; NE-gap test 3+8=11 | YES |
| §4 improvability datum: +13.4/step over uniform, paired 95% CI [+10.5, +16.6] | io | +13.374 [+10.490, +16.628] | YES |
| §5 anchors: NE ≤248.67 (=2984.04/12); always_rest 288.55 [285.20, 291.65] n=2000; random 302.87 point / 302.94 [301.46, 304.31] n=10k; specialist 386.60 [386.17, 387.03] n=10k; paired Δ +83.67 [+82.36, +84.89] | nej, sb | 2984.0437/12=248.670; 288.548 [285.200, 291.649]; 302.87 / 302.936 [301.464, 304.307]; 386.603 [386.173, 387.026]; +83.667 [+82.355, +84.886] | YES |
| §5 1× het_ppo: 306.26, CI [302.95, 309.33], at_random; uplift +3.39±7.34; clears random point by 0.08 | t1 | identical; 302.95−302.87=0.08 | YES |
| §5 ladder: 4× het_ppo CI-lo 304.03 (mean 307.66); 4×/16× ippo CI-lo 301.77/303.46 (at_random); 16× het_ppo 307.83 [305.00, 310.71] escaped_trap, clearing 304.31 by +0.69 | te | all identical, incl. verdict strings | YES |
| §5 dose-response: means 306.26→307.66→307.83; 16×−1× = +1.57±4.80 (t=1.46); uplift std 8.36→6.59; mean uplift +4.96 ≈6% of 83.7 headroom; best seed ≈27%, ≈61/step short; ≈79/step gap to scripted_best | te, 16x notes | 307.83−306.26=1.57; notes record +1.57±4.80 (t=+1.46), ≈27%, ≈61 short; stds 8.363→6.586; 4.962/83.667=5.9%; 386.60−307.83=78.77 | YES |
| §5 ladder line ≤248.67 < 302.87 < 307.83 < 386.60 | above | consistent | YES |
| §5 seeded DO: min improvement 11.79 vs ε=0.01; payoff oscillating [1611.56, 2316.61]; final mixture exploitable ≥407.50 (best deviation always_rest); specialist genome image exploitable +605 to +870/ep per position; support scores −258 to +237 | sd | 11.79 (iter 44) / ε=0.01; [1611.56, 2316.61]; +407.50 always_rest; per-position +604.91/+851.29/+869.56/+836.45; −258 to +237 | YES |
| §5 tier-1 battery context: ippo/influence/hca/lola gap_closed mean 0.005–0.125, all insufficient | t1 | 0.005 (lola) … 0.125 (ippo), 4× insufficient | YES |
| §5 vacuous historical gate: random alone mapped to gap_closed≈6.58, gate 0.49 | t1 notes | iter0 6.573→"≈6.58"; 0.49 gate | YES |
| §6: 12/12 frozen-scenario equilibrium coverage (11 converged + rest_trap annotated) | sd §closing / repo | consistent with sd RESULTS.md closing claim | YES |
| Abstract: 2304/8; 39-cell; 3–10×; 37-cell; two retired predictors; ≈80/step gap; one marginal variance-driven 16× escape | body + artifacts above | all consistent with the verified body numbers (v10 abstract no longer carries per-test statistics) | YES |
| NC-4 fix: §4 footnote — 4× label = iterations (200×4096 = 8× env steps); §5 ladder = true multiples at fixed 2048 | lb configs / te | 200·4096/(50·2048)=8; 200/800·2048 = 4×/16× | YES |

## Figure source-of-truth check (informational)

- `figures/recalibrated_heatmap.pdf` — source `figures/src/recalibrated_heatmap.py`
  present; render mtime ≥ source mtime. Rendered values re-extracted and
  matched to the HEAD JSON (see CF-1 table). **Not stale.**
- `figures/phase_diagram.pdf` — source `figures/src/phase_diagram.py` present;
  identical mtimes. **Not stale.**

## Cited-path existence check

All 16 committed artifact paths cited via `\path{}`/`\texttt{}` in the body
exist at HEAD (per_cell_baselines.json, per_cell.py,
entropy_vs_trainability.{py,json,md}, kstar_vs_trainability.{py,json,md},
noise_buydown_precision.md, improvability_oracle.md, registry.py,
docs/PARITY.md, docs/PAPER_RESULTS.md, baselines/parity.py,
rest_trap_seeded_do/RESULTS.md, scripted_battery/rest_trap.{json,md},
tier1_runs{,_trap_escape}, phase_diagram_ppo_longbudget,
paper/anvil_pub.bb-workshop.8/{changelog.md,stale_claims_audit.md},
rest_trap-v1.json), and `run_tier1_cell.classify_trap_verdict` exists.

## Discrepancies

None found. 0 numerical inconsistencies.
