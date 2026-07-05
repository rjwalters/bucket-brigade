# Numerical audit — anvil_pub.bb-workshop.9

Every quantitative claim in the abstract, §3–§6, the Conclusion, both table
bodies, and both figure captions was traced to the committed artifact the
paper (or `docs/PAPER_RESULTS.md`) names. Repo convention: every number
traces to a committed artifact. Verification below was performed by
re-reading the artifacts and, where the paper aggregates (class means,
separations, pooled columns), recomputing from the committed JSON.

Legend: ✓ = matches committed artifact at HEAD; ✓* = matches a **superseded
committed revision** of the artifact (HEAD regenerated since; see flag CF-1);
✗ = mismatch.

## 1. §4 / abstract — k\* coordination-threshold join (v9's new material)

Artifact: `experiments/nash/phase_diagram/kstar_vs_trainability.{json,md}` (HEAD).

| Text claim | Source value | Match |
|---|---|---|
| Primary test k\*=1 (3 cols) vs k\*≥2 (8 cols) on pooled NE gap: rank-biserial 0.00 | 0.00 (U = 12.0, exactly the null center) | ✓ |
| exact one-sided Mann–Whitney p = 0.539 (abstract: 0.54) | 0.5394 | ✓ |
| exhaustive 165-assignment permutation Δmean = −0.002, two-sided p = 0.994 | −0.0018, 0.9939 | ✓ |
| power floor: min achievable one-sided p = 0.006 (3 vs 8) | 0.0061 | ✓ |
| same split, homogeneous gap, 13 columns: rank-biserial 0.40, exact one-sided p = 0.19 | 0.40, 0.1853 | ✓ |
| post-hoc k\*=k_max=4 zone: 3 cols below all 10, rank-biserial −1.00, exact one-sided p = 0.0035 (combinatorial floor at 3 vs 10), permutation two-sided p = 0.021 | −1.00, 0.0035 (= floor), 0.0210; registration: post_hoc | ✓ |
| κ→k\* mapping: 0.1→4, 0.3–0.7→2, 0.9→1 (pure function of κ) | identical | ✓ |
| 15 effective columns, 13 joinable, 2 no-NE columns drop → 11 for NE-gap | identical | ✓ |
| Spearman k\* vs homogeneous gap ρ = −0.556 (p = 0.0004, n = 37 cells); ρ = −0.635 (p = 0.020, columns) | −0.556 / 0.0004 / 37; −0.635 / 0.0196 / 13 | ✓ |
| retired entropy comparison ρ = 0.109 / 0.342 (both n.s.) | 0.109 (p 0.560) / 0.342 (p 0.060) | ✓ |
| cross-tab: every k\* level mixes ≥2 verdict classes; asymmetric_only at all three k\* levels; all no_convergence in k\*=4 | table: asym 2/6/3 across k\*=1/2/4; nc 6 at k\*=4 only | ✓ |
| supersedes prior 2-vs-6 partial test p = 0.43 | noise_buydown_precision.md: p = 0.4286 | ✓ |
| **Issue #476 medians check**: v9 quotes no group median from the class_comparison display | grep confirms: only means, Δmeans, rank-biserials, exact/permutation p, ρ, power floors quoted | ✓ |

## 2. §4 — trainability sweep, entropy retirement, noise buy-down

Artifacts: `experiments/p3_specialization/phase_diagram_ppo_v2/recalibrated_verdict.json`,
`experiments/nash/phase_diagram/entropy_vs_trainability.{json,md}`,
`experiments/nash/phase_diagram/noise_buydown_precision.md`,
`experiments/p3_specialization/phase_diagram_ppo_longbudget/cell_*/cell_summary.json`.

**Supersession caveat (flag CF-1)**: `recalibrated_verdict.json` and
`entropy_vs_trainability.{json,md}` were regenerated at commit `a5b8ccdc`
(2026-07-04, the #456 20-seed buy-down) with n=20 means substituted on 8
cells. The paper's primary-sweep numbers match the **prior committed
revisions** (`73b49b08` for the verdict JSON; `22b1fda6` for the entropy
artifact) — the all-n=4 population §4 describes — not HEAD.

| Text claim | Source (revision) | Source value | Match |
|---|---|---|---|
| class means gap_closed_ne: sym 0.180 (n=11) > mixed 0.107 (n=9) > asym 0.059 (n=11); no_conv −0.049 (n=6, homogeneous fallback) | verdict JSON @ `73b49b08` (recomputed) | 0.180 / 0.107 / 0.059 / −0.049 | ✓* (HEAD gives 0.142 / 0.096 / 0.060 / −0.024) |
| homogeneous metric same ordering: 0.051 > 0.050 > 0.033 > −0.049 | same | 0.0507 / 0.0502 / 0.0328 / −0.0485 | ✓* (HEAD: 0.041 / **0.046** / 0.030 / −0.024 — **ordering sym>mixed inverts at HEAD**) |
| class-mean separations on homogeneous (0.001, 0.017, 0.082) | same | 0.0005 / 0.0174 / 0.0813; consistent as differences of the quoted rounded means | ✓* |
| per-class within-cell std ≈ 0.21–0.44 | same | mixed 0.21, asym 0.42, sym 0.44 | ✓* |
| 37 cells, 4 seeds, N=148 runs; skipped b0.10_k0.50_c0.50 + b0.10_k0.90_c0.50 | verdict JSON (both revisions) + Fig 2 n/a cells | identical | ✓ |
| 4×-budget no-convergence sweep worsens homogeneous mean to −0.108 (from −0.049) | longbudget cell summaries (HEAD, recomputed) | mean of 6 cells = −0.1079 | ✓ |
| entropy: 31 cells, mean h_cond vs gap_closed_ne ρ = 0.007 (p = 0.97) | entropy artifact @ `22b1fda6` | 0.007 / 0.970, n=31 | ✓* (HEAD headline: ρ = 0.109, p = 0.56) |
| "single nominally significant entry (spread vs homogeneous, p = 0.039)" fails Bonferroni α = 0.0063 | same | spread/homog −0.373, p = 0.039; α = 0.0063 | ✓* (HEAD: the nominal entry is **min** vs homog, p = 0.038) |
| gap_closed_ne varies across β by up to 0.36; e.g. −0.32/−0.11/−0.00 at (κ=0.1, c=1.0) | same | max range 0.36 at (0.3,1.0); (0.1,1.0) row −0.32/−0.11/−0.00 | ✓* (HEAD: −0.32/0.01/−0.00, max range 0.34) |
| per-cell std reaches 0.99 (median 0.24) | entropy artifact (HEAD) | 0.99 / 0.24 | ✓ |
| buy-down: CI half-widths shrink 2.00–3.64× (median 2.65), √5 ≈ 2.24; 1.84–3.59× homogeneous | noise_buydown_precision.md (HEAD) | identical | ✓ |
| 0/8 verdict flips, all remain insufficient; seeds 46–61 added to 42–45 | same | identical | ✓ |
| entropy retirement re-verified at n=20: ρ = 0.109, p = 0.56, all four aggregates insignificant (vs gap_closed_ne) | entropy artifact (HEAD) | mean 0.109/0.560, max 0.792, min 0.428, spread 0.195 | ✓ |
| largest per-cell shift 0.372 → 0.043 at (κ=0.3, c=1.0) | noise_buydown_precision.md | identical | ✓ |
| improvability oracle: +13.4/step, paired 95% CI [+10.5, +16.6] (≈14%) on c=0.5 no-convergence cells | improvability_oracle.md | +13.374 [+10.490, +16.628]; uniform base −92.9 → 14.4% | ✓ |
| v1 preview inversion 0.262 / 0.091 / −0.176 under old single-cell baseline | carried forward; verified against old-baseline columns by the v4/v5 audits | carry-forward | ✓ (accepted) |

## 3. §3 / Fig 1 / Table 1 — Nash phase diagram

Artifact: `experiments/nash/phase_diagram/results.json` (HEAD).

| Text claim | Source value | Match |
|---|---|---|
| 39-cell grid = 3β × 5κ × 3c − 6 unsampled c=0.5 cells | grid block: 45 − 6 = 39 | ✓ (count) — but see ✗ row below on *which* cells |
| verdict counts: sym 12, asym 11, mixed 10, no_conv 6 | verdict_counts identical | ✓ |
| 12 of 13 (κ,c) rows with ≥2 β samples verdict-identical; lone split (κ=0.9, c=0.5): β=0.1 mixed vs β∈{0.5,0.9} asymmetric | recomputed: 12/13; split row identical | ✓ |
| solver payoff noise 80.9 vs 72.0 at (κ=0.9, c=0.5) | best_team_payoff 80.915 / 72.0095 | ✓ |
| appendix: −614.4 vs −648.0 at (κ=0.5, c=0.5) | −614.433 / −648.015 | ✓ |
| Table 1 body: κ=0.1 → 6 collapse + 3 asym (0/9); κ=0.3 → 6 sym (6/6); κ=0.5 → 6 sym + 3 mixed (6/9); κ=0.7 → 6 asym (0/6); κ=0.9 → 7 mixed + 2 asym (7/9) | recomputed per-κ Counter: identical | ✓ |
| Table 1 caption: "row totals are 9 except at κ=0.7 (6 cells…)" | κ=0.3 row total is **also 6** (the table body itself shows 6/6) | ✗ **CF-2** |
| "six unsampled cells (high-κ × c=0.5 corner, subsumed by the c=1.0 row)" (§3 prose, Table 1 caption, Fig 1 caption) | the unsampled c=0.5 cells sit at κ∈{0.3, 0.7} (mid-κ); κ=0.9, c=0.5 IS sampled (it is the lone splitting row) | ✗ **CF-2** |
| predicted thresholds: collapse κ ≈ 0.028, symmetric band [0.030, 0.65], asymmetric onset κ ≳ 0.972; Ã = 36.24; A = 1812 | recomputed: 1/36.24 = 0.0276; A = 50·12+100·12+10·12/10 = 1812; f(0.03)≈0.0274, f(0.65)≈0.0279; g roots 0.028/0.972 | ✓ |
| state space 2304 (= 3²·2⁴·2⁴·… = 9·16·16); ≈9.44×10⁹ at H=10; action 8/40; joint 8⁴ = 4096 | arithmetic + env_spec notes | ✓ |

## 4. §5 / Table 2 — rest_trap anchors, budget ladder, seeded DO

Artifacts: `experiments/p3_specialization/scripted_battery/rest_trap.{json,md}`,
`experiments/p3_specialization/tier1_runs/tier1_verdict.md`,
`experiments/p3_specialization/tier1_runs_trap_escape/{4x,16x}/`,
`experiments/nash/rest_trap_seeded_do/RESULTS.md`,
`bucket_brigade/baselines/release/local/nash/rest_trap-v1.json`.

| Text claim | Source value | Match |
|---|---|---|
| frozen NE 3×FreeRider + 1×Firefighter, 2984.04/ep → ≤248.67/step (÷12) | team_payoff 2984.0437; profile label matches; 2984.04/12 = 248.67 | ✓ |
| always_rest ×4: 288.55 [285.20, 291.65], n=2000 | 288.548 [285.200, 291.649] | ✓ |
| uniform random 302.87 committed; 302.94 [301.46, 304.31] at n=10k; ±1.4/step noise | 302.87; 302.936 [301.464, 304.307]; half-width 1.43 | ✓ |
| het_ppo 1×: 306.26 [302.95, 309.33] → at_random; clears the point by 0.08; uplift +3.39 ± 7.34 | identical (tier1_verdict.md) | ✓ |
| het_ppo 16×: 307.83 [305.00, 310.71] → escaped_trap; clears anchor by +0.69 | identical (16x notes) | ✓ |
| specialist ×4: 386.60 [386.17, 387.03], n=10k paired; Δ +83.67 [+82.36, +84.89] | 386.603 [386.173, 387.026]; +83.667 [+82.355, +84.886] | ✓ |
| ladder: 4× het_ppo lo 304.03; ippo 4×/16× lo 301.77 / 303.46, both at_random | identical | ✓ |
| mean flat 306.26 → 307.66 → 307.83; 16×−1× = +1.57 ± 4.80 (t = 1.46), n.s. | identical | ✓ |
| uplift std 8.36 → 6.59; CI lo 302.95 → 304.03 → 305.00 | identical | ✓ |
| mean uplift +4.96 ≈ 6% of 83.7 headroom; best seed ≈27% of headroom, ≈61/step short of 386.60 | 16x notes: +4.96 ± 6.59; best seed 325.84 = +22.97 ≈ 27%, ≈ 61 short | ✓ |
| 300/300 bootstrap RNG/resample combinations; t-interval [304.67, 310.99] | PR #460 judge recomputation, recorded in the committed v8 paper trail (`anvil_pub.bb-workshop.8/changelog.md`, `.8/stale_claims_audit.md`); **no regenerable experiments/ artifact** | ✓ (note NC-5) |
| ladder statement ≤248.67 < 302.87 < 307.83 < 386.60; ≈79/step gap (386.60 − 307.83 = 78.77) | consistent | ✓ |
| vacuous historical gate: uniform-random alone mapped to gap_closed ≈ 6.58 under mis-scaled references | tier1_verdict.md: 6.58 (iter0 6.573) | ✓ |
| tier-1 battery on minimal_specialization: ippo/influence/hca/lola gap_closed 0.005–0.125, all insufficient, 3 seeds | 0.125/0.120/0.081/0.005 | ✓ |
| seeded DO: 50 iterations, min improvement 11.79 vs ε = 0.01; payoff oscillates [1611.56, 2316.61]; same basin as unseeded [1590.60, 2400.88] | RESULTS.md identical | ✓ |
| final mixture exploitable ≥407.50/ep, best deviation always_rest; rest-leaning deviations gain, work-leaning lose | +407.50, always_rest; free_rider +304.04 | ✓ |
| specialist genome image exploitable at every position, +605 to +870/ep | +604.91 / +851.29 / +869.56 / +836.45 | ✓ |
| cycling mechanism: support strategies score −258 to +237/ep vs mixture at verification budget | RESULTS.md identical | ✓ |
| 12/12 frozen-scenario coverage: 11 converged symmetric + rest_trap annotated | RESULTS.md line 148 | ✓ |
| escape robustness anchors (trap ladder rule): lo > scripted_best.ci95_hi / random_ci95_hi (304.31) / ne_bound | run_tier1_cell.classify_trap_verdict exists; anchors match | ✓ |

## 5. §6 — reproducibility pointers

| Claim | Check | Match |
|---|---|---|
| `per_cell_baselines.json`, `bucket_brigade/baselines/per_cell.py`, `envs/registry.py`, `docs/PARITY.md`, `docs/PAPER_RESULTS.md`, parity CLI, frozen IDs (`minimal_specialization-v1`, `rest_trap-v1`) | all exist and are git-tracked | ✓ |
| "the v2 recalibrated artifacts this paper reports reproduce byte-for-byte from the committed per-seed summaries" | at HEAD the committed per-seed summaries include seeds 46–61 on 8 cells; regeneration reproduces the **HEAD** artifact (n=20 blend), **not** the n=4 values §4 quotes | ✗ part of **CF-1** |
| Figure 2 "reads from …/recalibrated_verdict.json" (header comment) and its `gap_closed_ne` column "can be re-derived" | Fig 2's rendered per-cell values (extracted from the PDF) match the `73b49b08` revision; re-running `figures/src/recalibrated_heatmap.py` against HEAD JSON would change the β=0.5 column on 8 cells | ✗ part of **CF-1** |

## 6. Figure source-of-truth check (informational)

- `figures/phase_diagram.pdf` ↔ `figures/src/phase_diagram.py`: same mtime;
  rendered classes match `results.json` verdicts (spot-checked all four
  class regions + n/a hatching). Not stale.
- `figures/recalibrated_heatmap.pdf` ↔ `figures/src/recalibrated_heatmap.py`:
  same mtime — not stale by the mtime rule. However the script's declared
  data source (`recalibrated_verdict.json` at HEAD) no longer reproduces the
  rendered figure; see CF-1. The figure is *consistent with the paper text*
  (both quote the n=4-era revision); it is the on-disk artifact that moved.

## 7. Internal consistency (abstract ↔ body ↔ tables)

- Anchor ladder identical at every occurrence: abstract (386.6 / 306.3 /
  302.9 / ≥407.5 / 305.00 vs 304.31), §1 (386.60 / 302.87), Table 2, §5
  closing ladder (≤248.67 < 302.87 < 307.83 < 386.60). Consistent (roundings
  only). ✓
- k\* statistics identical across abstract, §1(3c), §4, §6-Threats,
  Conclusion (0.00 / 0.539–0.54 / −0.002 / −1.00 / 0.0035 / 0.006 floor). ✓
- Entropy/buy-down numbers identical across abstract, §1, §4, Conclusion
  (ρ = 0.007; 2.0–3.6×; 0/8 flips; ρ = 0.109 n=20). ✓
- Class means identical in abstract and §4 (0.180 / 0.107 / 0.059 / −0.049);
  −0.108 4×-budget figure identical in abstract, §1, §4, Conclusion. ✓
- 37/39-cell bookkeeping consistent everywhere (39 Nash cells, 37 PPO cells,
  class n = 11/9/11/6, mixed 9/37 in abstract = mixed n=9). ✓
- Sole internal contradiction found: Table 1 caption vs Table 1 body (CF-2).

## Non-critical numeric observations

- **"4×" naming**: §4's no-convergence budget sweep is 200 iter × 4096 steps
  (an 8× env-step multiple of the 50 × 2048 base; the "4×" counts iterations
  only), while §5's ladder "4×" is 200 × 2048 (a true 4× in env steps). Each
  matches its committed artifact spec; the shared label denotes different
  multipliers. Consider a footnote.
- Abstract quotes ρ = 0.007 (n=4 era) and 0.109 (n=20) without revision
  pinning; fully reconcilable only by reading §4 closely. Folded into CF-1's
  recommended fix.
