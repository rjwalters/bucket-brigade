---
title: P3 Plateau — 3-Lever Ablation (Issue #174)
date: 2026-05-15
status: complete; acceptance criterion not met; recommend follow-up
issue: https://github.com/rjwalters/bucket-brigade/issues/174
companion_diagnostics: ../experiments/p3_specialization/diagnostics/summary.md
companion_results: ../experiments/p3_specialization/results_174_ablation/summary.md
---

# P3 Plateau — 3-Lever Ablation (Issue #174)

## TL;DR

The pre-registered acceptance criterion of issue #174 — *at least one condition with reward CI lower bound > 320* — is **not met**. None of the six conditions clears even the 308 random baseline. The three diagnosed pathologies (value-loss dominance, entropy collapse, plateaued reward) were addressable, but **fixing them individually or in pairs is not sufficient to unlock learning**.

A small positive slope in `L1_norm` (+0.173 reward/iter over iters 25–50) and `L1L2` (+0.125) is the only signal hinting at follow-up. Recommendation: defer the defaults change and run a long-horizon (≥500 iter) test of `L1`-anchored configs as a separate issue.

## Headline numbers

5 seeds (42..46) × 50 iters × 2048 rollout steps, scenario `default`, λ_red=0.
Bootstrap 95% CI over seeds for final-iteration mean step-reward.

| condition | norm | vc | ec | reward iter49 (95% CI) | Δ vs rand | crosses bar (>320)? | ent49 | vc·v_loss/\|p_loss\| 0→49 | slope iter25→50 |
|---|---|---|---|---|---|---|---|---|---|
| baseline    | F | 0.50 | 0.01 | 296.66 [290.65, 302.66] | −11.34 | no | 0.346 | 6.1e+06 → 2.7e+06 | +0.090 |
| L1 (norm)   | T | 0.50 | 0.01 | 298.83 [292.47, 305.18] | −9.17  | no | **1.807** | **6.7e+02 → 2.0e+01** | **+0.173** |
| L2 (low vf) | F | 0.05 | 0.01 | 295.66 [290.35, 301.53] | −12.34 | no | 0.413 | 6.1e+05 → 5.8e+05 | −0.008 |
| L3 (high ent) | F | 0.50 | 0.10 | 296.14 [290.75, 301.06] | −11.86 | no | 2.354 | 6.5e+06 → 2.8e+06 | +0.066 |
| L1+L2       | T | 0.05 | 0.01 | 298.10 [292.23, 302.94] | −9.90  | no | 1.645 | 6.4e+01 → **5.8e+00** | +0.125 |
| L1+L3       | T | 0.50 | 0.10 | 297.46 [293.59, 301.30] | −10.54 | no | **3.983** | 7.7e+02 → 5.8e+01 | +0.074 |

Random per-step team-reward baseline on `default` = 308 (from #145). Acceptance bar = 320 (CI lower bound).

See `experiments/p3_specialization/results_174_ablation/reward_trajectories.png` and `gradient_ratios.png` for the per-iter trajectories.

## What the levers actually did

The instrumentation worked exactly as the diagnostic in `experiments/p3_specialization/diagnostics/summary.md` predicted.

**L1 (return normalization) crushes the value-loss dominance.** Baseline iter-49 ratio is `vc · v_loss / |p_loss| ≈ 2.7e+06`; adding return normalization brings it to `2.0e+01` — five orders of magnitude. L1+L2 takes it further to `5.8e+00`, the most balanced gradient of the six.

**L1 also prevents entropy collapse.** Baseline final-iter action entropy is `0.346` nats; L1 holds it at `1.807`. L1+L3 reaches `3.983` nats — the policy stays broadly exploratory.

**L2 alone barely moves the dominance.** Going from `vc=0.5` to `vc=0.05` only shifts the iter-49 ratio from `2.7e+06` to `5.8e+05` (one decade). The value-loss magnitude is so much larger than the policy-loss magnitude that a 10× drop in the coefficient leaves the value term still ~10⁶× dominant. To bring the ratio anywhere near unity by `vc` alone, the coefficient would need to be ~10⁻⁵, well outside the #174 range.

**L3 alone is mechanically inert.** Without normalization, the entropy bonus is dwarfed by the value-loss gradient. Entropy ends at 2.354 (vs baseline 0.346, so the bonus did do *something*), but the action-distribution stays uniform because the policy gradient never gets a chance to shape it.

## And yet, no condition learns

Every condition ends within a 296–299 reward band. None crosses 308 (random), let alone 320.

The reward trajectory shape is the same in every condition:

| condition | r[0] | r[10] | r[25] | r[49] | argmax iter | max reward |
|---|---|---|---|---|---|---|
| baseline    | 291.84 | 289.85 | 293.37 | 296.66 |  7 | 299.45 |
| L1_norm     | 291.84 | 292.15 | 292.26 | 298.83 |  7 | 299.71 |
| L2_low_vf   | 291.84 | 293.00 | 293.21 | 295.66 |  7 | 300.16 |
| L3_high_ent | 291.84 | 290.08 | 294.54 | 296.14 | 47 | 297.75 |
| L1L2        | 291.84 | 291.87 | 293.18 | 298.10 | 46 | 298.88 |
| L1L3        | 291.84 | 291.15 | 293.76 | 297.46 |  7 | 300.44 |

All conditions start at 291.84 (random-init policy is slightly biased below uniform), reach roughly 299–300 within the first ~7 iters, and then drift upward at a small positive rate of 0.07–0.17 reward/iter for the remaining ~40 iters. The reward never crosses 308 in any seed of any condition.

## Interpretation

The three hypotheses from #145 that #174 was designed to test:

| hypothesis | result |
|---|---|
| Value-loss dominance starves the policy gradient | **Falsified as sufficient cause.** L1+L2 reduces the ratio to ~6× and reward still doesn't move. |
| Entropy collapse locks the policy in early | **Falsified as sufficient cause.** L1+L3 maintains 3.98 nats of entropy and reward still doesn't move. |
| Some combination of the two fixes the plateau | **Falsified.** No combination of the three pre-registered levers crosses the random baseline. |

The plateau has a deeper cause than the gradient pathologies the diagnostics identified. The candidates remaining from #145 (now better-motivated by this data):

1. **Horizon.** L1_norm's +0.173 reward/iter slope, if held linearly, would reach 380 at iter 500. Even if the slope decays, the trajectory is non-flat in the right direction. 50 iters may simply be too short to see PPO escape this basin.
2. **Reward attribution structure** (#135). The Rust core emits only per-step work/rest cost; house outcomes only appear in `compute_final_rewards()`. If the per-step signal that PPO consumes is mostly noise relative to the terminal signal, the value head fits noise and the policy doesn't get useful credit assignment.
3. **GAE temporal discount mismatch.** With γ=0.99 and ~13 environment steps per episode, terminal rewards are heavily attenuated by the time they reach the per-step value targets. Either γ or rollout horizon may need adjustment.
4. **Algorithm mismatch.** PPO with this advantage-estimation setup may not be the right tool — random uniform-over-actions outperforms a trained PPO policy on `default`, which is unusual and may indicate a structural rather than hyperparameter problem.

None of these is testable from the data already on disk. They are follow-up work.

## Recommendation — defaults change

**No defaults change recommended at this time.** Issue #174's acceptance criterion required identifying one specific defaults change that produces a reward CI lower bound > 320. No such configuration was found.

In particular: do *not* set `normalize_returns=True` as the new default based on this data alone. The mechanical effect of L1 is real and matches theory, but it does not by itself produce learning beyond random. Changing the default to `True` would only encode an unverified prior into the codebase.

`CellConfig` defaults stay as-is. Authors of future P3 sweeps should set `normalize_returns=True` as a working hypothesis but treat it as an open question pending follow-up.

## What this unblocks (and what it doesn't)

- **#176 (validate state-summary CMI conditioner)** is still blocked. Until policies actually learn the task, the CMI conditioner cannot be tested for the property that conditioning on a sufficient state summary makes `I(Ẑ_i;Ẑ_j|S) < I(Ẑ_i;Ẑ_j)`. The state-summary infrastructure landed in #168; it's ready when training is.
- **P3 v2 sweep (240 cells overnight)** is *not* yet justified. Re-running the 3 × 4 × 20 grid with these hyperparameters would reproduce the May 14 result with cleaner gradients but no actual learning.
- **Phase 3 follow-up** is now the gating work — see "Recommended next" below.

## Recommended next (separate issue)

**P3 plateau Phase 3: long-horizon ablation of L1-anchored configs.** Specifically:

| axis | values |
|---|---|
| scenario | `default`, `chain_reaction` |
| λ_red | 0 |
| condition | `L1_norm`, `L1L2`, `L1L3` |
| seeds | 42..46 (5) |
| num_iterations | 500 |
| rollout_steps | 2048 |

That's 2 × 3 × 5 = 30 cells × ~10× the per-cell wall clock (since iters dominate) ≈ 5 hours on Mac Studio CPU, ~30 min on an alc-cluster node at 8-way parallel. Acceptance: same bar, CI lower bound > 320 in at least one condition.

If Phase 3 also fails to cross 320, the answer is *not* in the PPO hyperparameter space and the next step is one of:

- Re-examine the reward-attribution path through `compute_final_rewards()` (#135 territory).
- Try a different advantage estimator (e.g., raw n-step returns, V-trace).
- Try a different algorithm (REINFORCE with baseline, A2C, IMPALA).

## Reproducibility

- Sweep script: `experiments/p3_specialization/run_174_ablation.sh`
- Analysis script: `experiments/p3_specialization/analyze_174.py`
- Sweep grid: 6 conditions × `default` × λ=0 × 5 seeds × 50 iters × 2048 rollout × 4 PPO epochs
- Per-cell wall clock: ~100 s on Mac Studio CPU (~63 s estimated in #174; revise upward for future planning)
- Total sweep: ~60 min wall clock (14:52:49 → 15:52:46 UTC, 2026-05-15)
- Per-cell outputs (gitignored): `experiments/p3_specialization/runs/p3_174_ablation/{condition}/default/lambda_0e0/seed_{N}/{metrics.json,config.json,policies/}`.
- Aggregated outputs (committed): `experiments/p3_specialization/results_174_ablation/{summary.json,summary.md,reward_trajectories.png,gradient_ratios.png}`.
- Branch: `feature/issue-174`. Worktree: `.loom/worktrees/issue-174`.

## Amendment 2026-05-15

**Trigger:** Issue #202 baseline audit + H3 diagnostic (#192 / PR #196).

**What changed.** The `default` random per-step team-reward baseline used
throughout this notebook (308) and the derived #174 acceptance bar (320 =
308 + 12) were both built on an uncommitted n=50 measurement from issue
#145. PR #196 re-derived the random baseline at n=1000 episodes across 5
seeds: **293.39 with 95% bootstrap CI [288.87, 297.78]**. The cited 308
sits outside this CI. The 320 acceptance bar was therefore ~26 reward too
high relative to the corrected baseline (and was anyway an arbitrary "+12
buffer" with no statistical justification).

Per issue #202's hybrid policy, `analyze_174.py` now uses:

- `RAND_BASELINE = 293.4` (re-derived mean)
- `ACCEPTANCE_BAR = 297.78` (re-derived CI upper bound — "statistically
  distinguishable from random," not "exceeds an arbitrary buffer")

The regenerated `results_174_ablation/summary.md` and `summary.json`
reflect the new constants.

**Effect on this notebook's conclusions.** None of the qualitative
conclusions change:

- *Acceptance criterion not met.* The original criterion ("CI lower bound
  > 320") is not met by any condition. The corrected criterion ("CI lower
  bound > 297.78") is also not met by any condition — the best lower bound
  is L1L3's 293.59, which barely clears the corrected random mean (293.4)
  but still falls below the corrected acceptance bar.
- *No condition unlocked learning.* All conditions land in the 296–299
  band, which under the corrected baseline is "at random ± a few points,"
  not "below random." The "below random" framing in the TL;DR ("None of
  the six conditions clears even the 308 random baseline") should be read
  as "None of the six conditions is statistically distinguishable from
  random under the corrected n=1000 baseline."
- *L1_norm slope is still the only positive signal.* +0.173 reward/iter
  over iters 25–50 is unchanged; the long-horizon Phase 3 recommendation
  still stands.
- *Defaults change recommendation.* Still "no change recommended at this
  time" — corrected or not, no condition clears the new acceptance bar.

**Why an amendment and not an edit.** The body above is the dated research
record. Silent edits would erase the provenance of the original analysis.
Future readers should see the original 308/320 numbers and this footer
together.
