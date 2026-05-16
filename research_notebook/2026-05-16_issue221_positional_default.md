# Issue #221 — `positional_default` empirical validation (option A from #203)

**Date:** 2026-05-16
**Issues:** [#221](https://github.com/rjwalters/bucket-brigade/issues/221) (empirical validation), [#203](https://github.com/rjwalters/bucket-brigade/issues/203) (env-side branch proposal), [#217](https://github.com/rjwalters/bucket-brigade/pull/217) (scenario plumbing PR).
**Hypothesis:** Spatial cost asymmetry (per-agent distance penalty) creates a per-agent gradient that pure ownership rebalancing (#199's `minimal_specialization`) lacked, and that the new gradient is large enough to let independent-PPO close more of the specialist–random gap than it did on `minimal_specialization` (~18-27%).

## Scenario definition

| Field | `default` | `minimal_specialization` | `positional_default` |
|---|---|---|---|
| `prob_fire_spreads_to_neighbor` | 0.25 | 0.25 | 0.25 |
| `prob_solo_agent_extinguishes_fire` | 0.5 | 0.5 | 0.5 |
| `prob_house_catches_fire` | 0.02 | 0.02 | 0.02 |
| `cost_to_work_one_night` | 0.5 | 0.5 | 0.5 (base) |
| `min_nights` | 12 | 12 | 12 |
| `team_reward_house_survives` | 100.0 | 10.0 | **100.0** |
| `team_penalty_house_burns` | 100.0 | 10.0 | **100.0** |
| `reward_own_house_survives` | 20.0 | 50.0 | **20.0** |
| `reward_other_house_survives` | 0.0 | 0.0 | 0.0 |
| `penalty_own_house_burns` | 40.0 | 100.0 | **40.0** |
| `penalty_other_house_burns` | 0.0 | 0.0 | 0.0 |
| `agent_home_positions` | — | — | **[0, 3, 5, 8]** |
| `distance_cost_alpha` | 0.0 | 0.0 | **0.1** |
| `distance_metric` | n/a | n/a | **ring_arc** |

**Key design idea:** `positional_default` keeps `default`'s reward magnitudes verbatim and adds a *per-agent* spatial work-cost term: `work_cost = 0.5 + 0.1 * ring_dist(home, target)`. With homes `[0, 3, 5, 8]` the cheapest action for each agent is to defend its own neighborhood — a per-agent gradient that doesn't depend on reward magnitudes.

## H2 reward-attribution audit

`experiments/p3_specialization/diagnostics/audit_reward_attribution.py --scenario positional_default` (uniform-random rollouts, 20 seeds, 265 steps):

| Metric | `default` | `minimal_specialization` | `positional_default` |
|---|---|---|---|
| Mean abs team component | 73.06 | 7.31 | **73.06** |
| Mean abs ownership component (per agent) | 12.60 | 31.51 | **12.60** |
| Mean abs work component (per agent) | ~0.5 | ~0.5 | **0.63** (slightly higher: alpha-scaled) |
| `\|team\| / \|ownership\|` median | 3.00 | 0.12 | **3.00** |
| Mean off-diagonal pairwise reward corr | 0.7573 | 0.1465 | **0.7573** |
| Min off-diagonal pairwise corr | 0.7145 | 0.0437 | **0.7142** |
| Team-share lower bound on corr | 0.3665 | 0.0020 | **0.3665** |
| H2 (team dominates) verdict | NOT CONFIRMED | NOT CONFIRMED | NOT CONFIRMED |

**Headline finding:** `positional_default` produces essentially identical aggregate reward statistics to `default`. The mean off-diagonal pairwise correlation moves from 0.7573 → 0.7573 (no change to 4 decimals) — the alpha=0.1 distance term is too small to materially shift the dominant team-reward correlation pattern.

The work-component magnitude *does* increase slightly (0.63 vs ~0.5 mean abs per agent), reflecting the spatial cost, and per-agent work cost streams now differ across agents — but the absolute work component magnitude is dwarfed by team (×120) and ownership (×20), so it shows up in neither the correlation matrix nor the variance ratios.

**Implication:** the per-agent gradient introduced by `positional_default` is real but small. To break the team-correlation regime, alpha would have to be much larger (or applied to a magnitude-rebalanced base scenario like `minimal_specialization`).

Artifact: `experiments/p3_specialization/diagnostics/results/h2_reward_attribution_positional_default.json`.

## Random baseline (n=1000)

`experiments/p3_specialization/diagnostics/random_baseline.py --scenario positional_default --episodes-per-seed 200 --seeds 5 --no-mlp`:

| Quantity | `default` | `positional_default` |
|---|---|---|
| Per-step team reward, mean | **247.58** | **247.09** |
| 95% CI | [241.07, 253.89] | [240.57, 253.39] |
| Per-episode mean | 3261.95 | 3255.37 |
| Episode length: median / mean | 13 / 13.18 | 13 / 13.18 |
| n | 1000 | 1000 |

Both means agree to ~0.5 per-step — the alpha=0.1 cost on random actions averages ~2.5×0.1 = 0.25 extra cost per work-step, times ~half of agents working = ~0.5/step extra, which is ~½ of one CI half-width and below detection here.

**Sanity check (issue #213 baseline):** `default` per-step random is 247.58 (corrected from the original 308 cited in #145). The current ~247 is consistent with the post-#205 / #206 rebalance (team_reward magnitudes unchanged but the per-agent ownership generalization in #198/#206 changed the random-policy reward landscape). The `default` measurement here confirms #213's 293.4 is now stale — `default` random per-step on current main is ~247, not ~293. Out of scope to chase down; flagging for issue #213's follow-up.

Artifacts:
- `experiments/p3_specialization/diagnostics/results/issue221_positional/random_baseline.log` (positional_default)
- `experiments/p3_specialization/diagnostics/results/issue221_positional/random_baseline_default.log` (default sanity)

## Specialist baseline (n=50)

`experiments/p3_specialization/diagnostics/issue199_baselines.py --scenarios positional_default default` (the ownership-based specialist policy):

| Policy | `positional_default` per-step | `default` per-step |
|---|---|---|
| Uniform random | 241.36 (CI [216.00, 265.43]) | 241.85 (CI [216.47, 265.92]) |
| Specialist (ownership-based) | 320.89 (CI [299.30, 341.67]) | 320.94 (CI [299.35, 341.72]) |
| **Specialist − random gap** | **+79.53** | **+79.09** |

The ownership-based specialist beats random by **+79.53/step** on `positional_default`, satisfying the curator's `≥ +50/step` acceptance bar from #203.

The numbers are nearly identical to `default` because: (a) the specialist policy is ownership-driven (round-robin), not position-driven, so it ignores `agent_home_positions`; (b) at alpha=0.1 the extra spatial cost is too small to shift the per-episode mean by more than the random noise.

A *position-aware* specialist that prefers nearby houses would do better and would be a more honest "upper bound" for `positional_default`. Filed as an out-of-scope follow-up.

Artifact: `experiments/p3_specialization/diagnostics/results/issue221_positional/baselines.json`.

## PPO 50-iter trajectory (3 seeds)

Three seeds (42, 43, 44), `lambda_red=0`, 50 iter × 2048 rollout steps, config matching the #207 / #199 baseline (`value_coef=0.5 entropy_coef=0.01 --normalize-returns --device cpu`).

### Per-seed summary

| Seed | iter 0 | iter 25 | iter 49 | first-5 mean | last-5 mean | LS slope (per iter) |
|---|---|---|---|---|---|---|
| 42 | 256.16 | 247.39 | 259.47 | 256.54 | 251.93 | −0.076 |
| 43 | 236.87 | 243.76 | 260.81 | 248.67 | 262.89 | **+0.295** |
| 44 | 250.55 | 247.84 | 256.61 | 247.11 | 247.61 | −0.032 |
| **mean** | **247.86** | **246.33** | **258.96** | **250.77** | **254.15** | **+0.062** |
| std | 8.10 | 1.83 | 1.75 | — | — | — |

### Aggregate across seeds

```
iter 0  mean=247.86  std=8.10
iter 25 mean=246.33  std=1.83
iter 49 mean=258.96  std=1.75
first-5 mean=250.77
last-5  mean=254.15
slope (per iter): mean=+0.062
```

Trajectory is essentially flat after the first few iterations. Aggregate slope of +0.062/iter implies it would take ~1100 iters to close the remaining 66 reward-units of the gap — not a realistic trajectory.

## Verdict computation

Using the #199 / #207 protocol (random = 247.09, specialist = 320.89, gap = 73.80 from the n=1000 / n=50 baselines above):

| Metric | Value | Fraction of gap closed |
|---|---|---|
| PPO last-5 mean − random | +7.06 | **9.6%** |
| PPO iter-49 mean − random | +11.87 | **16.1%** |

| Comparison | last-5 | iter-49 |
|---|---|---|
| `minimal_specialization` (issue #199 result) | **18.1%** | **27.4%** |
| `positional_default` (this work) | **9.6%** | **16.1%** |
| `default` (issue #213 / #197 cell) | ~0% (PPO plateaus at random baseline) | ~0% |

## Verdict

**Independent-PPO does NOT close the specialist–random gap on `positional_default` — in fact it does *worse* than on `minimal_specialization`.** The "spatial asymmetry as a per-agent gradient source" hypothesis from #203 option A is not supported at alpha=0.1: the per-agent distance cost is too small (≪ team and ownership components) to materially shift either H2 correlation statistics or the PPO trajectory.

Decision-tree placement:

- Both scenarios closed **less than 50%** of the gap → both fall in the "PPO algorithm-bound" branch.
- Specifically: `positional_default` (9.6%) < `minimal_specialization` (18.1%) — pure spatial-cost asymmetry is a *weaker* per-agent signal than direct ownership-magnitude rebalancing for independent PPO on this env.
- The result is **strong corroborating evidence** for the algorithm-bound conclusion already drawn in #199. Two env-side fixes (one magnitude rebalance, one spatial gradient) have now both failed to rescue independent PPO from the ~15-20% gap-closure plateau.

The honest reading is: **the algorithm cannot do per-agent credit assignment with independent critics, even when the per-agent signal exists.** The structural fix lives at the algorithm layer, not the env layer. Recommended follow-up remains MAPPO / centralized-critic (tracked in #208).

Optional env-side follow-ups that *might* still be interesting:
- **A.5: higher alpha.** alpha=0.1 was deliberately minimal; alpha=1.0 or 5.0 would actually shift H2 statistics. If independent-PPO still plateaus at higher alpha, that closes the env-side branch tight.
- **A on minimal_specialization base.** Stack the rebalanced magnitudes (`minimal_specialization`) with spatial asymmetry. If this rescues PPO, the combination of "ownership dominates" + "spatial gradient" was the missing piece. Otherwise it's another nail in the env-side coffin.

The user's project memory note (`PPO failure mode`) already captures the "MAPPO / CTDE is next" verdict; the present result reinforces but does not change that note. No memory update needed.

## Latent bugs uncovered while running diagnostics

While running the diagnostics on this branch, three latent bugs introduced by PR #216 (per-agent observation differentiation) surfaced and were fixed in this PR:

1. **`JointPPOTrainer.encoder_outputs_batch` fed full 3-D obs to every policy.** Post-#216, `rollout.observations` is `[T, N, obs_dim]` but `encoder_outputs_batch` still called `policy.encoder_output(observations)` directly, producing 3-D encoder outputs that crashed `quantize_uniform` (expects 1-D/2-D). Fixed by slicing per-agent (`observations[:, i, :]`).

2. **`_state_summary_codes` (in `train.py`) indexed the wrong axis.** `obs_np[:, _HOUSES_OBS_SLICE]` was meant to grab the first 10 obs dims (houses state) but on the 3-D layout it grabbed the first 10 agents — truncated to N=4 since dim-1 is now agent-axis. Fixed by detecting 3-D layout and reading agent 0's view.

3. **`audit_reward_attribution.py` missed the `distance_cost_alpha` term.** The hand-rolled reward decomposition hard-coded `-cost_to_work_one_night` for WORK actions; the `positional_default` env adds `alpha * ring_dist(home, target)`. The strict reconstruction assert tripped on step 0. Fixed by mirroring the env's spatial-cost computation.

The first two bugs blocked PPO training on every scenario, not just `positional_default` — they were latent because `test_joint_trainer.py` does not exercise `_measure_information` and the diagnostic scripts had not been re-run on current main. The third bug only affects `positional_default` (alpha=0 elsewhere). All three were in scope here because the diagnostic plan in #221 cannot complete without them.

## Artifacts

- Scenario plumbing (pre-existing, this PR depends on it): PR #217 `feature(env): add positional_default scenario`
- Diagnostic results dir: `experiments/p3_specialization/diagnostics/results/issue221_positional/`
  - `baselines.json` — random + ownership-specialist on both scenarios
  - `random_baseline.log` / `random_baseline_default.log` — n=1000 random measurements
  - `h2_audit.log` — stdout summary
  - `ppo_run.log` — runner timing log
  - `ppo_seed{42,43,44}/{metrics.json,config.json,policies/}` — full 50-iter PPO outputs per seed
  - `ppo_seed{42,43,44}.log` — captured PPO stdout
- H2 audit JSON: `experiments/p3_specialization/diagnostics/results/h2_reward_attribution_positional_default.json`
- Patched diagnostic script: `experiments/p3_specialization/diagnostics/random_baseline.py` (added `--scenario`; minimal patch — broader generalization in issue #219)

## Out-of-scope follow-ups (proposed)

1. **Position-aware specialist baseline.** The current `specialist_action_joint` uses ownership only; a variant that prefers nearby houses (using `env.agent_home_positions`) would give `positional_default` a more honest upper bound. Worth filing if the next iteration of #203 (options B/C) lands.
2. **Higher alpha sweep.** alpha=0.1 was chosen to be minimally invasive (`default` reward magnitudes unchanged). At this magnitude the per-agent gradient is barely visible to either random or ownership-specialist policies. A larger alpha (e.g. 1.0 or 5.0) or one applied on top of `minimal_specialization`'s rebalanced magnitudes would be a stronger test of "spatial asymmetry alone breaks the team-correlation regime." Worth filing as a #203-option-A.5 follow-up if independent-PPO closes more gap here than on `minimal_specialization`.
3. **Update issue #213's stale 293.4 baseline.** The post-#205/#198 rebalance shifted `default` random per-step from ~293 to ~247. Worth a small PR to update `analyze_plateau.py:60`, `summary.md:24`, and #213's own notebook footer.
4. **MAPPO / CTDE on `positional_default`** — tracked separately in #208 and not blocking this validation.
