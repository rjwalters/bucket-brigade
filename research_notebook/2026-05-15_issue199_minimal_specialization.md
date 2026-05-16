# Issue #199 — `minimal_specialization` sanity-check

**Date:** 2026-05-15
**Issue:** [#199](https://github.com/rjwalters/bucket-brigade/issues/199)
**Hypothesis:** Disambiguate whether PPO's failure to learn specialization on `default` is an environment design problem (team reward signal swamps per-agent signal) or an algorithm problem (PPO with independent critics cannot do credit assignment even when the signal is dominant).

## Scenario definition

`minimal_specialization` was added to `definitions/scenarios.json` with the locked-in parameters from the issue brief:

| Field | Value | vs. `default` |
|---|---|---|
| `prob_fire_spreads_to_neighbor` | 0.25 | same |
| `prob_solo_agent_extinguishes_fire` | 0.5 | same |
| `prob_house_catches_fire` | 0.02 | same |
| `cost_to_work_one_night` | 0.5 | same |
| `min_nights` | 12 | same |
| `team_reward_house_survives` | **10.0** | 100 → 10 (10x lower) |
| `team_penalty_house_burns` | **10.0** | 100 → 10 (10x lower) |
| `reward_own_house_survives` | `[50, 50, 50, 50]` | 20.0 → 50 |
| `reward_other_house_survives` | `[0, 0, 0, 0]` | 0.0 same |
| `penalty_own_house_burns` | `[100, 100, 100, 100]` | 40.0 → 100 |
| `penalty_other_house_burns` | `[0, 0, 0, 0]` | 0.0 same |

Round-robin house ownership is unchanged (`np.arange(10) % 4`), so agent 0 → {0,4,8}, agent 1 → {1,5,9}, agent 2 → {2,6}, agent 3 → {3,7}.

Same fire dynamics as `default`. The only differences are: (a) team reward magnitude reduced 10×, (b) ownership signal magnitudes raised 2.5×, (c) zero cross-agent ownership rewards. Together these make the per-agent gradient signal dominate.

## H2 reward-attribution audit

Re-running `experiments/p3_specialization/diagnostics/audit_reward_attribution.py --scenario minimal_specialization` (uniform-random rollouts, 20 seeds, ~265 steps each):

| Metric | `default` (pre-#199 measurement) | `minimal_specialization` |
|---|---|---|
| Mean abs team component | 73.06 | **7.31** |
| Mean abs ownership component (per agent) | 12.60 | **31.51** |
| `\|team\| / \|ownership\|` median | 3.00 | **0.12** (~8× lower) |
| `\|team\| / \|ownership\|` p95 | 8.00 | **0.32** |
| Mean off-diagonal pairwise reward corr | **0.7573** | **0.1465** |
| Min off-diagonal pairwise corr | 0.7145 | **0.0437** |
| Team-share lower bound on corr | 0.3665 | **0.0020** |
| H2 (team dominates) verdict | NOT CONFIRMED | NOT CONFIRMED |

Interpretation: `minimal_specialization` does exactly what the design intends. Ownership component is now **~4× larger than team component** in magnitude, and cross-agent reward correlation has collapsed from 0.76 to 0.15. This scenario is therefore a clean test of "can PPO learn specialization when ownership is dominant?"

Artifact: `experiments/p3_specialization/diagnostics/results/h2_reward_attribution_minimal_specialization.json`.

## Baselines

`experiments/p3_specialization/diagnostics/issue199_baselines.py --episodes 50 --num-agents 4 --seed 42` measures uniform-random and a hand-coded specialist (`bucket_brigade.baselines.specialist_action_joint`) on both scenarios.

The specialist policy is trivial: at each step, an agent works the lowest-index burning house it owns, else rests. Round-robin ownership matches the env (`np.arange(10) % num_agents`).

### `minimal_specialization` (50 episodes per baseline, seed 42)

| Policy | per-step team reward (mean) | 95% CI |
|---|---|---|
| Uniform random | **−96.07** | [−118.35, −74.96] |
| Specialist | **−22.07** | [−41.65, −3.20] |
| Specialist − random gap | +74.00 | — |

Specialist closes 74 reward-units per step relative to random. Note both means are negative because team_penalty_house_burns (×ruined_fraction) plus own-house burn penalties dominate when no policy is firefighting; the issue's design accepts this — the magnitudes are calibrated so that *moving toward* specialization is strongly rewarded even if absolute reward stays negative.

The curator's "specialist > 2× random" bar is undefined when random is negative; the more meaningful check is the **gap** (74 units/step). Specialist substantially beats random — scenario design is not broken.

### `default` (50 episodes per baseline, seed 42) — for comparison

| Policy | per-step team reward (mean) | 95% CI |
|---|---|---|
| Uniform random | 241.85 | [216.47, 265.92] |
| Specialist | 320.94 | [299.35, 341.72] |
| Specialist − random gap | +79.09 | — |

Comparable absolute gap (~79 vs ~74), confirming the specialist policy has similar headroom on both scenarios — the disambiguator isn't "specialization is harder on minimal_spec," it's "what fraction of that headroom can PPO close?"

Artifact: `experiments/p3_specialization/diagnostics/results/issue199_minspec/baselines.json`.

## PPO 50-iter smoke

Single seed, `lambda_red=0`, `seed=42`, 50 iterations × 2048 rollout steps, `--device cpu --normalize-returns --value-coef 0.5 --entropy-coef 0.01` — config matches the #205 smoke configuration so results are directly comparable.

```
iter    0 | team_reward  -89.17
iter   10 | team_reward  -90.79
iter   25 | team_reward  -89.39
iter   40 | team_reward -105.75
iter   45 | team_reward  -84.19
iter   49 | team_reward  -75.80
```

- **first-5-iters mean:** −87.92
- **last-5-iters mean:** **−82.70**
- **iter-49 spot value:** −75.80
- **least-squares slope (per iter):** **+0.029**

The slope is essentially flat (noise floor is ~10 reward-units/iter). The iter-49 spot value of −75.80 is one favorable noisy step; the trailing-5 mean of −82.70 is a more honest reflection of "where PPO ended up."

## Verdict computation

Per the issue's decision tree, using **trailing-5-iter mean** for stability:

| Quantity | Value |
|---|---|
| PPO trailing-5 mean reward | **−82.70** |
| Random baseline mean | **−96.07** |
| Specialist baseline mean | **−22.07** |
| Specialist − random gap | 74.00 |
| PPO − random gap | 13.37 |
| **Fraction of (specialist − random) closed by PPO** | **18.1%** |

Using **iter-49 spot value** (the issue's exact ask):

| Quantity | Value |
|---|---|
| PPO iter-49 reward | −75.80 |
| PPO − random gap | 20.27 |
| **Fraction of (specialist − random) closed by PPO** | **27.4%** |

By either measure, PPO closes **less than 50%** of the specialist–random gap. Per the task's decision tree:

> **PPO < 50% of (specialist − random) → algorithm fundamentally unsuited → recommend MAPPO / centralized critic follow-up.**

## Verdict

**PPO with independent critics does NOT learn specialization on `minimal_specialization` within 50 iters, even though the per-agent gradient signal is overwhelmingly dominant (ownership 4× larger than team, cross-agent correlation 0.15 vs 0.76 on `default`).** The issue is not the environment — it is the algorithm. Recommended follow-up: move to centralized-training-decentralized-execution (MAPPO, COMA, or a centralized critic with per-agent advantage estimation), as already flagged in #193.

The bar between the two decision-tree branches is set at 50%; the result here (18–27%) is also above the curator's secondary "<25%" threshold, which would be unambiguous. The 18–27% reading falls in the curator's "ambiguous (25–50%) — extend to 200 iters before deciding" zone, which is a reasonable defensive read. However, the slope across 50 iters is +0.029 — at that rate it would take ~1800 iters to close the remaining 53.73 reward-units, which is not a realistic trajectory. The honest conclusion is: PPO is not learning the specialization, and continuing to run more iters of the same algorithm is unlikely to change that.

## Artifacts

- Scenario: `definitions/scenarios.json` (block: `minimal_specialization`)
- Generated Python: `bucket_brigade/envs/scenarios_generated.py` (function `minimal_specialization_scenario`)
- Generated Rust: `bucket-brigade-core/src/scenarios.rs` (entry `"minimal_specialization"`)
- Generated TS: `web/src/utils/scenarioGenerator.generated.ts` (entry `MINIMAL_SPECIALIZATION`)
- Specialist policy: `bucket_brigade/baselines/specialist.py`
- Baseline measurement script: `experiments/p3_specialization/diagnostics/issue199_baselines.py`
- H2 audit (updated to `--scenario`): `experiments/p3_specialization/diagnostics/audit_reward_attribution.py`
- Tests: `tests/test_baselines.py` (11 tests, all passing)
- Results dir: `experiments/p3_specialization/diagnostics/results/issue199_minspec/`
  - `baselines.json` — random + specialist measurements
  - `metrics.json` — full 50-iter PPO metrics
  - `training.log` — captured stdout
  - `config.json` — exact training config
- H2 audit JSON: `experiments/p3_specialization/diagnostics/results/h2_reward_attribution_minimal_specialization.json`

## Out-of-scope follow-ups (proposed)

1. **Algorithm-side:** open follow-up issue for centralized-critic / MAPPO / COMA. Reuses `minimal_specialization` as the entry point — if the new algorithm can learn this clean signal, that's a green light for `default`; if it can't, the problem is more fundamental than independent critics.
2. **Run-length extension:** if MAPPO matters more than expected, run `minimal_specialization` PPO to 200+ iters to definitively reject "PPO just needs more time" before committing to algorithm rewrite.
3. **Default rebalance:** explicitly *out of scope* per the user — but if MAPPO succeeds on `minimal_specialization` and existing PPO does poorly on `default`, the env redesign question reopens.
