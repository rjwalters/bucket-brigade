# Issue #231 — MAPPO Phase 2: paired IPPO vs MAPPO across three scenarios

**Date:** 2026-05-16
**Issue:** [#231](https://github.com/rjwalters/bucket-brigade/issues/231)
**PR (this commit):** TBD — see PR body for tmux session names and result status.

## Hypothesis

The independent-PPO baseline plateaus at ~15 % of the specialist-random gap on
`minimal_specialization` (project memory, "PPO failure mode"). The two env-side
interventions tested so far have both **failed** the pre-registered ≥50 %
gap-closure bar:

* Per-agent obs differentiation (#216 / PR #220): closes ~15-20 % — not the
  bottleneck (per `research_notebook/2026-05-16_issue220_obsfix_validation.md`).
* Positional reward shaping (PR #228, alpha = 0.1 on `positional_default`):
  closes 9.6 %/16.1 % — likewise sub-bar.

Algorithmic intervention is the next natural step. MAPPO Option A
(centralized critic, decentralized policies; landed in PR #225) is the
canonical CTDE upgrade. This experiment asks: **does sharing the value
baseline across agents close the gap on at least two of three P3 scenarios?**

## Arms

- **IPPO** (current behavior): per-agent critic, no information sharing.
- **MAPPO** (`--centralized-critic`): shared critic over the global-obs
  portion (identity tail stripped). Per-agent rewards still drive per-agent
  advantages — only the value baseline is shared. `redundancy_coef` is gated
  to zero (both arms run `--lambda-red 0.0`).

## Protocol

18 cells = 2 arms × 3 scenarios × 3 seeds, pinned to `main` HEAD
`89ed2ebb` (`diag(p3): re-derive chain_reaction random baseline`).

| Field | Value |
|---|---|
| scenarios | `default`, `minimal_specialization`, `positional_default` |
| arms | `ippo`, `mappo` (`--centralized-critic`) |
| seeds | 42, 43, 44 |
| num_iterations | 100 |
| rollout_steps | 2048 |
| num_agents | 4 |
| value_coef / entropy_coef / normalize_returns | 0.5 / 0.01 / on |
| lambda_red | 0.0 |
| device | cpu (env is CPU-bound) |

All cells launched on `COMPUTE_HOST_PRIMARY` (robbs-mac-studio) in 6 parallel
tmux sessions, one per (arm, scenario) pair, each running 3 cells sequentially:

```
issue231-ippo-default       issue231-mappo-default
issue231-ippo-minspec       issue231-mappo-minspec
issue231-ippo-positional    issue231-mappo-positional
```

Curator estimate: ~6 min/cell × 3 seeds = ~18 min per session, ~20-25 min
total wall-clock with the 6 sessions in parallel. (Mac Studio has 28
physical cores, so the original 4-/6-way restriction in the issue body was
loose; all 6 sessions launched concurrently.)

A pre-flight smoke test (5 iters, 512 steps, `--centralized-critic`) ran
cleanly before the sweep, confirming the MAPPO code path on current `main`.

## References

Per-step team reward baselines used for gap-closure
(`experiments/p3_specialization/analyze_231.py`):

| scenario | random | specialist | spec − random |
|---|---|---|---|
| default | +241.85 | +320.94 | +79.09 |
| minimal_specialization | −96.07 | −22.07 | +73.99 |
| positional_default | +241.36 | +320.89 | +79.53 |

Sources: `diagnostics/results/issue199_minspec/baselines.json` and
`diagnostics/results/issue221_positional/baselines.json`.

## Pre-registered verdict ladder

Per-scenario gap-closed delta `(mappo_gap − ippo_gap)`:

| Delta | Per-scenario tier |
|---|---|
| ≥ 0.50 | tier 1 — MAPPO succeeds for that scenario |
| 0.25–0.50 | tier 2 — partial / helps but does not solve |
| < 0.25 | tier 3 — insufficient |
| < −0.10 | harmful — block headline; investigate |

Headline verdict:

| Trigger | Outcome |
|---|---|
| ≥2 of 3 scenarios at tier 1 | `MAPPO_SUCCEEDS` → update PPO failure-mode memory; deprioritize #193 |
| All 3 at tier 2 | `MAPPO_HELPS_GLOBALLY_NOT_DECISIVE` → longer training / alpha sweep follow-on |
| Some tier 1 / some tier 3 | `PARTIAL_SCENARIO_DEPENDENT` → emit per-scenario verdicts |
| All 3 at tier 3 | `INSUFFICIENT` → promote #193 reward shaping to next priority |
| Any scenario harmful | `MAPPO_HARMFUL_ON_<SCEN>` → block headline; investigate critic instability |

## Results

All 18 cells completed on `COMPUTE_HOST_PRIMARY` at 2026-05-16T10:04Z
(~14 min wall-clock; 6 sessions in parallel). Full per-cell table is
in `experiments/p3_specialization/diagnostics/results/issue231_mappo/summary.{md,json}`.

### Headline numbers (trailing-5 team-reward, 3 seeds per cell)

| scenario | IPPO mean ± std | MAPPO mean ± std |
|---|---|---|
| default | +253.70 ± 2.38 | +247.18 ± 11.50 |
| minimal_specialization | −91.71 ± 7.38 | −94.52 ± 5.62 |
| positional_default | +253.19 ± 4.00 | +251.15 ± 5.04 |

### Gap-closed deltas

| scenario | gap_ippo | gap_mappo | delta (mappo−ippo) | per-scenario tier |
|---|---|---|---|---|
| default | +0.150 | +0.067 | **−0.083** | `tier_3_insufficient` |
| minimal_specialization | +0.059 | +0.021 | **−0.038** | `tier_3_insufficient` |
| positional_default | +0.149 | +0.123 | **−0.026** | `tier_3_insufficient` |

### Headline verdict: `INSUFFICIENT`

All 3 scenarios at tier 3 (`delta < 0.25`). MAPPO does **not** close the
PPO learning gap; in fact, the centralized critic produces a *slight* mean
underperformance on every scenario (mean delta = −0.049 across the three
gap-closure metrics), though every per-scenario delta is well above the
`< −0.10` "harmful" threshold so we do **not** flag MAPPO as actively
harmful — the differences are within seed-level noise.

Notable secondary findings (full table in `summary.md`):

- **Mean-entropy collapse on MAPPO**: on `default` and `positional_default`
  MAPPO drives mean-action-entropy *lower* than IPPO (1.50 vs 2.29 on
  `default`; 1.78 vs 2.17 on `positional_default`) without translating
  that lower entropy into higher reward — the shared critic is pulling all
  agents toward a narrower action distribution, not a better one. On
  `minimal_specialization` the gap is much smaller (2.77 vs 2.83).
- **Pairwise action KL** is unavailable for this PR because
  `experiments/p3_specialization/diagnostics/pairwise_action_kl.py`
  has a stale `from bucket_brigade.env import BucketBrigadeEnv` import
  that doesn't resolve on current `main` (module is now
  `bucket_brigade.envs.bucket_brigade_env`). Per builder scope discipline
  this is a separate issue and is filed for follow-up; the headline
  team-reward + gap-closed measurement does not depend on it.

### Implications per the pre-reg ladder

- **CTDE alone is insufficient.** The independent-PPO plateau is not
  primarily a value-function-baseline problem. Sharing the critic does
  not change the asymptotic team reward on any of the three scenarios.
- **Promote #193 (team-vs-ownership reward rebalancing) to next priority.**
  The reward signal itself is now the prime suspect — both env-side fixes
  (#220 obs-fix, #228 positional shaping) and the algorithm-side fix
  (this PR's MAPPO sweep) have all returned tier-3 results on the same
  scenarios. The bottleneck is upstream of the algorithm.
- **Cheap MAPPO follow-ups (alpha sweep, longer training) are not the
  natural next step** given the cross-scenario consistency of the
  negative result. A reward-shaping intervention should run first.

## Project-memory update

The "PPO failure mode" note in user memory is updated alongside this PR to
reflect that **MAPPO does not close the gap** and that #193 reward
shaping is now the next intervention to try (rather than further CTDE
ladder rungs like QMIX or value mixing).
