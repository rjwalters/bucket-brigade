# Issue #220 — Obs-fix empirical validation (paired comparison)

**Date:** 2026-05-16
**Issue:** [#220](https://github.com/rjwalters/bucket-brigade/issues/220)
**PR (this commit):** TBD — kickoff PR; results pending tmux completion on `COMPUTE_HOST_PRIMARY`.

## Hypothesis

Does fixing the latent obs-aliasing bug (PR #216 / commit `a38667b5`) measurably
improve independent-PPO learning on the P3 scenarios? Per-cell unit tests for
the fix already pass; this experiment is the empirical follow-up the Judge and
Builder of #216 explicitly deferred to a compute host.

## Arms

- **Baseline (pre-#216):** commit `19afcd76` — last commit before the obs-fix
  landed (`feat(env): generalize Scenario ownership reward fields to per-agent
  vectors`). `flatten_dict_obs` does **not** take an `agent_id` argument; all
  N policies see the same flat row.
- **Treatment (post-#216):** commit `a38667b5` (the obs-fix landing commit;
  also the current `main` HEAD as of this PR). `flatten_dict_obs` appends a
  length-N identity one-hot indexing the receiving agent.

The treatment arm is **explicitly pinned to `a38667b5`** so concurrent MAPPO
work on `main` (#208) cannot pollute the measurement.

## Protocol

12 cells = 2 arms × 2 scenarios × 3 seeds.

| Field | Value |
|---|---|
| scenarios | `default`, `minimal_specialization` |
| lambdas | 0.0 (single arm; orthogonal to obs-fix question) |
| seeds | 42, 43, 44 |
| num_iterations | 100 |
| rollout_steps | 2048 |
| num_agents | 4 |
| value_coef / entropy_coef / normalize_returns | 0.5 / 0.01 / on |
| device | cpu (env is CPU-bound) |

Wall-clock estimate (per curator analysis on Mac Studio): ≈ 6 min/cell × 12 cells
≈ 75 min per arm sequentially, ≈ 80 min total with the two tmux sessions
running in parallel.

## Metrics

Five per-cell metrics (per `experiments/p3_specialization/analyze_220.py` and
`experiments/p3_specialization/diagnostics/pairwise_action_kl.py`, both added
in this PR):

1. **Trailing-5 team reward** — `mean_step_reward_team` averaged over the last
   5 iterations.
2. **Per-agent action entropy** at iter-final (`action_entropy/agent_i` plus
   the mean).
3. **Per-agent CV + action-reward R²** — H1 inspector (existing
   `inspect_rollout_rewards.py`).
4. **Pairwise action-distribution KL** — new helper. Baseline arm should
   show KL ≈ 0 (identical-input pathology); treatment arm should show KL > 0
   iff agents specialize.
5. **Gap closed on `minimal_specialization`** —
   `(ppo_trailing5 − random) / (specialist − random)` against the 2026-05-15
   references (random = −96.07, specialist = −22.07). Pre-#216 reference was
   18.1%. Pass bar = **treatment ≥ 50%**.

## Status

**Results pending.** Two tmux sessions launched on `COMPUTE_HOST_PRIMARY`:

- `issue220-baseline` — runs `run_sweep.py` from a separate worktree pinned
  to `19afcd76`.
- `issue220-treatment` — runs `run_sweep.py` from a worktree pinned to
  `a38667b5`.

This entry will be updated with the metrics table and verdict once both
sessions complete. The PR was opened pre-results so that the analysis-code
review can proceed in parallel with the compute.

## Pre-registered verdicts

| treatment gap-closed | interpretation | action |
|---|---|---|
| ≥ 50% | Obs-differentiation was the bottleneck; PPO is viable. | Update "PPO failure mode" memory note; downgrade #208 priority. |
| 25–50% | Partial fix; some PPO failure remains. | Keep #208 active; document joint-cause. |
| < 25% | Obs-differentiation wasn't the bottleneck. | Tighten memory note to "MAPPO is necessary"; raise #208. |

Per-agent action-entropy divergence and pairwise KL > 0 are *necessary*
conditions for the fix to have done anything structurally — even if team
reward doesn't move, divergence > 0 confirms the bug is gone and points the
next intervention at credit assignment rather than identical-gradient collapse.
