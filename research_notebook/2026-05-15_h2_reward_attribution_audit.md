# H2 Reward-Attribution Audit

**Date**: 2026-05-15
**Issue**: [#191](https://github.com/rjwalters/bucket-brigade/issues/191)
**Status**: H2 **confirmed** — the per-step team reward dwarfs both ownership and work-cost components by 1–2 orders of magnitude, and per-agent rewards are nearly identical at every timestep.

## Setup

- `BucketBrigadeEnv(scenario=default_scenario(num_agents=4))`
- 20 episodes, uniform-random actions (random house index × random {REST, WORK}), max 200 nights per episode
- 265 total env steps analyzed
- Each per-step reward decomposed offline into three components:
  - `team_component[t]` — scalar, identical for all agents (computed from current `SAFE`/`RUINED` counts every step)
  - `ownership_component[i][t]` — per-agent (save bonus + currently-ruined penalty over owned houses)
  - `work_cost[i][t]` — per-agent (`-cost_to_work_one_night` if WORK, `+0.5` if REST)
- Sanity check: `team + ownership + work_cost == env.rewards` verified to within `1e-5` at every step.

## Magnitude results

| Component | Mean |x| | Notes |
|---|---|---|
| Team component | **73.06** | Same value broadcast to all 4 agents |
| Ownership component | **0.63** | Per agent, averaged over the 4 agents |
| Work cost | **0.50** | Per agent (always exactly 0.5 in either direction) |

### Per-step magnitude ratios

| Ratio | median | p75 | p95 | n |
|---|---|---|---|---|
| `|team| / |ownership|` | **60×** | 160× | 160× | 171 |
| `|team| / |work_cost|` | **160×** | 200× | 200× | 265 |
| Fraction of steps with ownership ≡ 0 | 35.5% | — | — | — |

The curator's static-analysis estimate (15–60× team:ownership) lines up with the empirical **median of 60×**, but actually under-states tail behaviour: at p75/p95 the ratio jumps to **160×**. Two reasons:
1. Late-episode steps with many `SAFE` and few `RUINED` houses can push `|team|` to ~80, while ownership stays bounded by `2 × houses_owned ≤ 6`.
2. About a third of steps have ownership exactly zero (agent's owned houses are all SAFE, and no save event fired this step), so the team component is the *only* non-work signal.

## Pairwise agent reward correlation (4×4)

```
        agent0   agent1   agent2   agent3
agent0 +1.0000  +0.9977  +0.9984  +0.9982
agent1 +0.9977  +1.0000  +0.9976  +0.9980
agent2 +0.9984  +0.9976  +1.0000  +0.9983
agent3 +0.9982  +0.9980  +0.9983  +1.0000
```

- **Mean off-diagonal correlation**: **0.998**
- **Minimum off-diagonal correlation**: **0.998**
- Curator threshold for confirming H2 (correlation > 0.95): **massively exceeded**.

### Variance decomposition (analytic lower bound)

- `var(team_t)` = 580.47
- mean `var(reward_a)` = 609.41
- **Team-share lower bound on cross-agent correlation** = `var(team) / mean var(reward)` = **0.953**

In other words, even before the (very small) extra positive co-movement from joint episode dynamics, mathematics alone guarantees pairwise correlation ≥ 0.95 just from the shared team term. The empirical 0.998 sits right at that ceiling because the per-agent residual (ownership + work_cost) is so small relative to team.

## Verdict

**H2 is confirmed.** The shared team reward component is so dominant that every agent receives an almost-identical scalar at every timestep, leaving very little gradient signal to drive per-agent specialization. Both diagnostic criteria fire:

- Median per-step `|team| / |ownership|` ratio is **60×**, vastly exceeding the 10× threshold.
- Minimum pairwise reward correlation is **0.998**, well above the 0.95 threshold.

## Implications (informational — not actioned in this PR)

This audit is diagnostic-only. The 100× team-vs-ownership magnitude split is **deliberate** per the #170/#179 "preserve magnitudes" decision and is not changed here. If P3 specialization is determined to be gradient-starved, candidate follow-ups would include:

- Rebalance `team_reward` vs `reward_own_house_survives` / `penalty_own_house_burns`.
- Switch to CTDE-with-centralized-critic so the value function absorbs the shared team signal and the policy gradient sees only the per-agent advantage.
- Subtract a per-step team mean (variance-reduction baseline) before computing advantages, so the team term doesn't dominate the policy update.

A separate H4 follow-up issue is appropriate for any of these.

## Artefacts

- Diagnostic script: `experiments/p3_specialization/diagnostics/audit_reward_attribution.py`
- Raw JSON output: `experiments/p3_specialization/diagnostics/results/h2_reward_attribution.json`
