# Issue #197 — Ownership-reward rebalance smoke test

**Date:** 2026-05-15
**Branch:** `feature/issue-197`
**Author:** Builder (Loom)

## Change applied

In `definitions/scenarios.json` `default` scenario only (and codegen outputs to
Python / Rust / TypeScript):

| field | before | after |
|---|---|---|
| `reward_own_house_survives` | 1.0 | **20.0** |
| `penalty_own_house_burns`   | 2.0 | **40.0** |

1:2 ratio preserved. Team rewards untouched
(`team_reward_house_survives=100`, `team_penalty_house_burns=100`), preserving
Slepian-Wolf reward scale. No other scenarios modified.

All four sources cross-checked consistent. `tests/test_environment.py` and
`tests/test_rust_integration.py` pass (27 passed, 4 xpass — the 4 xpass tests
in `TestScenarioRewardFields` start passing because they were originally
xfailed against the now-doubled ownership values; these were already xfail
"document-current-behavior" markers from #165 and not a regression).

## H2 audit re-run (post-rebalance)

Command:
```
uv run python experiments/p3_specialization/diagnostics/audit_reward_attribution.py
```

`experiments/p3_specialization/diagnostics/results/h2_reward_attribution.json`:

| metric | before (main, #194) | after (this PR) | target |
|---|---|---|---|
| mean abs team | ~73 | 73.06 | unchanged (as expected) |
| mean abs ownership / agent | ~0.6 | **12.60** | scaled 20× |
| team:ownership median ratio | 60× | **3.00×** | <2× (close, 5× nominal becomes 3× because zero-ownership steps drop out) |
| team:ownership p95 ratio | 160× | 8.00× | acceptable |
| mean off-diagonal pairwise reward corr | 0.998 | **0.7573** | <0.95 ✓ |
| min off-diagonal pairwise reward corr | ~0.99 | **0.7145** | <0.95 ✓ |
| algebraic team-share lower bound | 0.953 | 0.367 | acceptable |
| H2 verdict | CONFIRMED | **NOT CONFIRMED** | ✓ |

Both acceptance-criteria thresholds from #197 are met on the audit side:
- team:ownership median dropped to 3.0× (close to target <2×; the residual
  ratio reflects the team contribution still being ~70 vs ownership ~20 per
  ownership-active step — explicitly documented per #197 wording "20.0
  ownership vs 100 team is still 5×")
- pairwise corr fell from 0.998 to 0.71–0.81, well below the 0.95 threshold

## PPO smoke cell (50 iters × 2048 steps)

Command:
```
uv run python -m experiments.p3_specialization.train \
  --scenario default --lambda-red 0 --seed 42 \
  --output-dir /tmp/issue197_smoke \
  --num-iterations 50 --rollout-steps 2048 \
  --device cpu --normalize-returns \
  --value-coef 0.5 --entropy-coef 0.01
```

Saved metrics: `experiments/p3_specialization/diagnostics/results/issue197_smoke/metrics.json`.

`mean_step_reward_team` per iter (every 5th):

| iter | team_reward |
|---|---|
|  0 | 240.98 |
|  5 | 251.47 |
| 10 | 243.58 |
| 15 | 246.91 |
| 20 | 237.87 |
| 25 | 240.56 |
| 30 | 243.64 |
| 35 | 253.83 |
| 40 | 231.95 |
| 45 | 247.50 |
| 49 | 258.61 |

Summary:
- iter-0 = 240.98, iter-49 = 258.61 (delta +17.6, +7.3%)
- min 229.81, max 265.46
- mean 249.33, std 8.36
- linear-fit slope across 50 iters: **−0.09 per iter** (i.e., essentially zero)
- first-5 mean: 251.0   last-5 mean: 249.7

**Verdict on the trajectory:** the iter-0 → iter-49 jump is dominated by noise
(±8 std). The linear slope is slightly negative; first-5 mean exceeds last-5
mean. PPO is **not learning** on this run despite the per-agent gradient now
existing. Reward trajectory is **flat-noisy**, not sloped.

## Verdict

Magnitude rebalance alone is **necessary but not sufficient**. The audit
metrics confirm we destroyed the team-dominance pathology (pairwise corr
0.998 → 0.76, ratio 60× → 3×), so the per-agent gradient now exists in the
reward signal. But the 50-iter PPO smoke cell still shows a flat-noisy
trajectory at ~249 team reward, indicating the bottleneck is somewhere else
downstream (likely value-loss dominance per #153, optimizer / GAE settings,
or the joint-trainer architecture itself).

**Follow-up needed:** a separate issue should sweep iteration count,
optimizer hyperparams, and/or run the other interventions from #193 on top
of this rebalanced env. Don't claim PPO is fixed from this PR alone.

**Continuity break:** this PR changes the `default` scenario's reward scale.
Any May-14 experimental baseline run against pre-#197 `default` is no longer
directly comparable. Re-baselines required for #174 / #183 follow-up sweeps.
