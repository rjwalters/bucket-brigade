# 2026-05-17 — Issue #261/#262 action-shaping calibration verdict

## Context

PR #261 (#259) landed action-conditioned per-step reward shaping
(`Scenario.action_shaping_alpha` for credit-shared extinguish bonus,
`Scenario.action_shaping_beta` for preventive-presence bonus).
Infrastructure was bit-identical default-off and well-tested.
**This entry records the result of running the actual calibration sweep**
that answers "does this rescue PPO?"

This is **intervention #2** in the P3 plateau decision frame (#193):
- #1 magnitudes (#197/#198) — landed, did not break plateau
- **#2 action shaping (this experiment)**
- #3 curriculum (#260)
- #4 dense progress reward
- #5 MAPPO (#225) — known tier-3
- #6 rules-level (#251/#253)

## Protocol (pre-registered)

- Scenario: `minimal_specialization` (direct comparison to PR #257's 0.182
  obs-fix verdict).
- Grid: `alpha ∈ {0.1, 0.5, 2.0}` × `beta ∈ {0.0, 0.1, 0.5}` = 9 cells
  plus baseline `(alpha=0, beta=0)` = 10 configurations.
- Seeds: 42, 43, 44 → 30 cells total.
- Trainer: IPPO, 50 iterations × 2048 rollout steps.
- Wall-clock estimate: ~20-30 min on Mac Studio at 4-way parallel.

Gap closed uses the `minimal_specialization` PR #244/#243 references:
`random = -87.72`, `specialist = -22.07`, denominator = 65.65.

## Verdict ladder

| Outcome | Action |
|---|---|
| best gap_closed ≥ 0.50 | Reward shaping breaks the plateau. Update project memory. File production sweep follow-up. |
| best gap_closed 0.25–0.50 | Partial win. File follow-up to combine with intervention #3 (curriculum) or #4 (dense progress). |
| best gap_closed < 0.25 | Reward shaping insufficient. Promote intervention #4 or rules-level. |
| over-shaping detected (entropy collapse > 100×) | Confirms curator warning; reduce alpha; flag the entropy threshold. |

## Status

**Results pending.** Sweep launched in tmux session
`issue261_calibration` on `COMPUTE_HOST_PRIMARY`. The aggregator
(`experiments/p3_specialization/analyze_261_calibration.py`) and
summary outputs (`experiments/p3_specialization/diagnostics/results/
issue261_calibration/summary.{json,md}`) will be committed in a
follow-up once the sweep completes (per PR #248/#227 precedent).

## Verdict (to be filled in)

- _Best cell_: (alpha, beta) = …
- _Best gap_closed_mean_: …
- _Tier_: …
- _Over-shaping flagged_: …

## Implementation artifacts

- `experiments/p3_specialization/train.py` — added
  `--action-shaping-alpha` / `--action-shaping-beta` CLI flags that
  mutate the loaded `Scenario` instance post-load. Default 0.0 keeps
  bit-identical pre-#259 behavior via the env's fast-path skip.
- `experiments/p3_specialization/run_issue262_sweep.sh` — driver
  modeled on `run_issue239_sweep.sh` (xargs `-P` parallelism, per-cell
  stdout, git-rev provenance stamp).
- `experiments/p3_specialization/analyze_261_calibration.py` —
  aggregator modeled on `analyze_231.py`. Picks best `(alpha, beta)`
  by `gap_closed_mean`, flags over-shaping at 100× baseline entropy.
- `experiments/p3_specialization/runs/issue261_calibration/` — output
  root following the `alpha_{A}/beta_{B}/seed_{N}/` layout consumed
  by the aggregator.

## References

- PR #261 (infrastructure)
- PR #257 (obs-fix verdict — anchors the 0.182 baseline gap)
- #259 (curator's calibration spec)
- #193 (decision frame)
