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

**Sweep complete.** 36/36 cells finished on `COMPUTE_HOST_PRIMARY`
overnight; results harvested locally and aggregated in this follow-up.

Note: the analyzer committed in PR #263 used `f"{alpha:g}"` to rebuild
the run-dir path, which silently drops the trailing `.0` and skipped 8
of 12 cells (baseline and `alpha=2.0` rows came back as "missing").
Fixed to `f"{alpha:.1f}"` in this PR so all 12 cells aggregate.

## Verdict

- **Best cell**: `(alpha, beta) = (0.1, 0.0)`
- **Best gap_closed_mean**: **`+0.164`** (mean over 3 seeds)
- **Tier**: **`tier_3_insufficient`** — best cell is well below the
  pre-registered `0.25` tier-2 boundary.
- **Over-shaping flagged**: No. Max entropy collapse multiple was
  `3.4×` at `(alpha=2.0, beta=0.0)`; well under the conservative `100×`
  threshold. Action shaping at the swept magnitudes does **not**
  collapse the policy.

### Per-cell numbers

All 12 `(alpha, beta)` cells × 3 seeds (n_seeds=3 everywhere):

| alpha | beta | team_reward (mean ± std) | gap_closed_mean | entropy | collapse_x |
|---|---|---|---|---|---|
| 0.0 | 0.0 | -80.31 ± 5.95 | +0.113 | 0.915 | 1.0 |
| 0.0 | 0.1 | -80.28 ± 5.73 | +0.113 | 0.620 | 1.5 |
| 0.0 | 0.5 | -77.98 ± 5.04 | +0.148 | 0.497 | 1.8 |
| 0.1 | 0.0 | -76.98 ± 5.07 | **+0.164** | 0.752 | 1.2 |
| 0.1 | 0.1 | -81.83 ± 9.43 | +0.090 | 0.563 | 1.6 |
| 0.1 | 0.5 | -79.97 ± 3.54 | +0.118 | 0.492 | 1.9 |
| 0.5 | 0.0 | -80.31 ± 6.61 | +0.113 | 0.550 | 1.7 |
| 0.5 | 0.1 | -80.45 ± 9.38 | +0.111 | 0.496 | 1.8 |
| 0.5 | 0.5 | -77.69 ± 4.60 | +0.153 | 0.552 | 1.7 |
| 2.0 | 0.0 | -79.37 ± 7.14 | +0.127 | 0.270 | 3.4 |
| 2.0 | 0.1 | -80.82 ± 7.82 | +0.105 | 0.792 | 1.2 |
| 2.0 | 0.5 | -78.48 ± 3.61 | +0.141 | 0.499 | 1.8 |

Gap-closed range across cells: `[+0.090, +0.164]`. The grid is essentially
flat — action shaping moves the trailing-5 team reward by ~3 reward
units (out of a 65.65 random→specialist span), regardless of magnitude.

### Interpretation

1. **Baseline anchor**: `(α=0, β=0)` lands at gap_closed `+0.113`, near
   (slightly under) the PR #257 IPPO obs-fix verdict of `~0.182`. Same
   plateau, no surprise.
2. **No interior peak**: Increasing α (credit-shared extinguish bonus)
   or β (preventive-presence bonus) does not produce a monotonic or
   even directional improvement. The best cell (`α=0.1, β=0`) beats
   the baseline by only `+0.051`, well within seed noise (std ≈ 5–9).
3. **No over-shaping**: Even `α=2.0` produces only a 3.4× entropy
   collapse — orders of magnitude below MAPPO's 1874× signature.
   Action shaping is not "too strong"; it is **not strong enough to
   matter**.
4. **Plateau is robust to per-step shaping**: This is the **4th of
   5 interventions** to fail to break the plateau on
   `minimal_specialization` (after #197 magnitudes, #198 per-agent
   ownership-reward vectors, #225 MAPPO, and #236 signal-channel
   work that motivated the post-#236 re-derivation).

### Next steps per the verdict ladder

Tier-3 outcome promotes **intervention #4 (dense progress reward)** or
rules-level interventions (#251/#253) per the #193 decision frame.
Action shaping is **not** the lever that breaks the IPPO plateau.

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
