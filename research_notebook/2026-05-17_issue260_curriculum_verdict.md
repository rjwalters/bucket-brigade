# Issue #260 — episode-length curriculum verdict

**Date:** 2026-05-17
**Issue:** [#260](https://github.com/rjwalters/bucket-brigade/issues/260)
**PR (this commit):** TBD — see PR body for tmux session and result status.

## Hypothesis

Curriculum on `min_nights`: train initially on shorter episodes (5 nights), then
8, then the native 12-night floor of `minimal_specialization`. The hypothesis
(per the issue body, intervention #3 from #193) is that shorter episodes give
more end-of-episode terminations per env-step, yielding denser PPO gradient
signal, and that policy learned on the short variant transfers to the longer
floor.

## Arms

- **baseline** (`curriculum=[]`): native `min_nights=12` for the full 50 iters.
- **curriculum** (`--curriculum '0:5,17:8,34:12'`): 17 iters at floor 5, 17 at
  floor 8, 16 at floor 12. Final-iter evaluation lands on the canonical
  12-night floor (apples-to-apples with baseline).

Both arms run on `minimal_specialization` (per-agent ownership-dominated reward
structure that PPO has historically plateaued on at ~15% of the
specialist-random gap; see project memory "PPO failure mode").

## Protocol

6 cells = 2 arms × 3 seeds, run on `COMPUTE_HOST_PRIMARY` in a `tmux` session
named `issue260-curriculum`.

| Field | Value |
|---|---|
| scenario | `minimal_specialization` |
| arms | `baseline`, `curriculum` (`--curriculum '0:5,17:8,34:12'`) |
| seeds | 0, 1, 2 |
| num_iterations | 50 |
| rollout_steps | 2048 (default) |
| num_agents | 4 |
| lambda_red | 0.0 |
| device | cpu |

Branch pin: `feature/issue-260` at HEAD `0a7b616d` (curriculum implementation).

**Coordination caveat (logging only, not reward):** baseline seeds 1 and 2
were collected when the host workspace had a sibling branch
(`feature/issue-262`, action-shaping calibration) checked out at the time of
launch. Because every scenario including `minimal_specialization` defaults to
`action_shaping_alpha=beta=0.0`, the team-reward signal is identical on both
branches and the verdict numbers are sound. The only consequence is that
seeds 1 and 2 of the baseline arm log `min_nights_floor=-1` (the analyzer's
`.get()` default) instead of `12`; the underlying scenario was unchanged.
Baseline seed 0 and all three curriculum seeds were collected on
`feature/issue-260` and log `min_nights_floor` correctly.

## Pre-registered verdict ladder (per curator)

| delta = curriculum_gap − baseline_gap | tier | follow-up |
|---|---|---|
| ≥ 0.50 | tier 1 — clear win | file production-sweep follow-up |
| 0.25 – 0.50 | tier 2 — combine with intervention #2 (action shaping #261) | follow-up |
| < 0.25 | tier 3 — curriculum unhelpful | promote intervention #4 (dense progress signal) |

## References

Per-step (random, specialist) team-reward references on
`minimal_specialization` (issue199_minspec baselines, 4-agent honest-signaling
specialist, seed=42, n=50):

| scenario | random | specialist | spec − random |
|---|---|---|---|
| minimal_specialization | −92.92 | −22.07 | +70.85 |

Source: `experiments/p3_specialization/diagnostics/results/issue199_minspec/baselines.json`.

## Results

### Team reward (trailing-5 mean of `mean_step_reward_team`)

| arm | n | mean ± std | per-seed |
|---|---|---|---|
| baseline | 3 | −87.62 ± 11.84 | [−81.41, −101.27, −80.17] |
| curriculum | 3 | −87.79 ± 6.94 | [−84.20, −95.80, −83.38] |

### Curriculum floors observed

| arm | seed | floors_unique | first | last |
|---|---|---|---|---|
| baseline | 0 | [12] | 12 | 12 |
| baseline | 1 | [−1]* | −1 | −1 |
| baseline | 2 | [−1]* | −1 | −1 |
| curriculum | 0 | [5, 8, 12] | 5 | 12 |
| curriculum | 1 | [5, 8, 12] | 5 | 12 |
| curriculum | 2 | [5, 8, 12] | 5 | 12 |

*See coordination caveat above; `-1` is the analyzer's `.get()` sentinel and
does not indicate a non-12 floor — the scenario factory's native
`min_nights=12` was unchanged for both branches.

The curriculum cells correctly show the floor progression `5 → 8 → 12`
matching `'0:5,17:8,34:12'`, confirming that mid-training mutation of
`trainer.env.scenario.min_nights` took effect at the right iteration
boundaries.

### Verdict numbers

| metric | value |
|---|---|
| `gap_baseline` | +0.075 |
| `gap_curriculum` | +0.072 |
| `delta = curriculum − baseline` | **−0.002** |
| Tier | **`tier_3_curriculum_unhelpful`** |

### Headline verdict: `TIER 3 — INSUFFICIENT`

Both arms close ~7% of the specialist-random gap. The curriculum produces
no measurable improvement (delta = −0.002, well inside seed-level noise:
baseline std 11.84, curriculum std 6.94). Curriculum actually has slightly
*lower* variance across seeds, but the means are statistically
indistinguishable.

The headline number is consistent with the project-memory "PPO failure mode"
note: independent-PPO continues to plateau at ~7-15% of the optimum on
`minimal_specialization`, and the cheapest entry from #193's intervention
ladder does not move that plateau. The pattern matches the prior tier-3
results from MAPPO (#231), per-agent obs-fix (#220), and positional shaping
(#228 / #228).

## Implications per the pre-reg ladder

- **Curriculum on episode length alone is insufficient.** The PPO plateau is
  not primarily a reward-density / termination-frequency problem. Compressing
  episodes to 5 nights for the first phase does not give PPO traction it
  could carry across the phase boundary.
- **Promote intervention #4 (dense progress signal) per the explicit
  template in the curator enrichment.** Per #260 acceptance criteria, the
  tier-3 outcome triggers a follow-up issue titled "curriculum unhelpful —
  try intervention #4 (dense progress signal)".
- **Intervention #2 (action-conditioned reward shaping, #261/#262)** is
  already in flight and is the parallel cheap entry from #193's list; we
  don't deduplicate that effort here.

## Project-memory update

The "PPO failure mode" note in user memory will be updated alongside this
PR to reflect that **episode-length curriculum does not close the gap** and
that intervention #4 (dense progress signal) is the next entry on the #193
ladder to try (after action-shaping #261/#262 lands).
