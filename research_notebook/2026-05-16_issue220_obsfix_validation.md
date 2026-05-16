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

**Results pending; treatment arm blocked by latent post-#216 bug.** Two tmux
sessions launched on `COMPUTE_HOST_PRIMARY` (Mac Studio, `robbs-mac-studio`):

- `issue220-baseline` — runs `run_sweep.py` from worktree
  `~/GitHub/bb_issue220_baseline` pinned to `19afcd76`. **Progressing
  normally** as of kickoff (first cell completed within ~3 min, sweep
  ETA ~20–40 min for 6 cells).
- `issue220-treatment` — runs `run_sweep.py` from worktree
  `~/GitHub/bb_issue220_treatment` pinned to `a38667b5`. **CRASHED on
  first cell** with::

      File ".../experiments/p3_specialization/train.py", line 293, in _measure_information
        codes = [quantize_uniform(f, n_bins=n_bins) for f in feats_np]
      File ".../bucket_brigade/analysis/info_theory.py", line 262, in quantize_uniform
        raise ValueError(f"values must be 1-D or 2-D, got shape {values.shape}")
      ValueError: values must be 1-D or 2-D, got shape (2048, 4, 3)

  The new `[T, N, obs_dim]` rollout buffer shape introduced by #216 isn't
  flattened before being fed into `_measure_information`'s
  `quantize_uniform` call (which expects 1-D or 2-D features). Unit tests
  for #216 didn't cover the MI path. **This is a follow-up bug** — a
  separate issue should be filed against `experiments/p3_specialization/
  train.py:293` (or `info_theory.py:262`) to either accept 3-D input
  (preferred — reshape per-agent) or to flatten in `train.py` before
  calling.

This entry will be updated with the metrics table and verdict once the
treatment arm is unblocked and both arms re-run to completion. The PR
was opened pre-results so the analysis-code review can proceed in parallel.

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

## Results (2026-05-16, rerun via PR #230)

The treatment arm was rerun on the post-#228 main HEAD; the baseline arm
was rebased onto commit `2b963630` (the parent of the obs-fix commit
`a38667b5`) because `19afcd76` (the originally-cited baseline pin) predates
`minimal_specialization` (scenario added by PR #207 / commit `6d8bcc66`)
and so could not have produced the missing baseline-arm minspec cells. The
3 pre-existing baseline `default` cells on disk (from the PR #227 kickoff
on `19afcd76`) are scientifically equivalent for the obs-fix question (the
diff between `19afcd76` and `2b963630` is `positional_default` scenario
addition; `default` and the rest of the env code are unchanged) and were
retained via `--skip-existing`.

**Treatment HEAD:** `89ed2ebb` (current main at rerun time; first commit
post-#228 fix). **Baseline HEAD:** `2b963630` (parent of `a38667b5`
obs-fix commit; contains `minimal_specialization`).

All 12 cells completed (2 arms × 2 scenarios × 3 seeds). Pairwise action-KL
was computed for the 6 treatment cells; baseline cells produced 38-dim
policies that can't be loaded by the post-#216 trainer (42-dim input) so
KL on the baseline arm is reported as `n/a` — see the "divergence" table
interpretation below.

### Team reward (trailing-5 mean of `mean_step_reward_team`)

| scenario | arm | n | mean ± std | per-seed |
|---|---|---|---|---|
| default | baseline | 3 | +249.49 ± 1.40 | [251.11, 248.69, 248.66] |
| default | treatment | 3 | +253.70 ± 2.38 | [256.23, 253.38, 251.51] |
| minimal_specialization | baseline | 3 | -86.24 ± 6.21 | [-79.72, -92.07, -86.94] |
| minimal_specialization | treatment | 3 | -91.71 ± 7.38 | [-87.34, -100.23, -87.55] |

### Policy divergence (final-iter entropy spread + pairwise action KL)

| scenario | arm | mean_entropy | entropy_spread (max-min across agents) | KL off-diag mean |
|---|---|---|---|---|
| default | baseline | 2.399 | 1.628 | n/a |
| default | treatment | 2.293 | 1.653 | 3.7178 |
| minimal_specialization | baseline | 2.907 | 1.370 | n/a |
| minimal_specialization | treatment | 2.829 | 1.871 | 2.7252 |

The KL off-diagonal mean is strongly positive (2.7-3.7) on every treatment
cell. The necessary-condition check **passes**: the obs-fix produces
distinguishable per-agent policies, with mean pairwise KL well above zero
(the identical-input pathology). The baseline arm policies have 38-dim
input (no identity one-hot) and cannot be loaded by the post-#216 trainer
for KL computation, but their state was, by construction, the
identical-gradient pathology PR #216 was written to fix.

### Gap-closed on `minimal_specialization`

| arm | gap_closed (trailing-5 mean) | per-seed | pre-reg verdict |
|---|---|---|---|
| baseline | 0.133 | [0.221, 0.054, 0.123] | < 25% — MAPPO needed |
| treatment | 0.059 | [0.118, -0.056, 0.115] | **< 25% — MAPPO needed** |

## Verdict (pre-registered)

**Treatment gap-closed = 5.9 %** (per-seed range: -5.6 % to +11.8 %).
This falls into the **< 25 %** bin of the pre-registered verdict table:

> Obs-differentiation wasn't the bottleneck. Tighten memory note to
> "MAPPO is necessary"; raise #208.

Treatment is actually slightly **worse** than baseline (13.3 % vs 5.9 % on
minimal_specialization), within seed-level noise. The most informative
finding is the necessary-condition split: pairwise KL on the treatment arm
is 2.7-3.7 (strongly differentiated policies — the obs-fix worked
structurally), and team reward on `default` improves modestly
(+249.49 → +253.70). But that structural fix does **not** translate into
the credit-assignment gap closure that `minimal_specialization` was
designed to measure. Per the existing project-memory analysis: this points
the next intervention squarely at **independent-PPO's credit-assignment
failure**, not at observation structure.

### Action items applied

- Project memory `project_ppo_failure_mode.md`: tightened from "obs-fix
  verified working" to "obs-fix verified structurally but insufficient —
  gap-closure still 5.9 %; MAPPO/CTDE remains necessary".
- Issue #208 (MAPPO): should remain active / be raised in priority (this
  PR does not touch #208 labels — leaves that to Guide/Curator).

### Artifacts

- `experiments/p3_specialization/runs/issue220_baseline/**/metrics.json` (6 cells)
- `experiments/p3_specialization/runs/issue220_treatment/**/metrics.json` (6 cells)
- `experiments/p3_specialization/runs/issue220_treatment/**/pairwise_action_kl.json` (6 cells)
- `experiments/p3_specialization/diagnostics/results/issue220_obsfix/summary.{md,json}`
- Remote worktrees on `robbs-mac-studio`: `~/GitHub/bb_issue220_baseline`
  (HEAD `2b963630`), `~/GitHub/bb_issue220_treatment` (HEAD `89ed2ebb`).
