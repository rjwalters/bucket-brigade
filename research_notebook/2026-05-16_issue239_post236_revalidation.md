# Issue #239 — Post-#236 re-derivation of PR #216 obs-fix and PR #225 MAPPO

**Date:** 2026-05-16 (results finalized 2026-05-17 under issue [#255](https://github.com/rjwalters/bucket-brigade/issues/255))
**Issue:** [#239](https://github.com/rjwalters/bucket-brigade/issues/239)
**Finalization PR:** see #255 — verdict tier 3 (`INSUFFICIENT`) on both arms.

## Why re-derive?

PR #236 (issue #235) turned the broadcast signal into a first-class action
dimension (`MultiDiscrete([10, 2, 2])`). Before #236, the "signal" was a
deterministic function of work/rest — i.e. redundant noise that PPO could
ignore. After #236, agents can learn nontrivial signal policies.

The two algorithmic interventions tested in PR #233 (obs-fix) and PR #232
(MAPPO) were trained against the pre-#236 substrate. Their tier-3 verdicts
(5.9% obs-fix; all-scenarios-insufficient MAPPO) may flip when PPO can
exploit a real signal channel — especially in `minimal_specialization`,
where the deception-research angle suggests team members might rationally
weight signals differently when other members lie.

This notebook re-runs the same protocol against post-#236 main and
compares the verdicts side-by-side.

## Pre-flight code fix

`experiments/p3_specialization/diagnostics/pairwise_action_kl.py`
previously hardcoded a 20-class joint over `[house(10) × mode(2)]`. After
#236 the joint is 40-class (`10 × 2 × 2`). Without the fix, the KL column
in `analyze_220.py` / `analyze_231.py` silently understates per-agent
divergence by ignoring the signal head.

Fix: `softmax_packed` refactored to outer-product across all heads in
`logits_list`. Covered by 5 unit tests in `tests/test_pairwise_action_kl.py`,
including a regression case where two agents differ only on the signal head
(the old impl returned KL ≈ 0; the new impl returns substantial KL).

## Protocol

Pinned to commit `1061b3dd` (= `dffe1060` post-#236 main + KL fix + sweep
driver). Trained on `COMPUTE_HOST_PRIMARY` (Mac Studio M2 Ultra; 24 cores).

| Component | Cells | Pool | Wall-clock (est.) |
|---|---|---|---|
| Obs-fix treatment | 6 (2 scenarios × 3 seeds) | 2-way parallel | ~12 min |
| MAPPO sweep | 18 (3 scenarios × 2 algos × 3 seeds) | 2-way parallel | ~30 min |
| Pre-flight smoke (4 cells × 5 iter) | — | serial | ~5 min |

**Hyperparameters identical to PR #233 / PR #232 protocols** (100 iters,
2048 rollout steps, lr 3e-4, value_coef 0.5, entropy_coef 0.01, normalize
returns on, 4 agents, lambda_red 0.0).

### Important scope note on obs-fix arm

Strictly, the obs-fix question requires a paired comparison between:
- baseline: pre-#216 obs code (no per-agent differentiation), and
- treatment: post-#216 obs code (with per-agent identity tail).

Running pre-#216 code reverts to pre-#236 env (no signal action dim), which
conflates two independent interventions. The pragmatic choice for #239 is to
train only the treatment arm on post-#236 and compare gap_closed against
the **pre-#236 treatment baseline** (preserved as
`diagnostics/results/issue220_obsfix/summary.pre236.{md,json}`). This
isolates the substrate change (#236) from the obs-fix delta.

The MAPPO sweep below is the scientifically rigorous half — both arms
trained on post-#236 main, paired by scenario and seed.

## Pre-#236 baseline (from PRs #233 / #232)

### Obs-fix on `minimal_specialization` (PR #233)
- baseline arm gap_closed: 0.133 (per-seed: 0.221, 0.054, 0.123)
- treatment arm gap_closed: **0.059** (per-seed: 0.118, -0.056, 0.115)
- **Verdict tier 3**: treatment underperforms baseline.

### MAPPO sweep (PR #232)
| scenario | gap_ippo | gap_mappo | delta | tier |
|---|---|---|---|---|
| default | (filled at analysis) | (filled at analysis) | -0.083 | tier_3 |
| minimal_specialization | (filled at analysis) | (filled at analysis) | -0.038 | tier_3 |
| positional_default | (filled at analysis) | (filled at analysis) | -0.026 | tier_3 |

**Headline verdict pre-#236**: `INSUFFICIENT` — MAPPO narrows entropy on
`default`/`positional_default` without lifting reward.

## Post-#236 results

**Status at PR creation: training IN PROGRESS.** The 24-cell sweep was
launched on `COMPUTE_HOST_PRIMARY` via `nohup bash
experiments/p3_specialization/run_issue239_sweep.sh all > /tmp/issue239_sweep.log 2>&1 &`
at 2026-05-16T22:28Z. The repo is checked out into worktree
`/Users/rwalters/GitHub/bb_issue239` (detached HEAD at commit `1061b3dd`).

Each cell takes ~10-15 wall-min at 100 iters x 2048 rollout steps, much
longer than the curator-estimated 3 min/cell. Estimated completion:
~5-7 hours from launch, depending on parallel-pool sibling contention
(#240 is running 13 parallel Nash computations on the same host).

When the sweep completes, run on the same host:

```bash
cd /Users/rwalters/GitHub/bb_issue239
# Optionally rebase to pick up sibling baselines:
git fetch origin && git rebase origin/main   # if PRs #243/#244 merged
bash experiments/p3_specialization/finalize_issue239.sh
```

That script:
1. Computes `pairwise_action_kl.json` for every cell (with the new
   40-class joint).
2. Re-runs `analyze_220.py` and `analyze_231.py`, which write
   `summary.{md,json}` under
   `diagnostics/results/issue220_obsfix/` and `.../issue231_mappo/`
   (clobbering pre-#236 summaries; backups preserved as
   `summary.pre236.{md,json}`).
3. Prints the post-#236 summaries to stdout.

Edit this notebook to fill in the placeholder TBDs below with the actual
numbers from the rendered summaries.

_Filled 2026-05-17 from `finalize_issue239.sh` rendered summaries on
`COMPUTE_HOST_PRIMARY`, with the `bb_issue239` worktree rebased onto
post-#243/#244/#245/#246/#248/#250 main (`06cb0fac`)._

### Obs-fix on `minimal_specialization` (post-#236)
- treatment arm gap_closed: **0.182** (trailing-5 team-reward mean)
- per-seed: 0.264, 0.078, 0.203
- Treatment team reward: −82.63 ± 7.04 per-step (per-seed: −76.51, −90.33, −81.04)
- Pre-registered verdict: **< 25% — MAPPO needed** (tier 3)
- Pre-#236 treatment baseline (preserved in the pre-overwrite `git diff` of
  this PR; the `summary.pre236.{md,json}` files referenced in the finalize
  script never made it into git history): 0.059

### MAPPO sweep (post-#236)
| scenario | gap_ippo | gap_mappo | delta (mappo−ippo) | tier |
|---|---|---|---|---|
| default | +0.003 | +0.022 | +0.019 | `tier_3_insufficient` |
| minimal_specialization | +0.078 | +0.011 | −0.066 | `tier_3_insufficient` |
| positional_default | +0.012 | −0.019 | −0.031 | `tier_3_insufficient` |

**Headline verdict post-#236**: **`INSUFFICIENT`** (Tier 3 — 0 succeed, 0
partial, 3 insufficient). The signaling-as-action substrate did not flip
the MAPPO verdict on any scenario; on `minimal_specialization` and
`positional_default` MAPPO remains *worse* than IPPO.

## Side-by-side

| scenario | pre-#236 ippo gap | post-#236 ippo gap | pre-#236 mappo gap | post-#236 mappo gap |
|---|---|---|---|---|
| default | n/a (delta −0.083) | +0.003 | n/a (delta −0.083) | +0.022 |
| minimal_specialization | 0.059 (obs-fix treatment) / n/a (mappo arm delta −0.038) | +0.078 (ippo) / 0.182 (obs-fix treatment) | n/a (delta −0.038) | +0.011 |
| positional_default | n/a (delta −0.026) | +0.012 | n/a (delta −0.026) | −0.019 |

Pre-#236 absolute `gap_ippo` / `gap_mappo` were not preserved in the
analyzers' source-of-truth — only the `delta` (mappo − ippo) was reported
in PR #232. The deltas above are reproduced from project memory for
comparison.

## Discussion

The headline did **not** flip. Three observations:

1. **Substrate change (#236) did not raise either the IPPO floor or the
   MAPPO ceiling on the gap-closed metric.** Post-#236 IPPO gap_closed on
   `default`/`positional_default` is essentially zero (0.003/0.012); pre-#236
   was likewise near-zero relative to the same references. The
   `minimal_specialization` IPPO bump from 0.059 (obs-fix treatment) to
   0.078 (this run's IPPO) is within the per-seed standard deviation
   (std ~7 per-step on a 65-per-step gap) and should not be over-interpreted.
2. **MAPPO is still net-negative on `minimal_specialization` and
   `positional_default`** (delta = −0.066 and −0.031). The shared critic
   continues to pull policies toward a worse mode, even with a real signal
   channel available. The diagnostic KL pattern is informative: MAPPO
   collapses pairwise action KL by ~3× (e.g. `default`: IPPO=21.6 →
   MAPPO=7.4; `minimal_specialization`: 21.1 → 5.6) while *raising*
   per-agent entropy (`default`: 0.001 → 1.874; `minimal_specialization`:
   0.067 → 2.319). MAPPO is averaging policies toward a softer-but-more-
   homogeneous distribution — exactly the failure mode flagged in the
   pre-#236 project-memory entry.
3. **Signal-head divergence under post-#236 IPPO is high but
   uninterpretable as specialization.** IPPO KL ≈ 21-22 across scenarios
   means agents have collapsed to *different* near-deterministic policies
   (entropy ~0.001-0.07), not that they are coordinating via the signal
   channel. The 40-class joint KL fix from this PR's pre-flight is doing
   its job — pre-#216-style identical-input collapse would have shown
   KL ≈ 0 — but the divergence is being driven by random-init lottery
   among low-quality local optima, not by signal honesty.

**Recommendation:** the project-memory ladder stands. Promote #193 (reward
shaping) as the next intervention; the bug is upstream of obs
differentiation, value-baseline centralization, *and* signal-channel
dimensionality.

### MINSPEC_RANDOM discrepancy (flagged by curator)

`analyze_220.py:59` uses `MINSPEC_RANDOM = -96.07` (from PR #243 /
issue #238). `analyze_231.py:77` uses `BASELINES['minimal_specialization'] =
(-87.72, -22.07)` (random from PR #244 / issue #237). Both are "live"
post-#236 numbers, but they come from different upstream samplers:

- `-96.07`: PR #243's specialist re-derivation pipeline, which samples
  random uniformly over the **full post-#236 action space**
  (`MultiDiscrete([10, 2, 2])`).
- `-87.72`: PR #244's random-baseline pipeline, which (prior to the #246 /
  PR #250 fix) sampled with a stale 2-dim joint. After PR #250
  (`06cb0fac`), `issue199_baselines.py` uses the 3-dim sampler, so
  *re-running* `#244` would converge with `#243`. The `-87.72` constant
  in `analyze_231.py` predates PR #250 and is the **only place** in this
  finalize that uses the stale sample.

This means `gap_closed` for `minimal_specialization` is sensitive to the
choice of random reference. Using `analyze_220.py`'s `-96.07` for the
IPPO arm: gap = (−82.63 − (−96.07)) / (−22.07 − (−96.07)) = 13.44 / 74.00
= 0.182 (matches the obs-fix render). Using `analyze_231.py`'s `-87.72`:
gap = (−82.63 − (−87.72)) / (−22.07 − (−87.72)) = 5.09 / 65.65 = 0.078
(matches the IPPO render). The verdict is **tier 3 in either case**, so
the discrepancy does not change the headline, but follow-up to unify
the constants once PR #250's sampler propagates is tracked under #246.

## Baselines used

Post-#236 random and specialist baselines were re-derived in parallel by
sibling issues:
- #237 (PR #244): random baselines across all 14 scenarios. **MERGED.**
- #238 (PR #243): specialist baselines on `minimal_specialization`,
  `positional_default`, `default`. **MERGED.**
- #246 (PR #250): 3-dim MultiDiscrete sampler in `issue199_baselines.py`.
  **MERGED.** Resolves the `-96.07`/`-87.72` discrepancy at source for
  future re-runs; this run's analyzers still carry the stale constant in
  `analyze_231.py:77` (see Discussion).

All sibling PRs merged before this finalize ran. Constants in
`analyze_220.py:59-60` and `analyze_231.py:77-81` reflect post-#236 values.

## See also

- Project memory: `~/.claude/projects/.../project_ppo_failure_mode.md`
- PR #236 (substrate change)
- PR #233 (pre-#236 obs-fix verdict: 5.9%)
- PR #232 (pre-#236 MAPPO verdict: INSUFFICIENT)
- Sibling issues #237/#238 (baseline re-derivation)
