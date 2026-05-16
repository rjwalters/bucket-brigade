# Issue #239 — Post-#236 re-derivation of PR #216 obs-fix and PR #225 MAPPO

**Date:** 2026-05-16
**Issue:** [#239](https://github.com/rjwalters/bucket-brigade/issues/239)
**PR (this commit):** TBD — see PR body for tmux/nohup session names and result status.

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

_Filled when the sweep completes and `analyze_220.py` + `analyze_231.py`
have been re-run with the post-#236 baselines from PRs #243/#244 (siblings
#237/#238)._

### Obs-fix on `minimal_specialization` (post-#236)
- treatment arm gap_closed: **TBD**
- per-seed: TBD

### MAPPO sweep (post-#236)
| scenario | gap_ippo | gap_mappo | delta | tier |
|---|---|---|---|---|
| default | TBD | TBD | TBD | TBD |
| minimal_specialization | TBD | TBD | TBD | TBD |
| positional_default | TBD | TBD | TBD | TBD |

**Headline verdict post-#236**: TBD

## Side-by-side

| scenario | pre-#236 ippo gap | post-#236 ippo gap | pre-#236 mappo gap | post-#236 mappo gap |
|---|---|---|---|---|
| default | TBD | TBD | TBD | TBD |
| minimal_specialization | 0.059 | TBD | TBD | TBD |
| positional_default | TBD | TBD | TBD | TBD |

## Discussion

(To be added when results are in. Key questions to address:
1. Did the substrate change in #236 raise the IPPO floor, the MAPPO ceiling,
   or neither?
2. If MAPPO now succeeds, does the headline shift from `INSUFFICIENT` to
   `MAPPO_SUCCEEDS` or `MAPPO_HELPS_GLOBALLY_NOT_DECISIVE`?
3. Does the post-#236 KL diagnostic show signal-head divergence between
   agents? (Now that we capture the 40-class joint.))

## Baselines used

Post-#236 random and specialist baselines were re-derived in parallel by
sibling issues:
- #237 (PR #244): random baselines across all 14 scenarios. Status at PR
  creation: TBD.
- #238 (PR #243): specialist baselines on `minimal_specialization`,
  `positional_default`, `default`. Status at PR creation: TBD.

If both sibling PRs have merged when this PR runs the analyzers, the
constants in `analyze_220.py:53-54` and `analyze_231.py:61-65` will reflect
post-#236 values; otherwise the analyzers retain pre-#236 constants and
the verdict is marked PENDING.

## See also

- Project memory: `~/.claude/projects/.../project_ppo_failure_mode.md`
- PR #236 (substrate change)
- PR #233 (pre-#236 obs-fix verdict: 5.9%)
- PR #232 (pre-#236 MAPPO verdict: INSUFFICIENT)
- Sibling issues #237/#238 (baseline re-derivation)
