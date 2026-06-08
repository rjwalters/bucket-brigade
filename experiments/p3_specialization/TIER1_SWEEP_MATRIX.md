# Tier-1 sweep matrix — design decisions for the #343 trainer screen

This document is the concrete decomposition of the Tier-1 row of the
[`#343`](https://github.com/rjwalters/bucket-brigade/issues/343) sweep
matrix into the actual cells the operator will launch via
[`experiments/scripts/launch_tier1_sweep.sh`](../scripts/launch_tier1_sweep.sh).

It complements:

- The trainer / env-mode / scenario tables in the #343 body (the
  exhaustive enumeration of the design space).
- [`run_tier1_cell.py`](run_tier1_cell.py) (the uniform per-cell driver
  shipped in #346) and its 14-row [`TRAINERS`](run_tier1_cell.py) table.
- [`aggregate_tier1.py`](aggregate_tier1.py) (the post-sweep verdict
  aggregator shipped in the same PR).
- [`TIER1_LAUNCH_RUNBOOK.md`](TIER1_LAUNCH_RUNBOOK.md) (the operator's
  copy-paste companion to the launcher).

This doc answers "which combos should we actually run, and which should we
defer or exclude, and why?" — the curator's question that the parent issue
explicitly leaves to a sub-decision. It does **not** define the per-cell
schema; that lives in `run_tier1_cell.py`'s `build_cell_summary` and the
parent issue's curator note.

## TL;DR — the Tier-1 launch set

| # | Trainer key (driver) | Scenario              | Seeds       | Iters | Status / notes                          |
|---|----------------------|-----------------------|-------------|-------|-----------------------------------------|
| 1 | `mappo`              | minimal_specialization| 42 43 44    | 50    | already verified ≤ IPPO (#208/#232); rerun for matrix completeness |
| 2 | `high_lambda`        | minimal_specialization| 42 43 44    | 50    | no sweep yet (#282 / #295)              |
| 3 | `bc_init_continuation`| minimal_specialization| 42 43 44   | 50    | tests if BC fit (gap_closed=0.934) survives PPO continuation (#270/#278) |
| 4 | `bc_init_high_lambda`| minimal_specialization| 42 43 44    | 50    | composition test (#285/#304); depends on #3 |
| 5 | `lola`               | minimal_specialization| 42 43 44    | 50    | (#287/#303)                             |
| 6 | `hca`                | minimal_specialization| 42 43 44    | 50    | (#289/#301)                             |
| 7 | `influence`          | minimal_specialization| 42 43 44    | 50    | (#290/#302), α=0.5 from driver default  |
| 8 | `nhr`                | minimal_specialization| 42 43 44    | 50    | (#283/#300), λ=0.5 from driver default  |
| 9 | `progress`           | minimal_specialization| 42 43 44    | 50    | (#265/#299), coef=1.0 from driver default |
| 10| `macro_actions`      | minimal_specialization| 42 43 44    | 50    | (#286/#298)                             |
| 11| `reinforce`          | minimal_specialization| 42 43 44    | 50    | positive-control / off-PPO diagnostic (#273/#320) |
| 12| `pbt`                | minimal_specialization| 42 43 44    | 50    | orchestrator-shaped (#288/#306); slowest cell |

**Excluded from Tier-1**:

- `ippo` — already the baseline. Re-running adds no new information past
  `0.182` (#257). Included in `aggregate_tier1.py` output only if a prior
  cell exists in `tier1_runs/`.
- `coma` — author-deprioritized per #271 thesis pivot. The driver still
  dispatches it for completeness, but it is not in the Tier-1 launch set
  unless the operator explicitly asks for it via the launcher.

## Decision rationale (per axis)

### Trainers — why 12, not 14?

The parent issue enumerates 14 rows. We collapse to 12 in the launch set
because:

1. **`ippo` is the baseline.** Its `gap_closed ≈ 0.182` is the number every
   other cell is compared against; running it again is wasted compute.
   The aggregator (`aggregate_tier1.py`) already treats any IPPO cell in
   `tier1_runs/` as the comparison row.
2. **`coma` is deprioritized.** The author note in #271 explicitly pivots
   away from COMA after the thesis update. Including it in the auto-launch
   would consume a remote cell (~30 min) on a known-low-value arm.
   Operators can re-enable it via `--trainers coma` if they want to close
   the loop empirically.

Everything else gets one cell. The 12 cells are independent; the launcher
dispatches them sequentially within one tmux session by default (operator
can shard with `--trainers` for parallel hosts).

### Scenario — why only `minimal_specialization`?

The parent issue lists 6 registered scenarios. Tier-1 runs only on
`minimal_specialization` because:

1. **It is the canonical P3 substrate** — the entire learnability gap is
   defined on it (`MINSPEC_RANDOM = 1.067`, `MINSPEC_SPECIALIST = 2.0`,
   PPO ceiling ≈ `1.237`; see `bucket_brigade/baselines.py`).
2. **Cross-scenario rotation is a Tier-3 question.** The parent issue's
   "Open questions" section explicitly asks whether to rotate across
   `{default, minimal_specialization, chain_reaction}`. That is a
   robustness check after a candidate is found, not a screen.
3. **Compute budget.** Adding 2 more scenarios trebles Tier-1 to
   ~54 hours. That is acceptable, but only if Tier-1 actually produces a
   candidate worth robustifying. The launcher supports `--scenario` so
   the rotation can be added post-hoc without a code change.

If Tier-1 produces a trainer with `gap_closed ≥ 0.49`, the natural
follow-up is to re-launch that trainer with
`--scenarios default,minimal_specialization,chain_reaction` and require
the verdict to hold across all three.

### Seeds — why `42 43 44`?

The parent issue's "How to file a Tier-1 sub-issue" template fixes
`--seeds 42 43 44`. Three seeds is the minimum to compute a meaningful
mean ± std with the standard deviation calculation in
`build_cell_summary` (`statistics.pstdev` requires ≥ 2 samples). The
parent issue committed to 3 seeds in its compute-budget estimate
(~30 min × 3 seeds × 50 iters per cell).

Five seeds would double the wall-clock to ~60 hours total and is reserved
for Tier-3 cross-product cells where we want tighter error bars on a
candidate that already cleared the Tier-1 0.49 bar.

### `--num-iterations 50` — why?

This is the parent issue's commitment in both the compute-budget table
and the sub-issue template. 50 iterations is enough to clear the
"first-iteration random play" floor and observe the trailing-5 plateau
that `gap_closed_mean` is computed on (see
`run_tier1_cell.TRAILING_N = 5`).

Shorter (e.g. 25) would underreport plateau cells whose trajectories are
still climbing; longer (e.g. 100) doubles the wall-clock without changing
verdicts in the historical diagnostic literature (`analyze_270.py`,
`analyze_282.py`, etc., all use 50–100 iters and converge on the same
verdict tier).

### `--rollout-steps 2048` — why the driver default?

Driver default from `run_tier1_cell.py` (`p.add_argument("--rollout-steps",
type=int, default=2048)`). Matches the canonical training budget in
`experiments/p3_specialization/train.py` and the historical sweeps under
`experiments/p3_specialization/diagnostics/`. Smoke tests use
`--rollout-steps 64` to keep wall-clock under 1 s; that is **only** for
verifying argv plumbing, never for verdict-eligible cells.

## Sub-issue plan — what gets filed downstream

Per the parent issue's "How to file a Tier-1 sub-issue" section, each of
the 12 launch-set rows should become a thin sub-issue that:

1. Cites this matrix doc.
2. Names a single trainer (the value of `--trainer`).
3. References the launch command from `TIER1_LAUNCH_RUNBOOK.md`.
4. Lists the verdict thresholds the aggregator will apply.

The sub-issues are uniform-shaped, so a single Curator pass can file all
12 from the trainer keys in the table above. We deliberately do **not**
file them in this PR — the parent issue is the meta-tracker, the launcher
+ runbook + matrix doc are the buildable artifacts, and the per-trainer
sub-issues are the next decomposition step (handled by Curator after
review).

## Excluded combos and what to do if Tier-1 fails

If **all 12 Tier-1 cells land with `gap_closed_mean < 0.20`**, the
algorithm space alone is insufficient. The parent issue's decision
framework then escalates to **Tier-2** (env-mode screens on
`minimal_specialization` with IPPO as the trainer) — a separate launcher
will be authored if that path is needed. The trainer-only matrix is
self-contained; this PR does not block Tier-2.

If **some Tier-1 cells land in `[0.20, 0.49)` but none above 0.49**, the
parent issue specifies **Tier-3** (top-3 × top-3 cross product). That,
too, is a separate launcher. The hooks in `aggregate_tier1.py` (sorting
by `gap_closed_mean` desc, surfacing top cells) are designed to feed that
selection directly.

If **any Tier-1 cell lands `gap_closed_mean ≥ 0.49`**, declare "PPO gap
closed-by-trainer" and the parent issue can be closed with the verdict
ladder applied. No Tier-2 / Tier-3 launches are needed.

## What this PR does *not* do

- It does **not** launch the sweep. Per `CLAUDE.md`, multi-hour cluster
  work is operator-driven, not safe for an autonomous Builder session to
  start and walk away from.
- It does **not** file the 12 Tier-1 sub-issues. That is the natural next
  Curator action once this PR merges.
- It does **not** modify `run_tier1_cell.py` or `aggregate_tier1.py`. The
  driver + aggregator shipped in #346 are sufficient for the launch set
  above; the launcher only orchestrates them on a remote host.
- It does **not** define Tier-2 or Tier-3 launchers. Those are filed as
  follow-ups if and only if Tier-1 fails to produce a `≥ 0.49` cell.

## References

- Parent issue: [#343](https://github.com/rjwalters/bucket-brigade/issues/343) (meta).
- Driver issue: [#345](https://github.com/rjwalters/bucket-brigade/issues/345) / PR [#346](https://github.com/rjwalters/bucket-brigade/pull/346).
- Verdict ladder source: [`experiments/p3_specialization/diagnostics/random_mlp_search.py`](diagnostics/random_mlp_search.py) (`_classify_verdict`).
- Baseline constants: [`bucket_brigade/baselines.py`](../../bucket_brigade/baselines.py).
- Launcher: [`experiments/scripts/launch_tier1_sweep.sh`](../scripts/launch_tier1_sweep.sh).
- Runbook: [`TIER1_LAUNCH_RUNBOOK.md`](TIER1_LAUNCH_RUNBOOK.md).
- Option-A precedent (Nash phase-diagram fill): PR [#393](https://github.com/rjwalters/bucket-brigade/pull/393).
