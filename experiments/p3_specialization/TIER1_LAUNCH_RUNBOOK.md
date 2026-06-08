# Tier-1 sweep — operator runbook

This runbook is the operator companion to
[`experiments/scripts/launch_tier1_sweep.sh`](../scripts/launch_tier1_sweep.sh).
It runs the Tier-1 row of the trainer matrix tracked by
[issue #343][issue-343] using the per-cell driver shipped in
[`run_tier1_cell.py`](run_tier1_cell.py) (PR #346) and the post-sweep
aggregator [`aggregate_tier1.py`](aggregate_tier1.py).

[issue-343]: https://github.com/rjwalters/bucket-brigade/issues/343

The launch script does **not** run any compute locally — it shells into a
remote host listed in `.env`, pulls latest `main`, builds the Rust
extension, and starts a detached `tmux` session running each Tier-1 cell
sequentially. The actual sweep completes hours later inside the tmux
session; the operator's job is to launch, wait, then rsync and read the
verdict table.

## Prerequisites

- `.env` at the repo root defines at least one `COMPUTE_HOST_*` alias
  resolvable via the local `~/.ssh/config` (see `.env.example`).
- The remote host can build the Rust extension (`cargo`, `rustc`,
  Python ≥ 3.10, `uv`). The launcher's bootstrap handles the two
  CLAUDE.md gotchas (bare PATH under non-interactive ssh, missing pip in
  uv-created venvs) automatically.
- PR [#346][pr-346] (Tier-1 driver) is on `main`. Verify via
  `git log main --grep '#346'` — the launcher's remote import-check will
  fail loudly otherwise.

[pr-346]: https://github.com/rjwalters/bucket-brigade/pull/346

## Scope — what gets launched

By default the launcher runs the canonical Tier-1 launch set defined in
[`TIER1_SWEEP_MATRIX.md`](TIER1_SWEEP_MATRIX.md):

| Trainers (12)        | Scenario               | Seeds        | Iters | Rollout |
|----------------------|------------------------|--------------|-------|---------|
| `mappo`, `high_lambda`, `bc_init_continuation`, `bc_init_high_lambda`, `lola`, `hca`, `influence`, `nhr`, `progress`, `macro_actions`, `reinforce`, `pbt` | `minimal_specialization` | `42 43 44`   | `50`  | `2048`  |

Excluded by default: `ippo` (baseline; rerun adds no info) and `coma`
(author-deprioritized per #271). Both are re-enableable via `--trainers`.

### Compute estimate

Per the parent issue's budget (~30 min / cell on `COMPUTE_HOST_PRIMARY`):

| Path                    | Cells | Wall-clock @ primary |
|-------------------------|-------|----------------------|
| Default launch (12 cells) | 12    | ~6 h                 |
| + ippo + coma (14 cells)  | 14    | ~7 h                 |
| Cross-scenario × 3 scenarios | 36    | ~18 h                |

PBT is the slowest individual cell (~1 h+ depending on `--iters-per-gen`);
budget accordingly if running a single-trainer launch.

## Launch commands (copy-paste)

All commands assume a checkout of `main` (or the merged result of this PR)
and a working `.env` with at least one `COMPUTE_HOST_*` alias.

### Plan A — default Tier-1 launch (12 cells, ~6 h)

```bash
./experiments/scripts/launch_tier1_sweep.sh
```

Auto-resolves the host from `.env` (PRIMARY > CLUSTER > LAMBDA > GCP),
runs the full launch set on `minimal_specialization`, and ends with
`aggregate_tier1.py` writing `tier1_verdict.md` + `.json`.

### Plan B — sharded across multiple hosts

Split the launch set across two or three hosts to halve / third wall
clock. Each shard runs `--skip-aggregate` so the per-host outputs don't
race on the verdict file; the operator aggregates after rsync.

```bash
# Host 1 — cheap PPO-family arms (3 cells, ~1.5 h)
./experiments/scripts/launch_tier1_sweep.sh \
    --host alc-2 \
    --trainers mappo,high_lambda,reinforce \
    --skip-aggregate

# Host 2 — credit-assignment family (4 cells, ~2 h)
./experiments/scripts/launch_tier1_sweep.sh \
    --host alc-6 \
    --trainers lola,hca,influence,coma \
    --skip-aggregate

# Host 3 — env-side + BC-init + PBT (5 cells, ~3 h)
./experiments/scripts/launch_tier1_sweep.sh \
    --host alc-9 \
    --trainers nhr,progress,macro_actions,bc_init_continuation,bc_init_high_lambda,pbt \
    --skip-aggregate
```

After all three hosts finish, rsync each into the same local
`tier1_runs/` tree and aggregate locally (see "After cells finish"
below).

Host reliability ranking (from cluster operations): prefer alc-2 /
alc-6 / alc-9. **Avoid alc-5** (12.5% reliable per
`reference_cluster_host_reliability.md`). **Avoid alc-10** until
hardware is replaced.

### Plan C — robustness recheck on a Tier-1 candidate

If Plan A surfaces a trainer with `gap_closed_mean ≥ 0.49`, re-run that
single trainer on the three-scenario rotation called out in the parent
issue's "Open questions":

```bash
./experiments/scripts/launch_tier1_sweep.sh \
    --trainers <winning_trainer> \
    --scenarios minimal_specialization,default,chain_reaction \
    --output-root experiments/p3_specialization/tier1_runs_robustness
```

A verdict that holds at `≥ 0.49` on all three scenarios closes #343
with "PPO gap closed-by-trainer". A verdict that survives only on
`minimal_specialization` is reported in the comment thread and motivates
Tier-3 cross-product.

### Smoke test (local, ~30 s) — not for verdicts

```bash
./experiments/scripts/launch_tier1_sweep.sh \
    --trainers mappo \
    --num-iterations 1 \
    --rollout-steps 64 \
    --dry-run
```

Confirms argv plumbing without consuming compute. **Never use `--num-iterations
1 --rollout-steps 64` for verdict-eligible cells** — the trailing-5 mean is
meaningless at that horizon.

## Monitoring

The launch script prints the tmux session name (e.g. `tier1-sweep-n12-mappo`)
and log path on success. From local:

```bash
# Live attach
ssh <host> -t 'tmux attach -t tier1-sweep-n12-mappo'

# Tail the log
ssh <host> 'tail -f bucket-brigade/experiments/p3_specialization/tier1_runs/tier1-sweep-n12-mappo.log'

# Per-cell progress (as cells complete, each writes its cell_summary.json)
ssh <host> 'ls -la bucket-brigade/experiments/p3_specialization/tier1_runs/*/cell_summary.json'
```

## After cells finish — rsync, aggregate, report

Once **all** assigned plans on **all** hosts have completed, do the merge
locally:

```bash
# 1. Pull per-host outputs into the canonical tier1_runs/ tree.
#    Cells from different hosts land in disjoint <trainer>_<scenario>/
#    subdirs (Host 1 writes mappo_*/+high_lambda_*/+reinforce_*/, Host 2
#    writes lola_*/+hca_*/+influence_*/+coma_*/, etc.) so rsync is
#    non-destructive.
for host in alc-2 alc-6 alc-9; do
    rsync -avz "$host:bucket-brigade/experiments/p3_specialization/tier1_runs/" \
        experiments/p3_specialization/tier1_runs/
done

# 2. Regenerate the verdict table from the merged cells.
uv run python experiments/p3_specialization/aggregate_tier1.py \
    --tier1-root experiments/p3_specialization/tier1_runs

# 3. Inspect.
cat experiments/p3_specialization/tier1_runs/tier1_verdict.md
```

The verdict table sorts trainers by `gap_closed_mean` descending and tags
each with one of:

- `closed` (`>= 0.88`) — stunning near-specialist; PPO failure becomes mysterious
- `partial_upper` (`[0.49, 0.88)`) — **anti-attractor confirmed**, closes the PPO gap
- `partial_lower` (`[0.20, 0.49)`) — basin-trap consistent, ≈ PPO ceiling
- `insufficient` (`< 0.20`) — random-play basin
- `no_data` — cell missing or all seeds failed

## Reporting the verdict back to #343

Per the parent issue's "Test plan", #343 is closed when at least one
Tier-1 cell reaches `gap_closed ≥ 0.49`. Comment template:

```markdown
Tier-1 launch complete (`<sweep tag>`, git SHA `<sha>`).

Verdict table: `experiments/p3_specialization/tier1_runs/tier1_verdict.md`.

Top cell:
- `<trainer>` — `gap_closed_mean = <value>` (verdict: `<tier>`).

[ ] Above the 0.49 bar → "PPO gap closed-by-trainer", #343 closeable.
[ ] All below 0.20 → algorithm space alone insufficient, file Tier-2.
[ ] Mixed → file Tier-3 (top-3 × top-3 cross product).
```

Then commit the merged `tier1_runs/` tree (the per-cell directories +
`tier1_verdict.md` + `tier1_verdict.json`) under
`experiments/p3_specialization/diagnostics/results/issue343_tier1/` or
similar, mirroring the precedent set by prior diagnostic runs
(`issue220_obsfix/`, `issue231_mappo/`, etc.).

## Troubleshooting

- **Host unreachable**: the launcher aborts with exit 4 before consuming
  any compute. Check `ssh -v <host>` and the host's reachability/power.
- **Unknown trainer**: the launcher aborts with exit 5 before connecting.
  See the `KNOWN_TRAINERS` block in `launch_tier1_sweep.sh` and the
  authoritative `TRAINERS` table in `run_tier1_cell.py`.
- **Build failure on remote**: if `bucket-brigade-core/build.sh` complains
  about a stale `.so`, the remote venv is missing pip. The script seeds
  pip automatically; if that fails, ssh in and run
  `uv pip install pip && bash bucket-brigade-core/build.sh` once by hand.
- **One cell crashes mid-sweep**: the launcher uses `;` (not `&&`) between
  cells so a single failure does not abort the remaining sweep. The
  failed cell still gets a `cell_summary.json` with `verdict_tier =
  no_data`; the aggregator surfaces that explicitly in the verdict table.
- **PBT writes wrong-shaped artifacts**: the PBT orchestrator
  (`run_issue288_pbt.py`) writes per-seed metrics under
  `seed_<S>/metrics.json` matching the schema the aggregator expects.
  If the PBT cell shows `n_seeds_completed = 0` despite running, inspect
  `tier1_runs/pbt_minimal_specialization/seed_*/metrics.json` and check
  the PBT script's `--output-dir` semantics.
- **Cross-host clock skew**: rsync uses size+mtime, which can lose a
  partial cell if the remote clock is significantly off. Use
  `rsync -avz --checksum` if you suspect this.

## See also

- [Matrix decision doc](TIER1_SWEEP_MATRIX.md) — why these 12 cells.
- [Driver source](run_tier1_cell.py) — the `TRAINERS` dispatch table.
- [Aggregator source](aggregate_tier1.py) — verdict ladder + table format.
- [Parent issue #343](https://github.com/rjwalters/bucket-brigade/issues/343).
- [Driver PR #346](https://github.com/rjwalters/bucket-brigade/pull/346).
- [Option-A precedent (Nash phase-diagram)](../nash/phase_diagram/LAUNCH_RUNBOOK.md).
