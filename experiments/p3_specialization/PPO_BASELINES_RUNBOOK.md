# PPO baselines runbook (issue #384 → feeds #371)

Operator-driven launch + retrieve guide for the PPO baseline training
sweep that produces the frozen checkpoints shipped with the M4 release
(parent: [#371][issue-371], grandparent: [#365][issue-365] / [#357][issue-357]).

This runbook is the companion to:

- [`experiments/scripts/launch_ppo_baselines.sh`](../scripts/launch_ppo_baselines.sh) — the launcher
- [`experiments/p3_specialization/run_tier1_cell.py`](run_tier1_cell.py) — the `ippo` entry in `TRAINERS` (line 201)
- [`bucket_brigade/envs/scenarios_generated.py`](../../bucket_brigade/envs/scenarios_generated.py) — `SCENARIO_REGISTRY` (line 649)
- [`tests/test_launch_ppo_baselines.py`](../../tests/test_launch_ppo_baselines.py) — launcher contract tests

[issue-371]: https://github.com/rjwalters/bucket-brigade/issues/371
[issue-365]: https://github.com/rjwalters/bucket-brigade/issues/365
[issue-357]: https://github.com/rjwalters/bucket-brigade/issues/357

## What `ippo` is (and why it is the "baseline PPO")

`ippo` (Independent PPO) is the simplest PPO arm in `run_tier1_cell.py`'s
`TRAINERS` dispatch:

```python
"ippo": TrainerSpec(
    name="ippo",
    description="Independent PPO baseline (--algorithm ppo, no extras).",
    train_extra=("--algorithm", "ppo"),
),
```

No centralized critic (MAPPO), no opponent shaping (LOLA), no asymmetric
init (HetGPPO / #386), no value-loss tricks. Each agent runs its own PPO
update against its own GAE. This is the headline "PPO baseline" the paper
reports against and the artifact `#371` packages into
`bucket_brigade/baselines/release/ppo/`.

## Scope — what gets launched

By default the launcher runs the release scenario set defined in #384:

| Scenario                  | Source                                            | Why included                              |
|---------------------------|---------------------------------------------------|-------------------------------------------|
| `minimal_specialization`  | `minimal_specialization_scenario` (#199-family)   | Canonical P3 substrate                    |
| `default`                 | `default_scenario`                                | Vanilla 10-house environment              |
| `positional_default`      | `positional_default_scenario`                     | Positional variant of default             |
| `chain_reaction`          | `chain_reaction_scenario`                         | Cascade dynamics                          |
| `trivial_cooperation`     | `trivial_cooperation_scenario`                    | Simplest cooperation diagnostic           |
| `v2_minimal`              | `v2_minimal_scenario` (#254)                      | 2-house × 4-agent PPO learnability diagnostic |

All six names are keys of `SCENARIO_REGISTRY`. The launcher passes them
through to `run_tier1_cell.py --scenario <name>` verbatim. The versioned
release IDs (e.g. `minimal_specialization-v1`) are #371's concern when it
packages — at training time we use the bare names because that is what
the dispatcher consumes.

**Caveat**: `positional_default` is in `SCENARIO_REGISTRY` (the legacy
name-keyed registry) but does NOT have an entry in `SCENARIO_VERSIONS`
(`bucket_brigade/envs/registry.py`) as of this writing. Training works
fine. If #371 needs the versioned ID for its loader, the scenario needs
a `positional_default-v1` entry added to `SCENARIO_VERSIONS` — that is a
small follow-up and not in scope here.

### Compute estimate

Per the #384 budget:

| Path                                    | Cells | Wall-clock @ COMPUTE_HOST_CLUSTER |
|-----------------------------------------|-------|-----------------------------------|
| Default release set (6 scenarios × 3 seeds) | 6 | ~18 GPU-hours total              |
| Single scenario, 3 seeds                | 1     | ~1–2 h                            |
| Smoke (1 seed × 1 iter × 32 rollout)    | 1     | < 1 minute                        |

The bucket-brigade env is CPU-bound (see CLAUDE.md), so "GPU-hours" is
shorthand for wall-clock on the chosen host — the actual compute is CPU
even on a GPU box. `COMPUTE_HOST_CLUSTER` (alcubierre) and
`COMPUTE_HOST_PRIMARY` (Mac Studio) are both reasonable.

## Launch commands

All launches go through `launch_ppo_baselines.sh`. The script reads
`COMPUTE_HOST_PRIMARY` from `.env` and falls back to `_CLUSTER` /
`_LAMBDA` / `_GCP`. Pass `--host <alias>` to override.

### Plan A — Full release sweep (the canonical #384 launch)

```bash
./experiments/scripts/launch_ppo_baselines.sh
```

- 6 scenarios × 3 seeds × 50 iter × 2048 rollout
- Expected wall-clock: ~18 h total (one tmux session, scenarios run
  serially; seeds within a scenario also run serially inside
  `run_tier1_cell.py`)
- Outputs land under
  `experiments/p3_specialization/baselines/ippo_<scenario>/` on the
  remote host

### Plan B — Single scenario sanity (debug / positive-control rerun)

```bash
./experiments/scripts/launch_ppo_baselines.sh \
    --scenarios minimal_specialization \
    --seeds 42,43,44 \
    --num-iterations 25
```

- 1 scenario × 3 seeds × 25 iter — finishes in 1–2 h on most hosts
- Use this to confirm the trainer is wired correctly on a new host
  before burning compute on the full 6-scenario sweep

### Plan C — Shard across hosts (multi-host parallelism)

The launcher runs scenarios serially in one tmux session. To shard
across multiple hosts, launch the script once per host with a disjoint
scenario slice:

```bash
# Host 1: cheap scenarios
./experiments/scripts/launch_ppo_baselines.sh \
    --host alc-2 \
    --scenarios minimal_specialization,trivial_cooperation,v2_minimal

# Host 2: expensive scenarios
./experiments/scripts/launch_ppo_baselines.sh \
    --host alc-6 \
    --scenarios default,positional_default,chain_reaction
```

Each invocation creates an independent tmux session
(`ppo-baselines-<first_scenario>-and<N>more`) so concurrent launches on
the same host do not collide either.

### Plan D — Smoke test (local, safe)

```bash
# 1 seed × 1 iter × 32 rollout on minimal_specialization; finishes in seconds.
uv run python experiments/p3_specialization/run_tier1_cell.py \
    --trainer ippo \
    --scenario minimal_specialization \
    --seeds 42 \
    --num-iterations 1 \
    --rollout-steps 32 \
    --output-root /tmp/ppo_baseline_smoke \
    --skip-precheck
```

Expected: `== cell ippo_minimal_specialization complete: verdict=<some-tier>
gap_closed_mean=<some-float>`. The verdict on a 1-iteration smoke run is
meaningless — this only checks the dispatch + trainer + env wiring.

## Host assignments

Per `~/.claude/.../reference_cluster_host_reliability.md`:

- **Prefer**: `alc-2`, `alc-6`, `alc-9` (high reliability)
- **Avoid**: `alc-5` (12.5% reliable, confirmed flaky)
- **Personal CPU**: `COMPUTE_HOST_PRIMARY` (Mac Studio) — the env is
  CPU-bound, so the Mac Studio is competitive with GPU cluster nodes
  for this workload

## Retrieving results

```bash
# After the tmux session reports "cell complete" for every scenario:
rsync -avz <host>:bucket-brigade/experiments/p3_specialization/baselines/ \
    experiments/p3_specialization/baselines/

# Each cell writes:
#   ippo_<scenario>/cell_summary.json     -- gap_closed + verdict per seed
#   ippo_<scenario>/seed_<N>/             -- per-seed train.py output dir
#                                            (checkpoints, metrics.jsonl, config)
```

## Handoff to issue #371

This issue produces raw artifacts only. Issue #371 owns:

1. Selecting the best checkpoint per scenario from the per-seed sweep
2. Copying it to `bucket_brigade/baselines/release/ppo/<scenario>/checkpoint.pt`
3. Writing the sibling `config.json` (training hyperparameters + gap_closed)
4. Authoring `bucket_brigade/baselines/release/ppo/scores.json` with
   `{seed, gap_closed, config_hash}` per scenario
5. Adding the loader API (`bucket_brigade.baselines.load_ppo(scenario_id)`)
6. Smoke-testing the loader deserializes and runs

The handoff contract: as long as
`experiments/p3_specialization/baselines/ippo_<scenario>/` exists with a
non-empty `cell_summary.json` and at least one `seed_<N>/` subdirectory
containing a checkpoint, #371 can package it.

## What the launcher does NOT do

- It does **not** launch automatically. The operator runs it once and
  the remote tmux session does the work over hours.
- It does **not** package checkpoints into the release directory — that
  is #371's job.
- It does **not** parallelize seeds across hosts. For sharded
  multi-host launches, see Plan C above.
- It does **not** auto-merge results. The operator pulls results back,
  inspects the per-cell `cell_summary.json`, and PRs the artifacts.

## See also

- [`experiments/p3_specialization/het_ppo_runbook.md`](het_ppo_runbook.md) — sibling runbook for the asymmetry-aware PPO arm (#386)
- [`experiments/p3_specialization/TIER1_LAUNCH_RUNBOOK.md`](TIER1_LAUNCH_RUNBOOK.md) — sibling runbook for the full Tier-1 trainer matrix (#343)
- [`experiments/nash/phase_diagram/LAUNCH_RUNBOOK.md`](../nash/phase_diagram/LAUNCH_RUNBOOK.md) — sibling runbook for phase-diagram gap-fill (#390)
- [`bucket_brigade/envs/scenarios_generated.py`](../../bucket_brigade/envs/scenarios_generated.py) — `SCENARIO_REGISTRY`
- [`bucket_brigade/envs/registry.py`](../../bucket_brigade/envs/registry.py) — `SCENARIO_VERSIONS` (versioned IDs for #371's loader)
