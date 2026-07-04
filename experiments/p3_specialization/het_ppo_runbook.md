# het_ppo runbook (issue #386)

Operator-driven launch + retrieve guide for the asymmetry-aware PPO arm.

This runbook is the companion to:

- `experiments/scripts/launch_het_ppo_sweep.sh` — the launcher
- `bucket_brigade/training/joint_trainer.py` — the `per_agent_init_seed_offset` kwarg
- `experiments/p3_specialization/run_tier1_cell.py` — the `het_ppo` entry in `TRAINERS`
- `tests/test_joint_trainer.py::TestAsymmetryAwareInit` — the trainer-side tests
- `tests/test_tier1_driver.py::test_het_ppo_*` — the dispatch tests
- `tests/test_launch_het_ppo_sweep.py` — the launcher contract tests

## What `het_ppo` is

`het_ppo` is the HetGPPO-style (Bettini et al. AAMAS 2023, arXiv:2301.07137)
positive baseline for `asymmetric_only` phase-diagram cells. The architectural
contribution is small: the existing `JointPPOTrainer` already maintains
`num_agents` independent `PolicyNetwork` instances with `num_agents`
independent Adam optimizers (see `joint_trainer.py` line 462). What changed
in issue #386:

- A new `per_agent_init_seed_offset` kwarg on `JointPPOTrainer`. When set
  (recommended: `1000`), each policy `i` is initialized under its own seed
  `seed + i * offset` so the per-position parameter draws come from disjoint
  RNG streams. Default (`None`) preserves bit-identical legacy single-stream
  init.
- A new `het_ppo` entry in `run_tier1_cell.py::TRAINERS` that dispatches to
  `train.py --algorithm ppo --per-agent-init-seed-offset 1000`.

**Why the offset matters.** The legacy path constructs all `N` policies off a
single `torch.manual_seed(seed)` stream. The per-agent parameter tensors end
up *different* (each `nn.Linear` consumes RNG state in order) but their
starting points are correlated. On `asymmetric_only` cells (where the Nash
equilibrium is per-position-distinct), that shared-stream coupling can bias
SGD into a symmetric basin. Forcing disjoint init streams completes the
symmetry break at step 0.

## Acceptance criteria (per issue #386)

### Phase 1: rest_trap anchor

- 20 seeds × default budget (50 iter × 2048 rollout)
- Target: `gap_closed_mean >= 0.49` (`partial_upper` tier or better)

Tier-1 baselines (IPPO, MAPPO, COMA, HCA, social-influence, etc.) fail by
construction on `rest_trap` (#356); the asymmetric variant is expected to
beat them clearly. If `het_ppo` returns `insufficient` on `rest_trap`, the
asymmetry-aware story collapses and needs reframing.

### Phase 2: phase-diagram `asymmetric_only` cells

From `experiments/nash/phase_diagram/results.json` (per-cell ground truth;
note that `phase_diagram_table.md` renders columns in `c | β | κ` order — an
earlier revision of this runbook misread it as `β | κ | c` and derived two
non-existent c=0.90 cells, corrected under issue #435):

| β | κ | c | verdict | NE team payoff (per episode) | scenario name |
|---|---|---|---------|------------------------------|---------------|
| 0.5 | 0.9 | 0.5 | `asymmetric_only` | 72.0095 (14/20 converged) | `asym_b05_k09_c05` |
| 0.9 | 0.9 | 0.5 | `asymmetric_only` | 72.0095 (14/20 converged) | `asym_b09_k09_c05` |

These are the only `asymmetric_only` cells in the committed phase diagram.
Issue #435 registered both as first-class named scenarios
(`bucket_brigade/envs/scenarios_generated.py`, frozen IDs
`asym_b05_k09_c05-v1` / `asym_b09_k09_c05-v1` in
`bucket_brigade/envs/registry.py`), bit-identical to the
`make_phase_diagram_scenario(β, κ, c)` construction the #358 NE search used
— so the NE artifacts above remain citable for the named scenarios. Random
baselines + gap references are wired through
`bucket_brigade.baselines.SCENARIO_RANDOM_BASELINES` /
`SCENARIO_GAP_REFERENCES` (measured under #435: -78.27/step, 95% CI
[-83.88, -72.81], n=1000, #237 protocol).

**Replication-pair caveat (found during the #435 baseline measurement)**: β
(`prob_fire_spreads_to_neighbor`) is inert in the bernoulli extinguish mode
these scenarios use — the engine step order is extinguish → burn_out →
spread → spark, and burn_out ruins every still-burning house each night, so
no house is ever BURNING when the spread phase runs. The two cells are
therefore the *same effective environment*: identical NE payoffs in the
committed phase diagram, bit-identical random baselines under the same
seeds. Treat the Phase 2 pair as a replication pair (two names, one
dynamics), not a β sweep dimension.

### Phase 3 (optional): `symmetric_only` sanity check

Per the issue:

> sanity check: does dropping parameter sharing hurt on `symmetric_only`
> cells where the symmetric NE is optimal? Expected: yes, mildly
> (sample-efficiency loss without equilibrium-structure gain).

Run `het_ppo` on `minimal_specialization` (the canonical `symmetric_only`
anchor — see #354) and compare to IPPO. Useful framing for the paper but not
strictly required by the issue's acceptance criteria.

## Launch commands

All launches go through `launch_het_ppo_sweep.sh`. The script reads
`COMPUTE_HOST_PRIMARY` from `.env` and falls back to `_CLUSTER` / `_LAMBDA` /
`_GCP`. Pass `--host <alias>` to override.

### Plan A — Phase 1 anchor (smallest viable launch)

```bash
./experiments/scripts/launch_het_ppo_sweep.sh \
    --scenarios rest_trap
```

- 20 seeds (the default) × 50 iter × 2048 rollout × 1 cell
- Expected wall-clock: 30 min – 2 h per seed on a CPU host, ~20 h total at
  16-way parallelism (depends on host). The trainer is single-process per
  seed; parallelism comes from running multiple seeds concurrently, which
  this launcher does NOT do — each seed runs sequentially inside one tmux
  session. For multi-host parallelism, launch this command on each host with
  a disjoint seed slice (`--seeds 42,43,44,45,46`) and rsync results into a
  shared directory.

### Plan B — Phase 1 fast turnaround (debug / positive-control rerun)

```bash
./experiments/scripts/launch_het_ppo_sweep.sh \
    --scenarios rest_trap \
    --seeds 42,43,44 \
    --num-iterations 25
```

- 3 seeds × 25 iter — finishes in 1–3 h on most hosts
- Use this to confirm the trainer is wired correctly on a new host before
  burning compute on the full 20-seed sweep

### Plan C — Phase 2 (asymmetric_only scenarios registered by #435)

```bash
./experiments/scripts/launch_het_ppo_sweep.sh \
    --scenarios rest_trap,asym_b05_k09_c05,asym_b09_k09_c05
```

Each cell appends a fresh `het_ppo_<scenario>/cell_summary.json` under
`experiments/p3_specialization/tier1_runs/`. The launcher loops over
scenarios serially inside one tmux session.

## Host assignments

Per `~/.claude/.../reference_cluster_host_reliability.md`:

- **Prefer**: `alc-2`, `alc-6`, `alc-9` (high reliability)
- **Avoid**: `alc-5` (12.5% reliable, confirmed flaky)
- **Personal CPU**: `COMPUTE_HOST_PRIMARY` (Mac Studio) — default for
  evolution / Nash / P3 sweeps. The env is CPU-bound; parallelism wins, not
  GPU.

## Retrieving results

```bash
# After the tmux session reports "cell complete" for every scenario:
rsync -avz <host>:bucket-brigade/experiments/p3_specialization/tier1_runs/ \
    experiments/p3_specialization/tier1_runs/

# Regenerate the tier-1 aggregate table (this picks up het_ppo automatically
# as long as the cell_summary.json files are under tier1_runs/):
uv run python experiments/p3_specialization/aggregate_tier1.py \
    --root experiments/p3_specialization/tier1_runs/

# Commit results to a feature branch (do NOT push to main without review):
git checkout -b results/het-ppo-issue-386
git add experiments/p3_specialization/tier1_runs/het_ppo_*
git commit -m "exp(p3): het_ppo sweep results — issue #386"
```

## What the launcher does NOT do

- It does **not** launch automatically. The operator runs it once and the
  remote tmux session does the work over hours.
- It does **not** parallelize seeds across multiple hosts. For an asymmetric
  N-host launch, run the script N times with `--seeds <slice>` on each host
  and rsync results into a shared directory.
- It does **not** auto-merge results. The operator pulls results back,
  inspects the per-cell `cell_summary.json`, and commits/PRs.

## Smoke test (local, safe)

```bash
# 1 seed × 1 iter × 32 rollout on rest_trap; finishes in seconds.
uv run python experiments/p3_specialization/run_tier1_cell.py \
    --trainer het_ppo \
    --scenario rest_trap \
    --seeds 42 \
    --num-iterations 1 \
    --rollout-steps 32 \
    --output-root /tmp/het_ppo_smoke \
    --skip-precheck
```

Expected: `== cell het_ppo_rest_trap complete: verdict=<some-tier>
gap_closed_mean=<some-float>`. The verdict on a 1-iteration smoke run is
meaningless — this only checks the dispatch + trainer + env wiring.

## See also

- `experiments/nash/phase_diagram/LAUNCH_RUNBOOK.md` — sibling runbook for
  the phase-diagram fill operation (the #358/#390/#393 workflow this one
  follows).
- `bucket_brigade/training/joint_trainer.py` (lines ~290, 458–520) — kwarg
  docstring + init logic.
- `experiments/p3_specialization/tier1_runs/tier1_verdict.md` — the existing
  tier-1 verdict table. `het_ppo` results will land alongside the 14 trainers
  already there.
