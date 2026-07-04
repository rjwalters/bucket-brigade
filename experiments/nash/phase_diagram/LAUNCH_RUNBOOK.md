# Phase-diagram preview gap-fill — operator runbook

This runbook is the operator companion to
[`experiments/scripts/launch_phase_diagram_fill.sh`](../../scripts/launch_phase_diagram_fill.sh).
It is used to fill the missing cells of the `(β, κ, c)` heterogeneous Nash
preview grid documented in `RECOVERY_NOTES.md` (committed in PR #387; the
file will sit alongside this runbook once #387 merges) and tracked by
[issue #390][issue-390].

[issue-390]: https://github.com/rjwalters/bucket-brigade/issues/390

The launch script does **not** run any compute locally — it shells into a
remote host listed in `.env`, pulls latest `main`, builds the Rust
extension, and starts a detached `tmux` session running the driver
([`compute_nash_phase_diagram.py`](../../scripts/compute_nash_phase_diagram.py))
scoped to the requested cells. The actual cell-fill completes hours later
inside the tmux session; the operator's job is to launch, wait, then rsync
and replot.

## β-axis policy for future sweeps (issue #442)

**β (`prob_fire_spreads_to_neighbor`) is inert in bernoulli extinguish
mode** — the engine phase order (extinguish → burn_out → spread →
spontaneous_ignition) ruins every still-burning house before the spread
phase runs, and the spread phase draws zero RNG in this mode. Cells that
differ only in β are repeat solves of the same game with a bit-identical
RNG stream. The committed 39-cell grid confirms this empirically: 11/13
(κ, c) columns are bit-identical across β, and the 2 residual columns are
exactly the cells re-solved in a separate post-crash batch (pure
double-oracle restart lottery — see `beta_residuals.md`).

**Decision: collapse the β axis in all future bernoulli-mode sweeps.**
Run a single canonical β (use β = 0.5, the value behind the registered
`asym_b05_k09_c05` scenario) — the 3-value β axis turns 13 effective
cells into 39 solves, i.e. 3× compute (~5.5 h/cell × 26 redundant cells
≈ 140 core-hours per grid) for zero information.

We considered the alternative — switching the sweep to **continuous
extinguish mode**, where β is live (issue #253) — and rejected it as the
*default*: it changes the environment family, invalidating comparability
with every committed artifact (NE cells, PPO trainability sweeps,
per-cell baselines, the workshop paper). If β-response is a research
question, launch a continuous-mode grid as a deliberately separate
experiment family with its own baselines; do not mix modes within this
grid.

If you re-run individual cells (e.g. the issue #445 seeded retry), the
redundant β=0.5/β=0.9 twins should be treated as one cell.

## Prerequisites

The two performance / reliability fixes merged on 2026-06-08 must be on
`main`:

- **#391** — per-restart checkpointing in `HeterogeneousDoubleOracle.solve()`
  (a host outage now wastes at most 1/20 of a cell instead of a whole cell).
- **#392** (#389) — persistent `multiprocessing.Pool` + lifted worker cap
  (cells now saturate all cores instead of running on 1, ≈32× speedup ceiling
  on 32-core boxes; design budget of ~5.5 h/cell becomes realistic).

Without #392 each cell costs ~14 h wall-clock, which makes the fill
operationally infeasible on a single host. With #392 the full 11-cell fill
should complete in ~60 h on a healthy 32-core box, or ~22 h across three
hosts running in parallel.

The `.env` file at the repo root must define at least one `COMPUTE_HOST_*`
alias resolvable via the local `~/.ssh/config` (see `.env.example`).

## What is missing

The preview grid is `3 × 3 × N` over `(β, κ, c)` where:

- `β ∈ {0.10, 0.50, 0.90}`
- `κ ∈ {0.10, 0.50, 0.90}`
- `c` is operator-chosen — `PREVIEW_C_VALUES` in the driver is `(0.5, 2.0)`
  giving 18 cells; including `c = 1.0` gives 27.

Cells already committed in `preview/` (sibling of this runbook; arrives
with PR #387 — 10 cells, all at `c = 0.5`):

| β\κ  | 0.10 | 0.50 | 0.90 |
|------|------|------|------|
| 0.10 | ✅    | ❌    | ❌    |
| 0.50 | ✅    | ✅    | ✅    |
| 0.90 | ✅    | ✅    | ✅    |

`(7 unique cells + 3 alc-5 freq-cap duplicates)`

**Gaps required by issue #390 acceptance criteria**:

1. **β = 0.10 row, c = 0.50, κ ∈ {0.50, 0.90}** — 2 cells.
2. **All of c = 2.00** (`3 × 3 = 9` cells).
3. **All of c = 1.00** (`3 × 3 = 9` cells). *Optional under the 18-cell AC
   reading; required under the title's "(β=0.1 row, c=1.0, c=2.0)" reading.
   See the "Scope ambiguity" note below.*

Total: 11 cells (18-cell preview), or 20 cells (27-cell preview).

### Scope ambiguity

The issue body lists "All of c=1.0 and c=2.0 (12 cells)" but its acceptance
criterion is "All 18 cells of the (β, κ, c) preview grid". `PREVIEW_C_VALUES`
in the driver is `(0.5, 2.0)`, so 3×3×2 = 18 is the smaller reading and
excludes c=1.0. The title explicitly mentions c=1.0 though, suggesting the
author intends to extend the preview to 3×3×3 = 27.

**Operator decision required**: choose one before launching.

- **Minimal (18-cell)**: fill only Plan A and Plan B below. Stops at AC.
- **Extended (27-cell)**: fill Plan A + Plan B + Plan C. Closes the issue
  with full title-level coverage and gives the c-axis triangulation the
  body's rationale section asks for.

The script is identical for both; the extended path just adds the Plan C
launch.

## Host assignment

The preview can be parallelized across hosts cleanly because the driver's
per-cell artifacts live under `preview/cells/<tag>/` and are independently
written (one process per host, but they all write into disjoint cell
directories that get rsynced back). **Avoid running multiple hosts into
the *same* `preview/cells/` tree** — rsync afterwards.

Recommended split (assumes 3 healthy 32-core hosts; substitute with what
you actually have in `.env`):

| Plan | Cells           | Suggested host                  | Approx wall-clock @ 32 cores |
|------|-----------------|----------------------------------|------------------------------|
| A    | β=0.10 row gaps | alc-2 (or any reliable 32-core)  | ~11 h (2 cells × ~5.5 h)     |
| B    | c=2.0 plane     | alc-6 (or replacement for alc-10)| ~50 h (9 cells × ~5.5 h)     |
| C    | c=1.0 plane     | alc-9                            | ~50 h (9 cells × ~5.5 h)     |

Host reliability ranking (from cluster operations): prefer alc-2 / alc-6 /
alc-9. **Avoid alc-5** (12.5% reliable, confirmed flaky — requires the
4.5 GHz frequency cap workaround for any sustained load). **Avoid alc-10**
until hardware is replaced (offline as of 2026-06-08).

If you only have **one** healthy host, run Plans A → B → C sequentially on
the same host; the driver caches completed cells via `summary.json` so it
is safe to re-launch.

## Launch commands (copy-paste)

All commands assume a checkout of `main` (or the merged result of
PR #387 + this PR) and a working `.env`.

### Plan A — β=0.1 row gaps (c=0.5, κ ∈ {0.5, 0.9})

```bash
./experiments/scripts/launch_phase_diagram_fill.sh \
    --host alc-2 \
    --beta-values 0.1 \
    --kappa-values 0.5,0.9 \
    --c-values 0.5
```

Expected: 2 cells, ~11 h on a 32-core box.

### Plan B — c=2.0 plane (all 9 cells)

```bash
./experiments/scripts/launch_phase_diagram_fill.sh \
    --host alc-6 \
    --beta-values 0.1,0.5,0.9 \
    --kappa-values 0.1,0.5,0.9 \
    --c-values 2.0
```

Expected: 9 cells, ~50 h on a 32-core box.

### Plan C — c=1.0 plane (all 9 cells, optional)

```bash
./experiments/scripts/launch_phase_diagram_fill.sh \
    --host alc-9 \
    --beta-values 0.1,0.5,0.9 \
    --kappa-values 0.1,0.5,0.9 \
    --c-values 1.0
```

Expected: 9 cells, ~50 h on a 32-core box.

### One-host serial fallback (minimal, 18-cell coverage)

```bash
# Both A and B on one box. The driver appends results into the same
# preview/cells/ tree, so don't run these concurrently on the same host.
./experiments/scripts/launch_phase_diagram_fill.sh --host alc-9 \
    --beta-values 0.1 --kappa-values 0.5,0.9 --c-values 0.5

# Wait for plan A to finish before launching plan B (or use a different
# tmux session name and accept that the host will time-slice; the
# checkpointing in #391 makes this safe but slower).
./experiments/scripts/launch_phase_diagram_fill.sh --host alc-9 \
    --beta-values 0.1,0.5,0.9 --kappa-values 0.1,0.5,0.9 --c-values 2.0
```

## Monitoring

The launch script prints the tmux session name (`nash-fill-...`) and log
path on success. From local:

```bash
# Live attach
ssh <host> -t 'tmux attach -t nash-fill-c2.0'

# Tail the log
ssh <host> 'tail -f bucket-brigade/experiments/nash/phase_diagram/preview/nash-fill-c2.0.log'

# Check checkpoints (per-restart progress, from #391)
ssh <host> 'find bucket-brigade/experiments/nash/phase_diagram/preview/cells -name restarts_progress.json -newer /tmp/check'
```

## After cells finish — rsync, regenerate, replot

Once **all** assigned plans on **all** hosts have completed, do the merge
locally:

```bash
# 1. Pull per-host outputs into the canonical preview/ tree.
#    Cells from different hosts land in disjoint cells/<tag>/ subdirs
#    (Plan A writes b0.10_k0.50_c0.50/ + b0.10_k0.90_c0.50/, Plan B
#    writes b*_k*_c2.00/, etc.) so rsync is non-destructive.
for host in alc-2 alc-6 alc-9; do
    rsync -avz "$host:bucket-brigade/experiments/nash/phase_diagram/preview/cells/" \
        experiments/nash/phase_diagram/preview/cells/
done

# 2. Regenerate the aggregate results.json. The driver's --preview default
#    is 3×3×2; if you ran the optional Plan C, pass explicit --c-values to
#    include 1.0 in the aggregate.
# Minimal (18-cell preview):
uv run python experiments/scripts/compute_nash_phase_diagram.py --preview \
    --output-dir experiments/nash/phase_diagram/preview

# Extended (27-cell preview):
uv run python experiments/scripts/compute_nash_phase_diagram.py \
    --beta-values 0.1,0.5,0.9 --kappa-values 0.1,0.5,0.9 --c-values 0.5,1.0,2.0 \
    --output-dir experiments/nash/phase_diagram/preview

# 3. Regenerate the PNG + table.
uv run python experiments/scripts/plot_phase_diagram.py \
    --results experiments/nash/phase_diagram/preview/results.json \
    --out-png experiments/nash/phase_diagram/phase_diagram.png \
    --out-md  experiments/nash/phase_diagram/phase_diagram_table.md
```

The final `phase_diagram.png` should have no white `—` placeholder cells
in the `c = 0.50` (and `c = 1.00` if extended) and `c = 2.00` subplots —
that visual check is the AC for issue #390.

## Updating the notes file

Append a "Plane comparison" section to `RECOVERY_NOTES.md` (sibling file,
arrives with PR #387) describing whether the c=1.0 and/or c=2.0 planes
reveal a β-driven transition (the c=0.5 plane was purely κ-driven). That
answer is the analytical payoff the issue exists to produce.

## Troubleshooting

- **Host unreachable**: the script aborts with exit 4 before consuming
  any compute. Check `ssh -v <host>` and the host's reachability/power.
- **Build failure on remote**: if `bucket-brigade-core/build.sh` complains
  about a stale `.so`, the remote venv is missing pip. The script seeds
  pip automatically; if that fails, ssh in and run
  `uv pip install pip && bash bucket-brigade-core/build.sh` once by hand.
- **Driver crashes mid-cell**: thanks to #391, re-launching the script
  with the same arguments resumes from the last completed restart in
  each cell. No `--force` needed.
- **alc-5 / alc-10 hardware**: alc-5 needs the 4.5 GHz frequency cap
  (see the freq-cap section of `RECOVERY_NOTES.md` once PR #387 merges:
  bit-identical results vs boost clocks, with no crashes for 12 h+);
  alc-10 is still offline as of recovery. Substitute another reliable
  host from `.env`.
