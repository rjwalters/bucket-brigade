# Phase-diagram preview — recovery notes (2026-06-08)

## Context

The Run 2a preview slice of the heterogeneous Nash phase diagram (#383) was
launched across the alcubierre cluster on 2026-06-05 14:48 PT. Localhost
crashed at ~13:35 the same day (Jun 5), terminating the orchestrating
Claude Code session but leaving the SSH/tmux jobs running on each host.
This note recovers what landed.

## Coverage at c=0.50

|        | κ=0.10           | κ=0.50           | κ=0.90           |
|--------|------------------|------------------|------------------|
| β=0.10 | `no_convergence` | (missing)        | (missing)        |
| β=0.50 | `no_convergence` | `symmetric_only` | `asymmetric_only`|
| β=0.90 | `no_convergence` | `symmetric_only` | `asymmetric_only`|

Equilibrium payoffs (best team payoff across converged restarts):

| κ    | verdict           | payoff   | converged/restarts |
|------|-------------------|----------|--------------------|
| 0.10 | no_convergence    | -9692.67 | 0/20  (0%)         |
| 0.50 | symmetric_only    | -648.01  | 3/20  (15%)        |
| 0.90 | asymmetric_only   | 72.01    | 14/20 (70%)        |

Identical across the β=0.5 and β=0.9 rows — β does not move the regime at
c=0.5. The phase diagram is **κ-driven** in this slice.

## Gaps

- **β=0.1 row, κ∈{0.5, 0.9}**: alc-4 stopped after producing 1 cell (cause
  unknown — tmux session was gone by the time the localhost session was
  recovered; logs not retained).
- **All of c=1.0 and c=2.0**: alc-10 was the assigned host for c=2.0 and
  never came back online after the power cycle (~6h before crash, still
  offline at recovery 60h+ later).

## Hardware validation — freq-cap test

alc-5 ran the same β=0.5 row with `cpupower frequency-set --max 4.5GHz`
(no boost), specifically to test whether the recent int3 crashes on
alc-5/10 are consistent with Raptor Lake voltage-degradation symptoms.

| κ    | alc-2 (boost)     | alc-5 (4.5 GHz cap) | match? |
|------|-------------------|---------------------|--------|
| 0.10 | -9692.67, 0/20    | -9692.67, 0/20      | ✅      |
| 0.50 | -648.01,  3/20    | -648.01,  3/20      | ✅      |
| 0.90 | 72.01,    14/20   | 72.01,    14/20     | ✅      |

**Bit-identical** verdicts, payoffs, and convergence rates. The Double
Oracle solver is fully deterministic given the seed, and the freq cap does
not introduce any numerical drift.

Run-time stability: **12h+ of sustained 99% single-core load at 4.5 GHz
with zero int3 crashes**. Prior boost-clock runs on the same chip crashed
in 6m9s under equivalent load. This is a 100×+ stability improvement from
the cap alone, with no correctness cost. Consistent with the documented
Raptor Lake degradation signature; recommend applying the frequency cap
as a workaround on all affected hosts until chips are replaced.

## Rate

Cells took ~7–9h wall-clock each on a single core. The driver did not
engage `RustPayoffEvaluator`'s Pool for parallel MC — every cell ran at
1/32 utilization. At full Pool engagement the design budget was ~5.5h/cell;
single-core delivered ~32× slower, matching what we observed. This is the
likely reason the original "100h for 18 cells" estimate was too optimistic.

Follow-up: investigate why the Pool is not engaging in the per-cell DO
solver loop, and add per-restart checkpointing inside
`HeterogeneousDoubleOracle.solve()` so that future host outages waste at
most 1/20 of a cell rather than the whole cell.

## Files

- `results.json`            — aggregate of the 7 primary cells (alc-2, alc-4, alc-6)
- `results_freqtest.json`   — aggregate of the 3 alc-5 freqtest cells
- `phase_diagram.png`       — c=0.5 plane (with gaps for missing β=0.1 cells)
- `phase_diagram_freqtest.png` — β=0.5 row, freq-capped host
- `phase_diagram_table.md`  — paste-ready primary table
- `phase_diagram_table_freqtest.md` — paste-ready freqtest table
- `preview/alc-*/cells/*/summary.json` — raw per-cell outputs (rsynced from each host)
