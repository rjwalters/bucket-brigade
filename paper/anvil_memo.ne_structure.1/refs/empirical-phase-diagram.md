# Source: empirical 7-cell phase diagram preview (PRs #387, #391, #392, #393)

Empirical anchor for the analytical NE predictions of `ne_structure.md`.

## Files

- `experiments/nash/phase_diagram/results.json` — per-cell aggregate written by `compute_nash_phase_diagram.py` after a localhost crash on 2026-06-05. 7 cells from the originally-planned 75-cell grid (3×3×1 slice over $(\beta, \kappa)$ at $c{=}0.5$; the $c \in \{1.0, 2.0\}$ slices were not collected before the gap-fill launcher of #393).
- `experiments/nash/phase_diagram/phase_diagram_table.md` — human-readable per-cell verdict table generated from the same data.

## Per-cell verdicts (as of 2026-06-08)

| c | β | κ | verdict | best team payoff | converged restarts |
|---|---|---|---|---|---|
| 0.50 | 0.10 | 0.10 | `no_convergence` | -9692.67 | 0/20 |
| 0.50 | 0.50 | 0.10 | `no_convergence` | -9692.67 | 0/20 |
| 0.50 | 0.50 | 0.50 | `symmetric_only` | -648.01 | 3/20 |
| 0.50 | 0.50 | 0.90 | `asymmetric_only` |   72.01 | 14/20 |
| 0.50 | 0.90 | 0.10 | `no_convergence` | -9692.67 | 0/20 |
| 0.50 | 0.90 | 0.50 | `symmetric_only` | -648.01 | 3/20 |
| 0.50 | 0.90 | 0.90 | `asymmetric_only` |   72.01 | 14/20 |

## Solver and verdict definitions

- Driver: `experiments/scripts/compute_nash_phase_diagram.py`. `BASE_SCENARIO_NAME = "minimal_specialization"` (line 135) — the per-agent ownership-dominant 4-agent / 10-house scenario from `bucket_brigade/envs/scenarios_generated.py:minimal_specialization_scenario` (line 570).
- Equilibrium solver: heterogeneous Double-Oracle in `bucket_brigade/equilibrium/double_oracle_heterogeneous.py` over the continuous 10-D heuristic-parameter strategy menu (FF / FR / Hero / Coord / Liar archetypes plus discovered best-responses).
- Verdict labels: `symmetric_only` / `asymmetric_only` / `mixed` / `no_convergence` per the driver's classification logic (gap between best symmetric and best asymmetric team payoff, with $\epsilon{=}50$ payoff units as the deduplication tolerance).
- "Converged restarts" = restarts whose DO loop reached fixed-point within `max_iterations=20`.

## Headline pattern (the prediction target)

Holding $c{=}0.5$ fixed, the verdict moves only with $\kappa$:

- $\kappa = 0.1$: `no_convergence` regardless of $\beta$ (-9692.67 payoff; the all-RUINED collapse).
- $\kappa = 0.5$: `symmetric_only` regardless of $\beta$ (-648.01 payoff).
- $\kappa = 0.9$: `asymmetric_only` regardless of $\beta$ (+72.01 payoff).

$\beta$ does not move the verdict at $c{=}0.5$ in the 4 cells where both $\beta$ values were sampled ($\beta \in \{0.5, 0.9\}$, $\kappa \in \{0.5, 0.9\}$). The $\beta=0.1$ row is empty except for the $\kappa{=}0.1$ cell. The $c \in \{1.0, 2.0\}$ slices are unsampled — extending the test of the analytical prediction beyond $c{=}0.5$ requires the gap-fill launcher of #393 to complete (tracked as issue #358).

## Replication

```bash
uv run python experiments/scripts/compute_nash_phase_diagram.py \
  --beta 0.1,0.5,0.9 --kappa 0.1,0.5,0.9 --c 0.5
```

Per-cell wall-clock on the localhost cluster ran 2.5–18 hours per cell (see `elapsed_seconds` in `results.json`); the full 75-cell grid is the M1.1 deliverable of issue #358 on the alc-N cluster.
