# Heterogeneous Nash Phase Diagram — per-cell table
Base scenario: `minimal_specialization`
Grid: 3×5×3 = 39 cells

> **β-inertness caveat (issue #442)**: in bernoulli extinguish mode β (`prob_fire_spreads_to_neighbor`) is inert — burning houses are ruined before the spread phase runs and the spread phase draws zero RNG, so rows differing only in β are *repeat solves of the same game*. Any cross-β payoff/verdict difference below is double-oracle solver nondeterminism, quantified in `experiments/nash/phase_diagram/beta_residuals.md`. Future bernoulli-mode sweeps should collapse the β axis (see `LAUNCH_RUNBOOK.md`).

| c | β | κ | verdict | equilibrium_payoff | convergence_rate |
|---|---|---|---|---|---|
| 0.50 | 0.10 | 0.10 | `no_convergence` | -9692.67 | 0/20 (0%) |
| 0.50 | 0.10 | 0.50 | `symmetric_only` | -614.43 | 7/20 (35%) |
| 0.50 | 0.10 | 0.90 | `mixed` | 80.91 | 17/20 (85%) |
| 0.50 | 0.50 | 0.10 | `no_convergence` | -9692.67 | 0/20 (0%) |
| 0.50 | 0.50 | 0.50 | `symmetric_only` | -648.01 | 3/20 (15%) |
| 0.50 | 0.50 | 0.90 | `asymmetric_only` | 72.01 | 14/20 (70%) |
| 0.50 | 0.90 | 0.10 | `no_convergence` | -9692.67 | 0/20 (0%) |
| 0.50 | 0.90 | 0.50 | `symmetric_only` | -648.01 | 3/20 (15%) |
| 0.50 | 0.90 | 0.90 | `asymmetric_only` | 72.01 | 14/20 (70%) |
| 1.00 | 0.10 | 0.10 | `asymmetric_only` | -9599.59 | 1/20 (5%) |
| 1.00 | 0.10 | 0.30 | `symmetric_only` | -3479.64 | 2/20 (10%) |
| 1.00 | 0.10 | 0.50 | `symmetric_only` | -705.66 | 7/20 (35%) |
| 1.00 | 0.10 | 0.70 | `asymmetric_only` | -57.31 | 4/20 (20%) |
| 1.00 | 0.10 | 0.90 | `mixed` | 73.80 | 13/20 (65%) |
| 1.00 | 0.50 | 0.10 | `asymmetric_only` | -9599.59 | 1/20 (5%) |
| 1.00 | 0.50 | 0.30 | `symmetric_only` | -3479.64 | 2/20 (10%) |
| 1.00 | 0.50 | 0.50 | `symmetric_only` | -705.66 | 7/20 (35%) |
| 1.00 | 0.50 | 0.70 | `asymmetric_only` | -57.31 | 4/20 (20%) |
| 1.00 | 0.50 | 0.90 | `mixed` | 73.80 | 13/20 (65%) |
| 1.00 | 0.90 | 0.10 | `asymmetric_only` | -9599.59 | 1/20 (5%) |
| 1.00 | 0.90 | 0.30 | `symmetric_only` | -3479.64 | 2/20 (10%) |
| 1.00 | 0.90 | 0.50 | `symmetric_only` | -705.66 | 7/20 (35%) |
| 1.00 | 0.90 | 0.70 | `asymmetric_only` | -57.31 | 4/20 (20%) |
| 1.00 | 0.90 | 0.90 | `mixed` | 73.80 | 13/20 (65%) |
| 2.00 | 0.10 | 0.10 | `no_convergence` | -9681.69 | 0/20 (0%) |
| 2.00 | 0.10 | 0.30 | `symmetric_only` | -3538.30 | 1/20 (5%) |
| 2.00 | 0.10 | 0.50 | `mixed` | -740.02 | 6/20 (30%) |
| 2.00 | 0.10 | 0.70 | `asymmetric_only` | -77.87 | 3/20 (15%) |
| 2.00 | 0.10 | 0.90 | `mixed` | 57.63 | 9/20 (45%) |
| 2.00 | 0.50 | 0.10 | `no_convergence` | -9681.69 | 0/20 (0%) |
| 2.00 | 0.50 | 0.30 | `symmetric_only` | -3538.30 | 1/20 (5%) |
| 2.00 | 0.50 | 0.50 | `mixed` | -740.02 | 6/20 (30%) |
| 2.00 | 0.50 | 0.70 | `asymmetric_only` | -77.87 | 3/20 (15%) |
| 2.00 | 0.50 | 0.90 | `mixed` | 57.63 | 9/20 (45%) |
| 2.00 | 0.90 | 0.10 | `no_convergence` | -9681.69 | 0/20 (0%) |
| 2.00 | 0.90 | 0.30 | `symmetric_only` | -3538.30 | 1/20 (5%) |
| 2.00 | 0.90 | 0.50 | `mixed` | -740.02 | 6/20 (30%) |
| 2.00 | 0.90 | 0.70 | `asymmetric_only` | -77.87 | 3/20 (15%) |
| 2.00 | 0.90 | 0.90 | `mixed` | 57.63 | 9/20 (45%) |
