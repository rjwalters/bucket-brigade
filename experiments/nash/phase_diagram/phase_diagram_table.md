# Heterogeneous Nash Phase Diagram — per-cell table
Base scenario: `minimal_specialization`
Grid: 3×5×3 = 39 cells

> **β-inertness caveat (issue #442)**: in bernoulli extinguish mode β (`prob_fire_spreads_to_neighbor`) is inert — burning houses are ruined before the spread phase runs and the spread phase draws zero RNG, so rows differing only in β are *repeat solves of the same game*. Any cross-β payoff/verdict difference below is double-oracle solver nondeterminism, quantified in `experiments/nash/phase_diagram/beta_residuals.md`. Future bernoulli-mode sweeps should collapse the β axis (see `LAUNCH_RUNBOOK.md`).

> **(κ = 0.90, c = 0.50) anchor update (#459 / #466)**: the two `asymmetric_only` rows below (β = 0.50 / 0.90, payoff 72.01) record the solver's best-found profile `hero|FF|FF|FF`; the β = 0.10 row of the same game (`mixed`, 80.91) recorded `FF|hero|hero|FF`. The #459 exploitability audit found **both** committed profiles are ε-NE at the repo-standard ε = 50, and `FF|hero|hero|FF` is decisively better under winner's-curse-free CRN re-evaluation (55.36 ± 3.44 vs 45.80 ± 3.58 per episode; CRN paired +9.55 ± 2.73, t = +3.5). `FF|hero|hero|FF` is therefore the adopted NE anchor for the named scenarios `asym_b05_k09_c05` / `asym_b09_k09_c05` (#466 coordinated update). This table and `results.json` are the historical solve record and are retained unchanged. See `experiments/nash/phase_diagram/exploitability/RESULTS.md`.

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
