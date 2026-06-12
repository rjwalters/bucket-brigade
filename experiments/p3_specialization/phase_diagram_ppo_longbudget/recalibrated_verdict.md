# #360 phase-diagram PPO results — recalibrated per-cell (issue #413)

Re-aggregation of the existing #360 PPO sweep results using the per-cell
Random / Specialist baselines from
`experiments/nash/phase_diagram/per_cell_baselines.json`. Columns:

* `OLD gap_closed`: original #360 metric — every cell scored against the
  single canonical MINSPEC_RANDOM / MINSPEC_SPECIALIST constants (current
  values: -87.72 / -28.38; the historical n=50 MINSPEC_SPECIALIST was
  -22.07 prior to the issue #416 n=10k re-derivation).
* `NEW gap_closed_homogeneous`: per-cell Random→SpecialistPolicy×4 (apples-
  to-apples drop-in for the MINSPEC tradition).
* `NEW gap_closed_ne`: per-cell Random→1×Hero+3×Firefighter (the heterogeneous
  NE asymmetric profile from the DO search). The metric appropriate for the
  paper §3/§4 NE-structure-vs-PPO-success hypothesis.

Sort: by NE verdict, then by `gap_closed_homogeneous_mean` descending.

| cell | NE verdict | seeds | OLD gap_closed | NEW gap_closed_homogeneous | NEW gap_closed_ne | cell random | cell homo | cell NE |
|------|------------|------:|---------------:|---------------------------:|------------------:|------------:|----------:|--------:|
| b0.90_k0.10_c0.50 | no_convergence | 4 | -0.058±0.272 | -0.058±0.272 | —±— | -98.87 | -86.33 | — |
| b0.50_k0.10_c0.50 | no_convergence | 4 | -0.084±0.179 | -0.084±0.179 | —±— | -98.87 | -86.33 | — |
| b0.50_k0.10_c2.00 | no_convergence | 4 | -0.111±0.212 | -0.111±0.212 | —±— | -101.87 | -86.61 | — |
| b0.10_k0.10_c0.50 | no_convergence | 4 | -0.115±0.261 | -0.115±0.261 | —±— | -98.87 | -86.33 | — |
| b0.10_k0.10_c2.00 | no_convergence | 4 | -0.131±0.189 | -0.131±0.189 | —±— | -101.87 | -86.61 | — |
| b0.90_k0.10_c2.00 | no_convergence | 4 | -0.149±0.187 | -0.149±0.187 | —±— | -101.87 | -86.61 | — |

## Ordering check

* **OLD gap_closed**: no_convergence (-0.108)
* **NEW gap_closed_homogeneous**: no_convergence (-0.108)
* **NEW gap_closed_ne**: no data
