# #360 phase-diagram PPO results — recalibrated per-cell (issue #413)

Re-aggregation of the existing #360 PPO sweep results using the per-cell
Random / Specialist baselines from
`experiments/nash/phase_diagram/per_cell_baselines.json`. Columns:

* `OLD gap_closed`: original #360 metric — every cell scored against the
  single canonical MINSPEC_RANDOM = -87.72 / MINSPEC_SPECIALIST = -22.07.
* `NEW gap_closed_homogeneous`: per-cell Random→SpecialistPolicy×4 (apples-
  to-apples drop-in for the MINSPEC tradition).
* `NEW gap_closed_ne`: per-cell Random→1×Hero+3×Firefighter (the heterogeneous
  NE asymmetric profile from the DO search). The metric appropriate for the
  paper §3/§4 NE-structure-vs-PPO-success hypothesis.

Sort: by NE verdict, then by `gap_closed_homogeneous_mean` descending.

| cell | NE verdict | seeds | OLD gap_closed | NEW gap_closed_homogeneous | NEW gap_closed_ne | cell random | cell homo | cell NE |
|------|------------|------:|---------------:|---------------------------:|------------------:|------------:|----------:|--------:|
| b0.50_k0.90_c0.50 | asymmetric_only | 4 | 0.319±0.182 | 0.098±0.109 | 0.162±0.179 | -77.59 | 31.97 | -10.88 |
| b0.90_k0.90_c0.50 | asymmetric_only | 4 | 0.206±0.193 | 0.031±0.115 | 0.051±0.190 | -77.59 | 31.97 | -10.88 |
| b0.50_k0.50_c0.50 | symmetric_only | 4 | 0.093±0.096 | 0.106±0.105 | 0.334±0.331 | -87.93 | -28.38 | -69.01 |
| b0.90_k0.50_c0.50 | symmetric_only | 4 | 0.088±0.099 | 0.101±0.109 | 0.317±0.344 | -87.93 | -28.38 | -69.01 |
| b0.10_k0.10_c0.50 | no_convergence | 4 | -0.161±0.048 | 0.048±0.253 | —±— | -98.87 | -86.33 | — |
| b0.90_k0.10_c0.50 | no_convergence | 4 | -0.183±0.047 | -0.069±0.248 | —±— | -98.87 | -86.33 | — |
| b0.50_k0.10_c0.50 | no_convergence | 4 | -0.185±0.062 | -0.078±0.323 | —±— | -98.87 | -86.33 | — |

## Ordering check

* **OLD gap_closed**: asymmetric_only (0.262) > symmetric_only (0.091) > no_convergence (-0.176)
* **NEW gap_closed_homogeneous**: symmetric_only (0.103) > asymmetric_only (0.065) > no_convergence (-0.033)
* **NEW gap_closed_ne**: symmetric_only (0.326) > asymmetric_only (0.106)
