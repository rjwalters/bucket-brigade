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
| b0.50_k0.90_c0.50 | asymmetric_only | 4 | 0.098±0.109 | 0.098±0.109 | 0.162±0.179 | -77.59 | 31.97 | -10.88 |
| b0.10_k0.70_c1.00 | asymmetric_only | 4 | 0.071±0.100 | 0.071±0.100 | 0.168±0.236 | -83.60 | 1.01 | -47.90 |
| b0.90_k0.70_c1.00 | asymmetric_only | 4 | 0.067±0.097 | 0.067±0.097 | 0.158±0.230 | -83.60 | 1.01 | -47.90 |
| b0.10_k0.70_c2.00 | asymmetric_only | 4 | 0.062±0.092 | 0.062±0.092 | 0.162±0.241 | -85.60 | 0.81 | -52.60 |
| b0.50_k0.70_c1.00 | asymmetric_only | 4 | 0.057±0.099 | 0.057±0.099 | 0.135±0.234 | -83.60 | 1.01 | -47.90 |
| b0.90_k0.70_c2.00 | asymmetric_only | 4 | 0.056±0.115 | 0.056±0.115 | 0.145±0.301 | -85.60 | 0.81 | -52.60 |
| b0.50_k0.70_c2.00 | asymmetric_only | 4 | 0.037±0.078 | 0.037±0.078 | 0.097±0.203 | -85.60 | 0.81 | -52.60 |
| b0.90_k0.90_c0.50 | asymmetric_only | 4 | 0.031±0.115 | 0.031±0.115 | 0.051±0.190 | -77.59 | 31.97 | -10.88 |
| b0.90_k0.10_c1.00 | asymmetric_only | 4 | -0.000±0.275 | -0.000±0.275 | -0.001±0.993 | -99.87 | -86.42 | -96.14 |
| b0.50_k0.10_c1.00 | asymmetric_only | 4 | -0.029±0.241 | -0.029±0.241 | -0.106±0.870 | -99.87 | -86.42 | -96.14 |
| b0.10_k0.10_c1.00 | asymmetric_only | 4 | -0.088±0.258 | -0.088±0.258 | -0.317±0.932 | -99.87 | -86.42 | -96.14 |
| b0.50_k0.50_c0.50 | symmetric_only | 4 | 0.106±0.105 | 0.106±0.105 | 0.334±0.331 | -87.93 | -28.38 | -69.01 |
| b0.50_k0.30_c1.00 | symmetric_only | 4 | 0.103±0.152 | 0.103±0.152 | 0.372±0.547 | -94.20 | -58.20 | -84.19 |
| b0.90_k0.50_c0.50 | symmetric_only | 4 | 0.101±0.109 | 0.101±0.109 | 0.317±0.344 | -87.93 | -28.38 | -69.01 |
| b0.90_k0.30_c1.00 | symmetric_only | 4 | 0.066±0.124 | 0.066±0.124 | 0.237±0.445 | -94.20 | -58.20 | -84.19 |
| b0.90_k0.30_c2.00 | symmetric_only | 4 | 0.057±0.113 | 0.057±0.113 | 0.269±0.531 | -96.20 | -58.39 | -88.19 |
| b0.90_k0.50_c1.00 | symmetric_only | 4 | 0.047±0.085 | 0.047±0.085 | 0.159±0.286 | -88.93 | -28.47 | -71.01 |
| b0.10_k0.50_c1.00 | symmetric_only | 4 | 0.037±0.099 | 0.037±0.099 | 0.125±0.333 | -88.93 | -28.47 | -71.01 |
| b0.10_k0.30_c2.00 | symmetric_only | 4 | 0.018±0.121 | 0.018±0.121 | 0.085±0.573 | -96.20 | -58.39 | -88.19 |
| b0.50_k0.50_c1.00 | symmetric_only | 4 | 0.015±0.110 | 0.015±0.110 | 0.051±0.373 | -88.93 | -28.47 | -71.01 |
| b0.50_k0.30_c2.00 | symmetric_only | 4 | 0.005±0.180 | 0.005±0.180 | 0.023±0.850 | -96.20 | -58.39 | -88.19 |
| b0.10_k0.30_c1.00 | symmetric_only | 4 | 0.002±0.062 | 0.002±0.062 | 0.007±0.223 | -94.20 | -58.20 | -84.19 |
| b0.10_k0.10_c0.50 | no_convergence | 4 | 0.048±0.253 | 0.048±0.253 | —±— | -98.87 | -86.33 | — |
| b0.50_k0.10_c2.00 | no_convergence | 4 | -0.057±0.173 | -0.057±0.173 | —±— | -101.87 | -86.61 | — |
| b0.10_k0.10_c2.00 | no_convergence | 4 | -0.059±0.251 | -0.059±0.251 | —±— | -101.87 | -86.61 | — |
| b0.90_k0.10_c0.50 | no_convergence | 4 | -0.069±0.248 | -0.069±0.248 | —±— | -98.87 | -86.33 | — |
| b0.90_k0.10_c2.00 | no_convergence | 4 | -0.075±0.290 | -0.075±0.290 | —±— | -101.87 | -86.61 | — |
| b0.50_k0.10_c0.50 | no_convergence | 4 | -0.078±0.323 | -0.078±0.323 | —±— | -98.87 | -86.33 | — |
| b0.10_k0.90_c2.00 | mixed | 4 | 0.097±0.113 | 0.097±0.113 | 0.207±0.240 | -80.59 | 31.67 | -27.82 |
| b0.10_k0.90_c1.00 | mixed | 4 | 0.070±0.123 | 0.070±0.123 | 0.142±0.248 | -78.59 | 31.87 | -23.92 |
| b0.90_k0.90_c1.00 | mixed | 4 | 0.068±0.105 | 0.068±0.105 | 0.138±0.213 | -78.59 | 31.87 | -23.92 |
| b0.90_k0.90_c2.00 | mixed | 4 | 0.059±0.095 | 0.059±0.095 | 0.127±0.201 | -80.59 | 31.67 | -27.82 |
| b0.50_k0.90_c2.00 | mixed | 4 | 0.057±0.101 | 0.057±0.101 | 0.121±0.214 | -80.59 | 31.67 | -27.82 |
| b0.50_k0.90_c1.00 | mixed | 4 | 0.043±0.091 | 0.043±0.091 | 0.086±0.184 | -78.59 | 31.87 | -23.92 |
| b0.90_k0.50_c2.00 | mixed | 4 | 0.034±0.082 | 0.034±0.082 | 0.085±0.204 | -90.93 | -28.67 | -66.08 |
| b0.10_k0.50_c2.00 | mixed | 4 | 0.013±0.065 | 0.013±0.065 | 0.032±0.164 | -90.93 | -28.67 | -66.08 |
| b0.50_k0.50_c2.00 | mixed | 4 | 0.010±0.083 | 0.010±0.083 | 0.026±0.208 | -90.93 | -28.67 | -66.08 |

## Ordering check

* **OLD gap_closed**: symmetric_only (0.051) > asymmetric_only (0.033) > no_convergence (-0.049)
* **NEW gap_closed_homogeneous**: symmetric_only (0.051) > asymmetric_only (0.033) > no_convergence (-0.049)
* **NEW gap_closed_ne**: symmetric_only (0.180) > asymmetric_only (0.059)
