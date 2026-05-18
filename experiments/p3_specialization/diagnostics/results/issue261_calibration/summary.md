# Issue #261/#262 — Action-shaping calibration sweep

Scenario: ``minimal_specialization``  
References (per-step mean team reward): random=-87.72, specialist=-22.07 (denominator=+65.65).

## Per-cell results

| alpha | beta | n | team_reward (mean±std) | gap_closed_mean | mean_action_entropy_final | entropy_collapse_x |
|---|---|---|---|---|---|---|
| 0.0 | 0.0 | 3 | -80.31 ± 5.95 | +0.113 | 0.915 | 1.0 |
| 0.0 | 0.1 | 3 | -80.28 ± 5.73 | +0.113 | 0.620 | 1.5 |
| 0.0 | 0.5 | 3 | -77.98 ± 5.04 | +0.148 | 0.497 | 1.8 |
| 0.1 | 0.0 | 3 | -76.98 ± 5.07 | +0.164 | 0.752 | 1.2 |
| 0.1 | 0.1 | 3 | -81.83 ± 9.43 | +0.090 | 0.563 | 1.6 |
| 0.1 | 0.5 | 3 | -79.97 ± 3.54 | +0.118 | 0.492 | 1.9 |
| 0.5 | 0.0 | 3 | -80.31 ± 6.61 | +0.113 | 0.550 | 1.7 |
| 0.5 | 0.1 | 3 | -80.45 ± 9.38 | +0.111 | 0.496 | 1.8 |
| 0.5 | 0.5 | 3 | -77.69 ± 4.60 | +0.153 | 0.552 | 1.7 |
| 2.0 | 0.0 | 3 | -79.37 ± 7.14 | +0.127 | 0.270 | 3.4 |
| 2.0 | 0.1 | 3 | -80.82 ± 7.82 | +0.105 | 0.792 | 1.2 |
| 2.0 | 0.5 | 3 | -78.48 ± 3.61 | +0.141 | 0.499 | 1.8 |

## Verdict

**Headline**: ``ACTION_SHAPING_INSUFFICIENT``
**Tier**: ``tier_3_insufficient``

Best cell:
- alpha = ``0.1``, beta = ``0.0``
- gap_closed_mean = ``+0.164``
- team_reward_mean = ``-76.976``
- mean_action_entropy_final = ``0.752``
- n_seeds = ``3``

_No over-shaping flagged at the 100× threshold._

