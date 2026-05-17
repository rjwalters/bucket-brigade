# Issue #231 — IPPO vs MAPPO across three scenarios

Pre-registered references (per-step mean team reward):

| scenario | random | specialist | spec-rand gap |
|---|---|---|---|
| default | +251.23 | +320.94 | +69.71 |
| minimal_specialization | -87.72 | -22.07 | +65.65 |
| positional_default | +250.73 | +320.89 | +70.16 |

## Team reward (trailing-5 mean of `mean_step_reward_team`)

| scenario | arm | n | mean ± std | per-seed |
|---|---|---|---|---|
| default | ippo | 3 | +251.41 ± 6.31 | ['+254.58', '+244.14', '+255.50'] |
| default | mappo | 3 | +252.74 ± 3.70 | ['+255.61', '+248.57', '+254.04'] |
| minimal_specialization | ippo | 3 | -82.63 ± 7.04 | ['-76.51', '-90.33', '-81.04'] |
| minimal_specialization | mappo | 3 | -86.99 ± 2.02 | ['-89.12', '-86.74', '-85.12'] |
| positional_default | ippo | 3 | +251.60 ± 4.79 | ['+253.70', '+246.12', '+254.97'] |
| positional_default | mappo | 3 | +249.39 ± 9.61 | ['+258.87', '+239.66', '+249.64'] |

## Policy divergence (final-iter entropy spread + pairwise action KL)

| scenario | arm | mean_entropy | entropy_spread | KL off-diag mean |
|---|---|---|---|---|
| default | ippo | 0.001 | 0.002 | 21.6452 |
| default | mappo | 1.874 | 1.871 | 7.4051 |
| minimal_specialization | ippo | 0.067 | 0.224 | 21.0646 |
| minimal_specialization | mappo | 2.319 | 2.538 | 5.6038 |
| positional_default | ippo | 0.000 | 0.000 | 21.7437 |
| positional_default | mappo | 1.535 | 1.924 | 9.7635 |

## Gap-closed by scenario (per-arm and delta)

| scenario | gap_ippo | gap_mappo | delta (mappo−ippo) | per-scenario tier |
|---|---|---|---|---|
| default | +0.003 | +0.022 | +0.019 | `tier_3_insufficient` |
| minimal_specialization | +0.078 | +0.011 | -0.066 | `tier_3_insufficient` |
| positional_default | +0.012 | -0.019 | -0.031 | `tier_3_insufficient` |

## Headline verdict

**`INSUFFICIENT`**

Tier counts: tier_1_succeeds=0, tier_2_partial=0, tier_3_insufficient=3.

