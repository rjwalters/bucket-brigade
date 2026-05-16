# Issue #231 — IPPO vs MAPPO across three scenarios

Pre-registered references (per-step mean team reward):

| scenario | random | specialist | spec-rand gap |
|---|---|---|---|
| default | +241.85 | +320.94 | +79.09 |
| minimal_specialization | -96.07 | -22.07 | +74.00 |
| positional_default | +241.36 | +320.89 | +79.53 |

## Team reward (trailing-5 mean of `mean_step_reward_team`)

| scenario | arm | n | mean ± std | per-seed |
|---|---|---|---|---|
| default | ippo | 3 | +253.70 ± 2.38 | ['+256.23', '+253.38', '+251.51'] |
| default | mappo | 3 | +247.18 ± 11.50 | ['+257.07', '+234.56', '+249.92'] |
| minimal_specialization | ippo | 3 | -91.71 ± 7.38 | ['-87.34', '-100.23', '-87.55'] |
| minimal_specialization | mappo | 3 | -94.52 ± 5.62 | ['-97.19', '-98.30', '-88.06'] |
| positional_default | ippo | 3 | +253.19 ± 4.00 | ['+254.94', '+248.61', '+256.02'] |
| positional_default | mappo | 3 | +251.15 ± 5.04 | ['+256.78', '+247.04', '+249.65'] |

## Policy divergence (final-iter entropy spread + pairwise action KL)

| scenario | arm | mean_entropy | entropy_spread | KL off-diag mean |
|---|---|---|---|---|
| default | ippo | 2.293 | 1.653 | n/a |
| default | mappo | 1.499 | 2.180 | n/a |
| minimal_specialization | ippo | 2.829 | 1.871 | n/a |
| minimal_specialization | mappo | 2.768 | 1.406 | n/a |
| positional_default | ippo | 2.168 | 2.905 | n/a |
| positional_default | mappo | 1.783 | 2.757 | n/a |

## Gap-closed by scenario (per-arm and delta)

| scenario | gap_ippo | gap_mappo | delta (mappo−ippo) | per-scenario tier |
|---|---|---|---|---|
| default | +0.150 | +0.067 | -0.083 | `tier_3_insufficient` |
| minimal_specialization | +0.059 | +0.021 | -0.038 | `tier_3_insufficient` |
| positional_default | +0.149 | +0.123 | -0.026 | `tier_3_insufficient` |

## Headline verdict

**`INSUFFICIENT`**

Tier counts: tier_1_succeeds=0, tier_2_partial=0, tier_3_insufficient=3.
