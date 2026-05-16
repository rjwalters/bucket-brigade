# Issue #220 — Obs-fix paired comparison

Random/specialist references on `minimal_specialization` (from 2026-05-15 notebook): random = -96.07, specialist = -22.07. Pass bar = treatment closes ≥ 50%.

## Team reward (trailing-5 mean of `mean_step_reward_team`)

| scenario | arm | n | mean ± std | per-seed |
|---|---|---|---|---|
| default | baseline | 3 | +249.49 ± 1.40 | [251.10693359375, 248.6869140625, 248.6642578125] |
| default | treatment | 3 | +253.70 ± 2.38 | [256.22998046875, 253.37646484375, 251.508203125] |
| minimal_specialization | baseline | 3 | -86.24 ± 6.21 | [-79.71875, -92.0703125, -86.941015625] |
| minimal_specialization | treatment | 3 | -91.71 ± 7.38 | [-87.34228515625, -100.2322265625, -87.5513671875] |

## Policy divergence (final-iter entropy spread + pairwise action KL)

| scenario | arm | mean_entropy | entropy_spread (max−min across agents) | KL off-diag mean |
|---|---|---|---|---|
| default | baseline | 2.399 | 1.628 | n/a |
| default | treatment | 2.293 | 1.653 | 3.7178 |
| minimal_specialization | baseline | 2.907 | 1.370 | n/a |
| minimal_specialization | treatment | 2.829 | 1.871 | 2.7252 |

## Gap-closed on `minimal_specialization`

| arm | gap_closed (trailing-5 mean) | per-seed | pre-reg verdict |
|---|---|---|---|
| baseline | 0.133 | ['0.221', '0.054', '0.123'] | < 25% — MAPPO needed |
| treatment | 0.059 | ['0.118', '-0.056', '0.115'] | < 25% — MAPPO needed |
