# Issue #220 — Obs-fix paired comparison

Random/specialist references on `minimal_specialization` (from 2026-05-15 notebook): random = -96.07, specialist = -22.07. Pass bar = treatment closes ≥ 50%.

## Team reward (trailing-5 mean of `mean_step_reward_team`)

| scenario | arm | n | mean ± std | per-seed |
|---|---|---|---|---|
| default | baseline | 0 | — | missing |
| default | treatment | 3 | +251.41 ± 6.31 | [254.576171875, 244.13662109375, 255.50390625] |
| minimal_specialization | baseline | 0 | — | missing |
| minimal_specialization | treatment | 3 | -82.63 ± 7.04 | [-76.5126953125, -90.32919921875, -81.04072265625] |

## Policy divergence (final-iter entropy spread + pairwise action KL)

| scenario | arm | mean_entropy | entropy_spread (max−min across agents) | KL off-diag mean |
|---|---|---|---|---|
| default | treatment | 0.001 | 0.002 | 21.6452 |
| minimal_specialization | treatment | 0.067 | 0.224 | 21.0646 |

## Gap-closed on `minimal_specialization`

| arm | gap_closed (trailing-5 mean) | per-seed | pre-reg verdict |
|---|---|---|---|
| baseline | — | missing | — |
| treatment | 0.182 | ['0.264', '0.078', '0.203'] | < 25% — MAPPO needed |
