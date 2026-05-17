# Issue #260 — episode-length curriculum on `minimal_specialization`

Pre-registered references (per-step mean team reward) on `minimal_specialization`:

- random: `-92.921`
- specialist: `-22.072`
- spec-rand gap: `+70.850`

## Team reward (trailing-5 mean of `mean_step_reward_team`)

| arm | n | mean ± std | per-seed |
|---|---|---|---|
| baseline | 3 | -87.615 ± 11.842 | ['-81.409', '-101.270', '-80.167'] |
| curriculum | 3 | -87.792 ± 6.944 | ['-84.198', '-95.797', '-83.382'] |

## Curriculum floors observed

| arm | seed | floors_unique | first | last |
|---|---|---|---|---|
| baseline | 0 | [12] | 12 | 12 |
| baseline | 1 | [-1] | -1 | -1 |
| baseline | 2 | [-1] | -1 | -1 |
| curriculum | 0 | [5, 8, 12] | 5 | 12 |
| curriculum | 1 | [5, 8, 12] | 5 | 12 |
| curriculum | 2 | [5, 8, 12] | 5 | 12 |

## Verdict

- `gap_baseline = +0.075`
- `gap_curriculum = +0.072`
- `delta = -0.002`
- **Tier: `tier_3_curriculum_unhelpful`**
