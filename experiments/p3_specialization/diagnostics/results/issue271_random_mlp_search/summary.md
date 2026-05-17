# Issue #271 — Random-init MLP best-of-N on `minimal_specialization`
Question: does naive best-of-N random search over policy networks beat trained PPO? If yes, the gradient is actively misleading → active anti-attractor.
Invocation: `experiments/p3_specialization/diagnostics/random_mlp_search.py --seeds 1000 --episodes-per-seed 20 --restability-episodes 200 --protocol both --workers 24 --out-dir experiments/p3_specialization/diagnostics/results/issue271_random_mlp_search`

## Reference values

- Uniform-random per-step team reward: **-87.72** (from `random_baseline.SCENARIO_CITED_VALUES['minimal_specialization']`)
- PPO best across 5 tier-3 interventions: **~-75** (gap_closed ≈ 0.18)
- Specialist: **-22.07** (from `issue199_baselines.json`)
- Denominator (specialist − random): **65.65**

## Protocol: `independent`
- n_seeds = 1000, phase-1 episodes/seed = 20, phase-2 episodes/seed = 200
- Population mean = **-90.21** (std = 20.31)
- Percentiles (per-step team reward):
  - p01 = -136.97, p05 = -125.55, p10 = -116.94
  - p50 = -89.27
  - p90 = -64.73, p95 = -56.75, p99 = -47.20
- Top-10 phase-1 mean = **-41.96** (gap_closed = 0.697)
- Top-10 phase-2 (stability-re-eval) mean = **-88.17** (gap_closed = -0.007)
- Best phase-2 seed: 577 → mean = -79.68 (gap_closed = 0.122), 95% CI = [-91.17, -68.52]
- Phase-1 → phase-2 drift (mean): -46.22
- Mean iter-0 action-distribution entropy: 0.790 nats (uniform 10×2×2 ≈ 3.689 nats)

### Verdict — `independent`: `random_play_basin`

Random MLPs ≈ uniform random play. PPO sits in same basin random init lives in.

## Protocol: `shared`
- n_seeds = 1000, phase-1 episodes/seed = 20, phase-2 episodes/seed = 200
- Population mean = **-90.47** (std = 20.34)
- Percentiles (per-step team reward):
  - p01 = -141.17, p05 = -124.30, p10 = -117.36
  - p50 = -90.10
  - p90 = -65.11, p95 = -58.54, p99 = -45.23
- Top-10 phase-1 mean = **-41.05** (gap_closed = 0.711)
- Top-10 phase-2 (stability-re-eval) mean = **-87.39** (gap_closed = 0.005)
- Best phase-2 seed: 612 → mean = -78.15 (gap_closed = 0.146), 95% CI = [-90.03, -66.21]
- Phase-1 → phase-2 drift (mean): -46.34
- Mean iter-0 action-distribution entropy: 0.791 nats (uniform 10×2×2 ≈ 3.689 nats)

### Verdict — `shared`: `random_play_basin`

Random MLPs ≈ uniform random play. PPO sits in same basin random init lives in.

## Verdict table (from issue body, restated)
| Best-of-N outcome (top-1% phase-2 mean) | gap_closed | Interpretation |
| --- | --- | --- |
| ≥ -55 (clearly above PPO ceiling) | ≥ 0.49 | **Anti-attractor confirmed.** PPO is worse than naive search. |
| -75 to -55 | 0.20 to 0.49 | PPO ≈ best-of-N random. Basin trap consistent. |
| -85 to -75 | < 0.20 | Random MLPs ≈ uniform random play; PPO sits in same basin. |
| ≥ -30 (near specialist) | ≥ 0.88 | Stunning — random init lives near specialist. |
