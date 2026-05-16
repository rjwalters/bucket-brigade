# P3 Specialization Plateau Diagnostics (Phase 1)

Per-scenario summary at lambda_red=0 across all seeds.
All values are means across seeds.

## trivial_cooperation
- n_seeds = 20, n_iters = 50
- reward iter 0 -> iter 49: 400.44 -> 400.29  (baseline random = 400.0)
- mean value_loss iter 0 -> iter 49: 3.74e+05 -> 2.53e+05  (scaled by value_coef=0.5: 1.87e+05)
- mean |policy_loss| iter 0 -> iter 49: 1.713e-02 -> 2.225e-02
- mean entropy iter 0 -> iter 49: 6.646e-01 -> 1.743e-01  (scaled by entropy_coef=0.01: 1.743e-03)
- iter 0 dominance: value_term / policy_term = 1.1e+07, value_term / entropy_term = 2.8e+07

## default
- n_seeds = 20, n_iters = 50
- reward iter 0 -> iter 49: 293.46 -> 294.58  (baseline random = 293.4)
- mean value_loss iter 0 -> iter 49: 2.03e+05 -> 1.27e+05  (scaled by value_coef=0.5: 1.01e+05)
- mean |policy_loss| iter 0 -> iter 49: 2.036e-02 -> 2.299e-02
- mean entropy iter 0 -> iter 49: 6.658e-01 -> 1.216e-01  (scaled by entropy_coef=0.01: 1.216e-03)
- iter 0 dominance: value_term / policy_term = 5.0e+06, value_term / entropy_term = 1.5e+07

## chain_reaction
- n_seeds = 20, n_iters = 50
- reward iter 0 -> iter 49: 224.21 -> 224.47  (baseline random = 220.75)
- mean value_loss iter 0 -> iter 49: 1.56e+05 -> 1.09e+05  (scaled by value_coef=0.5: 7.80e+04)
- mean |policy_loss| iter 0 -> iter 49: 2.587e-02 -> 2.096e-02
- mean entropy iter 0 -> iter 49: 6.654e-01 -> 1.394e-01  (scaled by entropy_coef=0.01: 1.394e-03)
- iter 0 dominance: value_term / policy_term = 3.0e+06, value_term / entropy_term = 1.2e+07
