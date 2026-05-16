# Nash Equilibrium Computation Benchmarks

## Provenance Notes

**2026-05-16 (issue #240)**: The Rust-backed payoff evaluator
(`bucket_brigade.equilibrium.payoff_evaluator_rust`) and its Rust
counterpart `HeuristicAgent::select_action` hardcoded `signal == mode`
in the action vector, regardless of the agent's `honesty_bias`
parameter. Post-#236 (signaling first-class), this silently disagreed
with `HeuristicAgent.act`, which honors `honesty_bias`. The Liar
archetype (`honesty_bias = 0.1`) was therefore evaluated as honest
through every Rust path, biasing equilibrium results that include the
Liar archetype or any strategy with `honesty_bias < 1.0`. The Nov 4 2025
benchmarks below were computed before this fix. A post-#240 re-run is
underway (see `experiments/nash/v1_results_python_post240/`); table
will be updated when complete.

**2025-11-04 (original)**: Initial 12-scenario benchmark on an AWS
c5.16xlarge with the pre-#236 deterministic signaling semantics
(`signal == work bit` enforced by the engine itself). Numbers below
predate both #236 and #240, so the Liar archetype's strength is
understated and dominant-strategy verdicts should not be relied upon
without the post-#240 re-derivation.

## Hardware Configuration

**Machine**: SkyPilot rwalters-sandbox-1
- **CPU**: 64 vCPUs (AWS c5.16xlarge or similar)
- **Memory**: ~128 GB RAM
- **Region**: US-based AWS region
- **OS**: Ubuntu 24.04 LTS

## Computation Settings

**Algorithm**: Double Oracle with Rust-accelerated payoff evaluation
- **Simulations per evaluation**: 200 Monte Carlo rollouts
- **Optimization method**: Local (L-BFGS-B)
- **Max iterations**: 50 (Double Oracle algorithm)
- **Convergence threshold**: ε = 0.01

## Timing Results (November 4, 2025, pre-#240)

### Overall Performance

- **Total scenarios**: 12
- **Wall-clock time**: 47 minutes (parallel execution)
- **Total CPU time**: ~135 minutes (sum of all scenarios)
- **Parallelization efficiency**: ~95% (12 scenarios in <3x longest time)

### Per-Scenario Timings

| Scenario             | Time (seconds) | Time (minutes) | Iterations | Equilibrium Type |
|---------------------|----------------|----------------|------------|------------------|
| hard                | 222.2          | 3.7            | 2          | Pure             |
| default             | 256.7          | 4.3            | 2          | Pure             |
| easy                | 252.1          | 4.2            | 2          | Pure             |
| overcrowding        | 407.9          | 6.8            | 1          | Pure             |
| rest_trap           | 407.7          | 6.8            | 1          | Pure             |
| mixed_motivation    | 434.8          | 7.2            | 1          | Pure             |
| deceptive_calm      | 436.4          | 7.3            | 1          | Pure             |
| sparse_heroics      | 457.1          | 7.6            | 1          | Pure             |
| greedy_neighbor     | 641.4          | 10.7           | 2          | Pure             |
| chain_reaction      | 750.9          | 12.5           | 2          | Pure             |
| early_containment   | 1646.9         | 27.4           | 3          | Mixed            |
| trivial_cooperation | 2189.5         | 36.5           | 3          | Mixed            |

**Summary Statistics**:
- **Mean**: 675.3 seconds (11.3 minutes)
- **Median**: 445.6 seconds (7.4 minutes)
- **Min**: 222.2 seconds (3.7 minutes) - hard
- **Max**: 2189.5 seconds (36.5 minutes) - trivial_cooperation
- **Std Dev**: 587.7 seconds

### Observations

1. **Pure equilibria converge faster** (mean: 416s) than mixed equilibria (mean: 1918s)
2. **Iteration count correlates with time**: 1-iteration scenarios average 435s, 2-iteration average 437s, 3-iteration average 1918s
3. **Mixed equilibria take 4-5x longer** due to more complex strategy spaces
4. **CPU utilization**: ~40 parallel worker processes per scenario (Rust parallelization)

## Performance Comparison

### Before Optimizations (Estimated)
- **Simulations**: 2000 per evaluation (10x more)
- **Optimization**: Global differential evolution (10-100x slower)
- **Estimated time per scenario**: 4-8 hours
- **Estimated total (sequential)**: 48-96 hours
- **Estimated total (parallel, 12 cores)**: 48-96 hours (CPU bound)

### After Optimizations (Actual)
- **Simulations**: 200 per evaluation
- **Optimization**: Local L-BFGS-B
- **Actual time per scenario**: 3.7-36.5 minutes
- **Actual total (parallel)**: 47 minutes
- **Speedup**: **60-120x faster**

## Resource Usage

- **Peak CPU usage**: 12 main Python processes + ~480 Rust worker processes
- **Memory per scenario**: ~67-70 MB per main process
- **Disk I/O**: Minimal (only writing final JSON results)
- **Network**: None (local computation)

## Recommendations for Future Computations

### For Similar Scenarios (12 scenarios)
- **64 vCPU machine**: ~45-60 minutes
- **32 vCPU machine**: ~60-90 minutes (some serialization)
- **16 vCPU machine**: ~90-150 minutes (more serialization)

### For Larger Workloads
- **24 scenarios**: ~75-90 minutes on 64 vCPU (2x parallelization)
- **50 scenarios**: ~150-180 minutes on 64 vCPU
- **100 scenarios**: ~300-360 minutes on 64 vCPU

### Cost Estimation (AWS c5.16xlarge @ $2.72/hour)
- **12 scenarios**: ~$2.13 (47 minutes)
- **24 scenarios**: ~$3.40 (75 minutes)
- **100 scenarios**: ~$13.60 (300 minutes)

## Optimization Trade-offs

### Simulations: 200 vs 2000
- **Accuracy loss**: Minimal for Nash computation (±2-3% payoff variance)
- **Speedup**: 10x
- **Recommendation**: 200 is sufficient for Nash equilibria

### Optimization Method: Local vs Global
- **Risk**: May miss global optimum in complex landscapes
- **Mitigation**: Double Oracle iteratively explores strategy space
- **Observed**: 100% convergence rate across all scenarios
- **Speedup**: 10-100x
- **Recommendation**: Local optimization acceptable for Nash computation

## Future Work

1. **Adaptive simulation count**: Use fewer simulations early in Double Oracle, more for final verification
2. **Warm-start strategies**: Initialize with heuristic strategies to reduce iterations
3. **GPU acceleration**: Port Monte Carlo simulation to CUDA for 10-100x speedup
4. **Distributed computation**: Use Ray for multi-node scaling to 100+ scenarios

---

## Post-#240 Re-Derivation (in progress)

A canonical re-derivation of all 12 v1 scenarios on `COMPUTE_HOST_PRIMARY`
(Mac Studio, M-series; ~16 performance cores) was launched as part of
issue #240 alongside the `honesty_bias`-signaling fix. Results land in
`experiments/nash/v1_results_python_post240/{scenario}/equilibrium.json`.

**Why a re-run**: with the fix, Liar (`honesty_bias=0.1`) now broadcasts
deceptive signals through every evaluator path (Rust full-rust,
Rust+Python helper, pure-Python). Equilibria in scenarios where
signal-conditioned strategies could exploit deception may shift; in
particular, mixed equilibria containing Liar should be re-evaluated.

**Expected verdict diff** (to be filled in when the run completes):

| Scenario | Pre-#240 equilibrium | Post-#240 equilibrium | Liar mass change |
|----------|----------------------|------------------------|------------------|
| chain_reaction      | Pure Free_Rider (2.94)       | (pending) | (pending) |
| greedy_neighbor     | Pure Coordinator (58.77)      | (pending) | (pending) |
| trivial_cooperation | Mixed (Free_Rider, 108.78)    | (pending) | (pending) |
| (9 more)            | (see v1_results_python README) | (pending) | (pending) |

This section will be updated by a follow-up commit once the remote
re-derivation finishes.

---

*Benchmarks collected on rwalters-sandbox-1 (64 vCPU) on November 4, 2025*
*Post-#240 re-derivation on robbs-mac-studio (Mac Studio M-series), 2026-05-16*
*For issues #85, #240: Compute Nash Equilibria for All Scenarios*
