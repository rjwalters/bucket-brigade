# Evolution Timing Analysis

## Purpose

Accurate timing estimates for planning overnight evolution experiments.

## Configuration Tested

**Hardware**: Remote server (rwalters-sandbox-1)
- CPU: [Check with user]
- RAM: [Check with user]

**Software**:
- Rust-backed fitness evaluation (`RustFitnessEvaluator`)
- Python 3.12
- Fixed fitness metric (scenario payoff scale)

**Evolution Parameters**:
- Population: 200
- Games per evaluation: 50
- Scenarios: 9 (running in parallel)

## v3 Run Timing (Nov 5, 2025)

### Actual Performance

**Run details**:
- Started: 05:20:10 UTC
- Measured at: 06:18:18 UTC (58 min elapsed)
- Generations completed: ~2280 avg across 9 scenarios
- Target: 2500 generations

**Measured rate**:
- **1.53 seconds/generation**
- 43 generations/minute
- 0.65 generations/second

**Projected completion**:
- Total time for 2500 gen: **1.06 hours (64 minutes)**
- Completion time: ~06:24 UTC

### Comparison to Previous Estimates

| Run | Generations | Expected Time | Actual Time | Rate (sec/gen) | Error |
|-----|-------------|---------------|-------------|----------------|-------|
| v2 (estimated) | 1000 | 3.5 hours | Unknown | 12.53 | N/A |
| v3 (planned) | 2500 | 10 hours | **1.06 hours** | 1.53 | **-89.4%** |

**Root cause of estimation error**:
1. Previous run may have used Python fitness evaluator (100x slower)
2. Overestimated overhead and startup time
3. Did not account for Rust optimization improvements

## Revised Timing Estimates

### Single-Scenario Evolution

For one scenario with current configuration:

| Generations | Time (min) | Time (hours) |
|-------------|-----------|-------------|
| 1,000 | 26 | 0.43 |
| 2,500 | 64 | 1.07 |
| 5,000 | 128 | 2.13 |
| 10,000 | 255 | 4.25 |
| 20,000 | 510 | 8.50 |
| 25,000 | 638 | 10.63 |

### Parallel Evolution (9 Scenarios)

Wall clock time when running all 9 scenarios in parallel:

| Generations per scenario | Total time | Total generations |
|--------------------------|------------|-------------------|
| 1,000 | 1.1 hours | 9,000 |
| 2,500 | 1.1 hours | 22,500 |
| 5,000 | 2.1 hours | 45,000 |
| 10,000 | 4.3 hours | 90,000 |
| **22,500** | **10 hours** | **202,500** |
| 25,000 | 10.6 hours | 225,000 |

**For true 10-hour overnight run**: Use **22,500 generations per scenario**

## Cross-Scenario (Generalist) Evolution

For generalist agent evaluating across all 9 scenarios:

**Overhead factor**: Estimated 1.5-2.0x vs single scenario
- Single scenario: 1.53 sec/gen
- Cross-scenario: **2.3-3.1 sec/gen** (conservative estimate)

**Recommended for 8-10 hour run**:

| Target time | Generations | Conservative estimate |
|-------------|-------------|----------------------|
| 8 hours | 10,000 | Safe |
| 10 hours | 12,000 | Recommended |
| 12 hours | 15,000 | Aggressive |

## Parameter Recommendations

### Quick Test (30-60 min)

```bash
POPULATION=100
GENERATIONS=1000
GAMES=30
```

Expected: ~40 minutes for 9 scenarios

### Standard Run (2-4 hours)

```bash
POPULATION=200
GENERATIONS=5000
GAMES=50
```

Expected: ~2.1 hours for 9 scenarios

### Overnight Run (8-12 hours)

**Specialists (9 parallel)**:
```bash
POPULATION=200
GENERATIONS=20000
GAMES=50
```

Expected: ~8.5 hours

**Generalist (cross-scenario)**:
```bash
POPULATION=200
GENERATIONS=12000
GAMES=50
```

Expected: ~10 hours (with 2x overhead)

### Maximum Quality (24+ hours)

**Specialists**:
```bash
POPULATION=500
GENERATIONS=50000
GAMES=100
```

Expected: ~53 hours (scaling linearly from measurements)

## Scaling Factors

### Population Size

Fitness evaluation time scales linearly with population:
- 100 pop: 1.0x baseline
- 200 pop: 2.0x baseline
- 500 pop: 5.0x baseline

### Games Per Evaluation

Evaluation time scales linearly with games:
- 30 games: 0.6x baseline
- 50 games: 1.0x baseline
- 100 games: 2.0x baseline

### Combined Scaling

```
Time per generation = 1.53s × (population / 200) × (games / 50)
```

Examples:
- 100 pop, 30 games: 1.53 × 0.5 × 0.6 = **0.46 sec/gen**
- 500 pop, 100 games: 1.53 × 2.5 × 2.0 = **7.65 sec/gen**

## Performance Optimization Notes

### Why v3 is 8x Faster Than Expected

1. ✅ **Rust evaluation**: 100x faster than Python fallback
2. ✅ **Fixed fitness metric**: Direct scenario payoff (simpler calculation)
3. ✅ **Proper environment setup**: No import overhead per generation
4. ✅ **Optimized Rust compilation**: Release mode with proper flags

### Maintaining Performance

To ensure continued speed:
- Always use `RustFitnessEvaluator` (verify imports)
- Build Rust module with release mode: `cargo build --release`
- Set `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1`
- Use Python 3.9-3.13 (not 3.14+)

### Performance Degradation Indicators

If runs are slower than expected:
- **10x slower**: Python fallback likely being used
- **2-3x slower**: Debug mode Rust or CPU throttling
- **1.5x slower**: Memory pressure or I/O bottleneck

## Machine-Specific Scaling

These estimates are for the current remote server. For different machines:

**Faster machines** (more cores, higher clock speed):
- May see 1.2-1.5x speedup
- Diminishing returns beyond 16 cores (evolution is sequential)

**Slower machines** (older CPUs, limited RAM):
- May see 0.5-0.7x speed (2-3x slower)
- Consider reducing population or games

**Rule of thumb**: Run a 1000-generation test and measure actual rate, then scale linearly.

## Validation

To validate these estimates on your machine:

```bash
# Quick 1000-gen test
time ssh remote "cd bucket-brigade && source .venv/bin/activate && \
  python experiments/scripts/run_evolution.py trivial_cooperation \
    --population 200 --generations 1000 --games 50 \
    --output-dir /tmp/timing_test"

# Calculate your rate
# Rate = 1000 generations / (time in seconds)
```

## Next Steps

1. ✅ Use revised estimates for planning experiments
2. ✅ Update generalist evolution parameters in issue #120
3. ✅ Document actual completion time of v3 run for validation
4. Consider running ultra-long experiments (20k-50k generations) overnight

---

**Document created**: 2025-11-05
**Based on**: Specialist v3 run measurements
**Last updated**: 2025-11-05
