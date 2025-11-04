# Nash Equilibrium Computation Optimizations

## Issue #85: Compute Nash Equilibria for All Scenarios

This document summarizes the optimization work performed to make Nash equilibrium computation feasible for all 12 scenarios.

## Initial Problem

When running `compute_nash.py` with the original settings:
- **Simulations**: 2000 Monte Carlo rollouts per payoff evaluation
- **Optimization method**: Global optimization via `differential_evolution`
- **Maxiter**: 50 iterations for differential evolution
- **Time per iteration**: 4+ minutes
- **Estimated total time**: 4-8 hours per scenario × 12 scenarios = **48-96 hours**

This was computationally infeasible on a 48-vCPU sandbox.

## Root Cause Analysis

The bottleneck was in `bucket_brigade/equilibrium/best_response.py::compute_best_response_to_mixture()`:

1. **Differential Evolution**: Each iteration evaluates 10-20 candidate solutions
2. **High maxiter**: 50 iterations × 15 candidates = 750 payoff evaluations per best response
3. **Many simulations**: 2000 simulations × 750 evaluations = **1.5M Monte Carlo episodes** per best response
4. **Multiple iterations**: Double Oracle algorithm requires multiple best responses

Even with Rust-accelerated simulation, this was taking 4+ minutes per Double Oracle iteration.

## Optimization Strategy

### 1. Reduce Simulations (10x speedup)

**Change**: Reduced from 2000 to 200 simulations per payoff evaluation

**Rationale**:
- Nash equilibrium computation is robust to Monte Carlo noise
- 200 simulations still provides reasonable confidence intervals
- Most game-theoretic analysis uses 100-1000 simulations

**Impact**: 10x faster payoff evaluation

### 2. Reduce Differential Evolution Iterations (3x speedup)

**File**: `bucket_brigade/equilibrium/best_response.py`

**Changes**:
```python
# Before
result = differential_evolution(
    objective,
    bounds=bounds,
    seed=seed,
    maxiter=50,  # Too many iterations
    polish=True,
    workers=1,
)

# After
result = differential_evolution(
    objective,
    bounds=bounds,
    seed=seed,
    maxiter=15,  # Reduced from 50
    polish=True,
    workers=1,  # Kept at 1 to avoid pickling issues
    updating='immediate',
    atol=0.01,  # Absolute tolerance
    tol=0.005,  # Relative tolerance
)
```

**Rationale**:
- Added convergence tolerances to stop early when improvement plateaus
- Reduced maxiter from 50 to 15 for faster convergence
- `polish=True` refines solution with L-BFGS-B after DE converges
- `updating='immediate'` improves convergence rate

**Impact**: 3x faster global optimization

### 3. Switch to Local Optimization (10-100x speedup)

**File**: `bucket_brigade/equilibrium/double_oracle.py`

**Changes**:
```python
# Line 196-203
br_strategy, br_payoff = compute_best_response_to_mixture(
    strategy_mixture=eq_dict,
    strategy_pool=strategy_pool,
    scenario=self.scenario,
    num_simulations=self.num_simulations,
    method="local",  # Changed from "global" to "local"
    seed=self.seed + iteration if self.seed is not None else None,
)
```

**Rationale**:
- Local optimization (L-BFGS-B) is 10-100x faster than differential evolution
- Starts from existing strategies in the pool (good initial guesses)
- Nash computation context makes local optima less problematic:
  - Double Oracle iteratively refines solutions
  - Starting from existing best responses in pool
  - If local optimum is suboptimal, next iteration will find better response

**Trade-off**:
- **Global**: More robust but much slower (4+ minutes/iteration)
- **Local**: Faster but may miss global optimum (30-60 seconds/iteration)
- **Decision**: Local is "good enough" for Nash computation given iterative nature

**Impact**: 10-100x faster best response computation

### 4. Rust Module Build Fix

**File**: `scripts/sandbox.sh`

**Problem**: Maturin was building cffi bindings instead of PyO3, causing ImportError

**Solution**:
```bash
# Install Rust if not present
if ! command -v rustc &> /dev/null; then
    echo "🦀 Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
fi

# Build Rust module with PyO3
echo "🔧 Building Rust module (bucket-brigade-core)..."
cd bucket-brigade-core

# Clean any stale build artifacts to avoid cffi/PyO3 conflicts
rm -rf target bucket_brigade_core/__pycache__ bucket_brigade_core/bucket_brigade_core

# Ensure maturin is installed
uv pip install maturin

# Source Rust environment and build with PyO3 feature
source "$HOME/.cargo/env"
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 maturin develop --release --features python

cd ..
```

**Impact**: Ensures fresh sandbox environments build Rust module correctly

## Expected Performance

### Before Optimization
- **Time per scenario**: 4-8 hours
- **Total time**: 48-96 hours
- **Cost**: High (48 vCPUs for 2-4 days)

### After Optimization
- **Time per scenario**: 30-120 minutes (estimated)
- **Total time**: 6-24 hours for all 12 scenarios
- **Cost**: Lower (can use 1-GPU sandbox)
- **Speedup**: ~10-50x overall improvement

## Testing

Tests were launched on rwalters-sandbox-1 but sandbox was terminated before completion. Need to:

1. Re-run test on stable sandbox
2. Verify local optimization produces reasonable Nash equilibria
3. Compare convergence vs global optimization (if time permits)
4. Launch all 12 scenarios in parallel

## Files Modified

1. `bucket_brigade/equilibrium/best_response.py` - Reduced maxiter, added convergence tolerances
2. `bucket_brigade/equilibrium/double_oracle.py` - Switched to local optimization
3. `scripts/sandbox.sh` - Added Rust build setup for fresh environments

## Commits

- `aa3ea95` - Optimize Nash computation: reduce simulations and switch to local optimization
- `29e0a24` - Add Rust build setup to sandbox environment script

## Next Steps

1. Test on stable sandbox environment
2. Launch all 12 scenarios with optimized settings
3. Analyze and compare results
4. Create PR for review
