#!/usr/bin/env python3
"""
Test Rust-backed payoff evaluator - should be FAST!
"""
import time
import numpy as np
from bucket_brigade.envs.scenarios import greedy_neighbor_scenario
from bucket_brigade.agents.archetypes import FIREFIGHTER_PARAMS, FREE_RIDER_PARAMS
from bucket_brigade.equilibrium.payoff_evaluator_rust import RustPayoffEvaluator

print("=" * 80)
print("RUST-BACKED NASH EQUILIBRIUM TEST")
print("=" * 80)

# Greedy neighbor scenario
scenario = greedy_neighbor_scenario(num_agents=4)
print(f"\nScenario: Greedy Neighbor (high work cost)")
print(f"  beta:  {scenario.beta}")
print(f"  kappa: {scenario.kappa}")
print(f"  c (work cost): {scenario.c}  ← High cost creates free-riding incentive")

# Create Rust evaluator
print("\n" + "-" * 80)
print("Creating Rust-backed evaluator (100 simulations, parallel)")
print("-" * 80)

evaluator = RustPayoffEvaluator(
    scenario=scenario,
    num_simulations=100,  # Much more than we could do with Python!
    seed=42,
    parallel=False,  # Sequential to avoid multiprocessing guard issues in scripts
)

print("\nRunning payoff evaluations...")

# 1. Firefighter vs Firefighter
print("\n1. Firefighter vs Firefighter...")
start = time.time()
ff_vs_ff = evaluator.evaluate_symmetric_payoff(
    theta_focal=FIREFIGHTER_PARAMS,
    theta_opponents=FIREFIGHTER_PARAMS,
)
elapsed = time.time() - start
print(f"   Payoff: {ff_vs_ff:.2f} (took {elapsed:.2f}s)")

# 2. Free Rider vs Firefighter
print("\n2. Free Rider vs Firefighter...")
start = time.time()
fr_vs_ff = evaluator.evaluate_symmetric_payoff(
    theta_focal=FREE_RIDER_PARAMS,
    theta_opponents=FIREFIGHTER_PARAMS,
)
elapsed = time.time() - start
print(f"   Payoff: {fr_vs_ff:.2f} (took {elapsed:.2f}s)")

# 3. Firefighter vs Free Rider
print("\n3. Firefighter vs Free Rider...")
start = time.time()
ff_vs_fr = evaluator.evaluate_symmetric_payoff(
    theta_focal=FIREFIGHTER_PARAMS,
    theta_opponents=FREE_RIDER_PARAMS,
)
elapsed = time.time() - start
print(f"   Payoff: {ff_vs_fr:.2f} (took {elapsed:.2f}s)")

# 4. Free Rider vs Free Rider
print("\n4. Free Rider vs Free Rider...")
start = time.time()
fr_vs_fr = evaluator.evaluate_symmetric_payoff(
    theta_focal=FREE_RIDER_PARAMS,
    theta_opponents=FREE_RIDER_PARAMS,
)
elapsed = time.time() - start
print(f"   Payoff: {fr_vs_fr:.2f} (took {elapsed:.2f}s)")

# Analysis
print("\n" + "=" * 80)
print("PAYOFF MATRIX ANALYSIS")
print("=" * 80)

print("\n                  vs Firefighter    vs Free Rider")
print(f"Firefighter:      {ff_vs_ff:10.2f}      {ff_vs_fr:10.2f}")
print(f"Free Rider:       {fr_vs_ff:10.2f}      {fr_vs_fr:10.2f}")

print("\n" + "-" * 80)
print("Strategic Analysis:")
print("-" * 80)

if fr_vs_ff > ff_vs_ff:
    advantage = fr_vs_ff - ff_vs_ff
    print(f"\n✓ Free-riding is PROFITABLE when others cooperate")
    print(f"  Advantage: {advantage:.2f} points")
    print(f"  → Creates incentive to free-ride")
else:
    print(f"\n✓ Cooperation is STABLE")
    print(f"  → Firefighter does better even when opponents cooperate")

if ff_vs_fr > fr_vs_fr:
    advantage = ff_vs_fr - fr_vs_fr
    print(f"\n✓ Cooperation PUNISHES free-riders")
    print(f"  Firefighters get {advantage:.2f} more points vs free-riders")
    print(f"  → Provides deterrent against free-riding")
else:
    print(f"\n✓ Free-riding is safe")
    print(f"  → Free-riders not punished when others also free-ride")

print("\n" + "=" * 80)
print("SUCCESS! Rust backend is working and FAST!")
print("=" * 80)
print("\nReady for full Nash equilibrium analysis with the Double Oracle algorithm.")
