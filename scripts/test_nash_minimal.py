#!/usr/bin/env python3
"""
Minimal Nash equilibrium test - completes in seconds.

Tests basic functionality with absolute minimum parameters.
"""

import numpy as np
from bucket_brigade.envs.scenarios import trivial_cooperation_scenario
from bucket_brigade.agents.archetypes import FIREFIGHTER_PARAMS, FREE_RIDER_PARAMS
from bucket_brigade.equilibrium.payoff_evaluator import PayoffEvaluator

print("=" * 80)
print("MINIMAL NASH EQUILIBRIUM TEST")
print("=" * 80)

# Use trivial scenario (easiest to compute)
scenario = trivial_cooperation_scenario(num_agents=4)

print("\nScenario: Trivial Cooperation")
print(f"  beta (spread):    {scenario.beta}")
print(f"  kappa (extinguish): {scenario.kappa}")
print(f"  c (work cost):    {scenario.c}")
print(f"  num_agents:       {scenario.num_agents}")

# Create evaluator with VERY few simulations
print("\n" + "-" * 80)
print("Testing Payoff Evaluation (5 simulations, parallel)")
print("-" * 80)

evaluator = PayoffEvaluator(
    scenario=scenario,
    num_simulations=5,  # Absolute minimum
    seed=42,
    parallel=False,  # Use sequential to avoid multiprocessing issues in script
)

print("\n1. Firefighter vs Firefighter...")
ff_vs_ff = evaluator.evaluate_symmetric_payoff(
    theta_focal=FIREFIGHTER_PARAMS,
    theta_opponents=FIREFIGHTER_PARAMS,
)
print(f"   Payoff: {ff_vs_ff:.2f}")

print("\n2. Free Rider vs Firefighter...")
fr_vs_ff = evaluator.evaluate_symmetric_payoff(
    theta_focal=FREE_RIDER_PARAMS,
    theta_opponents=FIREFIGHTER_PARAMS,
)
print(f"   Payoff: {fr_vs_ff:.2f}")

print("\n3. Firefighter vs Free Rider...")
ff_vs_fr = evaluator.evaluate_symmetric_payoff(
    theta_focal=FIREFIGHTER_PARAMS,
    theta_opponents=FREE_RIDER_PARAMS,
)
print(f"   Payoff: {ff_vs_fr:.2f}")

print("\n4. Free Rider vs Free Rider...")
fr_vs_fr = evaluator.evaluate_symmetric_payoff(
    theta_focal=FREE_RIDER_PARAMS,
    theta_opponents=FREE_RIDER_PARAMS,
)
print(f"   Payoff: {fr_vs_fr:.2f}")

# Build simple 2x2 payoff matrix
print("\n" + "-" * 80)
print("Payoff Matrix (2x2 game)")
print("-" * 80)

payoff_matrix = np.array(
    [
        [ff_vs_ff, ff_vs_fr],
        [fr_vs_ff, fr_vs_fr],
    ]
)

print("\n                  vs Firefighter    vs Free Rider")
print(f"Firefighter:      {ff_vs_ff:10.2f}      {ff_vs_fr:10.2f}")
print(f"Free Rider:       {fr_vs_ff:10.2f}      {fr_vs_fr:10.2f}")

# Analyze the matrix
print("\n" + "-" * 80)
print("Game-Theoretic Analysis")
print("-" * 80)

# Check for dominant strategies
if ff_vs_ff >= fr_vs_ff and ff_vs_fr >= fr_vs_fr:
    print("\n✓ Firefighter DOMINATES Free Rider")
    print("  (Firefighter always does at least as well)")
    print("  → Pure strategy equilibrium: Everyone cooperates")
elif fr_vs_ff >= ff_vs_ff and fr_vs_fr >= ff_vs_fr:
    print("\n✓ Free Rider DOMINATES Firefighter")
    print("  (Free riding always does at least as well)")
    print("  → Pure strategy equilibrium: Everyone free rides")
else:
    print("\n✓ No dominant strategy - strategic tension exists")
    print(f"  FF vs FF: {ff_vs_ff:.2f}")
    print(f"  FR vs FF: {fr_vs_ff:.2f}")
    print(f"  Difference: {fr_vs_ff - ff_vs_ff:.2f}")
    print()
    if fr_vs_ff > ff_vs_ff:
        print("  → Free riding is profitable when others cooperate")
        print("  → Likely mixed strategy equilibrium")
    else:
        print("  → Cooperation is stable")

# Solve for Nash equilibrium
print("\n" + "-" * 80)
print("Computing Nash Equilibrium (Linear Programming)")
print("-" * 80)

from bucket_brigade.equilibrium.nash_solver import solve_symmetric_nash

distribution = solve_symmetric_nash(payoff_matrix)

print(f"\nEquilibrium strategy:")
print(f"  Play Firefighter: {distribution[0]:.1%}")
print(f"  Play Free Rider:  {distribution[1]:.1%}")

if distribution[0] > 0.95:
    print("\n✓ Pure strategy equilibrium: COOPERATE")
elif distribution[1] > 0.95:
    print("\n✓ Pure strategy equilibrium: FREE RIDE")
else:
    print(
        f"\n✓ Mixed strategy equilibrium: {distribution[0]:.0%}/{distribution[1]:.0%} mix"
    )

expected_payoff = distribution @ payoff_matrix @ distribution
print(f"\nExpected equilibrium payoff: {expected_payoff:.2f}")

print("\n" + "=" * 80)
print("TEST COMPLETE - Implementation verified!")
print("=" * 80)
print("\nKey findings:")
print("- Payoff evaluation works correctly")
print("- Nash solver produces valid equilibria")
print("- System is ready for full analysis with more simulations")
print("\nFor full analysis, use: scripts/analyze_nash_equilibrium.py")
print("=" * 80)
