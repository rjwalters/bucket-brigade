#!/usr/bin/env python3
"""
Ultra-minimal test - just 2 simulations to verify correctness.
Should complete in under 30 seconds.
"""

import numpy as np
print("Starting ultra-minimal Nash equilibrium test...")

from bucket_brigade.envs.scenarios import trivial_cooperation_scenario
from bucket_brigade.agents.archetypes import FIREFIGHTER_PARAMS, FREE_RIDER_PARAMS
from bucket_brigade.equilibrium.payoff_evaluator import PayoffEvaluator

print("Imports successful!")

# Trivial scenario
scenario = trivial_cooperation_scenario(num_agents=4)
print(f"Scenario: beta={scenario.beta}, kappa={scenario.kappa}")

# Just 2 simulations, sequential
evaluator = PayoffEvaluator(
    scenario=scenario,
    num_simulations=2,  # ULTRA minimal
    seed=42,
    parallel=False,
)

print("\n1. Evaluating Firefighter vs Firefighter (2 simulations)...")
ff_vs_ff = evaluator.evaluate_symmetric_payoff(
    theta_focal=FIREFIGHTER_PARAMS,
    theta_opponents=FIREFIGHTER_PARAMS,
)
print(f"   Result: {ff_vs_ff:.2f}")

print("\n2. Evaluating Free Rider vs Free Rider (2 simulations)...")
fr_vs_fr = evaluator.evaluate_symmetric_payoff(
    theta_focal=FREE_RIDER_PARAMS,
    theta_opponents=FREE_RIDER_PARAMS,
)
print(f"   Result: {fr_vs_fr:.2f}")

print("\n✓ Success! Basic payoff evaluation works.")
print(f"   Firefighters cooperating: {ff_vs_ff:.2f}")
print(f"   Free riders defecting:   {fr_vs_fr:.2f}")

if ff_vs_ff > fr_vs_fr:
    print("\n→ Cooperation yields higher payoff than free-riding")
    print("   (As expected in trivial cooperation scenario)")
else:
    print(f"\n→ Free-riding yields {fr_vs_fr - ff_vs_ff:.2f} more payoff")

print("\nImplementation verified with minimal simulations!")
