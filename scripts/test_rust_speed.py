#!/usr/bin/env python3
"""
Quick test to show Rust core speed.
"""
import time
import bucket_brigade_core as core

print("Testing Rust core speed...")
print(f"Available scenarios: {len(core.SCENARIOS)}")

# Get a scenario
scenario_name = list(core.SCENARIOS.keys())[0]
scenario = core.SCENARIOS[scenario_name]
print(f"Using scenario: {scenario_name}")

# Create game
game = core.BucketBrigade(scenario)

# Run 100 simulations with random actions
num_sims = 100
print(f"\nRunning {num_sims} full game simulations with Rust core...")

start = time.time()
for i in range(num_sims):
    game = core.BucketBrigade(scenario)

    while not game.is_done():
        # Random actions for all agents
        actions = [[game.rng.gen_range(0, 10), game.rng.gen_range(0, 2)]
                   for _ in range(scenario.num_agents)]
        game.step(actions)

    if (i + 1) % 10 == 0:
        elapsed = time.time() - start
        rate = (i + 1) / elapsed
        print(f"  Completed {i+1}/{num_sims} simulations ({rate:.1f} sims/sec)")

elapsed = time.time() - start
print(f"\n✓ Completed {num_sims} simulations in {elapsed:.2f} seconds")
print(f"  Rate: {num_sims/elapsed:.1f} simulations/second")
print(f"  Time per simulation: {elapsed/num_sims*1000:.1f} ms")

print("\nThis is the speed we should be getting with Rust!")
