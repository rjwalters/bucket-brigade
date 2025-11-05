#!/usr/bin/env python3
"""Test Rust environment consistency for the same agent.

This script now tests only Rust implementation to verify consistency across runs.
Python environment is deprecated - Rust is the single source of truth.
"""

import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from bucket_brigade.envs.scenarios import get_scenario_by_name
from bucket_brigade.evolution.fitness_rust import _heuristic_action, _convert_scenario_to_rust
import bucket_brigade_core as core

def test_rust_env(scenario_name: str, genome: np.ndarray, seed: int = 42) -> float:
    """Run agent in Rust environment."""
    python_scenario = get_scenario_by_name(scenario_name, num_agents=4)
    rust_scenario = _convert_scenario_to_rust(python_scenario)
    num_agents = python_scenario.num_agents

    # Create Rust game
    game = core.BucketBrigade(rust_scenario, seed=seed)

    # Python RNG for heuristic decisions
    rng = np.random.RandomState(seed)

    # Run game with all 4 agents using same genome
    done = False
    step_count = 0
    max_steps = 100

    while not done and step_count < max_steps:
        actions = []
        for agent_id in range(num_agents):
            obs = game.get_observation(agent_id)
            obs_dict = {
                "houses": obs.houses,
                "signals": obs.signals,
                "locations": obs.locations,
            }
            action = _heuristic_action(genome, obs_dict, agent_id, rng)
            actions.append(action)

        rewards, done, info = game.step(actions)
        step_count += 1

    result = game.get_result()
    # Return mean for consistency
    return result.final_score / num_agents


def main():
    if len(sys.argv) < 3:
        print("Usage: python test_rust_python_parity.py <scenario> <evolution_version>")
        print("Example: python test_rust_python_parity.py chain_reaction evolved")
        sys.exit(1)

    scenario_name = sys.argv[1]
    version = sys.argv[2]

    # Load agent genome
    agent_file = Path(f"experiments/scenarios/{scenario_name}/{version}/best_agent.json")
    if not agent_file.exists():
        print(f"Error: Agent file not found: {agent_file}")
        sys.exit(1)

    with open(agent_file) as f:
        agent_data = json.load(f)
        genome = np.array(list(agent_data["parameters"].values()))

    print(f"Testing Rust environment consistency")
    print(f"Scenario: {scenario_name}")
    print(f"Agent: {version}")
    print()

    # Run 5 games with different seeds, twice each to verify determinism
    seeds = [42, 123, 456, 789, 1024]
    first_run = []
    second_run = []

    for i, seed in enumerate(seeds, 1):
        print(f"Game {i}/5 (seed={seed})...", end=" ")

        # Run twice with same seed to verify determinism
        score1 = test_rust_env(scenario_name, genome, seed)
        score2 = test_rust_env(scenario_name, genome, seed)

        first_run.append(score1)
        second_run.append(score2)

        diff = abs(score1 - score2)
        match = "✓" if diff < 0.01 else "✗"
        print(f"Run 1: {score1:7.2f}, Run 2: {score2:7.2f}, Diff: {diff:7.2f} {match}")

    print()
    print("Summary:")
    print(f"  Run 1 mean: {np.mean(first_run):.2f} ± {np.std(first_run):.2f}")
    print(f"  Run 2 mean: {np.mean(second_run):.2f} ± {np.std(second_run):.2f}")
    print(f"  Mean diff:  {np.mean([abs(r1 - r2) for r1, r2 in zip(first_run, second_run)]):.2f}")
    print()

    # Determine if Rust is deterministic
    mean_diff = np.mean([abs(r1 - r2) for r1, r2 in zip(first_run, second_run)])
    if mean_diff < 0.01:
        print("✅ PASS: Rust environment is deterministic (same seed → same result)")
        return 0
    else:
        print("❌ FAIL: Rust environment is non-deterministic")
        print("   This indicates a bug in the Rust implementation!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
