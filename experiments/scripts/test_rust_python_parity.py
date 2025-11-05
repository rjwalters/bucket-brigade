#!/usr/bin/env python3
"""Test if Rust and Python environments produce identical results for the same agent."""

import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from bucket_brigade.envs import BucketBrigadeEnv
from bucket_brigade.envs.scenarios import get_scenario_by_name
from bucket_brigade.agents.heuristic_agent import HeuristicAgent
from bucket_brigade.evolution.fitness_rust import _run_rust_game, _heuristic_action
from bucket_brigade_core import BucketBrigadeGame

def test_python_env(scenario_name: str, genome: np.ndarray, seed: int = 42) -> float:
    """Run agent in Python environment."""
    scenario = get_scenario_by_name(scenario_name, num_agents=4)
    agents = [HeuristicAgent(params=genome, agent_id=i, name=f"agent-{i}") for i in range(4)]

    env = BucketBrigadeEnv(scenario)
    obs = env.reset(seed=seed)

    total_rewards = np.zeros(4)
    while not env.done:
        actions = np.array([agent.act(obs) for agent in agents])
        obs, rewards, dones, info = env.step(actions)
        total_rewards += rewards

    return float(np.mean(total_rewards))


def test_rust_env(scenario_name: str, genome: np.ndarray, seed: int = 42) -> float:
    """Run agent in Rust environment (via evaluator)."""
    from bucket_brigade.envs.scenarios.definitions import SCENARIOS

    python_scenario = get_scenario_by_name(scenario_name, num_agents=4)

    # Create Rust game
    game = BucketBrigadeGame(SCENARIOS[scenario_name], seed)

    # Run game with all 4 agents using same genome
    rng = np.random.default_rng(seed)
    num_agents = python_scenario.num_agents

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
    # Convert sum to mean for comparison with Python
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

    print(f"Testing Rust vs Python environment parity")
    print(f"Scenario: {scenario_name}")
    print(f"Agent: {version}")
    print()

    # Run 5 games with different seeds
    seeds = [42, 123, 456, 789, 1024]
    python_results = []
    rust_results = []

    for i, seed in enumerate(seeds, 1):
        print(f"Game {i}/5 (seed={seed})...", end=" ")

        py_score = test_python_env(scenario_name, genome, seed)
        rust_score = test_rust_env(scenario_name, genome, seed)

        python_results.append(py_score)
        rust_results.append(rust_score)

        diff = abs(py_score - rust_score)
        match = "✓" if diff < 0.01 else "✗"
        print(f"Python: {py_score:7.2f}, Rust: {rust_score:7.2f}, Diff: {diff:7.2f} {match}")

    print()
    print("Summary:")
    print(f"  Python mean: {np.mean(python_results):.2f} ± {np.std(python_results):.2f}")
    print(f"  Rust mean:   {np.mean(rust_results):.2f} ± {np.std(rust_results):.2f}")
    print(f"  Mean diff:   {np.mean([abs(p - r) for p, r in zip(python_results, rust_results)]):.2f}")
    print()

    # Determine if environments match
    mean_diff = np.mean([abs(p - r) for p, r in zip(python_results, rust_results)])
    if mean_diff < 0.1:
        print("✅ PASS: Rust and Python environments produce identical results")
        return 0
    else:
        print("❌ FAIL: Rust and Python environments produce DIFFERENT results")
        print("   This explains why training and tournament metrics don't match!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
