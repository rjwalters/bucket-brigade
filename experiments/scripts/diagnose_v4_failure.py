#!/usr/bin/env python3
"""
Diagnose why v4 training fitness doesn't match tournament performance.

This script will:
1. Test the original "evolved" agent in both training and tournament modes
2. Test the v4 agent in both modes
3. Compare the results to identify where the discrepancy occurs
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from bucket_brigade.envs.scenarios import get_scenario_by_name
from bucket_brigade.evolution import FitnessEvaluator, Individual
from bucket_brigade.evolution.fitness_rust import _heuristic_action, _convert_scenario_to_rust
import bucket_brigade_core as core


def test_tournament_mode(scenario_name: str, genome: np.ndarray, num_games: int = 10) -> dict:
    """Test agent in tournament mode (Rust environment).

    This ensures train/test consistency - both use the same Rust implementation.
    """
    # Get scenario and convert to Rust
    python_scenario = get_scenario_by_name(scenario_name, num_agents=4)
    rust_scenario = _convert_scenario_to_rust(python_scenario)
    num_agents = python_scenario.num_agents

    payoffs = []
    for game_idx in range(num_games):
        # Create Rust game
        game = core.BucketBrigade(rust_scenario, seed=game_idx)

        # Python RNG for heuristic decisions
        rng = np.random.RandomState(game_idx)

        # Run until done
        done = False
        step_count = 0
        max_steps = 100

        while not done and step_count < max_steps:
            # Get actions for all 4 agents using same genome
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

            # Step the game
            rewards, done, info = game.step(actions)
            step_count += 1

        # Get final result
        result = game.get_result()
        # Use mean of agent scores for consistency with training
        payoffs.append(result.final_score / num_agents)

    return {
        "mean": float(np.mean(payoffs)),
        "std": float(np.std(payoffs)),
        "min": float(np.min(payoffs)),
        "max": float(np.max(payoffs)),
        "raw_payoffs": payoffs,
    }


def test_training_mode(scenario_name: str, genome: np.ndarray, num_games: int = 10) -> dict:
    """Test agent in training mode (Rust evaluator)."""
    scenario = get_scenario_by_name(scenario_name, num_agents=4)

    evaluator = FitnessEvaluator(
        scenario=scenario,
        games_per_individual=num_games,
        seed=42,
        parallel=False,  # Disable parallel for easier debugging
    )

    # Create Individual object for evaluation
    individual = Individual(genome=genome, generation=0)

    # Evaluate
    fitness = evaluator.evaluate_individual(individual)

    # Note: fitness is MEAN of agent scores (same as tournament now)
    return {
        "fitness_mean": float(fitness),
        "note": "Training evaluator returns mean(agent_scores) across games",
    }


def diagnose_agent(scenario_name: str, version: str, num_games: int = 20):
    """Full diagnostic for one agent."""
    print(f"\n{'='*80}")
    print(f"Diagnosing: {scenario_name} / {version}")
    print(f"{'='*80}\n")

    # Load agent
    agent_file = Path(f"experiments/scenarios/{scenario_name}/{version}/best_agent.json")
    if not agent_file.exists():
        print(f"❌ Agent file not found: {agent_file}")
        return None

    with open(agent_file) as f:
        agent_data = json.load(f)
        genome = np.array(list(agent_data["parameters"].values()))
        stored_fitness = agent_data.get("fitness", "N/A")

    print(f"Agent parameters:")
    for param, value in agent_data["parameters"].items():
        print(f"  {param:20s}: {value:.4f}")
    print()

    # Test in training mode
    print("Testing in TRAINING mode (Rust evaluator)...")
    training_result = test_training_mode(scenario_name, genome, num_games)
    print(f"  Fitness (mean): {training_result['fitness_mean']:.2f}")
    print(f"  Stored fitness: {stored_fitness}")
    print()

    # Test in tournament mode
    print("Testing in TOURNAMENT mode (Rust environment)...")
    tournament_result = test_tournament_mode(scenario_name, genome, num_games)
    print(f"  Mean payoff: {tournament_result['mean']:.2f} ± {tournament_result['std']:.2f}")
    print(f"  Range: [{tournament_result['min']:.2f}, {tournament_result['max']:.2f}]")
    print()

    # Compare
    diff = abs(training_result['fitness_mean'] - tournament_result['mean'])
    print(f"COMPARISON:")
    print(f"  Training (mean):   {training_result['fitness_mean']:8.2f}")
    print(f"  Tournament (mean): {tournament_result['mean']:8.2f}")
    print(f"  Difference:        {diff:8.2f}")

    if diff < 1.0:
        print(f"  ✅ MATCH - Environments produce similar results")
    else:
        print(f"  ❌ MISMATCH - {diff:.1f} point discrepancy!")
        print(f"     This indicates training and tournament use different logic!")

    return {
        "version": version,
        "training": training_result,
        "tournament": tournament_result,
        "difference": diff,
        "parameters": agent_data["parameters"],
    }


def main():
    parser = argparse.ArgumentParser(description="Diagnose v4 training/tournament mismatch")
    parser.add_argument("scenario", help="Scenario name (e.g., chain_reaction)")
    parser.add_argument("--num-games", type=int, default=20, help="Games per test")
    args = parser.parse_args()

    results = {}

    # Test original "evolved" agent
    print("\n" + "="*80)
    print("BASELINE: Testing original 'evolved' agent")
    print("="*80)
    results["evolved"] = diagnose_agent(args.scenario, "evolved", args.num_games)

    # Test v4 agent
    print("\n" + "="*80)
    print("V4: Testing 'evolved_v4' agent")
    print("="*80)
    results["evolved_v4"] = diagnose_agent(args.scenario, "evolved_v4", args.num_games)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    for version, result in results.items():
        if result:
            print(f"\n{version}:")
            print(f"  Training:   {result['training']['fitness_mean']:8.2f}")
            print(f"  Tournament: {result['tournament']['mean']:8.2f}")
            print(f"  Difference: {result['difference']:8.2f}")
            print(f"  Status: {'✅ MATCH' if result['difference'] < 1.0 else '❌ MISMATCH'}")

    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)

    if results["evolved"] and results["evolved_v4"]:
        evolved_match = results["evolved"]["difference"] < 1.0
        v4_match = results["evolved_v4"]["difference"] < 1.0

        if evolved_match and not v4_match:
            print("✅ Original 'evolved' agents: Training matches tournament")
            print("❌ V4 agents: Training DOES NOT match tournament")
            print()
            print("This suggests the v4 fix is not working correctly!")
            print("Possible causes:")
            print("  1. Fix not actually applied during v4 training")
            print("  2. Different bug introduced in v4")
            print("  3. Training and tournament using different configurations")
        elif not evolved_match and not v4_match:
            print("❌ BOTH agents show train/test mismatch")
            print()
            print("This suggests a systematic difference in configuration between")
            print("training and tournament evaluation.")
        elif evolved_match and v4_match:
            print("✅ BOTH agents match training and tournament")
            print()
            print("The environments are consistent (both use Rust)!")
            print("V4 failure must be due to poor evolution, not evaluation bug.")
        else:
            print("⚠️  Unexpected pattern - evolved mismatch but v4 match")

    # Save detailed results
    output_file = Path(f"experiments/scenarios/{args.scenario}/v4_diagnostic_results.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to: {output_file}")


if __name__ == "__main__":
    main()
