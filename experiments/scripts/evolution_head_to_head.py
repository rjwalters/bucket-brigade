#!/usr/bin/env python3
"""
Head-to-head comparison: evolved agent vs best heuristic.

This script runs direct tournaments between evolved agents and hand-designed
heuristics to measure the improvement from evolutionary optimization.
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import List, Tuple

from bucket_brigade.agents.heuristic_agent import HeuristicAgent
from bucket_brigade.envs.bucket_brigade_env import BucketBrigadeEnv
from bucket_brigade.scenarios import get_scenario_by_name


def run_head_to_head(
    scenario_name: str,
    evolved_genome: List[float],
    heuristic_genome: List[float],
    num_games: int = 100,
    team_sizes: List[int] = None,
) -> dict:
    """
    Run head-to-head tournament between evolved and heuristic agents.

    Args:
        scenario_name: Name of the scenario
        evolved_genome: Parameter vector for evolved agent
        heuristic_genome: Parameter vector for heuristic agent
        num_games: Number of games to play
        team_sizes: List of team sizes to test (default: [4])

    Returns:
        Dictionary with tournament results
    """
    if team_sizes is None:
        team_sizes = [4]

    results = {
        "scenario": scenario_name,
        "num_games": num_games,
        "team_sizes": {},
    }

    for team_size in team_sizes:
        print(f"\n  Team size: {team_size} agents")

        scenario = get_scenario_by_name(scenario_name, num_agents=team_size)

        # Test 1: Homogeneous evolved team
        evolved_payoffs = []
        print(f"    Testing homogeneous evolved team...", end=" ", flush=True)
        for _ in range(num_games):
            agents = [HeuristicAgent(i, evolved_genome) for i in range(team_size)]
            env = BucketBrigadeEnv(scenario=scenario)
            _, _, _, info = env.reset()

            done = False
            while not done:
                actions = {i: agent.select_action(env.get_observation(i)) for i, agent in enumerate(agents)}
                _, _, done, info = env.step(actions)

            evolved_payoffs.append(sum(info["payoffs"]))

        evolved_mean = np.mean(evolved_payoffs)
        evolved_std = np.std(evolved_payoffs)
        print(f"✓ Mean: {evolved_mean:.2f} ± {evolved_std:.2f}")

        # Test 2: Homogeneous heuristic team
        heuristic_payoffs = []
        print(f"    Testing homogeneous heuristic team...", end=" ", flush=True)
        for _ in range(num_games):
            agents = [HeuristicAgent(i, heuristic_genome) for i in range(team_size)]
            env = BucketBrigadeEnv(scenario=scenario)
            _, _, _, info = env.reset()

            done = False
            while not done:
                actions = {i: agent.select_action(env.get_observation(i)) for i, agent in enumerate(agents)}
                _, _, done, info = env.step(actions)

            heuristic_payoffs.append(sum(info["payoffs"]))

        heuristic_mean = np.mean(heuristic_payoffs)
        heuristic_std = np.std(heuristic_payoffs)
        print(f"✓ Mean: {heuristic_mean:.2f} ± {heuristic_std:.2f}")

        # Test 3: Mixed team (1 evolved + rest heuristic)
        mixed_evolved_payoffs = []
        print(f"    Testing mixed team (1 evolved + {team_size-1} heuristic)...", end=" ", flush=True)
        for _ in range(num_games):
            agents = [HeuristicAgent(0, evolved_genome)] + [
                HeuristicAgent(i, heuristic_genome) for i in range(1, team_size)
            ]
            env = BucketBrigadeEnv(scenario=scenario)
            _, _, _, info = env.reset()

            done = False
            while not done:
                actions = {i: agent.select_action(env.get_observation(i)) for i, agent in enumerate(agents)}
                _, _, done, info = env.step(actions)

            mixed_evolved_payoffs.append(sum(info["payoffs"]))

        mixed_mean = np.mean(mixed_evolved_payoffs)
        mixed_std = np.std(mixed_evolved_payoffs)
        print(f"✓ Mean: {mixed_mean:.2f} ± {mixed_std:.2f}")

        # Calculate advantage
        homogeneous_advantage = ((evolved_mean - heuristic_mean) / abs(heuristic_mean)) * 100 if heuristic_mean != 0 else 0
        mixed_advantage = ((mixed_mean - heuristic_mean) / abs(heuristic_mean)) * 100 if heuristic_mean != 0 else 0

        print(f"\n    Evolved Advantage:")
        print(f"      Homogeneous: {homogeneous_advantage:+.1f}%")
        print(f"      Mixed team:  {mixed_advantage:+.1f}%")

        results["team_sizes"][team_size] = {
            "evolved_homogeneous": {
                "mean": float(evolved_mean),
                "std": float(evolved_std),
                "payoffs": [float(p) for p in evolved_payoffs],
            },
            "heuristic_homogeneous": {
                "mean": float(heuristic_mean),
                "std": float(heuristic_std),
                "payoffs": [float(p) for p in heuristic_payoffs],
            },
            "mixed_team": {
                "mean": float(mixed_mean),
                "std": float(mixed_std),
                "payoffs": [float(p) for p in mixed_evolved_payoffs],
            },
            "advantage": {
                "homogeneous_pct": float(homogeneous_advantage),
                "mixed_pct": float(mixed_advantage),
            },
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="Head-to-head: evolved vs heuristic")
    parser.add_argument("scenario", type=str, help="Scenario name")
    parser.add_argument("--mode", type=str, default="extended", help="Evolution mode (extended, competitive, warm_start)")
    parser.add_argument("--games", type=int, default=100, help="Number of games per test")
    parser.add_argument("--vary-team-size", action="store_true", help="Test with team sizes 3-9")

    args = parser.parse_args()

    print("=" * 80)
    print("HEAD-TO-HEAD: Evolved vs Best Heuristic")
    print("=" * 80)
    print()

    # Load evolved agent
    evolved_file = Path(f"experiments/scenarios/{args.scenario}/evolved/{args.mode}/best_agent.json")
    if not evolved_file.exists():
        print(f"❌ Evolved agent not found: {evolved_file}")
        print(f"   Run extended evolution first:")
        print(f"   python experiments/scripts/run_extended_evolution.py {args.scenario}")
        return

    with open(evolved_file) as f:
        evolved_data = json.load(f)
    evolved_genome = evolved_data["genome"]

    print(f"Scenario: {args.scenario}")
    print(f"Evolution mode: {args.mode}")
    print(f"Games per test: {args.games}")
    print()

    # Load best heuristic
    heuristics_file = Path(f"experiments/scenarios/{args.scenario}/heuristics/results.json")
    if not heuristics_file.exists():
        print(f"❌ Heuristics not found: {heuristics_file}")
        return

    with open(heuristics_file) as f:
        heuristics_data = json.load(f)

    # Get best heuristic genome
    best_heuristic_name = heuristics_data["best_homogeneous"]["composition"].split()[0].lower()

    # Map to genome
    heuristic_genomes = {
        "firefighter": [0.7, 0.2, 0.0, 0.9, 0.0, 0.0, 0.1, 0.0, 0.9, 0.0],
        "free_rider": [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.9, 0.0],
        "hero": [1.0, 0.9, 0.5, 0.8, 0.5, 0.7, 0.1, 0.0, 0.0, 0.8],
        "coordinator": [0.9, 0.6, 0.9, 0.5, 0.3, 0.9, 0.3, 0.5, 0.2, 0.6],
        "liar": [0.0, 0.3, 0.0, 0.9, 0.8, 0.0, 0.5, 0.0, 0.8, 0.0],
    }
    heuristic_genome = heuristic_genomes.get(best_heuristic_name, [0.5] * 10)

    print(f"Best Heuristic: {best_heuristic_name.capitalize()}")
    print(f"  Performance: {heuristics_data['best_homogeneous']['mean_payoff']:.2f}")
    print()

    print(f"Evolved Agent:")
    print(f"  Fitness: {evolved_data['fitness']:.2f}")
    print(f"  Generation: {evolved_data['generation']}")
    print()

    # Determine team sizes to test
    if args.vary_team_size:
        team_sizes = [3, 5, 7, 9]
        print("Testing team sizes: 3, 5, 7, 9")
    else:
        team_sizes = [4]
        print("Testing team size: 4")

    print()
    print("Running tournament...")

    # Run head-to-head
    results = run_head_to_head(
        scenario_name=args.scenario,
        evolved_genome=evolved_genome,
        heuristic_genome=heuristic_genome,
        num_games=args.games,
        team_sizes=team_sizes,
    )

    # Save results
    output_dir = Path(f"experiments/scenarios/{args.scenario}/evolved/{args.mode}")
    output_file = output_dir / "head_to_head.json"

    results["best_heuristic_name"] = best_heuristic_name
    results["evolved_genome"] = evolved_genome
    results["heuristic_genome"] = heuristic_genome

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()

    # Overall summary
    if len(team_sizes) == 1:
        team_size = team_sizes[0]
        data = results["team_sizes"][team_size]
        print(f"Team Size: {team_size}")
        print(f"  Evolved (homogeneous):  {data['evolved_homogeneous']['mean']:.2f} ± {data['evolved_homogeneous']['std']:.2f}")
        print(f"  Heuristic (homogeneous): {data['heuristic_homogeneous']['mean']:.2f} ± {data['heuristic_homogeneous']['std']:.2f}")
        print(f"  Mixed (1 evolved):       {data['mixed_team']['mean']:.2f} ± {data['mixed_team']['std']:.2f}")
        print()
        print(f"  Advantage (homogeneous): {data['advantage']['homogeneous_pct']:+.1f}%")
        print(f"  Advantage (mixed):       {data['advantage']['mixed_pct']:+.1f}%")
    else:
        print("Average advantage across team sizes:")
        print()
        homogeneous_advantages = [results["team_sizes"][size]["advantage"]["homogeneous_pct"] for size in team_sizes]
        mixed_advantages = [results["team_sizes"][size]["advantage"]["mixed_pct"] for size in team_sizes]

        print(f"  Homogeneous: {np.mean(homogeneous_advantages):+.1f}% (±{np.std(homogeneous_advantages):.1f}%)")
        print(f"  Mixed:       {np.mean(mixed_advantages):+.1f}% (±{np.std(mixed_advantages):.1f}%)")

    print()
    print(f"✅ Results saved to: {output_file}")
    print()


if __name__ == "__main__":
    main()
