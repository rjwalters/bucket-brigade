#!/usr/bin/env python3
"""
Analyze hand-tuned heuristic agents in a scenario.

Usage:
    python experiments/scripts/analyze_heuristics.py greedy_neighbor
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from bucket_brigade.envs import BucketBrigadeEnv
from bucket_brigade.envs.scenarios import get_scenario_by_name
from bucket_brigade.agents import create_archetype_agent
from bucket_brigade.agents.archetypes import (
    FIREFIGHTER_PARAMS,
    FREE_RIDER_PARAMS,
    HERO_PARAMS,
    COORDINATOR_PARAMS,
    LIAR_PARAMS,
)


def run_game(agents: List, scenario, seed: int = None) -> Dict[str, Any]:
    """Run a single game with given agents."""
    env = BucketBrigadeEnv(scenario)
    obs = env.reset(seed=seed)

    total_rewards = np.zeros(len(agents))

    while not env.done:
        actions = np.array([agent.act(obs) for agent in agents])
        obs, rewards, dones, info = env.step(actions)
        total_rewards += rewards

    return {
        "individual_payoffs": total_rewards.tolist(),
        "mean_payoff": float(np.mean(total_rewards)),
        "saved_houses": int(np.sum(obs["houses"] == 0)),
        "ruined_houses": int(np.sum(obs["houses"] == 2)),
        "nights_played": env.night,
    }


def analyze_heuristics(scenario_name: str, output_dir: Path, num_games: int = 100):
    """Run tournament with heuristic archetypes."""

    print(f"Analyzing heuristics for scenario: {scenario_name}")
    print(f"Output directory: {output_dir}")
    print()

    # Load scenario
    scenario = get_scenario_by_name(scenario_name, num_agents=4)

    print("Scenario Parameters:")
    print(f"  beta (spread):       {scenario.beta:.2f}")
    print(f"  kappa (extinguish):  {scenario.kappa:.2f}")
    print(f"  c (work cost):       {scenario.c:.2f}")
    print(f"  num_agents:          {scenario.num_agents}")
    print()

    # Define agent types
    agent_types = [
        ("firefighter", FIREFIGHTER_PARAMS),
        ("free_rider", FREE_RIDER_PARAMS),
        ("hero", HERO_PARAMS),
        ("coordinator", COORDINATOR_PARAMS),
        ("liar", LIAR_PARAMS),
    ]

    print(f"Agent Types: {', '.join([name for name, _ in agent_types])}")
    print()

    # Test homogeneous teams (all same type)
    print("Testing homogeneous teams...")
    homogeneous_results = []

    for agent_name, agent_params in agent_types:
        print(f"  {agent_name.title()}: ", end="", flush=True)

        game_results = []
        for game_idx in range(num_games):
            # Create team of 4 identical agents
            agents = [create_archetype_agent(agent_name, i) for i in range(4)]

            result = run_game(agents, scenario, seed=game_idx)
            game_results.append(result)

            if (game_idx + 1) % 20 == 0:
                print(f"{game_idx + 1}", end=" ", flush=True)

        # Aggregate results
        mean_payoff = np.mean([r["mean_payoff"] for r in game_results])
        std_payoff = np.std([r["mean_payoff"] for r in game_results])

        homogeneous_results.append(
            {
                "agent_type": agent_name,
                "team_composition": [agent_name] * 4,
                "num_games": num_games,
                "mean_payoff": float(mean_payoff),
                "std_payoff": float(std_payoff),
                "games": game_results,
            }
        )

        print(f"✓ Mean: {mean_payoff:.2f} ± {std_payoff:.2f}")

    print()

    # Test mixed teams (3 cooperators + 1 free rider, etc.)
    print("Testing mixed teams...")
    mixed_results = []

    test_teams = [
        # 3 cooperators + 1 defector
        (
            ["firefighter", "firefighter", "firefighter", "free_rider"],
            "3 Firefighters + 1 Free Rider",
        ),
        (
            ["coordinator", "coordinator", "coordinator", "free_rider"],
            "3 Coordinators + 1 Free Rider",
        ),
        (["hero", "hero", "hero", "free_rider"], "3 Heroes + 1 Free Rider"),
        # 2-2 split
        (
            ["firefighter", "firefighter", "free_rider", "free_rider"],
            "2 Firefighters + 2 Free Riders",
        ),
        (["coordinator", "coordinator", "liar", "liar"], "2 Coordinators + 2 Liars"),
        # Diverse teams
        (["firefighter", "coordinator", "hero", "free_rider"], "Diverse Team"),
    ]

    for team_composition, description in test_teams:
        print(f"  {description}: ", end="", flush=True)

        game_results = []
        for game_idx in range(num_games):
            agents = [
                create_archetype_agent(agent_type, i)
                for i, agent_type in enumerate(team_composition)
            ]

            result = run_game(agents, scenario, seed=game_idx)
            game_results.append(result)

        mean_payoff = np.mean([r["mean_payoff"] for r in game_results])
        std_payoff = np.std([r["mean_payoff"] for r in game_results])

        mixed_results.append(
            {
                "team_composition": team_composition,
                "description": description,
                "num_games": num_games,
                "mean_payoff": float(mean_payoff),
                "std_payoff": float(std_payoff),
                "games": game_results,
            }
        )

        print(f"✓ Mean: {mean_payoff:.2f} ± {std_payoff:.2f}")

    print()

    # Prepare output
    results = {
        "scenario": scenario_name,
        "parameters": {
            "beta": scenario.beta,
            "kappa": scenario.kappa,
            "c": scenario.c,
            "A": scenario.A,
            "L": scenario.L,
            "rho_ignite": scenario.rho_ignite,
            "num_agents": scenario.num_agents,
        },
        "agents": [
            {"name": name, "params": params.tolist()} for name, params in agent_types
        ],
        "homogeneous_teams": homogeneous_results,
        "mixed_teams": mixed_results,
    }

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "results.json"

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✅ Results saved to: {output_file}")

    # Print summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print()
    print("Homogeneous Teams (Best to Worst):")
    sorted_homogeneous = sorted(
        homogeneous_results, key=lambda x: x["mean_payoff"], reverse=True
    )
    for i, result in enumerate(sorted_homogeneous):
        print(
            f"  {i + 1}. {result['agent_type'].title()}: {result['mean_payoff']:.2f} ± {result['std_payoff']:.2f}"
        )

    print()
    print("Mixed Teams (Best to Worst):")
    sorted_mixed = sorted(mixed_results, key=lambda x: x["mean_payoff"], reverse=True)
    for i, result in enumerate(sorted_mixed):
        print(
            f"  {i + 1}. {result['description']}: {result['mean_payoff']:.2f} ± {result['std_payoff']:.2f}"
        )

    return results


def main():
    parser = argparse.ArgumentParser(description="Analyze heuristic agents")
    parser.add_argument("scenario", type=str, help="Scenario name")
    parser.add_argument(
        "--num-games", type=int, default=100, help="Number of games per team"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None, help="Output directory"
    )

    args = parser.parse_args()

    # Default output directory
    if args.output_dir is None:
        args.output_dir = Path(f"experiments/scenarios/{args.scenario}/heuristics")

    analyze_heuristics(args.scenario, args.output_dir, args.num_games)


if __name__ == "__main__":
    main()
