#!/usr/bin/env python3
"""
Run heterogeneous tournament with random team compositions.

This script samples random teams from an agent pool and plays games across
multiple scenarios to generate data for ranking model fitting.

Usage:
    # Tournament with heuristics only
    python experiments/scripts/run_heterogeneous_tournament.py \
      --agents firefighter free_rider hero coordinator liar \
      --num-games 1000

    # Tournament with evolved agents included
    python experiments/scripts/run_heterogeneous_tournament.py \
      --agents firefighter evolved evolved_v3 evolved_v4 evolved_v5 \
      --num-games 1000

    # Specific scenarios only
    python experiments/scripts/run_heterogeneous_tournament.py \
      --agents firefighter free_rider evolved_v4 \
      --scenarios chain_reaction greedy_neighbor \
      --num-games 500
"""

import sys
import json
import random
import argparse
from pathlib import Path
from typing import Any, Dict, List
from datetime import datetime

import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from bucket_brigade.envs.scenarios_generated import get_scenario_by_name
from bucket_brigade.evolution.fitness_rust import _heuristic_action
from bucket_brigade.equilibrium.payoff_evaluator_rust import _convert_scenario_to_rust
from bucket_brigade.agents.archetypes import (
    FIREFIGHTER_PARAMS,
    FREE_RIDER_PARAMS,
    HERO_PARAMS,
    COORDINATOR_PARAMS,
    LIAR_PARAMS,
)
import bucket_brigade_core as core


# Heuristic genomes
HEURISTIC_GENOMES = {
    "firefighter": FIREFIGHTER_PARAMS,
    "free_rider": FREE_RIDER_PARAMS,
    "hero": HERO_PARAMS,
    "coordinator": COORDINATOR_PARAMS,
    "liar": LIAR_PARAMS,
}


def load_agent_genome(agent_name: str, scenario: str = None) -> np.ndarray:
    """
    Load genome for a named agent.

    Args:
        agent_name: Name of agent (e.g., 'firefighter', 'evolved_v3')
        scenario: Scenario name (needed for evolved agents)

    Returns:
        Genome as numpy array
    """
    # Check if it's a heuristic
    if agent_name in HEURISTIC_GENOMES:
        return np.array(HEURISTIC_GENOMES[agent_name])

    # Check if it's an evolved agent
    if agent_name.startswith("evolved"):
        if scenario is None:
            # For scenario-agnostic loading, use chain_reaction as default
            scenario = "chain_reaction"

        evolved_dir = agent_name if agent_name == "evolved" else agent_name
        genome_file = Path(
            f"experiments/scenarios/{scenario}/{evolved_dir}/best_agent.json"
        )

        if not genome_file.exists():
            raise FileNotFoundError(f"No genome found for {agent_name} in {scenario}")

        with open(genome_file) as f:
            data = json.load(f)
            return np.array(data["genome"])

    raise ValueError(f"Unknown agent: {agent_name}")


def load_agent_pool(agent_names: List[str], scenario: str) -> Dict[str, np.ndarray]:
    """
    Load genomes for all agents in the pool.

    Args:
        agent_names: List of agent names
        scenario: Scenario name (for evolved agents)

    Returns:
        Dictionary mapping agent_name -> genome
    """
    agent_pool = {}
    for name in agent_names:
        try:
            genome = load_agent_genome(name, scenario)
            agent_pool[name] = genome
        except FileNotFoundError as e:
            print(f"⚠️  Skipping {name}: {e}")

    return agent_pool


def play_heterogeneous_game(
    team_genomes: List[np.ndarray], scenario_name: str, seed: int
) -> Dict[str, Any]:
    """
    Play a single game with a heterogeneous team.

    Args:
        team_genomes: List of genomes (one per agent)
        scenario_name: Name of scenario
        seed: Random seed

    Returns:
        Dictionary with game results
    """
    # Get scenario and convert to Rust
    python_scenario = get_scenario_by_name(scenario_name, num_agents=len(team_genomes))
    rust_scenario = _convert_scenario_to_rust(python_scenario)

    # Create Rust game
    game = core.BucketBrigade(rust_scenario, num_agents=len(team_genomes), seed=seed)

    # Python RNG for heuristic decisions
    rng = np.random.RandomState(seed)

    # Track cumulative rewards for each agent
    total_rewards = np.zeros(len(team_genomes))

    # Run game
    done = False
    step_count = 0
    max_steps = 100

    while not done and step_count < max_steps:
        # Get actions for all agents
        actions = []
        for agent_id, genome in enumerate(team_genomes):
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
        total_rewards += rewards  # Accumulate individual rewards
        step_count += 1

    # Get final result
    result = game.get_result()

    return {
        "agent_rewards": total_rewards.tolist(),
        "team_reward": float(result.final_score),
        "mean_reward": float(np.mean(total_rewards)),
        "steps": step_count,
    }


def run_heterogeneous_tournament(
    agent_pool: Dict[str, np.ndarray],
    scenarios: List[str],
    num_games: int,
    team_size: int = 4,
    seed: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run tournament with random team compositions.

    Args:
        agent_pool: Dictionary mapping agent_name -> genome
        scenarios: List of scenario names
        num_games: Total number of games to play
        team_size: Number of agents per team
        seed: Random seed for reproducibility
        verbose: Print progress

    Returns:
        DataFrame with game observations
    """
    random.seed(seed)
    np.random.seed(seed)

    agent_names = list(agent_pool.keys())
    observations = []

    if verbose:
        print("Running heterogeneous tournament...")
        print(f"  Agents: {', '.join(agent_names)}")
        print(f"  Scenarios: {', '.join(scenarios)}")
        print(f"  Games: {num_games}")
        print(f"  Team size: {team_size}")
        print()

    for game_idx in range(num_games):
        # Sample random scenario
        scenario = random.choice(scenarios)

        # Load agent pool for this scenario (evolved agents may differ)
        scenario_pool = load_agent_pool(agent_names, scenario)

        if len(scenario_pool) < team_size:
            if verbose and game_idx == 0:
                print(f"⚠️  Not enough agents for {scenario}, skipping...")
            continue

        # Sample random team (with replacement)
        team_names = random.choices(list(scenario_pool.keys()), k=team_size)
        team_genomes = [scenario_pool[name] for name in team_names]

        # Play game
        try:
            result = play_heterogeneous_game(team_genomes, scenario, game_idx)

            observations.append(
                {
                    "game_id": game_idx,
                    "scenario": scenario,
                    "team": team_names,  # List of agent names
                    "individual_payoffs": result["agent_rewards"],
                    "team_payoff": result["mean_reward"],
                    "steps": result["steps"],
                }
            )

            if verbose and (game_idx + 1) % 100 == 0:
                print(f"  {game_idx + 1}/{num_games} games completed...")

        except Exception as e:
            if verbose:
                print(f"⚠️  Game {game_idx} failed: {e}")

    if verbose:
        print(f"\n✅ Tournament complete: {len(observations)} games")

    return pd.DataFrame(observations)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run heterogeneous tournament with random teams",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--agents",
        type=str,
        nargs="+",
        required=True,
        help="Agent names (e.g., firefighter evolved_v3 evolved_v4)",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        nargs="+",
        default=None,
        help="Scenario names (default: all 9 scenarios)",
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=1000,
        help="Total number of games to play (default: 1000)",
    )
    parser.add_argument(
        "--team-size",
        type=int,
        default=4,
        help="Number of agents per team (default: 4)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV file (default: experiments/tournaments/tournament_{timestamp}.csv)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    # Default scenarios
    if args.scenarios is None:
        args.scenarios = [
            "chain_reaction",
            "deceptive_calm",
            "early_containment",
            "greedy_neighbor",
            "mixed_motivation",
            "overcrowding",
            "rest_trap",
            "sparse_heroics",
            "trivial_cooperation",
        ]

    # Default output path
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = Path(f"experiments/tournaments/tournament_{timestamp}.csv")

    # Load initial agent pool (for validation)
    print("Loading agent pool...")
    agent_pool = load_agent_pool(args.agents, args.scenarios[0])

    if len(agent_pool) == 0:
        print("❌ No valid agents found!")
        return

    if len(agent_pool) < args.team_size:
        print(
            f"⚠️  Warning: Only {len(agent_pool)} agents available, but team size is {args.team_size}"
        )
        print("    Some games may be skipped.")
        print()

    # Run tournament
    df = run_heterogeneous_tournament(
        agent_pool=agent_pool,
        scenarios=args.scenarios,
        num_games=args.num_games,
        team_size=args.team_size,
        seed=args.seed,
        verbose=not args.quiet,
    )

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    if not args.quiet:
        print(f"\n✅ Tournament data saved to: {args.output}")
        print("\nDataset statistics:")
        print(f"  Games: {len(df)}")

        if len(df) > 0:
            print(f"  Scenarios: {df['scenario'].nunique()}")

            # Count agent appearances
            agent_counts = {}
            for team in df["team"]:
                for agent in team:
                    agent_counts[agent] = agent_counts.get(agent, 0) + 1

            print("\nAgent appearances:")
            for agent, count in sorted(
                agent_counts.items(), key=lambda x: x[1], reverse=True
            ):
                print(f"  {agent:20s}: {count:4d} games")

            print("\nTo fit ranking model:")
            print(
                f"  uv run python experiments/scripts/fit_ranking_model.py --data {args.output}"
            )


if __name__ == "__main__":
    main()
