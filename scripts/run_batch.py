#!/usr/bin/env python3
"""
Run a batch of Bucket Brigade games for ranking experiments.
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm
import typer
import sys

# Add the bucket_brigade package to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bucket_brigade.envs import BucketBrigadeEnv, random_scenario
from bucket_brigade.agents import create_random_agent
from bucket_brigade.orchestration.summary import (
    generate_statistical_summary,
    load_game_replays,
)


def run_batch_games(
    num_games: int = 50,
    num_agents: int = 6,
    output_dir: str = "results",
    seed: int = 42,
    generate_summary: bool = False,
    experiment_id: str = "default",
    scenario_name: str = "random",
):
    """
    Run a batch of games with random agents and scenarios.

    Args:
        num_games: Number of games to run
        num_agents: Number of agents per game
        output_dir: Directory to save results
        seed: Random seed
        generate_summary: Whether to generate statistical summary
        experiment_id: Unique identifier for this experiment
        scenario_name: Name of the scenario being tested
    """
    print(f"Running {num_games} games with {num_agents} agents each...")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    replays_path = output_path / "replays"
    replays_path.mkdir(exist_ok=True)

    # Initialize results storage
    results = []

    # Set random seed
    np.random.seed(seed)
    game_seed = seed

    for game_idx in tqdm(range(num_games), desc="Running games"):
        # Create random scenario
        scenario = random_scenario(num_agents, seed=game_seed)
        game_seed += 1

        # Create environment
        env = BucketBrigadeEnv(scenario)

        # Create random agents (each game gets fresh random agents)
        agent_params = [np.random.uniform(0, 1, 10) for _ in range(num_agents)]
        agents = [create_random_agent(i) for i in range(num_agents)]
        team_agent_ids = list(range(num_agents))  # Agent IDs for this team

        # Run game
        obs = env.reset(seed=game_seed)
        game_seed += 1

        total_rewards = np.zeros(num_agents)

        while not env.done:
            actions = np.array([agent.act(obs) for agent in agents])
            obs, rewards, dones, info = env.step(actions)
            total_rewards += rewards

        # Save replay
        replay_file = replays_path / f"game_{game_idx:04d}.json"
        env.save_replay(str(replay_file))

        # Record results
        result = {
            "game_id": game_idx,
            "scenario_id": game_idx,  # Simple scenario ID
            "team": team_agent_ids,  # Agent IDs in this team
            "agent_params": [params.tolist() for params in agent_params],
            "team_reward": float(np.sum(total_rewards)),
            "agent_rewards": total_rewards.tolist(),
            "nights_played": env.night,
            "saved_houses": int(np.sum(obs["houses"] == 0)),
            "ruined_houses": int(np.sum(obs["houses"] == 2)),
            "replay_path": str(replay_file.relative_to(output_path)),
        }
        results.append(result)

    # Save results to CSV
    df = pd.DataFrame(results)
    csv_path = output_path / "batch_results.csv"
    df.to_csv(csv_path, index=False)

    # Save summary stats
    summary = {
        "num_games": num_games,
        "num_agents": num_agents,
        "avg_team_reward": float(df["team_reward"].mean()),
        "avg_nights": float(df["nights_played"].mean()),
        "avg_saved_houses": float(df["saved_houses"].mean()),
        "total_games": len(results),
    }

    summary_path = output_path / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nBatch complete!")
    print(f"Results saved to: {output_path}")
    print(f"Games: {summary['num_games']}")
    print(f"Avg team reward: {summary['avg_team_reward']:.2f}")
    print(f"Avg nights: {summary['avg_nights']:.1f}")
    print(f"Avg saved houses: {summary['avg_saved_houses']:.1f}")

    # Generate statistical summary if requested
    if generate_summary:
        print("\nGenerating statistical summary...")
        summaries_path = output_path / "summaries"
        summaries_path.mkdir(exist_ok=True)

        # Load all replays
        replays = load_game_replays(replays_path)

        # Infer team archetypes (for now, just use "random_agent")
        team_archetypes = [f"random_agent_{i}" for i in range(num_agents)]

        # Generate summary
        stat_summary = generate_statistical_summary(
            experiment_id=experiment_id,
            scenario_name=scenario_name,
            team_archetypes=team_archetypes,
            replays=replays,
        )

        # Save summary
        summary_file = summaries_path / f"{experiment_id}.json"
        stat_summary.to_json(str(summary_file))
        print(f"Statistical summary saved to: {summary_file}")
        print(f"  Mean reward: {stat_summary.team_reward_mean:.1f}")
        print(
            f"  95% CI: [{stat_summary.team_reward_ci_lower:.1f}, {stat_summary.team_reward_ci_upper:.1f}]"
        )
        print(f"  Success rate: {stat_summary.success_rate * 100:.1f}%")


def main(
    num_games: int = typer.Option(50, help="Number of games to run"),
    num_agents: int = typer.Option(6, help="Number of agents per game"),
    output_dir: str = typer.Option("results", help="Output directory"),
    seed: int = typer.Option(42, help="Random seed"),
    generate_summary: bool = typer.Option(
        False, "--generate-summary", help="Generate statistical summary"
    ),
    experiment_id: str = typer.Option(
        "default", help="Unique identifier for this experiment"
    ),
    scenario_name: str = typer.Option("random", help="Name of the scenario"),
):
    """Run a batch of Bucket Brigade games."""
    run_batch_games(
        num_games,
        num_agents,
        output_dir,
        seed,
        generate_summary,
        experiment_id,
        scenario_name,
    )


if __name__ == "__main__":
    typer.run(main)
