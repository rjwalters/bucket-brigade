#!/usr/bin/env python3
"""
Analyze ranking results from batch experiments.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import typer
import json


def analyze_results(results_dir: str):
    """
    Analyze batch results and generate basic ranking statistics.

    Args:
        results_dir: Directory containing batch_results.csv and summary.json
    """
    results_path = Path(results_dir)

    # Load results
    csv_path = results_path / "batch_results.csv"
    summary_path = results_path / "summary.json"

    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        return

    df = pd.read_csv(csv_path)

    print("=== Batch Analysis Results ===")
    print(f"Games analyzed: {len(df)}")
    print(f"Agents per game: {len(df.iloc[0]['agent_rewards'].split(','))}")
    print()

    # Basic statistics
    print("Team Reward Statistics:")
    print(".2f")
    print(".2f")
    print(f"  Median: {df['team_reward'].median():.2f}")
    print()

    print("Game Length Statistics:")
    print(f"  Average nights: {df['nights_played'].mean():.1f}")
    print(f"  Min nights: {df['nights_played'].min()}")
    print(f"  Max nights: {df['nights_played'].max()}")
    print()

    print("Outcome Statistics:")
    print(f"  Average saved houses: {df['saved_houses'].mean():.1f}")
    print(f"  Average ruined houses: {df['ruined_houses'].mean():.1f}")
    print()

    # Agent-level analysis (simplified)
    all_agent_rewards = []
    for _, row in df.iterrows():
        rewards = [float(x) for x in row['agent_rewards'].strip('[]').split(',')]
        all_agent_rewards.extend(rewards)

    print("Agent Reward Statistics (across all games):")
    print(f"  Average agent reward: {np.mean(all_agent_rewards):.2f}")
    print(f"  Agent reward std: {np.std(all_agent_rewards):.2f}")
    print(f"  Best agent reward: {np.max(all_agent_rewards):.2f}")
    print(f"  Worst agent reward: {np.min(all_agent_rewards):.2f}")
    print()

    # Save analysis
    analysis = {
        "num_games": len(df),
        "team_reward_mean": float(df['team_reward'].mean()),
        "team_reward_std": float(df['team_reward'].std()),
        "nights_mean": float(df['nights_played'].mean()),
        "saved_houses_mean": float(df['saved_houses'].mean()),
        "agent_reward_mean": float(np.mean(all_agent_rewards)),
        "agent_reward_std": float(np.std(all_agent_rewards))
    }

    analysis_path = results_path / "analysis.json"
    with open(analysis_path, 'w') as f:
        json.dump(analysis, f, indent=2)

    print(f"Analysis saved to: {analysis_path}")

    # Note: Full ranking system would fit surrogate models here
    print("\nNote: This is basic analysis. Full ranking system with surrogate models")
    print("and uncertainty quantification is planned for the next development stage.")


def main(results_dir: str = typer.Argument("results", help="Results directory to analyze")):
    """Analyze ranking experiment results."""
    analyze_results(results_dir)


if __name__ == "__main__":
    typer.run(main)
