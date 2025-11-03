#!/usr/bin/env python3
"""
Analyze and compare statistical summaries from Bucket Brigade experiments.

This script loads summary JSON files, compares team performance across scenarios,
generates visualizations, and exports results for further analysis.
"""

import sys
from pathlib import Path
import json
import typer
import pandas as pd
import numpy as np
from typing import List, Optional

# Add the bucket_brigade package to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bucket_brigade.orchestration.summary import (
    StatisticalSummary,
    compare_summaries,
    rank_teams_by_scenario,
)

app = typer.Typer()


def load_all_summaries(summary_dir: Path) -> List[StatisticalSummary]:
    """Load all summary JSON files from a directory."""
    summaries = []
    for json_file in sorted(summary_dir.glob("*.json")):
        try:
            summary = StatisticalSummary.from_json(str(json_file))
            summaries.append(summary)
        except Exception as e:
            typer.echo(f"Error loading {json_file}: {e}", err=True)

    return summaries


def create_comparison_table(
    summaries: List[StatisticalSummary], scenario_name: Optional[str] = None
) -> pd.DataFrame:
    """
    Create a comparison table of all summaries.

    Args:
        summaries: List of statistical summaries
        scenario_name: Optional scenario to filter by

    Returns:
        DataFrame with comparison metrics
    """
    # Filter by scenario if specified
    if scenario_name:
        summaries = [s for s in summaries if s.scenario_name == scenario_name]

    rows = []
    for summary in summaries:
        row = {
            "experiment_id": summary.experiment_id,
            "scenario": summary.scenario_name,
            "team": ", ".join(summary.team_composition),
            "num_replays": summary.num_replays,
            "mean_reward": summary.team_reward_mean,
            "reward_ci": f"[{summary.team_reward_ci_lower:.1f}, {summary.team_reward_ci_upper:.1f}]",
            "success_rate": f"{summary.success_rate * 100:.1f}%",
            "avg_saved": f"{summary.avg_saved_houses:.1f}",
            "avg_nights": f"{summary.avg_nights_played:.1f}",
            "work_reward_r": f"{summary.work_reward_correlation:.3f}",
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def create_agent_contribution_table(
    summaries: List[StatisticalSummary],
) -> pd.DataFrame:
    """
    Create a table showing individual agent contributions across all experiments.

    Args:
        summaries: List of statistical summaries

    Returns:
        DataFrame with per-agent statistics
    """
    rows = []
    for summary in summaries:
        for agent_stats in summary.agent_contributions:
            row = {
                "experiment_id": summary.experiment_id,
                "scenario": summary.scenario_name,
                "agent_id": agent_stats["agent_id"],
                "archetype": agent_stats["archetype"],
                "reward_mean": agent_stats["reward_mean"],
                "reward_std": agent_stats["reward_std"],
                "work_rate": f"{agent_stats['work_participation_rate'] * 100:.1f}%",
                "avg_work_nights": agent_stats["avg_work_nights"],
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    return df


def compare_teams_pairwise(summaries: List[StatisticalSummary]) -> pd.DataFrame:
    """
    Perform pairwise comparisons between all teams on same scenarios.

    Args:
        summaries: List of statistical summaries

    Returns:
        DataFrame with pairwise comparison results
    """
    # Group by scenario
    by_scenario = {}
    for summary in summaries:
        scenario = summary.scenario_name
        if scenario not in by_scenario:
            by_scenario[scenario] = []
        by_scenario[scenario].append(summary)

    rows = []
    for scenario, scenario_summaries in by_scenario.items():
        # Compare all pairs
        for i, summary_a in enumerate(scenario_summaries):
            for summary_b in scenario_summaries[i + 1 :]:
                comparison = compare_summaries(summary_a, summary_b)

                row = {
                    "scenario": scenario,
                    "team_a": ", ".join(summary_a.team_composition),
                    "team_b": ", ".join(summary_b.team_composition),
                    "reward_diff": comparison["reward_difference"],
                    "reward_diff_pct": f"{comparison['reward_difference_percent']:.1f}%",
                    "success_diff": f"{comparison['success_rate_difference'] * 100:.1f}%",
                    "ci_overlap": "Yes"
                    if comparison["confidence_intervals_overlap"]
                    else "No",
                }
                rows.append(row)

    df = pd.DataFrame(rows)
    return df


@app.command()
def summarize(
    summary_dir: Path = typer.Option(
        "results/summaries", help="Directory containing summary JSON files"
    ),
    output_dir: Path = typer.Option(
        "results/analysis", help="Directory to save analysis results"
    ),
    scenario: Optional[str] = typer.Option(
        None, help="Filter by specific scenario name"
    ),
):
    """
    Load summaries and generate comprehensive analysis reports.
    """
    if not summary_dir.exists():
        typer.echo(f"Error: Directory {summary_dir} does not exist", err=True)
        raise typer.Exit(1)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all summaries
    typer.echo(f"Loading summaries from {summary_dir}...")
    summaries = load_all_summaries(summary_dir)

    if not summaries:
        typer.echo("No summaries found!", err=True)
        raise typer.Exit(1)

    typer.echo(f"Loaded {len(summaries)} summaries")

    # Generate comparison table
    typer.echo("\n=== Team Performance Comparison ===")
    comparison_df = create_comparison_table(summaries, scenario)
    typer.echo(comparison_df.to_string(index=False))

    # Save to CSV
    csv_path = output_dir / "team_comparison.csv"
    comparison_df.to_csv(csv_path, index=False)
    typer.echo(f"\nSaved to: {csv_path}")

    # Generate agent contribution table
    typer.echo("\n=== Individual Agent Contributions ===")
    agent_df = create_agent_contribution_table(summaries)
    typer.echo(agent_df.to_string(index=False))

    # Save to CSV
    agent_csv_path = output_dir / "agent_contributions.csv"
    agent_df.to_csv(agent_csv_path, index=False)
    typer.echo(f"\nSaved to: {agent_csv_path}")

    # Generate pairwise comparisons
    if len(summaries) > 1:
        typer.echo("\n=== Pairwise Team Comparisons ===")
        pairwise_df = compare_teams_pairwise(summaries)
        typer.echo(pairwise_df.to_string(index=False))

        # Save to CSV
        pairwise_csv_path = output_dir / "pairwise_comparisons.csv"
        pairwise_df.to_csv(pairwise_csv_path, index=False)
        typer.echo(f"\nSaved to: {pairwise_csv_path}")


@app.command()
def rank(
    summary_dir: Path = typer.Option(
        "results/summaries", help="Directory containing summary JSON files"
    ),
    scenario: str = typer.Argument(..., help="Scenario name to rank teams by"),
):
    """
    Rank teams by performance on a specific scenario.
    """
    if not summary_dir.exists():
        typer.echo(f"Error: Directory {summary_dir} does not exist", err=True)
        raise typer.Exit(1)

    # Load all summaries
    summaries = load_all_summaries(summary_dir)

    if not summaries:
        typer.echo("No summaries found!", err=True)
        raise typer.Exit(1)

    # Rank teams
    rankings = rank_teams_by_scenario(summaries, scenario)

    if not rankings:
        typer.echo(f"No summaries found for scenario: {scenario}", err=True)
        raise typer.Exit(1)

    # Display rankings
    typer.echo(f"\n=== Team Rankings for '{scenario}' ===\n")
    for rank, (exp_id, reward, ci_width) in enumerate(rankings, 1):
        typer.echo(f"{rank}. {exp_id}: {reward:.1f} ± {ci_width / 2:.1f}")


@app.command()
def compare(
    summary_a: Path = typer.Argument(..., help="Path to first summary JSON"),
    summary_b: Path = typer.Argument(..., help="Path to second summary JSON"),
):
    """
    Compare two specific summaries in detail.
    """
    if not summary_a.exists():
        typer.echo(f"Error: File {summary_a} does not exist", err=True)
        raise typer.Exit(1)

    if not summary_b.exists():
        typer.echo(f"Error: File {summary_b} does not exist", err=True)
        raise typer.Exit(1)

    # Load summaries
    sum_a = StatisticalSummary.from_json(str(summary_a))
    sum_b = StatisticalSummary.from_json(str(summary_b))

    # Display basic info
    typer.echo("\n=== Summary A ===")
    typer.echo(f"Experiment: {sum_a.experiment_id}")
    typer.echo(f"Scenario: {sum_a.scenario_name}")
    typer.echo(f"Team: {', '.join(sum_a.team_composition)}")
    typer.echo(f"Mean reward: {sum_a.team_reward_mean:.1f}")
    typer.echo(
        f"95% CI: [{sum_a.team_reward_ci_lower:.1f}, {sum_a.team_reward_ci_upper:.1f}]"
    )
    typer.echo(f"Success rate: {sum_a.success_rate * 100:.1f}%")

    typer.echo("\n=== Summary B ===")
    typer.echo(f"Experiment: {sum_b.experiment_id}")
    typer.echo(f"Scenario: {sum_b.scenario_name}")
    typer.echo(f"Team: {', '.join(sum_b.team_composition)}")
    typer.echo(f"Mean reward: {sum_b.team_reward_mean:.1f}")
    typer.echo(
        f"95% CI: [{sum_b.team_reward_ci_lower:.1f}, {sum_b.team_reward_ci_upper:.1f}]"
    )
    typer.echo(f"Success rate: {sum_b.success_rate * 100:.1f}%")

    # Perform comparison
    comparison = compare_summaries(sum_a, sum_b)

    typer.echo("\n=== Comparison ===")
    typer.echo(f"Reward difference: {comparison['reward_difference']:.1f}")
    typer.echo(f"Reward difference: {comparison['reward_difference_percent']:.1f}%")
    typer.echo(
        f"Success rate difference: {comparison['success_rate_difference'] * 100:.1f}%"
    )
    typer.echo(
        f"Confidence intervals overlap: {'Yes' if comparison['confidence_intervals_overlap'] else 'No'}"
    )

    if not comparison["confidence_intervals_overlap"]:
        if sum_a.team_reward_mean > sum_b.team_reward_mean:
            typer.echo("\n✓ Summary A is significantly better than Summary B")
        else:
            typer.echo("\n✓ Summary B is significantly better than Summary A")
    else:
        typer.echo(
            "\n⚠ Confidence intervals overlap - difference may not be significant"
        )


@app.command()
def export(
    summary_dir: Path = typer.Option(
        "results/summaries", help="Directory containing summary JSON files"
    ),
    output_file: Path = typer.Option(
        "results/all_summaries.json", help="Output file for combined JSON"
    ),
):
    """
    Export all summaries to a single JSON file for external analysis.
    """
    if not summary_dir.exists():
        typer.echo(f"Error: Directory {summary_dir} does not exist", err=True)
        raise typer.Exit(1)

    # Load all summaries
    summaries = load_all_summaries(summary_dir)

    if not summaries:
        typer.echo("No summaries found!", err=True)
        raise typer.Exit(1)

    # Convert to dict
    export_data = {
        "num_experiments": len(summaries),
        "summaries": [s.to_dict() for s in summaries],
    }

    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(export_data, f, indent=2)

    typer.echo(f"Exported {len(summaries)} summaries to: {output_file}")


if __name__ == "__main__":
    app()
