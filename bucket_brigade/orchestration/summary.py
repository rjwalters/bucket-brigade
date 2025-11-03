"""
Statistical summary generation for Bucket Brigade experiments.

This module provides functions to aggregate results from multiple game replays
and generate comprehensive statistical summaries for research analysis.
"""

import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict

from bucket_brigade.utils.statistics import (
    confidence_interval,
    pearson_correlation,
)


@dataclass
class StatisticalSummary:
    """
    Comprehensive statistical summary for a team on a scenario.

    This captures both aggregate statistics and individual agent contributions
    across multiple game replays.
    """

    # Experiment metadata
    experiment_id: str
    scenario_name: str
    team_composition: List[str]  # Agent archetypes
    num_replays: int

    # Team performance statistics
    team_reward_mean: float
    team_reward_std: float
    team_reward_ci_lower: float
    team_reward_ci_upper: float

    # Game outcome statistics
    success_rate: float  # Fraction of games where >= 7 houses saved
    avg_saved_houses: float
    saved_houses_std: float
    avg_ruined_houses: float
    ruined_houses_std: float

    # Game dynamics
    avg_nights_played: float
    nights_played_std: float
    avg_total_work: float  # Total work actions across all agents
    work_std: float

    # Individual agent statistics
    agent_contributions: List[Dict[str, Any]]  # Per-agent stats

    # Correlations and insights
    work_reward_correlation: float  # Correlation between work and reward
    work_reward_p_value: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def to_json(self, filepath: str) -> None:
        """Save summary to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, filepath: str) -> "StatisticalSummary":
        """Load summary from JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls(**data)


def load_game_replays(replay_dir: Path) -> List[Dict[str, Any]]:
    """
    Load all JSON game replays from a directory.

    Args:
        replay_dir: Directory containing game_{idx:04d}.json files

    Returns:
        List of replay dictionaries
    """
    replays = []
    replay_files = sorted(replay_dir.glob("game_*.json"))

    for replay_file in replay_files:
        with open(replay_file, "r") as f:
            replay = json.load(f)
            replays.append(replay)

    return replays


def compute_agent_contribution_stats(
    replays: List[Dict[str, Any]], agent_id: int
) -> Dict[str, Any]:
    """
    Compute statistics for a specific agent across multiple replays.

    Args:
        replays: List of game replay dictionaries
        agent_id: Index of the agent (0-based)

    Returns:
        Dictionary of agent statistics
    """
    total_rewards = []
    work_counts = []
    rest_counts = []

    for replay in replays:
        nights = replay["nights"]
        num_nights = len(nights)

        # Extract final reward for this agent
        final_reward = nights[-1]["rewards"][agent_id]
        total_rewards.append(final_reward)

        # Count work vs rest actions
        work_count = sum(1 for night in nights if night["actions"][agent_id][1] == 1)
        rest_count = num_nights - work_count

        work_counts.append(work_count)
        rest_counts.append(rest_count)

    # Compute statistics
    reward_mean, reward_ci_lower, reward_ci_upper = confidence_interval(
        np.array(total_rewards)
    )

    return {
        "agent_id": agent_id,
        "reward_mean": float(reward_mean),
        "reward_std": float(np.std(total_rewards)),
        "reward_ci_lower": float(reward_ci_lower),
        "reward_ci_upper": float(reward_ci_upper),
        "avg_work_nights": float(np.mean(work_counts)),
        "avg_rest_nights": float(np.mean(rest_counts)),
        "work_participation_rate": float(
            np.mean(work_counts) / (np.mean(work_counts) + np.mean(rest_counts))
        ),
    }


def generate_statistical_summary(
    experiment_id: str,
    scenario_name: str,
    team_archetypes: List[str],
    replays: List[Dict[str, Any]],
) -> StatisticalSummary:
    """
    Generate a comprehensive statistical summary from multiple game replays.

    Args:
        experiment_id: Unique identifier for this experiment
        scenario_name: Name of the scenario (e.g., "early_containment")
        team_archetypes: List of agent archetypes (e.g., ["firefighter", "coordinator"])
        replays: List of game replay dictionaries

    Returns:
        StatisticalSummary object with aggregated statistics

    Example:
        >>> replays = load_game_replays(Path("results/replays"))
        >>> summary = generate_statistical_summary(
        ...     "exp001",
        ...     "early_containment",
        ...     ["firefighter", "firefighter", "coordinator", "hero"],
        ...     replays
        ... )
        >>> summary.to_json("results/summaries/exp001.json")
    """
    num_replays = len(replays)
    if num_replays == 0:
        raise ValueError("No replays provided")

    num_agents = len(team_archetypes)

    # Extract team-level metrics
    team_rewards = []
    saved_houses = []
    ruined_houses = []
    nights_played = []
    total_work_actions = []

    for replay in replays:
        nights = replay["nights"]
        final_night = nights[-1]

        # Team reward is sum of all agent rewards
        team_reward = sum(final_night["rewards"])
        team_rewards.append(team_reward)

        # House outcomes (0=SAFE, 1=BURNING, 2=RUINED)
        houses = final_night["houses"]
        saved = sum(1 for h in houses if h == 0)
        ruined = sum(1 for h in houses if h == 2)
        saved_houses.append(saved)
        ruined_houses.append(ruined)

        # Game length
        nights_played.append(len(nights))

        # Total work actions across all agents and nights
        work_count = sum(
            1
            for night in nights
            for action in night["actions"]
            if action[1] == 1  # WORK mode
        )
        total_work_actions.append(work_count)

    # Compute team statistics
    team_reward_mean, team_ci_lower, team_ci_upper = confidence_interval(
        np.array(team_rewards)
    )

    # Success rate (7+ houses saved)
    success_count = sum(1 for saved in saved_houses if saved >= 7)
    success_rate = success_count / num_replays

    # Compute individual agent statistics
    agent_contributions = []
    for agent_id in range(num_agents):
        agent_stats = compute_agent_contribution_stats(replays, agent_id)
        agent_stats["archetype"] = team_archetypes[agent_id]
        agent_contributions.append(agent_stats)

    # Compute correlations
    work_reward_corr, work_reward_p = pearson_correlation(
        np.array(total_work_actions), np.array(team_rewards)
    )

    # Create summary object
    summary = StatisticalSummary(
        experiment_id=experiment_id,
        scenario_name=scenario_name,
        team_composition=team_archetypes,
        num_replays=num_replays,
        team_reward_mean=float(team_reward_mean),
        team_reward_std=float(np.std(team_rewards)),
        team_reward_ci_lower=float(team_ci_lower),
        team_reward_ci_upper=float(team_ci_upper),
        success_rate=float(success_rate),
        avg_saved_houses=float(np.mean(saved_houses)),
        saved_houses_std=float(np.std(saved_houses)),
        avg_ruined_houses=float(np.mean(ruined_houses)),
        ruined_houses_std=float(np.std(ruined_houses)),
        avg_nights_played=float(np.mean(nights_played)),
        nights_played_std=float(np.std(nights_played)),
        avg_total_work=float(np.mean(total_work_actions)),
        work_std=float(np.std(total_work_actions)),
        agent_contributions=agent_contributions,
        work_reward_correlation=float(work_reward_corr),
        work_reward_p_value=float(work_reward_p),
    )

    return summary


def compare_summaries(
    summary_a: StatisticalSummary, summary_b: StatisticalSummary
) -> Dict[str, Any]:
    """
    Compare two statistical summaries to identify significant differences.

    Args:
        summary_a: First summary
        summary_b: Second summary

    Returns:
        Dictionary containing comparison metrics

    Example:
        >>> summary_firefighters = StatisticalSummary.from_json("results/team_a.json")
        >>> summary_freeriders = StatisticalSummary.from_json("results/team_b.json")
        >>> comparison = compare_summaries(summary_firefighters, summary_freeriders)
        >>> print(f"Reward difference: {comparison['reward_difference']:.1f}")
    """
    comparison = {
        "summary_a_id": summary_a.experiment_id,
        "summary_b_id": summary_b.experiment_id,
        "scenario_a": summary_a.scenario_name,
        "scenario_b": summary_b.scenario_name,
        "reward_difference": summary_a.team_reward_mean - summary_b.team_reward_mean,
        "reward_difference_percent": (
            (summary_a.team_reward_mean - summary_b.team_reward_mean)
            / summary_b.team_reward_mean
            * 100
        ),
        "success_rate_difference": summary_a.success_rate - summary_b.success_rate,
        "avg_saved_houses_difference": summary_a.avg_saved_houses
        - summary_b.avg_saved_houses,
        "confidence_intervals_overlap": not (
            summary_a.team_reward_ci_lower > summary_b.team_reward_ci_upper
            or summary_b.team_reward_ci_lower > summary_a.team_reward_ci_upper
        ),
    }

    return comparison


def rank_teams_by_scenario(
    summaries: List[StatisticalSummary], scenario_name: str
) -> List[Tuple[str, float, float]]:
    """
    Rank teams by performance on a specific scenario.

    Args:
        summaries: List of statistical summaries
        scenario_name: Name of scenario to filter by

    Returns:
        List of (experiment_id, mean_reward, ci_width) tuples, sorted by reward

    Example:
        >>> all_summaries = [
        ...     StatisticalSummary.from_json(f)
        ...     for f in Path("results/summaries").glob("*.json")
        ... ]
        >>> rankings = rank_teams_by_scenario(all_summaries, "early_containment")
        >>> for rank, (exp_id, reward, ci_width) in enumerate(rankings, 1):
        ...     print(f"{rank}. {exp_id}: {reward:.1f} ± {ci_width:.1f}")
    """
    # Filter by scenario
    filtered = [s for s in summaries if s.scenario_name == scenario_name]

    # Compute rankings
    rankings = []
    for summary in filtered:
        ci_width = summary.team_reward_ci_upper - summary.team_reward_ci_lower
        rankings.append((summary.experiment_id, summary.team_reward_mean, ci_width))

    # Sort by mean reward descending
    rankings.sort(key=lambda x: x[1], reverse=True)

    return rankings
