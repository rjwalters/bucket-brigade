"""
Statistical utilities for analyzing Bucket Brigade game results.

This module provides functions for computing confidence intervals, Shapley values,
performance rankings, and significance tests for multi-agent cooperation experiments.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy import stats


def confidence_interval(
    data: np.ndarray, confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    Compute mean and confidence interval for a dataset.

    Args:
        data: Array of values
        confidence: Confidence level (default 0.95 for 95% CI)

    Returns:
        Tuple of (mean, lower_bound, upper_bound)

    Example:
        >>> rewards = np.array([100, 105, 95, 110, 98])
        >>> mean, lower, upper = confidence_interval(rewards)
        >>> print(f"Mean: {mean:.1f}, CI: [{lower:.1f}, {upper:.1f}]")
    """
    n = len(data)
    if n < 2:
        return float(np.mean(data)), float(np.mean(data)), float(np.mean(data))

    mean = np.mean(data)
    std_err = stats.sem(data)
    margin = std_err * stats.t.ppf((1 + confidence) / 2, n - 1)

    return float(mean), float(mean - margin), float(mean + margin)


def bootstrap_confidence_interval(
    data: np.ndarray, confidence: float = 0.95, n_bootstrap: int = 10000
) -> Tuple[float, float, float]:
    """
    Compute mean and confidence interval using bootstrap resampling.

    More robust than parametric CI when distribution is non-normal.

    Args:
        data: Array of values
        confidence: Confidence level (default 0.95)
        n_bootstrap: Number of bootstrap samples

    Returns:
        Tuple of (mean, lower_bound, upper_bound)
    """
    n = len(data)
    if n < 2:
        return float(np.mean(data)), float(np.mean(data)), float(np.mean(data))

    means = np.array(
        [
            np.mean(np.random.choice(data, size=n, replace=True))
            for _ in range(n_bootstrap)
        ]
    )

    alpha = 1 - confidence
    lower = np.percentile(means, alpha / 2 * 100)
    upper = np.percentile(means, (1 - alpha / 2) * 100)

    return float(np.mean(data)), float(lower), float(upper)


def shapley_value_estimate(
    agent_id: int,
    team_rewards: Dict[Tuple[int, ...], float],
    num_agents: int,
    n_samples: int = 1000,
) -> float:
    """
    Estimate Shapley value for an agent's contribution to team performance.

    The Shapley value measures an agent's average marginal contribution across
    all possible team formations.

    Args:
        agent_id: ID of the agent to evaluate
        team_rewards: Dict mapping team compositions to their average rewards
        num_agents: Total number of agents in the pool
        n_samples: Number of random orderings to sample

    Returns:
        Estimated Shapley value (average marginal contribution)

    Example:
        >>> team_rewards = {
        ...     (0,): 50.0,
        ...     (1,): 60.0,
        ...     (0, 1): 120.0,
        ... }
        >>> shapley = shapley_value_estimate(0, team_rewards, 2)
    """
    contributions = []

    for _ in range(n_samples):
        # Random ordering of all agents
        ordering = list(np.random.permutation(num_agents))
        idx = ordering.index(agent_id)

        # Team before adding this agent
        team_before = tuple(sorted(ordering[:idx]))
        # Team after adding this agent
        team_after = tuple(sorted(ordering[: idx + 1]))

        # Marginal contribution
        reward_before = team_rewards.get(team_before, 0.0)
        reward_after = team_rewards.get(team_after, 0.0)
        contribution = reward_after - reward_before

        contributions.append(contribution)

    return float(np.mean(contributions))


def rank_agents_by_performance(
    agent_rewards: Dict[int, List[float]], method: str = "mean"
) -> List[Tuple[int, float, float]]:
    """
    Rank agents by their performance across multiple games.

    Args:
        agent_rewards: Dict mapping agent IDs to lists of their rewards
        method: Ranking method - 'mean', 'median', or 'robust_mean' (trimmed)

    Returns:
        List of (agent_id, score, uncertainty) tuples, sorted by score descending

    Example:
        >>> rewards = {
        ...     0: [100, 105, 95],
        ...     1: [110, 115, 105],
        ...     2: [90, 92, 88]
        ... }
        >>> rankings = rank_agents_by_performance(rewards)
        >>> print(f"Best agent: {rankings[0][0]} with score {rankings[0][1]:.1f}")
    """
    rankings = []

    for agent_id, rewards in agent_rewards.items():
        rewards_array = np.array(rewards)

        if method == "mean":
            score = np.mean(rewards_array)
            uncertainty = np.std(rewards_array) / np.sqrt(len(rewards_array))
        elif method == "median":
            score = np.median(rewards_array)
            # Use median absolute deviation for uncertainty
            mad = np.median(np.abs(rewards_array - score))
            uncertainty = 1.4826 * mad / np.sqrt(len(rewards_array))
        elif method == "robust_mean":
            # Trimmed mean (remove top and bottom 10%)
            sorted_rewards = np.sort(rewards_array)
            trim_count = max(1, int(0.1 * len(sorted_rewards)))
            trimmed = sorted_rewards[trim_count:-trim_count]
            score = np.mean(trimmed)
            uncertainty = np.std(trimmed) / np.sqrt(len(trimmed))
        else:
            raise ValueError(f"Unknown ranking method: {method}")

        rankings.append((agent_id, float(score), float(uncertainty)))

    # Sort by score descending
    rankings.sort(key=lambda x: x[1], reverse=True)
    return rankings


def mann_whitney_test(
    group_a: np.ndarray, group_b: np.ndarray, alternative: str = "two-sided"
) -> Tuple[float, float]:
    """
    Perform Mann-Whitney U test to compare two independent samples.

    Non-parametric test for whether two samples have different distributions.
    Useful when data is not normally distributed.

    Args:
        group_a: First sample
        group_b: Second sample
        alternative: 'two-sided', 'less', or 'greater'

    Returns:
        Tuple of (U statistic, p-value)

    Example:
        >>> team_a_rewards = np.array([100, 105, 95, 110])
        >>> team_b_rewards = np.array([80, 85, 75, 90])
        >>> u_stat, p_value = mann_whitney_test(team_a_rewards, team_b_rewards)
        >>> if p_value < 0.05:
        ...     print("Significant difference between teams")
    """
    u_stat, p_value = stats.mannwhitneyu(
        group_a, group_b, alternative=alternative, method="auto"
    )
    return float(u_stat), float(p_value)


def welch_t_test(
    group_a: np.ndarray, group_b: np.ndarray, alternative: str = "two-sided"
) -> Tuple[float, float]:
    """
    Perform Welch's t-test to compare two independent samples.

    Parametric test that doesn't assume equal variances. More robust than
    standard t-test when variances differ.

    Args:
        group_a: First sample
        group_b: Second sample
        alternative: 'two-sided', 'less', or 'greater'

    Returns:
        Tuple of (t statistic, p-value)
    """
    t_stat, p_value = stats.ttest_ind(
        group_a, group_b, equal_var=False, alternative=alternative
    )
    return float(t_stat), float(p_value)


def effect_size_cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """
    Compute Cohen's d effect size for comparing two groups.

    Cohen's d measures the standardized difference between two means.
    Interpretation:
    - |d| < 0.2: negligible
    - 0.2 <= |d| < 0.5: small
    - 0.5 <= |d| < 0.8: medium
    - |d| >= 0.8: large

    Args:
        group_a: First sample
        group_b: Second sample

    Returns:
        Cohen's d effect size
    """
    n_a, n_b = len(group_a), len(group_b)
    var_a, var_b = np.var(group_a, ddof=1), np.var(group_b, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))

    # Cohen's d
    d = (np.mean(group_a) - np.mean(group_b)) / pooled_std
    return float(d)


def compute_win_rate(
    agent_rewards: List[float], opponent_rewards: List[List[float]]
) -> float:
    """
    Compute agent's win rate against opponents.

    An agent "wins" a game if its reward is higher than the median opponent reward.

    Args:
        agent_rewards: List of agent's rewards across games
        opponent_rewards: List of lists, where each inner list contains opponent
                         rewards for the corresponding game

    Returns:
        Win rate (fraction of games won)
    """
    if len(agent_rewards) != len(opponent_rewards):
        raise ValueError("agent_rewards and opponent_rewards must have same length")

    wins = 0
    for agent_reward, opp_rewards in zip(agent_rewards, opponent_rewards):
        median_opp = np.median(opp_rewards)
        if agent_reward > median_opp:
            wins += 1

    return wins / len(agent_rewards)


def pearson_correlation(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Compute Pearson correlation coefficient and p-value.

    Measures linear relationship between two variables.

    Args:
        x: First variable
        y: Second variable

    Returns:
        Tuple of (correlation coefficient, p-value)

    Example:
        >>> work_effort = np.array([10, 15, 8, 20, 12])
        >>> rewards = np.array([100, 150, 80, 200, 120])
        >>> r, p = pearson_correlation(work_effort, rewards)
        >>> print(f"Correlation: {r:.3f}, p-value: {p:.3f}")
    """
    r, p = stats.pearsonr(x, y)
    return float(r), float(p)


def spearman_correlation(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Compute Spearman rank correlation coefficient and p-value.

    Non-parametric measure of monotonic relationship. More robust to outliers
    than Pearson correlation.

    Args:
        x: First variable
        y: Second variable

    Returns:
        Tuple of (correlation coefficient, p-value)
    """
    rho, p = stats.spearmanr(x, y)
    return float(rho), float(p)
