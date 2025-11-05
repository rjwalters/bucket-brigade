#!/usr/bin/env python3
"""
Fit Bayesian additive model to estimate individual agent skill from mixed team data.

Uses ridge regression to estimate each agent's marginal contribution to team performance.

Model:
    team_payoff = intercept + scenario_effect + sum(agent_skills) + noise

Usage:
    # Fit model to existing heuristics data
    python experiments/scripts/fit_ranking_model.py --data heuristics

    # Fit to custom tournament data
    python experiments/scripts/fit_ranking_model.py --data path/to/tournament.csv
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from scipy import stats

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_heuristics_data(scenarios: List[str]) -> pd.DataFrame:
    """
    Load mixed team data from heuristics results.

    Args:
        scenarios: List of scenario names to load

    Returns:
        DataFrame with columns: [game_id, scenario, team, individual_payoffs, team_payoff]
    """
    observations = []
    game_id = 0

    for scenario in scenarios:
        results_file = Path(f"experiments/scenarios/{scenario}/heuristics/results.json")

        if not results_file.exists():
            print(f"⚠️  Skipping {scenario} (no heuristics data)")
            continue

        with open(results_file) as f:
            data = json.load(f)

        # Process mixed teams
        for team_result in data.get("mixed_teams", []):
            team_composition = team_result["team_composition"]

            for game in team_result["games"]:
                observations.append({
                    "game_id": game_id,
                    "scenario": scenario,
                    "team": tuple(team_composition),  # Make hashable
                    "individual_payoffs": game["individual_payoffs"],
                    "team_payoff": game["mean_payoff"],
                })
                game_id += 1

        # Also process homogeneous teams
        for team_result in data.get("homogeneous_teams", []):
            agent_type = team_result["agent_type"]
            team_composition = [agent_type] * 4

            for game in team_result["games"]:
                observations.append({
                    "game_id": game_id,
                    "scenario": scenario,
                    "team": tuple(team_composition),
                    "individual_payoffs": game["individual_payoffs"],
                    "team_payoff": game["mean_payoff"],
                })
                game_id += 1

    df = pd.DataFrame(observations)
    print(f"Loaded {len(df)} games from {len(scenarios)} scenarios")
    return df


def build_design_matrix(
    df: pd.DataFrame, include_scenarios: bool = True
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Build design matrix for ridge regression.

    X has columns for:
    - Intercept (always 1)
    - Scenario indicators (one-hot, if include_scenarios=True)
    - Agent count per game (how many times each agent appears)

    Args:
        df: DataFrame with game observations
        include_scenarios: Whether to include scenario effects

    Returns:
        X: Design matrix (n_games × n_features)
        y: Response vector (team_payoffs)
        agent_names: List of agent names (ordered by X columns)
        scenario_names: List of scenario names (if include_scenarios)
    """
    n_games = len(df)

    # Get unique agents and scenarios
    all_agents = set()
    for team in df["team"]:
        all_agents.update(team)
    agent_names = sorted(all_agents)
    scenario_names = sorted(df["scenario"].unique())

    # Build feature names
    feature_names = ["intercept"]

    if include_scenarios:
        # One-hot encode scenarios (drop first for identifiability)
        feature_names.extend([f"scenario_{s}" for s in scenario_names[1:]])

    feature_names.extend([f"agent_{a}" for a in agent_names])

    # Build design matrix
    n_features = len(feature_names)
    X = np.zeros((n_games, n_features))

    # Intercept
    X[:, 0] = 1.0

    col_idx = 1

    # Scenario indicators
    if include_scenarios:
        scenario_to_idx = {s: i for i, s in enumerate(scenario_names[1:])}
        for i, scenario in enumerate(df["scenario"]):
            if scenario in scenario_to_idx:
                X[i, col_idx + scenario_to_idx[scenario]] = 1.0
        col_idx += len(scenario_names) - 1

    # Agent counts (how many times each agent appears in team)
    agent_to_idx = {a: i for i, a in enumerate(agent_names)}
    for i, team in enumerate(df["team"]):
        for agent in team:
            X[i, col_idx + agent_to_idx[agent]] += 1.0

    # Response variable
    y = df["team_payoff"].values

    return X, y, agent_names, scenario_names


def fit_ridge_regression(
    X: np.ndarray, y: np.ndarray, alpha: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit ridge regression and compute posterior covariance.

    Args:
        X: Design matrix (n_games × n_features)
        y: Response vector (n_games,)
        alpha: Ridge penalty parameter (λ)

    Returns:
        theta: Coefficient estimates (n_features,)
        cov: Posterior covariance matrix (n_features × n_features)
    """
    # Fit ridge regression
    model = Ridge(alpha=alpha, fit_intercept=False)  # Intercept already in X
    model.fit(X, y)
    theta = model.coef_

    # Compute residual variance
    y_pred = X @ theta
    residuals = y - y_pred
    n, p = X.shape
    sigma_squared = np.sum(residuals**2) / (n - p)

    # Posterior covariance: σ² (X^T X + αI)^(-1)
    XtX = X.T @ X
    cov = sigma_squared * np.linalg.inv(XtX + alpha * np.eye(p))

    return theta, cov


def extract_agent_ratings(
    theta: np.ndarray,
    cov: np.ndarray,
    agent_names: List[str],
    scenario_names: List[str],
    df: pd.DataFrame,
    include_scenarios: bool = True,
) -> Dict[str, Any]:
    """
    Extract agent skill ratings from fitted model.

    Args:
        theta: Fitted coefficients
        cov: Posterior covariance matrix
        agent_names: List of agent names
        scenario_names: List of scenario names
        df: Original dataframe (for counting games)
        include_scenarios: Whether scenario effects were included

    Returns:
        Dictionary with agent ratings and metadata
    """
    # Determine column offsets
    col_idx = 1  # Skip intercept
    if include_scenarios:
        col_idx += len(scenario_names) - 1

    # Extract agent coefficients and standard errors
    agent_ratings = {}
    for i, agent_name in enumerate(agent_names):
        theta_i = theta[col_idx + i]
        std_err = np.sqrt(cov[col_idx + i, col_idx + i])

        # 95% confidence interval (1.96 * SE)
        ci_lower = theta_i - 1.96 * std_err
        ci_upper = theta_i + 1.96 * std_err

        # Count games this agent appeared in
        num_games = sum(agent_name in team for team in df["team"])

        agent_ratings[agent_name] = {
            "theta": float(theta_i),
            "std_error": float(std_err),
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "num_games": int(num_games),
        }

    # Sort by theta (descending)
    sorted_agents = sorted(
        agent_ratings.items(), key=lambda x: x[1]["theta"], reverse=True
    )

    # Add ranks
    for rank, (agent_name, rating) in enumerate(sorted_agents, 1):
        rating["rank"] = rank

    return dict(sorted_agents)


def fit_scenario_specific_models(
    df: pd.DataFrame, alpha: float = 1.0
) -> Dict[str, Dict[str, Any]]:
    """
    Fit separate model for each scenario.

    Args:
        df: DataFrame with all game observations
        alpha: Ridge penalty

    Returns:
        Dictionary mapping scenario -> agent_ratings
    """
    scenario_rankings = {}

    for scenario in sorted(df["scenario"].unique()):
        print(f"\nFitting model for {scenario}...")

        # Filter to this scenario
        df_scenario = df[df["scenario"] == scenario].copy()

        # Build design matrix (no scenario effects needed)
        X, y, agent_names, _ = build_design_matrix(df_scenario, include_scenarios=False)

        # Fit model
        theta, cov = fit_ridge_regression(X, y, alpha=alpha)

        # Extract ratings
        ratings = extract_agent_ratings(
            theta, cov, agent_names, [], df_scenario, include_scenarios=False
        )

        scenario_rankings[scenario] = ratings

        # Print top 3
        print(f"  Top 3 agents:")
        for agent_name, rating in list(ratings.items())[:3]:
            print(
                f"    {rating['rank']}. {agent_name:15s}: θ={rating['theta']:6.2f} [{rating['ci_lower']:6.2f}, {rating['ci_upper']:6.2f}]"
            )

    return scenario_rankings


def fit_aggregate_model(df: pd.DataFrame, alpha: float = 1.0) -> Dict[str, Any]:
    """
    Fit single model across all scenarios.

    Args:
        df: DataFrame with all game observations
        alpha: Ridge penalty

    Returns:
        Dictionary with agent ratings
    """
    print("\nFitting aggregate model across all scenarios...")

    # Build design matrix (include scenario effects)
    X, y, agent_names, scenario_names = build_design_matrix(df, include_scenarios=True)

    # Fit model
    theta, cov = fit_ridge_regression(X, y, alpha=alpha)

    # Extract ratings
    ratings = extract_agent_ratings(
        theta, cov, agent_names, scenario_names, df, include_scenarios=True
    )

    # Print all agents
    print(f"\n{'Rank':<6} {'Agent':<20} {'θ (Skill)':<12} {'95% CI':<20} {'Games':<8}")
    print("=" * 70)
    for agent_name, rating in ratings.items():
        ci_str = f"[{rating['ci_lower']:.2f}, {rating['ci_upper']:.2f}]"
        print(
            f"{rating['rank']:<6} {agent_name:<20} {rating['theta']:<12.2f} {ci_str:<20} {rating['num_games']:<8}"
        )

    return ratings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit ranking model to mixed team tournament data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data",
        type=str,
        default="heuristics",
        help="Data source: 'heuristics' or path to CSV file",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        nargs="+",
        default=None,
        help="Scenarios to include (default: all available)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Ridge penalty parameter (default: 1.0)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON file (default: experiments/rankings/heuristics_rankings.json)",
    )

    args = parser.parse_args()

    # Load data
    if args.data == "heuristics":
        # Use all available scenarios if not specified
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

        df = load_heuristics_data(args.scenarios)

        if args.output is None:
            args.output = Path("experiments/rankings/heuristics_rankings.json")
    else:
        # Load from CSV
        df = pd.read_csv(args.data)
        df["team"] = df["team"].apply(eval)  # Convert string to tuple

        if args.output is None:
            args.output = Path("experiments/rankings/custom_rankings.json")

    print(f"\nDataset statistics:")
    print(f"  Total games: {len(df)}")
    print(f"  Scenarios: {df['scenario'].nunique()}")
    print(f"  Unique agents: {len(set(agent for team in df['team'] for agent in team))}")
    print()

    # Fit models
    scenario_rankings = fit_scenario_specific_models(df, alpha=args.alpha)
    aggregate_rankings = fit_aggregate_model(df, alpha=args.alpha)

    # Combine results
    results = {
        "metadata": {
            "num_games": len(df),
            "num_scenarios": df["scenario"].nunique(),
            "scenarios": sorted(df["scenario"].unique()),
            "alpha": args.alpha,
        },
        "aggregate": aggregate_rankings,
        "by_scenario": scenario_rankings,
    }

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results saved to: {args.output}")


if __name__ == "__main__":
    main()
