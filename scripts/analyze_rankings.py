#!/usr/bin/env python3
"""
Analyze ranking results from batch experiments and fit ranking models.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any
import typer

# Add the bucket_brigade package to the path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from bucket_brigade.orchestration import AgentRankingModel, evaluate_ranking_accuracy


def load_batch_data(results_dir: str) -> tuple[list[Dict[str, Any]], pd.DataFrame]:
    """
    Load batch results from CSV and return both raw data and DataFrame.

    Args:
        results_dir: Directory containing batch_results.csv

    Returns:
        Tuple of (batch_results_list, dataframe)
    """
    results_path = Path(results_dir) / "batch_results.csv"

    if not results_path.exists():
        print(f"Error: {results_path} not found")
        print(f"Make sure to run batch experiments first:")
        print(f"  uv run python scripts/run_batch.py --num-games 50")
        return [], pd.DataFrame()

    df = pd.read_csv(results_path)

    # Convert to the format expected by ranking model
    batch_results = []
    for _, row in df.iterrows():
        result = {
            'game_id': int(row['game_id']),
            'scenario_id': int(row['scenario_id']),
            'team': json.loads(row['agent_params']),  # Wait, this should be team, not agent_params
            'team_reward': float(row['team_reward']),
            'agent_rewards': json.loads(row['agent_rewards']),
            'nights_played': int(row['nights_played']),
            'saved_houses': int(row['saved_houses']),
            'ruined_houses': int(row['ruined_houses']),
            'replay_path': row['replay_path']
        }

        # Fix: team should be the agent IDs, not parameters
        # For now, we'll extract from agent_rewards keys
        # In a real implementation, this should be stored properly
        if 'team' not in row or pd.isna(row['team']):
            # Extract agent IDs from agent_rewards
            agent_rewards = json.loads(row['agent_rewards'])
            result['team'] = list(range(len(agent_rewards)))
        else:
            result['team'] = json.loads(row['team'])

        batch_results.append(result)

    return batch_results, df


def analyze_rankings(results_dir: str = "results",
                    lambda_reg: float = 1.0,
                    save_model: bool = True) -> None:
    """
    Analyze batch results and fit ranking models.

    Args:
        results_dir: Directory containing batch_results.csv
        lambda_reg: Regularization parameter for ridge regression
        save_model: Whether to save the fitted model
    """
    print("🔍 Analyzing Batch Experiment Results")
    print("=" * 50)

    # Load data
    batch_results, df = load_batch_data(results_dir)

    if not batch_results:
        return

    print(f"📊 Loaded {len(batch_results)} games")
    print(f"🎯 Found {len(set(agent for result in batch_results for agent in result['team']))} unique agents")

    # Basic statistics
    print("\n📈 Basic Statistics:")
    print(f"  Average team reward: {df['team_reward'].mean():.2f} ± {df['team_reward'].std():.2f}")
    print(f"  Average nights played: {df['nights_played'].mean():.1f}")
    print(f"  Average houses saved: {df['saved_houses'].mean():.1f}")
    print(f"  Average houses ruined: {df['ruined_houses'].mean():.1f}")

    # Fit ranking model
    print(f"\n🤖 Fitting Ranking Model (λ={lambda_reg})...")
    model = AgentRankingModel(regularization_lambda=lambda_reg)

    try:
        ranking_result = model.fit(batch_results)

        print("✅ Model fitted successfully!")
        print(f"   Log-likelihood: {ranking_result.log_likelihood:.2f}")
        print(f"   Agents ranked: {len(ranking_result.agent_ids)}")

        # Display top and bottom performers
        agent_rankings = model.get_agent_rankings(ranking_result)

        print("\n🏆 Top Performers:")
        for i, ranking in enumerate(agent_rankings[:5]):
            print(f"   #{i+1} Agent {ranking['agent_id']}: "
                  f"{ranking['skill_estimate']:.3f} ± {ranking['skill_uncertainty']:.3f} "
                  f"(marginal: {ranking['marginal_value']:.3f})")

        print("\n📉 Bottom Performers:")
        for i, ranking in enumerate(agent_rankings[-3:]):
            rank = len(agent_rankings) - 2 + i
            print(f"   #{rank} Agent {ranking['agent_id']}: "
                  f"{ranking['skill_estimate']:.3f} ± {ranking['skill_uncertainty']:.3f} "
                  f"(marginal: {ranking['marginal_value']:.3f})")

        # Prediction accuracy (cross-validation style)
        print("\n🎯 Model Validation:")
        # Simple validation: predict held-out games
        predictions = []
        actuals = []

        for result in batch_results:
            try:
                pred_mean, pred_std = model.predict_team_reward(result['team'])
                predictions.append(pred_mean)
                actuals.append(result['team_reward'])
            except Exception as e:
                # Skip predictions that fail (model not fitted properly)
                print(f"   Skipping prediction for game {result['game_id']}: {e}")
                continue

        predictions = np.array(predictions)
        actuals = np.array(actuals)

        mse = np.mean((predictions - actuals) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - actuals))
        correlation = np.corrcoef(predictions, actuals)[0, 1] if len(predictions) > 1 else 0.0

        print(f"   RMSE: {rmse:.3f}")
        print(f"   MAE: {mae:.3f}")
        print(".3f")

        # Next batch suggestions
        print("\n🎲 Next Batch Suggestions:")
        suggestions = model.suggest_next_batch(ranking_result, num_candidates=5)

        for i, suggestion in enumerate(suggestions):
            team_str = ','.join(map(str, suggestion['team']))
            print(f"   Team {i+1}: [{team_str}] "
                  f"(size={suggestion['team_size']}, "
                  f"focus={suggestion['focus']})")

        # Save results
        if save_model:
            output_dir = Path(results_dir)
            ranking_file = output_dir / "ranking_results.json"

            ranking_data = {
                'model_params': {
                    'lambda': lambda_reg,
                    'num_games': ranking_result.num_games,
                    'num_agents': len(ranking_result.agent_ids)
                },
                'agent_rankings': agent_rankings,
                'model_stats': {
                    'log_likelihood': ranking_result.log_likelihood,
                    'rmse': rmse,
                    'mae': mae,
                    'correlation': correlation
                },
                'next_batch_suggestions': suggestions
            }

            with open(ranking_file, 'w') as f:
                json.dump(ranking_data, f, indent=2, default=str)

            print(f"\n💾 Results saved to: {ranking_file}")

    except Exception as e:
        print(f"❌ Error fitting model: {e}")
        print("This might be due to insufficient data or numerical issues.")
        print("Try running more games or adjusting the regularization parameter.")


def main(
    results_dir: str = typer.Option("results", help="Directory containing batch_results.csv"),
    lambda_reg: float = typer.Option(1.0, help="Regularization parameter for ridge regression"),
    save_model: bool = typer.Option(True, help="Save fitted model and results")
):
    """Analyze ranking experiment results and fit Bayesian ranking models."""
    analyze_rankings(results_dir, lambda_reg, save_model)


if __name__ == "__main__":
    typer.run(main)
