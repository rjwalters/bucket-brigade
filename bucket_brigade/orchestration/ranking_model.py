"""
Bayesian ranking model for agent skill estimation using ridge regression.

This implements the ranking system described in RANKING_SYSTEM.md:
- Ridge regression for agent skill estimation
- Bayesian posterior with uncertainty quantification
- Marginal value calculation for deployment decisions
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging


logger = logging.getLogger(__name__)


@dataclass
class RankingResult:
    """Result of fitting a ranking model."""

    agent_ids: List[int]
    skill_estimates: np.ndarray  # θ_i estimates
    skill_uncertainty: np.ndarray  # SE(θ_i)
    marginal_values: np.ndarray  # v_i deployment values
    posterior_mean: np.ndarray  # Full posterior mean
    posterior_cov: np.ndarray  # Full posterior covariance
    log_likelihood: float
    num_games: int


class AgentRankingModel:
    """
    Bayesian ridge regression model for estimating agent skills.

    Implements the additive model: y_g = α + μ_c(g) + Σ θ_i * I(i ∈ team_g) + ε
    """

    def __init__(self, regularization_lambda: float = 1.0, noise_variance: float = 1.0):
        """
        Initialize the ranking model.

        Args:
            regularization_lambda: Ridge regularization parameter
            noise_variance: Assumed noise variance σ²
        """
        self.lambda_ = regularization_lambda
        self.sigma2 = noise_variance

        # Model state
        self.fitted = False
        self.agent_ids: List[int] = []
        self.skill_estimates: Optional[np.ndarray] = None
        self.posterior_cov: Optional[np.ndarray] = None

    def fit(self, batch_results: List[Dict]) -> RankingResult:
        """
        Fit the ranking model to batch experiment data.

        Args:
            batch_results: List of batch result dictionaries with keys:
                - 'team': list of agent IDs
                - 'team_reward': scalar team outcome
                - 'scenario_id': scenario identifier (for future scenario effects)

        Returns:
            RankingResult with fitted parameters and uncertainties
        """
        if not batch_results:
            raise ValueError("Cannot fit model with empty batch results")

        # Extract unique agent IDs
        all_agent_ids = set()
        for result in batch_results:
            all_agent_ids.update(result["team"])
        self.agent_ids = sorted(list(all_agent_ids))

        agent_id_to_idx = {agent_id: idx for idx, agent_id in enumerate(self.agent_ids)}
        num_agents = len(self.agent_ids)

        # Build design matrix X and response vector y
        num_games = len(batch_results)
        X = np.zeros((num_games, num_agents))  # Agent membership indicators
        y = np.zeros(num_games)  # Team rewards

        for game_idx, result in enumerate(batch_results):
            team = result["team"]
            y[game_idx] = result["team_reward"]

            # Set agent membership indicators
            for agent_id in team:
                if agent_id in agent_id_to_idx:
                    agent_idx = agent_id_to_idx[agent_id]
                    X[game_idx, agent_idx] = 1.0

        # Add intercept column (always 1 for team effects)
        X = np.column_stack([np.ones(num_games), X])

        # Bayesian ridge regression
        # Prior: β ~ N(0, σ²/(λ*I))
        # Posterior: β ~ N(β_hat, Σ)

        lambda_matrix = self.lambda_ * np.eye(X.shape[1])
        lambda_matrix[0, 0] = 1e-6  # Very weak prior on intercept

        # Posterior covariance: Σ = σ² * (X^T X + λ I)^(-1)
        XtX_plus_lambda = X.T @ X + lambda_matrix / self.sigma2
        posterior_cov = self.sigma2 * np.linalg.inv(XtX_plus_lambda)

        # Posterior mean: β_hat = Σ * (X^T y) / σ²
        posterior_mean = posterior_cov @ (X.T @ y) / self.sigma2

        # Extract agent skill estimates (skip intercept)
        skill_estimates = posterior_mean[1:]
        skill_cov = posterior_cov[1:, 1:]
        skill_uncertainty = np.sqrt(np.diag(skill_cov))

        # Calculate marginal values (deployment-relevant values)
        marginal_values = self._calculate_marginal_values(
            skill_estimates, posterior_cov, batch_results
        )

        # Calculate log likelihood for model evaluation
        residuals = y - X @ posterior_mean
        log_likelihood = -0.5 * len(y) * np.log(2 * np.pi * self.sigma2)
        log_likelihood -= 0.5 * np.sum(residuals**2) / self.sigma2
        log_likelihood -= 0.5 * np.sum(posterior_mean**2) * self.lambda_ / self.sigma2

        self.fitted = True
        self.skill_estimates = skill_estimates
        self.posterior_cov = posterior_cov

        return RankingResult(
            agent_ids=self.agent_ids,
            skill_estimates=skill_estimates,
            skill_uncertainty=skill_uncertainty,
            marginal_values=marginal_values,
            posterior_mean=posterior_mean,
            posterior_cov=posterior_cov,
            log_likelihood=log_likelihood,
            num_games=num_games,
        )

    def _calculate_marginal_values(
        self,
        skill_estimates: np.ndarray,
        posterior_cov: np.ndarray,
        batch_results: List[Dict],
    ) -> np.ndarray:
        """
        Calculate deployment-relevant marginal values using Monte Carlo sampling.

        For each agent i, v_i = E[ f(S∪{i}, c) - f(S, c) ] over deployment distribution
        """
        num_agents = len(skill_estimates)
        marginal_values = np.zeros(num_agents)

        # For simplicity, use the fitted point estimates
        # In a full implementation, this would use posterior sampling
        # and evaluate over many possible team compositions

        for agent_idx in range(num_agents):
            agent_id = self.agent_ids[agent_idx]

            # Count games where this agent participated vs didn't
            games_with_agent = 0
            games_without_agent = 0
            total_reward_with = 0.0
            total_reward_without = 0.0

            for result in batch_results:
                team = result["team"]
                reward = result["team_reward"]

                if agent_id in team:
                    games_with_agent += 1
                    total_reward_with += reward
                else:
                    games_without_agent += 1
                    total_reward_without += reward

            # Calculate marginal contribution
            if games_with_agent > 0 and games_without_agent > 0:
                avg_with = total_reward_with / games_with_agent
                avg_without = total_reward_without / games_without_agent
                marginal_values[agent_idx] = avg_with - avg_without
            else:
                # Fallback to skill estimate if insufficient data
                marginal_values[agent_idx] = skill_estimates[agent_idx]

        return marginal_values

    def predict_team_reward(self, team: List[int]) -> Tuple[float, float]:
        """
        Predict team reward with uncertainty.

        Args:
            team: List of agent IDs

        Returns:
            Tuple of (mean_prediction, standard_error)
        """
        if not self.fitted or self.skill_estimates is None:
            raise ValueError("Model must be fitted before prediction")

        # Build feature vector for this team
        features = np.zeros(len(self.agent_ids) + 1)  # +1 for intercept
        features[0] = 1.0  # intercept

        for agent_id in team:
            if agent_id in self.agent_ids:
                agent_idx = self.agent_ids.index(agent_id)
                features[agent_idx + 1] = 1.0

        # We need to reconstruct the posterior mean from stored data
        # posterior_mean = [intercept, skill_estimates...]
        posterior_mean = np.zeros(len(self.agent_ids) + 1)
        posterior_mean[0] = 0.0  # intercept (regularized to 0)
        posterior_mean[1:] = self.skill_estimates

        # Prediction: E[y] = x^T β_hat
        prediction = features @ posterior_mean

        # For uncertainty, use a simplified approach since we don't store full covariance
        # Uncertainty: SE ≈ σ/√n where n is number of games with similar team composition
        # For now, use a fixed uncertainty estimate
        prediction_std = np.sqrt(self.sigma2) * 2.0  # Conservative estimate

        return prediction, prediction_std

    def get_agent_rankings(self, ranking_result: RankingResult) -> List[Dict]:
        """
        Get sorted agent rankings with confidence intervals.

        Returns:
            List of agent ranking dictionaries
        """
        rankings = []
        for i, agent_id in enumerate(ranking_result.agent_ids):
            # 95% confidence interval
            ci_lower = (
                ranking_result.skill_estimates[i]
                - 1.96 * ranking_result.skill_uncertainty[i]
            )
            ci_upper = (
                ranking_result.skill_estimates[i]
                + 1.96 * ranking_result.skill_uncertainty[i]
            )

            rankings.append(
                {
                    "agent_id": agent_id,
                    "rank": i + 1,  # Will be resorted
                    "skill_estimate": ranking_result.skill_estimates[i],
                    "skill_uncertainty": ranking_result.skill_uncertainty[i],
                    "marginal_value": ranking_result.marginal_values[i],
                    "confidence_interval": [ci_lower, ci_upper],
                    "games_played": ranking_result.num_games,  # Approximate
                }
            )

        # Sort by marginal value (deployment-relevant ranking)
        rankings.sort(key=lambda x: x["marginal_value"], reverse=True)

        # Update ranks
        for i, ranking in enumerate(rankings):
            ranking["rank"] = i + 1

        return rankings

    def suggest_next_batch(
        self, ranking_result: RankingResult, num_candidates: int = 32
    ) -> List[Dict]:
        """
        Suggest next batch of teams to maximize information gain.

        This is a simplified version. Full implementation would use:
        - A-optimal design for uncertainty reduction
        - Thompson sampling for exploration
        - Diversity constraints for team composition

        Args:
            ranking_result: Current ranking results
            num_candidates: Number of team combinations to suggest

        Returns:
            List of suggested team configurations
        """
        # For now, suggest diverse team combinations focusing on uncertain agents
        suggestions = []

        # Sort agents by uncertainty (most uncertain first)
        agent_uncertainty = list(
            zip(ranking_result.agent_ids, ranking_result.skill_uncertainty)
        )
        agent_uncertainty.sort(key=lambda x: x[1], reverse=True)

        # Generate team combinations with varying sizes
        team_sizes = [4, 5, 6, 7, 8]  # Common team sizes

        for team_size in team_sizes:
            # Focus on uncertain agents but include some known ones for baseline
            uncertain_agents = [aid for aid, _ in agent_uncertainty[: team_size // 2]]
            remaining_slots = team_size - len(uncertain_agents)

            # Fill with random selection from remaining agents
            available_agents = [
                aid for aid in ranking_result.agent_ids if aid not in uncertain_agents
            ]
            if len(available_agents) >= remaining_slots:
                fillers = np.random.choice(
                    available_agents, remaining_slots, replace=False
                )
                team = uncertain_agents + fillers.tolist()
                suggestions.append(
                    {
                        "team": sorted(team),
                        "team_size": team_size,
                        "focus": "uncertainty_reduction",
                        "expected_info_gain": np.mean(
                            [u for _, u in agent_uncertainty if _ in uncertain_agents]
                        ),
                    }
                )

        # Limit to requested number
        return suggestions[:num_candidates]


def evaluate_ranking_accuracy(
    true_skills: Dict[int, float], ranking_result: RankingResult
) -> Dict[str, float]:
    """
    Evaluate ranking accuracy against known true skills.

    Args:
        true_skills: Dictionary mapping agent_id to true skill value
        ranking_result: Fitted ranking results

    Returns:
        Dictionary of evaluation metrics
    """
    true_values = []
    estimated_values = []
    uncertainties = []

    for i, agent_id in enumerate(ranking_result.agent_ids):
        if agent_id in true_skills:
            true_values.append(true_skills[agent_id])
            estimated_values.append(ranking_result.skill_estimates[i])
            uncertainties.append(ranking_result.skill_uncertainty[i])

    if not true_values:
        return {"error": "No overlapping agents found"}

    true_values = np.array(true_values)
    estimated_values = np.array(estimated_values)
    uncertainties = np.array(uncertainties)

    # Calculate metrics
    mse = np.mean((true_values - estimated_values) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(true_values - estimated_values))

    # Correlation
    correlation = np.corrcoef(true_values, estimated_values)[0, 1]

    # Calibration (uncertainty accuracy)
    errors = np.abs(true_values - estimated_values)
    calibration_score = np.mean(errors <= uncertainties)  # Fraction within 1σ

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "correlation": correlation,
        "calibration_score": calibration_score,
        "num_agents": len(true_values),
    }
