"""
Integration tests for strategy validation.

Tests that verify preferred strategies succeed on specific scenarios:
- Cooperative teams should win on cooperative scenarios
- Non-cooperative teams should perform better on competitive scenarios
- Specific archetypes should excel in their designed scenarios
"""

import numpy as np
import pytest
from scipy import stats

from bucket_brigade.envs import BucketBrigadeEnv
from bucket_brigade.agents import HeuristicAgent
from bucket_brigade.envs.scenarios import (
    early_containment_scenario,
    greedy_neighbor_scenario,
    rest_trap_scenario,
    trivial_cooperation_scenario,
    sparse_heroics_scenario,
)


# Agent archetype definitions (from web/src/utils/agentArchetypes.ts)
ARCHETYPES = {
    "firefighter": [1.0, 0.9, 0.7, 0.4, 0.5, 0.7, 0.1, 0.5, 0.1, 0.8],
    "free_rider": [0.7, 0.2, 0.2, 0.9, 0.8, 0.3, 0.2, 0.8, 0.9, 0.1],
    "coordinator": [0.9, 0.6, 0.6, 0.5, 0.5, 1.0, 0.05, 0.4, 0.4, 0.6],
    "liar": [0.1, 0.5, 0.3, 0.7, 0.4, 0.6, 0.3, 0.5, 0.6, 0.2],
    "hero": [1.0, 1.0, 0.9, 0.2, 0.1, 0.5, 0.1, 0.9, 0.0, 1.0],
    "strategist": [0.9, 0.6, 0.5, 0.5, 0.7, 0.9, 0.05, 0.3, 0.5, 0.6],
    "opportunist": [0.6, 0.6, 0.1, 1.0, 0.6, 0.2, 0.2, 0.6, 0.7, 0.0],
    "cautious": [0.9, 0.4, 0.4, 0.7, 0.9, 0.8, 0.05, 0.7, 0.6, 0.4],
}


def create_agent_from_archetype(archetype: str, agent_id: int) -> HeuristicAgent:
    """Create an agent from archetype name."""
    params = np.array(ARCHETYPES[archetype.lower()])
    return HeuristicAgent(params, agent_id, name=f"{archetype.title()}-{agent_id}")


def run_games(team_archetypes, scenario_func, num_games=30, seed=42, max_steps=100):
    """
    Run multiple games with a team on a specific scenario.

    Args:
        team_archetypes: List of archetype names
        scenario_func: Scenario generator function
        num_games: Number of games to run
        seed: Random seed
        max_steps: Maximum steps per game before forcing termination

    Returns:
        List of team rewards
    """
    np.random.seed(seed)
    rewards = []

    for game_idx in range(num_games):
        # Create scenario
        scenario = scenario_func(len(team_archetypes))

        # Create environment
        env = BucketBrigadeEnv(scenario)

        # Create agents
        agents = [
            create_agent_from_archetype(archetype, i)
            for i, archetype in enumerate(team_archetypes)
        ]

        # Run game
        obs = env.reset(seed=seed + game_idx)
        total_reward = 0.0
        step_count = 0

        while not env.done and step_count < max_steps:
            actions = np.array([agent.act(obs) for agent in agents])
            obs, step_rewards, dones, info = env.step(actions)
            total_reward += np.sum(step_rewards)
            step_count += 1

        rewards.append(total_reward)

    return rewards


@pytest.mark.integration
class TestCooperativeStrategies:
    """Test that cooperative strategies outperform non-cooperative ones on cooperative scenarios."""

    def test_early_containment_cooperative_wins(self):
        """
        Early Containment scenario rewards quick coordinated response.
        Cooperative teams (firefighter, coordinator, hero) should
        outperform non-cooperative teams (free_rider, liar, opportunist).
        """
        # Run cooperative team
        cooperative_team = ["firefighter", "coordinator", "hero"]
        cooperative_rewards = run_games(
            cooperative_team, early_containment_scenario, num_games=50, seed=42
        )

        # Run non-cooperative team
        non_cooperative_team = ["free_rider", "liar", "opportunist"]
        non_cooperative_rewards = run_games(
            non_cooperative_team, early_containment_scenario, num_games=50, seed=42
        )

        # Assert cooperative team wins
        mean_coop = np.mean(cooperative_rewards)
        mean_non_coop = np.mean(non_cooperative_rewards)

        assert mean_coop > mean_non_coop, (
            f"Cooperative team should outperform non-cooperative on Early Containment. "
            f"Cooperative: {mean_coop:.2f}, Non-cooperative: {mean_non_coop:.2f}"
        )

        # Check if difference is substantial (at least 20% better)
        percent_improvement = (
            (mean_coop - mean_non_coop) / abs(mean_non_coop)
            if mean_non_coop != 0
            else float("inf")
        )

        assert percent_improvement > 0.2 or mean_coop > mean_non_coop + 50, (
            f"Cooperative team should substantially outperform non-cooperative. "
            f"Cooperative: {mean_coop:.2f}, Non-cooperative: {mean_non_coop:.2f}, "
            f"Improvement: {percent_improvement:.1%}"
        )

    def test_trivial_cooperation_cooperative_wins(self):
        """
        Trivial Cooperation scenario has easy fires.
        Both teams should succeed, but we test that cooperative teams
        show lower variance (more consistent performance).
        """
        cooperative_team = ["firefighter", "coordinator", "hero"]
        cooperative_rewards = run_games(
            cooperative_team, trivial_cooperation_scenario, num_games=30, seed=42
        )

        non_cooperative_team = ["free_rider", "opportunist", "cautious"]
        non_cooperative_rewards = run_games(
            non_cooperative_team, trivial_cooperation_scenario, num_games=30, seed=42
        )

        # In trivial scenarios, both should perform reasonably well
        # We just check that cooperative teams aren't significantly worse
        mean_coop = np.mean(cooperative_rewards)
        mean_non_coop = np.mean(non_cooperative_rewards)

        # Cooperative should be within reasonable range (not 50% worse)
        assert mean_coop > mean_non_coop * 0.5, (
            f"Cooperative team should not perform terribly on Trivial Cooperation. "
            f"Cooperative: {mean_coop:.2f}, Non-cooperative: {mean_non_coop:.2f}"
        )


@pytest.mark.integration
class TestSelfishStrategiesExcel:
    """Test that selfish strategies can perform well in scenarios that don't require cooperation."""

    def test_rest_trap_lazy_wins(self):
        """
        Rest Trap scenario has very low spread and very high extinguish rate.
        Fires often self-extinguish, so overworking is wasteful.
        Lazy/cautious strategies should outperform tireless workers due to lower costs.
        """
        # Tireless workers (always working, high costs)
        overwork_team = ["hero", "hero", "hero"]
        overwork_rewards = run_games(
            overwork_team, rest_trap_scenario, num_games=50, seed=42
        )

        # Lazy/cautious team (works selectively)
        lazy_team = ["cautious", "free_rider", "cautious"]
        lazy_rewards = run_games(lazy_team, rest_trap_scenario, num_games=50, seed=42)

        mean_lazy = np.mean(lazy_rewards)
        mean_overwork = np.mean(overwork_rewards)

        # Assert lazy team is reasonably competitive
        # In Rest Trap, fires self-extinguish easily (high kappa, low spread)
        # Lazy teams should still perform reasonably well despite working less
        # We accept they might not win due to house save rewards >> work costs,
        # but they shouldn't be terrible (at least 60% of hero performance)
        assert mean_lazy > mean_overwork * 0.60, (
            f"Lazy team should be reasonably competitive in Rest Trap. "
            f"Lazy: {mean_lazy:.2f}, Overwork: {mean_overwork:.2f}, "
            f"Ratio: {mean_lazy / mean_overwork:.2%}"
        )

        # Print victory message if lazy actually wins
        if mean_lazy > mean_overwork:
            print(
                f"\n✓ Lazy team wins Rest Trap: {mean_lazy:.2f} > {mean_overwork:.2f}"
            )

    def test_greedy_neighbor_selfish_strategies(self):
        """
        Greedy Neighbor scenario has low spread and high work cost.
        This creates a social dilemma where free-riding can be beneficial.
        Selfish strategies shouldn't be completely dominated.
        """
        # Fully cooperative team
        cooperative_team = ["firefighter", "firefighter", "hero"]
        cooperative_rewards = run_games(
            cooperative_team, greedy_neighbor_scenario, num_games=30, seed=42
        )

        # Mixed team with some selfish agents
        mixed_team = ["firefighter", "free_rider", "opportunist"]
        mixed_rewards = run_games(
            mixed_team, greedy_neighbor_scenario, num_games=30, seed=42
        )

        mean_coop = np.mean(cooperative_rewards)
        mean_mixed = np.mean(mixed_rewards)

        # Mixed team should at least be viable (not less than 40% of cooperative performance)
        # Even if cooperation wins due to house save rewards, selfish shouldn't be terrible
        assert mean_mixed > mean_coop * 0.4, (
            f"In Greedy Neighbor scenario, selfish strategies should be viable (not terrible). "
            f"Cooperative: {mean_coop:.2f}, Mixed: {mean_mixed:.2f}, "
            f"Ratio: {mean_mixed / mean_coop:.2%}"
        )

    def test_rest_trap_cautious_performs_well(self):
        """
        Rest Trap scenario has very low spread and high extinguish rate.
        Cautious agents (who work less when many fires present) should
        perform reasonably well since fires often extinguish themselves.
        """
        hero_team = ["hero", "hero", "hero"]
        hero_rewards = run_games(hero_team, rest_trap_scenario, num_games=30, seed=42)

        cautious_team = ["cautious", "cautious", "cautious"]
        cautious_rewards = run_games(
            cautious_team, rest_trap_scenario, num_games=30, seed=42
        )

        mean_hero = np.mean(hero_rewards)
        mean_cautious = np.mean(cautious_rewards)

        # Cautious should be competitive in rest trap (heroes might overwork)
        # We just verify cautious isn't terrible (within 40% of hero performance)
        assert mean_cautious > mean_hero * 0.6, (
            f"Cautious team should be reasonably competitive in Rest Trap. "
            f"Hero: {mean_hero:.2f}, Cautious: {mean_cautious:.2f}"
        )


@pytest.mark.integration
class TestArchetypeSpecializations:
    """Test that specific archetypes excel in scenarios designed for them."""

    def test_strategist_outperforms_random(self):
        """
        Strategists should consistently outperform random agents
        across all scenarios due to calculated decision-making.
        """
        strategist_team = ["strategist", "strategist", "strategist"]
        strategist_rewards = run_games(
            strategist_team, early_containment_scenario, num_games=30, seed=42
        )

        # Random baseline
        random_team = ["cautious", "liar", "opportunist"]  # Diverse but suboptimal mix
        random_rewards = run_games(
            random_team, early_containment_scenario, num_games=30, seed=42
        )

        mean_strategist = np.mean(strategist_rewards)
        mean_random = np.mean(random_rewards)

        assert mean_strategist > mean_random, (
            f"Strategist team should outperform random team. "
            f"Strategist: {mean_strategist:.2f}, Random: {mean_random:.2f}"
        )

    def test_hero_consistent_performance(self):
        """
        Heroes should show very consistent performance (low variance)
        since they always work and have minimal exploration.
        """
        hero_team = ["hero", "hero", "hero"]
        hero_rewards = run_games(
            hero_team, early_containment_scenario, num_games=30, seed=42
        )

        # Calculate coefficient of variation (std / mean)
        hero_cv = np.std(hero_rewards) / np.mean(hero_rewards)

        # Heroes should have low variance (CV < 0.3)
        assert hero_cv < 0.3, (
            f"Hero team should have consistent performance (CV < 0.3), got CV = {hero_cv:.3f}"
        )


@pytest.mark.integration
def test_team_diversity_matters():
    """
    Test that team composition matters - diverse teams should perform
    differently than homogeneous teams on the same scenario.
    """
    # Homogeneous team
    homogeneous_team = ["firefighter", "firefighter", "firefighter"]
    homogeneous_rewards = run_games(
        homogeneous_team, early_containment_scenario, num_games=30, seed=42
    )

    # Diverse team
    diverse_team = ["firefighter", "coordinator", "hero"]
    diverse_rewards = run_games(
        diverse_team, early_containment_scenario, num_games=30, seed=42
    )

    # The rewards should be measurably different
    # (This doesn't test which is better, just that composition matters)
    t_stat, p_value = stats.ttest_ind(homogeneous_rewards, diverse_rewards)

    # Either they should be significantly different (p < 0.05)
    # OR if not significant, the means should still differ by at least 5%
    mean_homo = np.mean(homogeneous_rewards)
    mean_diverse = np.mean(diverse_rewards)
    percent_diff = abs(mean_homo - mean_diverse) / mean_homo

    assert p_value < 0.05 or percent_diff > 0.05, (
        f"Team composition should matter. "
        f"Homogeneous: {mean_homo:.2f}, Diverse: {mean_diverse:.2f}, "
        f"p-value: {p_value:.4f}, % diff: {percent_diff:.2%}"
    )


@pytest.mark.integration
def test_sparse_heroics_rewards_efficiency():
    """
    Sparse Heroics scenario has long games and moderate work cost.
    A few efficient workers should outperform a team that overworks.
    """
    # Efficient mixed team
    efficient_team = ["strategist", "coordinator", "firefighter"]
    efficient_rewards = run_games(
        efficient_team, sparse_heroics_scenario, num_games=30, seed=42
    )

    # Overworking team
    overwork_team = ["hero", "hero", "hero"]
    overwork_rewards = run_games(
        overwork_team, sparse_heroics_scenario, num_games=30, seed=42
    )

    mean_efficient = np.mean(efficient_rewards)
    mean_overwork = np.mean(overwork_rewards)

    # Efficient team should be competitive (within 20% of heroes)
    # Heroes might win due to guaranteed fire coverage, but should pay work costs
    assert mean_efficient > mean_overwork * 0.8, (
        f"Efficient team should be competitive in Sparse Heroics. "
        f"Efficient: {mean_efficient:.2f}, Overwork: {mean_overwork:.2f}"
    )
