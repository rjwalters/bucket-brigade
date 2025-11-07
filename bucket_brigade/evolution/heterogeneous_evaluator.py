"""Heterogeneous tournament fitness evaluator for V7 evolution.

This evaluator trains agents against a diverse opponent pool, including
archetypes like firefighter, free_rider, hero, and coordinator. This is
the key innovation in V7 that was missing from V6.

V6 Problem: Agents evolved against homogeneous teams (all clones)
V7 Solution: Agents evolve against heterogeneous teams (mixed opponents)
"""

from __future__ import annotations

from typing import Optional
import numpy as np
import bucket_brigade_core as core

from bucket_brigade.agents.archetypes import (
    FIREFIGHTER_PARAMS,
    FREE_RIDER_PARAMS,
    HERO_PARAMS,
    COORDINATOR_PARAMS,
)


def _heuristic_action(
    theta: np.ndarray, obs: dict, agent_id: int, rng: np.random.RandomState
) -> list[int]:
    """
    Simplified heuristic action selection based on parameters.

    Fast approximation of HeuristicAgent behavior for fitness evaluation.
    """
    # Unpack key parameters
    work_tendency = theta[1]
    own_house_priority = theta[3]
    rest_reward_bias = theta[8]

    # Simple decision: work with probability based on work_tendency
    if rng.random() < work_tendency * (1 - rest_reward_bias):
        # Work - choose which house
        owned_house = agent_id % 10

        # Prioritize owned house if burning
        if obs["houses"][owned_house] == 1 and rng.random() < own_house_priority:
            house = owned_house
        else:
            # Choose a burning house
            burning = [i for i, h in enumerate(obs["houses"]) if h == 1]
            if burning:
                house = rng.choice(burning)
            else:
                house = owned_house

        mode = 1  # WORK
    else:
        # Rest
        house = agent_id % 10
        mode = 0  # REST

    return [house, mode]


class HeterogeneousEvaluator:
    """Evaluates agent fitness in heterogeneous tournament settings.

    This is the core innovation in V7: agents are evaluated by playing
    games with diverse opponents, forcing them to learn robust strategies
    that work in mixed teams.

    Key difference from V6:
    - V6: team = [candidate, candidate, candidate, candidate]
    - V7: team = [candidate, firefighter, free_rider, hero] (random mix)

    This forces agents to handle:
    - Hard workers (firefighter, hero)
    - Defectors (free_rider) <- KEY!
    - Coordinators (balanced strategies)
    - Self-play (some games with clones)
    """

    # Opponent archetypes (fixed throughout evolution)
    OPPONENT_POOL = {
        "firefighter": FIREFIGHTER_PARAMS,
        "free_rider": FREE_RIDER_PARAMS,
        "hero": HERO_PARAMS,
        "coordinator": COORDINATOR_PARAMS,
    }

    def __init__(
        self,
        scenario_name: str,
        num_agents: int = 4,
        opponent_types: Optional[list[str]] = None,
        seed: Optional[int] = None
    ):
        """Initialize heterogeneous evaluator.

        Args:
            scenario_name: Name of scenario to evaluate on
            num_agents: Number of agents per game (default: 4)
            opponent_types: List of opponent types to include in pool
                (default: all archetypes)
            seed: Random seed for opponent sampling reproducibility
        """
        self.scenario_name = scenario_name
        self.num_agents = num_agents
        self.seed = seed

        # Use all opponents if not specified
        if opponent_types is None:
            opponent_types = list(self.OPPONENT_POOL.keys())

        # Validate opponent types
        invalid = set(opponent_types) - set(self.OPPONENT_POOL.keys())
        if invalid:
            raise ValueError(f"Invalid opponent types: {invalid}")

        self.opponent_types = opponent_types

        # Get Rust scenario (single source of truth)
        self.rust_scenario = core.SCENARIOS[scenario_name]

        # RNG for opponent sampling
        self.rng = np.random.RandomState(seed)

    def evaluate(
        self,
        candidate_genome: np.ndarray,
        num_games: int = 100,
        verbose: bool = False
    ) -> float:
        """Evaluate candidate in heterogeneous tournament.

        Args:
            candidate_genome: Genome to evaluate (10-dimensional)
            num_games: Number of games to run
            verbose: If True, print per-game details

        Returns:
            Mean payoff across all games (from candidate's perspective)
        """
        total_payoff = 0.0

        # Track opponent statistics
        opponent_counts = {op: 0 for op in self.opponent_types}

        for game_idx in range(num_games):
            # Sample teammates (candidate is always agent 0)
            num_teammates = self.num_agents - 1
            teammate_types = self.rng.choice(
                self.opponent_types,
                size=num_teammates,
                replace=True
            )

            # Track opponent usage
            for op in teammate_types:
                opponent_counts[op] += 1

            # Get genomes for all agents
            genomes = [candidate_genome]  # Agent 0 is candidate
            for teammate_type in teammate_types:
                genomes.append(self.OPPONENT_POOL[teammate_type])

            # Run game
            payoff = self._run_game(genomes, game_idx)
            total_payoff += payoff

            if verbose and game_idx < 5:  # Show first 5 games
                print(f"  Game {game_idx}: team={['candidate'] + list(teammate_types)}, payoff={payoff:.2f}")

        mean_payoff = total_payoff / num_games

        if verbose:
            print(f"  Opponent exposure: {opponent_counts}")
            print(f"  Mean payoff: {mean_payoff:.2f}")

        return mean_payoff

    def _run_game(self, genomes: list[np.ndarray], seed: int) -> float:
        """Run single game with heterogeneous team.

        Args:
            genomes: List of genomes (one per agent)
            seed: Random seed for game

        Returns:
            Payoff for agent 0 (candidate)
        """
        # Create Rust game
        game = core.BucketBrigade(
            self.rust_scenario,
            self.num_agents,
            seed=seed
        )

        # Python RNG for heuristic decisions
        rng = np.random.RandomState(seed)

        # Run until done
        done = False
        step_count = 0
        max_steps = 100  # Safety limit

        candidate_total_reward = 0.0

        while not done and step_count < max_steps:
            # Get actions for ALL agents
            actions = []
            for agent_id in range(self.num_agents):
                obs = game.get_observation(agent_id)
                obs_dict = {
                    "houses": obs.houses,
                    "signals": obs.signals,
                    "locations": obs.locations,
                }

                # Get action from heuristic using this agent's genome
                genome = genomes[agent_id]
                action = _heuristic_action(genome, obs_dict, agent_id, rng)
                actions.append(action)

            # Step the Rust game
            rewards, done, info = game.step(actions)

            # Track candidate (agent 0) reward
            candidate_total_reward += rewards[0]

            step_count += 1

        return candidate_total_reward

    def evaluate_batch(
        self,
        genomes: list[np.ndarray],
        num_games: int = 100
    ) -> list[float]:
        """Evaluate multiple candidates (for parallel evolution).

        Args:
            genomes: List of genomes to evaluate
            num_games: Number of games per genome

        Returns:
            List of mean payoffs (one per genome)
        """
        return [self.evaluate(genome, num_games) for genome in genomes]

    def evaluate_individual(self, individual) -> float:
        """Evaluate individual (GeneticAlgorithm interface compatibility).

        Args:
            individual: Individual with genome attribute

        Returns:
            Mean payoff across tournament games
        """
        return self.evaluate(individual.genome, num_games=100)

    def evaluate_population(self, population, parallel: bool = False) -> None:
        """Evaluate all individuals in population (GeneticAlgorithm interface).

        Args:
            population: Population to evaluate
            parallel: Ignored (sequential only for now)

        Note:
            This method modifies individuals in-place by setting their fitness.
        """
        for individual in population:
            if individual.fitness is None:
                individual.fitness = self.evaluate_individual(individual)


def create_heterogeneous_evaluator(
    scenario_name: str,
    num_agents: int = 4,
    opponent_types: Optional[list[str]] = None,
    seed: Optional[int] = None
) -> HeterogeneousEvaluator:
    """Factory function for creating heterogeneous evaluator.

    Args:
        scenario_name: Name of scenario
        num_agents: Agents per game
        opponent_types: Opponent types to include
        seed: Random seed

    Returns:
        Configured HeterogeneousEvaluator
    """
    return HeterogeneousEvaluator(
        scenario_name=scenario_name,
        num_agents=num_agents,
        opponent_types=opponent_types,
        seed=seed
    )
