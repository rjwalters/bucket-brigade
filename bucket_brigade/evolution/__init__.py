"""Evolutionary algorithms for agent strategy optimization.

This module provides genetic algorithms for evolving agent parameters
through tournament-based fitness evaluation.
"""

from .fitness import (
    FitnessEvaluator,
    create_fitness_function,
    multi_objective_fitness,
    robustness_fitness,
    win_rate_fitness,
)
from .genetic_algorithm import EvolutionConfig, EvolutionResult, GeneticAlgorithm
from .operators import (
    adaptive_mutation,
    arithmetic_crossover,
    gaussian_mutation,
    rank_selection,
    roulette_selection,
    single_point_crossover,
    tournament_selection,
    uniform_crossover,
    uniform_mutation,
)
from .population import (
    Individual,
    Population,
    create_random_individual,
    create_random_population,
)

__all__ = [
    # Core classes
    "GeneticAlgorithm",
    "EvolutionConfig",
    "EvolutionResult",
    # Population
    "Individual",
    "Population",
    "create_random_individual",
    "create_random_population",
    # Fitness
    "FitnessEvaluator",
    "create_fitness_function",
    "win_rate_fitness",
    "robustness_fitness",
    "multi_objective_fitness",
    # Operators
    "tournament_selection",
    "roulette_selection",
    "rank_selection",
    "uniform_crossover",
    "single_point_crossover",
    "arithmetic_crossover",
    "gaussian_mutation",
    "uniform_mutation",
    "adaptive_mutation",
]
