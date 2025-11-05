"""Evolutionary algorithms for agent strategy optimization.

This module provides genetic algorithms for evolving agent parameters
through tournament-based fitness evaluation.

Uses Rust-backed fitness evaluation for 100x speedup when available.
Falls back to Python implementation if Rust module not built.
"""

# Try to import Rust fitness evaluator, fall back to Python if not available
try:
    from .fitness_rust import RustFitnessEvaluator as FitnessEvaluator
except (ImportError, ModuleNotFoundError):
    from .fitness_python import PythonFitnessEvaluator as FitnessEvaluator

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
