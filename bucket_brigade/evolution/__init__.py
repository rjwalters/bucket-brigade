"""Evolutionary algorithms for agent strategy optimization.

This module provides genetic algorithms for evolving agent parameters
through tournament-based fitness evaluation.

Requires Rust-backed fitness evaluation for acceptable performance.
Build the Rust module with: cd bucket-brigade-core && ./build.sh
"""

try:
    from .fitness_rust import RustFitnessEvaluator as FitnessEvaluator
except (ImportError, ModuleNotFoundError) as e:
    raise ImportError(
        "Rust fitness evaluator required but not found. "
        "The Python fallback is too slow for practical use (100x slower). "
        "Please build the Rust module:\n\n"
        "  cd bucket-brigade-core && ./build.sh\n\n"
        "See bucket-brigade-core/README.md for troubleshooting (e.g. the "
        "CFFI-shadow trap if you see 'cannot import name PyBucketBrigade')."
    ) from e

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
