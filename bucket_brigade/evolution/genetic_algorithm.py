"""Main genetic algorithm implementation for agent evolution.

This module provides the GeneticAlgorithm class that orchestrates the
evolutionary process: initialization, selection, crossover, mutation,
and fitness evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional

import numpy as np

from . import (
    FitnessEvaluator,
)  # Import from package __init__ which handles Rust/Python fallback
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


@dataclass
class EvolutionConfig:
    """Configuration for evolutionary algorithm."""

    population_size: int = 50
    num_generations: int = 20
    elite_size: int = 5

    # Selection
    selection_strategy: Literal["tournament", "roulette", "rank"] = "tournament"
    tournament_size: int = 3

    # Crossover
    crossover_strategy: Literal["uniform", "single_point", "arithmetic"] = "uniform"
    crossover_rate: float = 0.7
    arithmetic_alpha: float = 0.5

    # Mutation
    mutation_strategy: Literal["gaussian", "uniform", "adaptive"] = "gaussian"
    mutation_rate: float = 0.1
    mutation_scale: float = 0.1
    adaptive_initial_rate: float = 0.3
    adaptive_final_rate: float = 0.05

    # Fitness
    fitness_type: str = "mean_reward"
    games_per_individual: int = 20

    # Parallelization
    parallel: bool = True
    num_workers: Optional[int] = None  # None = cpu_count()

    # Diversity
    maintain_diversity: bool = True
    min_diversity: float = 0.1

    # Early stopping
    early_stopping: bool = True
    convergence_generations: int = 5
    convergence_threshold: float = 0.01

    # Random seed
    seed: Optional[int] = None


@dataclass
class EvolutionResult:
    """Results from an evolution run."""

    best_individual: Individual
    final_population: Population
    fitness_history: list[
        dict[str, float]
    ]  # [{"min": ..., "max": ..., "mean": ..., "std": ...}, ...]
    diversity_history: list[float]
    converged_at: Optional[int] = None


class GeneticAlgorithm:
    """Genetic algorithm for evolving agent strategies.

    The GA follows this process:
    1. Initialize population (random or from seed individuals)
    2. Evaluate fitness of all individuals
    3. For each generation:
       a. Select parents based on fitness
       b. Create offspring via crossover
       c. Apply mutation to offspring
       d. Keep elite individuals
       e. Form new population
       f. Evaluate new individuals
       g. Track progress and check convergence
    """

    def __init__(
        self, config: EvolutionConfig, fitness_evaluator: Optional[Any] = None
    ) -> None:
        """Initialize genetic algorithm.

        Args:
            config: Evolution configuration
            fitness_evaluator: Optional custom fitness evaluator (uses default if None)
        """
        self.config = config
        self.rng = np.random.default_rng(config.seed)

        # Create fitness evaluator (use provided or create default)
        if fitness_evaluator is not None:
            self.fitness_evaluator = fitness_evaluator
        else:
            self.fitness_evaluator = FitnessEvaluator(
                scenario=None,  # Use default scenario
                games_per_individual=config.games_per_individual,
                seed=int(self.rng.integers(0, 2**31))
                if config.seed is not None
                else None,
                num_workers=config.num_workers,
            )

        # History tracking
        self.fitness_history: list[dict[str, float]] = []
        self.diversity_history: list[float] = []
        self.generation = 0

    def initialize_population(
        self, seed_individuals: Optional[list[Individual]] = None
    ) -> Population:
        """Initialize population randomly or from seed individuals.

        Args:
            seed_individuals: Optional list of individuals to seed population

        Returns:
            Initial population
        """
        if seed_individuals is None or len(seed_individuals) == 0:
            # Random initialization
            return create_random_population(
                size=self.config.population_size,
                generation=0,
                seed=int(self.rng.integers(0, 2**31)),
            )

        # Use seed individuals and fill rest with random
        individuals = [ind.clone(new_generation=0) for ind in seed_individuals]

        while len(individuals) < self.config.population_size:
            individuals.append(create_random_individual(generation=0, rng=self.rng))

        # If too many seed individuals, keep only the best
        if len(individuals) > self.config.population_size:
            # Evaluate if needed
            for ind in individuals:
                if ind.fitness is None:
                    ind.fitness = self.fitness_evaluator.evaluate_individual(ind)

            # Sort and keep best
            individuals.sort(key=lambda ind: ind.fitness or 0, reverse=True)
            individuals = individuals[: self.config.population_size]

        return Population(individuals)

    def select_parents(self, population: Population) -> list[Individual]:
        """Select parents for breeding.

        Args:
            population: Current population

        Returns:
            List of selected parents (2 * number of offspring to create)
        """
        # Number of offspring = population size - elite size
        num_offspring = self.config.population_size - self.config.elite_size

        # Need pairs of parents, so select 2 * num_offspring
        # (we'll create num_offspring from pairs)
        num_parents_needed = (
            num_offspring  # Each pair creates 2 children, we'll keep num_offspring
        )

        # Adjust to ensure even number
        if num_parents_needed % 2 != 0:
            num_parents_needed += 1

        parents = []

        if self.config.selection_strategy == "tournament":
            parents = tournament_selection(
                population,
                self.config.tournament_size,
                num_parents_needed,
                rng=self.rng,
            )

        elif self.config.selection_strategy == "roulette":
            parents = roulette_selection(population, num_parents_needed, rng=self.rng)

        elif self.config.selection_strategy == "rank":
            parents = rank_selection(population, num_parents_needed, rng=self.rng)

        return parents

    def crossover(
        self, parent1: Individual, parent2: Individual, generation: int
    ) -> tuple[Individual, Individual]:
        """Perform crossover between two parents.

        Args:
            parent1: First parent
            parent2: Second parent
            generation: Current generation number

        Returns:
            Two offspring individuals
        """
        # Apply crossover with probability crossover_rate
        if self.rng.random() < self.config.crossover_rate:
            if self.config.crossover_strategy == "uniform":
                return uniform_crossover(parent1, parent2, generation, rng=self.rng)

            elif self.config.crossover_strategy == "single_point":
                return single_point_crossover(
                    parent1, parent2, generation, rng=self.rng
                )

            elif self.config.crossover_strategy == "arithmetic":
                return arithmetic_crossover(
                    parent1,
                    parent2,
                    generation,
                    alpha=self.config.arithmetic_alpha,
                    rng=self.rng,
                )

        # No crossover: return clones of parents
        return parent1.clone(new_generation=generation), parent2.clone(
            new_generation=generation
        )

    def mutate(self, individual: Individual) -> Individual:
        """Apply mutation to an individual.

        Args:
            individual: Individual to mutate

        Returns:
            Mutated individual
        """
        if self.config.mutation_strategy == "gaussian":
            return gaussian_mutation(
                individual,
                mutation_rate=self.config.mutation_rate,
                mutation_scale=self.config.mutation_scale,
                rng=self.rng,
            )

        elif self.config.mutation_strategy == "uniform":
            return uniform_mutation(
                individual, mutation_rate=self.config.mutation_rate, rng=self.rng
            )

        elif self.config.mutation_strategy == "adaptive":
            return adaptive_mutation(
                individual,
                generation=self.generation,
                max_generations=self.config.num_generations,
                initial_rate=self.config.adaptive_initial_rate,
                final_rate=self.config.adaptive_final_rate,
                mutation_scale=self.config.mutation_scale,
                rng=self.rng,
            )

        return individual

    def create_next_generation(self, population: Population) -> Population:
        """Create next generation through selection, crossover, and mutation.

        Args:
            population: Current population

        Returns:
            Next generation population
        """
        # Sort population by fitness (best first)
        population.sort_by_fitness(reverse=True)

        # Keep elite individuals
        elite = population.get_best(self.config.elite_size)

        # Select parents and create offspring
        parents = self.select_parents(population)
        offspring = []

        for i in range(0, len(parents) - 1, 2):
            # Crossover
            child1, child2 = self.crossover(
                parents[i], parents[i + 1], self.generation + 1
            )

            # Mutation
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)

            offspring.extend([child1, child2])

        # Ensure we have exactly population_size - elite_size offspring
        offspring = offspring[: self.config.population_size - self.config.elite_size]

        # Combine elite and offspring
        new_individuals = [
            ind.clone(new_generation=self.generation + 1) for ind in elite
        ] + offspring

        return Population(new_individuals)

    def check_convergence(self) -> bool:
        """Check if evolution has converged.

        Convergence = fitness improvement < threshold for N generations

        Returns:
            True if converged
        """
        if len(self.fitness_history) < self.config.convergence_generations + 1:
            return False

        # Get recent fitness history
        recent = self.fitness_history[-self.config.convergence_generations - 1 :]

        # Compute fitness improvement over last N generations
        initial_max = recent[0]["max"]
        final_max = recent[-1]["max"]

        improvement = final_max - initial_max

        # Check if improvement is below threshold
        return improvement < self.config.convergence_threshold

    def evolve(
        self,
        seed_individuals: Optional[list[Individual]] = None,
        progress_callback: Optional[Callable[[int, Population], None]] = None,
    ) -> EvolutionResult:
        """Run the evolutionary algorithm.

        Args:
            seed_individuals: Optional initial individuals to seed population
            progress_callback: Optional callback(generation, population) called each generation

        Returns:
            Evolution results with best individual and history
        """
        # Initialize population
        population = self.initialize_population(seed_individuals)

        # Evaluate initial population
        self.fitness_evaluator.evaluate_population(
            population, parallel=self.config.parallel
        )

        # Track initial stats
        self.fitness_history.append(population.get_fitness_stats())
        self.diversity_history.append(population.get_diversity())

        # Main evolution loop
        converged_at = None

        for gen in range(self.config.num_generations):
            self.generation = gen

            # Progress callback
            if progress_callback is not None:
                progress_callback(gen, population)

            # Create next generation
            population = self.create_next_generation(population)

            # Evaluate new individuals
            self.fitness_evaluator.evaluate_population(
                population, parallel=self.config.parallel
            )

            # Track stats
            self.fitness_history.append(population.get_fitness_stats())
            self.diversity_history.append(population.get_diversity())

            # Diversity maintenance
            if self.config.maintain_diversity:
                diversity = population.get_diversity()
                if diversity < self.config.min_diversity:
                    # Inject random individuals
                    num_inject = max(1, self.config.population_size // 10)
                    for i in range(num_inject):
                        population.individuals[-(i + 1)] = create_random_individual(
                            generation=gen + 1, rng=self.rng
                        )
                        population.individuals[
                            -(i + 1)
                        ].fitness = self.fitness_evaluator.evaluate_individual(
                            population.individuals[-(i + 1)]
                        )

            # Check convergence
            if self.config.early_stopping and self.check_convergence():
                converged_at = gen
                break

        # Final callback
        if progress_callback is not None:
            progress_callback(self.generation, population)

        # Get best individual
        best = population.get_best(1)[0]

        return EvolutionResult(
            best_individual=best,
            final_population=population,
            fitness_history=self.fitness_history,
            diversity_history=self.diversity_history,
            converged_at=converged_at,
        )
