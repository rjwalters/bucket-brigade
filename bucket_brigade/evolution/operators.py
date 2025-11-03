"""Genetic operators for evolutionary algorithms.

This module provides standard genetic algorithm operators:
- Selection (tournament, roulette wheel, rank-based)
- Crossover (uniform, single-point, arithmetic)
- Mutation (Gaussian, uniform)
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .population import Individual, Population


# ============================================================================
# Selection Operators
# ============================================================================


def tournament_selection(
    population: Population,
    tournament_size: int = 3,
    num_parents: int = 2,
    rng: Optional[np.random.Generator] = None,
) -> list[Individual]:
    """Select parents using tournament selection.

    Tournament selection:
    1. Randomly sample K individuals (tournament size)
    2. Select the best from the tournament
    3. Repeat until we have enough parents

    Args:
        population: Population to select from
        tournament_size: Number of individuals in each tournament
        num_parents: Number of parents to select
        rng: Random number generator

    Returns:
        List of selected parent individuals
    """
    if rng is None:
        rng = np.random.default_rng()

    # Filter to only evaluated individuals
    evaluated = [ind for ind in population if ind.fitness is not None]

    if len(evaluated) < tournament_size:
        raise ValueError(f"Population has only {len(evaluated)} evaluated individuals, need at least {tournament_size}")

    parents = []
    for _ in range(num_parents):
        # Randomly sample tournament_size individuals
        tournament = rng.choice(evaluated, size=tournament_size, replace=False)

        # Select the best from the tournament
        winner = max(tournament, key=lambda ind: ind.fitness or 0)
        parents.append(winner)

    return parents


def roulette_selection(
    population: Population, num_parents: int = 2, rng: Optional[np.random.Generator] = None
) -> list[Individual]:
    """Select parents using roulette wheel selection (fitness-proportional).

    Roulette wheel selection:
    - Probability of selection proportional to fitness
    - Higher fitness = higher chance of selection

    Args:
        population: Population to select from
        num_parents: Number of parents to select
        rng: Random number generator

    Returns:
        List of selected parent individuals
    """
    if rng is None:
        rng = np.random.default_rng()

    # Filter to only evaluated individuals
    evaluated = [ind for ind in population if ind.fitness is not None]

    if len(evaluated) == 0:
        raise ValueError("No evaluated individuals in population")

    # Get fitness values and shift to positive (add min if negative)
    fitnesses = np.array([ind.fitness for ind in evaluated])
    min_fitness = np.min(fitnesses)
    if min_fitness < 0:
        fitnesses = fitnesses - min_fitness + 1e-6  # Shift to positive

    # Normalize to probabilities
    total_fitness = np.sum(fitnesses)
    if total_fitness == 0:
        # All fitness values are zero, use uniform selection
        probabilities = np.ones(len(evaluated)) / len(evaluated)
    else:
        probabilities = fitnesses / total_fitness

    # Select parents according to probabilities
    indices = rng.choice(len(evaluated), size=num_parents, p=probabilities, replace=True)
    parents = [evaluated[i] for i in indices]

    return parents


def rank_selection(
    population: Population, num_parents: int = 2, rng: Optional[np.random.Generator] = None
) -> list[Individual]:
    """Select parents using rank-based selection.

    Rank selection:
    - Sort by fitness and assign selection probability based on rank
    - Less sensitive to fitness differences than roulette wheel
    - Prevents premature convergence when fitness values are very different

    Args:
        population: Population to select from
        num_parents: Number of parents to select
        rng: Random number generator

    Returns:
        List of selected parent individuals
    """
    if rng is None:
        rng = np.random.default_rng()

    # Filter to only evaluated individuals
    evaluated = [ind for ind in population if ind.fitness is not None]

    if len(evaluated) == 0:
        raise ValueError("No evaluated individuals in population")

    # Sort by fitness (ascending)
    sorted_individuals = sorted(evaluated, key=lambda ind: ind.fitness or 0)

    # Assign probabilities based on rank (linear ranking)
    # Rank 1 (worst) = 1, Rank 2 = 2, ..., Rank N (best) = N
    ranks = np.arange(1, len(sorted_individuals) + 1)
    probabilities = ranks / np.sum(ranks)

    # Select parents according to rank-based probabilities
    indices = rng.choice(len(sorted_individuals), size=num_parents, p=probabilities, replace=True)
    parents = [sorted_individuals[i] for i in indices]

    return parents


# ============================================================================
# Crossover Operators
# ============================================================================


def uniform_crossover(
    parent1: Individual, parent2: Individual, generation: int, rng: Optional[np.random.Generator] = None
) -> tuple[Individual, Individual]:
    """Perform uniform crossover between two parents.

    Uniform crossover:
    - For each gene, randomly choose from parent1 or parent2
    - Each child gets a random mix of parent genes

    Args:
        parent1: First parent
        parent2: Second parent
        generation: Generation number for offspring
        rng: Random number generator

    Returns:
        Tuple of two offspring individuals
    """
    if rng is None:
        rng = np.random.default_rng()

    # Create mask: True = take from parent1, False = take from parent2
    mask = rng.random(10) < 0.5

    # Create offspring genomes
    child1_genome = np.where(mask, parent1.genome, parent2.genome)
    child2_genome = np.where(mask, parent2.genome, parent1.genome)

    # Create offspring individuals
    child1 = Individual(genome=child1_genome.copy(), generation=generation, parents=(parent1.id, parent2.id))

    child2 = Individual(genome=child2_genome.copy(), generation=generation, parents=(parent1.id, parent2.id))

    return child1, child2


def single_point_crossover(
    parent1: Individual, parent2: Individual, generation: int, rng: Optional[np.random.Generator] = None
) -> tuple[Individual, Individual]:
    """Perform single-point crossover between two parents.

    Single-point crossover:
    - Choose random point in genome
    - Child1: genes 0:point from parent1, point:end from parent2
    - Child2: genes 0:point from parent2, point:end from parent1

    Args:
        parent1: First parent
        parent2: Second parent
        generation: Generation number for offspring
        rng: Random number generator

    Returns:
        Tuple of two offspring individuals
    """
    if rng is None:
        rng = np.random.default_rng()

    # Choose crossover point (1 to 9, so we always mix genes)
    point = rng.integers(1, 10)

    # Create offspring genomes
    child1_genome = np.concatenate([parent1.genome[:point], parent2.genome[point:]])
    child2_genome = np.concatenate([parent2.genome[:point], parent1.genome[point:]])

    # Create offspring individuals
    child1 = Individual(genome=child1_genome.copy(), generation=generation, parents=(parent1.id, parent2.id))

    child2 = Individual(genome=child2_genome.copy(), generation=generation, parents=(parent1.id, parent2.id))

    return child1, child2


def arithmetic_crossover(
    parent1: Individual,
    parent2: Individual,
    generation: int,
    alpha: float = 0.5,
    rng: Optional[np.random.Generator] = None,
) -> tuple[Individual, Individual]:
    """Perform arithmetic crossover (blending) between two parents.

    Arithmetic crossover:
    - Child1 = alpha * parent1 + (1 - alpha) * parent2
    - Child2 = alpha * parent2 + (1 - alpha) * parent1

    Args:
        parent1: First parent
        parent2: Second parent
        generation: Generation number for offspring
        alpha: Blending coefficient (0.5 = equal blend)
        rng: Random number generator (unused, for API consistency)

    Returns:
        Tuple of two offspring individuals
    """
    # Blend genomes
    child1_genome = alpha * parent1.genome + (1 - alpha) * parent2.genome
    child2_genome = alpha * parent2.genome + (1 - alpha) * parent1.genome

    # Ensure values stay in [0, 1]
    child1_genome = np.clip(child1_genome, 0, 1)
    child2_genome = np.clip(child2_genome, 0, 1)

    # Create offspring individuals
    child1 = Individual(genome=child1_genome.copy(), generation=generation, parents=(parent1.id, parent2.id))

    child2 = Individual(genome=child2_genome.copy(), generation=generation, parents=(parent1.id, parent2.id))

    return child1, child2


# ============================================================================
# Mutation Operators
# ============================================================================


def gaussian_mutation(
    individual: Individual, mutation_rate: float = 0.1, mutation_scale: float = 0.1, rng: Optional[np.random.Generator] = None
) -> Individual:
    """Apply Gaussian mutation to an individual.

    Gaussian mutation:
    - For each gene, with probability mutation_rate, add Gaussian noise
    - Noise ~ N(0, mutation_scale)
    - Clip to [0, 1] range

    Args:
        individual: Individual to mutate
        mutation_rate: Probability of mutating each gene
        mutation_scale: Standard deviation of Gaussian noise
        rng: Random number generator

    Returns:
        Mutated individual (original is not modified)
    """
    if rng is None:
        rng = np.random.default_rng()

    # Clone the individual
    mutated = individual.clone(new_generation=individual.generation)

    # Determine which genes to mutate
    mutate_mask = rng.random(10) < mutation_rate

    # Add Gaussian noise to selected genes
    noise = rng.normal(0, mutation_scale, size=10)
    mutated.genome = mutated.genome + mutate_mask * noise

    # Clip to valid range
    mutated.genome = np.clip(mutated.genome, 0, 1)

    return mutated


def uniform_mutation(
    individual: Individual, mutation_rate: float = 0.1, rng: Optional[np.random.Generator] = None
) -> Individual:
    """Apply uniform mutation to an individual.

    Uniform mutation:
    - For each gene, with probability mutation_rate, replace with random value in [0, 1]

    Args:
        individual: Individual to mutate
        mutation_rate: Probability of mutating each gene
        rng: Random number generator

    Returns:
        Mutated individual (original is not modified)
    """
    if rng is None:
        rng = np.random.default_rng()

    # Clone the individual
    mutated = individual.clone(new_generation=individual.generation)

    # Determine which genes to mutate
    mutate_mask = rng.random(10) < mutation_rate

    # Replace selected genes with random values
    random_values = rng.uniform(0, 1, size=10)
    mutated.genome = np.where(mutate_mask, random_values, mutated.genome)

    return mutated


def adaptive_mutation(
    individual: Individual,
    generation: int,
    max_generations: int,
    initial_rate: float = 0.3,
    final_rate: float = 0.05,
    mutation_scale: float = 0.1,
    rng: Optional[np.random.Generator] = None,
) -> Individual:
    """Apply Gaussian mutation with adaptive rate that decreases over generations.

    Adaptive mutation:
    - High mutation rate early (exploration)
    - Low mutation rate late (exploitation)
    - Linear decrease from initial_rate to final_rate

    Args:
        individual: Individual to mutate
        generation: Current generation number
        max_generations: Total number of generations
        initial_rate: Mutation rate at generation 0
        final_rate: Mutation rate at final generation
        mutation_scale: Standard deviation of Gaussian noise
        rng: Random number generator

    Returns:
        Mutated individual (original is not modified)
    """
    # Compute adaptive mutation rate
    progress = generation / max(max_generations, 1)
    current_rate = initial_rate + (final_rate - initial_rate) * progress

    return gaussian_mutation(individual, mutation_rate=current_rate, mutation_scale=mutation_scale, rng=rng)
