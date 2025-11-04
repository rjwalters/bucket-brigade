"""Population management for evolutionary algorithms.

This module provides classes for managing populations of agents during evolution,
including individual genomes, fitness tracking, and lineage recording.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Optional
from uuid import uuid4

import numpy as np


@dataclass
class Individual:
    """Represents a single individual in the evolutionary population.

    Each individual contains:
    - A genome (10-parameter vector for HeuristicAgent)
    - Fitness score (evaluated via tournament performance)
    - Generation number
    - Lineage information (parent IDs)
    - Unique identifier
    """

    genome: np.ndarray  # Shape (10,), dtype float32, range [0, 1]
    generation: int
    id: str = field(default_factory=lambda: str(uuid4()))
    fitness: Optional[float] = None
    parents: Optional[tuple[str, str]] = None  # Parent IDs for lineage tracking

    def __post_init__(self) -> None:
        """Validate genome after initialization."""
        if not isinstance(self.genome, np.ndarray):
            self.genome = np.array(self.genome, dtype=np.float32)

        if self.genome.shape != (10,):
            raise ValueError(f"Genome must have shape (10,), got {self.genome.shape}")

        if not np.all((self.genome >= 0) & (self.genome <= 1)):
            raise ValueError("All genome values must be in range [0, 1]")

    def __eq__(self, other: object) -> bool:
        """Compare individuals by ID to avoid numpy array comparison issues.

        Args:
            other: Another object to compare with

        Returns:
            True if other is an Individual with the same ID
        """
        if not isinstance(other, Individual):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        """Hash by ID for use in sets and dicts.

        Returns:
            Hash of the individual's ID
        """
        return hash(self.id)

    def clone(self, new_generation: Optional[int] = None) -> Individual:
        """Create a copy of this individual for mutation or new generation.

        Args:
            new_generation: Generation number for the clone (defaults to same generation)

        Returns:
            New Individual with copied genome but new ID
        """
        return Individual(
            genome=self.genome.copy(),
            generation=new_generation
            if new_generation is not None
            else self.generation,
            parents=self.parents,
        )

    def to_dict(self) -> dict:
        """Convert individual to dictionary for serialization.

        Returns:
            Dictionary with all individual data
        """
        return {
            "id": self.id,
            "genome": self.genome.tolist(),
            "generation": self.generation,
            "fitness": self.fitness,
            "parents": list(self.parents) if self.parents else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Individual:
        """Create individual from dictionary.

        Args:
            data: Dictionary with individual data

        Returns:
            New Individual instance
        """
        return cls(
            id=data["id"],
            genome=np.array(data["genome"], dtype=np.float32),
            generation=data["generation"],
            fitness=data.get("fitness"),
            parents=tuple(data["parents"]) if data.get("parents") else None,
        )


class Population:
    """Manages a population of individuals for evolutionary algorithms.

    Provides utilities for:
    - Sorting by fitness
    - Selecting top individuals
    - Computing population statistics
    - Tracking diversity
    """

    def __init__(self, individuals: list[Individual]) -> None:
        """Initialize population with a list of individuals.

        Args:
            individuals: List of Individual instances
        """
        self.individuals = individuals

    def __len__(self) -> int:
        """Return population size."""
        return len(self.individuals)

    def __getitem__(self, index: int) -> Individual:
        """Get individual by index."""
        return self.individuals[index]

    def __iter__(self) -> Iterator[Individual]:
        """Iterate over individuals."""
        return iter(self.individuals)

    def sort_by_fitness(self, reverse: bool = True) -> None:
        """Sort population by fitness (descending by default).

        Args:
            reverse: If True, sort descending (best first). If False, ascending.
        """
        # Move individuals without fitness to the end
        self.individuals.sort(
            key=lambda ind: (
                ind.fitness is None,
                -(ind.fitness or 0) if reverse else (ind.fitness or float("inf")),
            )
        )

    def get_best(self, n: int = 1) -> list[Individual]:
        """Get the top N individuals by fitness.

        Args:
            n: Number of top individuals to return

        Returns:
            List of top N individuals (sorted descending by fitness)
        """
        # Filter out individuals without fitness
        evaluated = [ind for ind in self.individuals if ind.fitness is not None]
        evaluated.sort(
            key=lambda ind: ind.fitness if ind.fitness is not None else 0.0,
            reverse=True,
        )
        return evaluated[:n]

    def get_fitness_stats(self) -> dict[str, float]:
        """Compute fitness statistics for the population.

        Returns:
            Dictionary with min, max, mean, std of fitness values
        """
        fitnesses = np.array(
            [ind.fitness for ind in self.individuals if ind.fitness is not None]
        )

        if len(fitnesses) == 0:
            return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}

        return {
            "min": float(np.min(fitnesses)),
            "max": float(np.max(fitnesses)),
            "mean": float(np.mean(fitnesses)),
            "std": float(np.std(fitnesses)),
        }

    def get_diversity(self) -> float:
        """Compute population diversity as average pairwise distance.

        Diversity is measured as the average Euclidean distance between
        all pairs of genomes in the population.

        Returns:
            Average pairwise distance in parameter space
        """
        if len(self.individuals) <= 1:
            return 0.0

        genomes = np.array([ind.genome for ind in self.individuals])

        # Compute pairwise distances
        total_distance = 0.0
        count = 0

        for i in range(len(genomes)):
            for j in range(i + 1, len(genomes)):
                distance = np.linalg.norm(genomes[i] - genomes[j])
                total_distance += distance
                count += 1

        return float(total_distance / count) if count > 0 else 0.0

    def to_dict(self) -> dict:
        """Convert population to dictionary for serialization.

        Returns:
            Dictionary with population data
        """
        return {"individuals": [ind.to_dict() for ind in self.individuals]}

    @classmethod
    def from_dict(cls, data: dict) -> Population:
        """Create population from dictionary.

        Args:
            data: Dictionary with population data

        Returns:
            New Population instance
        """
        individuals = [
            Individual.from_dict(ind_data) for ind_data in data["individuals"]
        ]
        return cls(individuals)


def create_random_individual(
    generation: int = 0, rng: Optional[np.random.Generator] = None
) -> Individual:
    """Create a random individual with genome in [0, 1]^10.

    Args:
        generation: Generation number for the individual
        rng: Random number generator (creates new one if None)

    Returns:
        New Individual with random genome
    """
    if rng is None:
        rng = np.random.default_rng()

    genome = rng.uniform(0, 1, size=10).astype(np.float32)
    return Individual(genome=genome, generation=generation)


def create_random_population(
    size: int, generation: int = 0, seed: Optional[int] = None
) -> Population:
    """Create a population of random individuals.

    Args:
        size: Number of individuals to create
        generation: Generation number for all individuals
        seed: Random seed for reproducibility

    Returns:
        New Population with random individuals
    """
    rng = np.random.default_rng(seed)
    individuals = [
        create_random_individual(generation=generation, rng=rng) for _ in range(size)
    ]
    return Population(individuals)
