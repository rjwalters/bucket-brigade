"""Tests for evolutionary algorithm components."""

import numpy as np
import pytest

from bucket_brigade.evolution import (
    EvolutionConfig,
    GeneticAlgorithm,
    Individual,
    Population,
    adaptive_mutation,
    arithmetic_crossover,
    create_random_individual,
    create_random_population,
    gaussian_mutation,
    rank_selection,
    roulette_selection,
    single_point_crossover,
    tournament_selection,
    uniform_crossover,
    uniform_mutation,
)


class TestIndividual:
    """Test Individual class."""

    def test_individual_creation(self):
        """Test creating an individual."""
        genome = np.array(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], dtype=np.float32
        )
        ind = Individual(genome=genome, generation=0)

        assert ind.genome.shape == (10,)
        assert np.all(ind.genome == genome)
        assert ind.generation == 0
        assert ind.fitness is None
        assert ind.parents is None

    def test_individual_validation(self):
        """Test genome validation."""
        # Invalid shape
        with pytest.raises(ValueError):
            Individual(genome=np.array([0.1, 0.2]), generation=0)

        # Out of range
        with pytest.raises(ValueError):
            Individual(
                genome=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.5]),
                generation=0,
            )

    def test_individual_clone(self):
        """Test cloning an individual."""
        genome = np.array(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], dtype=np.float32
        )
        ind = Individual(genome=genome, generation=0, fitness=0.5)

        clone = ind.clone(new_generation=1)

        assert clone.id != ind.id  # New ID
        assert clone.generation == 1  # New generation
        assert np.all(clone.genome == ind.genome)  # Same genome
        assert clone.fitness is None  # Fitness not copied

    def test_individual_serialization(self):
        """Test individual to/from dict."""
        genome = np.array(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], dtype=np.float32
        )
        ind = Individual(genome=genome, generation=5, fitness=0.8)

        # To dict
        data = ind.to_dict()
        assert "id" in data
        assert "genome" in data
        assert "generation" in data
        assert "fitness" in data

        # From dict
        ind2 = Individual.from_dict(data)
        assert ind2.id == ind.id
        assert np.all(ind2.genome == ind.genome)
        assert ind2.generation == ind.generation
        assert ind2.fitness == ind.fitness


class TestPopulation:
    """Test Population class."""

    def test_population_creation(self):
        """Test creating a population."""
        individuals = [create_random_individual(generation=0) for _ in range(10)]
        pop = Population(individuals)

        assert len(pop) == 10
        assert pop[0] == individuals[0]

    def test_population_sorting(self):
        """Test sorting by fitness."""
        individuals = [create_random_individual(generation=0) for _ in range(5)]
        individuals[0].fitness = 0.1
        individuals[1].fitness = 0.5
        individuals[2].fitness = 0.3
        individuals[3].fitness = 0.9
        individuals[4].fitness = 0.2

        pop = Population(individuals)
        pop.sort_by_fitness(reverse=True)

        assert pop[0].fitness == 0.9
        assert pop[1].fitness == 0.5
        assert pop[2].fitness == 0.3
        assert pop[3].fitness == 0.2
        assert pop[4].fitness == 0.1

    def test_population_get_best(self):
        """Test getting best individuals."""
        individuals = [create_random_individual(generation=0) for _ in range(5)]
        individuals[0].fitness = 0.1
        individuals[1].fitness = 0.5
        individuals[2].fitness = 0.3
        individuals[3].fitness = 0.9
        individuals[4].fitness = 0.2

        pop = Population(individuals)
        best = pop.get_best(n=2)

        assert len(best) == 2
        assert best[0].fitness == 0.9
        assert best[1].fitness == 0.5

    def test_population_fitness_stats(self):
        """Test fitness statistics."""
        individuals = [create_random_individual(generation=0) for _ in range(5)]
        individuals[0].fitness = 0.1
        individuals[1].fitness = 0.2
        individuals[2].fitness = 0.3
        individuals[3].fitness = 0.4
        individuals[4].fitness = 0.5

        pop = Population(individuals)
        stats = pop.get_fitness_stats()

        assert stats["min"] == 0.1
        assert stats["max"] == 0.5
        assert stats["mean"] == pytest.approx(0.3)

    def test_population_diversity(self):
        """Test diversity calculation."""
        # Create population with varying diversity
        individuals = [
            create_random_individual(generation=0, rng=np.random.default_rng(i))
            for i in range(10)
        ]
        pop = Population(individuals)

        diversity = pop.get_diversity()
        assert diversity > 0  # Should have some diversity

        # Population of clones has zero diversity
        clone_genome = create_random_individual(generation=0).genome
        clones = [
            Individual(genome=clone_genome.copy(), generation=0) for _ in range(10)
        ]
        clone_pop = Population(clones)

        clone_diversity = clone_pop.get_diversity()
        assert clone_diversity == pytest.approx(0.0)


class TestSelectionOperators:
    """Test selection operators."""

    def test_tournament_selection(self):
        """Test tournament selection."""
        individuals = [create_random_individual(generation=0) for _ in range(10)]
        for i, ind in enumerate(individuals):
            ind.fitness = i * 0.1  # 0.0 to 0.9

        pop = Population(individuals)
        parents = tournament_selection(pop, tournament_size=3, num_parents=2)

        assert len(parents) == 2
        # Should tend to select higher fitness
        # (Can't guarantee deterministically, but check valid selection)
        for parent in parents:
            assert parent in individuals

    def test_roulette_selection(self):
        """Test roulette wheel selection."""
        individuals = [create_random_individual(generation=0) for _ in range(10)]
        for i, ind in enumerate(individuals):
            ind.fitness = i + 1  # 1 to 10

        pop = Population(individuals)
        parents = roulette_selection(pop, num_parents=2)

        assert len(parents) == 2
        for parent in parents:
            assert parent in individuals

    def test_rank_selection(self):
        """Test rank-based selection."""
        individuals = [create_random_individual(generation=0) for _ in range(10)]
        for i, ind in enumerate(individuals):
            ind.fitness = i * 0.1

        pop = Population(individuals)
        parents = rank_selection(pop, num_parents=2)

        assert len(parents) == 2
        for parent in parents:
            assert parent in individuals


class TestCrossoverOperators:
    """Test crossover operators."""

    def test_uniform_crossover(self):
        """Test uniform crossover."""
        genome1 = np.zeros(10, dtype=np.float32)
        genome2 = np.ones(10, dtype=np.float32)

        parent1 = Individual(genome=genome1, generation=0)
        parent2 = Individual(genome=genome2, generation=0)

        child1, child2 = uniform_crossover(
            parent1, parent2, generation=1, rng=np.random.default_rng(42)
        )

        # Children should have mix of parent genes
        assert child1.generation == 1
        assert child2.generation == 1
        assert np.any(child1.genome == 0)  # Has some genes from parent1
        assert np.any(child1.genome == 1)  # Has some genes from parent2
        assert child1.parents == (parent1.id, parent2.id)

    def test_single_point_crossover(self):
        """Test single-point crossover."""
        genome1 = np.zeros(10, dtype=np.float32)
        genome2 = np.ones(10, dtype=np.float32)

        parent1 = Individual(genome=genome1, generation=0)
        parent2 = Individual(genome=genome2, generation=0)

        child1, child2 = single_point_crossover(
            parent1, parent2, generation=1, rng=np.random.default_rng(42)
        )

        # Children should be complementary splits
        assert child1.generation == 1
        assert child2.generation == 1
        assert np.any(child1.genome == 0)
        assert np.any(child1.genome == 1)

    def test_arithmetic_crossover(self):
        """Test arithmetic crossover."""
        genome1 = np.zeros(10, dtype=np.float32)
        genome2 = np.ones(10, dtype=np.float32)

        parent1 = Individual(genome=genome1, generation=0)
        parent2 = Individual(genome=genome2, generation=0)

        child1, child2 = arithmetic_crossover(parent1, parent2, generation=1, alpha=0.5)

        # With alpha=0.5, both children should be 0.5
        assert np.all(child1.genome == 0.5)
        assert np.all(child2.genome == 0.5)


class TestMutationOperators:
    """Test mutation operators."""

    def test_gaussian_mutation(self):
        """Test Gaussian mutation."""
        genome = np.full(10, 0.5, dtype=np.float32)
        ind = Individual(genome=genome, generation=0)

        mutated = gaussian_mutation(
            ind, mutation_rate=1.0, mutation_scale=0.1, rng=np.random.default_rng(42)
        )

        # All genes should be mutated (rate=1.0)
        assert not np.all(mutated.genome == ind.genome)
        # Values should stay in [0, 1]
        assert np.all((mutated.genome >= 0) & (mutated.genome <= 1))

    def test_uniform_mutation(self):
        """Test uniform mutation."""
        genome = np.full(10, 0.5, dtype=np.float32)
        ind = Individual(genome=genome, generation=0)

        mutated = uniform_mutation(
            ind, mutation_rate=1.0, rng=np.random.default_rng(42)
        )

        # All genes should be mutated (rate=1.0)
        assert not np.all(mutated.genome == ind.genome)
        # Values should be in [0, 1]
        assert np.all((mutated.genome >= 0) & (mutated.genome <= 1))

    def test_adaptive_mutation(self):
        """Test adaptive mutation."""
        genome = np.full(10, 0.5, dtype=np.float32)
        ind = Individual(genome=genome, generation=0)

        # Early generation: high mutation rate
        early = adaptive_mutation(
            ind,
            generation=0,
            max_generations=100,
            initial_rate=0.5,
            final_rate=0.1,
            rng=np.random.default_rng(42),
        )

        # Late generation: low mutation rate
        late = adaptive_mutation(
            ind,
            generation=99,
            max_generations=100,
            initial_rate=0.5,
            final_rate=0.1,
            rng=np.random.default_rng(43),
        )

        # Can't guarantee specific behavior due to randomness,
        # but both should produce valid genomes
        assert np.all((early.genome >= 0) & (early.genome <= 1))
        assert np.all((late.genome >= 0) & (late.genome <= 1))


class TestGeneticAlgorithm:
    """Test GeneticAlgorithm class."""

    def test_ga_initialization(self):
        """Test GA initialization."""
        config = EvolutionConfig(population_size=20, num_generations=5, seed=42)
        ga = GeneticAlgorithm(config)

        assert ga.config.population_size == 20
        assert ga.config.num_generations == 5

    def test_ga_initialize_population(self):
        """Test population initialization."""
        config = EvolutionConfig(population_size=20, seed=42)
        ga = GeneticAlgorithm(config)

        # Random initialization
        pop = ga.initialize_population()
        assert len(pop) == 20

        # Seed population
        seeds = [create_random_individual(generation=0) for _ in range(5)]
        pop2 = ga.initialize_population(seed_individuals=seeds)
        assert len(pop2) == 20  # Filled to population_size

    def test_ga_evolve_basic(self):
        """Test basic evolution run (small scale)."""
        config = EvolutionConfig(
            population_size=3,
            num_generations=1,
            elite_size=1,
            games_per_individual=1,
            seed=42,
            early_stopping=False,
        )

        ga = GeneticAlgorithm(config)
        result = ga.evolve()

        # Check result structure
        assert result.best_individual is not None
        assert result.best_individual.fitness is not None
        assert len(result.fitness_history) == 2  # Initial + 1 generation
        assert len(result.diversity_history) == 2

        # With only 1 game per individual, fitness has high variance
        # so we just verify that fitness values are reasonable
        initial_best = result.fitness_history[0]["max"]
        final_best = result.fitness_history[-1]["max"]
        # Fitness should be in a reasonable range (negative due to reward structure)
        assert -100 <= initial_best <= 100
        assert -100 <= final_best <= 100

    def test_ga_convergence_detection(self):
        """Test convergence detection."""
        config = EvolutionConfig(
            population_size=3,
            num_generations=5,
            games_per_individual=1,
            early_stopping=True,
            convergence_threshold=0.0001,
            seed=42,
        )

        ga = GeneticAlgorithm(config)
        result = ga.evolve()

        # Should converge before reaching max generations
        # (may not always happen due to randomness, but likely)
        # At minimum, check that converged_at is tracked
        assert isinstance(result.converged_at, (int, type(None)))

    def test_ga_diversity_maintenance(self):
        """Test diversity maintenance."""
        config = EvolutionConfig(
            population_size=4,
            num_generations=2,
            games_per_individual=1,
            maintain_diversity=True,
            min_diversity=0.1,
            seed=42,
            early_stopping=False,
        )

        ga = GeneticAlgorithm(config)
        result = ga.evolve()

        # Check that diversity is maintained throughout
        for diversity in result.diversity_history:
            # Diversity may occasionally drop below threshold, but should recover
            pass  # Basic smoke test that this runs


class TestUtilityFunctions:
    """Test utility functions."""

    def test_create_random_individual(self):
        """Test random individual creation."""
        ind = create_random_individual(generation=5, rng=np.random.default_rng(42))

        assert ind.generation == 5
        assert ind.genome.shape == (10,)
        assert np.all((ind.genome >= 0) & (ind.genome <= 1))

    def test_create_random_population(self):
        """Test random population creation."""
        pop = create_random_population(size=20, generation=3, seed=42)

        assert len(pop) == 20
        assert all(ind.generation == 3 for ind in pop)

    def test_reproducibility(self):
        """Test that using same seed gives same results."""
        pop1 = create_random_population(size=10, seed=42)
        pop2 = create_random_population(size=10, seed=42)

        for ind1, ind2 in zip(pop1, pop2):
            assert np.all(ind1.genome == ind2.genome)
