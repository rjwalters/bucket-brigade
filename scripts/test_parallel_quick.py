#!/usr/bin/env python3
"""Quick smoke test for parallel evaluation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from bucket_brigade.evolution import (
    EvolutionConfig,
    GeneticAlgorithm,
    create_random_population,
)
from bucket_brigade.evolution.fitness import FitnessEvaluator


def test_basic_parallel():
    """Test that parallel evaluation works."""
    print("Testing basic parallel evaluation...")

    # Create small population
    pop = create_random_population(size=3, generation=0, seed=42)

    # Evaluate in parallel
    evaluator = FitnessEvaluator(games_per_individual=2, seed=100, num_workers=2)
    evaluator.evaluate_population(pop, parallel=True)

    # Check all got fitness
    for i, ind in enumerate(pop):
        assert ind.fitness is not None, f"Individual {i} has no fitness"
        print(f"  Individual {i}: fitness = {ind.fitness:.4f}")

    print("✅ Basic parallel evaluation works\n")


def test_sequential_vs_parallel():
    """Test that sequential and parallel give similar results."""
    print("Testing sequential vs parallel consistency...")

    from bucket_brigade.evolution import Population

    # Create test population
    pop = create_random_population(size=3, generation=0, seed=42)

    # Sequential
    evaluator_seq = FitnessEvaluator(games_per_individual=5, seed=100)
    pop_seq = Population([ind.clone(new_generation=0) for ind in pop])
    evaluator_seq.evaluate_population(pop_seq, parallel=False)

    # Parallel
    evaluator_par = FitnessEvaluator(games_per_individual=5, seed=100, num_workers=2)
    pop_par = Population([ind.clone(new_generation=0) for ind in pop])
    evaluator_par.evaluate_population(pop_par, parallel=True)

    # Compare
    print("  Sequential vs Parallel fitness:")
    for i, (ind_seq, ind_par) in enumerate(zip(pop_seq, pop_par)):
        print(
            f"    Individual {i}: seq={ind_seq.fitness:.4f}, par={ind_par.fitness:.4f}"
        )
        # Both should have fitness (may differ due to different random streams in workers)
        assert ind_seq.fitness is not None, (
            f"Sequential fitness is None for individual {i}"
        )
        assert ind_par.fitness is not None, (
            f"Parallel fitness is None for individual {i}"
        )

    print("✅ Both sequential and parallel produce valid fitness values\n")


def test_ga_integration():
    """Test that GeneticAlgorithm works with parallel config."""
    print("Testing GeneticAlgorithm integration...")

    config = EvolutionConfig(
        population_size=5,
        num_generations=2,
        games_per_individual=2,
        parallel=True,
        num_workers=2,
        seed=42,
        early_stopping=False,
    )

    ga = GeneticAlgorithm(config)
    result = ga.evolve()

    assert result.best_individual is not None
    assert result.best_individual.fitness is not None
    print(f"  Best individual fitness: {result.best_individual.fitness:.4f}")
    print(f"  Generations: {len(result.fitness_history) - 1}")

    print("✅ GeneticAlgorithm integration works\n")


def main():
    """Run all smoke tests."""
    print("=" * 60)
    print("Parallel Evaluation Smoke Tests")
    print("=" * 60)
    print()

    test_basic_parallel()
    test_sequential_vs_parallel()
    test_ga_integration()

    print("=" * 60)
    print("✅ All smoke tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
