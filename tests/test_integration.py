"""
Integration Tests for End-to-End Pipelines.

This module tests complete workflows that span multiple components:
- Evolution pipeline (genetic algorithm end-to-end)
- Nash equilibrium computation pipeline
- Rust-Python integration for full episodes
- Research data pipeline (loading evolved agents, evaluating)

These tests ensure all components work together correctly.
"""

import pytest
import numpy as np
from pathlib import Path
import json

from bucket_brigade.envs.scenarios import (
    easy_scenario,
    trivial_cooperation_scenario,
)
from bucket_brigade.evolution.genetic_algorithm import GeneticAlgorithm, EvolutionConfig
from bucket_brigade.evolution.fitness_rust import RustFitnessEvaluator
from bucket_brigade.evolution.population import Individual
from bucket_brigade.agents.archetypes import FIREFIGHTER_PARAMS, FREE_RIDER_PARAMS
from bucket_brigade.equilibrium.payoff_evaluator import PayoffEvaluator
from bucket_brigade.equilibrium.nash_solver import solve_symmetric_nash


@pytest.mark.integration
@pytest.mark.slow
class TestEvolutionPipeline:
    """Test end-to-end evolution pipeline."""

    def test_evolution_minimal_run(self):
        """Test complete evolution run with minimal parameters (slow: ~30s)."""
        scenario = easy_scenario(num_agents=4)

        # Create genetic algorithm with minimal parameters
        config = EvolutionConfig(
            population_size=10,  # Small population
            num_generations=5,  # Few generations
            seed=42,
        )

        evaluator = RustFitnessEvaluator(
            scenario=scenario,
            games_per_individual=10,
            seed=42,
        )

        ga = GeneticAlgorithm(config=config, fitness_evaluator=evaluator)

        # Initialize population
        ga.initialize_population()

        assert len(ga.population.individuals) == 10
        assert all(ind.fitness is None for ind in ga.population.individuals)

        # Run evolution
        result = ga.evolve()

        # Check result structure
        assert result.best_individual is not None
        assert result.best_individual.fitness is not None
        assert result.best_individual.fitness > 0  # Should achieve positive fitness
        assert len(result.fitness_history) == 5  # Should have 5 generations
        assert result.converged_at is None or isinstance(result.converged_at, int)  # Should have convergence status

        # Check that fitness improved or stayed reasonable
        first_gen_best = result.fitness_history[0]["max"]
        final_gen_best = result.fitness_history[-1]["max"]

        assert final_gen_best > 0, "Final generation should achieve positive fitness"
        assert (
            final_gen_best >= first_gen_best * 0.8
        ), "Fitness should not degrade significantly"

    def test_evolution_produces_valid_genome(self):
        """Test that evolution produces a valid genome."""
        scenario = easy_scenario(num_agents=4)

        config = EvolutionConfig(
            population_size=5,
            num_generations=3,
            seed=42,
        )

        evaluator = RustFitnessEvaluator(
            scenario=scenario,
            games_per_individual=5,
            seed=42,
        )

        ga = GeneticAlgorithm(config=config, fitness_evaluator=evaluator)

        result = ga.evolve()

        # Genome should be valid
        genome = result.best_individual.genome
        assert genome.shape == (10,), "Genome should have 10 parameters"
        assert np.all(genome >= 0.0), "All parameters should be >= 0"
        assert np.all(genome <= 1.0), "All parameters should be <= 1"

    def test_evolution_history_tracking(self):
        """Test that evolution tracks history correctly."""
        scenario = easy_scenario(num_agents=4)

        config = EvolutionConfig(
            population_size=5,
            num_generations=4,
            seed=42,
        )

        evaluator = RustFitnessEvaluator(
            scenario=scenario,
            games_per_individual=5,
            seed=42,
        )

        ga = GeneticAlgorithm(config=config, fitness_evaluator=evaluator)

        result = ga.evolve()

        # History should have one entry per generation
        assert len(result.fitness_history) == 4

        # Each history entry should have required fields
        for gen_data in result.fitness_history:
            assert "min" in gen_data
            assert "max" in gen_data
            assert "mean" in gen_data
            assert "std" in gen_data

        # Check diversity history matches
        assert len(result.diversity_history) == 4


@pytest.mark.integration
class TestNashEquilibriumPipeline:
    """Test end-to-end Nash equilibrium computation pipeline."""

    def test_nash_computation_two_strategies(self):
        """Test Nash equilibrium computation with two pure strategies."""
        scenario = trivial_cooperation_scenario(num_agents=4)

        # Create payoff evaluator
        evaluator = PayoffEvaluator(
            scenario=scenario,
            num_simulations=20,  # Small number for speed
            seed=42,
            parallel=False,
        )

        # Evaluate payoff matrix for two strategies
        strategies = [FIREFIGHTER_PARAMS, FREE_RIDER_PARAMS]
        payoff_matrix = evaluator.evaluate_payoff_matrix(strategies)

        # Check payoff matrix structure
        assert payoff_matrix.shape == (2, 2)
        assert np.all(np.isfinite(payoff_matrix)), "All payoffs should be finite"

        # Compute Nash equilibrium
        distribution = solve_symmetric_nash(payoff_matrix)

        # Check distribution is valid probability distribution
        assert distribution.shape == (2,)
        assert np.all(distribution >= 0.0), "Probabilities should be non-negative"
        assert np.abs(distribution.sum() - 1.0) < 1e-6, "Probabilities should sum to 1"


@pytest.mark.integration
class TestRustPythonIntegration:
    """Test Rust-Python integration for full episode execution."""

    def test_rust_evaluator_full_episode(self):
        """Test that Rust evaluator can run complete episodes."""
        scenario = easy_scenario(num_agents=4)

        evaluator = RustFitnessEvaluator(
            scenario=scenario,
            games_per_individual=10,
            num_workers=1,
            seed=42,
        )

        # Evaluate a genome
        genome = FIREFIGHTER_PARAMS.copy()
        individual = Individual(genome=genome, generation=0)
        fitness = evaluator.evaluate_individual(individual)

        # Should produce valid fitness
        assert isinstance(fitness, (int, float))
        assert np.isfinite(fitness)
        assert fitness > -1000  # Reasonable lower bound
        assert fitness < 2000  # Reasonable upper bound

    def test_rust_evaluator_reproducibility(self):
        """Test that Rust evaluator produces reproducible results."""
        scenario = easy_scenario(num_agents=4)

        # Run twice with same seed
        evaluator1 = RustFitnessEvaluator(
            scenario=scenario,
            games_per_individual=10,
            num_workers=1,
            seed=42,
        )

        evaluator2 = RustFitnessEvaluator(
            scenario=scenario,
            games_per_individual=10,
            num_workers=1,
            seed=42,
        )

        genome = FIREFIGHTER_PARAMS.copy()
        individual1 = Individual(genome=genome, generation=0)
        individual2 = Individual(genome=genome, generation=0)
        fitness1 = evaluator1.evaluate_individual(individual1)
        fitness2 = evaluator2.evaluate_individual(individual2)

        # Should be identical
        assert fitness1 == fitness2, (
            f"Same seed should produce same fitness: {fitness1} != {fitness2}"
        )


@pytest.mark.integration
class TestResearchDataPipeline:
    """Test research data loading and evaluation pipeline."""

    def test_load_evolved_genome_if_exists(self):
        """Test loading evolved genome from disk (if available)."""
        # Try to load a known evolved genome
        genome_path = (
            Path("experiments")
            / "scenarios"
            / "trivial_cooperation"
            / "evolved_v4"
            / "best_agent.json"
        )

        if not genome_path.exists():
            pytest.skip(f"No evolved genome found at {genome_path}")

        # Load genome
        with open(genome_path) as f:
            data = json.load(f)

        # Check structure
        assert "genome" in data
        assert "fitness" in data
        assert "scenario" in data

        genome = np.array(data["genome"])

        # Validate genome
        assert genome.shape == (10,)
        assert np.all(genome >= 0.0)
        assert np.all(genome <= 1.0)

        # Evaluate the genome
        scenario = trivial_cooperation_scenario(num_agents=4)
        evaluator = RustFitnessEvaluator(
            scenario=scenario,
            games_per_individual=20,
            num_workers=1,
            seed=42,
        )

        individual = Individual(genome=genome, generation=0)
        fitness = evaluator.evaluate_individual(individual)

        # Should achieve reasonable fitness (within 50% of recorded fitness)
        recorded_fitness = data["fitness"]
        assert (
            fitness > recorded_fitness * 0.5
        ), f"Genome fitness {fitness} too low compared to recorded {recorded_fitness}"

    def test_scenario_evolution_roundtrip(self):
        """Test complete scenario → evolution → evaluation roundtrip."""
        # Create scenario
        scenario = easy_scenario(num_agents=4)

        # Run minimal evolution
        config = EvolutionConfig(
            population_size=5,
            num_generations=2,
            seed=42,
        )

        evaluator = RustFitnessEvaluator(
            scenario=scenario,
            games_per_individual=5,
            seed=42,
        )

        ga = GeneticAlgorithm(config=config, fitness_evaluator=evaluator)

        result = ga.evolve()
        best_genome = result.best_individual.genome

        # Re-evaluate the best genome
        evaluator = RustFitnessEvaluator(
            scenario=scenario,
            games_per_individual=10,
            num_workers=1,
            seed=42,
        )

        individual = Individual(genome=best_genome, generation=0)
        fitness = evaluator.evaluate_individual(individual)

        # Fitness should be positive and reasonable
        assert fitness > 0, "Best evolved genome should have positive fitness"
        assert (
            fitness <= result.best_individual.fitness * 1.5
        ), "Re-evaluation shouldn't differ by more than 50% (stochasticity)"


@pytest.mark.integration
class TestComponentIntegration:
    """Test that different components integrate correctly."""

    def test_scenario_to_evaluator_to_ga(self):
        """Test scenario → evaluator → GA integration."""
        # Create scenario
        scenario = easy_scenario(num_agents=4)

        # Create evaluator from scenario
        evaluator = RustFitnessEvaluator(
            scenario=scenario,
            games_per_individual=5,
            num_workers=1,
            seed=42,
        )

        # Create GA with evaluator
        config = EvolutionConfig(
            population_size=3,
            num_generations=2,
            seed=42,
        )

        ga = GeneticAlgorithm(config=config, fitness_evaluator=evaluator)

        # Should work without errors
        result = ga.evolve()
        assert result.best_individual.fitness > 0

    def test_evolution_to_nash_pipeline(self):
        """Test evolution → Nash equilibrium pipeline."""
        scenario = trivial_cooperation_scenario(num_agents=4)

        # Step 1: Run evolution to get strategies
        config = EvolutionConfig(
            population_size=5,
            num_generations=3,
            seed=42,
        )

        evaluator = RustFitnessEvaluator(
            scenario=scenario,
            games_per_individual=5,
            seed=42,
        )

        ga = GeneticAlgorithm(config=config, fitness_evaluator=evaluator)

        result = ga.evolve()

        # Get top 2 individuals
        result.final_population.sort_by_fitness()
        strategy1 = result.final_population.individuals[0].genome
        strategy2 = result.final_population.individuals[1].genome

        # Step 2: Compute Nash equilibrium over evolved strategies
        evaluator = PayoffEvaluator(
            scenario=scenario,
            num_simulations=10,
            seed=42,
            parallel=False,
        )

        payoff_matrix = evaluator.evaluate_payoff_matrix([strategy1, strategy2])

        # Step 3: Solve for Nash equilibrium
        distribution = solve_symmetric_nash(payoff_matrix)

        # Should produce valid distribution
        assert distribution.shape == (2,)
        assert np.abs(distribution.sum() - 1.0) < 1e-6


if __name__ == "__main__":
    # Quick manual verification
    print("=" * 80)
    print("Integration Tests")
    print("=" * 80)

    print("\n1. Testing Rust-Python integration...")
    test = TestRustPythonIntegration()
    test.test_rust_evaluator_full_episode()
    print("✓ Rust evaluator works")

    print("\n2. Testing Rust reproducibility...")
    test.test_rust_evaluator_reproducibility()
    print("✓ Rust evaluator is reproducible")

    print("\n3. Testing Nash computation...")
    test_nash = TestNashEquilibriumPipeline()
    test_nash.test_nash_computation_two_strategies()
    print("✓ Nash equilibrium computation works")

    print("\n4. Testing component integration...")
    test_comp = TestComponentIntegration()
    test_comp.test_scenario_to_evaluator_to_ga()
    print("✓ Scenario → Evaluator → GA integration works")

    print("\n" + "=" * 80)
    print("Core integration tests passed!")
    print("Run full suite with: pytest tests/test_integration.py -v")
    print("=" * 80)
