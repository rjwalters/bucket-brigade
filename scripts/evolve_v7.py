#!/usr/bin/env python3
"""
V7 Evolution: TRUE Heterogeneous Tournament Training

This implements the ACTUAL V6 plan that was never run:
- Heterogeneous opponent pool (firefighter, free_rider, hero, coordinator)
- Tournament-based fitness (mixed teams, not self-play clones)
- Tests hypothesis: Training against diverse opponents → robust strategies

KEY DIFFERENCE from V6:
- V6 (as run): fitness_type="mean_reward" → homogeneous self-play
- V7 (proper): HeterogeneousEvaluator → mixed opponent teams

This is the experiment we THOUGHT we ran in V6!

See experiments/evolution/V7_PLAN.md for full strategy.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import time

from bucket_brigade.evolution import (
    EvolutionConfig,
    GeneticAlgorithm,
    Population,
)
from bucket_brigade.evolution.heterogeneous_evaluator import (
    HeterogeneousEvaluator,
    create_heterogeneous_evaluator,
)
from bucket_brigade.envs.scenarios import list_scenarios


def progress_callback(generation: int, population: Population, output_dir: Path) -> None:
    """Print evolution progress and save checkpoints."""
    stats = population.get_fitness_stats()
    diversity = population.get_diversity()

    print(
        f"[Gen {generation:4d}] "
        f"Best: {stats['max']:7.2f} | "
        f"Mean: {stats['mean']:7.2f} ± {stats['std']:5.2f} | "
        f"Diversity: {diversity:.3f}"
    )

    # Save checkpoint every 20 generations
    if generation > 0 and generation % 20 == 0:
        checkpoint_path = output_dir / f"checkpoint_gen{generation}.json"
        best_individual = population.get_best(n=1)[0]
        checkpoint_data = {
            "generation": generation,
            "population": population.to_dict(),
            "best_individual": best_individual.to_dict(),
            "fitness_stats": stats,
            "diversity": diversity,
        }

        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint_data, f, indent=2)

        print(f"  → Checkpoint saved: {checkpoint_path.name}")


def save_results(result, output_dir: Path, config: EvolutionConfig, scenario: str) -> None:
    """Save evolution results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save best agent in standard format
    best_agent_path = output_dir / "best_agent.json"
    best_agent_data = {
        "genome": result.best_individual.genome.tolist(),
        "fitness": float(result.best_individual.fitness),
        "generation": result.best_individual.generation,
        "version": "v7",
        "scenario": scenario,
        "fitness_type": "heterogeneous_tournament",  # KEY DIFFERENCE from V6
    }
    with open(best_agent_path, "w") as f:
        json.dump(best_agent_data, f, indent=2)
    print(f"✓ Best agent saved: {best_agent_path}")

    # Save full results
    results_path = output_dir / "evolution_results.json"
    results_data = {
        "version": "v7",
        "scenario": scenario,
        "config": {
            "population_size": config.population_size,
            "num_generations": config.num_generations,
            "mutation_rate": config.mutation_rate,
            "mutation_scale": config.mutation_scale,
            "elite_size": config.elite_size,
            "games_per_individual": config.games_per_individual,
            "fitness_type": "heterogeneous_tournament",  # KEY DIFFERENCE
            "seed": config.seed,
        },
        "best_individual": result.best_individual.to_dict(),
        "final_population": result.final_population.to_dict(),
        "fitness_history": result.fitness_history,
        "diversity_history": result.diversity_history,
        "converged_at": result.converged_at,
    }
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"✓ Full results saved: {results_path}")

    # Save summary log
    log_path = output_dir / "evolution_log.txt"
    with open(log_path, "w") as f:
        f.write("V7 Evolution Results - Heterogeneous Tournament Training\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Scenario: {scenario}\n")
        f.write(f"Population: {config.population_size}\n")
        f.write(f"Generations: {config.num_generations}\n")
        f.write(f"Mutation Rate: {config.mutation_rate}\n")
        f.write(f"Fitness Type: heterogeneous_tournament\n")
        f.write(f"Converged At: {result.converged_at}\n\n")
        f.write("Best Individual:\n")
        f.write(f"  Fitness: {result.best_individual.fitness:.4f}\n")
        f.write(f"  Generation: {result.best_individual.generation}\n")
        f.write(f"  Genome: {result.best_individual.genome.tolist()}\n\n")
        f.write("Final Population Stats:\n")
        final_stats = result.final_population.get_fitness_stats()
        f.write(f"  Best: {final_stats['max']:.4f}\n")
        f.write(f"  Mean: {final_stats['mean']:.4f}\n")
        f.write(f"  Std:  {final_stats['std']:.4f}\n")
        f.write(f"  Diversity: {result.final_population.get_diversity():.4f}\n")
    print(f"✓ Evolution log saved: {log_path}")


def main():
    parser = argparse.ArgumentParser(
        description="V7 Evolution: TRUE Heterogeneous Tournament Training"
    )

    # Core parameters
    parser.add_argument(
        "--scenario",
        type=str,
        required=True,
        help="Scenario name (e.g., chain_reaction, trivial_cooperation)",
    )
    parser.add_argument(
        "--population",
        type=int,
        default=200,
        help="Population size (default: 200)",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=200,
        help="Number of generations (default: 200)",
    )
    parser.add_argument(
        "--mutation-rate",
        type=float,
        default=0.15,
        help="Mutation probability per gene (default: 0.15)",
    )
    parser.add_argument(
        "--games-per-eval",
        type=int,
        default=100,
        help="Games per fitness evaluation (default: 100)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)",
    )

    # Opponent pool configuration
    parser.add_argument(
        "--opponent-types",
        type=str,
        nargs="+",
        default=["firefighter", "free_rider", "hero", "coordinator"],
        help="Opponent types to include (default: all archetypes)",
    )

    # Output
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for results",
    )

    # Optional
    parser.add_argument(
        "--num-agents",
        type=int,
        default=4,
        help="Number of agents per game (default: 4)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed fitness evaluation info",
    )

    args = parser.parse_args()

    # Validate scenario
    available_scenarios = list_scenarios()
    if args.scenario not in available_scenarios:
        print(f"Error: Scenario '{args.scenario}' not found")
        print(f"Available scenarios: {', '.join(available_scenarios)}")
        sys.exit(1)

    print("=" * 80)
    print("V7 Evolution: TRUE Heterogeneous Tournament Training")
    print("=" * 80)
    print(f"Scenario: {args.scenario}")
    print(f"Population: {args.population}")
    print(f"Generations: {args.generations}")
    print(f"Mutation Rate: {args.mutation_rate}")
    print(f"Games per eval: {args.games_per_eval}")
    print(f"Opponent types: {', '.join(args.opponent_types)}")
    print(f"Workers: {args.workers}")
    print(f"Output: {args.output}")
    print(f"Seed: {args.seed if args.seed else 'random'}")
    print()
    print("KEY DIFFERENCE from V6:")
    print("  V6: Homogeneous self-play (mean_reward)")
    print("  V7: Heterogeneous opponents (tournament)")
    print()

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Create heterogeneous evaluator
    print(f"Creating heterogeneous evaluator...")
    print(f"  Opponent pool: {args.opponent_types}")
    evaluator = create_heterogeneous_evaluator(
        scenario_name=args.scenario,
        num_agents=args.num_agents,
        opponent_types=args.opponent_types,
        seed=args.seed,
    )

    # Test evaluator on random genome
    if args.verbose:
        print("\nTesting evaluator on random genome...")
        import numpy as np
        test_genome = np.random.rand(10)
        test_fitness = evaluator.evaluate(test_genome, num_games=5, verbose=True)
        print(f"Test fitness: {test_fitness:.2f}")
        print()

    # Create evolution configuration
    config = EvolutionConfig(
        population_size=args.population,
        num_generations=args.generations,
        elite_size=10,
        selection_strategy="tournament",
        tournament_size=5,
        crossover_strategy="uniform",
        crossover_rate=0.7,
        mutation_strategy="gaussian",
        mutation_rate=args.mutation_rate,
        mutation_scale=0.1,
        fitness_type="heterogeneous_tournament",  # Document this!
        games_per_individual=args.games_per_eval,
        parallel=(args.workers > 1),
        num_workers=args.workers if args.workers > 1 else None,
        maintain_diversity=True,
        min_diversity=0.1,
        early_stopping=False,  # Run full generations
        seed=args.seed,
    )

    print("Evolution Configuration:")
    print(f"  Elite size: {config.elite_size}")
    print(f"  Tournament size: {config.tournament_size}")
    print(f"  Games per individual: {config.games_per_individual}")
    print(f"  Parallel: {config.parallel}")
    print(f"  Early stopping: {config.early_stopping}")
    print(f"  Maintain diversity: {config.maintain_diversity}")
    print()

    # Create genetic algorithm with HeterogeneousEvaluator
    # The evaluator implements evaluate_individual(individual) -> float
    # which is the interface expected by GeneticAlgorithm
    ga = GeneticAlgorithm(config, fitness_evaluator=evaluator)

    # Run evolution with progress callback
    print("Starting evolution...")
    print()
    start_time = time.time()

    result = ga.evolve(progress_callback=lambda gen, pop: progress_callback(gen, pop, args.output))

    elapsed_time = time.time() - start_time

    # Print summary
    print()
    print("=" * 80)
    print("Evolution Complete!")
    print("=" * 80)
    print(f"Best Fitness: {result.best_individual.fitness:.4f}")
    print(f"Best Generation: {result.best_individual.generation}")
    print(f"Converged At: {result.converged_at}")
    print(f"Elapsed Time: {elapsed_time:.1f}s ({elapsed_time/60:.1f}min)")
    print()

    # Save results
    save_results(result, args.output, config, args.scenario)

    print()
    print("✓ V7 Evolution complete!")
    print(f"✓ Results saved to: {args.output}")
    print()
    print("Next steps:")
    print("  1. Run tournament validation:")
    print(f"     uv run python experiments/scripts/run_heterogeneous_tournament.py \\")
    print(f"       --agents evolved_v7 evolved_v6 firefighter hero free_rider coordinator \\")
    print(f"       --scenarios {args.scenario} --num-games 100")
    print()
    print("  2. Compare V7 vs V6:")
    print(f"     Check if V7 ranks better against free_rider!")
    print()


if __name__ == "__main__":
    main()
