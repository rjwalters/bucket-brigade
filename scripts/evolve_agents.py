#!/usr/bin/env python3
"""
Evolution script for discovering optimal agent strategies using genetic algorithms.

This script evolves agent parameters through tournament-based fitness evaluation,
producing high-performing agents that can be saved and deployed.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import time
from typing import Optional

import numpy as np

from bucket_brigade.evolution import (
    EvolutionConfig,
    GeneticAlgorithm,
    Individual,
    Population,
)
from bucket_brigade.envs.scenarios import default_scenario, greedy_neighbor_scenario


def progress_callback(generation: int, population: Population) -> None:
    """Print evolution progress.

    Args:
        generation: Current generation number
        population: Current population
    """
    stats = population.get_fitness_stats()
    diversity = population.get_diversity()

    print(
        f"Gen {generation:3d} | "
        f"Best: {stats['max']:.3f} | "
        f"Mean: {stats['mean']:.3f} | "
        f"Std: {stats['std']:.3f} | "
        f"Diversity: {diversity:.3f}"
    )


def save_results(result, output_path: Path, config: EvolutionConfig) -> None:
    """Save evolution results to file.

    Args:
        result: EvolutionResult instance
        output_path: Path to save results
        config: Evolution configuration
    """
    data = {
        "config": {
            "population_size": config.population_size,
            "num_generations": config.num_generations,
            "elite_size": config.elite_size,
            "selection_strategy": config.selection_strategy,
            "crossover_strategy": config.crossover_strategy,
            "crossover_rate": config.crossover_rate,
            "mutation_strategy": config.mutation_strategy,
            "mutation_rate": config.mutation_rate,
            "mutation_scale": config.mutation_scale,
            "fitness_type": config.fitness_type,
            "games_per_individual": config.games_per_individual,
            "seed": config.seed,
        },
        "best_individual": result.best_individual.to_dict(),
        "final_population": result.final_population.to_dict(),
        "fitness_history": result.fitness_history,
        "diversity_history": result.diversity_history,
        "converged_at": result.converged_at,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to {output_path}")


def load_seed_population(seed_path: Optional[Path]) -> Optional[list[Individual]]:
    """Load seed individuals from file.

    Args:
        seed_path: Path to seed population file

    Returns:
        List of seed individuals or None
    """
    if seed_path is None or not seed_path.exists():
        return None

    with open(seed_path, "r") as f:
        data = json.load(f)

    if "individuals" in data:
        # Population file
        individuals = [
            Individual.from_dict(ind_data) for ind_data in data["individuals"]
        ]
    elif "final_population" in data:
        # Evolution result file
        individuals = [
            Individual.from_dict(ind_data)
            for ind_data in data["final_population"]["individuals"]
        ]
    else:
        raise ValueError(f"Invalid seed file format: {seed_path}")

    print(f"Loaded {len(individuals)} seed individuals from {seed_path}")
    return individuals


def main():
    parser = argparse.ArgumentParser(
        description="Evolve agent strategies using genetic algorithms"
    )

    # Population
    parser.add_argument(
        "--population-size", type=int, default=50, help="Population size (default: 50)"
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=20,
        help="Number of generations (default: 20)",
    )
    parser.add_argument(
        "--elite-size", type=int, default=5, help="Elite size (default: 5)"
    )

    # Selection
    parser.add_argument(
        "--selection",
        type=str,
        default="tournament",
        choices=["tournament", "roulette", "rank"],
        help="Selection strategy (default: tournament)",
    )
    parser.add_argument(
        "--tournament-size",
        type=int,
        default=3,
        help="Tournament size for tournament selection (default: 3)",
    )

    # Crossover
    parser.add_argument(
        "--crossover",
        type=str,
        default="uniform",
        choices=["uniform", "single_point", "arithmetic"],
        help="Crossover strategy (default: uniform)",
    )
    parser.add_argument(
        "--crossover-rate",
        type=float,
        default=0.7,
        help="Crossover probability (default: 0.7)",
    )
    parser.add_argument(
        "--arithmetic-alpha",
        type=float,
        default=0.5,
        help="Blending coefficient for arithmetic crossover (default: 0.5)",
    )

    # Mutation
    parser.add_argument(
        "--mutation",
        type=str,
        default="gaussian",
        choices=["gaussian", "uniform", "adaptive"],
        help="Mutation strategy (default: gaussian)",
    )
    parser.add_argument(
        "--mutation-rate",
        type=float,
        default=0.1,
        help="Mutation probability per gene (default: 0.1)",
    )
    parser.add_argument(
        "--mutation-scale",
        type=float,
        default=0.1,
        help="Mutation scale (std dev for Gaussian) (default: 0.1)",
    )
    parser.add_argument(
        "--adaptive-initial-rate",
        type=float,
        default=0.3,
        help="Initial mutation rate for adaptive mutation (default: 0.3)",
    )
    parser.add_argument(
        "--adaptive-final-rate",
        type=float,
        default=0.05,
        help="Final mutation rate for adaptive mutation (default: 0.05)",
    )

    # Fitness
    parser.add_argument(
        "--fitness-type",
        type=str,
        default="mean_reward",
        choices=["mean_reward", "win_rate", "robustness", "multi_objective"],
        help="Fitness function (default: mean_reward)",
    )
    parser.add_argument(
        "--games-per-individual",
        type=int,
        default=20,
        help="Games per individual for fitness evaluation (default: 20)",
    )

    # Parallelization
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel fitness evaluation (sequential mode)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        help="Number of parallel workers (default: cpu_count)",
    )

    # Diversity
    parser.add_argument(
        "--no-diversity-maintenance",
        action="store_true",
        help="Disable diversity maintenance",
    )
    parser.add_argument(
        "--min-diversity",
        type=float,
        default=0.1,
        help="Minimum diversity threshold (default: 0.1)",
    )

    # Early stopping
    parser.add_argument(
        "--no-early-stopping", action="store_true", help="Disable early stopping"
    )
    parser.add_argument(
        "--convergence-generations",
        type=int,
        default=5,
        help="Generations for convergence check (default: 5)",
    )
    parser.add_argument(
        "--convergence-threshold",
        type=float,
        default=0.01,
        help="Fitness improvement threshold for convergence (default: 0.01)",
    )

    # Seed population
    parser.add_argument(
        "--seed-population", type=Path, help="Path to seed population JSON file"
    )

    # Output
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evolution_results.json"),
        help="Output file path",
    )

    # Random seed
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")

    args = parser.parse_args()

    # Create configuration
    config = EvolutionConfig(
        population_size=args.population_size,
        num_generations=args.generations,
        elite_size=args.elite_size,
        selection_strategy=args.selection,
        tournament_size=args.tournament_size,
        crossover_strategy=args.crossover,
        crossover_rate=args.crossover_rate,
        arithmetic_alpha=args.arithmetic_alpha,
        mutation_strategy=args.mutation,
        mutation_rate=args.mutation_rate,
        mutation_scale=args.mutation_scale,
        adaptive_initial_rate=args.adaptive_initial_rate,
        adaptive_final_rate=args.adaptive_final_rate,
        fitness_type=args.fitness_type,
        games_per_individual=args.games_per_individual,
        parallel=not args.no_parallel,
        num_workers=args.num_workers,
        maintain_diversity=not args.no_diversity_maintenance,
        min_diversity=args.min_diversity,
        early_stopping=not args.no_early_stopping,
        convergence_generations=args.convergence_generations,
        convergence_threshold=args.convergence_threshold,
        seed=args.seed,
    )

    print("Evolution Configuration:")
    print(f"  Population: {config.population_size}")
    print(f"  Generations: {config.num_generations}")
    print(f"  Selection: {config.selection_strategy}")
    print(f"  Crossover: {config.crossover_strategy} (rate={config.crossover_rate})")
    print(
        f"  Mutation: {config.mutation_strategy} (rate={config.mutation_rate}, scale={config.mutation_scale})"
    )
    print(
        f"  Fitness: {config.fitness_type} ({config.games_per_individual} games/individual)"
    )
    parallel_mode = "parallel" if config.parallel else "sequential"
    workers_info = f", {config.num_workers} workers" if config.num_workers else ", auto workers"
    print(f"  Evaluation: {parallel_mode}{workers_info if config.parallel else ''}")
    print(f"  Seed: {config.seed}")
    print()

    # Load seed population if provided
    seed_individuals = load_seed_population(args.seed_population)

    # Create GA and run evolution
    print("Starting evolution...")
    print()

    ga = GeneticAlgorithm(config)

    start_time = time.time()
    result = ga.evolve(
        seed_individuals=seed_individuals, progress_callback=progress_callback
    )
    elapsed_time = time.time() - start_time

    print()
    print("=" * 80)
    print("Evolution Complete!")
    print("=" * 80)
    print(f"Time elapsed: {elapsed_time:.1f}s")
    print(
        f"Converged at generation: {result.converged_at if result.converged_at is not None else 'N/A'}"
    )
    print()
    print("Best Individual:")
    print(f"  Fitness: {result.best_individual.fitness:.4f}")
    print(f"  Generation: {result.best_individual.generation}")
    print(f"  Genome: {result.best_individual.genome}")
    print()
    print("Final Population Stats:")
    final_stats = result.final_population.get_fitness_stats()
    print(f"  Best: {final_stats['max']:.4f}")
    print(f"  Mean: {final_stats['mean']:.4f}")
    print(f"  Std:  {final_stats['std']:.4f}")
    print(f"  Diversity: {result.final_population.get_diversity():.4f}")
    print()

    # Save results
    save_results(result, args.output, config)

    print("\nTo use the best agent:")
    print(f"  from bucket_brigade.agents.heuristic_agent import HeuristicAgent")
    print(f"  params = {result.best_individual.genome.tolist()}")
    print(
        f"  agent = HeuristicAgent(params=np.array(params), agent_id=0, name='Evolved Champion')"
    )


if __name__ == "__main__":
    main()
