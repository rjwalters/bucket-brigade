#!/usr/bin/env python3
"""
Optimized evolutionary training script for remote GPU sandbox.

This script is designed to run on remote hardware with many vCPUs (48+),
using Rust-backed parallel fitness evaluation to maximize CPU utilization.

Features:
- Rust-backed fitness evaluation with multiprocessing (100x speedup)
- Configurable worker count to match available vCPUs
- Extended training runs with periodic checkpointing
- Comprehensive logging for remote monitoring
- Headless operation support (no display required)
- Resume from checkpoints for long-running experiments
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import time
from datetime import datetime
from typing import Optional

import numpy as np

from bucket_brigade.evolution import (
    EvolutionConfig,
    GeneticAlgorithm,
    Individual,
    Population,
)

# Try to import Rust evaluator, fall back to Python if unavailable
try:
    from bucket_brigade.evolution.fitness_rust import RustFitnessEvaluator
    RUST_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    print(f"⚠️  Rust evaluator not available ({e}), using Python implementation")
    from bucket_brigade.evolution.fitness import FitnessEvaluator
    RUST_AVAILABLE = False

from bucket_brigade.envs.scenarios import default_scenario


class RemoteEvolutionRunner:
    """Runner for long-running evolutionary experiments on remote hardware."""

    def __init__(
        self,
        config: EvolutionConfig,
        output_dir: Path,
        checkpoint_interval: int = 10,
        num_workers: int = 48,
    ):
        """Initialize remote evolution runner.

        Args:
            config: Evolution configuration
            output_dir: Directory for outputs and checkpoints
            checkpoint_interval: Save checkpoint every N generations
            num_workers: Number of parallel workers (should match vCPU count)
        """
        self.config = config
        self.output_dir = output_dir
        self.checkpoint_interval = checkpoint_interval
        self.num_workers = num_workers

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"evolution_{timestamp}"
        self.log_file = self.output_dir / f"{self.run_id}.log"

        # Initialize log
        self._log("=" * 80)
        self._log(f"Remote Evolution Run: {self.run_id}")
        self._log("=" * 80)
        self._log(f"Output directory: {self.output_dir}")
        self._log(f"Workers: {self.num_workers}")
        self._log(f"Population: {config.population_size}")
        self._log(f"Generations: {config.num_generations}")
        self._log(f"Games per individual: {config.games_per_individual}")
        self._log(f"Checkpoint interval: {self.checkpoint_interval} generations")
        self._log("")

    def _log(self, message: str):
        """Log message to both console and file."""
        print(message)
        with open(self.log_file, "a") as f:
            f.write(message + "\n")

    def save_checkpoint(self, generation: int, population: Population, ga: GeneticAlgorithm):
        """Save checkpoint for resuming later.

        Args:
            generation: Current generation number
            population: Current population
            ga: GeneticAlgorithm instance
        """
        checkpoint = {
            "run_id": self.run_id,
            "generation": generation,
            "config": {
                "population_size": self.config.population_size,
                "num_generations": self.config.num_generations,
                "elite_size": self.config.elite_size,
                "selection_strategy": self.config.selection_strategy,
                "crossover_strategy": self.config.crossover_strategy,
                "crossover_rate": self.config.crossover_rate,
                "mutation_strategy": self.config.mutation_strategy,
                "mutation_rate": self.config.mutation_rate,
                "mutation_scale": self.config.mutation_scale,
                "fitness_type": self.config.fitness_type,
                "games_per_individual": self.config.games_per_individual,
                "seed": self.config.seed,
            },
            "population": population.to_dict(),
            "best_fitness_history": ga.fitness_history if hasattr(ga, 'fitness_history') else [],
            "diversity_history": ga.diversity_history if hasattr(ga, 'diversity_history') else [],
        }

        checkpoint_path = self.output_dir / f"checkpoint_gen{generation:04d}.json"
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint, f, indent=2)

        self._log(f"💾 Checkpoint saved: {checkpoint_path}")

    def progress_callback(self, generation: int, population: Population, ga: GeneticAlgorithm):
        """Progress callback with logging and checkpointing.

        Args:
            generation: Current generation number
            population: Current population
            ga: GeneticAlgorithm instance
        """
        stats = population.get_fitness_stats()
        diversity = population.get_diversity()

        log_msg = (
            f"Gen {generation:4d}/{self.config.num_generations} | "
            f"Best: {stats['max']:7.3f} | "
            f"Mean: {stats['mean']:7.3f} | "
            f"Std: {stats['std']:6.3f} | "
            f"Diversity: {diversity:.3f}"
        )
        self._log(log_msg)

        # Save checkpoint periodically
        if generation % self.checkpoint_interval == 0:
            self.save_checkpoint(generation, population, ga)

    def run(self, seed_individuals: Optional[list[Individual]] = None):
        """Run evolutionary training.

        Args:
            seed_individuals: Optional seed population

        Returns:
            EvolutionResult
        """
        # Create fitness evaluator with parallel execution
        if RUST_AVAILABLE:
            self._log(f"Using Rust fitness evaluator with {self.num_workers} workers")
            evaluator = RustFitnessEvaluator(
                scenario=default_scenario(num_agents=1),
                games_per_individual=self.config.games_per_individual,
                seed=self.config.seed,
                parallel=True,
                num_workers=self.num_workers,
            )
        else:
            self._log(f"Using Python fitness evaluator (slower, no parallelism)")
            evaluator = FitnessEvaluator(
                scenario=default_scenario(num_agents=1),
                games_per_individual=self.config.games_per_individual,
                seed=self.config.seed,
            )

        self._log("Starting evolution...")
        self._log("")

        # Create GA with evaluator
        ga = GeneticAlgorithm(self.config, fitness_evaluator=evaluator)

        start_time = time.time()

        # Run evolution
        result = ga.evolve(
            seed_individuals=seed_individuals,
            progress_callback=lambda gen, pop: self.progress_callback(gen, pop, ga),
        )

        elapsed_time = time.time() - start_time

        # Log results
        self._log("")
        self._log("=" * 80)
        self._log("Evolution Complete!")
        self._log("=" * 80)
        self._log(f"Time elapsed: {elapsed_time:.1f}s ({elapsed_time/60:.1f} min)")
        self._log(
            f"Converged at generation: {result.converged_at if result.converged_at is not None else 'N/A'}"
        )
        self._log("")
        self._log("Best Individual:")
        self._log(f"  Fitness: {result.best_individual.fitness:.4f}")
        self._log(f"  Generation: {result.best_individual.generation}")
        self._log(f"  Genome: {result.best_individual.genome.tolist()}")
        self._log("")
        self._log("Final Population Stats:")
        final_stats = result.final_population.get_fitness_stats()
        self._log(f"  Best: {final_stats['max']:.4f}")
        self._log(f"  Mean: {final_stats['mean']:.4f}")
        self._log(f"  Std:  {final_stats['std']:.4f}")
        self._log(f"  Diversity: {result.final_population.get_diversity():.4f}")
        self._log("")

        # Save final results
        results_path = self.output_dir / f"{self.run_id}_final.json"
        data = {
            "run_id": self.run_id,
            "config": {
                "population_size": self.config.population_size,
                "num_generations": self.config.num_generations,
                "elite_size": self.config.elite_size,
                "selection_strategy": self.config.selection_strategy,
                "crossover_strategy": self.config.crossover_strategy,
                "crossover_rate": self.config.crossover_rate,
                "mutation_strategy": self.config.mutation_strategy,
                "mutation_rate": self.config.mutation_rate,
                "mutation_scale": self.config.mutation_scale,
                "fitness_type": self.config.fitness_type,
                "games_per_individual": self.config.games_per_individual,
                "seed": self.config.seed,
                "num_workers": self.num_workers,
            },
            "best_individual": result.best_individual.to_dict(),
            "final_population": result.final_population.to_dict(),
            "fitness_history": result.fitness_history,
            "diversity_history": result.diversity_history,
            "converged_at": result.converged_at,
            "elapsed_time": elapsed_time,
        }

        with open(results_path, "w") as f:
            json.dump(data, f, indent=2)

        self._log(f"Final results saved: {results_path}")
        self._log("")

        return result


def load_checkpoint(checkpoint_path: Path) -> tuple[EvolutionConfig, list[Individual], int]:
    """Load checkpoint to resume evolution.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        Tuple of (config, population_individuals, generation)
    """
    with open(checkpoint_path, "r") as f:
        data = json.load(f)

    # Reconstruct config
    config = EvolutionConfig(
        population_size=data["config"]["population_size"],
        num_generations=data["config"]["num_generations"],
        elite_size=data["config"]["elite_size"],
        selection_strategy=data["config"]["selection_strategy"],
        crossover_strategy=data["config"]["crossover_strategy"],
        crossover_rate=data["config"]["crossover_rate"],
        mutation_strategy=data["config"]["mutation_strategy"],
        mutation_rate=data["config"]["mutation_rate"],
        mutation_scale=data["config"]["mutation_scale"],
        fitness_type=data["config"]["fitness_type"],
        games_per_individual=data["config"]["games_per_individual"],
        seed=data["config"]["seed"],
    )

    # Load population
    individuals = [
        Individual.from_dict(ind_data)
        for ind_data in data["population"]["individuals"]
    ]

    generation = data["generation"]

    print(f"Loaded checkpoint from generation {generation}")
    return config, individuals, generation


def main():
    parser = argparse.ArgumentParser(
        description="Run evolutionary training on remote GPU sandbox"
    )

    # Hardware settings
    parser.add_argument(
        "--num-workers",
        type=int,
        default=48,
        help="Number of parallel workers (default: 48, should match vCPU count)",
    )

    # Evolution settings
    parser.add_argument(
        "--population-size",
        type=int,
        default=100,
        help="Population size (default: 100)",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=500,
        help="Number of generations (default: 500 for extended run)",
    )
    parser.add_argument(
        "--games-per-individual",
        type=int,
        default=50,
        help="Games per individual for fitness evaluation (default: 50)",
    )
    parser.add_argument(
        "--elite-size",
        type=int,
        default=10,
        help="Elite size (default: 10)",
    )

    # Selection
    parser.add_argument(
        "--selection",
        type=str,
        default="tournament",
        choices=["tournament", "roulette", "rank"],
        help="Selection strategy (default: tournament)",
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
        help="Mutation scale (default: 0.1)",
    )

    # Fitness
    parser.add_argument(
        "--fitness-type",
        type=str,
        default="mean_reward",
        choices=["mean_reward", "win_rate", "robustness", "multi_objective"],
        help="Fitness function (default: mean_reward)",
    )

    # Checkpointing
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10,
        help="Save checkpoint every N generations (default: 10)",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Resume from checkpoint file",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/remote_evolution"),
        help="Output directory (default: runs/remote_evolution)",
    )

    # Random seed
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    # Load checkpoint if resuming
    if args.resume:
        config, seed_individuals, start_generation = load_checkpoint(args.resume)
        # Adjust generations to continue from checkpoint
        remaining_generations = config.num_generations - start_generation
        config.num_generations = remaining_generations
        print(f"Resuming from generation {start_generation}, running {remaining_generations} more generations")
    else:
        # Create configuration
        config = EvolutionConfig(
            population_size=args.population_size,
            num_generations=args.generations,
            elite_size=args.elite_size,
            selection_strategy=args.selection,
            crossover_strategy=args.crossover,
            crossover_rate=args.crossover_rate,
            mutation_strategy=args.mutation,
            mutation_rate=args.mutation_rate,
            mutation_scale=args.mutation_scale,
            fitness_type=args.fitness_type,
            games_per_individual=args.games_per_individual,
            seed=args.seed,
        )
        seed_individuals = None

    # Create runner
    runner = RemoteEvolutionRunner(
        config=config,
        output_dir=args.output_dir,
        checkpoint_interval=args.checkpoint_interval,
        num_workers=args.num_workers,
    )

    # Run evolution
    result = runner.run(seed_individuals=seed_individuals)

    print("\nEvolution complete! Check output directory for results and logs.")


if __name__ == "__main__":
    main()
