#!/usr/bin/env python3
"""
Benchmark script to compare sequential vs parallel fitness evaluation.

This script measures the performance improvement from parallel evaluation.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import time
from multiprocessing import cpu_count

from bucket_brigade.evolution import EvolutionConfig, GeneticAlgorithm


def benchmark_evolution(parallel: bool, num_workers: int = None, label: str = ""):
    """Run evolution and measure time.

    Args:
        parallel: Whether to use parallel evaluation
        num_workers: Number of workers (None = cpu_count)
        label: Label for output
    """
    config = EvolutionConfig(
        population_size=20,
        num_generations=5,
        games_per_individual=10,
        parallel=parallel,
        num_workers=num_workers,
        seed=42,
        early_stopping=False,
    )

    ga = GeneticAlgorithm(config)

    start_time = time.time()
    result = ga.evolve()
    elapsed_time = time.time() - start_time

    print(f"{label}:")
    print(f"  Time: {elapsed_time:.2f}s")
    print(f"  Best fitness: {result.best_individual.fitness:.4f}")
    print(f"  Generations completed: {len(result.fitness_history) - 1}")
    print()

    return elapsed_time


def main():
    """Run benchmarks comparing sequential and parallel evaluation."""
    print("=" * 80)
    print("Parallel Fitness Evaluation Benchmark")
    print("=" * 80)
    print(f"System CPUs: {cpu_count()}")
    print("Config: 20 individuals, 5 generations, 10 games/individual")
    print("Total evaluations: ~100 individuals (20 initial + 16 new per gen)")
    print()

    # Sequential baseline
    print("Running sequential (baseline)...")
    seq_time = benchmark_evolution(parallel=False, num_workers=None, label="Sequential")

    # Parallel with 2 workers
    print("Running parallel (2 workers)...")
    par2_time = benchmark_evolution(
        parallel=True, num_workers=2, label="Parallel (2 workers)"
    )

    # Parallel with 4 workers
    print("Running parallel (4 workers)...")
    par4_time = benchmark_evolution(
        parallel=True, num_workers=4, label="Parallel (4 workers)"
    )

    # Parallel with auto workers
    print(f"Running parallel (auto = {cpu_count()} workers)...")
    par_auto_time = benchmark_evolution(
        parallel=True,
        num_workers=None,
        label=f"Parallel (auto = {cpu_count()} workers)",
    )

    # Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Sequential time:    {seq_time:.2f}s (baseline)")
    print(
        f"Parallel (2 workers):  {par2_time:.2f}s ({seq_time / par2_time:.2f}x speedup)"
    )
    print(
        f"Parallel (4 workers):  {par4_time:.2f}s ({seq_time / par4_time:.2f}x speedup)"
    )
    print(
        f"Parallel (auto):       {par_auto_time:.2f}s ({seq_time / par_auto_time:.2f}x speedup)"
    )
    print()
    print(
        f"✅ Achieved {seq_time / par_auto_time:.1f}x speedup with {cpu_count()} workers"
    )
    print()


if __name__ == "__main__":
    main()
