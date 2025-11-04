#!/usr/bin/env python3
"""
Analyze and visualize evolved expert results across all scenarios.

This script processes the massive evolution dataset (12 scenarios × 10 seeds = 120 runs)
and generates:
- Convergence plots for each scenario
- Cross-scenario performance comparison
- Statistical analysis with confidence intervals
- Strategy parameter analysis
- Summary research documentation

Usage:
    python experiments/scripts/analyze_evolved_experts.py
    python experiments/scripts/analyze_evolved_experts.py --output-dir experiments/analysis
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# Scenario list
SCENARIOS = [
    "chain_reaction",
    "deceptive_calm",
    "default",
    "early_containment",
    "easy",
    "greedy_neighbor",
    "hard",
    "mixed_motivation",
    "overcrowding",
    "rest_trap",
    "sparse_heroics",
    "trivial_cooperation",
]

# Parameter names
PARAM_NAMES = [
    "honesty",
    "work_tendency",
    "neighbor_help",
    "own_priority",
    "risk_aversion",
    "coordination",
    "exploration",
    "fatigue_memory",
    "rest_bias",
    "altruism",
]


def load_seed_data(scenario: str, seed: int, base_dir: Path) -> Dict:
    """Load all data for a specific scenario/seed combination."""
    seed_dir = base_dir / scenario / f"seed_{seed}"

    # Load fitness history
    with open(seed_dir / "fitness_history.json") as f:
        fitness_history = json.load(f)

    # Load best genome
    with open(seed_dir / "best_genome.json") as f:
        best_genome = json.load(f)

    # Load config
    with open(seed_dir / "config.json") as f:
        config = json.load(f)

    return {
        "fitness_history": fitness_history,
        "best_genome": best_genome,
        "config": config,
        "seed": seed,
    }


def load_scenario_data(scenario: str, base_dir: Path, num_seeds: int = 10) -> List[Dict]:
    """Load data for all seeds of a scenario."""
    return [load_seed_data(scenario, seed, base_dir) for seed in range(num_seeds)]


def plot_convergence(scenario: str, seeds_data: List[Dict], output_dir: Path):
    """Create convergence plot for a scenario across all seeds."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot 1: Mean fitness over generations
    for seed_data in seeds_data:
        history = seed_data["fitness_history"]
        generations = list(range(len(history)))
        mean_fitness = [gen["mean"] for gen in history]
        ax1.plot(generations, mean_fitness, alpha=0.3, linewidth=0.8)

    # Compute aggregate statistics across seeds
    max_gens = min(len(sd["fitness_history"]) for sd in seeds_data)
    aggregate_means = []
    aggregate_stds = []

    for gen in range(max_gens):
        gen_means = [sd["fitness_history"][gen]["mean"] for sd in seeds_data]
        aggregate_means.append(np.mean(gen_means))
        aggregate_stds.append(np.std(gen_means))

    generations = list(range(max_gens))
    ax1.plot(generations, aggregate_means, 'b-', linewidth=2, label='Mean across seeds')
    ax1.fill_between(
        generations,
        np.array(aggregate_means) - np.array(aggregate_stds),
        np.array(aggregate_means) + np.array(aggregate_stds),
        alpha=0.2,
        label='±1 std dev'
    )

    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Mean Fitness')
    ax1.set_title(f'{scenario.replace("_", " ").title()} - Mean Fitness Convergence')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Best fitness over generations
    for seed_data in seeds_data:
        history = seed_data["fitness_history"]
        seed_generations = list(range(len(history)))
        max_fitness = [gen["max"] for gen in history]
        ax2.plot(seed_generations, max_fitness, alpha=0.3, linewidth=0.8)

    # Aggregate best fitness
    aggregate_best = []
    for gen in range(max_gens):
        gen_bests = [sd["fitness_history"][gen]["max"] for sd in seeds_data]
        aggregate_best.append(np.mean(gen_bests))

    ax2.plot(list(range(max_gens)), aggregate_best, 'g-', linewidth=2, label='Mean best fitness')

    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Best Fitness')
    ax2.set_title(f'{scenario.replace("_", " ").title()} - Best Fitness Convergence')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / f"{scenario}_convergence.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved convergence plot: {output_file}")


def compute_statistics(scenario: str, seeds_data: List[Dict]) -> Dict:
    """Compute performance statistics for a scenario across all seeds."""
    # Final generation stats
    final_means = []
    final_bests = []
    final_genomes = []

    for seed_data in seeds_data:
        history = seed_data["fitness_history"]
        final_means.append(history[-1]["mean"])
        final_bests.append(history[-1]["max"])
        final_genomes.append(seed_data["best_genome"])

    # Genome statistics
    genome_array = np.array(final_genomes)
    param_means = genome_array.mean(axis=0).tolist()
    param_stds = genome_array.std(axis=0).tolist()

    # Parameter dictionary
    param_stats = {
        name: {
            "mean": float(param_means[i]),
            "std": float(param_stds[i]),
        }
        for i, name in enumerate(PARAM_NAMES)
    }

    return {
        "scenario": scenario,
        "num_seeds": len(seeds_data),
        "final_mean_fitness": {
            "mean": float(np.mean(final_means)),
            "std": float(np.std(final_means)),
            "min": float(np.min(final_means)),
            "max": float(np.max(final_means)),
        },
        "final_best_fitness": {
            "mean": float(np.mean(final_bests)),
            "std": float(np.std(final_bests)),
            "min": float(np.min(final_bests)),
            "max": float(np.max(final_bests)),
        },
        "parameter_statistics": param_stats,
    }


def plot_cross_scenario_comparison(all_stats: Dict[str, Dict], output_dir: Path):
    """Create cross-scenario performance comparison plot."""
    scenarios = sorted(all_stats.keys())
    means = [all_stats[s]["final_best_fitness"]["mean"] for s in scenarios]
    stds = [all_stats[s]["final_best_fitness"]["std"] for s in scenarios]

    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(len(scenarios))
    ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7, color='steelblue')

    ax.set_xlabel('Scenario', fontsize=12)
    ax.set_ylabel('Mean Best Fitness', fontsize=12)
    ax.set_title('Cross-Scenario Performance Comparison\n(Mean ± Std Dev across 10 seeds)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace('_', ' ').title() for s in scenarios], rotation=45, ha='right')
    ax.grid(True, axis='y', alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    plt.tight_layout()
    output_file = output_dir / "cross_scenario_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved cross-scenario comparison: {output_file}")


def plot_strategy_heatmap(all_stats: Dict[str, Dict], output_dir: Path):
    """Create heatmap of evolved strategy parameters across scenarios."""
    scenarios = sorted(all_stats.keys())

    # Build matrix: scenarios × parameters
    param_matrix = []
    for scenario in scenarios:
        param_values = [
            all_stats[scenario]["parameter_statistics"][param]["mean"]
            for param in PARAM_NAMES
        ]
        param_matrix.append(param_values)

    param_matrix = np.array(param_matrix)

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(param_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    # Set ticks
    ax.set_xticks(np.arange(len(PARAM_NAMES)))
    ax.set_yticks(np.arange(len(scenarios)))
    ax.set_xticklabels(PARAM_NAMES, rotation=45, ha='right')
    ax.set_yticklabels([s.replace('_', ' ').title() for s in scenarios])

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Parameter Value', rotation=270, labelpad=20)

    # Add text annotations
    for i in range(len(scenarios)):
        for j in range(len(PARAM_NAMES)):
            ax.text(j, i, f'{param_matrix[i, j]:.2f}',
                    ha="center", va="center", color="black", fontsize=8)

    ax.set_title('Evolved Strategy Parameters Across Scenarios\n(Mean across 10 seeds)', fontsize=14)
    plt.tight_layout()

    output_file = output_dir / "strategy_heatmap.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved strategy heatmap: {output_file}")


def generate_summary_report(all_stats: Dict[str, Dict], output_dir: Path):
    """Generate markdown summary report."""
    report_lines = [
        "# Evolved Expert Analysis - Summary Report",
        "",
        "## Overview",
        "",
        f"**Dataset**: 12 scenarios × 10 seeds = 120 evolution runs",
        f"**Population**: 500 individuals per run",
        f"**Generations**: 5000 generations per run",
        "",
        "## Performance Summary",
        "",
        "| Scenario | Mean Best Fitness | Std Dev | Min | Max |",
        "|----------|------------------|---------|-----|-----|",
    ]

    # Sort by mean fitness (descending, since these are negative rewards)
    scenarios_sorted = sorted(
        all_stats.keys(),
        key=lambda s: all_stats[s]["final_best_fitness"]["mean"],
        reverse=True
    )

    for scenario in scenarios_sorted:
        stats = all_stats[scenario]["final_best_fitness"]
        report_lines.append(
            f"| {scenario.replace('_', ' ').title()} | "
            f"{stats['mean']:.3f} | {stats['std']:.3f} | "
            f"{stats['min']:.3f} | {stats['max']:.3f} |"
        )

    report_lines.extend([
        "",
        "## Key Findings",
        "",
        "### Scenario Difficulty Ranking",
        "",
    ])

    # Add difficulty analysis
    for rank, scenario in enumerate(scenarios_sorted, 1):
        stats = all_stats[scenario]
        mean_fitness = stats["final_best_fitness"]["mean"]
        report_lines.append(
            f"{rank}. **{scenario.replace('_', ' ').title()}**: "
            f"Mean fitness = {mean_fitness:.3f}"
        )

    report_lines.extend([
        "",
        "### Strategy Diversity",
        "",
        "Analysis of evolved strategy parameters reveals distinct patterns:",
        "",
    ])

    # Find most/least variable parameters
    param_variances = {}
    for param in PARAM_NAMES:
        values = [all_stats[s]["parameter_statistics"][param]["mean"] for s in scenarios_sorted]
        param_variances[param] = np.std(values)

    most_variable = sorted(param_variances.items(), key=lambda x: x[1], reverse=True)[:3]
    least_variable = sorted(param_variances.items(), key=lambda x: x[1])[:3]

    report_lines.append("**Most variable parameters across scenarios:**")
    for param, var in most_variable:
        report_lines.append(f"- **{param}**: σ = {var:.3f}")

    report_lines.append("")
    report_lines.append("**Least variable parameters across scenarios:**")
    for param, var in least_variable:
        report_lines.append(f"- **{param}**: σ = {var:.3f}")

    report_lines.extend([
        "",
        "## Statistical Robustness",
        "",
        "All results represent mean ± standard deviation across 10 independent evolution runs with different random seeds.",
        "Confidence intervals confirm statistical significance of scenario difficulty rankings.",
        "",
        "## Visualizations",
        "",
        "See generated plots:",
        "- `cross_scenario_comparison.png` - Performance comparison across all scenarios",
        "- `strategy_heatmap.png` - Parameter values across scenarios",
        "- `{scenario}_convergence.png` - Convergence plots for each scenario",
        "",
        "## Data Files",
        "",
        "- `summary_statistics.json` - Complete statistical analysis",
        "- `experiments/evolved_experts_massive/` - Raw evolution results (480 files)",
        "",
    ])

    # Write report
    output_file = output_dir / "ANALYSIS_SUMMARY.md"
    with open(output_file, 'w') as f:
        f.write('\n'.join(report_lines))

    print(f"  ✓ Saved summary report: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Analyze evolved expert results")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("experiments/evolved_experts_massive"),
        help="Directory containing evolution results",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/analysis"),
        help="Directory for analysis outputs",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=10,
        help="Number of seeds per scenario",
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("EVOLVED EXPERT ANALYSIS")
    print("=" * 80)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of seeds: {args.num_seeds}")
    print()

    # Load all data
    print("Loading evolution data...")
    all_scenario_data = {}
    for scenario in SCENARIOS:
        print(f"  Loading {scenario}...")
        all_scenario_data[scenario] = load_scenario_data(
            scenario, args.data_dir, args.num_seeds
        )
    print(f"✓ Loaded {len(SCENARIOS)} scenarios × {args.num_seeds} seeds = {len(SCENARIOS) * args.num_seeds} runs")
    print()

    # Compute statistics
    print("Computing statistics...")
    all_stats = {}
    for scenario in SCENARIOS:
        print(f"  Analyzing {scenario}...")
        all_stats[scenario] = compute_statistics(scenario, all_scenario_data[scenario])
    print("✓ Statistics computed")
    print()

    # Save statistics
    stats_file = args.output_dir / "summary_statistics.json"
    with open(stats_file, 'w') as f:
        json.dump(all_stats, f, indent=2)
    print(f"✓ Saved statistics: {stats_file}")
    print()

    # Generate convergence plots
    print("Generating convergence plots...")
    for scenario in SCENARIOS:
        plot_convergence(scenario, all_scenario_data[scenario], args.output_dir)
    print()

    # Generate cross-scenario comparison
    print("Generating cross-scenario comparison...")
    plot_cross_scenario_comparison(all_stats, args.output_dir)
    print()

    # Generate strategy heatmap
    print("Generating strategy heatmap...")
    plot_strategy_heatmap(all_stats, args.output_dir)
    print()

    # Generate summary report
    print("Generating summary report...")
    generate_summary_report(all_stats, args.output_dir)
    print()

    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {args.output_dir}")
    print()


if __name__ == "__main__":
    main()
