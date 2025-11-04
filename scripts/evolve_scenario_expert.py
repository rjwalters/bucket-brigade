#!/usr/bin/env python3
"""
Evolve expert agent for a specific scenario.

This script evolves agent parameters optimized for a single scenario,
testing robustness against diverse team compositions to produce generalist experts.
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
from bucket_brigade.envs.scenarios import get_scenario_by_name, list_scenarios
from bucket_brigade.agents.heuristic_agent import HeuristicAgent
from bucket_brigade.envs.bucket_brigade_env import BucketBrigadeEnv


def create_team_composition_fitness(scenario_name: str, num_agents: int = 4):
    """Create fitness function that tests against diverse team compositions.

    Args:
        scenario_name: Name of the scenario to evolve for
        num_agents: Number of agents in the team

    Returns:
        Fitness function that evaluates robustness across team types
    """
    # Load scenario once
    scenario = get_scenario_by_name(scenario_name, num_agents)

    # Define different team parameter sets
    team_types = {
        'random': np.random.randn(10).tolist(),  # Random baseline
        'greedy': [0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # High free-riding
        'fair': [0.0, 0.9, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0],  # Cooperative
        'mixed': None,  # Will generate mix of above
    }

    def fitness_fn(individual: Individual) -> float:
        """Evaluate individual against diverse teams.

        Args:
            individual: Individual to evaluate

        Returns:
            Fitness score (mean reward - variance penalty)
        """
        all_rewards = []
        rng = np.random.default_rng(42)  # Fixed seed for reproducibility

        for team_type_name, team_params in team_types.items():
            # Create team for this type
            if team_type_name == 'mixed':
                # Mixed team: random, greedy, fair teammates
                teammates = [
                    HeuristicAgent(params=np.array(team_types['random']), agent_id=i+1, name=f"random-{i}")
                    for i in range(min(1, num_agents - 1))
                ]
                if num_agents > 2:
                    teammates.append(HeuristicAgent(params=np.array(team_types['greedy']), agent_id=2, name="greedy"))
                if num_agents > 3:
                    teammates.append(HeuristicAgent(params=np.array(team_types['fair']), agent_id=3, name="fair"))
                # Pad with random if needed
                while len(teammates) < num_agents - 1:
                    teammates.append(HeuristicAgent(
                        params=np.array(team_types['random']),
                        agent_id=len(teammates) + 1,
                        name=f"random-{len(teammates)}"
                    ))
            else:
                # Homogeneous team of this type
                teammates = [
                    HeuristicAgent(params=np.array(team_params), agent_id=i+1, name=f"{team_type_name}-{i}")
                    for i in range(num_agents - 1)
                ]

            # Run multiple games per team type for stability
            for game_idx in range(10):
                try:
                    # Create focal agent
                    focal_agent = HeuristicAgent(params=individual.genome, agent_id=0, name="focal")

                    # Create environment and run episode
                    env = BucketBrigadeEnv(scenario=scenario, agents=[focal_agent] + teammates)
                    obs = env.reset(seed=rng.integers(0, 2**31))

                    done = False
                    total_reward = 0.0

                    while not done:
                        actions = []
                        for agent in env.agents:
                            action = agent.get_action(obs)
                            actions.append(action)

                        obs, rewards, terminated, truncated, info = env.step(actions)
                        total_reward += rewards[0]  # Focal agent reward
                        done = terminated or truncated

                    all_rewards.append(total_reward)

                except Exception as e:
                    # Handle evaluation errors gracefully
                    print(f"Warning: Evaluation failed for {team_type_name}, game {game_idx}: {e}")
                    all_rewards.append(-200.0)  # Penalty for failure

        # Fitness = mean reward - variance penalty (robustness)
        mean_reward = np.mean(all_rewards)
        variance = np.var(all_rewards)
        fitness = mean_reward - 0.1 * variance

        return fitness

    return fitness_fn


def progress_callback(generation: int, population: Population, output_dir: Path) -> None:
    """Print evolution progress and save checkpoints.

    Args:
        generation: Current generation number
        population: Current population
        output_dir: Output directory for checkpoints
    """
    stats = population.get_fitness_stats()
    diversity = population.get_diversity()

    print(
        f"[Gen {generation:4d}] "
        f"Best: {stats['max']:.3f} | "
        f"Mean: {stats['mean']:.3f} ± {stats['std']:.3f} | "
        f"Diversity: {diversity:.3f}"
    )

    # Save checkpoint every 50 generations
    if generation > 0 and generation % 50 == 0:
        checkpoint_path = output_dir / f"checkpoint_gen{generation}.json"
        checkpoint_data = {
            "generation": generation,
            "population": population.to_dict(),
            "best_individual": population.get_best_individual().to_dict(),
        }

        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint_data, f, indent=2)

        print(f"  → Checkpoint saved: {checkpoint_path.name}")


def save_results(
    result,
    output_dir: Path,
    config: EvolutionConfig,
    scenario_name: str,
) -> None:
    """Save evolution results to output directory.

    Args:
        result: EvolutionResult instance
        output_dir: Output directory
        config: Evolution configuration
        scenario_name: Name of the scenario
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save best genome
    genome_path = output_dir / "best_genome.json"
    with open(genome_path, "w") as f:
        json.dump(result.best_individual.genome.tolist(), f, indent=2)
    print(f"\n✓ Best genome saved: {genome_path}")

    # Save fitness history
    history_path = output_dir / "fitness_history.json"
    with open(history_path, "w") as f:
        json.dump(result.fitness_history, f, indent=2)
    print(f"✓ Fitness history saved: {history_path}")

    # Save configuration
    config_path = output_dir / "config.json"
    config_data = {
        "scenario": scenario_name,
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
        "converged_at": result.converged_at,
    }
    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=2)
    print(f"✓ Configuration saved: {config_path}")

    # Save evolution log
    log_path = output_dir / "evolution_log.txt"
    with open(log_path, "w") as f:
        f.write(f"Scenario: {scenario_name}\n")
        f.write(f"Population Size: {config.population_size}\n")
        f.write(f"Generations: {config.num_generations}\n")
        f.write(f"Converged At: {result.converged_at}\n")
        f.write(f"\nBest Individual:\n")
        f.write(f"  Fitness: {result.best_individual.fitness:.4f}\n")
        f.write(f"  Generation: {result.best_individual.generation}\n")
        f.write(f"  Genome: {result.best_individual.genome.tolist()}\n")
        f.write(f"\nFinal Population Stats:\n")
        final_stats = result.final_population.get_fitness_stats()
        f.write(f"  Best: {final_stats['max']:.4f}\n")
        f.write(f"  Mean: {final_stats['mean']:.4f}\n")
        f.write(f"  Std:  {final_stats['std']:.4f}\n")
        f.write(f"  Diversity: {result.final_population.get_diversity():.4f}\n")
    print(f"✓ Evolution log saved: {log_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evolve expert agent for a specific scenario"
    )

    # Scenario selection
    parser.add_argument(
        "--scenario",
        type=str,
        required=True,
        help="Scenario name (e.g., 'easy', 'default', 'greedy_neighbor')",
    )
    parser.add_argument(
        "--num-agents",
        type=int,
        default=4,
        help="Number of agents in the scenario (default: 4)",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for results",
    )

    # Evolution parameters
    parser.add_argument(
        "--generations",
        type=int,
        default=100,
        help="Number of generations (default: 100)",
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=50,
        help="Population size (default: 50)",
    )
    parser.add_argument(
        "--elite-size",
        type=int,
        default=5,
        help="Elite size (default: 5)",
    )

    # Genetic operators
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
    parser.add_argument(
        "--crossover-rate",
        type=float,
        default=0.7,
        help="Crossover probability (default: 0.7)",
    )

    # Random seed
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    # Validate scenario name
    available_scenarios = list_scenarios()
    if args.scenario not in available_scenarios:
        print(f"Error: Scenario '{args.scenario}' not found")
        print(f"Available scenarios: {', '.join(available_scenarios)}")
        sys.exit(1)

    print("=" * 80)
    print(f"Evolving Expert for Scenario: {args.scenario}")
    print("=" * 80)
    print(f"Output directory: {args.output_dir}")
    print(f"Number of agents: {args.num_agents}")
    print(f"Population size: {args.population_size}")
    print(f"Generations: {args.generations}")
    print(f"Seed: {args.seed if args.seed else 'random'}")
    print()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Create fitness function with team composition testing
    print("Creating robustness fitness function...")
    print("  Testing against: random, greedy, fair, and mixed teams")
    print("  10 games per team type (40 total evaluations per individual)")
    print()

    # Create evolution configuration
    config = EvolutionConfig(
        population_size=args.population_size,
        num_generations=args.generations,
        elite_size=args.elite_size,
        selection_strategy="tournament",
        tournament_size=3,
        crossover_strategy="uniform",
        crossover_rate=args.crossover_rate,
        mutation_strategy="gaussian",
        mutation_rate=args.mutation_rate,
        mutation_scale=args.mutation_scale,
        fitness_type="custom",  # Using custom fitness function
        games_per_individual=40,  # 4 team types × 10 games
        parallel=True,
        maintain_diversity=True,
        min_diversity=0.1,
        early_stopping=True,
        convergence_generations=5,
        convergence_threshold=0.01,
        seed=args.seed,
    )

    # Create GA with custom fitness function
    ga = GeneticAlgorithm(config)

    # Override fitness evaluator with team composition fitness
    team_fitness_fn = create_team_composition_fitness(args.scenario, args.num_agents)

    # Wrap fitness function to match expected signature
    def wrapped_fitness_fn(individual: Individual) -> float:
        return team_fitness_fn(individual)

    # Replace the fitness evaluator's fitness function
    ga.fitness_evaluator.fitness_fn = wrapped_fitness_fn

    # Run evolution
    print("Starting evolution...")
    print()

    start_time = time.time()

    # Create progress callback with output_dir
    def progress_with_checkpoint(generation: int, population: Population) -> None:
        progress_callback(generation, population, args.output_dir)

    result = ga.evolve(progress_callback=progress_with_checkpoint)

    elapsed_time = time.time() - start_time

    # Print results
    print()
    print("=" * 80)
    print("Evolution Complete!")
    print("=" * 80)
    print(f"Time elapsed: {elapsed_time:.1f}s ({elapsed_time/60:.1f} minutes)")
    print(f"Converged at generation: {result.converged_at if result.converged_at is not None else 'N/A'}")
    print()
    print("Best Individual:")
    print(f"  Fitness: {result.best_individual.fitness:.4f}")
    print(f"  Generation: {result.best_individual.generation}")
    print(f"  Genome: {result.best_individual.genome.tolist()}")
    print()

    # Save results
    save_results(result, args.output_dir, config, args.scenario)

    print()
    print("✓ All results saved successfully!")
    print()
    print("To use the evolved expert:")
    print(f"  from bucket_brigade.agents.heuristic_agent import HeuristicAgent")
    print(f"  import json")
    print(f"  with open('{args.output_dir / 'best_genome.json'}', 'r') as f:")
    print(f"      params = json.load(f)")
    print(f"  agent = HeuristicAgent(params=np.array(params), agent_id=0, name='Expert-{args.scenario}')")


if __name__ == "__main__":
    main()
