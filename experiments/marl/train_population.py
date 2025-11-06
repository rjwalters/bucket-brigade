#!/usr/bin/env python3
"""
Population-Based Multi-Agent Training.

Trains a population of agents simultaneously using:
- Single CPU process for game simulation (Rust environments)
- Multiple GPU processes for policy learning (PPO)
- On-policy learning (each agent learns from own experience)

Architecture:
    CPU: Game Simulator (Rust BucketBrigade envs, matchmaking)
        ↓ Experience Queues
    GPU: Policy Learners (8 parallel processes, each training one agent)
        ↑ Policy Update Queue
    CPU: Policy Repository (updated policies for simulation)

Usage:
    # Train 8 agents on trivial_cooperation
    uv run python experiments/marl/train_population.py \\
        --scenario trivial_cooperation \\
        --population-size 8 \\
        --num-episodes 10000 \\
        --run-name pop_v1

    # Quick test
    uv run python experiments/marl/train_population.py \\
        --scenario trivial_cooperation \\
        --population-size 4 \\
        --num-episodes 100 \\
        --run-name test

    # Large-scale training
    uv run python experiments/marl/train_population.py \\
        --scenario mixed_motivation \\
        --population-size 16 \\
        --num-games 128 \\
        --num-episodes 100000 \\
        --hidden-size 1024 \\
        --run-name large_pop_v1
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
from datetime import datetime
import multiprocessing as mp

from bucket_brigade.training.population_trainer import PopulationTrainer


def main():
    parser = argparse.ArgumentParser(description="Train agent population with PPO")

    # Scenario and population
    parser.add_argument("--scenario", type=str, default="trivial_cooperation",
                        help="Scenario name")
    parser.add_argument("--population-size", type=int, default=8,
                        help="Number of agents in population")
    parser.add_argument("--num-games", type=int, default=64,
                        help="Number of parallel game environments")
    parser.add_argument("--num-agents-per-game", type=int, default=4,
                        help="Number of agents per game")

    # Training
    parser.add_argument("--num-episodes", type=int, default=10000,
                        help="Total simulation episodes")
    parser.add_argument("--hidden-size", type=int, default=512,
                        help="Hidden layer size for policy networks")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                        help="Learning rate")

    # Learner parameters
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size for learners")
    parser.add_argument("--num-epochs", type=int, default=4,
                        help="PPO epochs per batch")
    parser.add_argument("--update-interval", type=int, default=100,
                        help="Learners send policy updates every N batches")

    # System
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for GPU learners (cuda or cpu)")
    parser.add_argument("--matchmaking", type=str, default="round_robin",
                        choices=["round_robin", "random"],
                        help="Matchmaking strategy")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed")

    # Logging and checkpoints
    parser.add_argument("--run-name", type=str, default=None,
                        help="Run name for logging")
    parser.add_argument("--log-interval", type=int, default=100,
                        help="Log progress every N episodes")
    parser.add_argument("--checkpoint-interval", type=int, default=1000,
                        help="Save checkpoint every N episodes")

    args = parser.parse_args()

    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{args.scenario}_pop{args.population_size}_{timestamp}"

    # Directories
    exp_dir = Path("experiments/marl")
    checkpoint_dir = exp_dir / "checkpoints" / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"🎮 Population-Based Multi-Agent Training")
    print(f"{'='*60}")
    print(f"Run: {run_name}")
    print(f"Scenario: {args.scenario}")
    print(f"Population: {args.population_size} agents")
    print(f"Parallel games: {args.num_games}")
    print(f"Episodes: {args.num_episodes:,}")
    print(f"Device: {args.device}")
    print(f"Matchmaking: {args.matchmaking}")
    print(f"Seed: {args.seed}")
    print(f"{'='*60}\n")

    # Create trainer
    trainer = PopulationTrainer(
        scenario_name=args.scenario,
        population_size=args.population_size,
        num_games=args.num_games,
        num_agents_per_game=args.num_agents_per_game,
        hidden_size=args.hidden_size,
        learning_rate=args.learning_rate,
        device=args.device,
        matchmaking_strategy=args.matchmaking,
        seed=args.seed,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        update_interval=args.update_interval,
        checkpoint_dir=checkpoint_dir,
        log_interval=args.log_interval,
    )

    # Save configuration
    config_path = checkpoint_dir / "config.json"
    trainer.save_config(config_path)

    try:
        # Run training
        trainer.train(num_episodes=args.num_episodes)

        # Save final checkpoint
        final_checkpoint = checkpoint_dir / f"final_checkpoint.pt"
        trainer.save_checkpoint(final_checkpoint)

    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        trainer.cleanup()

    except Exception as e:
        print(f"\n\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        trainer.cleanup()
        raise

    print(f"\n✅ Training complete! Results saved to: {checkpoint_dir}")


if __name__ == "__main__":
    # Required for multiprocessing on some platforms
    mp.set_start_method('spawn', force=True)
    main()
