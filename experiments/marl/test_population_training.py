#!/usr/bin/env python3
"""
Test script for population-based training on GPU.

This script tests the full population training pipeline with:
- 8 agents in the population
- 64 parallel game environments
- GPU-accelerated learning
- Checkpointing and logging
"""

from pathlib import Path
from bucket_brigade.training import PopulationTrainer


def main():
    """Run population training test."""
    print("=" * 80)
    print("Population-Based Training Test")
    print("=" * 80)

    # Create trainer
    trainer = PopulationTrainer(
        scenario_name='trivial_cooperation',
        population_size=8,
        num_games=64,
        num_agents_per_game=4,
        # Network architecture
        hidden_size=512,
        learning_rate=3e-4,
        # Training parameters
        device='cuda',  # Use GPU
        matchmaking_strategy='round_robin',
        seed=42,
        # Learner parameters
        batch_size=256,
        num_epochs=4,
        update_interval=100,
        # Checkpoint and logging
        checkpoint_dir=Path('experiments/marl/checkpoints/population_test'),
        log_interval=50,
    )

    # Train for 1000 episodes
    print("\nStarting training for 1000 episodes...")
    trainer.train(num_episodes=1000)

    # Save final checkpoint
    checkpoint_path = Path('experiments/marl/checkpoints/population_test/final.pt')
    trainer.save_checkpoint(checkpoint_path)

    # Save config
    config_path = Path('experiments/marl/checkpoints/population_test/config.json')
    trainer.save_config(config_path)

    print("\n" + "=" * 80)
    print("Training complete!")
    print(f"Checkpoint saved: {checkpoint_path}")
    print(f"Config saved: {config_path}")
    print("=" * 80)


if __name__ == '__main__':
    main()
