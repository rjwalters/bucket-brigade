#!/usr/bin/env python3
"""
Train agents with a curriculum of increasingly difficult scenarios.

This implements curriculum learning where agents start with simple scenarios
and progressively learn more complex strategies.
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path

# Add the bucket_brigade package to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pufferlib
from pufferlib.models import PolicyValueNetwork

from bucket_brigade.envs import make_vectorized_env


class CurriculumTrainer:
    """Manages curriculum learning across scenarios."""

    def __init__(self, base_config):
        self.base_config = base_config
        self.current_stage = 0

        # Define curriculum stages (easier to harder)
        self.curriculum = [
            {
                "name": "trivial_cooperation",
                "timesteps": 200_000,
                "opponents": 2,
                "description": "Learn basic cooperation",
            },
            {
                "name": "early_containment",
                "timesteps": 300_000,
                "opponents": 3,
                "description": "Learn timing and coordination",
            },
            {
                "name": "greedy_neighbor",
                "timesteps": 400_000,
                "opponents": 3,
                "description": "Learn social dilemmas",
            },
            {
                "name": "sparse_heroics",
                "timesteps": 500_000,
                "opponents": 4,
                "description": "Learn resource allocation",
            },
            {
                "name": "chain_reaction",
                "timesteps": 600_000,
                "opponents": 4,
                "description": "Learn distributed coordination",
            },
        ]

    def train_curriculum(self, run_name: str = "curriculum_training"):
        """Train through the full curriculum."""
        print("🎓 Starting Curriculum Training")
        print("=" * 50)

        # Initialize policy
        dummy_env = make_vectorized_env(
            scenario_name="trivial_cooperation", num_opponents=2
        )
        policy = PolicyValueNetwork(
            dummy_env.observation_space.shape[0],
            dummy_env.action_space.nvec.sum(),
            policy_channels=64,
            value_channels=64,
            policy_layers=3,
            value_layers=3,
        )

        total_timesteps = 0

        for stage_idx, stage in enumerate(self.curriculum):
            print(f"\n📚 Stage {stage_idx + 1}: {stage['name']}")
            print(f"   {stage['description']}")
            print(f"   Timesteps: {stage['timesteps']:,}")
            print(f"   Opponents: {stage['opponents']}")

            # Create environment for this stage
            env = make_vectorized_env(
                scenario_name=stage["name"], num_opponents=stage["opponents"]
            )

            # Train on this stage
            stage_timesteps = self._train_stage(policy, env, stage, run_name, stage_idx)

            total_timesteps += stage_timesteps
            print(f"   ✅ Stage complete! Total timesteps: {total_timesteps:,}")

        # Final evaluation
        print("\n🎯 Final Evaluation:")
        self._evaluate_curriculum_policy(policy, run_name)

        # Save final model
        output_dir = Path("models") / run_name
        output_dir.mkdir(parents=True, exist_ok=True)
        final_path = output_dir / "curriculum_final.pt"
        torch.save(policy.state_dict(), final_path)
        print(f"💾 Saved final curriculum model to {final_path}")

        print("🎓 Curriculum training complete!")

    def _train_stage(self, policy, env, stage, run_name, stage_idx):
        """Train on a single curriculum stage."""
        # Set up optimizer (use lower LR for later stages to fine-tune)
        lr = self.base_config["learning_rate"] * (0.8**stage_idx)
        optimizer = torch.optim.Adam(policy.parameters(), lr=lr, eps=1e-5)

        # Create trainer
        trainer = pufferlib.PPO(
            policy,
            optimizer,
            env,
            batch_size=self.base_config["batch_size"],
            learning_rate=lr,
            num_epochs=self.base_config["num_epochs"],
            num_minibatches=self.base_config["num_minibatches"],
            anneal_lr=False,  # Don't anneal within stages
            gamma=self.base_config["gamma"],
            gae_lambda=self.base_config["gae_lambda"],
            clip_coef=self.base_config["clip_coef"],
            ent_coef=self.base_config["ent_coef"],
            vf_coef=self.base_config["vf_coef"],
            max_grad_norm=self.base_config["max_grad_norm"],
        )

        # Train for specified timesteps
        stage_timesteps = 0
        target_timesteps = stage["timesteps"]

        while stage_timesteps < target_timesteps:
            stats = trainer.evaluate_and_train()
            stage_timesteps += self.base_config["batch_size"]

            if stage_timesteps % 50000 == 0:
                progress = min(stage_timesteps / target_timesteps, 1.0)
                print(
                    f"   Progress: {progress * 100:.1f}% ({stage_timesteps}/{target_timesteps})"
                )

        return stage_timesteps

    def _evaluate_curriculum_policy(self, policy, run_name):
        """Evaluate the final curriculum-trained policy."""
        from scripts.evaluate_policy import evaluate_policy_tournament

        # Create a temporary model file for evaluation
        temp_model_path = f"/tmp/curriculum_eval_{run_name}.pt"
        torch.save(policy.state_dict(), temp_model_path)

        try:
            # Evaluate on the hardest scenario
            results = evaluate_policy_tournament(
                temp_model_path,
                scenario_name="chain_reaction",
                num_games=30,
                num_opponents=4,
            )

            trained_perf = results["trained_agent"]
            print(f"   Win Rate: {trained_perf['win_rate']:.3f}")
            print(f"   Mean Reward: {trained_perf['mean_reward']:.3f}")
            print(f"   Std Reward: {trained_perf['std_reward']:.3f}")
            # Clean up
            os.unlink(temp_model_path)

        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            if os.path.exists(temp_model_path):
                os.unlink(temp_model_path)


def main():
    parser = argparse.ArgumentParser(
        description="Train Bucket Brigade agent with curriculum learning"
    )

    # Curriculum settings
    parser.add_argument(
        "--run-name",
        type=str,
        default="curriculum_training",
        help="Name for this training run",
    )

    # Training hyperparameters (similar to train_policy.py)
    parser.add_argument("--batch-size", type=int, default=2**12, help="Batch size")
    parser.add_argument(
        "--learning-rate", type=float, default=2.5e-4, help="Initial learning rate"
    )
    parser.add_argument(
        "--num-epochs", type=int, default=4, help="PPO epochs per update"
    )
    parser.add_argument(
        "--num-minibatches", type=int, default=4, help="Minibatches per epoch"
    )

    # PPO hyperparameters
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument(
        "--clip-coef", type=float, default=0.2, help="PPO clip coefficient"
    )
    parser.add_argument(
        "--ent-coef", type=float, default=0.01, help="Entropy coefficient"
    )
    parser.add_argument(
        "--vf-coef", type=float, default=0.5, help="Value function coefficient"
    )
    parser.add_argument(
        "--max-grad-norm", type=float, default=0.5, help="Maximum gradient norm"
    )

    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Set up base configuration
    base_config = {
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "num_epochs": args.num_epochs,
        "num_minibatches": args.num_minibatches,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "clip_coef": args.clip_coef,
        "ent_coef": args.ent_coef,
        "vf_coef": args.vf_coef,
        "max_grad_norm": args.max_grad_norm,
    }

    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create and run curriculum trainer
    trainer = CurriculumTrainer(base_config)
    trainer.train_curriculum(args.run_name)


if __name__ == "__main__":
    main()
