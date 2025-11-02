#!/usr/bin/env python3
"""
Train a reinforcement learning policy for Bucket Brigade using PufferLib.

This script trains an agent to play Bucket Brigade against a mix of opponent
policies, similar to the tournament setup. The trained agent learns optimal
strategies through interaction with the environment.
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path

# Add the bucket_brigade package to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pufferlib
import pufferlib.utils
import pufferlib.vectorization
from pufferlib.models import PolicyValueNetwork
import torch
import torch.nn as nn

from bucket_brigade.envs import make_env, make_vectorized_env


def make_policy(env):
    """Create a policy network for the environment."""
    return PolicyValueNetwork(
        env.observation_space.shape[0],
        env.action_space.nvec.sum(),
        policy_channels=64,
        value_channels=64,
        policy_layers=3,
        value_layers=3,
    )


def train_policy(
    scenario_name: str = 'default',
    num_opponents: int = 3,
    opponent_policies: list = None,
    total_timesteps: int = 1_000_000,
    batch_size: int = 2**12,
    learning_rate: float = 2.5e-4,
    num_envs: int = 8,
    num_epochs: int = 4,
    num_minibatches: int = 4,
    anneal_lr: bool = True,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_coef: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    target_kl: float = None,
    seed: int = 42,
    eval_interval: int = 1000,
    save_interval: int = 10000,
    run_name: str = None,
):
    """
    Train a policy using PPO.

    Args:
        scenario_name: Which scenario to train on
        num_opponents: Number of opponent agents
        opponent_policies: List of opponent policy types
        total_timesteps: Total training timesteps
        batch_size: Batch size for training
        learning_rate: Learning rate
        num_envs: Number of parallel environments
        num_epochs: Number of epochs per update
        num_minibatches: Number of minibatches per epoch
        anneal_lr: Whether to anneal learning rate
        gamma: Discount factor
        gae_lambda: GAE lambda
        clip_coef: PPO clip coefficient
        ent_coef: Entropy coefficient
        vf_coef: Value function coefficient
        max_grad_norm: Maximum gradient norm
        target_kl: Target KL divergence for early stopping
        seed: Random seed
        eval_interval: Evaluation interval
        save_interval: Model save interval
        run_name: Name for this training run
    """

    # Set up run name
    if run_name is None:
        run_name = f"bb_{scenario_name}_{num_opponents}opp_{seed}"

    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Create environments
    print(f"🎯 Training on scenario: {scenario_name}")
    print(f"🤖 Opponents: {num_opponents}")
    print(f"🏃 Environments: {num_envs}")
    print(f"📊 Total timesteps: {total_timesteps:,}")
    print()

    # Create training environments
    train_envs = make_vectorized_env(
        num_envs=num_envs,
        scenario_name=scenario_name,
        num_opponents=num_opponents
    )

    # Create evaluation environment
    eval_env = make_env(scenario_name, num_opponents)

    # Set up policy and optimizer
    policy = make_policy(train_envs)
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate, eps=1e-5)

    # Set up PufferLib PPO trainer
    trainer = pufferlib.PPO(
        policy,
        optimizer,
        train_envs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        num_minibatches=num_minibatches,
        anneal_lr=anneal_lr,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_coef=clip_coef,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        target_kl=target_kl,
    )

    # Create output directory
    output_dir = Path("models") / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    print("🚀 Starting training...")

    global_step = 0
    best_reward = float('-inf')

    while global_step < total_timesteps:
        # Training step
        stats = trainer.evaluate_and_train()

        global_step += batch_size

        # Logging
        if global_step % 1000 == 0:
            print(f"📈 Step {global_step:,} | "
                  f"Loss: {stats['policy_loss']:.3f} | "
                  f"Value Loss: {stats['value_loss']:.3f} | "
                  f"Entropy: {stats['entropy']:.3f} | "
                  f"Reward: {stats['mean_reward']:.2f}")

        # Evaluation
        if global_step % eval_interval == 0:
            eval_reward = evaluate_policy(policy, eval_env, num_episodes=10)
            print(f"🎯 Evaluation | Mean Reward: {eval_reward:.2f}")

            if eval_reward > best_reward:
                best_reward = eval_reward
                save_path = output_dir / f"best_policy_{global_step}.pt"
                torch.save(policy.state_dict(), save_path)
                print(f"💾 Saved best model to {save_path}")

        # Regular saving
        if global_step % save_interval == 0:
            save_path = output_dir / f"policy_{global_step}.pt"
            torch.save(policy.state_dict(), save_path)
            print(f"💾 Saved checkpoint to {save_path}")

    # Final save
    final_path = output_dir / "final_policy.pt"
    torch.save(policy.state_dict(), final_path)
    print(f"💾 Saved final model to {final_path}")

    print("🎉 Training complete!")


def evaluate_policy(policy, env, num_episodes: int = 10, deterministic: bool = True):
    """Evaluate a policy on the environment."""
    total_reward = 0

    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            # Get action from policy
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                action_logits, value = policy(obs_tensor)

                if deterministic:
                    action = torch.argmax(action_logits, dim=-1).squeeze().numpy()
                else:
                    action_dist = torch.distributions.Categorical(logits=action_logits)
                    action = action_dist.sample().squeeze().numpy()

            # Convert multi-discrete action back to [house, mode]
            house_idx = action // 2  # First dimension
            mode = action % 2        # Second dimension
            action_array = np.array([house_idx, mode])

            # Step environment
            obs, reward, terminated, truncated, _ = env.step(action_array)
            episode_reward += reward
            done = terminated or truncated

        total_reward += episode_reward

    return total_reward / num_episodes


def main():
    parser = argparse.ArgumentParser(description='Train Bucket Brigade RL Policy')

    # Environment settings
    parser.add_argument('--scenario', type=str, default='default',
                       choices=['default', 'trivial_cooperation', 'early_containment',
                               'greedy_neighbor', 'sparse_heroics'],
                       help='Training scenario')
    parser.add_argument('--num-opponents', type=int, default=3,
                       help='Number of opponent agents')
    parser.add_argument('--opponent-policies', type=str, nargs='+',
                       default=['random', 'firefighter', 'coordinator'],
                       help='Opponent policy types')

    # Training settings
    parser.add_argument('--total-timesteps', type=int, default=1_000_000,
                       help='Total training timesteps')
    parser.add_argument('--batch-size', type=int, default=2**12,
                       help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=2.5e-4,
                       help='Learning rate')
    parser.add_argument('--num-envs', type=int, default=8,
                       help='Number of parallel environments')
    parser.add_argument('--num-epochs', type=int, default=4,
                       help='PPO epochs per update')
    parser.add_argument('--num-minibatches', type=int, default=4,
                       help='Minibatches per epoch')

    # PPO hyperparameters
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor')
    parser.add_argument('--gae-lambda', type=float, default=0.95,
                       help='GAE lambda')
    parser.add_argument('--clip-coef', type=float, default=0.2,
                       help='PPO clip coefficient')
    parser.add_argument('--ent-coef', type=float, default=0.01,
                       help='Entropy coefficient')
    parser.add_argument('--vf-coef', type=float, default=0.5,
                       help='Value function coefficient')
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                       help='Maximum gradient norm')

    # Training options
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--eval-interval', type=int, default=1000,
                       help='Evaluation interval')
    parser.add_argument('--save-interval', type=int, default=10000,
                       help='Model save interval')
    parser.add_argument('--run-name', type=str, default=None,
                       help='Name for this training run')

    args = parser.parse_args()

    train_policy(
        scenario_name=args.scenario,
        num_opponents=args.num_opponents,
        opponent_policies=args.opponent_policies,
        total_timesteps=args.total_timesteps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_envs=args.num_envs,
        num_epochs=args.num_epochs,
        num_minibatches=args.num_minibatches,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_coef=args.clip_coef,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()
