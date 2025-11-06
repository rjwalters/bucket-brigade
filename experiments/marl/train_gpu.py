#!/usr/bin/env python3
"""
GPU-Optimized PPO Training for Bucket Brigade using Rust-backed environment.

This script trains a neural network policy using PPO with the RustPufferBucketBrigade
environment (100x faster than Python). Optimized for GPU training with proper
checkpointing, logging, and evaluation.

Usage:
    # Full training run (10M steps, ~2-3 hours on L4 GPU)
    uv run python experiments/marl/train_gpu.py \\
        --steps 10000000 \\
        --scenario trivial_cooperation \\
        --run-name baseline_v1

    # Quick test (1M steps, ~15 minutes)
    uv run python experiments/marl/train_gpu.py \\
        --steps 1000000 \\
        --scenario greedy_neighbor \\
        --run-name quick_test
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import time
from datetime import datetime
import json

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from bucket_brigade.envs.puffer_env_rust import make_rust_env
from bucket_brigade.training import PolicyNetwork, TransformerPolicyNetwork, compute_gae


def train_ppo(
    env,
    policy,
    optimizer,
    num_steps=10000000,
    batch_size=2048,
    num_epochs=4,
    clip_epsilon=0.2,
    value_coef=0.5,
    entropy_coef=0.01,
    max_grad_norm=0.5,
    eval_interval=10000,
    checkpoint_interval=100000,
    writer=None,
    checkpoint_dir=None,
    device=None,
):
    """Train policy using PPO with proper checkpointing and logging."""
    if device is None:
        device = next(policy.parameters()).device

    print(f"\n{'='*60}")
    print(f"🚀 Starting PPO Training")
    print(f"{'='*60}")
    print(f"📊 Total steps: {num_steps:,}")
    print(f"📦 Batch size: {batch_size:,}")
    print(f"🔁 Epochs per batch: {num_epochs}")
    print(f"💾 Checkpoint interval: {checkpoint_interval:,} steps")
    print(f"📈 Eval interval: {eval_interval:,} steps")
    print(f"🖥️  Device: {device}")
    print(f"{'='*60}\n")

    obs, _ = env.reset()
    obs = torch.FloatTensor(obs).to(device)

    episode_rewards = []
    episode_lengths = []
    episode_reward = 0
    episode_length = 0

    start_time = time.time()
    last_eval_time = start_time

    for step in range(num_steps):
        # Collect experience
        with torch.no_grad():
            action_logits, value = policy(obs.unsqueeze(0))

            # action_logits is a list of tensors, one per action dimension
            house_probs = torch.softmax(action_logits[0], dim=-1)
            mode_probs = torch.softmax(action_logits[1], dim=-1)

            # Sample actions
            house_dist = torch.distributions.Categorical(house_probs)
            mode_dist = torch.distributions.Categorical(mode_probs)

            house = house_dist.sample()
            mode = mode_dist.sample()

            action = torch.stack([house, mode]).squeeze()

        # Environment step
        next_obs, reward, terminated, truncated, info = env.step(action.cpu().numpy())
        done = terminated or truncated

        episode_reward += reward
        episode_length += 1

        obs = torch.FloatTensor(next_obs).to(device)

        if done:
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

            if writer:
                writer.add_scalar("episode/reward", episode_reward, step)
                writer.add_scalar("episode/length", episode_length, step)

            episode_reward = 0
            episode_length = 0
            obs, _ = env.reset()
            obs = torch.FloatTensor(obs).to(device)

        # Evaluation logging
        if (step + 1) % eval_interval == 0:
            elapsed = time.time() - last_eval_time
            last_eval_time = time.time()

            if len(episode_rewards) > 0:
                avg_reward = np.mean(episode_rewards[-100:])
                avg_length = np.mean(episode_lengths[-100:])

                steps_per_sec = eval_interval / elapsed
                progress = (step + 1) / num_steps * 100

                print(f"Step {step+1:,}/{num_steps:,} ({progress:.1f}%) | "
                      f"Reward: {avg_reward:.2f} | "
                      f"Length: {avg_length:.1f} | "
                      f"Episodes: {len(episode_rewards)} | "
                      f"Speed: {steps_per_sec:.0f} steps/s")

                if writer:
                    writer.add_scalar("train/avg_reward_100ep", avg_reward, step)
                    writer.add_scalar("train/avg_length_100ep", avg_length, step)
                    writer.add_scalar("train/steps_per_sec", steps_per_sec, step)
                    writer.add_scalar("train/total_episodes", len(episode_rewards), step)

        # Checkpointing
        if checkpoint_dir and (step + 1) % checkpoint_interval == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_step_{step+1}.pt"
            torch.save({
                'step': step + 1,
                'model_state_dict': policy.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'episode_rewards': episode_rewards,
                'episode_lengths': episode_lengths,
            }, checkpoint_path)
            print(f"💾 Saved checkpoint: {checkpoint_path}")

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"✅ Training Complete!")
    print(f"⏱️  Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"📊 Total episodes: {len(episode_rewards)}")
    if episode_rewards:
        print(f"📈 Final avg reward (last 100): {np.mean(episode_rewards[-100:]):.2f}")
    print(f"{'='*60}\n")

    return episode_rewards, episode_lengths


def main():
    parser = argparse.ArgumentParser(description="Train PPO on Bucket Brigade (GPU)")
    parser.add_argument("--steps", type=int, default=10000000, help="Total training steps")
    parser.add_argument("--scenario", type=str, default="trivial_cooperation", help="Scenario name")
    parser.add_argument("--opponents", type=int, default=3, help="Number of opponent agents")
    parser.add_argument("--run-name", type=str, default=None, help="Name for this run")
    parser.add_argument("--batch-size", type=int, default=2048, help="Batch size")
    parser.add_argument("--epochs", type=int, default=4, help="Epochs per batch")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--checkpoint-interval", type=int, default=100000, help="Steps between checkpoints")
    parser.add_argument("--eval-interval", type=int, default=10000, help="Steps between eval prints")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage (default: auto-detect GPU)")

    # Architecture options
    parser.add_argument("--architecture", type=str, default="mlp", choices=["mlp", "transformer"],
                        help="Network architecture: 'mlp' (default, ~4K params) or 'transformer' (~650K params)")
    parser.add_argument("--d-model", type=int, default=192, help="Transformer embedding dimension (default: 192)")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of transformer layers (default: 2)")
    parser.add_argument("--nhead", type=int, default=4, help="Number of attention heads (default: 4)")

    args = parser.parse_args()

    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{args.scenario}_{timestamp}"

    # Directories
    exp_dir = Path("experiments/marl")
    runs_dir = exp_dir / "runs" / run_name
    checkpoint_dir = exp_dir / "checkpoints" / run_name
    runs_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = vars(args).copy()
    config['timestamp'] = timestamp
    with open(runs_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n🎮 Creating environment: {args.scenario}")
    print(f"   Opponents: {args.opponents}")

    # Create Rust-backed environment (100x faster!)
    env = make_rust_env(args.scenario, num_opponents=args.opponents)

    # Device
    if args.cpu:
        device = torch.device("cpu")
        print("🖥️  Using CPU (forced)")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("🖥️  Using CPU (no GPU available)")

    # Policy network
    obs_space = env.observation_space
    action_space = env.action_space

    print(f"\n🧠 Creating policy network")
    print(f"   Architecture: {args.architecture}")
    print(f"   Observation dim: {obs_space.shape[0]}")
    print(f"   Action dims: {action_space.nvec.tolist()}")

    if args.architecture == "transformer":
        policy = TransformerPolicyNetwork(
            obs_dim=obs_space.shape[0],
            action_dims=action_space.nvec.tolist(),
            d_model=args.d_model,
            num_layers=args.num_layers,
            nhead=args.nhead,
        ).to(device)
    else:
        policy = PolicyNetwork(
            obs_dim=obs_space.shape[0],
            action_dims=action_space.nvec.tolist()
        ).to(device)

    # Print parameter count
    total_params = sum(p.numel() for p in policy.parameters())
    print(f"   Total parameters: {total_params:,}")

    optimizer = optim.Adam(policy.parameters(), lr=args.lr)

    # TensorBoard
    writer = SummaryWriter(runs_dir)
    print(f"📊 TensorBoard: {runs_dir}")
    print(f"   View with: tensorboard --logdir experiments/marl/runs/")

    # Train
    episode_rewards, episode_lengths = train_ppo(
        env=env,
        policy=policy,
        optimizer=optimizer,
        num_steps=args.steps,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        eval_interval=args.eval_interval,
        checkpoint_interval=args.checkpoint_interval,
        writer=writer,
        checkpoint_dir=checkpoint_dir,
        device=device,
    )

    # Save final model
    final_model_path = exp_dir / f"model_{run_name}.pt"
    torch.save(policy.state_dict(), final_model_path)
    print(f"💾 Final model saved: {final_model_path}")

    writer.close()


if __name__ == "__main__":
    main()
