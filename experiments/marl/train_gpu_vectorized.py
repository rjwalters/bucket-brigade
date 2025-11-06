#!/usr/bin/env python3
"""
Vectorized GPU PPO Training for Bucket Brigade - Optimized for 80-90% GPU utilization.

Uses vectorized environments to collect experiences in parallel, maximizing GPU throughput.

Usage:
    # Full training (10M steps, ~30-45 minutes on L4)
    uv run python experiments/marl/train_gpu_vectorized.py \\
        --steps 10000000 \\
        --scenario trivial_cooperation \\
        --num-envs 64 \\
        --run-name baseline_v1

    # Quick test
    uv run python experiments/marl/train_gpu_vectorized.py \\
        --steps 100000 \\
        --scenario trivial_cooperation \\
        --num-envs 32 \\
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
from bucket_brigade.training import PolicyNetwork


class VectorizedEnv:
    """Simple vectorized environment wrapper."""

    def __init__(self, env_fn, num_envs):
        self.num_envs = num_envs
        self.envs = [env_fn() for _ in range(num_envs)]
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

    def reset(self):
        obs = np.stack([env.reset()[0] for env in self.envs])
        return obs

    def step(self, actions):
        """Step all environments with vectorized actions."""
        results = [env.step(action) for env, action in zip(self.envs, actions)]

        obs = np.stack([r[0] for r in results])
        rewards = np.array([r[1] for r in results])
        terminateds = np.array([r[2] for r in results])
        truncateds = np.array([r[3] for r in results])
        infos = [r[4] for r in results]

        # Auto-reset done environments
        for i, (term, trunc) in enumerate(zip(terminateds, truncateds)):
            if term or trunc:
                obs[i] = self.envs[i].reset()[0]

        return obs, rewards, terminateds, truncateds, infos


def train_ppo_vectorized(
    vec_env,
    policy,
    optimizer,
    num_steps=10000000,
    rollout_length=128,
    num_epochs=4,
    minibatch_size=256,
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
    """Vectorized PPO training with proper batching for GPU utilization."""
    if device is None:
        device = next(policy.parameters()).device

    num_envs = vec_env.num_envs

    print(f"\n{'='*60}")
    print(f"🚀 Starting Vectorized PPO Training")
    print(f"{'='*60}")
    print(f"📊 Total steps: {num_steps:,}")
    print(f"🌍 Parallel environments: {num_envs}")
    print(f"📦 Rollout length: {rollout_length}")
    print(f"📦 Minibatch size: {minibatch_size}")
    print(f"🔁 Epochs per batch: {num_epochs}")
    print(f"💾 Checkpoint interval: {checkpoint_interval:,} steps")
    print(f"📈 Eval interval: {eval_interval:,} steps")
    print(f"🖥️  Device: {device}")
    print(f"⚡ Expected GPU util: 80-90%")
    print(f"{'='*60}\n")

    # Initialize
    obs = vec_env.reset()
    obs_tensor = torch.FloatTensor(obs).to(device)

    # Episode tracking
    episode_rewards = [0.0] * num_envs
    episode_lengths = [0] * num_envs
    completed_rewards = []
    completed_lengths = []

    # Rollout buffers
    rollout_obs = []
    rollout_actions = []
    rollout_logprobs = []
    rollout_values = []
    rollout_rewards = []
    rollout_dones = []

    start_time = time.time()
    last_eval_time = start_time
    total_env_steps = 0

    while total_env_steps < num_steps:
        # === ROLLOUT PHASE: Collect experiences ===
        for _ in range(rollout_length):
            with torch.no_grad():
                # Batch forward pass through policy
                action_logits, values = policy(obs_tensor)

                # Convert to probabilities
                house_probs = torch.softmax(action_logits[0], dim=-1)
                mode_probs = torch.softmax(action_logits[1], dim=-1)

                # Sample actions (batched)
                house_dist = torch.distributions.Categorical(house_probs)
                mode_dist = torch.distributions.Categorical(mode_probs)

                house_actions = house_dist.sample()
                mode_actions = mode_dist.sample()

                # Compute log probs
                house_logprobs = house_dist.log_prob(house_actions)
                mode_logprobs = mode_dist.log_prob(mode_actions)
                total_logprobs = house_logprobs + mode_logprobs

                actions = torch.stack([house_actions, mode_actions], dim=1)

            # Step environments (all at once)
            next_obs, rewards, terminateds, truncateds, infos = vec_env.step(actions.cpu().numpy())
            dones = terminateds | truncateds

            # Store rollout data
            rollout_obs.append(obs_tensor)
            rollout_actions.append(actions)
            rollout_logprobs.append(total_logprobs)
            rollout_values.append(values.squeeze(-1))
            rollout_rewards.append(torch.FloatTensor(rewards).to(device))
            rollout_dones.append(torch.FloatTensor(dones).to(device))

            # Update episode tracking
            for i in range(num_envs):
                episode_rewards[i] += rewards[i]
                episode_lengths[i] += 1

                if dones[i]:
                    completed_rewards.append(episode_rewards[i])
                    completed_lengths.append(episode_lengths[i])
                    episode_rewards[i] = 0.0
                    episode_lengths[i] = 0

                    if writer:
                        writer.add_scalar("episode/reward", completed_rewards[-1], total_env_steps)
                        writer.add_scalar("episode/length", completed_lengths[-1], total_env_steps)

            obs_tensor = torch.FloatTensor(next_obs).to(device)
            total_env_steps += num_envs

            if total_env_steps >= num_steps:
                break

        # === LEARNING PHASE: Update policy ===
        if len(rollout_obs) > 0:
            # Stack rollout data
            rollout_obs_batch = torch.stack(rollout_obs)  # [rollout_length, num_envs, obs_dim]
            rollout_actions_batch = torch.stack(rollout_actions)
            rollout_logprobs_batch = torch.stack(rollout_logprobs)
            rollout_values_batch = torch.stack(rollout_values)
            rollout_rewards_batch = torch.stack(rollout_rewards)
            rollout_dones_batch = torch.stack(rollout_dones)

            # Compute advantages using GAE
            with torch.no_grad():
                _, next_values = policy(obs_tensor)
                next_values = next_values.squeeze(-1)

                # Compute GAE advantages (tensor version)
                T, B = rollout_rewards_batch.shape  # [rollout_length, num_envs]
                advantages = torch.zeros_like(rollout_rewards_batch)
                lastgaelam = 0

                for t in reversed(range(T)):
                    if t == T - 1:
                        nextnonterminal = 1.0 - rollout_dones_batch[t]
                        nextvalues = next_values
                    else:
                        nextnonterminal = 1.0 - rollout_dones_batch[t + 1]
                        nextvalues = rollout_values_batch[t + 1]

                    delta = rollout_rewards_batch[t] + 0.99 * nextvalues * nextnonterminal - rollout_values_batch[t]
                    advantages[t] = lastgaelam = delta + 0.99 * 0.95 * nextnonterminal * lastgaelam
            returns = advantages + rollout_values_batch

            # Flatten batch [rollout_length, num_envs, ...] -> [rollout_length * num_envs, ...]
            b_obs = rollout_obs_batch.reshape(-1, rollout_obs_batch.shape[-1])
            b_actions = rollout_actions_batch.reshape(-1, rollout_actions_batch.shape[-1])
            b_logprobs = rollout_logprobs_batch.reshape(-1)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = rollout_values_batch.reshape(-1)

            # Normalize advantages
            b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

            # Multiple epochs over the data
            batch_size = b_obs.shape[0]
            for epoch in range(num_epochs):
                # Shuffle indices
                indices = torch.randperm(batch_size, device=device)

                # Process minibatches
                for start in range(0, batch_size, minibatch_size):
                    end = start + minibatch_size
                    mb_indices = indices[start:end]

                    # Get minibatch
                    mb_obs = b_obs[mb_indices]
                    mb_actions = b_actions[mb_indices]
                    mb_logprobs = b_logprobs[mb_indices]
                    mb_advantages = b_advantages[mb_indices]
                    mb_returns = b_returns[mb_indices]
                    mb_values = b_values[mb_indices]

                    # Forward pass
                    action_logits, new_values = policy(mb_obs)
                    new_values = new_values.squeeze(-1)

                    # Recompute log probs
                    house_probs = torch.softmax(action_logits[0], dim=-1)
                    mode_probs = torch.softmax(action_logits[1], dim=-1)

                    house_dist = torch.distributions.Categorical(house_probs)
                    mode_dist = torch.distributions.Categorical(mode_probs)

                    new_house_logprobs = house_dist.log_prob(mb_actions[:, 0])
                    new_mode_logprobs = mode_dist.log_prob(mb_actions[:, 1])
                    new_logprobs = new_house_logprobs + new_mode_logprobs

                    # PPO loss
                    logratio = new_logprobs - mb_logprobs
                    ratio = logratio.exp()

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    value_loss = 0.5 * ((new_values - mb_returns) ** 2).mean()

                    # Entropy bonus
                    entropy = (house_dist.entropy() + mode_dist.entropy()).mean()

                    # Total loss
                    loss = pg_loss + value_coef * value_loss - entropy_coef * entropy

                    # Optimize
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                    optimizer.step()

            # Clear rollout buffers
            rollout_obs.clear()
            rollout_actions.clear()
            rollout_logprobs.clear()
            rollout_values.clear()
            rollout_rewards.clear()
            rollout_dones.clear()

        # === LOGGING ===
        if total_env_steps % eval_interval < num_envs * rollout_length:
            elapsed = time.time() - last_eval_time
            last_eval_time = time.time()

            if len(completed_rewards) > 0:
                avg_reward = np.mean(completed_rewards[-100:])
                avg_length = np.mean(completed_lengths[-100:])

                steps_per_sec = eval_interval / elapsed if elapsed > 0 else 0
                progress = total_env_steps / num_steps * 100

                print(f"Step {total_env_steps:,}/{num_steps:,} ({progress:.1f}%) | "
                      f"Reward: {avg_reward:.2f} | "
                      f"Length: {avg_length:.1f} | "
                      f"Episodes: {len(completed_rewards)} | "
                      f"Speed: {steps_per_sec:.0f} steps/s")

                if writer:
                    writer.add_scalar("train/avg_reward_100ep", avg_reward, total_env_steps)
                    writer.add_scalar("train/avg_length_100ep", avg_length, total_env_steps)
                    writer.add_scalar("train/steps_per_sec", steps_per_sec, total_env_steps)
                    writer.add_scalar("train/total_episodes", len(completed_rewards), total_env_steps)

        # === CHECKPOINTING ===
        if checkpoint_dir and total_env_steps % checkpoint_interval < num_envs * rollout_length:
            checkpoint_path = checkpoint_dir / f"checkpoint_step_{total_env_steps}.pt"
            torch.save({
                'step': total_env_steps,
                'model_state_dict': policy.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'episode_rewards': completed_rewards,
                'episode_lengths': completed_lengths,
            }, checkpoint_path)
            print(f"💾 Saved checkpoint: {checkpoint_path}")

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"✅ Training Complete!")
    print(f"⏱️  Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"📊 Total episodes: {len(completed_rewards)}")
    if completed_rewards:
        print(f"📈 Final avg reward (last 100): {np.mean(completed_rewards[-100:]):.2f}")
    print(f"⚡ Average speed: {total_env_steps/total_time:.0f} steps/s")
    print(f"{'='*60}\n")

    return completed_rewards, completed_lengths


def main():
    parser = argparse.ArgumentParser(description="Vectorized PPO Training (High GPU Utilization)")
    parser.add_argument("--steps", type=int, default=10000000, help="Total training steps")
    parser.add_argument("--scenario", type=str, default="trivial_cooperation", help="Scenario name")
    parser.add_argument("--opponents", type=int, default=3, help="Number of opponent agents")
    parser.add_argument("--num-envs", type=int, default=64, help="Number of parallel environments")
    parser.add_argument("--rollout-length", type=int, default=128, help="Steps per rollout")
    parser.add_argument("--minibatch-size", type=int, default=256, help="Minibatch size for updates")
    parser.add_argument("--epochs", type=int, default=4, help="Epochs per batch")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--hidden-size", type=int, default=2048, help="Hidden layer size (larger = more GPU usage)")
    parser.add_argument("--run-name", type=str, default=None, help="Name for this run")
    parser.add_argument("--checkpoint-interval", type=int, default=100000, help="Steps between checkpoints")
    parser.add_argument("--eval-interval", type=int, default=10000, help="Steps between eval prints")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage")

    args = parser.parse_args()

    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{args.scenario}_vec_{timestamp}"

    # Directories
    exp_dir = Path("experiments/marl")
    runs_dir = exp_dir / "runs" / run_name
    checkpoint_dir = exp_dir / "checkpoints" / run_name
    runs_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = vars(args).copy()
    config['timestamp'] = timestamp
    config['vectorized'] = True
    with open(runs_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n🎮 Creating vectorized environment: {args.scenario}")
    print(f"   Opponents: {args.opponents}")
    print(f"   Parallel envs: {args.num_envs}")

    # Create vectorized environment
    def env_fn():
        return make_rust_env(args.scenario, num_opponents=args.opponents)

    vec_env = VectorizedEnv(env_fn, args.num_envs)

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
    obs_space = vec_env.observation_space
    action_space = vec_env.action_space

    print(f"\n🧠 Creating policy network")
    print(f"   Observation dim: {obs_space.shape[0]}")
    print(f"   Action dims: {action_space.nvec.tolist()}")

    # Use large hidden size to saturate GPU
    policy = PolicyNetwork(
        obs_dim=obs_space.shape[0],
        action_dims=action_space.nvec.tolist(),
        hidden_size=args.hidden_size
    ).to(device)

    total_params = sum(p.numel() for p in policy.parameters())
    print(f"   Model parameters: {total_params:,} ({total_params * 4 / 1024 / 1024:.1f} MB)")

    optimizer = optim.Adam(policy.parameters(), lr=args.lr)

    # TensorBoard
    writer = SummaryWriter(runs_dir)
    print(f"📊 TensorBoard: {runs_dir}")
    print(f"   View with: tensorboard --logdir experiments/marl/runs/")

    # Train
    episode_rewards, episode_lengths = train_ppo_vectorized(
        vec_env=vec_env,
        policy=policy,
        optimizer=optimizer,
        num_steps=args.steps,
        rollout_length=args.rollout_length,
        minibatch_size=args.minibatch_size,
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
