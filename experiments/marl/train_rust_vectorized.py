#!/usr/bin/env python3
"""
Rust-Vectorized PPO Training - Maximum Performance.

Uses PyVectorEnv from bucket-brigade-core for ultra-fast environment stepping.
All environments processed in a single Rust call, eliminating Python overhead.

This should provide 10-100x faster env stepping compared to Python-based vectorization,
while maintaining the 97% GPU utilization from the large model approach.

Usage:
    # Full training (10M steps)
    uv run python experiments/marl/train_rust_vectorized.py \
        --total-timesteps 10000000 \
        --scenario trivial_cooperation \
        --num-envs 256 \
        --run-name rust_vec_v1

    # Quick test
    uv run python experiments/marl/train_rust_vectorized.py \
        --total-timesteps 100000 \
        --scenario trivial_cooperation \
        --num-envs 64 \
        --run-name quick_test
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
from datetime import datetime
import json
import time

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from bucket_brigade_core import VectorEnv, SCENARIOS
from bucket_brigade.training import PolicyNetwork, compute_gae


def collect_rollout(
    vecenv,
    policy,
    num_steps,
    device,
):
    """Collect a rollout from vectorized environment."""
    num_envs = vecenv.num_envs

    # Storage
    obs_batch = []
    action_batch = []
    reward_batch = []
    done_batch = []
    value_batch = []
    logprob_batch = []

    # Get initial observations
    obs = vecenv.reset()
    obs_tensor = torch.FloatTensor(obs).to(device)  # [num_envs, num_agents, obs_dim]

    for step in range(num_steps):
        with torch.no_grad():
            # Forward pass for all environments at once
            action_logits, values = policy(obs_tensor)

            # Sample actions for all envs
            house_probs = torch.softmax(action_logits[0], dim=-1)
            mode_probs = torch.softmax(action_logits[1], dim=-1)

            house_dist = torch.distributions.Categorical(house_probs)
            mode_dist = torch.distributions.Categorical(mode_probs)

            houses = house_dist.sample()
            modes = mode_dist.sample()

            actions = torch.stack([houses, modes], dim=-1)  # [num_envs, 2]

            # Calculate log probabilities
            house_logprobs = house_dist.log_prob(houses)
            mode_logprobs = mode_dist.log_prob(modes)
            total_logprobs = house_logprobs + mode_logprobs

        # Store data
        obs_batch.append(obs_tensor.cpu())
        action_batch.append(actions.cpu())
        value_batch.append(values.squeeze(-1).cpu())
        logprob_batch.append(total_logprobs.cpu())

        # Step all environments at once in Rust
        actions_list = actions.cpu().numpy().tolist()
        next_obs, rewards, dones, infos = vecenv.step(actions_list)

        # Store rewards and dones
        reward_batch.append(torch.FloatTensor(rewards))
        done_batch.append(torch.BoolTensor(dones))

        # Update observations
        obs_tensor = torch.FloatTensor(next_obs).to(device)

    # Stack everything
    obs_batch = torch.stack(obs_batch)  # [num_steps, num_envs, num_agents, obs_dim]
    action_batch = torch.stack(action_batch)  # [num_steps, num_envs, 2]
    reward_batch = torch.stack(reward_batch)  # [num_steps, num_envs]
    done_batch = torch.stack(done_batch)  # [num_steps, num_envs]
    value_batch = torch.stack(value_batch)  # [num_steps, num_envs]
    logprob_batch = torch.stack(logprob_batch)  # [num_steps, num_envs]

    return {
        'obs': obs_batch,
        'actions': action_batch,
        'rewards': reward_batch,
        'dones': done_batch,
        'values': value_batch,
        'logprobs': logprob_batch,
    }


def train_ppo(
    vecenv,
    policy,
    optimizer,
    total_timesteps,
    num_steps=256,
    num_epochs=4,
    num_minibatches=4,
    clip_epsilon=0.2,
    value_coef=0.5,
    entropy_coef=0.01,
    max_grad_norm=0.5,
    gamma=0.99,
    gae_lambda=0.95,
    checkpoint_interval=100000,
    eval_interval=10000,
    writer=None,
    checkpoint_dir=None,
    device=None,
):
    """Train policy using PPO with Rust-vectorized environments."""
    if device is None:
        device = next(policy.parameters()).device

    num_envs = vecenv.num_envs
    num_updates = total_timesteps // (num_steps * num_envs)
    minibatch_size = (num_steps * num_envs) // num_minibatches

    print(f"\n{'='*60}")
    print(f"🚀 Starting Rust-Vectorized PPO Training")
    print(f"{'='*60}")
    print(f"📊 Total timesteps: {total_timesteps:,}")
    print(f"🌍 Num environments: {num_envs}")
    print(f"📦 Steps per rollout: {num_steps}")
    print(f"🔁 Update epochs: {num_epochs}")
    print(f"📦 Minibatches: {num_minibatches} (size {minibatch_size})")
    print(f"💾 Checkpoint interval: {checkpoint_interval:,} steps")
    print(f"📈 Eval interval: {eval_interval:,} steps")
    print(f"🖥️  Device: {device}")
    print(f"⚡ Using Rust VectorEnv for maximum speed!")
    print(f"{'='*60}\n")

    global_step = 0
    episode_rewards = []
    start_time = time.time()
    last_eval_time = start_time

    for update in range(num_updates):
        rollout_start = time.time()

        # Collect rollout (all envs processed in Rust)
        rollout = collect_rollout(vecenv, policy, num_steps, device)

        rollout_time = time.time() - rollout_start

        # Calculate advantages using GAE
        with torch.no_grad():
            next_obs = vecenv.reset()  # Get final observations
            next_obs_tensor = torch.FloatTensor(next_obs).to(device)
            _, next_values = policy(next_obs_tensor)
            next_values = next_values.squeeze(-1).cpu()

        advantages = compute_gae(
            rollout['rewards'],
            rollout['values'],
            rollout['dones'],
            next_values,
            gamma,
            gae_lambda,
        )
        returns = advantages + rollout['values']

        # Flatten batch dimensions
        b_obs = rollout['obs'].reshape(-1, *rollout['obs'].shape[2:])
        b_actions = rollout['actions'].reshape(-1, *rollout['actions'].shape[2:])
        b_logprobs = rollout['logprobs'].reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = rollout['values'].reshape(-1)

        # Normalize advantages
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        # PPO update
        update_start = time.time()

        for epoch in range(num_epochs):
            # Shuffle and create minibatches
            indices = torch.randperm(b_obs.shape[0])

            for start in range(0, b_obs.shape[0], minibatch_size):
                end = start + minibatch_size
                mb_indices = indices[start:end]

                mb_obs = b_obs[mb_indices].to(device)
                mb_actions = b_actions[mb_indices].to(device)
                mb_logprobs = b_logprobs[mb_indices].to(device)
                mb_advantages = b_advantages[mb_indices].to(device)
                mb_returns = b_returns[mb_indices].to(device)

                # Forward pass
                action_logits, values = policy(mb_obs)
                values = values.squeeze(-1)

                # Calculate log probabilities
                house_probs = torch.softmax(action_logits[0], dim=-1)
                mode_probs = torch.softmax(action_logits[1], dim=-1)

                house_dist = torch.distributions.Categorical(house_probs)
                mode_dist = torch.distributions.Categorical(mode_probs)

                house_logprobs = house_dist.log_prob(mb_actions[:, 0])
                mode_logprobs = mode_dist.log_prob(mb_actions[:, 1])
                new_logprobs = house_logprobs + mode_logprobs

                # PPO loss
                ratio = torch.exp(new_logprobs - mb_logprobs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.functional.mse_loss(values, mb_returns)

                # Entropy bonus
                entropy = (house_dist.entropy() + mode_dist.entropy()).mean()

                # Total loss
                loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

                # Optimize
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                optimizer.step()

        update_time = time.time() - update_start

        global_step += num_steps * num_envs

        # Logging
        if writer:
            writer.add_scalar("train/policy_loss", policy_loss.item(), global_step)
            writer.add_scalar("train/value_loss", value_loss.item(), global_step)
            writer.add_scalar("train/entropy", entropy.item(), global_step)
            writer.add_scalar("train/rollout_time", rollout_time, global_step)
            writer.add_scalar("train/update_time", update_time, global_step)
            writer.add_scalar("train/mean_reward", rollout['rewards'].mean().item(), global_step)

        # Evaluation logging
        if (update + 1) % (eval_interval // (num_steps * num_envs)) == 0:
            elapsed = time.time() - last_eval_time
            last_eval_time = time.time()

            steps_per_sec = eval_interval / elapsed
            progress = global_step / total_timesteps * 100
            mean_reward = rollout['rewards'].mean().item()

            print(f"Step {global_step:,}/{total_timesteps:,} ({progress:.1f}%) | "
                  f"Reward: {mean_reward:.3f} | "
                  f"Rollout: {rollout_time:.2f}s | "
                  f"Update: {update_time:.2f}s | "
                  f"Speed: {steps_per_sec:.0f} steps/s")

        # Checkpointing
        if checkpoint_dir and (global_step % checkpoint_interval == 0):
            checkpoint_path = checkpoint_dir / f"checkpoint_step_{global_step}.pt"
            torch.save({
                'step': global_step,
                'model_state_dict': policy.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)
            print(f"💾 Saved checkpoint: {checkpoint_path}")

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"✅ Training Complete!")
    print(f"⏱️  Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"⚡ Avg speed: {total_timesteps/total_time:.0f} steps/s")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Train PPO with Rust VectorEnv")
    parser.add_argument("--total-timesteps", type=int, default=10000000, help="Total training timesteps")
    parser.add_argument("--scenario", type=str, default="trivial_cooperation", help="Scenario name")
    parser.add_argument("--num-envs", type=int, default=256, help="Number of parallel environments")
    parser.add_argument("--num-agents", type=int, default=4, help="Number of agents per environment")
    parser.add_argument("--num-steps", type=int, default=256, help="Steps per rollout")
    parser.add_argument("--num-epochs", type=int, default=4, help="PPO epochs per update")
    parser.add_argument("--num-minibatches", type=int, default=4, help="Number of minibatches")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--hidden-size", type=int, default=4096, help="Hidden layer size (large for GPU util)")
    parser.add_argument("--checkpoint-interval", type=int, default=100000, help="Steps between checkpoints")
    parser.add_argument("--eval-interval", type=int, default=10000, help="Steps between eval prints")
    parser.add_argument("--run-name", type=str, default=None, help="Run name")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")

    args = parser.parse_args()

    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{args.scenario}_rust_vec_{timestamp}"

    # Directories
    exp_dir = Path("experiments/marl")
    runs_dir = exp_dir / "runs" / run_name
    checkpoint_dir = exp_dir / "checkpoints" / run_name
    runs_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = vars(args).copy()
    config['timestamp'] = timestamp
    config['rust_vectorized'] = True
    with open(runs_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n🎮 Creating Rust VectorEnv")
    print(f"   Scenario: {args.scenario}")
    print(f"   Num environments: {args.num_envs}")
    print(f"   Num agents: {args.num_agents}")
    print(f"   Seed: {args.seed}")

    # Create Rust-vectorized environment
    scenario = SCENARIOS[args.scenario]
    vecenv = VectorEnv(scenario, args.num_envs, args.num_agents, seed=args.seed)

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

    # Policy network (large for GPU utilization)
    print(f"\n🧠 Creating policy network")
    print(f"   Hidden size: {args.hidden_size} (large for GPU utilization)")

    # Get observation shape from a test reset
    test_obs = vecenv.reset()
    obs_dim = len(test_obs[0])  # Flattened observation dimension

    # Action space: house selection and mode
    action_dims = [scenario.num_houses, 3]  # 3 modes: idle, fill, pass

    print(f"   Observation dim: {obs_dim}")
    print(f"   Action dims: {action_dims}")

    policy = PolicyNetwork(
        obs_dim=obs_dim,
        action_dims=action_dims,
        hidden_size=args.hidden_size,
    ).to(device)

    optimizer = optim.Adam(policy.parameters(), lr=args.learning_rate)

    # TensorBoard
    writer = SummaryWriter(runs_dir)
    print(f"📊 TensorBoard: {runs_dir}")
    print(f"   View with: tensorboard --logdir experiments/marl/runs/")

    # Train
    train_ppo(
        vecenv=vecenv,
        policy=policy,
        optimizer=optimizer,
        total_timesteps=args.total_timesteps,
        num_steps=args.num_steps,
        num_epochs=args.num_epochs,
        num_minibatches=args.num_minibatches,
        checkpoint_interval=args.checkpoint_interval,
        eval_interval=args.eval_interval,
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
