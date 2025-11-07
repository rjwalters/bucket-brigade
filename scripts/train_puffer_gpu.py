#!/usr/bin/env python3
"""
GPU-accelerated training script for Bucket Brigade using PufferLib vectorized environments.

This script leverages PufferLib's vectorized environments to run multiple game instances
in parallel on the GPU, significantly speeding up training.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import time
from torch.utils.tensorboard import SummaryWriter
import pufferlib
import pufferlib.vector

from bucket_brigade.envs.puffer_env import PufferBucketBrigade
from bucket_brigade.training import PolicyNetwork


def make_env_func(scenario=None, num_opponents=3, **kwargs):
    """Factory function to create a single environment instance."""
    from bucket_brigade.envs import get_scenario_by_name

    if scenario is None:
        scenario_obj = get_scenario_by_name("default", num_agents=num_opponents + 1)
    else:
        scenario_obj = get_scenario_by_name(scenario, num_agents=num_opponents + 1)

    # PufferBucketBrigade is now a native PufferEnv, no wrapping needed
    env = PufferBucketBrigade(scenario=scenario_obj, num_opponents=num_opponents, **kwargs)
    return env


def train_ppo_vectorized(
    vecenv,
    policy,
    optimizer,
    device,
    num_steps=1000000,
    batch_size=2048,
    num_epochs=4,
    clip_epsilon=0.2,
    value_coef=0.5,
    entropy_coef=0.01,
    max_grad_norm=0.5,
    eval_interval=1000,
    gamma=0.99,
    gae_lambda=0.95,
    writer=None,
    minibatch_size=None,
):
    """Train the policy using vectorized PPO with optimized GPU utilization."""

    num_envs = vecenv.num_envs
    single_observation_space = vecenv.single_observation_space
    single_action_space = vecenv.single_action_space

    # Calculate number of steps per environment
    num_steps_per_env = batch_size // num_envs

    # Use minibatches for GPU efficiency
    if minibatch_size is None:
        minibatch_size = min(4096, batch_size)  # Default GPU-friendly size

    if batch_size % minibatch_size != 0:
        print(
            f"⚠️  Warning: batch_size ({batch_size}) not divisible by minibatch_size ({minibatch_size})"
        )

    print(f"🚀 Starting high-throughput vectorized training")
    print(
        f"📊 Num envs: {num_envs}, Steps/env: {num_steps_per_env}, Batch: {batch_size:,}"
    )
    print(f"🎯 Minibatch size: {minibatch_size:,} (GPU batches)")
    print(f"🔧 Device: {device}")
    print(f"⚡ Target: {num_steps:,} total steps")

    # Storage for rollout data
    obs_buffer = torch.zeros(
        (num_steps_per_env, num_envs) + single_observation_space.shape
    ).to(device)
    actions_buffer = torch.zeros(
        (num_steps_per_env, num_envs, len(single_action_space.nvec))
    ).to(device)
    logprobs_buffer = torch.zeros((num_steps_per_env, num_envs)).to(device)
    rewards_buffer = torch.zeros((num_steps_per_env, num_envs)).to(device)
    dones_buffer = torch.zeros((num_steps_per_env, num_envs)).to(device)
    values_buffer = torch.zeros((num_steps_per_env, num_envs)).to(device)

    episode_rewards = deque(maxlen=100)
    current_episode_rewards = np.zeros(num_envs)

    global_step = 0
    start_time = time.time()

    # Initialize environment
    next_obs = torch.Tensor(vecenv.reset()[0]).to(device)
    next_done = torch.zeros(num_envs).to(device)

    num_updates = num_steps // batch_size
    print(f"✅ Environments initialized, starting training loop ({num_updates} updates)...", flush=True)

    for update in range(1, num_updates + 1):
        # DEBUG: Log progress every update
        if update % 10 == 0:
            print(f"[DEBUG] Starting update {update}/{num_updates}, global_step={global_step}", flush=True)

        # Collect rollout
        for step in range(num_steps_per_env):
            global_step += num_envs
            obs_buffer[step] = next_obs
            dones_buffer[step] = next_done

            # Sample action
            with torch.no_grad():
                action, logprob, _, value = policy.get_action_and_value(next_obs)
                values_buffer[step] = value
                actions_buffer[step] = action
                logprobs_buffer[step] = logprob

            # Execute action in vectorized environment
            action_list = action.cpu().numpy()
            next_obs_np, reward, terminated, truncated, infos = vecenv.step(action_list)

            rewards_buffer[step] = torch.tensor(reward).to(device)
            next_obs = torch.Tensor(next_obs_np).to(device)
            next_done = torch.Tensor(np.logical_or(terminated, truncated)).to(device)

            # Track episode rewards
            current_episode_rewards += reward
            for idx in range(num_envs):
                if terminated[idx] or truncated[idx]:
                    episode_rewards.append(current_episode_rewards[idx])
                    current_episode_rewards[idx] = 0

        # Bootstrap value if not done
        with torch.no_grad():
            next_value = policy.get_action_and_value(next_obs)[3]
            advantages = torch.zeros_like(rewards_buffer).to(device)
            lastgaelam = 0
            for t in reversed(range(num_steps_per_env)):
                if t == num_steps_per_env - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones_buffer[t + 1]
                    nextvalues = values_buffer[t + 1]
                delta = (
                    rewards_buffer[t]
                    + gamma * nextvalues * nextnonterminal
                    - values_buffer[t]
                )
                advantages[t] = lastgaelam = (
                    delta + gamma * gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values_buffer

        # Flatten the batch
        b_obs = obs_buffer.reshape((-1,) + single_observation_space.shape)
        b_logprobs = logprobs_buffer.reshape(-1)
        b_actions = actions_buffer.reshape((-1, len(single_action_space.nvec)))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_buffer.reshape(-1)

        # Optimizing the policy and value network using minibatches
        b_inds = np.arange(batch_size)
        clipfracs = []

        for epoch in range(num_epochs):
            np.random.shuffle(b_inds)

            # Process in minibatches for GPU efficiency
            for start in range(0, batch_size, minibatch_size):
                end = min(start + minibatch_size, batch_size)
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = policy.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > clip_epsilon).float().mean().item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if len(mb_advantages) > 1:  # Avoid division by zero with single sample
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - clip_epsilon, 1 + clip_epsilon
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - entropy_coef * entropy_loss + v_loss * value_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                optimizer.step()

        # Logging
        if writer is not None and global_step % 100 == 0:
            writer.add_scalar("train/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("train/value_loss", v_loss.item(), global_step)
            writer.add_scalar("train/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("train/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("train/clipfrac", np.mean(clipfracs), global_step)
            if episode_rewards:
                writer.add_scalar(
                    "episode/mean_reward", np.mean(episode_rewards), global_step
                )

        if global_step % eval_interval == 0:
            elapsed = time.time() - start_time
            fps = global_step / elapsed
            avg_reward = np.mean(episode_rewards) if episode_rewards else 0
            eta_seconds = (num_steps - global_step) / fps if fps > 0 else 0
            eta_minutes = eta_seconds / 60

            print(
                f"\n{'=' * 60}\n"
                f"📊 Step {global_step:,}/{num_steps:,} ({100 * global_step / num_steps:.1f}%)\n"
                f"   Avg Reward: {avg_reward:.2f} | Episodes: {len(episode_rewards)}\n"
                f"   FPS: {fps:.0f} | Elapsed: {elapsed / 60:.1f}m | ETA: {eta_minutes:.1f}m\n"
                f"   Policy Loss: {pg_loss.item():.4f} | Value Loss: {v_loss.item():.4f}\n"
                f"{'=' * 60}\n",
                flush=True,
            )

        # Save periodic checkpoints every 500K steps
        if global_step > 0 and global_step % 500000 == 0:
            checkpoint_path = f"{writer.log_dir}/checkpoint_{global_step}.pt"
            torch.save(
                {
                    "policy_state_dict": policy.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "global_step": global_step,
                    "avg_reward": np.mean(episode_rewards) if episode_rewards else 0,
                },
                checkpoint_path,
            )
            print(f"💾 Checkpoint saved: {checkpoint_path}", flush=True)

    print(
        f"\n{'=' * 60}\n"
        f"✅ Training complete!\n"
        f"   Total time: {(time.time() - start_time) / 60:.1f} minutes\n"
        f"   Final avg reward: {np.mean(episode_rewards) if episode_rewards else 0:.2f}\n"
        f"   Total episodes: {len(episode_rewards)}\n"
        f"{'=' * 60}\n",
        flush=True,
    )

    return policy


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Train Bucket Brigade policy with vectorized PPO"
    )
    parser.add_argument(
        "--num-envs", type=int, default=8, help="Number of parallel environments"
    )
    parser.add_argument(
        "--num-opponents", type=int, default=3, help="Number of opponent agents"
    )
    parser.add_argument(
        "--num-steps", type=int, default=100000, help="Total training steps"
    )
    parser.add_argument(
        "--batch-size", type=int, default=2048, help="Batch size for training"
    )
    parser.add_argument(
        "--minibatch-size",
        type=int,
        default=None,
        help="Minibatch size for GPU (default: 4096)",
    )
    parser.add_argument(
        "--hidden-size", type=int, default=128, help="Hidden layer size"
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        choices=["fast", "balanced", "quality"],
        help="Use preset configuration (overrides other params)",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="models/puffer_gpu_policy.pt",
        help="Path to save trained model",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="default",
        help="Scenario name (e.g., 'default', 'trivial_cooperation')",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Name for TensorBoard run",
    )
    parser.add_argument(
        "--vectorization",
        type=str,
        default="serial",
        choices=["serial", "multiprocessing"],
        help="Vectorization backend (serial for debugging, multiprocessing for performance)",
    )

    args = parser.parse_args()

    # Apply presets
    if args.preset == "fast":
        # Maximum throughput: many environments, huge batches
        args.num_envs = 512
        args.batch_size = 262144  # 256K
        args.minibatch_size = 8192
        args.vectorization = "multiprocessing"
        print("🚀 Using FAST preset: 512 envs, 256K batch, 8K minibatch")
    elif args.preset == "balanced":
        # Balanced: good throughput with reasonable quality
        args.num_envs = 256
        args.batch_size = 131072  # 128K
        args.minibatch_size = 4096
        args.vectorization = "multiprocessing"
        print("⚖️  Using BALANCED preset: 256 envs, 128K batch, 4K minibatch")
    elif args.preset == "quality":
        # Quality: smaller batches, more updates
        args.num_envs = 128
        args.batch_size = 65536  # 64K
        args.minibatch_size = 4096
        args.vectorization = "multiprocessing"
        print("💎 Using QUALITY preset: 128 envs, 64K batch, 4K minibatch")

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Using device: {device}", flush=True)
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}", flush=True)
        print(
            f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB",
            flush=True,
        )

    # Create vectorized environment
    print(f"🎮 Creating {args.num_envs} vectorized environments...", flush=True)

    env_kwargs = {
        "scenario": args.scenario,
        "num_opponents": args.num_opponents,
    }

    # Choose vectorization backend
    if args.vectorization == "serial" or args.num_envs == 1:
        backend = pufferlib.vector.Serial
    else:
        backend = pufferlib.vector.Multiprocessing

    vecenv = pufferlib.vector.make(
        make_env_func,
        env_kwargs=env_kwargs,
        backend=backend,
        num_envs=args.num_envs,
        num_workers=min(args.num_envs, 8),  # Limit workers
    )

    # Create policy
    obs_dim = vecenv.single_observation_space.shape[0]
    action_dims = vecenv.single_action_space.nvec.tolist()

    print(f"🧠 Creating policy network", flush=True)
    print(f"   Observation dim: {obs_dim}")
    print(f"   Action dims: {action_dims}")

    policy = PolicyNetwork(obs_dim, action_dims, hidden_size=args.hidden_size).to(
        device
    )
    optimizer = optim.Adam(policy.parameters(), lr=args.lr)

    # Initialize TensorBoard
    if args.run_name is None:
        args.run_name = f"puffer_gpu_{args.scenario}_{int(time.time())}"

    writer = SummaryWriter(f"runs/{args.run_name}")
    print(f"📊 TensorBoard logging to: runs/{args.run_name}")

    # Train
    try:
        policy = train_ppo_vectorized(
            vecenv=vecenv,
            policy=policy,
            optimizer=optimizer,
            device=device,
            num_steps=args.num_steps,
            batch_size=args.batch_size,
            minibatch_size=args.minibatch_size,
            writer=writer,
        )
    finally:
        writer.close()
        vecenv.close()

    # Save model
    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "policy_state_dict": policy.state_dict(),
            "obs_dim": obs_dim,
            "action_dims": action_dims,
            "hidden_size": args.hidden_size,
        },
        args.save_path,
    )

    print(f"💾 Model saved to {args.save_path}")


if __name__ == "__main__":
    main()
