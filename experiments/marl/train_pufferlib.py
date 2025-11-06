#!/usr/bin/env python3
"""
PufferLib Native Training - Proper 80-90% GPU Utilization.

Uses PufferLib's proven vectorization and PPO implementation for maximum efficiency.

Usage:
    # Full training (10M steps, ~30-45 minutes)
    uv run python experiments/marl/train_pufferlib.py \\
        --total-timesteps 10000000 \\
        --scenario trivial_cooperation \\
        --run-name baseline_puffer_v1

    # Quick test
    uv run python experiments/marl/train_pufferlib.py \\
        --total-timesteps 100000 \\
        --scenario trivial_cooperation \\
        --run-name quick_test
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
from datetime import datetime
import json

import torch
import pufferlib
import pufferlib.vector
import pufferlib.frameworks.cleanrl

from bucket_brigade.envs.puffer_env_rust import make_rust_env
from bucket_brigade.training import PolicyNetwork


def main():
    parser = argparse.ArgumentParser(description="PufferLib Native Training")
    parser.add_argument("--total-timesteps", type=int, default=10000000, help="Total training timesteps")
    parser.add_argument("--scenario", type=str, default="trivial_cooperation", help="Scenario name")
    parser.add_argument("--opponents", type=int, default=3, help="Number of opponents")
    parser.add_argument("--num-envs", type=int, default=64, help="Parallel environments")
    parser.add_argument("--num-cores", type=int, default=1, help="CPU cores (1=serial, >1=multiprocessing)")
    parser.add_argument("--batch-size", type=int, default=65536, help="Batch size (num_envs * num_steps)")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--run-name", type=str, default=None, help="Run name")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")

    args = parser.parse_args()

    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{args.scenario}_puffer_{timestamp}"

    # Directories
    exp_dir = Path("experiments/marl")
    runs_dir = exp_dir / "runs" / run_name
    checkpoint_dir = exp_dir / "checkpoints" / run_name
    runs_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = vars(args).copy()
    config['timestamp'] = timestamp
    config['pufferlib_native'] = True
    with open(runs_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n🎮 Creating PufferLib vectorized environment")
    print(f"   Scenario: {args.scenario}")
    print(f"   Opponents: {args.opponents}")
    print(f"   Num envs: {args.num_envs}")
    print(f"   Cores: {args.num_cores} ({'serial' if args.num_cores == 1 else 'multiprocessing'})")

    # Environment creator
    def env_creator(name=None):
        return make_rust_env(args.scenario, num_opponents=args.opponents)

    # Create vectorized environment using PufferLib
    backend = 'serial' if args.num_cores == 1 else 'multiprocessing'
    vecenv = pufferlib.vector.make(
        env_creator,
        num_envs=args.num_envs,
        num_cores=args.num_cores,
        backend=backend,
    )

    # Device
    if args.cpu:
        device = "cpu"
        print(f"🖥️  Using CPU (forced)")
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print(f"🖥️  Using CPU (no GPU available)")

    # Policy network
    single_env = env_creator()
    obs_space = single_env.observation_space
    action_space = single_env.action_space

    print(f"\n🧠 Creating policy network")
    print(f"   Observation dim: {obs_space.shape[0]}")
    print(f"   Action dims: {action_space.nvec.tolist()}")

    policy = PolicyNetwork(
        obs_dim=obs_space.shape[0],
        action_dims=action_space.nvec.tolist()
    ).to(device)

    print(f"\n📊 TensorBoard: {runs_dir}")
    print(f"   View with: tensorboard --logdir experiments/marl/runs/")

    # PufferLib CleanRL PPO config
    config = pufferlib.namespace(
        # Environment settings
        num_envs=args.num_envs,
        env_pool=False,

        # Training settings
        total_timesteps=args.total_timesteps,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        batch_rows=args.batch_size // args.num_envs,  # num_steps
        bptt_horizon=16,

        # PPO settings
        num_minibatches=4,
        update_epochs=4,
        clip_coef=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
        max_grad_norm=0.5,
        gamma=0.99,
        gae_lambda=0.95,

        # Logging
        log_frequency=100,
        eval_frequency=0,
        checkpoint_interval=100000,

        # Paths
        runs_dir=str(runs_dir),
        checkpoint_dir=str(checkpoint_dir),

        # Device
        device=device,
        compile=False,
        compile_mode='default',
    )

    print(f"\n{'='*60}")
    print(f"🚀 Starting PufferLib CleanRL PPO Training")
    print(f"{'='*60}")
    print(f"📊 Total timesteps: {args.total_timesteps:,}")
    print(f"🌍 Parallel environments: {args.num_envs}")
    print(f"📦 Batch size: {args.batch_size:,}")
    print(f"📦 Batch rows (steps per env): {config.batch_rows}")
    print(f"🔁 Update epochs: {config.update_epochs}")
    print(f"🖥️  Device: {device}")
    print(f"⚡ Expected GPU util: 80-90%")
    print(f"{'='*60}\n")

    # Train using PufferLib's CleanRL PPO
    data = pufferlib.frameworks.cleanrl.train(
        vecenv=vecenv,
        agent=policy,
        config=config,
    )

    # Save final model
    final_model_path = exp_dir / f"model_{run_name}.pt"
    torch.save(policy.state_dict(), final_model_path)
    print(f"\n💾 Final model saved: {final_model_path}")

    vecenv.close()


if __name__ == "__main__":
    main()
