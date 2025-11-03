#!/usr/bin/env python3
"""
Multi-GPU training script for Bucket Brigade using PyTorch Distributed.

Usage:
    # Single node, 4 GPUs:
    torchrun --nproc_per_node=4 scripts/train_multi_gpu.py --num-steps 10000000

    # Or with python -m:
    python -m torch.distributed.run --nproc_per_node=4 scripts/train_multi_gpu.py --num-steps 10000000
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Import the existing training components
from scripts.train_puffer_gpu import PolicyNetwork, make_env_func, train_ppo_vectorized

import numpy as np
import pufferlib
import pufferlib.vector


def setup_distributed():
    """Initialize distributed training."""
    # torchrun sets these environment variables
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])

        # Initialize process group
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

        is_master = rank == 0
        device = torch.device(f"cuda:{local_rank}")

        if is_master:
            print(f"🌐 Distributed training initialized: {world_size} GPUs")
            print(f"   Rank {rank}/{world_size-1}, Device: {device}")

        return {
            "distributed": True,
            "device": device,
            "rank": rank,
            "world_size": world_size,
            "local_rank": local_rank,
            "is_master": is_master,
        }
    else:
        # Single GPU fallback
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return {
            "distributed": False,
            "device": device,
            "rank": 0,
            "world_size": 1,
            "local_rank": 0,
            "is_master": True,
        }


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    import argparse
    from torch.utils.tensorboard import SummaryWriter
    import time

    parser = argparse.ArgumentParser(description="Multi-GPU training for Bucket Brigade")
    parser.add_argument("--num-envs-per-gpu", type=int, default=32, help="Environments per GPU")
    parser.add_argument("--num-opponents", type=int, default=3, help="Number of opponents")
    parser.add_argument("--num-steps", type=int, default=10000000, help="Total training steps")
    parser.add_argument("--batch-size-per-gpu", type=int, default=4096, help="Batch size per GPU")
    parser.add_argument("--hidden-size", type=int, default=128, help="Hidden layer size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--save-path",
        type=str,
        default="models/multi_gpu_policy.pt",
        help="Path to save model",
    )
    parser.add_argument("--scenario", type=str, default="default", help="Scenario name")
    parser.add_argument("--run-name", type=str, default=None, help="TensorBoard run name")

    args = parser.parse_args()

    # Setup distributed training
    dist_info = setup_distributed()
    device = dist_info["device"]
    is_master = dist_info["is_master"]
    world_size = dist_info["world_size"]
    rank = dist_info["rank"]

    # Set seeds (different per rank for data diversity)
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    if is_master:
        print(f"🔧 Using device: {device}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(device)}")
            print(f"   Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")

    # Calculate per-GPU configuration
    num_envs = args.num_envs_per_gpu
    batch_size = args.batch_size_per_gpu

    # Global configuration (across all GPUs)
    total_envs = num_envs * world_size
    total_batch_size = batch_size * world_size

    if is_master:
        print(f"🎮 Creating {num_envs} environments per GPU ({total_envs} total)")
        print(f"📊 Batch size: {batch_size} per GPU ({total_batch_size} total)")

    # Create vectorized environment (each rank has its own)
    env_kwargs = {
        "scenario": args.scenario,
        "num_opponents": args.num_opponents,
    }

    vecenv = pufferlib.vector.make(
        make_env_func,
        env_kwargs=env_kwargs,
        backend=pufferlib.vector.Serial,  # Use Serial for stability
        num_envs=num_envs,
    )

    # Create policy
    obs_dim = vecenv.single_observation_space.shape[0]
    action_dims = vecenv.single_action_space.nvec.tolist()

    if is_master:
        print(f"🧠 Creating policy network")
        print(f"   Observation dim: {obs_dim}")
        print(f"   Action dims: {action_dims}")

    policy = PolicyNetwork(obs_dim, action_dims, hidden_size=args.hidden_size).to(device)

    # Wrap with DistributedDataParallel if using multiple GPUs
    if dist_info["distributed"]:
        policy = DDP(policy, device_ids=[dist_info["local_rank"]])
        if is_master:
            print(f"✅ Policy wrapped with DistributedDataParallel")

    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)

    # TensorBoard (master only)
    writer = None
    if is_master:
        if args.run_name is None:
            args.run_name = f"multi_gpu_{world_size}x_{args.scenario}_{int(time.time())}"
        writer = SummaryWriter(f"runs/{args.run_name}")
        print(f"📊 TensorBoard logging to: runs/{args.run_name}")

    try:
        # Get the actual policy for training (unwrap DDP if needed)
        training_policy = policy.module if dist_info["distributed"] else policy

        # Train
        train_ppo_vectorized(
            vecenv=vecenv,
            policy=training_policy,
            optimizer=optimizer,
            device=device,
            num_steps=args.num_steps // world_size,  # Divide steps across GPUs
            batch_size=batch_size,
            writer=writer if is_master else None,
        )

        # Save model (master only)
        if is_master:
            Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)

            # Save the unwrapped model
            model_to_save = policy.module if dist_info["distributed"] else policy

            torch.save(
                {
                    "policy_state_dict": model_to_save.state_dict(),
                    "obs_dim": obs_dim,
                    "action_dims": action_dims,
                    "hidden_size": args.hidden_size,
                },
                args.save_path,
            )
            print(f"💾 Model saved to {args.save_path}")

    finally:
        if writer:
            writer.close()
        vecenv.close()
        cleanup_distributed()


if __name__ == "__main__":
    main()
