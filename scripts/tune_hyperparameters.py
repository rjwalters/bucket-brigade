#!/usr/bin/env python3
"""
Hyperparameter tuning script for Bucket Brigade PPO training using Optuna.

This script systematically searches for optimal hyperparameters using Bayesian
optimization. It can be run in parallel across multiple processes for faster results.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import json
import time
from collections import deque

from bucket_brigade.envs.puffer_env import PufferBucketBrigade
from bucket_brigade.envs import get_scenario_by_name
from scripts.train_simple import PolicyNetwork, compute_gae


def train_trial(
    env,
    policy,
    optimizer,
    num_steps,
    batch_size,
    num_epochs,
    clip_epsilon,
    value_coef,
    entropy_coef,
    gamma,
    gae_lambda,
    max_grad_norm,
    device,
    trial=None,
    eval_interval=1000,
):
    """
    Train the policy and return mean reward for hyperparameter optimization.

    This is a simplified version of train_ppo that focuses on evaluation metrics
    for hyperparameter tuning.
    """
    obs, _ = env.reset()
    obs = torch.FloatTensor(obs)

    episode_rewards = deque(maxlen=100)
    current_episode_reward = 0

    global_step = 0
    last_eval_step = 0

    while global_step < num_steps:
        # Collect trajectories
        observations = []
        actions_taken = []
        log_probs_old = []
        rewards = []
        dones = []
        values = []

        for _ in range(batch_size):
            with torch.no_grad():
                obs_gpu = obs.unsqueeze(0).to(device)
                actions, log_prob, value = policy.get_action(obs_gpu)
                actions_list = [a.item() for a in actions]

            observations.append(obs.cpu())
            actions_taken.append(actions_list)
            log_probs_old.append(log_prob.cpu() if log_prob is not None else None)
            values.append(value.squeeze().cpu())

            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(actions_list)
            done = terminated or truncated

            rewards.append(reward)
            dones.append(done)
            current_episode_reward += reward

            obs = torch.FloatTensor(next_obs)
            global_step += 1

            if done:
                episode_rewards.append(current_episode_reward)
                current_episode_reward = 0
                obs, _ = env.reset()
                obs = torch.FloatTensor(obs)

        # Compute advantages
        values_np = torch.stack(values).detach().numpy()
        advantages = compute_gae(rewards, values_np, dones, gamma=gamma, gae_lambda=gae_lambda)
        advantages = torch.FloatTensor(advantages)
        returns = advantages + torch.FloatTensor(values_np)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert to tensors and move to device
        obs_batch = torch.stack(observations).to(device)
        actions_batch = torch.LongTensor(actions_taken).to(device)
        log_probs_old = torch.stack(log_probs_old).to(device)
        advantages = advantages.to(device)
        returns = returns.to(device)

        # PPO update
        for epoch in range(num_epochs):
            # Forward pass
            action_logits, values_pred = policy(obs_batch)

            # Compute log probs for taken actions
            log_probs_new = []
            entropies = []

            for i, logits in enumerate(action_logits):
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                log_probs_new.append(dist.log_prob(actions_batch[:, i]))
                entropies.append(dist.entropy())

            log_probs_new = torch.stack(log_probs_new).sum(0)
            entropy = torch.stack(entropies).mean()

            # PPO loss
            ratio = torch.exp(log_probs_new - log_probs_old)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = nn.functional.mse_loss(values_pred.squeeze(), returns)

            # Total loss
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()

        # Report intermediate values for pruning
        if trial is not None and global_step - last_eval_step >= eval_interval:
            if len(episode_rewards) > 0:
                mean_reward = np.mean(episode_rewards)
                trial.report(mean_reward, global_step)

                # Check if trial should be pruned
                if trial.should_prune():
                    raise optuna.TrialPruned()

            last_eval_step = global_step

    # Return final mean reward
    if len(episode_rewards) > 0:
        return np.mean(episode_rewards)
    else:
        return 0.0


def objective(trial, args, device):
    """
    Optuna objective function to optimize.

    This function defines the hyperparameter search space and trains a model
    with the sampled hyperparameters, returning the mean reward.
    """
    # Sample hyperparameters
    hidden_size = trial.suggest_categorical("hidden_size", [64, 128, 256, 512])
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [512, 1024, 2048, 4096])
    num_epochs = trial.suggest_int("num_epochs", 3, 10)
    clip_epsilon = trial.suggest_float("clip_epsilon", 0.1, 0.3)
    entropy_coef = trial.suggest_float("entropy_coef", 0.001, 0.05, log=True)
    value_coef = trial.suggest_float("value_coef", 0.3, 0.7)
    gamma = trial.suggest_float("gamma", 0.95, 0.999)
    gae_lambda = trial.suggest_float("gae_lambda", 0.9, 0.99)

    # Set random seed for reproducibility
    seed = args.seed + trial.number
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create environment
    scenario = get_scenario_by_name(args.scenario, num_agents=args.num_opponents + 1)
    env = PufferBucketBrigade(scenario=scenario, num_opponents=args.num_opponents)

    # Create policy
    obs_dim = env.observation_space.shape[0]
    action_dims = env.action_space.nvec.tolist()

    policy = PolicyNetwork(obs_dim, action_dims, hidden_size=hidden_size).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    # Train and evaluate
    try:
        mean_reward = train_trial(
            env=env,
            policy=policy,
            optimizer=optimizer,
            num_steps=args.num_steps,
            batch_size=batch_size,
            num_epochs=num_epochs,
            clip_epsilon=clip_epsilon,
            value_coef=value_coef,
            entropy_coef=entropy_coef,
            gamma=gamma,
            gae_lambda=gae_lambda,
            max_grad_norm=0.5,
            device=device,
            trial=trial,
            eval_interval=args.eval_interval,
        )
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        raise optuna.TrialPruned()

    return mean_reward


def main():
    parser = argparse.ArgumentParser(
        description="Tune PPO hyperparameters for Bucket Brigade using Optuna"
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="default",
        help="Scenario name",
    )
    parser.add_argument(
        "--num-opponents",
        type=int,
        default=2,
        help="Number of opponent agents",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=50000,
        help="Training steps per trial (shorter for faster tuning)",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Number of trials to run",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel jobs (-1 for all CPUs)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default=None,
        help="Optuna study name (default: auto-generated)",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optuna storage URL (default: in-memory)",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=5000,
        help="Steps between intermediate evaluations",
    )
    parser.add_argument(
        "--pruner",
        type=str,
        default="median",
        choices=["none", "median"],
        help="Pruning strategy",
    )

    args = parser.parse_args()

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Using device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    # Generate study name if not provided
    if args.study_name is None:
        args.study_name = f"ppo_tune_{args.scenario}_{int(time.time())}"

    print(f"\n{'=' * 70}")
    print(f"🔍 Hyperparameter Tuning Configuration")
    print(f"{'=' * 70}")
    print(f"Scenario: {args.scenario}")
    print(f"Study name: {args.study_name}")
    print(f"Number of trials: {args.n_trials}")
    print(f"Parallel jobs: {args.n_jobs}")
    print(f"Training steps per trial: {args.num_steps:,}")
    print(f"Storage: {args.storage or 'in-memory'}")
    print(f"{'=' * 70}\n")

    # Create pruner
    if args.pruner == "median":
        pruner = MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=args.eval_interval * 2,
        )
    else:
        pruner = optuna.pruners.NopPruner()

    # Create or load study
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="maximize",
        sampler=TPESampler(seed=args.seed),
        pruner=pruner,
        load_if_exists=True,
    )

    # Run optimization
    print(f"🚀 Starting hyperparameter optimization...\n")

    try:
        study.optimize(
            lambda trial: objective(trial, args, device),
            n_trials=args.n_trials,
            n_jobs=args.n_jobs,
            show_progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n⚠️  Optimization interrupted by user")

    # Print results
    print(f"\n{'=' * 70}")
    print(f"✅ Optimization Complete!")
    print(f"{'=' * 70}")
    print(f"Number of finished trials: {len(study.trials)}")
    print(f"Best trial:")

    trial = study.best_trial
    print(f"  Value (mean reward): {trial.value:.2f}")
    print(f"  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Save results
    results_dir = Path("experiments/hyperparameter_tuning")
    results_dir.mkdir(parents=True, exist_ok=True)

    results_file = results_dir / f"{args.study_name}_results.json"
    with open(results_file, "w") as f:
        json.dump({
            "study_name": args.study_name,
            "scenario": args.scenario,
            "best_value": trial.value,
            "best_params": trial.params,
            "n_trials": len(study.trials),
            "all_trials": [
                {
                    "number": t.number,
                    "value": t.value,
                    "params": t.params,
                    "state": str(t.state),
                }
                for t in study.trials
            ],
        }, f, indent=2)

    print(f"\n💾 Results saved to: {results_file}")

    # Generate visualization if possible
    try:
        import optuna.visualization as vis

        # Save optimization history
        fig = vis.plot_optimization_history(study)
        fig.write_html(results_dir / f"{args.study_name}_history.html")

        # Save parameter importances
        fig = vis.plot_param_importances(study)
        fig.write_html(results_dir / f"{args.study_name}_importances.html")

        # Save parallel coordinate plot
        fig = vis.plot_parallel_coordinate(study)
        fig.write_html(results_dir / f"{args.study_name}_parallel.html")

        print(f"📊 Visualizations saved to: {results_dir}")

    except ImportError:
        print("⚠️  Install plotly for visualizations: pip install plotly")

    print(f"{'=' * 70}\n")

    # Print recommended command
    print("🎯 To train with best hyperparameters, run:")
    print(f"\nuv run python scripts/train_simple.py \\")
    print(f"  --scenario {args.scenario} \\")
    print(f"  --hidden-size {trial.params['hidden_size']} \\")
    print(f"  --lr {trial.params['learning_rate']:.2e} \\")
    print(f"  --batch-size {trial.params['batch_size']} \\")
    print(f"  --num-steps 500000\n")


if __name__ == "__main__":
    main()
