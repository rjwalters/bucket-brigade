"""Single-controller (joint-action) PPO trainer for issue #291.

This is the scope-determining experiment for the misaligned-gradient thesis
(``research_notebook/2026-05-17_thesis_misaligned_gradients.md`` open question
#1). The question: does the basin trap that IPPO hits on
``minimal_specialization`` (#270/#271 verdicts) survive when multi-agent
credit-assignment is removed as a confound?

The training setup matches IPPO's budget exactly (50 iter * 2048 steps * 3
seeds) except that:

- The env is wrapped in :class:`SingleAgentJointWrapper` so a single
  controller emits the joint action for all 4 sub-agents.
- A single :class:`PolicyNetwork` produces a factorized joint distribution
  with ``action_dims = [num_houses, 2, 2] * num_subagents`` (12 heads for
  ``minimal_specialization``: 4 sub-agents x [house, mode, signal]).
- The reward signal is the **team sum** of per-sub-agent rewards (no
  per-agent advantage decomposition).

NB: this is **not** the macro-action wrapper from #286 (which coarsens a
single agent into temporally-extended options) and not the toy reduction
from #292. The env mechanics — episode length, fire dynamics, ignition,
ownership rewards — are bit-exact identical to the IPPO baseline.

CLI:

    python experiments/p3_specialization/train_single_agent.py \\
        --scenario minimal_specialization \\
        --seed 42 --num-iterations 50 --rollout-steps 2048 \\
        --output-dir runs/issue291/seed_42 --joint-control

Defaults match the #270 budget. The ``--joint-control`` flag is accepted
but ignored (this script is the joint-control path); it is preserved so
the run driver and docs can pass it explicitly.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from bucket_brigade.envs.bucket_brigade_env import BucketBrigadeEnv
from bucket_brigade.envs.scenarios_generated import get_scenario_by_name
from bucket_brigade.envs.single_agent_wrapper import SingleAgentJointWrapper
from bucket_brigade.training.networks import PolicyNetwork, compute_gae


@dataclass
class SingleAgentCellConfig:
    scenario: str
    seed: int
    num_iterations: int = 50
    rollout_steps: int = 2048
    num_subagents: int = 4
    hidden_size: int = 64
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    minibatch_size: int = 256
    device: str = "cpu"
    # Action layout per sub-agent. Matches the post-#235 ``[house, mode,
    # signal]`` MultiDiscrete([num_houses, 2, 2]). The joint factorized head
    # is built as ``per_agent * num_subagents``.
    per_agent_action_dims: List[int] = field(default_factory=lambda: [10, 2, 2])


def _build_env(scenario_name: str, num_subagents: int) -> SingleAgentJointWrapper:
    """Construct the joint-action wrapper over ``BucketBrigadeEnv``."""
    scenario = get_scenario_by_name(scenario_name, num_agents=num_subagents)
    inner = BucketBrigadeEnv(scenario=scenario)
    return SingleAgentJointWrapper(inner)


def _joint_obs_dim(env: SingleAgentJointWrapper) -> int:
    """Length of the controller's flattened observation (concat per-sub-agent rows)."""
    return int(env.num_agents * env.obs_dim_per_agent)


def _flatten_controller_obs(stacked: np.ndarray) -> np.ndarray:
    """Concatenate ``[N, obs_dim_per_agent]`` rows into a single controller vector."""
    return stacked.reshape(-1).astype(np.float32, copy=False)


def train_single_agent_cell(
    cfg: SingleAgentCellConfig, output_dir: Path
) -> None:
    """Run one (scenario, seed) cell of single-controller joint-action PPO."""
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(cfg.device)

    env = _build_env(cfg.scenario, cfg.num_subagents)
    # The wrapper exposes ``joint_action_dims`` already laid out as
    # [house_0, mode_0, signal_0, house_1, mode_1, signal_1, ...]. Use it
    # directly so the policy heads match the wrapper's split convention.
    joint_action_dims = list(env.joint_action_dims)
    obs_dim = _joint_obs_dim(env)

    policy = PolicyNetwork(
        obs_dim=obs_dim,
        action_dims=joint_action_dims,
        hidden_size=cfg.hidden_size,
    ).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=cfg.lr)

    stacked = env.reset(seed=cfg.seed)
    obs = _flatten_controller_obs(stacked)

    metrics_log = []

    for it in range(cfg.num_iterations):
        # ---------- rollout ----------
        T = cfg.rollout_steps
        rollout_obs = np.zeros((T, obs_dim), dtype=np.float32)
        rollout_actions = np.zeros((T, len(joint_action_dims)), dtype=np.int64)
        rollout_log_probs = np.zeros(T, dtype=np.float32)
        rollout_values = np.zeros(T, dtype=np.float32)
        rollout_rewards = np.zeros(T, dtype=np.float32)
        rollout_dones = np.zeros(T, dtype=np.float32)

        for t in range(T):
            obs_t = torch.from_numpy(obs).to(device).unsqueeze(0)  # [1, obs_dim]
            with torch.no_grad():
                actions, log_prob, _entropy, value = policy.get_action_and_value(obs_t)
            joint_action = actions[0].cpu().numpy()  # [12] flat joint action

            next_stacked, team_reward, done, _info = env.step(joint_action)

            rollout_obs[t] = obs
            rollout_actions[t] = joint_action
            rollout_log_probs[t] = float(log_prob[0].item())
            rollout_values[t] = float(value[0].item())
            rollout_rewards[t] = float(team_reward)
            rollout_dones[t] = float(done)

            if done:
                next_stacked = env.reset()
            obs = _flatten_controller_obs(next_stacked)

        # ---------- GAE + returns ----------
        adv = np.asarray(
            compute_gae(
                rollout_rewards.tolist(),
                rollout_values.tolist(),
                rollout_dones.astype(bool).tolist(),
                cfg.gamma,
                cfg.gae_lambda,
            ),
            dtype=np.float32,
        )
        returns = adv + rollout_values
        # Per-batch advantage standardization, matching JointPPOTrainer.update.
        adv_std = (adv - adv.mean()) / (adv.std() + 1e-8)

        # ---------- PPO update ----------
        obs_tensor = torch.from_numpy(rollout_obs).to(device)
        act_tensor = torch.from_numpy(rollout_actions).to(device)
        old_lp_tensor = torch.from_numpy(rollout_log_probs).to(device)
        adv_tensor = torch.from_numpy(adv_std).to(device)
        ret_tensor = torch.from_numpy(returns).to(device)

        mb_size = min(cfg.minibatch_size, T)

        epoch_policy_loss = 0.0
        epoch_value_loss = 0.0
        epoch_entropy = 0.0
        for _ in range(cfg.ppo_epochs):
            idx = torch.randperm(T, device=device)[:mb_size]
            obs_mb = obs_tensor[idx]
            act_mb = act_tensor[idx]
            old_lp_mb = old_lp_tensor[idx]
            adv_mb = adv_tensor[idx]
            ret_mb = ret_tensor[idx]

            _new_acts, new_lp, entropy, value = policy.get_action_and_value(
                obs_mb, action=act_mb
            )

            ratio = torch.exp(new_lp - old_lp_mb)
            surr1 = ratio * adv_mb
            surr2 = (
                torch.clamp(ratio, 1.0 - cfg.clip_epsilon, 1.0 + cfg.clip_epsilon)
                * adv_mb
            )
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = 0.5 * (value - ret_mb).pow(2).mean()
            entropy_mean = entropy.mean()

            loss = (
                policy_loss
                + cfg.value_coef * value_loss
                - cfg.entropy_coef * entropy_mean
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            optimizer.step()

            epoch_policy_loss += float(policy_loss.item())
            epoch_value_loss += float(value_loss.item())
            epoch_entropy += float(entropy_mean.item())

        n_e = max(1, cfg.ppo_epochs)
        # Match the IPPO trainer's "mean per-step team reward" metric so
        # downstream analyzers (gap_closed) can compare arms directly.
        mean_step_reward_team = float(rollout_rewards.mean())
        record = {
            "iteration": it,
            "mean_step_reward_team": mean_step_reward_team,
            "policy_loss": epoch_policy_loss / n_e,
            "value_loss": epoch_value_loss / n_e,
            "entropy": epoch_entropy / n_e,
        }
        metrics_log.append(record)

        if it % max(1, cfg.num_iterations // 10) == 0 or it == cfg.num_iterations - 1:
            print(
                f"  iter {it:4d} | team_reward {mean_step_reward_team:8.3f} | "
                f"policy_loss {record['policy_loss']:.4f} | "
                f"value_loss {record['value_loss']:.4f} | "
                f"entropy {record['entropy']:.4f}"
            )

    # Save final policy + metrics + config.
    pol_dir = output_dir / "policies"
    pol_dir.mkdir(exist_ok=True)
    torch.save(policy.state_dict(), pol_dir / "controller.pt")
    with (output_dir / "metrics.json").open("w") as f:
        json.dump(metrics_log, f, indent=2)
    with (output_dir / "config.json").open("w") as f:
        json.dump(asdict(cfg), f, indent=2)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--num-iterations", type=int, default=50)
    p.add_argument("--rollout-steps", type=int, default=2048)
    p.add_argument("--num-subagents", type=int, default=4)
    p.add_argument(
        "--joint-control",
        action="store_true",
        help=(
            "Issue #291: opt into the joint-action single-controller training "
            "path. This script implements only that path; the flag is "
            "accepted for explicitness and forward compatibility with the "
            "sweep driver. Future revisions may multiplex this script with "
            "other reductions (#286 macro actions, #292 toy dilemma)."
        ),
    )
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    cfg = SingleAgentCellConfig(
        scenario=args.scenario,
        seed=args.seed,
        num_iterations=args.num_iterations,
        rollout_steps=args.rollout_steps,
        num_subagents=args.num_subagents,
        device=args.device,
    )
    print(
        f"== Issue #291 single-controller cell: scenario={cfg.scenario} "
        f"seed={cfg.seed} num_subagents={cfg.num_subagents} "
        f"joint_control={args.joint_control} =="
    )
    train_single_agent_cell(cfg, args.output_dir)


if __name__ == "__main__":
    main()
