"""IPPO baseline + MAPPO arm for the minimal dilemma (issue #292).

Thin driver that invokes :class:`bucket_brigade.training.joint_trainer.JointPPOTrainer`
on :class:`MinimalDilemmaEnv` with **zero core modifications**. Two arms:

- ``--centralized-critic=False`` (default) — independent-PPO. The primary
  basin-trap claim (H1): IPPO from random init converges to mutual-defect
  (per-step reward ≈ 0.0).
- ``--centralized-critic=True`` — MAPPO (issue #208). The intervention-
  sensitivity probe (H2): does centralized-critic shared baseline let PPO
  reach the cooperative basin from random init?

Outputs (under ``--output-dir``):

- ``metrics.json``  — per-iteration scalars: ``iteration``,
  ``mean_step_reward_per_agent`` (team / num_agents → directly comparable
  to the analytic table in the issue), ``cooperation_fraction``, and the
  full ``JointPPOTrainer.update`` stats dict.
- ``config.json``   — exact arguments used.
- ``policies/agent_{i}.pt`` — final per-agent state dicts (consumed by BC
  / verdict scripts downstream).

Usage::

    # 5-iter smoke (~30 sec on laptop)
    uv run python -m experiments.p3_specialization.minimal_dilemma.train_ippo \\
        --seed 0 --num-iterations 5 --rollout-steps 256 \\
        --output-dir /tmp/dilemma_smoke

    # Full IPPO baseline (100 iters; runs on COMPUTE_HOST_PRIMARY).
    uv run python -m experiments.p3_specialization.minimal_dilemma.train_ippo \\
        --seed 0 --num-iterations 100 --rollout-steps 2048 \\
        --output-dir experiments/p3_specialization/minimal_dilemma/results/ippo_seed0

    # MAPPO arm (H2).
    uv run python -m experiments.p3_specialization.minimal_dilemma.train_ippo \\
        --seed 0 --num-iterations 100 --rollout-steps 2048 \\
        --centralized-critic \\
        --output-dir experiments/p3_specialization/minimal_dilemma/results/mappo_seed0
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import torch

from bucket_brigade.training.joint_trainer import JointPPOTrainer, flatten_dict_obs

from experiments.p3_specialization.minimal_dilemma.env import (
    ACTION_COOPERATE,
    EPISODE_LENGTH,
    MULTIPLIER,
    MinimalDilemmaEnv,
    NUM_AGENTS,
)


# Single Discrete(2) action head per agent — the dilemma is a one-dim choice.
ACTION_DIMS: List[int] = [2]


@dataclass
class TrainConfig:
    """Per-cell training config (mirrors ``experiments/p3_specialization/train.py``)."""

    seed: int
    num_iterations: int = 100
    rollout_steps: int = 2048
    hidden_size: int = 64
    lr: float = 3e-4
    ppo_epochs: int = 4
    minibatch_size: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    centralized_critic: bool = False
    multiplier: float = MULTIPLIER
    episode_length: int = EPISODE_LENGTH
    device: str = "cpu"
    bc_init_checkpoint_dir: Optional[str] = None


def make_env_fn(multiplier: float, episode_length: int):
    """Closure factory matching the ``env_fn`` shape expected by JointPPOTrainer."""

    def env_fn() -> MinimalDilemmaEnv:
        return MinimalDilemmaEnv(multiplier=multiplier, episode_length=episode_length)

    return env_fn


def train_one_cell(cfg: TrainConfig, output_dir: Path) -> List[dict]:
    """Run one training cell to completion. Returns the metrics log.

    Mirrors :func:`experiments.p3_specialization.train.train_one_cell` but
    stripped of the BB-specific MI / curriculum / shaping plumbing — the
    minimal-dilemma env has none of those signals.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    env_fn = make_env_fn(cfg.multiplier, cfg.episode_length)

    # Probe obs_dim from a single reset. Identity tail (#204) is appended by
    # ``flatten_dict_obs`` when ``agent_id`` is passed. For the minimal env:
    # base = last_actions(4) + scenario_info(1) = 5; identity tail = 2;
    # total = 7. We let the probe compute it rather than hardcoding so a
    # future env-shape tweak doesn't require touching this driver.
    probe = env_fn()
    probe_obs = probe.reset(seed=cfg.seed)
    obs_dim = flatten_dict_obs(probe_obs, agent_id=0, num_agents=NUM_AGENTS).shape[0]

    trainer = JointPPOTrainer(
        env_fn=env_fn,
        num_agents=NUM_AGENTS,
        obs_dim=obs_dim,
        action_dims=ACTION_DIMS,
        hidden_size=cfg.hidden_size,
        lr=cfg.lr,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        ppo_epochs=cfg.ppo_epochs,
        minibatch_size=cfg.minibatch_size,
        value_coef=cfg.value_coef,
        entropy_coef=cfg.entropy_coef,
        redundancy_coef=0.0,
        centralized_critic=cfg.centralized_critic,
        device=cfg.device,
        seed=cfg.seed,
    )

    # Optional BC warm-start. Same protocol as
    # ``experiments/p3_specialization/train.py:667`` (issue #270).
    if cfg.bc_init_checkpoint_dir is not None:
        bc_dir = Path(cfg.bc_init_checkpoint_dir)
        if not bc_dir.is_dir():
            raise FileNotFoundError(
                f"bc_init_checkpoint_dir does not exist or is not a directory: {bc_dir}"
            )
        for i, policy in enumerate(trainer.policies):
            ckpt_path = bc_dir / f"agent_{i}.pt"
            if not ckpt_path.exists():
                raise FileNotFoundError(
                    f"BC checkpoint missing for agent {i}: expected {ckpt_path}"
                )
            sd = torch.load(ckpt_path, map_location=cfg.device, weights_only=True)
            policy.load_state_dict(sd)
        print(f"  BC-init: loaded {NUM_AGENTS} per-agent state dicts from {bc_dir}")

    metrics_log: List[dict] = []
    for it in range(cfg.num_iterations):
        rollout = trainer.collect_rollout(cfg.rollout_steps)
        stats = trainer.update(rollout)

        # Per-step team reward (averaged over both agents). For the dilemma,
        # the natural unit is per-agent (matches the analytic payoff table);
        # we report both for ease of comparison.
        team_sum = sum(
            float(rollout.rewards[i].sum().item()) for i in range(NUM_AGENTS)
        )
        per_step_per_agent = team_sum / (cfg.rollout_steps * NUM_AGENTS)

        # Cooperation fraction across the rollout (both agents pooled).
        coop_count = 0
        total_actions = 0
        for i in range(NUM_AGENTS):
            acts = rollout.actions[i][:, 0].cpu().numpy()
            coop_count += int((acts == ACTION_COOPERATE).sum())
            total_actions += len(acts)
        coop_fraction = float(coop_count / max(1, total_actions))

        record = {
            "iteration": it,
            "mean_step_reward_per_agent": per_step_per_agent,
            "mean_step_reward_team": team_sum / cfg.rollout_steps,
            "cooperation_fraction": coop_fraction,
            **stats,
        }
        metrics_log.append(record)

        if it % max(1, cfg.num_iterations // 10) == 0 or it == cfg.num_iterations - 1:
            print(
                f"  iter {it:4d} | per_agent {per_step_per_agent:+.3f} | "
                f"coop_frac {coop_fraction:.3f} | "
                f"policy_loss {stats.get('policy_loss', float('nan')):.4f}"
            )

    # Save final policies.
    pol_dir = output_dir / "policies"
    pol_dir.mkdir(exist_ok=True)
    for i, policy in enumerate(trainer.policies):
        torch.save(policy.state_dict(), pol_dir / f"agent_{i}.pt")

    with (output_dir / "metrics.json").open("w") as f:
        json.dump(metrics_log, f, indent=2)
    with (output_dir / "config.json").open("w") as f:
        json.dump(asdict(cfg), f, indent=2)

    return metrics_log


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--num-iterations", type=int, default=100)
    p.add_argument("--rollout-steps", type=int, default=2048)
    p.add_argument("--hidden-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ppo-epochs", type=int, default=4)
    p.add_argument("--minibatch-size", type=int, default=256)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument(
        "--centralized-critic",
        action="store_true",
        help="Enable MAPPO (centralized critic) per #208 — H2 intervention arm.",
    )
    p.add_argument(
        "--bc-init-checkpoint-dir",
        type=str,
        default=None,
        help=(
            "Directory with per-agent BC-pretrained state dicts (agent_{i}.pt). "
            "Loaded into trainer.policies before the first PPO iteration."
        ),
    )
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    cfg = TrainConfig(
        seed=args.seed,
        num_iterations=args.num_iterations,
        rollout_steps=args.rollout_steps,
        hidden_size=args.hidden_size,
        lr=args.lr,
        ppo_epochs=args.ppo_epochs,
        minibatch_size=args.minibatch_size,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        centralized_critic=args.centralized_critic,
        bc_init_checkpoint_dir=args.bc_init_checkpoint_dir,
        device=args.device,
    )
    print(
        f"== minimal_dilemma train: seed={cfg.seed} "
        f"centralized_critic={cfg.centralized_critic} "
        f"bc_init={cfg.bc_init_checkpoint_dir} =="
    )
    train_one_cell(cfg, args.output_dir)


if __name__ == "__main__":
    main()
