"""Behaviorally clone the specialist policy and save per-agent checkpoints.

Issue #270 — Phase 1 discriminator between basin trap and anti-attractor.

This script does three things end-to-end:

1. **Demo gathering** — roll out the hand-coded specialist
   (:func:`bucket_brigade.baselines.specialist_action_joint`) on a scenario
   (default: ``minimal_specialization``) until we have ``--num-pairs`` per-agent
   ``(flat_obs, action)`` tuples. Observations are flattened with
   :func:`bucket_brigade.training.joint_trainer.flatten_dict_obs` using the
   per-agent identity one-hot tail (#204), so the trained policy is shape-
   compatible with :class:`JointPPOTrainer.policies[i]` at load time.

2. **Supervised fit** — instantiate one :class:`PolicyNetwork` per agent and
   train it with summed cross-entropy across the three action heads
   (``[house, mode, signal]``). Adam, batch 64, 10 epochs by default.

3. **Eval gate** — run ``--eval-episodes`` deterministic-argmax episodes and
   compute ``gap_closed = (mean_step_team − MINSPEC_RANDOM) / (MINSPEC_SPECIALIST
   − MINSPEC_RANDOM)`` using the canonical constants from
   :mod:`bucket_brigade.baselines`. The script exits non-zero when
   ``gap_closed < --min-gap-closed`` so a bad BC fit blocks the PPO
   continuation.

Per-agent state dicts are saved to ``<output_dir>/agent_{i}.pt``, the exact
filename layout :func:`train_one_cell` expects when loading from
``--bc-init-checkpoint-dir``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from bucket_brigade.baselines import (
    MINSPEC_RANDOM,
    MINSPEC_SPECIALIST,
    specialist_action_joint,
)
from bucket_brigade.envs.bucket_brigade_env import BucketBrigadeEnv
from bucket_brigade.envs.scenarios_generated import get_scenario_by_name
from bucket_brigade.training.joint_trainer import flatten_dict_obs
from bucket_brigade.training.networks import PolicyNetwork, TransformerPolicyNetwork

# Verdict reference constants imported from the canonical source
# (``bucket_brigade.baselines``; issue #293 unified the three previously
# scattered ``MINSPEC_RANDOM`` literals). See that module's docstring for
# derivation provenance.


def _gap_closed(mean_step_team: float) -> float:
    """Fraction of specialist−random gap closed (minimal_specialization)."""
    return (mean_step_team - MINSPEC_RANDOM) / (MINSPEC_SPECIALIST - MINSPEC_RANDOM)


def gather_demos(
    scenario_name: str,
    num_agents: int,
    num_pairs_per_agent: int,
    seed: int,
    num_houses: int = 10,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Roll the specialist until we have ``num_pairs_per_agent`` per-agent
    ``(flat_obs, action)`` tuples per agent.

    Returns:
        ``(obs_per_agent, act_per_agent)`` where ``obs_per_agent[i]`` is a
        ``[num_pairs_per_agent, obs_dim]`` float32 array of agent i's
        observations (with the identity one-hot tail set for agent i) and
        ``act_per_agent[i]`` is a ``[num_pairs_per_agent, 3]`` int64 array of
        the specialist's actions for agent i at those observations.
    """
    scenario = get_scenario_by_name(scenario_name, num_agents=num_agents)
    env = BucketBrigadeEnv(scenario=scenario)
    rng = np.random.default_rng(seed)

    obs_buf: List[List[np.ndarray]] = [[] for _ in range(num_agents)]
    act_buf: List[List[np.ndarray]] = [[] for _ in range(num_agents)]

    # The specialist is deterministic from (obs, agent_id), so each step
    # contributes exactly one (obs, action) pair per agent. Tally on
    # `obs_buf[0]` and stop when we've reached the per-agent target.
    while len(obs_buf[0]) < num_pairs_per_agent:
        obs = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        while not env.done:
            joint = specialist_action_joint(obs, num_agents, num_houses=num_houses)
            for i in range(num_agents):
                obs_buf[i].append(
                    flatten_dict_obs(obs, agent_id=i, num_agents=num_agents)
                )
                act_buf[i].append(joint[i])
            obs, _, _, _ = env.step(joint)
            if len(obs_buf[0]) >= num_pairs_per_agent:
                break

    obs_per_agent = [np.stack(buf, axis=0).astype(np.float32) for buf in obs_buf]
    act_per_agent = [np.stack(buf, axis=0).astype(np.int64) for buf in act_buf]
    return obs_per_agent, act_per_agent


def bc_fit_one_agent(
    obs: np.ndarray,
    acts: np.ndarray,
    obs_dim: int,
    action_dims: List[int],
    hidden_size: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    seed: int,
    architecture: str = "mlp",
    transformer_d_model: int = 256,
    transformer_nhead: int = 4,
    transformer_num_layers: int = 3,
    transformer_dim_feedforward: int = 512,
    transformer_dropout: float = 0.1,
) -> Tuple[torch.nn.Module, List[float]]:
    """Supervised fit of one policy network to (obs, acts) by summing
    cross-entropy on each of ``len(action_dims)`` heads.

    ``architecture`` selects between :class:`PolicyNetwork` (default; ``'mlp'``)
    and :class:`TransformerPolicyNetwork` (#281 escalation; ``'transformer'``).
    Default values preserve bit-identical behaviour with the pre-#281 MLP path.

    Returns the trained policy plus per-epoch mean training losses (for
    inspection only — convergence is verified by the eval gate, not the loss
    value).
    """
    # Reproducible per-agent fits: seed both the model init and the dataloader
    # shuffle by tying torch's RNG to the same seed used for the broader run.
    torch.manual_seed(seed)
    if architecture == "mlp":
        policy: torch.nn.Module = PolicyNetwork(
            obs_dim=obs_dim, action_dims=action_dims, hidden_size=hidden_size
        ).to(device)
    elif architecture == "transformer":
        policy = TransformerPolicyNetwork(
            obs_dim=obs_dim,
            action_dims=action_dims,
            d_model=transformer_d_model,
            nhead=transformer_nhead,
            num_layers=transformer_num_layers,
            dim_feedforward=transformer_dim_feedforward,
            dropout=transformer_dropout,
        ).to(device)
    else:
        raise ValueError(
            f"Unknown architecture {architecture!r}; expected 'mlp' or 'transformer'."
        )
    opt = torch.optim.Adam(policy.parameters(), lr=lr)

    x_t = torch.from_numpy(obs).float().to(device)
    a_t = torch.from_numpy(acts).long().to(device)
    ds = TensorDataset(x_t, a_t)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    epoch_losses: List[float] = []
    n_heads = len(action_dims)
    for ep in range(epochs):
        batch_losses: List[float] = []
        for x, a in dl:
            logits, _ = policy(x)
            # Sum of CE on each independent action head. PolicyNetwork emits
            # `n_heads` raw logits tensors of shape (batch, action_dims[k]).
            loss = sum(F.cross_entropy(logits[k], a[:, k]) for k in range(n_heads))
            opt.zero_grad()
            loss.backward()
            opt.step()
            batch_losses.append(float(loss.item()))
        epoch_losses.append(float(np.mean(batch_losses)))
    return policy, epoch_losses


def evaluate_policies(
    policies: List[torch.nn.Module],
    scenario_name: str,
    num_agents: int,
    num_episodes: int,
    device: str,
    seed: int = 0,
) -> dict:
    """Roll the per-agent policies (argmax) for ``num_episodes`` episodes and
    return per-step team reward statistics plus gap_closed."""
    scenario = get_scenario_by_name(scenario_name, num_agents=num_agents)
    env = BucketBrigadeEnv(scenario=scenario)

    for p in policies:
        p.eval()

    ep_per_step: List[float] = []
    ep_lengths: List[int] = []
    with torch.no_grad():
        for ep in range(num_episodes):
            obs = env.reset(seed=seed + ep)
            total = 0.0
            nights = 0
            while not env.done:
                joint_rows: List[List[int]] = []
                for i in range(num_agents):
                    x = flatten_dict_obs(obs, agent_id=i, num_agents=num_agents)
                    xt = torch.from_numpy(x).float().to(device).unsqueeze(0)
                    actions, _, _ = policies[i].get_action(xt, deterministic=True)
                    joint_rows.append([int(a.item()) for a in actions])
                joint = np.asarray(joint_rows, dtype=np.int64)
                obs, rewards, _, _ = env.step(joint)
                total += float(rewards.sum())
                nights = int(env.night)
            # ``env.night`` is the night count at termination; treat 0 defensively.
            if nights > 0:
                ep_per_step.append(total / nights)
                ep_lengths.append(nights)

    per_step = np.asarray(ep_per_step, dtype=np.float64)
    gc = _gap_closed(float(per_step.mean())) if per_step.size > 0 else float("nan")
    return {
        "n_episodes": int(per_step.size),
        "mean_step_reward_team": float(per_step.mean()) if per_step.size > 0 else 0.0,
        "std_step_reward_team": float(per_step.std(ddof=1))
        if per_step.size > 1
        else 0.0,
        "mean_episode_length": float(np.mean(ep_lengths)) if ep_lengths else 0.0,
        "gap_closed": float(gc),
        "minspec_random_ref": MINSPEC_RANDOM,
        "minspec_specialist_ref": MINSPEC_SPECIALIST,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--scenario",
        default="minimal_specialization",
        help="Scenario name to gather demos from and evaluate on.",
    )
    p.add_argument("--num-agents", type=int, default=4)
    p.add_argument(
        "--num-pairs-per-agent",
        type=int,
        default=10000,
        help="Per-agent (obs, action) pair count for BC training (#270 brief).",
    )
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-size", type=int, default=64)
    p.add_argument(
        "--architecture",
        type=str,
        choices=["mlp", "transformer"],
        default="mlp",
        help=(
            "Policy architecture: 'mlp' (default; PolicyNetwork) preserves the "
            "pre-#281 path. 'transformer' uses TransformerPolicyNetwork (#281 "
            "escalation). NB: saved state dicts are not cross-architecture "
            "compatible, so a transformer BC-init checkpoint can only be "
            "consumed by a transformer PPO run."
        ),
    )
    p.add_argument("--transformer-d-model", type=int, default=256)
    p.add_argument("--transformer-nhead", type=int, default=4)
    p.add_argument("--transformer-num-layers", type=int, default=3)
    p.add_argument("--transformer-dim-feedforward", type=int, default=512)
    p.add_argument("--transformer-dropout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--eval-episodes",
        type=int,
        default=20,
        help="Number of deterministic-argmax episodes for the gap_closed gate.",
    )
    p.add_argument(
        "--min-gap-closed",
        type=float,
        default=0.7,
        help=(
            "Gate threshold from issue #270: if BC eval gap_closed < this, "
            "exit non-zero so the PPO continuation isn't launched from a bad init."
        ),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save agent_{i}.pt and bc_summary.json.",
    )
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"== BC-init #270: scenario={args.scenario} num_agents={args.num_agents} "
        f"num_pairs_per_agent={args.num_pairs_per_agent} epochs={args.epochs} "
        f"batch_size={args.batch_size} lr={args.lr} seed={args.seed} =="
    )

    # ----- Phase 1: gather demos -----
    print(
        f"[1/3] Gathering specialist demos (target={args.num_pairs_per_agent}/agent)..."
    )
    obs_per_agent, act_per_agent = gather_demos(
        scenario_name=args.scenario,
        num_agents=args.num_agents,
        num_pairs_per_agent=args.num_pairs_per_agent,
        seed=args.seed,
    )
    obs_dim = int(obs_per_agent[0].shape[1])
    print(
        f"  collected per-agent obs shape: {obs_per_agent[0].shape}, obs_dim={obs_dim}"
    )

    # Sanity: at least some non-trivial action distribution. Specialist WORK
    # fraction should be > 0 on minimal_specialization (fires do ignite).
    work_frac = float((act_per_agent[0][:, 1] == 1).mean())
    print(f"  agent 0 specialist WORK fraction: {work_frac:.3f}")

    # ----- Phase 2: BC fit per agent -----
    print(
        f"[2/3] Fitting {args.num_agents} per-agent policies "
        f"(architecture={args.architecture})..."
    )
    policies: List[torch.nn.Module] = []
    per_agent_loss_history: List[List[float]] = []
    action_dims = [10, 2, 2]  # MultiDiscrete([num_houses, mode, signal])
    for i in range(args.num_agents):
        policy, losses = bc_fit_one_agent(
            obs=obs_per_agent[i],
            acts=act_per_agent[i],
            obs_dim=obs_dim,
            action_dims=action_dims,
            hidden_size=args.hidden_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
            # Per-agent seed nudge so the 4 nets don't init identically. The
            # supervised target is per-agent (identity one-hot differs), so the
            # init nudge is purely a numerical-stability nicety.
            seed=args.seed + i,
            architecture=args.architecture,
            transformer_d_model=args.transformer_d_model,
            transformer_nhead=args.transformer_nhead,
            transformer_num_layers=args.transformer_num_layers,
            transformer_dim_feedforward=args.transformer_dim_feedforward,
            transformer_dropout=args.transformer_dropout,
        )
        policies.append(policy)
        per_agent_loss_history.append(losses)
        print(
            f"  agent {i}: epoch_losses=[" + ", ".join(f"{x:.4f}" for x in losses) + "]"
        )

    # ----- Phase 3: save + eval gate -----
    print(f"[3/3] Saving policies and evaluating ({args.eval_episodes} episodes)...")
    for i, policy in enumerate(policies):
        torch.save(policy.state_dict(), args.output_dir / f"agent_{i}.pt")

    eval_stats = evaluate_policies(
        policies=policies,
        scenario_name=args.scenario,
        num_agents=args.num_agents,
        num_episodes=args.eval_episodes,
        device=args.device,
        seed=args.seed + 1000,  # disjoint from demo-gen seed stream
    )

    summary = {
        "issue": 270,
        "scenario": args.scenario,
        "num_agents": args.num_agents,
        "num_pairs_per_agent": args.num_pairs_per_agent,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "architecture": args.architecture,
        "hidden_size": args.hidden_size,
        "transformer_d_model": args.transformer_d_model,
        "transformer_nhead": args.transformer_nhead,
        "transformer_num_layers": args.transformer_num_layers,
        "transformer_dim_feedforward": args.transformer_dim_feedforward,
        "transformer_dropout": args.transformer_dropout,
        "obs_dim": obs_dim,
        "action_dims": action_dims,
        "seed": args.seed,
        "per_agent_loss_history": per_agent_loss_history,
        "agent_0_work_fraction_in_demos": work_frac,
        "eval": eval_stats,
        "min_gap_closed_gate": args.min_gap_closed,
        "gate_passed": bool(eval_stats["gap_closed"] >= args.min_gap_closed),
    }
    with (args.output_dir / "bc_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(
        f"  BC eval: mean_step_team={eval_stats['mean_step_reward_team']:.3f} "
        f"gap_closed={eval_stats['gap_closed']:.3f} "
        f"(gate >= {args.min_gap_closed})"
    )

    if not summary["gate_passed"]:
        print(
            f"FAIL: BC did not take (gap_closed={eval_stats['gap_closed']:.3f} "
            f"< {args.min_gap_closed}). PPO continuation should NOT be launched."
        )
        raise SystemExit(2)
    print(
        f"OK: BC took (gap_closed={eval_stats['gap_closed']:.3f}). Safe for PPO continuation."
    )


if __name__ == "__main__":
    main()
