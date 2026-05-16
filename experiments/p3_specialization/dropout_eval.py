"""Single-agent dropout robustness evaluation for P3.

Loads a trained cell's policies and runs evaluation episodes under N+1
conditions:

- ``none`` --- baseline; all agents act normally.
- ``agent_i`` for ``i ∈ {0, …, N-1}`` --- agent i is replaced by a no-op
  policy that always picks ``[house=0, mode=REST=0]``.

For each condition we report mean team reward across ``num_episodes``
evaluation rollouts. Robustness to specialization shows up as a *smaller*
team-reward drop when each agent is removed individually --- which is the
P3 prediction the redundancy penalty should improve.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from bucket_brigade.envs.bucket_brigade_env import BucketBrigadeEnv
from bucket_brigade.envs.scenarios_generated import get_scenario_by_name
from bucket_brigade.training.networks import PolicyNetwork
from bucket_brigade.training.joint_trainer import flatten_dict_obs


def _build_policies(
    cell_dir: Path,
    obs_dim: int,
    action_dims: List[int],
    hidden_size: int,
    num_agents: int,
    device: torch.device,
) -> List[PolicyNetwork]:
    policies = []
    for i in range(num_agents):
        p = PolicyNetwork(
            obs_dim=obs_dim, action_dims=action_dims, hidden_size=hidden_size
        ).to(device)
        state = torch.load(
            cell_dir / "policies" / f"agent_{i}.pt",
            map_location=device,
            weights_only=True,
        )
        p.load_state_dict(state)
        p.eval()
        policies.append(p)
    return policies


@torch.no_grad()
def _run_episode(
    env: BucketBrigadeEnv,
    policies: List[PolicyNetwork],
    dropout_agent: Optional[int],
    device: torch.device,
    seed: Optional[int] = None,
) -> float:
    """Run a single episode; return total team reward summed over steps & agents.

    ``dropout_agent`` is ``None`` for the baseline or an agent index for the
    single-agent dropout condition (no-op replaces that agent's actions).
    """
    obs_dict = env.reset(seed=seed)
    total_reward = 0.0
    num_agents = len(policies)
    n_action_dims = 2
    while not env.done:
        obs = flatten_dict_obs(obs_dict)
        obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
        joint = np.zeros((num_agents, n_action_dims), dtype=np.int64)
        for i, policy in enumerate(policies):
            if i == dropout_agent:
                joint[i] = [0, 0]  # no-op: rest at house 0
                continue
            actions, _, _, _ = policy.get_action_and_value(obs_t)
            joint[i] = actions[0].cpu().numpy()
        obs_dict, rewards, _, _ = env.step(joint)
        total_reward += float(rewards.sum())
    return total_reward


def evaluate_cell(
    cell_dir: Path,
    num_episodes: int = 50,
    device: str = "cpu",
    seed_offset: int = 10_000,
) -> Dict[str, Dict[str, float]]:
    """Evaluate a single trained cell under each dropout condition.

    Args:
        cell_dir: Directory containing ``config.json`` and ``policies/``.
        num_episodes: Evaluation episodes per condition.
        device: ``"cpu"`` or ``"cuda"``.
        seed_offset: Eval seeds are disjoint from training seeds:
            ``seed = seed_offset + k`` for ``k = 0..num_episodes-1``.

    Returns:
        Dict mapping condition name -> {"mean", "std", "n"} of team reward.
    """
    with (cell_dir / "config.json").open() as f:
        cfg = json.load(f)

    scenario = get_scenario_by_name(cfg["scenario"], num_agents=cfg["num_agents"])
    env_probe = BucketBrigadeEnv(scenario=scenario)
    obs_dim = flatten_dict_obs(env_probe.reset(seed=0)).shape[0]

    device_t = torch.device(device)
    policies = _build_policies(
        cell_dir=cell_dir,
        obs_dim=obs_dim,
        action_dims=cfg["action_dims"],
        hidden_size=cfg["hidden_size"],
        num_agents=cfg["num_agents"],
        device=device_t,
    )

    results: Dict[str, Dict[str, float]] = {}
    conditions: List[Optional[int]] = [None] + list(range(cfg["num_agents"]))
    for cond in conditions:
        name = "none" if cond is None else f"agent_{cond}"
        rewards = []
        env = BucketBrigadeEnv(scenario=scenario)
        for k in range(num_episodes):
            r = _run_episode(env, policies, cond, device_t, seed=seed_offset + k)
            rewards.append(r)
        results[name] = {
            "mean": float(np.mean(rewards)),
            "std": float(np.std(rewards)),
            "n": num_episodes,
        }

    with (cell_dir / "dropout_results.json").open("w") as f:
        json.dump(results, f, indent=2)
    return results


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cell-dir", type=Path, help="Single cell to evaluate.")
    p.add_argument(
        "--sweep-root",
        type=Path,
        help="Walk this directory and evaluate every cell with policies/.",
    )
    p.add_argument("--num-episodes", type=int, default=50)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    if args.cell_dir is not None:
        print(f"Evaluating cell: {args.cell_dir}")
        results = evaluate_cell(args.cell_dir, args.num_episodes, args.device)
        print(json.dumps(results, indent=2))
        return

    if args.sweep_root is None:
        p.error("either --cell-dir or --sweep-root is required")

    cells = sorted(args.sweep_root.rglob("config.json"))
    print(f"Found {len(cells)} cells under {args.sweep_root}")
    for cfg_path in cells:
        cell = cfg_path.parent
        if not (cell / "policies").is_dir():
            print(f"  skip (no policies): {cell}")
            continue
        print(f"  eval: {cell}")
        evaluate_cell(cell, args.num_episodes, args.device)


if __name__ == "__main__":
    main()
