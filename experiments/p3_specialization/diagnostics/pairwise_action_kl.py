"""Pairwise action-distribution KL between agents on a trained cell.

For issue #220: confirm that the obs-differentiation fix (#216) lets PPO
produce distinguishable per-agent policies. Pre-#216 policies should yield
KL ≈ 0 (the identical-input pathology); post-#216 policies should produce
KL > 0 iff agents specialize.

For each step of one rollout we recover each agent's full action-head
softmax over the packed 20-class action space (10 houses x 2 modes) and
average the pairwise KL ``KL(p_i || p_j)`` over all steps. The result
is an ``N x N`` matrix (zero diagonal); we also print the off-diagonal mean.

Usage::

    uv run python experiments/p3_specialization/diagnostics/pairwise_action_kl.py \\
        --cell experiments/p3_specialization/runs/issue220_treatment/\\
minimal_specialization/lambda_0e0/seed_42

The script writes ``pairwise_action_kl.json`` into the cell directory so
``analyze_220.py`` can aggregate without re-running rollouts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from bucket_brigade.envs import BucketBrigadeEnv
from bucket_brigade.envs.scenarios_generated import get_scenario_by_name
from bucket_brigade.training.joint_trainer import JointPPOTrainer, flatten_dict_obs


def softmax_packed(logits_list: list[torch.Tensor]) -> torch.Tensor:
    """logits_list = [logits_house [B,10], logits_mode [B,2]] -> [B, 20] joint."""
    p_house = torch.softmax(logits_list[0], dim=-1)  # [B, 10]
    p_mode = torch.softmax(logits_list[1], dim=-1)  # [B, 2]
    # Joint over independent heads: outer product flattened.
    joint = (p_house.unsqueeze(-1) * p_mode.unsqueeze(-2)).reshape(p_house.shape[0], -1)
    return joint  # [B, 20]


def compute_pairwise_kl(cell_dir: Path, rollout_steps: int | None = None) -> dict:
    cfg = json.loads((cell_dir / "config.json").read_text())
    scenario = get_scenario_by_name(cfg["scenario"], num_agents=cfg["num_agents"])
    env_fn = lambda: BucketBrigadeEnv(scenario=scenario)  # noqa: E731

    probe = env_fn()
    obs_dim = flatten_dict_obs(
        probe.reset(seed=cfg["seed"]), agent_id=0, num_agents=cfg["num_agents"]
    ).shape[0]

    trainer = JointPPOTrainer(
        env_fn=env_fn,
        num_agents=cfg["num_agents"],
        obs_dim=obs_dim,
        action_dims=cfg["action_dims"],
        hidden_size=cfg["hidden_size"],
        lr=cfg["lr"],
        ppo_epochs=cfg["ppo_epochs"],
        minibatch_size=cfg["minibatch_size"],
        value_coef=cfg["value_coef"],
        entropy_coef=cfg["entropy_coef"],
        normalize_returns=cfg.get("normalize_returns", False),
        device=cfg.get("device", "cpu"),
        seed=cfg["seed"],
    )
    n = cfg["num_agents"]
    for i in range(n):
        sd = torch.load(
            cell_dir / f"policies/agent_{i}.pt",
            map_location=cfg.get("device", "cpu"),
            weights_only=True,
        )
        trainer.policies[i].load_state_dict(sd)

    steps = rollout_steps if rollout_steps is not None else cfg["rollout_steps"]
    rollout = trainer.collect_rollout(steps)
    # rollout.observations: [T, N, obs_dim]
    obs = rollout.observations  # torch.Tensor
    T = obs.shape[0]

    # Each agent's softmax over its own observation tail.
    probs = []
    with torch.no_grad():
        for i in range(n):
            logits, _ = trainer.policies[i].forward(obs[:, i, :])
            probs.append(softmax_packed(logits).cpu().numpy())  # [T, 20]
    probs = np.stack(probs, axis=0)  # [N, T, 20]
    eps = 1e-10
    P = probs + eps
    P = P / P.sum(axis=-1, keepdims=True)

    # KL(p_i || p_j) = sum_a p_i log(p_i / p_j), averaged over steps.
    kl = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            ratio = np.log(P[i] / P[j])  # [T, 20]
            kl_step = (P[i] * ratio).sum(axis=-1)  # [T]
            kl[i, j] = float(kl_step.mean())

    off_diag = kl[~np.eye(n, dtype=bool)]
    return {
        "cell": str(cell_dir),
        "scenario": cfg["scenario"],
        "seed": cfg["seed"],
        "num_agents": n,
        "rollout_steps": T,
        "kl_matrix": kl.tolist(),
        "kl_off_diag_mean": float(off_diag.mean()),
        "kl_off_diag_max": float(off_diag.max()),
        "kl_off_diag_min": float(off_diag.min()),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cell", type=Path, required=True)
    p.add_argument("--rollout-steps", type=int, default=None)
    p.add_argument(
        "--no-write",
        action="store_true",
        help="Print only; do not write pairwise_action_kl.json into the cell dir.",
    )
    args = p.parse_args()
    out = compute_pairwise_kl(args.cell, args.rollout_steps)
    print(json.dumps(out, indent=2))
    if not args.no_write:
        (args.cell / "pairwise_action_kl.json").write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.cell / 'pairwise_action_kl.json'}")


if __name__ == "__main__":
    main()
