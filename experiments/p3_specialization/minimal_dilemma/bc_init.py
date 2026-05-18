"""BC + eval gate for the minimal dilemma (issue #292).

Adapts ``experiments/p3_specialization/bc_init.py`` (issue #270) to the
:class:`MinimalDilemmaEnv` with the new specialists. Three phases mirror the
bucket-brigade BC pipeline so the verdict semantics are directly comparable:

1. **Demo gathering** — roll the chosen specialist (``always_cooperate``
   or ``tit_for_tat``) on the dilemma env until we have ``--num-pairs``
   per-agent ``(flat_obs, action)`` tuples. Observations are flattened
   with :func:`flatten_dict_obs` including the per-agent identity one-hot
   tail (#204) so the resulting state dicts plug into ``trainer.policies``
   with zero shape massaging.

2. **Supervised fit** — one :class:`PolicyNetwork` per agent, cross-entropy
   on the single ``Discrete(2)`` head, Adam, ``--epochs`` (default 10).

3. **Eval gate** — ``--eval-episodes`` deterministic-argmax episodes against
   the env. Pass if action-prediction accuracy ≥ ``--min-action-accuracy``
   AND mean per-step per-agent reward ≥ ``--min-mean-reward`` (defaults
   reflect always_cooperate self-play: 99% accuracy, 0.5 per-step). The
   script exits non-zero on failure so the downstream PPO continuation
   isn't launched from a bad init.

Per-agent state dicts are saved to ``<output_dir>/agent_{i}.pt``, matching
the filename layout :func:`train_one_cell` (in ``train_ippo.py``) expects
for ``--bc-init-checkpoint-dir``.

Usage::

    # always_cooperate fit (default specialist)
    uv run python -m experiments.p3_specialization.minimal_dilemma.bc_init \\
        --output-dir experiments/p3_specialization/minimal_dilemma/results/bc_alwaysC \\
        --num-pairs-per-agent 5000

    # tit_for_tat fit
    uv run python -m experiments.p3_specialization.minimal_dilemma.bc_init \\
        --specialist tit_for_tat --output-dir .../bc_tft
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from bucket_brigade.training.joint_trainer import flatten_dict_obs
from bucket_brigade.training.networks import PolicyNetwork

from experiments.p3_specialization.minimal_dilemma.env import (
    ACTION_COOPERATE,
    EPISODE_LENGTH,
    MULTIPLIER,
    MinimalDilemmaEnv,
    NUM_AGENTS,
    REWARD_MUTUAL_COOPERATE,
    REWARD_MUTUAL_DEFECT,
)
from experiments.p3_specialization.minimal_dilemma.specialists import (
    always_cooperate,
    tit_for_tat,
)


ACTION_DIMS: List[int] = [2]

# Specialist registry: name → callable. Used by the CLI to pick a fit target.
SPECIALISTS = {
    "always_cooperate": always_cooperate,
    "tit_for_tat": tit_for_tat,
}


def _gap_closed(mean_per_agent: float) -> float:
    """Fraction of the defect→cooperate gap closed (mutual play self-evaluation)."""
    span = REWARD_MUTUAL_COOPERATE - REWARD_MUTUAL_DEFECT
    if span <= 0:
        return float("nan")
    return (mean_per_agent - REWARD_MUTUAL_DEFECT) / span


def gather_demos(
    specialist: Callable,
    num_pairs_per_agent: int,
    seed: int,
    multiplier: float = MULTIPLIER,
    episode_length: int = EPISODE_LENGTH,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Roll the specialist until we have ``num_pairs_per_agent`` demos per agent.

    Mirrors :func:`experiments.p3_specialization.bc_init.gather_demos`. Returns
    ``(obs_per_agent, act_per_agent)`` with the same dtype and shape contract.
    """
    env = MinimalDilemmaEnv(multiplier=multiplier, episode_length=episode_length)
    rng = np.random.default_rng(seed)

    obs_buf: List[List[np.ndarray]] = [[] for _ in range(NUM_AGENTS)]
    act_buf: List[List[np.ndarray]] = [[] for _ in range(NUM_AGENTS)]

    # Each step contributes exactly one (obs, action) pair per agent (the
    # specialist is deterministic). Stop when agent 0 reaches the target.
    while len(obs_buf[0]) < num_pairs_per_agent:
        obs = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        while not env.done:
            joint = specialist(obs, num_agents=NUM_AGENTS)
            for i in range(NUM_AGENTS):
                obs_buf[i].append(
                    flatten_dict_obs(obs, agent_id=i, num_agents=NUM_AGENTS)
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
    hidden_size: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    seed: int,
) -> Tuple[PolicyNetwork, List[float]]:
    """Supervised fit of one PolicyNetwork to ``(obs, acts)``.

    Mirrors :func:`experiments.p3_specialization.bc_init.bc_fit_one_agent` but
    for the single-head dilemma action space (``ACTION_DIMS = [2]``).
    """
    torch.manual_seed(seed)
    policy = PolicyNetwork(
        obs_dim=obs_dim, action_dims=ACTION_DIMS, hidden_size=hidden_size
    ).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)

    x_t = torch.from_numpy(obs).float().to(device)
    a_t = torch.from_numpy(acts).long().to(device)
    ds = TensorDataset(x_t, a_t)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    epoch_losses: List[float] = []
    for _ in range(epochs):
        batch_losses: List[float] = []
        for x, a in dl:
            logits, _ = policy(x)
            # Single head — sum-of-heads collapses to one CE term.
            loss = F.cross_entropy(logits[0], a[:, 0])
            opt.zero_grad()
            loss.backward()
            opt.step()
            batch_losses.append(float(loss.item()))
        epoch_losses.append(float(np.mean(batch_losses)))
    return policy, epoch_losses


def action_accuracy(
    policy: PolicyNetwork,
    obs: np.ndarray,
    acts: np.ndarray,
    device: str,
) -> float:
    """Action-prediction accuracy of the BC policy on held-out ``(obs, acts)``."""
    policy.eval()
    with torch.no_grad():
        x = torch.from_numpy(obs).float().to(device)
        logits, _ = policy(x)
        pred = logits[0].argmax(dim=-1).cpu().numpy()
    return float((pred == acts[:, 0]).mean())


def evaluate_policies(
    policies: List[PolicyNetwork],
    num_episodes: int,
    device: str,
    multiplier: float = MULTIPLIER,
    episode_length: int = EPISODE_LENGTH,
    seed: int = 0,
) -> dict:
    """Roll the BC policies (argmax) for ``num_episodes`` episodes.

    Returns mean per-step per-agent reward and gap_closed (defect → cooperate
    self-play span).
    """
    env = MinimalDilemmaEnv(multiplier=multiplier, episode_length=episode_length)

    for p in policies:
        p.eval()

    ep_per_step: List[float] = []
    coop_counts: List[float] = []
    with torch.no_grad():
        for ep in range(num_episodes):
            obs = env.reset(seed=seed + ep)
            total = 0.0
            n_steps = 0
            n_coop = 0
            while not env.done:
                joint_rows: List[List[int]] = []
                for i in range(NUM_AGENTS):
                    x = flatten_dict_obs(obs, agent_id=i, num_agents=NUM_AGENTS)
                    xt = torch.from_numpy(x).float().to(device).unsqueeze(0)
                    actions, _, _ = policies[i].get_action(xt, deterministic=True)
                    a_i = int(actions[0].item())
                    joint_rows.append([a_i])
                    if a_i == ACTION_COOPERATE:
                        n_coop += 1
                joint = np.asarray(joint_rows, dtype=np.int64)
                obs, rewards, _, _ = env.step(joint)
                total += float(rewards.sum())
                n_steps += 1
            if n_steps > 0:
                # per-step per-agent reward = team_total / (n_steps * num_agents).
                ep_per_step.append(total / (n_steps * NUM_AGENTS))
                coop_counts.append(n_coop / (n_steps * NUM_AGENTS))

    per_step = np.asarray(ep_per_step, dtype=np.float64)
    return {
        "n_episodes": int(per_step.size),
        "mean_step_reward_per_agent": float(per_step.mean())
        if per_step.size > 0
        else 0.0,
        "std_step_reward_per_agent": float(per_step.std(ddof=1))
        if per_step.size > 1
        else 0.0,
        "mean_cooperation_fraction": float(np.mean(coop_counts))
        if coop_counts
        else 0.0,
        "gap_closed": _gap_closed(float(per_step.mean()))
        if per_step.size > 0
        else float("nan"),
        "reward_mutual_defect_ref": REWARD_MUTUAL_DEFECT,
        "reward_mutual_cooperate_ref": REWARD_MUTUAL_COOPERATE,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--specialist",
        choices=sorted(SPECIALISTS.keys()),
        default="always_cooperate",
        help="Which hand-coded specialist to clone.",
    )
    p.add_argument("--num-pairs-per-agent", type=int, default=5000)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument(
        "--min-action-accuracy",
        type=float,
        default=0.99,
        help="Test-set accuracy threshold for the BC fit gate (issue spec).",
    )
    p.add_argument(
        "--min-mean-reward",
        type=float,
        default=0.5,
        help=(
            "Per-step per-agent reward floor for the BC eval gate. For the "
            "always_cooperate specialist this corresponds to ≥ 0.5 (mutual-C "
            "yields 0.6); set lower if cloning tit_for_tat with noise."
        ),
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    specialist = SPECIALISTS[args.specialist]

    print(
        f"== minimal_dilemma BC: specialist={args.specialist} "
        f"num_pairs_per_agent={args.num_pairs_per_agent} epochs={args.epochs} "
        f"seed={args.seed} =="
    )

    # ----- Phase 1: gather demos -----
    print(f"[1/3] Gathering demos (target={args.num_pairs_per_agent}/agent)...")
    obs_per_agent, act_per_agent = gather_demos(
        specialist=specialist,
        num_pairs_per_agent=args.num_pairs_per_agent,
        seed=args.seed,
    )
    obs_dim = int(obs_per_agent[0].shape[1])
    print(
        f"  collected per-agent obs shape: {obs_per_agent[0].shape}, obs_dim={obs_dim}"
    )

    # ----- Phase 2: BC fit per agent (90/10 train/test split) -----
    print(f"[2/3] Fitting {NUM_AGENTS} per-agent policies...")
    policies: List[PolicyNetwork] = []
    per_agent_loss_history: List[List[float]] = []
    per_agent_test_acc: List[float] = []
    for i in range(NUM_AGENTS):
        n = len(obs_per_agent[i])
        n_train = int(0.9 * n)
        # Deterministic split (no shuffle here — the dataloader handles shuffle
        # inside training). Holding out the *tail* of the demo stream gives a
        # disjoint set of episode segments per agent.
        x_train, x_test = obs_per_agent[i][:n_train], obs_per_agent[i][n_train:]
        a_train, a_test = act_per_agent[i][:n_train], act_per_agent[i][n_train:]
        policy, losses = bc_fit_one_agent(
            obs=x_train,
            acts=a_train,
            obs_dim=obs_dim,
            hidden_size=args.hidden_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
            seed=args.seed + i,
        )
        policies.append(policy)
        per_agent_loss_history.append(losses)
        acc = action_accuracy(policy, x_test, a_test, args.device)
        per_agent_test_acc.append(acc)
        print(f"  agent {i}: final_loss={losses[-1]:.5f} test_accuracy={acc:.4f}")

    # ----- Phase 3: save + eval gate -----
    print(f"[3/3] Saving policies and evaluating ({args.eval_episodes} episodes)...")
    for i, policy in enumerate(policies):
        torch.save(policy.state_dict(), args.output_dir / f"agent_{i}.pt")

    eval_stats = evaluate_policies(
        policies=policies,
        num_episodes=args.eval_episodes,
        device=args.device,
        seed=args.seed + 1000,
    )

    min_test_acc = float(min(per_agent_test_acc))
    summary = {
        "issue": 292,
        "specialist": args.specialist,
        "num_pairs_per_agent": args.num_pairs_per_agent,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "hidden_size": args.hidden_size,
        "obs_dim": obs_dim,
        "action_dims": ACTION_DIMS,
        "seed": args.seed,
        "per_agent_loss_history": per_agent_loss_history,
        "per_agent_test_accuracy": per_agent_test_acc,
        "min_test_accuracy": min_test_acc,
        "eval": eval_stats,
        "min_action_accuracy_gate": args.min_action_accuracy,
        "min_mean_reward_gate": args.min_mean_reward,
        "accuracy_gate_passed": bool(min_test_acc >= args.min_action_accuracy),
        "reward_gate_passed": bool(
            eval_stats["mean_step_reward_per_agent"] >= args.min_mean_reward
        ),
    }
    summary["gate_passed"] = bool(
        summary["accuracy_gate_passed"] and summary["reward_gate_passed"]
    )

    with (args.output_dir / "bc_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(
        f"  eval: per_agent={eval_stats['mean_step_reward_per_agent']:.3f} "
        f"coop_frac={eval_stats['mean_cooperation_fraction']:.3f} "
        f"gap_closed={eval_stats['gap_closed']:.3f}"
    )
    print(
        f"  gates: accuracy={summary['accuracy_gate_passed']} "
        f"(min={min_test_acc:.4f}, threshold={args.min_action_accuracy}); "
        f"reward={summary['reward_gate_passed']} "
        f"(mean={eval_stats['mean_step_reward_per_agent']:.3f}, "
        f"threshold={args.min_mean_reward})"
    )

    if not summary["gate_passed"]:
        print("FAIL: BC eval gate failed. PPO continuation should NOT be launched.")
        raise SystemExit(2)
    print("OK: BC eval gates passed. Safe for PPO continuation.")


if __name__ == "__main__":
    main()
