"""Specialist BC-fit-only diagnostic (issue #272).

Phase-1 discriminator for the PPO-plateau debugging effort: can the standard
``PolicyNetwork`` (obs_dim=46, action_dims=[10,2,2], hidden_size=64, ~8.1k
params) actually represent the hand-coded specialist policy on
``minimal_specialization``? If supervised behavioural-cloning loss collapses
and per-head accuracy is high, the PPO plateau is a path-finding problem
(strengthens the case for MAPPO/CTDE in #270 etc.). If BC plateaus high,
we have a representational gap and need a wider/deeper net.

No env rollouts during training — pure supervised cross-entropy on
``(flatten_dict_obs, specialist_action)`` pairs. See issue #272 + the
curator-enriched comment for thresholds.

Usage:
    uv run python experiments/p3_specialization/bc_fit_only.py
    uv run python experiments/p3_specialization/bc_fit_only.py --epochs 10 --seed 0
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

from bucket_brigade.baselines.specialist import (
    specialist_action,
    specialist_action_joint,
)
from bucket_brigade.envs.bucket_brigade_env import BucketBrigadeEnv
from bucket_brigade.envs.scenarios_generated import get_scenario_by_name
from bucket_brigade.training.joint_trainer import flatten_dict_obs
from bucket_brigade.training.networks import PolicyNetwork, TransformerPolicyNetwork


# ---------------------------------------------------------------------------
# Demonstration generation
# ---------------------------------------------------------------------------


def generate_demonstrations(
    num_steps: int,
    num_agents: int,
    scenario_name: str,
    seed: int,
    epsilon: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Roll a (mostly-specialist + epsilon-noise) policy and record
    ``(flat_obs, specialist_label)`` pairs across all agents.

    The behaviour policy is specialist + epsilon-greedy noise (random house +
    random mode/signal) so we visit some states the pure specialist would
    never reach. The *labels* are always the specialist's deterministic
    action on the current obs — that is what we want the BC net to fit.

    Returns
    -------
    obs_array : np.ndarray, shape (num_steps * num_agents, obs_dim), float32
    label_array : np.ndarray, shape (num_steps * num_agents, 3), int64
    info : dict with bookkeeping (label distribution, etc.)
    """
    rng = np.random.RandomState(seed)
    scenario = get_scenario_by_name(scenario_name, num_agents)
    env = BucketBrigadeEnv(scenario=scenario, num_agents=num_agents)
    obs = env.reset(seed=seed)

    num_houses = env.num_houses

    flat_dim = flatten_dict_obs(obs, agent_id=0, num_agents=num_agents).shape[0]

    obs_buf = np.zeros((num_steps * num_agents, flat_dim), dtype=np.float32)
    label_buf = np.zeros((num_steps * num_agents, 3), dtype=np.int64)

    cursor = 0
    episodes = 0
    for _ in range(num_steps):
        # Record (per-agent flat obs, specialist label) for the current state.
        # Specialist labels are deterministic functions of obs + agent_id.
        for agent_id in range(num_agents):
            obs_buf[cursor] = flatten_dict_obs(
                obs, agent_id=agent_id, num_agents=num_agents
            )
            label_buf[cursor] = specialist_action(obs, agent_id, num_agents, num_houses)
            cursor += 1

        # Behaviour policy: specialist + epsilon-greedy noise. Joint action
        # has shape (num_agents, 3).
        joint = specialist_action_joint(obs, num_agents, num_houses)
        for agent_id in range(num_agents):
            if rng.random() < epsilon:
                joint[agent_id, 0] = rng.randint(num_houses)
                joint[agent_id, 1] = rng.randint(2)
                joint[agent_id, 2] = rng.randint(2)

        obs, _rewards, dones, _info = env.step(joint)
        if bool(dones[0]):
            obs = env.reset(seed=seed + 1 + episodes)
            episodes += 1

    obs_buf = obs_buf[:cursor]
    label_buf = label_buf[:cursor]

    # Label distribution bookkeeping — verifies we see enough WORK rows.
    mode_counts = np.bincount(label_buf[:, 1], minlength=2).tolist()
    signal_counts = np.bincount(label_buf[:, 2], minlength=2).tolist()
    house_counts = np.bincount(label_buf[:, 0], minlength=num_houses).tolist()
    work_frac = float(label_buf[:, 1].mean())

    info = {
        "num_pairs": int(cursor),
        "num_episodes_completed": int(episodes),
        "flat_obs_dim": int(flat_dim),
        "mode_counts": mode_counts,
        "signal_counts": signal_counts,
        "house_counts": house_counts,
        "work_frac": work_frac,
    }
    return obs_buf, label_buf, info


# ---------------------------------------------------------------------------
# Supervised training
# ---------------------------------------------------------------------------


def _compute_class_weights(
    labels_train: torch.Tensor, num_houses: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute per-head class weights using the canonical
    ``N / (num_classes * bincount)`` formula.

    Returns (house_w, mode_w, signal_w) tensors, each summing to ``num_classes``
    in expectation when classes are balanced. Empty classes are floored to 1
    count to avoid division-by-zero (the weight for an unseen class becomes
    ``N / num_classes`` — large, but unused since no samples carry that label).
    """
    n = labels_train.shape[0]
    house_counts = torch.bincount(labels_train[:, 0], minlength=num_houses).clamp(min=1)
    mode_counts = torch.bincount(labels_train[:, 1], minlength=2).clamp(min=1)
    signal_counts = torch.bincount(labels_train[:, 2], minlength=2).clamp(min=1)

    house_w = n / (num_houses * house_counts.float())
    mode_w = n / (2 * mode_counts.float())
    signal_w = n / (2 * signal_counts.float())
    return house_w, mode_w, signal_w


def _build_network(
    architecture: str,
    obs_dim: int,
    action_dims: list,
    hidden_size: int,
    transformer_d_model: int = 256,
    transformer_nhead: int = 4,
    transformer_num_layers: int = 3,
    transformer_dim_feedforward: int = 512,
    transformer_dropout: float = 0.1,
):
    """Instantiate the chosen policy architecture.

    Default ``mlp`` preserves the pre-#281 :class:`PolicyNetwork` path
    (bit-identical when ``architecture='mlp'``). ``transformer`` uses
    :class:`TransformerPolicyNetwork` (#281 escalation).
    """
    if architecture == "mlp":
        return PolicyNetwork(
            obs_dim=obs_dim, action_dims=action_dims, hidden_size=hidden_size
        )
    if architecture == "transformer":
        return TransformerPolicyNetwork(
            obs_dim=obs_dim,
            action_dims=action_dims,
            d_model=transformer_d_model,
            nhead=transformer_nhead,
            num_layers=transformer_num_layers,
            dim_feedforward=transformer_dim_feedforward,
            dropout=transformer_dropout,
        )
    raise ValueError(
        f"Unknown --architecture {architecture!r}; expected 'mlp' or 'transformer'."
    )


def train_bc(
    obs: np.ndarray,
    labels: np.ndarray,
    num_houses: int,
    hidden_size: int,
    lr: float,
    batch_size: int,
    epochs: int,
    train_frac: float,
    seed: int,
    class_balanced: bool = False,
    architecture: str = "mlp",
    transformer_d_model: int = 256,
    transformer_nhead: int = 4,
    transformer_num_layers: int = 3,
    transformer_dim_feedforward: int = 512,
    transformer_dropout: float = 0.1,
) -> dict:
    """Train a policy network via sum-of-cross-entropy over its 3 heads.

    The ``architecture`` parameter selects between :class:`PolicyNetwork`
    (default; ``'mlp'``) and :class:`TransformerPolicyNetwork` (``'transformer'``,
    #281 escalation). Default values preserve bit-identical behaviour with
    the pre-#281 MLP path.

    Returns a dict with per-epoch eval losses + per-head accuracies and the
    final-epoch summary for verdict thresholding.

    When ``class_balanced=True`` (issue #279), enables BOTH:

    - Weighted cross-entropy per head, using
      ``class_weight = N / (num_classes * bincount)`` from the training labels.
    - ``WeightedRandomSampler``-style minibatch oversampling weighted by the
      inverse-frequency of the *mode* class (the primary imbalance lever:
      19:1 REST:WORK in the canonical scenario). This lifts the gradient
      signal on all 3 heads on WORK rows, which is where the headline
      ``house_acc_on_work_subset`` metric lives.

    When ``class_balanced=False`` (default), behaviour is bit-identical to
    the pre-#279 path (no weighting, ``torch.randperm`` minibatches).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    n = obs.shape[0]
    perm = np.random.RandomState(seed).permutation(n)
    n_train = int(round(train_frac * n))
    train_idx = perm[:n_train]
    eval_idx = perm[n_train:]

    x_train = torch.from_numpy(obs[train_idx])
    y_train = torch.from_numpy(labels[train_idx])
    x_eval = torch.from_numpy(obs[eval_idx])
    y_eval = torch.from_numpy(labels[eval_idx])

    obs_dim = obs.shape[1]
    action_dims = [num_houses, 2, 2]
    net = _build_network(
        architecture=architecture,
        obs_dim=obs_dim,
        action_dims=action_dims,
        hidden_size=hidden_size,
        transformer_d_model=transformer_d_model,
        transformer_nhead=transformer_nhead,
        transformer_num_layers=transformer_num_layers,
        transformer_dim_feedforward=transformer_dim_feedforward,
        transformer_dropout=transformer_dropout,
    )

    n_params = sum(p.numel() for p in net.parameters())

    optim = torch.optim.Adam(net.parameters(), lr=lr)

    # Class-balanced setup: build per-head CE weight tensors and per-row
    # sampler weights once, outside the training loop.
    if class_balanced:
        house_w, mode_w, signal_w = _compute_class_weights(y_train, num_houses)
        head_weights = [house_w, mode_w, signal_w]
        # Per-row sampler weights = inverse of (mode) class frequency. Using
        # only the mode head's frequency keeps the sampler interpretable
        # ("oversample WORK rows by ~19x") while still amplifying signal on
        # the house head (most of those WORK rows hit houses 4..N-1 which
        # are the under-represented houses).
        mode_counts = torch.bincount(y_train[:, 1], minlength=2).clamp(min=1)
        row_inv_freq = 1.0 / mode_counts.float()
        sample_weights = row_inv_freq[y_train[:, 1]]
        print(
            "[bc_fit_only] class-balanced ON: head class-weights:\n"
            f"  house ({num_houses}-way) min={float(house_w.min()):.3f} "
            f"max={float(house_w.max()):.3f}\n"
            f"  mode  (2-way)   = [{float(mode_w[0]):.3f}, {float(mode_w[1]):.3f}]\n"
            f"  signal(2-way)   = [{float(signal_w[0]):.3f}, {float(signal_w[1]):.3f}]\n"
            f"  row sampler: mode_counts={mode_counts.tolist()} "
            f"-> oversample-WORK factor ~ "
            f"{float(mode_counts[0]) / float(mode_counts[1]):.1f}x"
        )
    else:
        head_weights = [None, None, None]
        sample_weights = None

    history = []
    for epoch in range(epochs):
        net.train()
        if class_balanced:
            # WeightedRandomSampler with replacement: draw n_train indices per
            # epoch with probability proportional to sample_weights. The
            # generator is seeded per-epoch so two runs with the same seed
            # produce identical index sequences.
            gen = torch.Generator()
            gen.manual_seed(seed * 10_000 + epoch)
            sampler = torch.utils.data.WeightedRandomSampler(
                weights=sample_weights,
                num_samples=n_train,
                replacement=True,
                generator=gen,
            )
            idx = torch.tensor(list(sampler), dtype=torch.long)
        else:
            idx = torch.randperm(n_train)
        train_loss_sum = 0.0
        train_work_seen = 0
        for start in range(0, n_train, batch_size):
            batch = idx[start : start + batch_size]
            xb = x_train[batch]
            yb = y_train[batch]
            logits, _ = net(xb)
            loss = sum(
                F.cross_entropy(logits[k], yb[:, k], weight=head_weights[k])
                for k in range(3)
            )
            optim.zero_grad()
            loss.backward()
            optim.step()
            train_loss_sum += float(loss.detach()) * xb.shape[0]
            train_work_seen += int(yb[:, 1].sum())
        train_loss_avg = train_loss_sum / n_train
        train_work_frac = train_work_seen / n_train

        # Eval: NOTE eval loss is always *unweighted* CE, so it remains
        # comparable across class_balanced=True/False runs. Eval split
        # composition is also unchanged by oversampling — sampling only
        # affects the training minibatch path.
        net.eval()
        with torch.no_grad():
            logits, _ = net(x_eval)
            eval_loss = sum(F.cross_entropy(logits[k], y_eval[:, k]) for k in range(3))
            preds = [logits[k].argmax(dim=-1) for k in range(3)]
            head_acc = [
                float((preds[k] == y_eval[:, k]).float().mean()) for k in range(3)
            ]
            joint_correct = (
                (preds[0] == y_eval[:, 0])
                & (preds[1] == y_eval[:, 1])
                & (preds[2] == y_eval[:, 2])
            )
            joint_acc = float(joint_correct.float().mean())

        epoch_entry = {
            "epoch": epoch + 1,
            "train_loss": train_loss_avg,
            "eval_loss": float(eval_loss),
            "house_acc": head_acc[0],
            "mode_acc": head_acc[1],
            "signal_acc": head_acc[2],
            "joint_acc": joint_acc,
        }
        if class_balanced:
            epoch_entry["train_work_frac"] = train_work_frac
        history.append(epoch_entry)
        epoch_msg = (
            f"  epoch {epoch + 1:2d}/{epochs}  "
            f"train_loss={train_loss_avg:.4f}  "
            f"eval_loss={float(eval_loss):.4f}  "
            f"house={head_acc[0]:.3f}  mode={head_acc[1]:.3f}  "
            f"signal={head_acc[2]:.3f}  joint={joint_acc:.3f}"
        )
        if class_balanced:
            epoch_msg += f"  work_frac={train_work_frac:.3f}"
        print(epoch_msg)

    # Confusion bookkeeping on the house head conditioned on mode==WORK,
    # since that's the only non-trivial case for the specialist policy.
    net.eval()
    with torch.no_grad():
        logits, _ = net(x_eval)
        pred_house = logits[0].argmax(dim=-1)
    work_mask = y_eval[:, 1] == 1
    work_count = int(work_mask.sum())
    if work_count > 0:
        house_acc_work = float(
            (pred_house[work_mask] == y_eval[work_mask, 0]).float().mean()
        )
    else:
        house_acc_work = float("nan")

    final = history[-1]
    return {
        "architecture": architecture,
        "n_params": n_params,
        "obs_dim": obs_dim,
        "n_train": n_train,
        "n_eval": int(n - n_train),
        "history": history,
        "final": final,
        "house_acc_on_work_subset": house_acc_work,
        "eval_work_rows": work_count,
    }


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------


def compute_verdict(final: dict) -> str:
    """Apply curator's verdict thresholds verbatim (issue #272)."""
    eval_loss = final["eval_loss"]
    joint_acc = final["joint_acc"]
    min_head_acc = min(final["house_acc"], final["mode_acc"], final["signal_acc"])
    if eval_loss < 0.05 and joint_acc > 0.90:
        return "REPRESENTABLE"
    if eval_loss > 0.5 and min_head_acc < 0.50:
        return "ARCHITECTURE_MISMATCH"
    return "INDUCTIVE_BIAS_GAP"


def compute_verdict_279(house_acc_on_work_subset: float) -> str:
    """Apply the #279 class-balanced verdict ladder verbatim.

    | house_acc_on_work_subset | Verdict             | Implication                          |
    |--------------------------|---------------------|--------------------------------------|
    | >= 0.90                  | CLASS_IMBALANCE     | Architecture has capacity            |
    | 0.50 .. 0.90             | PARTIAL             | Both effects contribute              |
    | <= 0.50                  | CAPACITY            | Confirmed capacity gap               |
    """
    if house_acc_on_work_subset != house_acc_on_work_subset:  # NaN guard
        return "INSUFFICIENT_DATA"
    if house_acc_on_work_subset >= 0.90:
        return "CLASS_IMBALANCE"
    if house_acc_on_work_subset > 0.50:
        return "PARTIAL"
    return "CAPACITY"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Specialist BC-fit-only diagnostic (issue #272)"
    )
    parser.add_argument("--num-agents", type=int, default=4)
    parser.add_argument("--scenario", type=str, default="minimal_specialization")
    parser.add_argument(
        "--num-steps",
        type=int,
        default=2500,
        help="Env steps to roll; pairs = num_steps * num_agents (default 10k pairs).",
    )
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--class-balanced",
        action="store_true",
        help=(
            "Issue #279: enable weighted cross-entropy AND WeightedRandomSampler "
            "minibatch oversampling to discriminate capacity gap from "
            "class-imbalance underfitting. Eval split + eval loss remain "
            "unweighted/unsampled so the headline house_acc_on_work_subset is "
            "comparable across runs."
        ),
    )
    parser.add_argument(
        "--architecture",
        type=str,
        choices=["mlp", "transformer"],
        default="mlp",
        help=(
            "Policy architecture: 'mlp' (default; PolicyNetwork) preserves the "
            "pre-#281 path. 'transformer' uses TransformerPolicyNetwork (#281 "
            "escalation) — ~350K params vs ~14K for the MLP at hidden_size=64."
        ),
    )
    parser.add_argument(
        "--transformer-d-model",
        type=int,
        default=256,
        help="Transformer embedding dim (only if --architecture transformer).",
    )
    parser.add_argument(
        "--transformer-nhead",
        type=int,
        default=4,
        help="Transformer attention heads (only if --architecture transformer).",
    )
    parser.add_argument(
        "--transformer-num-layers",
        type=int,
        default=3,
        help="Transformer encoder layers (only if --architecture transformer).",
    )
    parser.add_argument(
        "--transformer-dim-feedforward",
        type=int,
        default=512,
        help="Transformer FFN dim (only if --architecture transformer).",
    )
    parser.add_argument(
        "--transformer-dropout",
        type=float,
        default=0.1,
        help="Transformer dropout (only if --architecture transformer).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="experiments/p3_specialization/bc_fit_only_result.json",
    )
    args = parser.parse_args()

    t0 = time.time()
    print(
        f"[bc_fit_only] scenario={args.scenario} num_agents={args.num_agents} "
        f"num_steps={args.num_steps} epsilon={args.epsilon} seed={args.seed}"
    )
    print("[bc_fit_only] generating demonstrations ...")
    obs, labels, demo_info = generate_demonstrations(
        num_steps=args.num_steps,
        num_agents=args.num_agents,
        scenario_name=args.scenario,
        seed=args.seed,
        epsilon=args.epsilon,
    )
    t_gen = time.time() - t0
    print(
        f"[bc_fit_only] generated {demo_info['num_pairs']} pairs in {t_gen:.1f}s "
        f"(work_frac={demo_info['work_frac']:.3f}, "
        f"episodes={demo_info['num_episodes_completed']})"
    )

    # Probe num_houses from the env so the action head matches the scenario.
    scenario = get_scenario_by_name(args.scenario, args.num_agents)
    num_houses = int(getattr(scenario, "num_houses", 10))

    if args.architecture == "mlp":
        print(
            f"[bc_fit_only] training PolicyNetwork(obs_dim={obs.shape[1]}, "
            f"action_dims=[{num_houses},2,2], hidden_size={args.hidden_size})"
        )
    else:
        print(
            f"[bc_fit_only] training TransformerPolicyNetwork(obs_dim={obs.shape[1]}, "
            f"action_dims=[{num_houses},2,2], d_model={args.transformer_d_model}, "
            f"nhead={args.transformer_nhead}, num_layers={args.transformer_num_layers})"
        )
    t1 = time.time()
    result = train_bc(
        obs=obs,
        labels=labels,
        num_houses=num_houses,
        hidden_size=args.hidden_size,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        train_frac=args.train_frac,
        seed=args.seed,
        class_balanced=args.class_balanced,
        architecture=args.architecture,
        transformer_d_model=args.transformer_d_model,
        transformer_nhead=args.transformer_nhead,
        transformer_num_layers=args.transformer_num_layers,
        transformer_dim_feedforward=args.transformer_dim_feedforward,
        transformer_dropout=args.transformer_dropout,
    )
    t_train = time.time() - t1
    print(f"[bc_fit_only] training done in {t_train:.1f}s")

    final = result["final"]
    verdict = compute_verdict(final)
    verdict_279 = compute_verdict_279(result["house_acc_on_work_subset"])

    print()
    print("=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"architecture:     {result['architecture']}")
    print(f"obs_dim:          {result['obs_dim']}")
    print(f"n_params:         {result['n_params']}")
    print(f"train/eval pairs: {result['n_train']} / {result['n_eval']}")
    print(f"final eval_loss:  {final['eval_loss']:.4f}")
    print("per-head accuracy (eval):")
    print(f"  house  (10-way): {final['house_acc']:.3f}")
    print(f"  mode   (2-way) : {final['mode_acc']:.3f}")
    print(f"  signal (2-way) : {final['signal_acc']:.3f}")
    print(f"joint accuracy   : {final['joint_acc']:.3f}")
    print(
        f"house acc | mode=WORK subset (n={result['eval_work_rows']}): "
        f"{result['house_acc_on_work_subset']:.3f}"
    )
    print()
    print(f"VERDICT: {verdict}")
    if args.class_balanced:
        print(f"VERDICT_279 (class-balanced ladder): {verdict_279}")
    print("=" * 60)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_payload = {
        "args": vars(args),
        "demo_info": demo_info,
        "result": result,
        "verdict": verdict,
        "timing": {"data_gen_sec": t_gen, "train_sec": t_train},
    }
    if args.class_balanced:
        out_payload["verdict_279"] = verdict_279
    with open(out_path, "w") as f:
        json.dump(out_payload, f, indent=2)
    print(f"[bc_fit_only] wrote {out_path}")


if __name__ == "__main__":
    main()
