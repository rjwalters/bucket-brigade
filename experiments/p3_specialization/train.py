"""Single-cell trainer for the P3 specialization experiment.

One invocation of :func:`train_one_cell` runs one ``(scenario, lambda_red, seed)``
training cell to completion: it spins up a :class:`JointPPOTrainer`, runs
``num_iterations`` rollout-and-update cycles, and writes the artifacts a
later analysis pass needs:

- ``policies/agent_{i}.pt`` --- final state dict for each agent.
- ``metrics.json`` --- per-iteration scalars (loss, reward, MI, etc.).
- ``config.json`` --- the exact arguments used (for reproducibility).

Plug-in conditional MI between encoder outputs is computed on each
rollout's most recent batch, conditioned on the quantized team reward
(per the paper's main-text definition ``I(Ẑ_i; Ẑ_j | R)``). Unconditional
MI is also logged so we can see the PMIC failure mode if it kicks in.
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from bucket_brigade.analysis.info_theory import (
    conditional_mutual_information,
    entropy_discrete,
    is_degenerate_conditioner,
    mutual_information,
    quantize_uniform,
)
from bucket_brigade.envs.bucket_brigade_env import BucketBrigadeEnv
from bucket_brigade.envs.scenarios_generated import get_scenario_by_name
from bucket_brigade.training.joint_trainer import (
    JointPPOTrainer,
    flatten_dict_obs,
)


@dataclass
class CellConfig:
    scenario: str
    lambda_red: float
    seed: int
    num_iterations: int = 50
    rollout_steps: int = 2048
    num_agents: int = 4
    hidden_size: int = 64
    lr: float = 3e-4
    ppo_epochs: int = 4
    minibatch_size: int = 256
    # PPO loss weights. Defaults match ``JointPPOTrainer.__init__`` so existing
    # callers see no behavior change. Phase 2 sweeps (issue #153) vary these to
    # test the value-loss-dominance hypothesis from the Phase 1 diagnostics
    # (see ``experiments/p3_specialization/diagnostics/summary.md``).
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    # Encoder outputs are quantized for plug-in MI. We first project from
    # ``hidden_size`` down to ``mi_proj_dims`` via a fixed random matrix
    # (seeded from ``seed``); then ``quantize_uniform`` packs each row into
    # a single integer code with up to ``n_bins ** mi_proj_dims`` values.
    n_bins: int = 4
    mi_proj_dims: int = 3
    device: str = "cpu"
    action_dims: List[int] = field(default_factory=lambda: [10, 2])


def _measure_information(
    trainer: JointPPOTrainer,
    rollout,
    n_bins: int,
    projection: np.ndarray,
) -> Dict[str, float]:
    """Plug-in MI/CMI on the rollout's encoder outputs.

    Conditioned on the *team reward* (sum over agents), quantized to
    ``n_bins`` levels. Also logs unconditional MI per pair.

    Encoder outputs are projected from ``hidden_size`` to a small dimension
    via the shared ``projection`` matrix before quantization, because a
    direct quantize-and-pack of a 64-D vector overflows the integer code.
    """
    with torch.no_grad():
        feats = trainer.encoder_outputs_batch(rollout.observations)
    feats_np = [f.cpu().numpy() @ projection for f in feats]

    # Quantize each (T, mi_proj_dims) projection into a single integer code.
    codes = [quantize_uniform(f, n_bins=n_bins) for f in feats_np]

    # Team reward conditioning variable, also quantized.
    rewards = (
        torch.stack([rollout.rewards[i] for i in range(trainer.num_agents)], dim=0)
        .sum(dim=0)
        .cpu()
        .numpy()
    )
    r_codes = quantize_uniform(rewards, n_bins=n_bins)

    out: Dict[str, float] = {}

    # Defensive measurement-quality check (see issue #146).
    #
    # When the conditioner ``r_codes`` is (near-)constant on the sample, the
    # plug-in CMI ``I(Ẑ_i; Ẑ_j | R)`` collapses to the unconditional MI
    # ``I(Ẑ_i; Ẑ_j)`` and the "conditional" claim is vacuous. The classic
    # offender in this experiment is the ``trivial_cooperation`` scenario,
    # where the team reward is essentially constant; near-deterministic-reward
    # scenarios fail less obviously but in the same way.
    #
    # NOTE: This check DETECTS the failure mode; it does not RESOLVE it. The
    # research-level question of which conditioner to use (state-coarsening,
    # per-agent reward, marginal-action codes, etc.) is deferred to the
    # Architect/Hermit framing described in the issue. We deliberately do not
    # pick a replacement here.
    is_degenerate, diag = is_degenerate_conditioner(r_codes)
    out["cmi/conditioner_n_distinct"] = float(diag["n_distinct"])
    out["cmi/conditioner_modal_fraction"] = diag["modal_fraction"]
    out["cmi/conditioner_entropy_bits"] = diag["entropy_bits"]
    out["cmi/conditioner_degenerate"] = float(is_degenerate)
    if is_degenerate:
        warnings.warn(
            (
                "P3 CMI conditioner appears degenerate "
                f"(n_distinct={diag['n_distinct']}, "
                f"modal_fraction={diag['modal_fraction']:.3f}, "
                f"entropy_bits={diag['entropy_bits']:.3f}). "
                "I(Ẑ_i; Ẑ_j | R) ≈ I(Ẑ_i; Ẑ_j) is mathematically guaranteed; "
                "see issue #146 for context. Reported CMI values for this "
                "iteration should not be interpreted as 'conditional'."
            ),
            RuntimeWarning,
            stacklevel=2,
        )

    n = trainer.num_agents
    mi_vals = []
    cmi_vals = []
    for i in range(n):
        for j in range(i + 1, n):
            mi = mutual_information(codes[i], codes[j])
            cmi = conditional_mutual_information(codes[i], codes[j], r_codes)
            out[f"mi/agent_{i}_{j}"] = mi
            out[f"cmi/agent_{i}_{j}"] = cmi
            mi_vals.append(mi)
            cmi_vals.append(cmi)
    out["mi/mean_pair"] = float(np.mean(mi_vals))
    out["cmi/mean_pair"] = float(np.mean(cmi_vals))

    # Marginal action entropy per agent (proxy for role entropy H(A_i^*)).
    for i in range(n):
        a = rollout.actions[i].cpu().numpy()
        # Pack multi-discrete action into a single label per step.
        packed = a[:, 0] * 2 + a[:, 1]  # 0..19 for [house, mode]
        out[f"action_entropy/agent_{i}"] = entropy_discrete(packed)
    out["action_entropy/mean"] = float(
        np.mean([out[f"action_entropy/agent_{i}"] for i in range(n)])
    )

    return out


def train_one_cell(cfg: CellConfig, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    scenario = get_scenario_by_name(cfg.scenario, num_agents=cfg.num_agents)

    def env_fn():
        env = BucketBrigadeEnv(scenario=scenario)
        return env

    # Probe obs_dim from a single reset.
    probe = env_fn()
    probe_obs = probe.reset(seed=cfg.seed)
    obs_dim = flatten_dict_obs(probe_obs).shape[0]

    trainer = JointPPOTrainer(
        env_fn=env_fn,
        num_agents=cfg.num_agents,
        obs_dim=obs_dim,
        action_dims=cfg.action_dims,
        hidden_size=cfg.hidden_size,
        lr=cfg.lr,
        ppo_epochs=cfg.ppo_epochs,
        minibatch_size=cfg.minibatch_size,
        value_coef=cfg.value_coef,
        entropy_coef=cfg.entropy_coef,
        redundancy_coef=cfg.lambda_red,
        device=cfg.device,
        seed=cfg.seed,
    )

    # Shared random projection matrix for MI measurement. Seeded from cfg.seed
    # so the metric is reproducible and comparable across iterations.
    rng = np.random.default_rng(cfg.seed)
    projection = rng.standard_normal((cfg.hidden_size, cfg.mi_proj_dims)).astype(
        np.float32
    )
    # Unit-norm columns make the projected values comparable across runs.
    projection /= np.linalg.norm(projection, axis=0, keepdims=True) + 1e-8

    metrics_log: List[Dict[str, float]] = []
    for it in range(cfg.num_iterations):
        rollout = trainer.collect_rollout(cfg.rollout_steps)
        stats = trainer.update(rollout)

        info_stats = _measure_information(
            trainer, rollout, n_bins=cfg.n_bins, projection=projection
        )

        mean_reward = float(
            torch.stack([rollout.rewards[i].sum() for i in range(cfg.num_agents)])
            .sum()
            .item()
            / cfg.rollout_steps
        )

        record = {
            "iteration": it,
            "mean_step_reward_team": mean_reward,
            **stats,
            **info_stats,
        }
        metrics_log.append(record)

        if it % max(1, cfg.num_iterations // 10) == 0 or it == cfg.num_iterations - 1:
            print(
                f"  iter {it:4d} | team_reward {mean_reward:8.3f} | "
                f"mi_mean {info_stats['mi/mean_pair']:.3f} | "
                f"cmi_mean {info_stats['cmi/mean_pair']:.3f} | "
                f"red_loss {stats['redundancy_loss']:.4f}"
            )

    # Save final policies.
    pol_dir = output_dir / "policies"
    pol_dir.mkdir(exist_ok=True)
    for i, policy in enumerate(trainer.policies):
        torch.save(policy.state_dict(), pol_dir / f"agent_{i}.pt")

    # Save metrics + config.
    with (output_dir / "metrics.json").open("w") as f:
        json.dump(metrics_log, f, indent=2)
    with (output_dir / "config.json").open("w") as f:
        json.dump(asdict(cfg), f, indent=2)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", required=True)
    p.add_argument("--lambda-red", type=float, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--num-iterations", type=int, default=50)
    p.add_argument("--rollout-steps", type=int, default=2048)
    p.add_argument("--num-agents", type=int, default=4)
    p.add_argument(
        "--value-coef",
        type=float,
        default=CellConfig.__dataclass_fields__["value_coef"].default,
        help=(
            "PPO value-loss weight (default matches JointPPOTrainer.__init__). "
            "Lowered in Phase 2 sweeps to test value-loss-dominance hypothesis "
            "(issue #153)."
        ),
    )
    p.add_argument(
        "--entropy-coef",
        type=float,
        default=CellConfig.__dataclass_fields__["entropy_coef"].default,
        help=(
            "PPO entropy bonus weight (default matches JointPPOTrainer.__init__). "
            "Raised in Phase 2 sweeps to prevent entropy collapse (issue #153)."
        ),
    )
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    cfg = CellConfig(
        scenario=args.scenario,
        lambda_red=args.lambda_red,
        seed=args.seed,
        num_iterations=args.num_iterations,
        rollout_steps=args.rollout_steps,
        num_agents=args.num_agents,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        device=args.device,
    )
    print(
        f"== P3 cell: scenario={cfg.scenario} lambda_red={cfg.lambda_red} "
        f"seed={cfg.seed} =="
    )
    train_one_cell(cfg, args.output_dir)


if __name__ == "__main__":
    main()
