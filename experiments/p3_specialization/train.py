"""Single-cell trainer for the P3 specialization experiment.

One invocation of :func:`train_one_cell` runs one ``(scenario, lambda_red, seed)``
training cell to completion: it spins up a :class:`JointPPOTrainer`, runs
``num_iterations`` rollout-and-update cycles, and writes the artifacts a
later analysis pass needs:

- ``policies/agent_{i}.pt`` --- final state dict for each agent.
- ``metrics.json`` --- per-iteration scalars (loss, reward, MI, etc.).
- ``config.json`` --- the exact arguments used (for reproducibility).

Plug-in conditional MI between encoder outputs is computed on each
rollout's most recent batch, conditioned on a coarse state summary
``(num_houses_burning, day_index)`` (Option 1 from issue #154). Unconditional
MI is also logged so we can see how much "specialization beyond shared
state" the encoders actually carry.
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
    # Issue #159: optional return normalization. Default False preserves
    # existing behavior; flip on for the ablation cells.
    normalize_returns: bool = False
    # Encoder outputs are quantized for plug-in MI. We first project from
    # ``hidden_size`` down to ``mi_proj_dims`` via a fixed random matrix
    # (seeded from ``seed``); then ``quantize_uniform`` packs each row into
    # a single integer code with up to ``n_bins ** mi_proj_dims`` values.
    n_bins: int = 4
    mi_proj_dims: int = 3
    device: str = "cpu"
    action_dims: List[int] = field(default_factory=lambda: [10, 2])


# Default number of day bins for the state-summary conditioner. Episodes in
# ``BucketBrigadeEnv`` are short (single-digit nights on most scenarios), so a
# small bin count is appropriate: too many bins fragments the conditioner across
# rare day-indices and silently shrinks the per-cell sample.
_STATE_SUMMARY_DAY_BINS = 4

# Per-house observation layout: the first 10 columns of ``rollout.observations``
# encode the 10-house state vector (SAFE=0, BURNING=1, RUINED=2). See
# ``BucketBrigadeEnv._get_observation`` and ``flatten_dict_obs`` for the layout.
_HOUSES_OBS_SLICE = slice(0, 10)
_BURNING_CODE = 1


def _state_summary_codes(
    rollout, day_bins: int = _STATE_SUMMARY_DAY_BINS
) -> np.ndarray:
    """Coarse ``(num_houses_burning, day_index)`` state-summary conditioner.

    Implements **Option 1** from issue #154 — replaces the previously-degenerate
    team-reward conditioner with a coarse summary of the shared environment
    state. Both components are reducible from rollout tensors without plumbing
    changes to the env or trainer:

    - ``num_houses_burning`` per timestep: count of ``BURNING`` entries in the
      per-house state vector at ``rollout.observations[:, 0:10]``. Range
      ``[0, 10]`` → 11-valued alphabet.
    - ``day_index`` per timestep: 0-based step counter within the current
      episode, reconstructed by resetting a running counter on each
      ``rollout.dones[t] == 1`` flag (the env auto-resets after a done, so the
      *next* observation begins a new episode). Quantized to ``day_bins`` equal
      bins over the empirical range of day indices in this rollout.

    Codes are packed into a single integer per timestep as
    ``burning_count + 11 * day_bin`` (range ``[0, 11 * day_bins)``) so the
    output matches the 1-D hashable-code shape consumed by
    :func:`conditional_mutual_information`.

    NOTE: Conditioner selected by Builder (issue #154 Option 1: state summary).
    The choice between state-summary vs. trajectory-bucket vs. other-agent-action
    is a research call; curator flagged this for Architect review (see #154
    builder-note for the Option 2/3 tradeoffs). If measurement values look
    wrong, revisit Options 2/3 in #154.
    """
    obs_np = rollout.observations.cpu().numpy()
    # Per-house state ∈ {SAFE=0, BURNING=1, RUINED=2}; count BURNING per step.
    num_burning = (obs_np[:, _HOUSES_OBS_SLICE] == _BURNING_CODE).sum(axis=1)
    num_burning = num_burning.astype(np.int64)  # range [0, 10]

    # Reconstruct per-step day_index from the shared dones flag. ``dones[t] == 1``
    # signals "episode ended at step t"; the env auto-resets so step t+1 belongs
    # to a fresh episode. We restart the counter at the step *after* each done.
    dones_np = rollout.dones.cpu().numpy().astype(bool)
    T = len(dones_np)
    day_index = np.zeros(T, dtype=np.int64)
    counter = 0
    for t in range(T):
        day_index[t] = counter
        # If this step terminates the episode, the next step is day 0.
        counter = 0 if dones_np[t] else counter + 1

    # Quantize day_index uniformly. ``quantize_uniform`` handles the constant
    # case (all-same input → all zeros) defensively.
    day_codes = quantize_uniform(day_index.astype(np.float64), n_bins=day_bins)

    # Pack into a single integer code per timestep. The +11 base matches the
    # full alphabet of ``num_burning`` (0..10 inclusive); using the actual max
    # would risk index collisions on rare empty rollouts.
    return num_burning + 11 * day_codes.astype(np.int64)


def _measure_information(
    trainer: JointPPOTrainer,
    rollout,
    n_bins: int,
    projection: np.ndarray,
) -> Dict[str, float]:
    """Plug-in MI/CMI on the rollout's encoder outputs.

    Conditioned on a coarse ``(num_houses_burning, day_index)`` state summary
    (issue #154 Option 1). Also logs unconditional MI per pair.

    Encoder outputs are projected from ``hidden_size`` to a small dimension
    via the shared ``projection`` matrix before quantization, because a
    direct quantize-and-pack of a 64-D vector overflows the integer code.

    NOTE: Conditioner selected by Builder (issue #154 Option 1: state summary).
    The choice between state-summary vs. trajectory-bucket vs. other-agent-action
    is a research call; curator flagged this for Architect review. If
    measurement values look wrong, revisit Options 2/3 in #154.
    """
    with torch.no_grad():
        feats = trainer.encoder_outputs_batch(rollout.observations)
    feats_np = [f.cpu().numpy() @ projection for f in feats]

    # Quantize each (T, mi_proj_dims) projection into a single integer code.
    codes = [quantize_uniform(f, n_bins=n_bins) for f in feats_np]

    # State-summary conditioning variable (issue #154 Option 1). See
    # ``_state_summary_codes`` for the Architect-deferred decision rationale.
    z_codes = _state_summary_codes(rollout)

    out: Dict[str, float] = {}

    # Defensive measurement-quality check (see issue #146).
    #
    # The team-reward conditioner used previously was near-constant on several
    # scenarios (``trivial_cooperation`` literally; ``default`` and
    # ``chain_reaction`` near-deterministically), which made the plug-in CMI
    # collapse to the unconditional MI. Issue #154 Option 1 replaces it with a
    # coarse ``(num_houses_burning, day_index)`` summary; this check stays in
    # place to detect future regressions (e.g., if a scenario ends on step 0
    # every time, both components collapse).
    is_degenerate, diag = is_degenerate_conditioner(z_codes)
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
                "I(Ẑ_i; Ẑ_j | Z) ≈ I(Ẑ_i; Ẑ_j) is mathematically guaranteed; "
                "see issue #154 for context. Reported CMI values for this "
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
            cmi = conditional_mutual_information(codes[i], codes[j], z_codes)
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
        normalize_returns=cfg.normalize_returns,
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
    p.add_argument(
        "--normalize-returns",
        action="store_true",
        help=(
            "Issue #159: normalize PPO returns by running std before the "
            "value-loss MSE. Default off preserves existing behavior; flip on "
            "for the 4-cell ablation."
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
        normalize_returns=args.normalize_returns,
        device=args.device,
    )
    print(
        f"== P3 cell: scenario={cfg.scenario} lambda_red={cfg.lambda_red} "
        f"seed={cfg.seed} =="
    )
    train_one_cell(cfg, args.output_dir)


if __name__ == "__main__":
    main()
