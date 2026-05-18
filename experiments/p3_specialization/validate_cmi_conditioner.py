"""Phase 1 BC-init smoke test for the state-summary CMI conditioner.

Issue #176 (Phase 1).

Loads a BC-init checkpoint (default: the issue #270 specialist from PR #278),
rolls out a fixed-seed episode on ``minimal_specialization`` with the loaded
policy in every agent slot, and reports the unconditioned mutual information
:math:`I(\\hat{Z}_i; \\hat{Z}_j)` and the state-summary conditional mutual
information :math:`I(\\hat{Z}_i; \\hat{Z}_j | S)` between agent encoder outputs.
The verdict ladder maps the CMI/MI ratio to one of ``informative``, ``weak``,
or ``vacuous`` --- the latter means the falsifier in P3 v2 cannot be
interpreted on this policy because conditioning on the coarse state-summary
does not reduce the encoder-to-encoder dependence.

This is intentionally **trainer-agnostic**: it consumes per-agent state-dict
files (``agent_{i}.pt``) via
:class:`bucket_brigade.agents.trained_policy_archetype.TrainedPolicyArchetype`
rather than reconstructing a :class:`JointPPOTrainer`. The same pathway will
be reused in Phase 2 for non-BC trainers (LOLA, COMA, HCA, social-influence,
etc.) once additional above-threshold checkpoints land.

**Refusal of legacy 2-head checkpoints**: BC-init checkpoints from PR #278
postdate PR #236 (signal is a first-class action dim), so they are 3-head
``[10, 2, 2]``. We load *without* ``allow_legacy_2head=True``; loading fails
loud if the checkpoint is 2-head, which would indicate a pre-#236 artifact
where ``signal := mode`` is silently fabricated. See issue #325.

Run::

    uv run python experiments/p3_specialization/validate_cmi_conditioner.py \\
        --checkpoint-dir experiments/p3_specialization/runs/issue270_bc_init/specialist_bc_v2

The default ``--checkpoint-dir`` is the canonical PR #278 path. If it does
not exist locally, the script fails loud with the absolute path expected.
The BC-init checkpoint typically lives on ``COMPUTE_HOST_PRIMARY`` --- copy
or rsync it down before running, or pass ``--checkpoint-dir`` to a local
override (including a synthetic fixture produced by the test suite).

Outputs::

    <output-dir>/results.json
    <output-dir>/verdict.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from bucket_brigade.agents.trained_policy_archetype import TrainedPolicyArchetype
from bucket_brigade.analysis.info_theory import (
    conditional_mutual_information,
    is_degenerate_conditioner,
    mutual_information,
    quantize_uniform,
)
from bucket_brigade.envs.bucket_brigade_env import BucketBrigadeEnv
from bucket_brigade.envs.scenarios_generated import get_scenario_by_name
from bucket_brigade.training.joint_trainer import flatten_dict_obs


# Canonical BC-init checkpoint path produced by PR #278 (issue #270).
# Mac Studio (``COMPUTE_HOST_PRIMARY``) is the host of record; the directory
# is gitignored, so a fresh clone has to rsync it from the remote.
DEFAULT_CHECKPOINT_DIR = (
    "experiments/p3_specialization/runs/issue270_bc_init/specialist_bc_v2"
)
DEFAULT_SCENARIO = "minimal_specialization"
DEFAULT_NUM_AGENTS = 4
DEFAULT_NUM_HOUSES = 10
DEFAULT_ROLLOUT_STEPS = 2048
DEFAULT_SEED = 42

# Match :mod:`train.py` so the validation result is comparable to the
# per-cell ``cmi/mean_pair`` numbers produced by the training harness.
DEFAULT_N_BINS = 4
DEFAULT_MI_PROJ_DIMS = 3
DEFAULT_STATE_SUMMARY_DAY_BINS = 4

# Verdict thresholds. See ``verdict_cmi_informative`` for the rationale.
INFORMATIVE_RATIO = 0.5
WEAK_RATIO = 0.9


@dataclass
class ValidationResult:
    """Per-checkpoint smoke test result.

    Mirrors the JSON schema written to ``results.json``.
    """

    checkpoint_dir: str
    checkpoint_hash: str
    scenario: str
    num_agents: int
    rollout_steps: int
    seed: int
    n_bins: int
    mi_proj_dims: int
    hidden_size: int
    obs_dim: int
    action_dims: List[int]

    mi_mean_pair: float
    cmi_mean_pair: float
    cmi_over_mi_ratio: float
    is_degenerate_conditioner: bool
    conditioner_n_distinct: int
    conditioner_modal_fraction: float
    conditioner_entropy_bits: float
    verdict: str

    mi_per_pair: Dict[str, float] = field(default_factory=dict)
    cmi_per_pair: Dict[str, float] = field(default_factory=dict)


def verdict_cmi_informative(mi: float, cmi: float) -> str:
    """Map the (MI, CMI) pair onto the 3-rung verdict ladder.

    The F1 falsifier in P3 v2 measures a monotone decrease in
    :math:`I(\\hat{Z}_i; \\hat{Z}_j | S)` as the redundancy penalty
    coefficient grows. For that test to *discriminate* trained behavior
    from random, the conditioner must explain at least *some* of the
    encoder-to-encoder dependence on a learning policy --- otherwise CMI
    tracks MI 1-for-1 and the falsifier is uninterpretable.

    Three rungs, all conditioned on MI > 0:

    - ``"informative"`` --- CMI < 0.5 * MI. The conditioner shaves at least
      half of the mutual information. F1 has discriminating power.
    - ``"weak"`` --- 0.5 * MI <= CMI < 0.9 * MI. Conditioning has *some*
      effect, but more than half of the dependence survives. F1 verdicts
      are still nominal but harder to interpret.
    - ``"vacuous"`` --- CMI >= 0.9 * MI. The conditioner barely moves the
      value; F1 cannot rule on this policy. This is the failure mode
      issue #146/#149 originally diagnosed for the team-reward conditioner.

    Edge case: when ``mi == 0`` (no encoder dependence at all), the
    ratio is undefined; we return ``"vacuous"`` because there is no signal
    for the conditioner to remove --- the falsifier has no quantity to
    track.
    """
    if mi <= 0.0:
        return "vacuous"
    ratio = cmi / mi
    if ratio < INFORMATIVE_RATIO:
        return "informative"
    if ratio < WEAK_RATIO:
        return "weak"
    return "vacuous"


def _state_summary_codes(
    burning_per_step: np.ndarray,
    dones_per_step: np.ndarray,
    day_bins: int = DEFAULT_STATE_SUMMARY_DAY_BINS,
) -> np.ndarray:
    """Reproduce the coarse ``(num_houses_burning, day_index)`` conditioner.

    This is a re-implementation of
    :func:`experiments.p3_specialization.train._state_summary_codes` that
    operates on numpy arrays we already have in hand from our local rollout
    (we do not have a :class:`RolloutBuffer`-shaped object here). The packing
    convention matches ``train.py:307``::

        code = num_houses_burning + 11 * day_bin

    so the alphabet is ``[0, 11 * day_bins)``. Keeping the same packing
    makes the smoke-test CMI numbers directly comparable to the per-iter
    ``cmi/mean_pair`` rows in any ``metrics.json`` produced by the training
    harness.
    """
    T = burning_per_step.shape[0]
    day_index = np.zeros(T, dtype=np.int64)
    counter = 0
    for t in range(T):
        day_index[t] = counter
        counter = 0 if bool(dones_per_step[t]) else counter + 1
    day_codes = quantize_uniform(day_index.astype(np.float64), n_bins=day_bins)
    return burning_per_step.astype(np.int64) + 11 * day_codes.astype(np.int64)


def _hash_checkpoint_dir(checkpoint_dir: Path, num_agents: int) -> str:
    """Stable 12-char hash of all per-agent state-dicts in the directory."""
    h = hashlib.sha256()
    for i in range(num_agents):
        p = checkpoint_dir / f"agent_{i}.pt"
        if not p.exists():
            # Hash whatever exists; the caller validates existence separately.
            continue
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


def _load_archetypes(
    checkpoint_dir: Path,
    num_agents: int,
) -> List[TrainedPolicyArchetype]:
    """Load one :class:`TrainedPolicyArchetype` per agent slot.

    The loader refuses 2-head checkpoints unconditionally
    (``allow_legacy_2head=False``). Per the issue brief: BC-init from
    PR #270/#278 postdates PR #236, so it is 3-head and must load
    cleanly without the legacy opt-in.
    """
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(
            f"Checkpoint directory not found: {checkpoint_dir.resolve()}\n"
            f"\n"
            f"The canonical BC-init checkpoint (PR #278) lives on the remote "
            f"compute host (typically `COMPUTE_HOST_PRIMARY`, the Mac Studio).\n"
            f"\n"
            f"To run this validation locally, either:\n"
            f"  1. rsync the directory down:\n"
            f"     rsync -avz <HOST>:~/GitHub/bucket-brigade/{DEFAULT_CHECKPOINT_DIR}/ "
            f"{DEFAULT_CHECKPOINT_DIR}/\n"
            f"  2. or pass --checkpoint-dir <path> pointing at a local override.\n"
        )

    archetypes: List[TrainedPolicyArchetype] = []
    for i in range(num_agents):
        path = checkpoint_dir / f"agent_{i}.pt"
        if not path.exists():
            raise FileNotFoundError(
                f"Per-agent checkpoint missing: {path.resolve()}\n"
                f"BC-init writes one `agent_{{i}}.pt` per agent slot "
                f"(see experiments/p3_specialization/bc_init.py)."
            )
        # NOTE: ``allow_legacy_2head=False`` is the issue #176 invariant.
        # A pre-#236 checkpoint here would fabricate a signal channel,
        # so we fail loud rather than silently produce a 2-head verdict.
        archetypes.append(
            TrainedPolicyArchetype(
                state_dict_path=path,
                agent_id=i,
                num_agents=num_agents,
                deterministic=True,
                allow_legacy_2head=False,
            )
        )

    # The first archetype's action-head count tells us whether we got a
    # 3-head modern checkpoint. (TrainedPolicyArchetype's __init__ would
    # have already raised for 2-head, but re-check for defense in depth
    # in case the contract drifts later.)
    if len(archetypes[0].action_dims) != 3:
        raise ValueError(
            f"Loaded checkpoint has {len(archetypes[0].action_dims)} action "
            f"heads; issue #176 requires the post-#236 3-head shape "
            f"[10, 2, 2]. Got action_dims={archetypes[0].action_dims}."
        )
    return archetypes


def _rollout_with_archetypes(
    archetypes: List[TrainedPolicyArchetype],
    scenario_name: str,
    num_agents: int,
    rollout_steps: int,
    seed: int,
    num_houses: int = DEFAULT_NUM_HOUSES,
) -> Dict[str, np.ndarray]:
    """Drive a fixed-seed rollout using the archetypes.

    Returns a dict with::

        observations: [T, N, obs_dim]   --- per-agent flattened obs
        dones:        [T]                --- episode boundary flag
        burning:      [T]                --- num_houses_burning at each step
    """
    scenario = get_scenario_by_name(scenario_name, num_agents=num_agents)
    env = BucketBrigadeEnv(scenario=scenario)

    obs_dim = archetypes[0].obs_dim

    observations = np.zeros((rollout_steps, num_agents, obs_dim), dtype=np.float32)
    dones = np.zeros(rollout_steps, dtype=np.float32)
    burning = np.zeros(rollout_steps, dtype=np.int64)

    rng = np.random.default_rng(seed)
    obs = env.reset(seed=int(rng.integers(0, 2**31 - 1)))

    for t in range(rollout_steps):
        # Snapshot per-agent flattened obs *before* stepping.
        for i in range(num_agents):
            observations[t, i, :] = flatten_dict_obs(
                obs, agent_id=i, num_agents=num_agents
            )
        # Houses-state slice is the first ``num_houses`` columns; BURNING=1.
        burning[t] = int((obs["houses"][:num_houses] == 1).sum())

        joint = np.stack(
            [archetypes[i].act(obs) for i in range(num_agents)], axis=0
        ).astype(np.int8)
        obs, _, _, _ = env.step(joint)
        dones[t] = 1.0 if env.done else 0.0
        if env.done:
            obs = env.reset(seed=int(rng.integers(0, 2**31 - 1)))

    return {
        "observations": observations,
        "dones": dones,
        "burning": burning,
    }


def _compute_mi_cmi(
    observations: np.ndarray,
    archetypes: List[TrainedPolicyArchetype],
    state_codes: np.ndarray,
    n_bins: int,
    mi_proj_dims: int,
    seed: int,
) -> Dict[str, object]:
    """Plug-in MI / CMI on the rollout's encoder outputs.

    Mirrors the relevant slice of
    :func:`experiments.p3_specialization.train._measure_information`.
    """
    num_agents = observations.shape[1]
    hidden_size = archetypes[0].hidden_size

    # Random projection seeded for reproducibility. Same recipe as train.py.
    rng = np.random.default_rng(seed)
    projection = rng.standard_normal((hidden_size, mi_proj_dims)).astype(np.float32)
    projection /= np.linalg.norm(projection, axis=0, keepdims=True) + 1e-8

    codes_per_agent: List[np.ndarray] = []
    for i in range(num_agents):
        x = torch.from_numpy(observations[:, i, :])
        with torch.no_grad():
            feats = archetypes[i].policy.encoder_output(x).cpu().numpy()
        proj = feats @ projection
        codes_per_agent.append(quantize_uniform(proj, n_bins=n_bins))

    mi_per_pair: Dict[str, float] = {}
    cmi_per_pair: Dict[str, float] = {}
    mi_vals: List[float] = []
    cmi_vals: List[float] = []
    for i in range(num_agents):
        for j in range(i + 1, num_agents):
            mi = mutual_information(codes_per_agent[i], codes_per_agent[j])
            cmi = conditional_mutual_information(
                codes_per_agent[i], codes_per_agent[j], state_codes
            )
            key = f"agent_{i}_{j}"
            mi_per_pair[key] = float(mi)
            cmi_per_pair[key] = float(cmi)
            mi_vals.append(float(mi))
            cmi_vals.append(float(cmi))

    return {
        "mi_per_pair": mi_per_pair,
        "cmi_per_pair": cmi_per_pair,
        "mi_mean_pair": float(np.mean(mi_vals)),
        "cmi_mean_pair": float(np.mean(cmi_vals)),
        "hidden_size": int(hidden_size),
    }


def validate(
    checkpoint_dir: Path,
    output_dir: Optional[Path] = None,
    scenario: str = DEFAULT_SCENARIO,
    num_agents: int = DEFAULT_NUM_AGENTS,
    rollout_steps: int = DEFAULT_ROLLOUT_STEPS,
    seed: int = DEFAULT_SEED,
    n_bins: int = DEFAULT_N_BINS,
    mi_proj_dims: int = DEFAULT_MI_PROJ_DIMS,
    state_summary_day_bins: int = DEFAULT_STATE_SUMMARY_DAY_BINS,
) -> ValidationResult:
    """Top-level entry point. Loads, rolls out, measures, writes results."""
    archetypes = _load_archetypes(checkpoint_dir, num_agents)
    obs_dim = archetypes[0].obs_dim
    hidden_size = archetypes[0].hidden_size
    action_dims = list(archetypes[0].action_dims)

    rollout = _rollout_with_archetypes(
        archetypes,
        scenario_name=scenario,
        num_agents=num_agents,
        rollout_steps=rollout_steps,
        seed=seed,
    )

    state_codes = _state_summary_codes(
        rollout["burning"], rollout["dones"], day_bins=state_summary_day_bins
    )
    is_deg, diag = is_degenerate_conditioner(state_codes)

    measurement = _compute_mi_cmi(
        rollout["observations"],
        archetypes,
        state_codes=state_codes,
        n_bins=n_bins,
        mi_proj_dims=mi_proj_dims,
        seed=seed,
    )

    mi_mean = measurement["mi_mean_pair"]
    cmi_mean = measurement["cmi_mean_pair"]
    ratio = (cmi_mean / mi_mean) if mi_mean > 0.0 else float("nan")
    verdict = verdict_cmi_informative(mi_mean, cmi_mean)

    result = ValidationResult(
        checkpoint_dir=str(checkpoint_dir.resolve()),
        checkpoint_hash=_hash_checkpoint_dir(checkpoint_dir, num_agents),
        scenario=scenario,
        num_agents=num_agents,
        rollout_steps=rollout_steps,
        seed=seed,
        n_bins=n_bins,
        mi_proj_dims=mi_proj_dims,
        hidden_size=hidden_size,
        obs_dim=obs_dim,
        action_dims=action_dims,
        mi_mean_pair=mi_mean,
        cmi_mean_pair=cmi_mean,
        cmi_over_mi_ratio=ratio,
        is_degenerate_conditioner=bool(is_deg),
        conditioner_n_distinct=int(diag["n_distinct"]),
        conditioner_modal_fraction=float(diag["modal_fraction"]),
        conditioner_entropy_bits=float(diag["entropy_bits"]),
        verdict=verdict,
        mi_per_pair=measurement["mi_per_pair"],
        cmi_per_pair=measurement["cmi_per_pair"],
    )

    if output_dir is None:
        output_dir = (
            Path("experiments/p3_specialization/cmi_validation")
            / result.checkpoint_hash
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "results.json"
    with results_path.open("w") as f:
        json.dump(asdict(result), f, indent=2)

    verdict_path = output_dir / "verdict.md"
    with verdict_path.open("w") as f:
        f.write(_format_verdict_md(result))

    return result


def _format_verdict_md(result: ValidationResult) -> str:
    """Human-readable verdict summary in Markdown."""
    next_steps_map = {
        "informative": (
            "The state-summary conditioner discriminates encoder-pair "
            "dependence on this trained policy: the F1 falsifier in P3 v2 "
            "has interpretable signal on at least one above-threshold "
            "trainer. Promote Phase 2 (cross-trainer power: re-run against "
            "LOLA, COMA, HCA, social-influence checkpoints once their "
            "sweeps land above `gap_closed > 0.25`)."
        ),
        "weak": (
            "Conditioning shaves some of the encoder-pair dependence, but "
            "more than half survives. F1 verdicts are still nominal but "
            "the signal is weak. Consider Phase 2 sweep across more "
            "trainers to estimate variance, and re-curate issue #146's "
            "alternative conditioners (per-agent reward, marginal-action "
            "codes, smaller-cardinality state coarsening) as fallbacks."
        ),
        "vacuous": (
            "CMI tracks MI to within 10% --- the state-summary conditioner "
            "is effectively vacuous on this policy. F1 cannot rule on the "
            "P3 v2 falsifier with this measurement setup. File a follow-up "
            "proposing an alternative conditioner from #146."
        ),
    }
    next_steps = next_steps_map[result.verdict]

    per_pair_table = "\n".join(
        f"| `{k}` | {result.mi_per_pair[k]:.4f} | {result.cmi_per_pair[k]:.4f} | "
        f"{(result.cmi_per_pair[k] / result.mi_per_pair[k] if result.mi_per_pair[k] > 0 else float('nan')):.3f} |"
        for k in sorted(result.mi_per_pair.keys())
    )

    return (
        f"# CMI Conditioner Validation --- Phase 1 (issue #176)\n\n"
        f"**Checkpoint**: `{result.checkpoint_dir}`  \n"
        f"**Hash**: `{result.checkpoint_hash}`  \n"
        f"**Scenario**: `{result.scenario}` "
        f"(N={result.num_agents}, T={result.rollout_steps}, "
        f"seed={result.seed})  \n"
        f"**Architecture**: hidden_size={result.hidden_size}, "
        f"obs_dim={result.obs_dim}, action_dims={result.action_dims}  \n\n"
        f"## Summary\n\n"
        f"| Quantity | Value |\n"
        f"|---|---|\n"
        f"| MI mean-pair (bits) | **{result.mi_mean_pair:.4f}** |\n"
        f"| CMI mean-pair (bits) | **{result.cmi_mean_pair:.4f}** |\n"
        f"| CMI / MI ratio | **{result.cmi_over_mi_ratio:.4f}** |\n"
        f"| Verdict | **`{result.verdict}`** |\n"
        f"| `is_degenerate_conditioner` | {result.is_degenerate_conditioner} |\n"
        f"| Conditioner n_distinct | {result.conditioner_n_distinct} |\n"
        f"| Conditioner modal_fraction | {result.conditioner_modal_fraction:.4f} |\n"
        f"| Conditioner entropy (bits) | {result.conditioner_entropy_bits:.4f} |\n\n"
        f"## Per-pair detail\n\n"
        f"| Pair | MI (bits) | CMI (bits) | CMI/MI |\n"
        f"|---|---|---|---|\n"
        f"{per_pair_table}\n\n"
        f"## Verdict ladder\n\n"
        f"- `informative` --- CMI < {INFORMATIVE_RATIO:.1f} * MI "
        f"(conditioner explains >50% of dependence; F1 has discriminating power)\n"
        f"- `weak` --- {INFORMATIVE_RATIO:.1f} * MI <= CMI < {WEAK_RATIO:.1f} * MI "
        f"(partial info but not great)\n"
        f"- `vacuous` --- CMI >= {WEAK_RATIO:.1f} * MI "
        f"(conditioner does not change the value; F1 uninterpretable)\n\n"
        f"## Next steps\n\n"
        f"{next_steps}\n\n"
        f"## Notes\n\n"
        f"- This is Phase 1 of issue #176. Phase 2 (cross-trainer power "
        f"validation across LOLA, COMA, HCA, social-influence, etc.) is "
        f"deferred to a follow-up issue once additional trainer sweeps "
        f"land above the `gap_closed > 0.25` trigger.\n"
        f"- The trainer-agnostic loading pathway (via "
        f"`TrainedPolicyArchetype`) used here generalizes directly to the "
        f"Phase 2 trainers; no further loader plumbing is needed.\n"
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 1 BC-init smoke test for the state-summary CMI "
            "conditioner (issue #176)."
        ),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path(DEFAULT_CHECKPOINT_DIR),
        help=(
            "Directory containing per-agent `agent_{i}.pt` state dicts "
            f"(default: {DEFAULT_CHECKPOINT_DIR}). The default points at "
            "the PR #278 BC-init specialist; rsync from "
            "`COMPUTE_HOST_PRIMARY` if not present locally."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for results.json and verdict.md. Defaults to "
            "experiments/p3_specialization/cmi_validation/<checkpoint_hash>/."
        ),
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=DEFAULT_SCENARIO,
    )
    parser.add_argument(
        "--num-agents",
        type=int,
        default=DEFAULT_NUM_AGENTS,
    )
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=DEFAULT_ROLLOUT_STEPS,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n-bins", type=int, default=DEFAULT_N_BINS)
    parser.add_argument("--mi-proj-dims", type=int, default=DEFAULT_MI_PROJ_DIMS)
    args = parser.parse_args(argv)

    try:
        result = validate(
            checkpoint_dir=args.checkpoint_dir,
            output_dir=args.output_dir,
            scenario=args.scenario,
            num_agents=args.num_agents,
            rollout_steps=args.rollout_steps,
            seed=args.seed,
            n_bins=args.n_bins,
            mi_proj_dims=args.mi_proj_dims,
        )
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(
        f"\nVerdict: {result.verdict.upper()}  "
        f"(MI={result.mi_mean_pair:.4f} bits, "
        f"CMI={result.cmi_mean_pair:.4f} bits, "
        f"CMI/MI={result.cmi_over_mi_ratio:.4f})"
    )
    print(f"Wrote: {args.output_dir or '(default)'} ...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
