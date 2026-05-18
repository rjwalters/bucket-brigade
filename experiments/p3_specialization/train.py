"""Single-cell trainer for the P3 specialization experiment.

One invocation of :func:`train_one_cell` runs one ``(scenario, lambda_red, seed)``
training cell to completion: it spins up a :class:`JointPPOTrainer`, runs
``num_iterations`` rollout-and-update cycles, and writes the artifacts a
later analysis pass needs:

- ``policies/agent_{i}.pt`` --- final state dict for each agent.
- ``metrics.json`` --- per-iteration scalars (loss, reward, MI, etc.).
- ``config.json`` --- the exact arguments used (for reproducibility).

Plug-in conditional MI between encoder outputs is computed on each
rollout's most recent batch, with two conditioners reported side-by-side:

- **Primary (Option 1, state summary):** coarse ``(num_houses_burning,
  day_index)``. Exogenous environment state; clean removal of "redundancy
  from looking at the same world."
- **Sensitivity check (Option 3, other-agent lagged action):** per pair
  ``(i, j)``, agent ``j``'s packed action at ``t-1``. Conditions on a
  signal that is downstream of agent ``j``'s encoder (via its policy
  head), so it can suppress CMI for the wrong reason — useful as a
  cross-check, not as a primary measurement.

See ``research_notebook/2026-05-15_p3_conditioner_decision.md`` for the
architect rationale and the four-way agreement table used to interpret
the two CMI series together. Unconditional MI is also logged.
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

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
from bucket_brigade.envs.macro_action_env import MacroActionEnv
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
    # Issue #208: MAPPO / centralized-critic flag. Default False preserves
    # the independent-PPO (IPPO) baseline; flip on for the MAPPO arm of
    # the Phase 2 sweep. Incompatible with lambda_red > 0 (the redundancy
    # penalty couples the per-agent actor trunks).
    centralized_critic: bool = False
    # Issue #284: COMA / counterfactual-advantage flag. Default False
    # preserves the IPPO baseline. Mutually exclusive with both
    # centralized_critic and lambda_red > 0. Installs a CentralizedQCritic
    # whose enumerated per-agent Q-values feed the COMA counterfactual
    # advantage instead of GAE-on-V. See JointPPOTrainer docstring.
    coma: bool = False
    coma_target_update_every: int = 200
    # Encoder outputs are quantized for plug-in MI. We first project from
    # ``hidden_size`` down to ``mi_proj_dims`` via a fixed random matrix
    # (seeded from ``seed``); then ``quantize_uniform`` packs each row into
    # a single integer code with up to ``n_bins ** mi_proj_dims`` values.
    n_bins: int = 4
    mi_proj_dims: int = 3
    device: str = "cpu"
    # Issue #235: action is now [house, mode, signal] (10*2*2 = 40 values).
    # Pre-#235 was [10, 2]; the third entry (signal alphabet) makes the
    # broadcast channel a learned action dimension.
    action_dims: List[int] = field(default_factory=lambda: [10, 2, 2])
    # Issue #260: optional episode-length curriculum. Each entry is
    # ``[start_iteration, min_nights]``. Empty list disables the
    # curriculum (default) and preserves bit-identical behavior with
    # pre-#260 runs. Phases are applied in iteration order; the floor
    # in effect at iteration ``i`` is the latest phase whose
    # ``start_iteration <= i``. The env exposes ``min_nights`` (the
    # *minimum* forced episode length), not ``max_nights``; shorter
    # floors increase termination density per env-step but episodes
    # can still run longer when fires remain active. See
    # ``bucket_brigade/envs/bucket_brigade_env.py::_check_termination``.
    curriculum: List[List[int]] = field(default_factory=list)
    # Issue #261/#262: action-conditioned reward shaping knobs that are
    # mutated onto the loaded ``Scenario`` instance in ``train_one_cell``.
    # Defaults match ``Scenario.action_shaping_alpha`` / ``_beta`` (both
    # 0.0), which keeps the env fast-path skip and bit-identical legacy
    # behavior. The calibration sweep driver passes per-cell values via
    # the ``--action-shaping-alpha`` / ``--action-shaping-beta`` flags.
    action_shaping_alpha: float = 0.0
    action_shaping_beta: float = 0.0
    # Issue #265: dense progress shaping coefficient. Mutated onto the loaded
    # ``Scenario`` instance in ``train_one_cell`` so the per-cell knob takes
    # effect inside ``_compute_rewards``. Default ``0.0`` matches the
    # ``Scenario.progress_shaping_coef`` default, keeps the env fast-path
    # skip, and preserves bit-identical pre-#265 behavior. The #265 sweep
    # driver passes per-cell values via ``--progress-shaping-coef``.
    progress_shaping_coef: float = 0.0
    # Issue #283: potential-based team-welfare shaping knobs (Ng-Harada-Russell
    # 1999). Mutated onto the loaded ``Scenario`` instance in ``train_one_cell``
    # next to the action-shaping mutation. Defaults match
    # ``Scenario.team_welfare_lambda`` / ``_gamma`` / ``_kind``:
    # lambda=0.0 and kind="none" keep the env fast-path skip and
    # bit-identical pre-#283 behavior.
    team_welfare_lambda: float = 0.0
    team_welfare_gamma: float = 1.0
    team_welfare_kind: str = "none"
    # Issue #270: optional BC-pretrained init. Directory containing per-agent
    # ``agent_{i}.pt`` state dicts produced by
    # ``experiments/p3_specialization/bc_init.py``. ``None`` (default)
    # preserves bit-identical pre-#270 random-init behavior. When set, the
    # state dicts are loaded into ``trainer.policies[i]`` immediately after
    # trainer construction, before the first PPO iteration.
    bc_init_checkpoint_dir: Optional[str] = None
    # Issue #282: GAE lambda for advantage estimation. Default 0.95 matches
    # ``JointPPOTrainer.__init__`` and preserves bit-identical pre-#282
    # behavior. Raised toward 1.0 in the issue #282 sweep to test whether
    # bootstrap bias is the source of PPO's plateau on
    # ``minimal_specialization`` (thesis: GAE bootstrap bias as the
    # misalignment source — see
    # ``research_notebook/2026-05-17_thesis_misaligned_gradients.md``).
    gae_lambda: float = 0.95
    # Issue #286: macro-action wrapper. When ``macro_actions=True`` the
    # env_fn returns a :class:`MacroActionEnv` instead of a raw
    # :class:`BucketBrigadeEnv`, the action space collapses to a single
    # ``Discrete(num_options)`` head, and the trainer is constructed with
    # ``action_dims=[num_options]`` instead of ``[10, 2, 2]``. Default
    # ``False`` preserves bit-identical pre-#286 behavior.
    macro_actions: bool = False
    commit_steps: int = 3
    # Issue #289: HCA advantage path selector. ``"gae"`` (default) preserves
    # bit-identical pre-#289 behavior. ``"hca"`` swaps in the Hindsight
    # Credit Assignment advantage estimator (Harutyunyan et al. 2019, Eq. 9)
    # and trains per-agent hindsight networks alongside the actors.
    advantage_estimator: str = "gae"
    # Issue #289: Adam LR for the per-agent hindsight networks. None defers
    # to ``lr`` (the actor LR) so the default sweep cell does not need to
    # think about a second hyperparameter.
    hindsight_lr: Optional[float] = None
    # Issue #289: alphabet size of the quantile-bucketed return-to-go
    # encoding used as the future statistic ``X_t``. Default 8 matches the
    # curator's recommendation.
    hindsight_num_return_buckets: int = 8
    # Issue #289: symmetric clip on the ``pi/h`` hindsight ratio for
    # numerical stability. Mirrors PPO's own clip philosophy.
    hindsight_ratio_clip: float = 10.0
    # Issue #290: Jaques-2019 social-influence intrinsic motivation knob.
    # When ``> 0`` the trainer adds ``influence_coef * influence_i(t)`` to
    # each agent's environmental reward before the advantage estimator in
    # ``update()`` (works with GAE, HCA, and COMA); the influence term is
    # the sum across teammates ``j != i`` of KL between
    # ``pi_j(. | obs_j^{t+1}_real)`` and the counterfactual marginal
    # averaged over MC samples of ``a'_i ~ pi_i(. | obs_i^t)``. ``0.0``
    # (default) preserves bit-identical pre-#290 IPPO behavior.
    influence_coef: float = 0.0
    influence_mc_samples: int = 4


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

    Architect-validated; see ``research_notebook/2026-05-15_p3_conditioner_decision.md``.
    """
    obs_np = rollout.observations.cpu().numpy()
    # Issue #221: post-#216, ``rollout.observations`` is ``[T, N, obs_dim]``;
    # before #216 it was ``[T, obs_dim]``. The houses-state slice lives in the
    # trailing ``obs_dim`` axis, and the first ``num_houses`` slots are shared
    # across agents (the per-agent identity one-hot is in the tail), so we
    # read agent 0's view as a stand-in.
    if obs_np.ndim == 3:
        houses = obs_np[:, 0, _HOUSES_OBS_SLICE]  # [T, 10]
    else:
        houses = obs_np[:, _HOUSES_OBS_SLICE]  # [T, 10]
    # Per-house state ∈ {SAFE=0, BURNING=1, RUINED=2}; count BURNING per step.
    num_burning = (houses == _BURNING_CODE).sum(axis=1)
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


# Packed-action alphabet used by ``_other_agent_action_codes`` and the
# ``action_entropy`` metric below.
#
# Issue #235: with ``action_dims = [10, 2, 2]`` (post-#235) the pack
# ``a[:, 0] * 4 + a[:, 1] * 2 + a[:, 2]`` covers ``0..39`` inclusive
# (40 distinct values). We reserve code 40 as a sentinel for the
# "no prior action" case at ``t=0`` (and just after an episode boundary),
# so the conditioner's alphabet is ``[0, 41)``.
#
# Pre-#235 was ``a[:, 0] * 2 + a[:, 1]`` (range ``0..19``) with sentinel 20.
_ACTION_PACK_BASE_MODE = 2  # mode dimension cardinality (action_dims[1])
_ACTION_PACK_BASE_SIGNAL = 2  # signal dimension cardinality (action_dims[2])
_ACTION_PACK_BASE = _ACTION_PACK_BASE_MODE * _ACTION_PACK_BASE_SIGNAL  # 4
_ACTION_NO_PRIOR_SENTINEL = 40
_ACTION_CODE_ALPHABET = 41  # 0..39 real + 40 sentinel


def _other_agent_action_codes(rollout, agent_j: int, lag: int = 1) -> np.ndarray:
    """Per-step codes for agent ``j``'s packed action at ``t - lag``.

    Implements **Option 3** from issue #154 — a sensitivity-check conditioner
    that asks: "are agents ``i`` and ``j`` redundant beyond what their
    *observed coordination* (agent ``j``'s recent action) forces?"
    See ``research_notebook/2026-05-15_p3_conditioner_decision.md`` for the
    architect rationale on why Option 3 is reported as a **secondary**
    check alongside Option 1, not as a primary measurement.

    Per-pair convention: when measuring ``CMI(Ẑ_i; Ẑ_j | Z_ij)`` with the
    Option 3 conditioner, ``Z_ij`` is the *other* agent's lagged action —
    here, agent ``j``'s action at ``t - lag``. This matches the convention
    used in the four-way diagnostic table in the architect notebook.

    Codes use the pack
    ``a[:, 0] * _ACTION_PACK_BASE + a[:, 1] * _ACTION_PACK_BASE_SIGNAL + a[:, 2]``
    (range ``0..39`` for ``action_dims = [10, 2, 2]``, issue #235).
    For ``t < lag`` (no prior action exists yet) we emit the sentinel
    ``_ACTION_NO_PRIOR_SENTINEL`` so the conditioner is defined on
    every step.

    Note: we do **not** reset the sentinel at episode boundaries here. The
    architect spec calls for a single-step lag and defensive ``t=0`` handling;
    leaking the last action of episode ``k-1`` into the first step of episode
    ``k`` is a small effect at lag=1 and the env auto-resets observations
    (which the encoder consumes) at the same step. If a future scenario uses
    very short episodes where this matters, revisit.
    """
    if lag < 1:
        raise ValueError(f"lag must be >= 1, got {lag}")
    a = rollout.actions[agent_j].cpu().numpy()
    # Pack the multi-discrete action (issue #235):
    # a[:, 0] in [0, 10), a[:, 1] in [0, 2), a[:, 2] in [0, 2).
    if a.shape[-1] >= 3:
        packed = (
            a[:, 0] * _ACTION_PACK_BASE + a[:, 1] * _ACTION_PACK_BASE_SIGNAL + a[:, 2]
        ).astype(np.int64)
    elif a.shape[-1] == 2:
        # Legacy 2-element actions (pre-#235): pretend the agent was honest
        # (signal == mode) so the packed code remains well-defined.
        packed = (
            a[:, 0] * _ACTION_PACK_BASE + a[:, 1] * _ACTION_PACK_BASE_SIGNAL + a[:, 1]
        ).astype(np.int64)
    else:
        # Issue #286: single-column macro-action — option index is already a
        # well-defined scalar label. Use it directly as the packed code.
        packed = a[:, 0].astype(np.int64)
    T = packed.shape[0]
    codes = np.full(T, _ACTION_NO_PRIOR_SENTINEL, dtype=np.int64)
    if T > lag:
        codes[lag:] = packed[:-lag]
    return codes


def _masked_mean_cmi_action(
    cmi_values: List[float],
    conditioning_agents: List[int],
    per_agent_degenerate: Dict[int, bool],
) -> tuple[float, int]:
    """Mean of per-pair Option-3 CMIs, masking pairs with a degenerate conditioner.

    Aggregate-only masking for the Option 3 (other-agent lagged action)
    sensitivity check. Per the post-PR-#180 amendment to
    ``research_notebook/2026-05-15_p3_conditioner_decision.md``, when any
    conditioning agent's action distribution collapses to (near-)deterministic
    — which is a structural outcome of PPO at λ=0 with no specialization
    pressure — that agent's contribution to the aggregate is mathematically
    vacuous (``I(X; Y | Z) ≈ I(X; Y)``). We exclude only the affected pairs
    from the aggregate; per-pair raw CMIs continue to be emitted unmasked for
    drilldown.

    Args:
        cmi_values: Per-pair Option-3 CMI values, in the order produced by
            the ``i < j`` pair loop.
        conditioning_agents: For each entry in ``cmi_values``, the index of
            the conditioning agent (the larger of ``i, j`` per the pair
            convention). Same length as ``cmi_values``.
        per_agent_degenerate: Map ``{agent_j: bool}`` marking which
            conditioning agents have a degenerate marginal action
            distribution at this iteration.

    Returns:
        ``(mean_pair, n_valid_pairs)`` where ``mean_pair`` is the mean over
        non-degenerate pairs, or ``float('nan')`` if every pair is degenerate
        (``n_valid_pairs == 0``).
    """
    assert len(cmi_values) == len(conditioning_agents), (
        "cmi_values and conditioning_agents must align"
    )
    kept = [
        v
        for v, j in zip(cmi_values, conditioning_agents)
        if not per_agent_degenerate.get(j, False)
    ]
    n_valid = len(kept)
    if n_valid == 0:
        return float("nan"), 0
    return float(np.mean(kept)), n_valid


def _measure_information(
    trainer: JointPPOTrainer,
    rollout,
    n_bins: int,
    projection: np.ndarray,
) -> Dict[str, float]:
    """Plug-in MI/CMI on the rollout's encoder outputs.

    Reports two conditional MI series per pair, side-by-side:

    - ``cmi/*`` — Option 1, conditioned on the coarse
      ``(num_houses_burning, day_index)`` state summary.
    - ``cmi_action/*`` — Option 3, conditioned on the *other* agent's
      packed action at ``t-1`` (per-pair conditioner).

    Also logs unconditional MI per pair.

    Encoder outputs are projected from ``hidden_size`` to a small dimension
    via the shared ``projection`` matrix before quantization, because a
    direct quantize-and-pack of a 64-D vector overflows the integer code.

    Aggregate masking (Option 3 only): per the post-PR-#180 amendment to the
    architect notebook, ``cmi_action/mean_pair`` is the mean over **non-
    degenerate** per-pair CMIs only (where "degenerate" is judged on the
    *conditioning* agent ``j``'s marginal action distribution via
    :func:`is_degenerate_conditioner`). ``cmi_action/n_valid_pairs`` is
    emitted as the explicit denominator; if every conditioning agent is
    degenerate the aggregate is ``NaN`` (JSON ``null``). Per-pair raw
    ``cmi_action/agent_{i}_{j}`` values are emitted unconditionally so
    drilldown into a degenerate pair is still possible. The aggregate
    ``cmi_action/conditioner_degenerate`` flag retains its "any-degenerate"
    semantics — diagnostic, not a gate. See
    ``research_notebook/2026-05-15_p3_conditioner_decision.md`` Amendment
    section.

    Architect-validated; see ``research_notebook/2026-05-15_p3_conditioner_decision.md``.
    """
    with torch.no_grad():
        feats = trainer.encoder_outputs_batch(rollout.observations)
    feats_np = [f.cpu().numpy() @ projection for f in feats]

    # Quantize each (T, mi_proj_dims) projection into a single integer code.
    codes = [quantize_uniform(f, n_bins=n_bins) for f in feats_np]

    # Primary conditioner (Option 1, state summary). See
    # ``_state_summary_codes`` and the architect notebook for rationale.
    z_codes = _state_summary_codes(rollout)

    out: Dict[str, float] = {}

    # Defensive measurement-quality check on the Option 1 conditioner (see
    # issue #146 for the failure mode this guards against). The team-reward
    # conditioner used previously was near-constant on several scenarios
    # (``trivial_cooperation`` literally; ``default`` and ``chain_reaction``
    # near-deterministically), which made the plug-in CMI collapse to the
    # unconditional MI. The current Option 1 state-summary conditioner has not
    # tripped this guard on the May 14 sweep; the check stays in place to
    # detect future regressions (e.g., if a scenario ends on step 0 every
    # time, both components collapse).
    is_degenerate, diag = is_degenerate_conditioner(z_codes)
    out["cmi/conditioner_n_distinct"] = float(diag["n_distinct"])
    out["cmi/conditioner_modal_fraction"] = diag["modal_fraction"]
    out["cmi/conditioner_entropy_bits"] = diag["entropy_bits"]
    out["cmi/conditioner_degenerate"] = float(is_degenerate)
    if is_degenerate:
        warnings.warn(
            (
                "P3 CMI Option 1 (state-summary) conditioner appears degenerate "
                f"(n_distinct={diag['n_distinct']}, "
                f"modal_fraction={diag['modal_fraction']:.3f}, "
                f"entropy_bits={diag['entropy_bits']:.3f}). "
                "I(Ẑ_i; Ẑ_j | Z) ≈ I(Ẑ_i; Ẑ_j) is mathematically guaranteed; "
                "see issue #146 for context. Reported `cmi/*` values for this "
                "iteration should not be interpreted as 'conditional'."
            ),
            RuntimeWarning,
            stacklevel=2,
        )

    n = trainer.num_agents

    # Sensitivity-check conditioners (Option 3): for each pair ``(i, j)`` with
    # ``i < j``, the conditioner is agent ``j``'s packed action at ``t - 1``.
    # Precompute once per agent ``j`` so the per-pair loop stays cheap, and
    # so the degenerate diagnostics are emitted per conditioning agent. See
    # the architect notebook for why this is reported alongside (not in place
    # of) the Option 1 measurement.
    action_codes_per_agent: Dict[int, np.ndarray] = {
        j: _other_agent_action_codes(rollout, agent_j=j, lag=1) for j in range(n)
    }
    # Aggregate degenerate-diagnostic stats across the conditioning agents
    # we actually use (j = 1..n-1, since pairs are i < j).
    action_cond_diag_agents = list(range(1, n))
    action_cond_degenerate_any = False
    action_cond_degenerate_per_agent: Dict[int, bool] = {}
    action_cond_modal_fractions: List[float] = []
    action_cond_entropy_bits: List[float] = []
    action_cond_n_distinct: List[int] = []
    for j in action_cond_diag_agents:
        is_deg_j, diag_j = is_degenerate_conditioner(action_codes_per_agent[j])
        out[f"cmi_action/conditioner_agent_{j}_n_distinct"] = float(
            diag_j["n_distinct"]
        )
        out[f"cmi_action/conditioner_agent_{j}_modal_fraction"] = diag_j[
            "modal_fraction"
        ]
        out[f"cmi_action/conditioner_agent_{j}_entropy_bits"] = diag_j["entropy_bits"]
        out[f"cmi_action/conditioner_agent_{j}_degenerate"] = float(is_deg_j)
        action_cond_degenerate_any = action_cond_degenerate_any or is_deg_j
        action_cond_degenerate_per_agent[j] = bool(is_deg_j)
        action_cond_modal_fractions.append(diag_j["modal_fraction"])
        action_cond_entropy_bits.append(diag_j["entropy_bits"])
        action_cond_n_distinct.append(int(diag_j["n_distinct"]))
        if is_deg_j:
            warnings.warn(
                (
                    "P3 CMI Option 3 (other-agent-action) conditioner appears "
                    f"degenerate for agent {j} "
                    f"(n_distinct={diag_j['n_distinct']}, "
                    f"modal_fraction={diag_j['modal_fraction']:.3f}, "
                    f"entropy_bits={diag_j['entropy_bits']:.3f}). "
                    "I(Ẑ_i; Ẑ_j | A_j[t-1]) ≈ I(Ẑ_i; Ẑ_j) is mathematically "
                    "guaranteed for pairs conditioned on this agent. "
                    f"Reported `cmi_action/agent_*_{j}` values for this "
                    "iteration should not be interpreted as 'conditional'."
                ),
                RuntimeWarning,
                stacklevel=2,
            )
    # Aggregate diagnostics across conditioning agents (mean over j).
    if action_cond_diag_agents:
        out["cmi_action/conditioner_n_distinct"] = float(
            np.mean(action_cond_n_distinct)
        )
        out["cmi_action/conditioner_modal_fraction"] = float(
            np.mean(action_cond_modal_fractions)
        )
        out["cmi_action/conditioner_entropy_bits"] = float(
            np.mean(action_cond_entropy_bits)
        )
        out["cmi_action/conditioner_degenerate"] = float(action_cond_degenerate_any)

    mi_vals = []
    cmi_vals = []
    cmi_action_vals: List[float] = []
    cmi_action_conditioning_agents: List[int] = []
    for i in range(n):
        for j in range(i + 1, n):
            mi = mutual_information(codes[i], codes[j])
            cmi = conditional_mutual_information(codes[i], codes[j], z_codes)
            cmi_action = conditional_mutual_information(
                codes[i], codes[j], action_codes_per_agent[j]
            )
            out[f"mi/agent_{i}_{j}"] = mi
            out[f"cmi/agent_{i}_{j}"] = cmi
            # Per-pair raw CMI is emitted unconditionally; masking applies only
            # to the aggregate below. See docstring + notebook Amendment.
            out[f"cmi_action/agent_{i}_{j}"] = cmi_action
            mi_vals.append(mi)
            cmi_vals.append(cmi)
            cmi_action_vals.append(cmi_action)
            cmi_action_conditioning_agents.append(j)
    out["mi/mean_pair"] = float(np.mean(mi_vals))
    out["cmi/mean_pair"] = float(np.mean(cmi_vals))
    cmi_action_mean, n_valid_pairs = _masked_mean_cmi_action(
        cmi_action_vals,
        cmi_action_conditioning_agents,
        action_cond_degenerate_per_agent,
    )
    out["cmi_action/mean_pair"] = cmi_action_mean
    out["cmi_action/n_valid_pairs"] = n_valid_pairs

    # Marginal action entropy per agent (proxy for role entropy H(A_i^*)).
    for i in range(n):
        a = rollout.actions[i].cpu().numpy()
        # Pack multi-discrete action into a single label per step.
        # Issue #235: pack [house, mode, signal] into 0..39 (action_dims
        # = [10, 2, 2]). Pre-#235 was [house, mode] into 0..19. Handle
        # legacy 2-element actions defensively by treating them as honest
        # (signal == mode).
        if a.shape[-1] >= 3:
            packed = (
                a[:, 0] * _ACTION_PACK_BASE
                + a[:, 1] * _ACTION_PACK_BASE_SIGNAL
                + a[:, 2]
            )
        elif a.shape[-1] == 2:
            packed = (
                a[:, 0] * _ACTION_PACK_BASE
                + a[:, 1] * _ACTION_PACK_BASE_SIGNAL
                + a[:, 1]
            )
        else:
            # Issue #286: single-column macro-action — option index serves
            # as the packed action code directly.
            packed = a[:, 0].astype(np.int64)
        out[f"action_entropy/agent_{i}"] = entropy_discrete(packed)
    out["action_entropy/mean"] = float(
        np.mean([out[f"action_entropy/agent_{i}"] for i in range(n)])
    )

    return out


def _validate_curriculum(curriculum: List[List[int]]) -> List[List[int]]:
    """Validate and normalize a curriculum schedule (issue #260).

    Each entry must be ``[start_iteration, min_nights]`` with non-negative
    integer ``start_iteration`` and strictly-positive integer ``min_nights``.
    The returned list is sorted by ``start_iteration`` ascending. Duplicate
    ``start_iteration`` values are rejected (ambiguous resolution).

    Raises:
        ValueError: on any malformed entry. No silent fallback.
    """
    if not curriculum:
        return []
    normalized: List[List[int]] = []
    seen_starts: set[int] = set()
    for idx, entry in enumerate(curriculum):
        if len(entry) != 2:
            raise ValueError(
                f"curriculum[{idx}] must be a 2-element [iter, min_nights] "
                f"pair, got {entry!r}"
            )
        start_it, floor = entry
        if not isinstance(start_it, int) or not isinstance(floor, int):
            raise ValueError(
                f"curriculum[{idx}] entries must be ints, got "
                f"({type(start_it).__name__}, {type(floor).__name__})"
            )
        if start_it < 0:
            raise ValueError(
                f"curriculum[{idx}] start_iteration must be >= 0, got {start_it}"
            )
        if floor <= 0:
            raise ValueError(
                f"curriculum[{idx}] min_nights floor must be > 0, got {floor}"
            )
        if start_it in seen_starts:
            raise ValueError(
                f"curriculum has duplicate start_iteration={start_it}; "
                f"each iteration boundary must be unique"
            )
        seen_starts.add(start_it)
        normalized.append([int(start_it), int(floor)])
    normalized.sort(key=lambda x: x[0])
    return normalized


def _curriculum_floor_for(
    curriculum: List[List[int]], iteration: int, default_floor: int
) -> int:
    """Return the ``min_nights`` floor in effect at ``iteration`` (issue #260).

    The active floor is the latest phase whose ``start_iteration <= iteration``.
    If no phase has started yet (``iteration < curriculum[0][0]``), the
    scenario's native ``default_floor`` is returned, preserving pre-curriculum
    behavior for early iterations.
    """
    if not curriculum:
        return default_floor
    active = default_floor
    for start_it, floor in curriculum:
        if start_it <= iteration:
            active = floor
        else:
            break
    return active


def _parse_curriculum_arg(value: str) -> List[List[int]]:
    """Parse the ``--curriculum`` CLI flag (issue #260).

    Format: ``'iter:min_nights,iter:min_nights,...'`` (e.g. ``'0:5,17:8,34:12'``).
    Empty string returns ``[]`` (curriculum disabled). Whitespace around
    tokens is tolerated. Malformed input raises ``argparse.ArgumentTypeError``
    so the CLI exits non-zero with a clear message.
    """
    if value is None or value.strip() == "":
        return []
    phases: List[List[int]] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" not in token:
            raise argparse.ArgumentTypeError(
                f"Curriculum phase {token!r} missing ':' separator. "
                f"Expected 'iter:min_nights', e.g. '0:5'."
            )
        left, _, right = token.partition(":")
        try:
            start_it = int(left.strip())
            floor = int(right.strip())
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Curriculum phase {token!r} has non-integer field: {exc}"
            ) from None
        phases.append([start_it, floor])
    # _validate_curriculum surfaces the same errors with clearer messages.
    try:
        return _validate_curriculum(phases)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def train_one_cell(cfg: CellConfig, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    scenario = get_scenario_by_name(cfg.scenario, num_agents=cfg.num_agents)

    # Issue #261/#262: mutate the loaded ``Scenario`` instance so the
    # per-cell action-shaping knobs take effect inside ``_compute_rewards``.
    # ``Scenario.__post_init__`` only validates the ownership-reward
    # promotion and ``num_houses`` — it does not touch the shaping fields
    # — so post-construction mutation is safe. Cells where both knobs are
    # 0.0 (e.g., the in-sweep baseline) still hit the env's fast-path
    # skip, preserving bit-identical pre-#259 behavior.
    scenario.action_shaping_alpha = float(cfg.action_shaping_alpha)
    scenario.action_shaping_beta = float(cfg.action_shaping_beta)

    # Issue #265: mutate the dense progress shaping coefficient. Default
    # ``0.0`` keeps the env fast-path skip and bit-identical pre-#265
    # behavior; the #265 sweep driver passes per-cell values.
    scenario.progress_shaping_coef = float(cfg.progress_shaping_coef)

    # Issue #283: potential-based team-welfare shaping (Ng-Harada-Russell 1999).
    # Mutate the same loaded ``Scenario`` instance. When lambda=0.0 (default)
    # the env takes the fast-path skip and per-step rewards are byte-identical
    # to pre-#283 behavior, so the in-sweep baseline cell can share the same
    # code path as treatment cells without distorting the comparison.
    scenario.team_welfare_lambda = float(cfg.team_welfare_lambda)
    scenario.team_welfare_gamma = float(cfg.team_welfare_gamma)
    scenario.team_welfare_kind = str(cfg.team_welfare_kind)

    # Issue #286: when macro_actions is on, wrap the base env in
    # ``MacroActionEnv``. The wrapper preserves the dict-obs / numpy-action
    # contract so it is a drop-in for ``JointPPOTrainer.env_fn``; only
    # ``action_dims`` changes (collapses to ``[num_options]``).
    if cfg.macro_actions:

        def env_fn():
            base = BucketBrigadeEnv(scenario=scenario)
            return MacroActionEnv(base, commit_steps=cfg.commit_steps)
    else:

        def env_fn():
            env = BucketBrigadeEnv(scenario=scenario)
            return env

    # Probe obs_dim from a single reset. Issue #204: include the per-agent
    # identity one-hot tail so obs_dim matches what JointPPOTrainer actually
    # consumes during rollouts.
    probe = env_fn()
    probe_obs = probe.reset(seed=cfg.seed)
    obs_dim = flatten_dict_obs(probe_obs, agent_id=0, num_agents=cfg.num_agents).shape[
        0
    ]

    # Issue #286: action_dims comes from the wrapper when macro-actions are
    # on (single Discrete head with num_options values); otherwise use the
    # configured ``cfg.action_dims`` (default ``[num_houses, 2, 2]``).
    # Mutate ``cfg.action_dims`` to the runtime value so the on-disk
    # ``config.json`` reflects what the trainer actually saw.
    if cfg.macro_actions:
        action_dims = [int(probe.num_options)]
        cfg.action_dims = action_dims
    else:
        action_dims = cfg.action_dims

    trainer = JointPPOTrainer(
        env_fn=env_fn,
        num_agents=cfg.num_agents,
        obs_dim=obs_dim,
        action_dims=action_dims,
        hidden_size=cfg.hidden_size,
        lr=cfg.lr,
        gae_lambda=cfg.gae_lambda,
        ppo_epochs=cfg.ppo_epochs,
        minibatch_size=cfg.minibatch_size,
        value_coef=cfg.value_coef,
        entropy_coef=cfg.entropy_coef,
        redundancy_coef=cfg.lambda_red,
        normalize_returns=cfg.normalize_returns,
        centralized_critic=cfg.centralized_critic,
        coma=cfg.coma,
        coma_target_update_every=cfg.coma_target_update_every,
        advantage_estimator=cfg.advantage_estimator,
        hindsight_lr=cfg.hindsight_lr,
        hindsight_num_return_buckets=cfg.hindsight_num_return_buckets,
        hindsight_ratio_clip=cfg.hindsight_ratio_clip,
        influence_coef=cfg.influence_coef,
        influence_mc_samples=cfg.influence_mc_samples,
        device=cfg.device,
        seed=cfg.seed,
    )

    # Issue #270: optionally warm-start the actor weights from a BC-pretrained
    # checkpoint directory produced by ``experiments/p3_specialization/bc_init.py``.
    # Loaded after the trainer is constructed (which sets the random-init
    # baseline) so the per-agent state dict overwrites only the actor; the
    # value head, optimizer state, and (when ``centralized_critic=True``)
    # the critic remain freshly initialized. This is the basin-trap vs
    # anti-attractor probe — BC sets the actor near specialist, then PPO
    # continues from there.
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
            # ``load_state_dict`` is strict by default; a shape mismatch (e.g.,
            # the BC fit used a different hidden_size) will raise here instead
            # of silently shifting the obs encoding mid-training.
            policy.load_state_dict(sd)
        print(
            f"  BC-init (#270): loaded {cfg.num_agents} per-agent state dicts "
            f"from {bc_dir}"
        )

    # Shared random projection matrix for MI measurement. Seeded from cfg.seed
    # so the metric is reproducible and comparable across iterations.
    rng = np.random.default_rng(cfg.seed)
    projection = rng.standard_normal((cfg.hidden_size, cfg.mi_proj_dims)).astype(
        np.float32
    )
    # Unit-norm columns make the projected values comparable across runs.
    projection /= np.linalg.norm(projection, axis=0, keepdims=True) + 1e-8

    # Issue #260: episode-length curriculum. Validated once up-front so a
    # malformed schedule fails before we burn any rollout budget. The
    # scenario's native ``min_nights`` is the default floor before the
    # first curriculum phase begins.
    curriculum = _validate_curriculum(cfg.curriculum)
    native_min_nights = int(scenario.min_nights)

    metrics_log: List[Dict[str, float]] = []
    for it in range(cfg.num_iterations):
        # Apply curriculum: mutate ``scenario.min_nights`` in place. The
        # trainer holds a single env via ``env_fn()`` and reads
        # ``scenario.min_nights`` on every ``_check_termination()`` call,
        # so the new floor takes effect on the next episode boundary
        # without reconstructing the env (preserves optimizer state).
        new_floor = _curriculum_floor_for(curriculum, it, native_min_nights)
        if trainer.env.scenario.min_nights != new_floor:
            trainer.env.scenario.min_nights = new_floor
        current_floor = int(trainer.env.scenario.min_nights)

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
            # Issue #260: record the curriculum floor in effect at this
            # iteration so the analysis pass can verify the schedule
            # applied and overlay it on the reward trajectory.
            "min_nights_floor": current_floor,
            **stats,
            **info_stats,
        }
        metrics_log.append(record)

        if it % max(1, cfg.num_iterations // 10) == 0 or it == cfg.num_iterations - 1:
            print(
                f"  iter {it:4d} | team_reward {mean_reward:8.3f} | "
                f"mi_mean {info_stats['mi/mean_pair']:.3f} | "
                f"cmi_mean {info_stats['cmi/mean_pair']:.3f} | "
                f"cmi_action_mean {info_stats['cmi_action/mean_pair']:.3f} "
                f"(n_valid={info_stats['cmi_action/n_valid_pairs']}) | "
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
    p.add_argument(
        "--centralized-critic",
        action="store_true",
        help=(
            "Issue #208: enable MAPPO (centralized critic, decentralized "
            "actors). One shared CentralizedCritic consumes the global "
            "obs (identity-tail stripped) and produces a value baseline "
            "shared across all agents; per-agent advantages still come "
            "from per-agent rewards. Default off preserves IPPO. "
            "Incompatible with --lambda-red > 0."
        ),
    )
    p.add_argument(
        "--use-coma",
        dest="coma",
        action="store_true",
        help=(
            "Issue #284: enable COMA (counterfactual multi-agent policy "
            "gradient). Installs a CentralizedQCritic and replaces the "
            "advantage with A_i = Q(s, a) - sum_u pi_i(u | s) Q(s, "
            "(a_{-i}, u)), enumerated over the joint per-agent action "
            "space (prod(action_dims) = 40 for [10, 2, 2]). Default off "
            "preserves IPPO. Mutually exclusive with --centralized-critic "
            "and --lambda-red > 0."
        ),
    )
    p.add_argument(
        "--coma-target-update-every",
        type=int,
        default=CellConfig.__dataclass_fields__["coma_target_update_every"].default,
        help=(
            "Issue #284: hard-copy cadence for the COMA target Q-network "
            "(in PPO update steps). Default 200 matches Foerster 2018. "
            "Set to 0 to disable target-network updates entirely."
        ),
    )
    p.add_argument(
        "--curriculum",
        type=_parse_curriculum_arg,
        default=[],
        help=(
            "Issue #260: optional episode-length curriculum. Format: "
            "'iter:min_nights,iter:min_nights,...' e.g. '0:5,17:8,34:12'. "
            "Empty (default) disables curriculum and preserves bit-identical "
            "behavior. Phases apply at their start iteration; the floor at "
            "iteration i is the latest phase whose start_iteration <= i. "
            "Mutates trainer.env.scenario.min_nights in place between "
            "iterations (no env reconstruction, optimizer state preserved)."
        ),
    )
    p.add_argument(
        "--action-shaping-alpha",
        type=float,
        default=CellConfig.__dataclass_fields__["action_shaping_alpha"].default,
        help=(
            "Issue #261/#262: credit-shared extinguish bonus knob. Mutated "
            "onto the loaded Scenario instance pre-rollout. Default 0.0 "
            "keeps the env fast-path skip and bit-identical pre-#259 "
            "behavior. Use 0.1/0.5/2.0 in the calibration sweep."
        ),
    )
    p.add_argument(
        "--action-shaping-beta",
        type=float,
        default=CellConfig.__dataclass_fields__["action_shaping_beta"].default,
        help=(
            "Issue #261/#262: flat preventive-presence bonus knob. Mutated "
            "onto the loaded Scenario instance pre-rollout. Default 0.0 "
            "keeps the env fast-path skip and bit-identical pre-#259 "
            "behavior. Use 0.0/0.1/0.5 in the calibration sweep."
        ),
    )
    p.add_argument(
        "--progress-shaping-coef",
        type=float,
        default=CellConfig.__dataclass_fields__["progress_shaping_coef"].default,
        help=(
            "Issue #265: dense Δsafe per-step team-shaping coefficient. "
            "Mutated onto the loaded Scenario instance pre-rollout. "
            "Default 0.0 keeps the env fast-path skip and bit-identical "
            "pre-#265 behavior. The #265 sweep uses {0.0, 1.0, 5.0, 25.0}."
        ),
    )
    p.add_argument(
        "--team-welfare-lambda",
        type=float,
        default=CellConfig.__dataclass_fields__["team_welfare_lambda"].default,
        help=(
            "Issue #283: potential-based team-welfare shaping multiplier "
            "(Ng-Harada-Russell 1999). Mutated onto the loaded Scenario "
            "instance pre-rollout. Default 0.0 keeps the env fast-path "
            "skip and bit-identical pre-#283 behavior. Sweep cells: "
            "{0.0, 0.1, 0.5, 1.0}."
        ),
    )
    p.add_argument(
        "--team-welfare-gamma",
        type=float,
        default=CellConfig.__dataclass_fields__["team_welfare_gamma"].default,
        help=(
            "Issue #283: NHR discount factor in `gamma*Phi(s') - Phi(s)`. "
            "Default 1.0 collapses shaping to a simple delta. Should "
            "match the trainer's PPO gamma for maximum invariance "
            "strength, but NHR holds for any gamma in [0, 1]."
        ),
    )
    p.add_argument(
        "--team-welfare-kind",
        type=str,
        default=CellConfig.__dataclass_fields__["team_welfare_kind"].default,
        choices=("none", "team_welfare_closed_form"),
        help=(
            "Issue #283: potential function selector. 'none' disables "
            "shaping. 'team_welfare_closed_form' = option B (cheap, no "
            "learning): Phi(s) = team_reward*safe/N - team_penalty*ruined/N "
            "- 0.5*team_penalty*burning/N. Future kinds (e.g., "
            "'specialist_mc_regressor' = option A) will be added when "
            "the headline Phi regressor lands."
        ),
    )
    p.add_argument(
        "--bc-init-checkpoint-dir",
        type=str,
        default=None,
        help=(
            "Issue #270: directory containing per-agent BC-pretrained "
            "state dicts (agent_{i}.pt) produced by "
            "experiments/p3_specialization/bc_init.py. When set, the actor "
            "weights are loaded into trainer.policies[i] after trainer "
            "construction, before the first PPO iteration. Default None "
            "preserves bit-identical pre-#270 random-init behavior."
        ),
    )
    p.add_argument(
        "--gae-lambda",
        type=float,
        default=CellConfig.__dataclass_fields__["gae_lambda"].default,
        help=(
            "Issue #282: GAE lambda for advantage estimation. Default 0.95 "
            "matches JointPPOTrainer's existing default and preserves "
            "bit-identical pre-#282 behavior. The issue #282 sweep raises "
            "this to {0.99, 0.999, 1.0} to test whether GAE bootstrap bias "
            "is the source of PPO's plateau on minimal_specialization "
            "(thesis: gradient locality / bootstrap bias as the "
            "misalignment source). lambda=1.0 is pure Monte Carlo (no "
            "bootstrap)."
        ),
    )
    p.add_argument(
        "--macro-actions",
        action="store_true",
        help=(
            "Issue #286: wrap the env in MacroActionEnv. Action space "
            "becomes a single Discrete head over 3+(num_agents-1) options "
            "(PATROL, DEFEND_OWN, REST_UNTIL_FIRE, FOLLOW_j). Each option "
            "commits the agent for --commit-steps base steps (or until "
            "env-done). Default off preserves bit-identical pre-#286 "
            "behavior."
        ),
    )
    p.add_argument(
        "--commit-steps",
        type=int,
        default=CellConfig.__dataclass_fields__["commit_steps"].default,
        help=(
            "Issue #286: number of base-env steps each macro-action commits "
            "to (subject to early termination on env-done). Only used when "
            "--macro-actions is set. Default 3. Sweep grid is {3, 5, 8}."
        ),
    )
    p.add_argument(
        "--advantage-estimator",
        type=str,
        choices=["gae", "hca"],
        default=CellConfig.__dataclass_fields__["advantage_estimator"].default,
        help=(
            "Issue #289: select the advantage estimator path. 'gae' "
            "(default) preserves bit-identical pre-#289 behavior. 'hca' "
            "swaps in Hindsight Credit Assignment (Harutyunyan et al. "
            "2019, Eq. 9) and trains per-agent hindsight nets alongside "
            "the actors."
        ),
    )
    p.add_argument(
        "--use-hca",
        action="store_true",
        help=(
            "Issue #289: convenience alias for "
            "'--advantage-estimator hca'. Mutually exclusive override of "
            "the --advantage-estimator flag."
        ),
    )
    p.add_argument(
        "--hindsight-lr",
        type=float,
        default=None,
        help=(
            "Issue #289: Adam LR for the per-agent hindsight networks. "
            "Default (None) defers to the actor --lr."
        ),
    )
    p.add_argument(
        "--hindsight-num-return-buckets",
        type=int,
        default=CellConfig.__dataclass_fields__["hindsight_num_return_buckets"].default,
        help=(
            "Issue #289: alphabet size of the quantile-bucketed return-to-"
            "go encoding used as the HCA future statistic X_t. Default 8."
        ),
    )
    p.add_argument(
        "--hindsight-ratio-clip",
        type=float,
        default=CellConfig.__dataclass_fields__["hindsight_ratio_clip"].default,
        help=(
            "Issue #289: symmetric clip on the pi/h hindsight ratio for "
            "numerical stability. Default 10.0."
        ),
    )
    p.add_argument(
        "--influence-coef",
        type=float,
        default=CellConfig.__dataclass_fields__["influence_coef"].default,
        help=(
            "Issue #290: alpha for Jaques-2019 social-influence intrinsic "
            "motivation. When > 0, each agent's reward is augmented by "
            "alpha * sum_{j != i} KL(pi_j(.|obs_j^{t+1}_real) || E_a'_i "
            "[pi_j(.|obs_j^{t+1}_cf(a'_i))]) before the advantage "
            "estimator (works with GAE, HCA, and COMA). Default 0.0 "
            "preserves bit-identical IPPO behavior. Sweep grid: "
            "{0.0, 0.1, 0.5, 1.0}."
        ),
    )
    p.add_argument(
        "--influence-mc-samples",
        type=int,
        default=CellConfig.__dataclass_fields__["influence_mc_samples"].default,
        help=(
            "Issue #290: K, Monte-Carlo samples used to estimate the "
            "counterfactual marginal in the influence reward. Default 4 "
            "matches the Jaques 2019 paper. Ignored when "
            "--influence-coef == 0."
        ),
    )
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    # Issue #289: ``--use-hca`` is a convenience alias for
    # ``--advantage-estimator hca``. It overrides whatever was passed via
    # ``--advantage-estimator`` so callers don't need to set both flags.
    advantage_estimator = "hca" if args.use_hca else args.advantage_estimator

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
        centralized_critic=args.centralized_critic,
        coma=args.coma,
        coma_target_update_every=args.coma_target_update_every,
        curriculum=args.curriculum,
        action_shaping_alpha=args.action_shaping_alpha,
        action_shaping_beta=args.action_shaping_beta,
        progress_shaping_coef=args.progress_shaping_coef,
        team_welfare_lambda=args.team_welfare_lambda,
        team_welfare_gamma=args.team_welfare_gamma,
        team_welfare_kind=args.team_welfare_kind,
        bc_init_checkpoint_dir=args.bc_init_checkpoint_dir,
        gae_lambda=args.gae_lambda,
        macro_actions=args.macro_actions,
        commit_steps=args.commit_steps,
        advantage_estimator=advantage_estimator,
        hindsight_lr=args.hindsight_lr,
        hindsight_num_return_buckets=args.hindsight_num_return_buckets,
        hindsight_ratio_clip=args.hindsight_ratio_clip,
        influence_coef=args.influence_coef,
        influence_mc_samples=args.influence_mc_samples,
        device=args.device,
    )
    print(
        f"== P3 cell: scenario={cfg.scenario} lambda_red={cfg.lambda_red} "
        f"seed={cfg.seed} "
        f"gae_lambda={cfg.gae_lambda} "
        f"action_shaping_alpha={cfg.action_shaping_alpha} "
        f"action_shaping_beta={cfg.action_shaping_beta} "
        f"progress_shaping_coef={cfg.progress_shaping_coef} "
        f"team_welfare_lambda={cfg.team_welfare_lambda} "
        f"team_welfare_kind={cfg.team_welfare_kind} "
        f"macro_actions={cfg.macro_actions} commit_steps={cfg.commit_steps} "
        f"influence_coef={cfg.influence_coef} =="
    )
    train_one_cell(cfg, args.output_dir)


if __name__ == "__main__":
    main()
