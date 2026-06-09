"""Per-position conditional action entropy from heterogeneous Nash profiles.

This module implements the measurement layer for the SW-MARL paper's
trainability predictor (#368). Given a converged heterogeneous Nash strategy
profile (one genome per position), it:

1. Drives ``BucketBrigadeEnv`` step-by-step from Python with 4
   ``HeuristicAgent`` instances and captures the joint action
   ``(a_0, a_1, a_2, a_3)`` at every step. Each agent action is a length-3
   ``[house, mode, signal]`` vector (issue #235); we use the full 3-tuple as
   the per-position discrete category so the joint label space is
   ``(num_houses * 2 * 2) ** 4`` in the worst case.
2. Tabulates the joint action distribution.
3. Estimates per-position ``\\tilde{H}_i := H(A_i^* | A_{-i}^*)`` in bits via
   :func:`bucket_brigade.analysis.info_theory.conditional_entropy` (plug-in
   plus Miller-Madow correction) with a bootstrap 95% CI that resamples
   *episodes* — not individual steps — to respect within-episode
   non-independence.

The episode-level bootstrap is the load-bearing piece. ``info_theory``'s
``bootstrap_ci`` resamples at the sample (step) level, which is invalid for
trajectory data; this module wraps it with an episode-aware sampler.

Compute placement note: the rollout is CPU-bound and parallelises trivially
across episodes via ``multiprocessing.Pool`` (mirrors the
``RustPayoffEvaluator`` pattern). 10k episodes of ~50 steps is well under a
CPU-minute per cell on a 32-thread cluster host. Do NOT run the full sweep
on a laptop — see ``CLAUDE.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from typing import Callable, Optional, Sequence

import numpy as np

from bucket_brigade.agents.heuristic_agent import HeuristicAgent
from bucket_brigade.analysis.info_theory import (
    conditional_entropy,
    entropy_discrete,
    joint_entropy,
)
from bucket_brigade.envs.bucket_brigade_env import BucketBrigadeEnv
from bucket_brigade.envs.scenarios_generated import Scenario

__all__ = [
    "EpisodeActions",
    "PositionEntropy",
    "CellEntropyResult",
    "rollout_joint_actions",
    "episode_bootstrap_ci",
    "estimate_cell_entropy",
]


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EpisodeActions:
    """Per-episode action trace.

    ``actions`` has shape ``(T, num_agents, action_dim)`` with ``action_dim``
    typically 3 (``[house, mode, signal]``). ``T`` varies across episodes
    because BucketBrigadeEnv terminates when no fires remain (or all houses
    are ruined / SAFE).
    """

    actions: np.ndarray  # shape (T, num_agents, action_dim), dtype int


@dataclass(frozen=True)
class PositionEntropy:
    """Per-position estimate of ``H(A_i^* | A_{-i}^*)`` with bootstrap CI.

    All entropies are in bits. ``h_cond`` may equal 0 when the conditional
    entropy is at or below the MM noise floor; the floor-at-zero is applied
    inside :func:`conditional_entropy`.
    """

    position: int
    h_joint: float  # H(A_0, ..., A_{N-1})
    h_minus_i: float  # H(A_{-i})
    h_cond: float  # H(A_i | A_{-i}) = H_joint - H_{-i}
    h_cond_ci_lo: float
    h_cond_ci_hi: float
    n_episodes: int
    n_steps: int


@dataclass(frozen=True)
class CellEntropyResult:
    """Container for the per-cell, per-position estimates."""

    cell_tag: str
    beta: float
    kappa: float
    c: float
    verdict: str
    n_episodes: int
    n_steps_total: int
    positions: list[PositionEntropy]


# ---------------------------------------------------------------------------
# Rollout driver
# ---------------------------------------------------------------------------


def _action_to_label(action_row: np.ndarray) -> tuple:
    """Pack a length-3 action ``[house, mode, signal]`` into a hashable label.

    We use a plain Python tuple of ints because ``entropy_discrete`` runs
    ``Counter`` over the object array directly; tuples are hashable and
    cheap to construct.
    """
    return (int(action_row[0]), int(action_row[1]), int(action_row[2]))


def _run_one_episode(
    args: tuple[list[np.ndarray], Scenario, int],
) -> np.ndarray:
    """Run one episode and return the action trace.

    Worker function for ``multiprocessing.Pool.map``. Returns a numpy array
    of shape ``(T, num_agents, 3)``.

    ``args``:
        ``(genomes, scenario, seed)`` where ``genomes`` is a list of length
        ``num_agents``, each a length-10 float vector.
    """
    genomes, scenario, seed = args

    env = BucketBrigadeEnv(scenario=scenario)
    obs = env.reset(seed=seed)

    # Build per-position HeuristicAgents from genomes. We seed numpy's global
    # RNG so the agents' internal np.random calls are reproducible across
    # workers; this is necessary because HeuristicAgent uses np.random.* rather
    # than a per-instance Generator. The seed is offset to avoid colliding
    # with the env RNG.
    np.random.seed((seed + 991) % (2**31 - 1))
    agents = [
        HeuristicAgent(params=np.asarray(genomes[i], dtype=np.float64), agent_id=i)
        for i in range(scenario.num_agents)
    ]
    for a in agents:
        a.reset()

    trace: list[np.ndarray] = []
    done = False
    while not done:
        action_list = [agent.act(obs) for agent in agents]
        actions = np.stack(action_list, axis=0).astype(np.int8)
        trace.append(actions)
        obs, _rewards, dones, _info = env.step(actions)
        done = bool(dones[0])

    return np.stack(trace, axis=0)  # (T, num_agents, action_dim)


def rollout_joint_actions(
    genomes: Sequence[np.ndarray],
    scenario: Scenario,
    n_episodes: int,
    seed: int = 0,
    num_workers: Optional[int] = None,
) -> list[EpisodeActions]:
    """Run ``n_episodes`` rollouts of the per-position Nash profile.

    Args:
        genomes: Sequence of length ``num_agents`` (4 for BB) of length-10
            float vectors. ``genomes[i]`` is the genome assigned to position
            ``i`` from the converged NE profile.
        scenario: The :class:`Scenario` defining the cell (β, κ, c, ...).
        n_episodes: Number of independent episodes to roll out.
        seed: Base seed; per-episode seeds are derived deterministically.
        num_workers: If ``None``, uses ``cpu_count()`` for parallel rollouts.
            Pass ``1`` (or any value ≤ 1) to disable multiprocessing — useful
            for tests and small ``n_episodes``.

    Returns:
        List of length ``n_episodes`` containing per-episode action traces.
    """
    if len(genomes) != scenario.num_agents:
        raise ValueError(
            f"genomes length {len(genomes)} != scenario.num_agents "
            f"{scenario.num_agents}"
        )

    rng = np.random.default_rng(seed)
    seeds = [int(s) for s in rng.integers(0, 2**31 - 1, size=n_episodes)]
    genome_list = [np.asarray(g, dtype=np.float64) for g in genomes]

    args_list = [(genome_list, scenario, s) for s in seeds]

    if num_workers is None:
        num_workers = cpu_count()

    if num_workers > 1 and n_episodes > 1:
        with Pool(processes=num_workers) as pool:
            traces = pool.map(_run_one_episode, args_list)
    else:
        traces = [_run_one_episode(args) for args in args_list]

    return [EpisodeActions(actions=t) for t in traces]


# ---------------------------------------------------------------------------
# Episode-level bootstrap
# ---------------------------------------------------------------------------


def episode_bootstrap_ci(
    estimator: Callable[..., float],
    episode_arrays: Sequence[Sequence[np.ndarray]],
    n_boot: int = 1000,
    confidence: float = 0.95,
    rng: Optional[np.random.Generator] = None,
) -> tuple[float, float, float]:
    """Episode-level bootstrap CI for an info-theoretic estimator.

    ``info_theory.bootstrap_ci`` resamples with replacement at the *sample*
    level — but adjacent timesteps within an episode are not i.i.d., so the
    sample-level bootstrap underestimates the CI. This wrapper resamples
    *episodes* with replacement, concatenates the per-episode arrays, then
    calls ``estimator``.

    Args:
        estimator: Function taking the unpacked concatenated arrays and
            returning a float. Same calling convention as
            :func:`info_theory.bootstrap_ci`.
        episode_arrays: A sequence of length ``num_variables``. Each element
            is itself a sequence of length ``n_episodes``: the per-episode
            1-D array for that variable. All variables must agree on
            per-episode lengths (they are tabulated against each other).
        n_boot: Number of bootstrap resamples.
        confidence: Coverage level for the CI (default 0.95).
        rng: Optional NumPy random Generator for reproducibility.

    Returns:
        ``(point_estimate, lower, upper)``.
    """
    if rng is None:
        rng = np.random.default_rng()

    num_vars = len(episode_arrays)
    if num_vars == 0:
        raise ValueError("episode_arrays must contain at least one variable")

    n_eps = len(episode_arrays[0])
    if n_eps == 0:
        raise ValueError("each variable must have at least one episode")
    for var_idx, var_eps in enumerate(episode_arrays):
        if len(var_eps) != n_eps:
            raise ValueError(
                f"variable {var_idx} has {len(var_eps)} episodes, expected {n_eps}"
            )
        # Validate per-episode length parity across variables.
        for ep_idx in range(n_eps):
            length_first = len(episode_arrays[0][ep_idx])
            length_here = len(var_eps[ep_idx])
            if length_here != length_first:
                raise ValueError(
                    f"variable {var_idx} episode {ep_idx} length "
                    f"{length_here} != first-variable length {length_first}"
                )

    # Point estimate on the full concatenation.
    full_arrays = [np.concatenate(list(var_eps), axis=0) for var_eps in episode_arrays]
    point = float(estimator(*full_arrays))

    boots = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n_eps, size=n_eps)
        boot_arrays = []
        for var_eps in episode_arrays:
            boot_arrays.append(np.concatenate([var_eps[i] for i in idx], axis=0))
        boots[b] = float(estimator(*boot_arrays))

    alpha = (1.0 - confidence) / 2.0
    lower = float(np.quantile(boots, alpha))
    upper = float(np.quantile(boots, 1.0 - alpha))
    return point, lower, upper


# ---------------------------------------------------------------------------
# Per-cell estimator
# ---------------------------------------------------------------------------


def _build_per_position_episode_arrays(
    episodes: list[EpisodeActions],
    num_agents: int,
) -> list[list[np.ndarray]]:
    """Convert episode action traces into per-position episode-array lists.

    Output shape: ``per_position[i]`` is a list of length ``n_episodes``, each
    a length-``T_ep`` object array whose entries are ``(house, mode, signal)``
    tuples for position ``i``.
    """
    per_position: list[list[np.ndarray]] = [[] for _ in range(num_agents)]
    for ep in episodes:
        actions = ep.actions  # (T, N, action_dim)
        T = actions.shape[0]
        for i in range(num_agents):
            # Pack each step's length-3 row into a tuple. We build an object
            # array of tuples directly so downstream Counter calls in
            # entropy_discrete work correctly (the same pattern as
            # _as_tuples in info_theory).
            labels = np.empty(T, dtype=object)
            for t in range(T):
                labels[t] = _action_to_label(actions[t, i])
            per_position[i].append(labels)
    return per_position


def _other_positions_episodes(
    per_position: list[list[np.ndarray]],
    i: int,
) -> list[np.ndarray]:
    """For each episode, build the joint label for positions ``{0..N-1} \\ {i}``."""
    others = [j for j in range(len(per_position)) if j != i]
    n_eps = len(per_position[0])
    joined: list[np.ndarray] = []
    for ep_idx in range(n_eps):
        T = len(per_position[0][ep_idx])
        labels = np.empty(T, dtype=object)
        for t in range(T):
            labels[t] = tuple(per_position[j][ep_idx][t] for j in others)
        joined.append(labels)
    return joined


def estimate_cell_entropy(
    genomes: Sequence[np.ndarray],
    scenario: Scenario,
    n_episodes: int = 10_000,
    n_boot: int = 1000,
    seed: int = 0,
    num_workers: Optional[int] = None,
    cell_tag: str = "",
    beta: float = float("nan"),
    kappa: float = float("nan"),
    c: float = float("nan"),
    verdict: str = "",
) -> CellEntropyResult:
    """Estimate per-position H(A_i | A_{-i}) for one cell.

    Drives ``n_episodes`` rollouts in parallel, then computes
    plug-in + Miller-Madow conditional entropies (in bits) with an
    episode-level bootstrap CI for each of the 4 positions.

    Args:
        genomes: List of length ``num_agents`` of length-10 NE genomes.
        scenario: The cell's :class:`Scenario`.
        n_episodes: Number of independent rollouts.
        n_boot: Bootstrap resamples for the episode-level CI.
        seed: Base RNG seed (per-episode seeds derived deterministically).
        num_workers: Worker pool size; ``None`` uses ``cpu_count()``.
        cell_tag, beta, kappa, c, verdict: Pass-through metadata for the
            returned :class:`CellEntropyResult`.

    Returns:
        :class:`CellEntropyResult` with per-position entropies and CIs.
    """
    num_agents = scenario.num_agents

    episodes = rollout_joint_actions(
        genomes=genomes,
        scenario=scenario,
        n_episodes=n_episodes,
        seed=seed,
        num_workers=num_workers,
    )

    per_position = _build_per_position_episode_arrays(episodes, num_agents)
    n_steps_total = sum(len(ep_arr) for ep_arr in per_position[0])

    # Build joint-action episode arrays once for h_joint reuse.
    joint_episodes: list[np.ndarray] = []
    for ep_idx in range(len(episodes)):
        T = len(per_position[0][ep_idx])
        labels = np.empty(T, dtype=object)
        for t in range(T):
            labels[t] = tuple(per_position[j][ep_idx][t] for j in range(num_agents))
        joint_episodes.append(labels)

    # H(A_0, ..., A_{N-1}) point estimate on the full concatenation. We don't
    # carry a CI for the joint or the marginal here — the only CI we report is
    # for h_cond, since that's the SW-relevant scalar.
    h_joint_point = float(entropy_discrete(np.concatenate(joint_episodes)))

    positions: list[PositionEntropy] = []
    for i in range(num_agents):
        minus_i_episodes = _other_positions_episodes(per_position, i)
        h_minus_point = float(entropy_discrete(np.concatenate(minus_i_episodes)))

        # Conditional entropy bootstrap: resample episodes once per bootstrap
        # draw, then recompute joint and h_{-i} on the same resample. This
        # captures the covariance between the two and gives a tighter CI than
        # propagating two independent intervals.
        def _h_cond_estimator(joint_arr: np.ndarray, minus_arr: np.ndarray) -> float:
            h_xy = entropy_discrete(joint_arr)
            h_y = entropy_discrete(minus_arr)
            return max(0.0, h_xy - h_y)

        rng_cond = np.random.default_rng(seed + 200 + i)
        h_cond_point, h_cond_lo, h_cond_hi = episode_bootstrap_ci(
            estimator=_h_cond_estimator,
            episode_arrays=[joint_episodes, minus_i_episodes],
            n_boot=n_boot,
            rng=rng_cond,
        )

        positions.append(
            PositionEntropy(
                position=i,
                h_joint=h_joint_point,
                h_minus_i=h_minus_point,
                h_cond=h_cond_point,
                h_cond_ci_lo=h_cond_lo,
                h_cond_ci_hi=h_cond_hi,
                n_episodes=len(episodes),
                n_steps=n_steps_total,
            )
        )

    return CellEntropyResult(
        cell_tag=cell_tag,
        beta=beta,
        kappa=kappa,
        c=c,
        verdict=verdict,
        n_episodes=len(episodes),
        n_steps_total=n_steps_total,
        positions=positions,
    )


# ---------------------------------------------------------------------------
# Convenience: plain ``conditional_entropy`` from action arrays
# ---------------------------------------------------------------------------


def conditional_action_entropy_step_level(
    actions: np.ndarray,
    position: int,
) -> float:
    """Step-level (no episode boundaries) ``H(A_i | A_{-i})`` in bits.

    ``actions`` is shape ``(N, num_agents, action_dim)``. This helper is used
    by sanity tests where the i.i.d.-step assumption is acceptable; for
    trajectory data, prefer :func:`estimate_cell_entropy` with the episode
    bootstrap.
    """
    num_agents = actions.shape[1]
    n = actions.shape[0]
    labels = [np.empty(n, dtype=object) for _ in range(num_agents)]
    for i in range(num_agents):
        for t in range(n):
            labels[i][t] = _action_to_label(actions[t, i])
    others = [j for j in range(num_agents) if j != position]
    minus = np.empty(n, dtype=object)
    for t in range(n):
        minus[t] = tuple(labels[j][t] for j in others)
    joint = np.empty(n, dtype=object)
    for t in range(n):
        joint[t] = tuple(labels[j][t] for j in range(num_agents))
    return (
        max(0.0, joint_entropy(joint, minus) - entropy_discrete(minus))
        if False
        else conditional_entropy(labels[position], minus)
    )
