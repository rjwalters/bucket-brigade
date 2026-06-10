"""Per-cell baseline measurement for the phase-diagram (β, κ, c) grid.

Issue #413: ``MINSPEC_RANDOM = -87.72`` / ``MINSPEC_SPECIALIST = -22.07``
are calibrated at a single point in the (β, κ, c) parameter space (the
canonical ``minimal_specialization`` scenario, β=0.25 κ=0.5 c=0.5). The
#360 phase-diagram PPO sweep scores every cell against that one yardstick,
which is invalid for cross-cell comparison: a hand-coded specialist at
β=0.9 κ=0.1 plays a very different game than at β=0.5 κ=0.9, and the
random→specialist gap can collapse or invert.

This module measures, **per cell**, three baselines:

1. ``random_baseline``  -- uniform-random ``MultiDiscrete([10, 2, 2])``
   actions for all 4 agents.
2. ``specialist_homogeneous``  -- :class:`SpecialistPolicy` ×4. This is the
   apples-to-apples per-cell drop-in for ``MINSPEC_SPECIALIST``.
3. ``specialist_ne``  -- the converged 1×Hero + 3×Firefighter heterogeneous
   NE profile from the phase-diagram DO search
   (``bucket_brigade/baselines/release/local/nash/phase_diagram/*.json``).
   This is the appropriate yardstick for the paper §3/§4 NE-structure-vs-
   PPO-success hypothesis.

The ``per_step_team`` definition matches the existing
``experiments/p3_specialization/diagnostics/random_baseline.py``: total
episode team reward (``sum(env.rewards)`` summed across all steps) divided
by ``env.night``. This is the same scale as
``mean_step_reward_team`` in the trainer ``metrics.json`` files, so the
gap_closed denominator stays bit-comparable.

Compute placement
-----------------
10k episodes per (cell × policy) is CPU-bound but parallelises trivially
across episodes. The #368 entropy sweep ran 10k × 4 cells in ~10 min on a
32-core cluster host; this is ~3× that. Do not run on a laptop — see
``CLAUDE.md``.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Optional

import numpy as np

from bucket_brigade.agents.heuristic_agent import HeuristicAgent
from bucket_brigade.baselines.specialist import SpecialistPolicy
from bucket_brigade.envs.bucket_brigade_env import BucketBrigadeEnv
from bucket_brigade.envs.scenarios_generated import Scenario

__all__ = [
    "BaselineEstimate",
    "make_phase_diagram_scenario",
    "measure_random",
    "measure_specialist_homogeneous",
    "measure_specialist_ne",
    "load_ne_genomes",
]


# ---------------------------------------------------------------------------
# Container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BaselineEstimate:
    """Per-cell per-policy baseline estimate with episode-bootstrap 95% CI.

    All fields are floats except ``n_episodes`` (int).
    """

    mean: float
    ci95_lo: float
    ci95_hi: float
    n_episodes: int

    def to_dict(self) -> dict:
        return {
            "mean": float(self.mean),
            "ci95_lo": float(self.ci95_lo),
            "ci95_hi": float(self.ci95_hi),
            "n_episodes": int(self.n_episodes),
        }


# ---------------------------------------------------------------------------
# Scenario factory
# ---------------------------------------------------------------------------


def make_phase_diagram_scenario(
    beta: float,
    kappa: float,
    cost: float,
    base_scenario_name: str = "minimal_specialization",
    num_agents: int = 4,
) -> Scenario:
    """Build a ``Scenario`` with (β, κ, c) overridden on the base family.

    Mirrors ``experiments/scripts/compute_nash_phase_diagram._make_scenario``
    so the per-cell baselines and the NE-search rewards are bit-comparable.
    """
    from bucket_brigade.envs.scenarios_generated import get_scenario_by_name

    base = get_scenario_by_name(base_scenario_name, num_agents=num_agents)
    return dataclasses.replace(
        base,
        prob_fire_spreads_to_neighbor=float(beta),
        prob_solo_agent_extinguishes_fire=float(kappa),
        cost_to_work_one_night=float(cost),
    )


# ---------------------------------------------------------------------------
# Per-episode rollout workers
# ---------------------------------------------------------------------------


def _per_step_team(env: BucketBrigadeEnv) -> float:
    """Live alias for ``env.rewards.sum()``; placeholder for symmetry."""
    return float(env.rewards.sum())


def _run_random_episode(args: tuple[Scenario, int]) -> float:
    """One uniform-random episode. Returns ``per_step_team``.

    Per-step team reward is total episode team reward (summed across agents
    and steps) divided by ``env.night``. Matches the convention in
    ``experiments/p3_specialization/diagnostics/random_baseline.py``.
    """
    scenario, seed = args
    env = BucketBrigadeEnv(scenario=scenario)
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    total_reward = 0.0
    while not env.done:
        actions = np.stack(
            [
                rng.integers(0, env.num_houses, size=env.num_agents),
                rng.integers(0, 2, size=env.num_agents),
                rng.integers(0, 2, size=env.num_agents),
            ],
            axis=-1,
        ).astype(np.int64)
        _, rewards, _, _ = env.step(actions)
        total_reward += float(rewards.sum())
    nights = max(1, int(env.night))
    return total_reward / nights


def _run_specialist_homogeneous_episode(args: tuple[Scenario, int]) -> float:
    """One ``SpecialistPolicy ×4`` episode. Returns ``per_step_team``."""
    scenario, seed = args
    env = BucketBrigadeEnv(scenario=scenario)
    obs = env.reset(seed=seed)
    policy = SpecialistPolicy(num_agents=env.num_agents, num_houses=env.num_houses)
    total_reward = 0.0
    while not env.done:
        actions = policy(obs).astype(np.int64)
        obs, rewards, _, _ = env.step(actions)
        total_reward += float(rewards.sum())
    nights = max(1, int(env.night))
    return total_reward / nights


def _run_ne_episode(args: tuple[list[np.ndarray], Scenario, int]) -> float:
    """One heuristic-agent episode from per-position genomes. Returns ``per_step_team``.

    Mirrors :func:`bucket_brigade.analysis.conditional_entropy._run_one_episode`'s
    construction (HeuristicAgent + numpy global seeding offset), but captures
    the per-step team reward instead of the action trace.
    """
    genomes, scenario, seed = args
    env = BucketBrigadeEnv(scenario=scenario)
    obs = env.reset(seed=seed)

    # Mirror _run_one_episode's seeding so reproducibility patterns line up.
    np.random.seed((seed + 991) % (2**31 - 1))
    agents = [
        HeuristicAgent(params=np.asarray(genomes[i], dtype=np.float64), agent_id=i)
        for i in range(env.num_agents)
    ]
    for a in agents:
        a.reset()

    total_reward = 0.0
    while not env.done:
        action_list = [agent.act(obs) for agent in agents]
        actions = np.stack(action_list, axis=0).astype(np.int64)
        obs, rewards, _, _ = env.step(actions)
        total_reward += float(rewards.sum())
    nights = max(1, int(env.night))
    return total_reward / nights


# ---------------------------------------------------------------------------
# Episode bootstrap
# ---------------------------------------------------------------------------


def _episode_bootstrap_ci(
    per_step_values: np.ndarray,
    n_boot: int = 1000,
    confidence: float = 0.95,
    rng: Optional[np.random.Generator] = None,
) -> tuple[float, float, float]:
    """Episode-level percentile bootstrap of the mean.

    Each episode contributes one scalar (its ``per_step_team``); bootstrapping
    the mean of those scalars is the right granularity because adjacent
    episodes are independent. Matches the
    ``conditional_entropy.episode_bootstrap_ci`` spirit but for a scalar
    estimator (not info-theoretic).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    n = len(per_step_values)
    if n == 0:
        raise ValueError("per_step_values must be non-empty")
    point = float(per_step_values.mean())
    boots = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[b] = per_step_values[idx].mean()
    alpha = (1.0 - confidence) / 2.0
    lo = float(np.quantile(boots, alpha))
    hi = float(np.quantile(boots, 1.0 - alpha))
    return point, lo, hi


# ---------------------------------------------------------------------------
# Public measurement functions
# ---------------------------------------------------------------------------


def _seeds_for(seed: int, n_episodes: int) -> list[int]:
    """Derive per-episode seeds deterministically from a base seed."""
    rng = np.random.default_rng(seed)
    return [int(s) for s in rng.integers(0, 2**31 - 1, size=n_episodes)]


def _parallel_map(worker, args_list, num_workers: Optional[int]) -> list[float]:
    """Run ``worker`` over ``args_list`` with an optional multiprocessing Pool.

    Mirrors the conditional_entropy parallelism pattern: ``None`` -> cpu_count,
    ``<= 1`` -> sequential (useful for tests).
    """
    if num_workers is None:
        num_workers = cpu_count()
    if num_workers > 1 and len(args_list) > 1:
        with Pool(processes=num_workers) as pool:
            return list(pool.map(worker, args_list))
    return [worker(a) for a in args_list]


def measure_random(
    beta: float,
    kappa: float,
    cost: float,
    n_episodes: int = 10_000,
    seed: int = 0,
    num_workers: Optional[int] = None,
    base_scenario_name: str = "minimal_specialization",
    n_boot: int = 1000,
) -> BaselineEstimate:
    """Measure per-step team reward for 4 uniform-random agents at (β, κ, c).

    Returns :class:`BaselineEstimate` with episode-bootstrap 95% CI.
    """
    scenario = make_phase_diagram_scenario(beta, kappa, cost, base_scenario_name)
    seeds = _seeds_for(seed, n_episodes)
    args_list = [(scenario, s) for s in seeds]
    values = np.asarray(_parallel_map(_run_random_episode, args_list, num_workers))
    mean, lo, hi = _episode_bootstrap_ci(
        values, n_boot=n_boot, rng=np.random.default_rng(seed + 7)
    )
    return BaselineEstimate(mean=mean, ci95_lo=lo, ci95_hi=hi, n_episodes=n_episodes)


def measure_specialist_homogeneous(
    beta: float,
    kappa: float,
    cost: float,
    n_episodes: int = 10_000,
    seed: int = 0,
    num_workers: Optional[int] = None,
    base_scenario_name: str = "minimal_specialization",
    n_boot: int = 1000,
) -> BaselineEstimate:
    """Measure per-step team reward for ``SpecialistPolicy ×4`` at (β, κ, c).

    This is the apples-to-apples per-cell drop-in for the
    ``MINSPEC_SPECIALIST = -22.07`` constant from ``bucket_brigade.baselines``.
    """
    scenario = make_phase_diagram_scenario(beta, kappa, cost, base_scenario_name)
    seeds = _seeds_for(seed, n_episodes)
    args_list = [(scenario, s) for s in seeds]
    values = np.asarray(
        _parallel_map(_run_specialist_homogeneous_episode, args_list, num_workers)
    )
    mean, lo, hi = _episode_bootstrap_ci(
        values, n_boot=n_boot, rng=np.random.default_rng(seed + 11)
    )
    return BaselineEstimate(mean=mean, ci95_lo=lo, ci95_hi=hi, n_episodes=n_episodes)


def load_ne_genomes(ne_genomes_path: Path | str) -> list[np.ndarray]:
    """Load the heterogeneous NE genomes from a phase-diagram NE JSON file.

    The schema (from ``bucket_brigade/baselines/release/local/nash/
    phase_diagram/*.json``) has a top-level ``positions`` array, each entry
    with a ``genome`` (length-10 float vector). Returns a list of length 4.
    """
    path = Path(ne_genomes_path)
    if not path.exists():
        raise FileNotFoundError(f"NE genomes file not found: {path}")
    data = json.loads(path.read_text())
    positions = data.get("positions")
    if not isinstance(positions, list) or len(positions) == 0:
        raise ValueError(f"{path}: no 'positions' list in NE genome file")
    genomes: list[np.ndarray] = []
    for entry in sorted(positions, key=lambda e: int(e.get("position", 0))):
        g = entry.get("genome")
        if g is None:
            raise ValueError(f"{path}: position {entry} missing 'genome'")
        genomes.append(np.asarray(g, dtype=np.float64))
    return genomes


def measure_specialist_ne(
    beta: float,
    kappa: float,
    cost: float,
    ne_genomes_path: Path | str,
    n_episodes: int = 10_000,
    seed: int = 0,
    num_workers: Optional[int] = None,
    base_scenario_name: str = "minimal_specialization",
    n_boot: int = 1000,
) -> BaselineEstimate:
    """Measure per-step team reward for the converged heterogeneous-NE profile.

    Loads per-position genomes from ``ne_genomes_path`` (the
    ``hetero_double_oracle`` JSON schema) and runs ``n_episodes`` rollouts
    with ``HeuristicAgent`` per position.
    """
    scenario = make_phase_diagram_scenario(beta, kappa, cost, base_scenario_name)
    genomes = load_ne_genomes(ne_genomes_path)
    if len(genomes) != scenario.num_agents:
        raise ValueError(
            f"NE genomes file has {len(genomes)} positions, "
            f"scenario.num_agents = {scenario.num_agents}"
        )
    seeds = _seeds_for(seed, n_episodes)
    args_list = [(genomes, scenario, s) for s in seeds]
    values = np.asarray(_parallel_map(_run_ne_episode, args_list, num_workers))
    mean, lo, hi = _episode_bootstrap_ci(
        values, n_boot=n_boot, rng=np.random.default_rng(seed + 13)
    )
    return BaselineEstimate(mean=mean, ci95_lo=lo, ci95_hi=hi, n_episodes=n_episodes)
